import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import torchvision.io as v_io
from tqdm import tqdm

from src import models, utils


VIDEO_EXTENSIONS = ("*.mp4", "*.mov", "*.mkv", "*.avi", "*.webm")
AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.aac", "*.ogg")


def load_models(
    video_weights, video_transformer_size, music_weights, music_transformer_size, device
):
    video_model = models.VideoTransformer(
        num_layers=video_transformer_size, query_dim=512
    )
    music_model = models.MusicTransformer(
        num_layers=music_transformer_size, query_dim=512
    )

    video_model.load_state_dict(torch.load(video_weights, map_location=device))
    music_model.load_state_dict(torch.load(music_weights, map_location=device))

    video_model.to(device).eval()
    music_model.to(device).eval()

    return video_model, music_model


def encode_video_segment(segment_path: Path, video_converter, device) -> torch.Tensor:
    """Load a video file and run it through the converter -> [1, 1, 512]."""
    video_tensor, _, __ = v_io.read_video(str(segment_path), pts_unit="sec")
    # [T, H, W, C] -> [1, 1, T, H, W, C]
    features = video_converter(video_tensor.unsqueeze(0).unsqueeze(0).to(device))
    return features  # [1, 1, embed_dim]


def encode_audio_segment(segment_path: Path, music_converter, device) -> torch.Tensor:
    """Load an audio file and run it through the converter -> [1, 1, 512]."""
    audio_tensor, _ = torchaudio.load(str(segment_path))
    # [C, samples] -> [1, 1, C, samples]
    features = music_converter(audio_tensor.unsqueeze(0).unsqueeze(0).to(device))
    return features  # [1, 1, embed_dim]


def compute_embedding(model, features: torch.Tensor, device) -> torch.Tensor:
    """Run [1, 1, D] features through the transformer, mean-pool over the sequence
    dimension, and return the L2-normalized vector [D]."""
    with torch.no_grad():
        out = model(features.to(device))  # [1, S, D]
        pooled = out.mean(dim=1)  # [1, D]
        normalized = F.normalize(pooled, p=2, dim=-1)
    return normalized.squeeze(0).detach().cpu()


def discover_files(db_dir: Path, db_type: str) -> list[Path]:
    patterns = VIDEO_EXTENSIONS if db_type == "video" else AUDIO_EXTENSIONS
    files: list[Path] = []
    for pattern in patterns:
        files.extend(db_dir.glob(pattern))
    return sorted(files)


def build_db_segments(db_dir, db_type, model, segment_length, device, tmpdir):
    """Split database files into segments and compute embeddings for them.

    Returns a list of dicts: {"path", "embedding", "source", "index"}.
    """
    db_dir = Path(db_dir)
    db_segments: list[dict] = []
    db_files = discover_files(db_dir, db_type)

    print(
        f"\n-> Building embeddings for the {db_type} database... Found {len(db_files)} files."
    )
    if not db_files:
        return db_segments

    if db_type == "video":
        converter = models.VideoConverter()
    else:
        converter = models.MusicConverter()

    db_segments_dir = Path(tmpdir) / "db_segments"
    db_segments_dir.mkdir(parents=True, exist_ok=True)

    for file_path in db_files:
        try:
            if db_type == "video":
                segments = utils.media.split_video_into_segments(
                    source_path=file_path,
                    output_dir=db_segments_dir,
                    segment_length=segment_length,
                )
            else:
                segments = utils.media.split_audio_into_segments(
                    source_path=file_path,
                    output_dir=db_segments_dir,
                    segment_length=segment_length,
                    sample_rate=16000,
                    channels=1,
                )
        except Exception as e:
            print(f"Failed to process database file {file_path}: {e}")
            continue

        for index, segment_path in enumerate(
            tqdm(segments, desc=f"Encoding {file_path.name}", leave=False)
        ):
            if db_type == "video":
                features = encode_video_segment(segment_path, converter, device)
            else:
                features = encode_audio_segment(segment_path, converter, device)
            embedding = compute_embedding(model, features, device)
            db_segments.append(
                {
                    "path": Path(segment_path),
                    "embedding": embedding,
                    "source": file_path.name,
                    "index": index,
                }
            )

    return db_segments


def assemble_output(
    query_path: Path,
    query_type: str,
    chosen_segment_paths: list[Path],
    output_path: Path,
    tmpdir: Path,
) -> None:
    """Assemble the final video file from the original query and matched segments."""
    tmpdir = Path(tmpdir)

    if query_type == "video":
        concatenated_audio = tmpdir / "matched_audio.wav"
        wav_segments = tmpdir / "matched_audio_segments"
        wav_segments.mkdir(parents=True, exist_ok=True)
        # Re-encode the selected segments into a consistent WAV format so the concat
        # demuxer receives uniform inputs from a single directory.
        normalized = []
        for i, seg in enumerate(chosen_segment_paths):
            dest = wav_segments / f"part_{i:04d}.wav"
            utils.media.run_command(
                [
                    "ffmpeg", "-y", "-i", str(seg),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(dest),
                ]
            )
            normalized.append(dest)
        utils.media.concat_segments(normalized, concatenated_audio)

        audio_duration = utils.media.probe_duration(concatenated_audio)
        utils.media.mux_video_with_audio(
            video_path=query_path,
            audio_path=concatenated_audio,
            output_path=output_path,
            max_duration=audio_duration,
        )
    else:
        concatenated_video = tmpdir / "matched_video.mp4"
        utils.media.concat_segments(chosen_segment_paths, concatenated_video)

        video_duration = utils.media.probe_duration(concatenated_video)
        utils.media.mux_video_with_audio(
            video_path=concatenated_video,
            audio_path=query_path,
            output_path=output_path,
            max_duration=video_duration,
        )


def match(
    query_path,
    db_dir,
    query_model,
    db_model,
    query_type,
    segment_length,
    device,
    output_path,
):
    """Main segment-matching and assembly pipeline."""

    query_path = Path(query_path)
    db_dir = Path(db_dir)

    if not query_path.exists():
        print(f"[ERROR]: File '{query_path}' doesn't exist.")
        sys.exit(1)
    if not db_dir.exists() or not db_dir.is_dir():
        print(f"[ERROR]: Database directory '{db_dir}' doesn't exist.")
        sys.exit(1)

    utils.media.ensure_ffmpeg_tools_available()

    db_type = "music" if query_type == "video" else "video"
    output_path = Path(output_path) if output_path else Path(
        f"{query_path.stem}_out.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. CANDIDATE DATABASE
        db_segments = build_db_segments(
            db_dir, db_type, db_model, segment_length, device, tmpdir_path
        )
        if not db_segments:
            print(
                "Error: database is empty or no segments could be extracted."
            )
            sys.exit(1)

        # 2. QUERY SEGMENTATION AND ENCODING
        print(f"\n-> Analyzing query file: {query_path.name}")
        query_segments_dir = tmpdir_path / "query_segments"
        query_segments_dir.mkdir(parents=True, exist_ok=True)

        if query_type == "video":
            query_segments = utils.media.split_video_into_segments(
                source_path=query_path,
                output_dir=query_segments_dir,
                segment_length=segment_length,
            )
            query_converter = models.VideoConverter()
        else:
            query_segments = utils.media.split_audio_into_segments(
                source_path=query_path,
                output_dir=query_segments_dir,
                segment_length=segment_length,
                sample_rate=16000,
                channels=1,
            )
            query_converter = models.MusicConverter()

        if not query_segments:
            print(
                f"Error: query '{query_path.name}' is shorter than the segment length "
                f"({segment_length}s)."
            )
            sys.exit(1)

        # 3. MATCHING
        db_embeddings = torch.stack([seg["embedding"] for seg in db_segments])  # [N, D]

        chosen_segment_paths: list[Path] = []
        for segment_index, segment in enumerate(
            tqdm(query_segments, desc="Matching segments")
        ):
            if query_type == "video":
                features = encode_video_segment(segment, query_converter, device)
            else:
                features = encode_audio_segment(segment, query_converter, device)
            query_emb = compute_embedding(query_model, features, device)  # [D]

            scores = db_embeddings @ query_emb  # [N]
            best_idx = int(torch.argmax(scores).item())
            best = db_segments[best_idx]

            start_t = segment_index * segment_length
            end_t = start_t + segment_length
            print(
                f"   [Time {start_t:05.1f}s - {end_t:05.1f}s] -> "
                f"{best['source']} #{best['index']:03d} (score={scores[best_idx]:.4f})"
            )
            chosen_segment_paths.append(best["path"])

        # 4. FINAL ASSEMBLY
        print("\n-> Rendering final video file...")
        assemble_output(
            query_path=query_path,
            query_type=query_type,
            chosen_segment_paths=chosen_segment_paths,
            output_path=output_path,
            tmpdir=tmpdir_path,
        )

        print(f"\nDone. Output written to: {output_path}\n")


def main(args: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_model, music_model = load_models(
        video_weights=args["video_weights"],
        video_transformer_size=args["video_transformer_size"],
        music_weights=args["music_weights"],
        music_transformer_size=args["music_transformer_size"],
        device=device,
    )

    if args["command"] == "video2music":
        match(
            query_path=args["video"],
            db_dir=args["music_db"],
            query_model=video_model,
            db_model=music_model,
            query_type="video",
            segment_length=args["segment_length"],
            device=device,
            output_path=args.get("output"),
        )
    elif args["command"] == "music2video":
        match(
            query_path=args["music"],
            db_dir=args["video_db"],
            query_model=music_model,
            db_model=video_model,
            query_type="music",
            segment_length=args["segment_length"],
            device=device,
            output_path=args.get("output"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for matching music to video (and vice versa) using contrastive Transformers."
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Select search mode"
    )

    # Subcommand: Video -> Music
    v2m_parser = subparsers.add_parser(
        "video2music", help="Find the best matching music for the given video"
    )
    v2m_parser.add_argument(
        "--video",
        "-v",
        required=True,
        type=str,
        help="Path to the input video file (.mp4)",
    )
    v2m_parser.add_argument(
        "--music-db",
        "-m",
        required=True,
        type=str,
        help="Directory containing the music database",
    )

    # Subcommand: Music -> Video
    m2v_parser = subparsers.add_parser(
        "music2video",
        help="Find the best matching video clips for the given music track",
    )
    m2v_parser.add_argument(
        "--music",
        "-m",
        required=True,
        type=str,
        help="Path to the input audio file (.mp3 / .wav)",
    )
    m2v_parser.add_argument(
        "--video-db", "-db", required=True, type=str, help="Directory containing the video database"
    )

    # Shared configuration arguments added to both subcommands
    for p in [v2m_parser, m2v_parser]:
        p.add_argument(
            "--video-weights",
            default="models/video_transformer.pth",
            type=str,
            help="Path to video_transformer weights",
        )
        p.add_argument(
            "--video-transformer-size",
            default=3,
            type=int,
            help="Number of layers in video_transformer",
        )
        p.add_argument(
            "--music-weights",
            default="models/music_transformer.pth",
            type=str,
            help="Path to music_transformer weights",
        )
        p.add_argument(
            "--music-transformer-size",
            default=3,
            type=int,
            help="Number of layers in music_transformer",
        )
        p.add_argument(
            "--segment-length", default=5, type=int, help="Segment length in seconds"
        )
        p.add_argument(
            "--output",
            "-o",
            default=None,
            type=str,
            help="Path to the output .mp4 file (default: <query_stem>_out.mp4)",
        )

    args = vars(parser.parse_args())
    main(args)
