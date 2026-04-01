from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}


@dataclass(frozen=True)
class SourcePair:
    pair_key: str
    video_path: Path
    audio_path: Path


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    split: str
    pair_key: str
    start_time: float
    end_time: float
    duration: float
    source_video_path: str
    source_audio_path: str
    video_path: str
    audio_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare train/validation AV dataset by cutting random aligned segments."
    )
    parser.add_argument(
        "--raw-video-dir",
        type=Path,
        default=Path("data/raw/videos"),
        help="Directory with source video files.",
    )
    parser.add_argument(
        "--raw-audio-dir",
        type=Path,
        default=Path("data/raw/audio"),
        help="Directory with source audio files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for prepared segments and manifests.",
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=5.0,
        help="Length of each aligned segment in seconds.",
    )
    parser.add_argument(
        "--segments-per-pair",
        type=int,
        default=4,
        help="How many random segments to generate from each matched pair.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of source pairs assigned to the training split.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=16000,
        help="Sample rate used for exported WAV clips.",
    )
    parser.add_argument(
        "--audio-channels",
        type=int,
        default=1,
        help="Channel count used for exported WAV clips.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling split and crop positions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove previous prepared data before writing new files.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Create raw data directories and exit without generating processed data.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.segment_length <= 0:
        raise ValueError("segment_length must be greater than 0.")
    if args.segments_per_pair <= 0:
        raise ValueError("segments_per_pair must be greater than 0.")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if args.audio_sample_rate <= 0:
        raise ValueError("audio_sample_rate must be greater than 0.")
    if args.audio_channels <= 0:
        raise ValueError("audio_channels must be greater than 0.")


def ensure_ffmpeg_tools_available() -> None:
    missing_tools = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing_tools:
        joined = ", ".join(missing_tools)
        raise RuntimeError(f"Missing required system tools: {joined}.")


def initialize_raw_directories(raw_video_dir: Path, raw_audio_dir: Path) -> None:
    raw_video_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_dir.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip()
        stdout = error.stdout.strip()
        details = stderr or stdout or "No subprocess output captured."
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}") from error


def discover_media_files(root_dir: Path, extensions: set[str]) -> dict[str, Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {root_dir}")

    files_by_stem: dict[str, Path] = {}
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        if path.stem in files_by_stem:
            previous = files_by_stem[path.stem]
            raise ValueError(
                f"Duplicate stem '{path.stem}' found for {previous} and {path}. "
                "Make stems unique so files can be paired unambiguously."
            )

        files_by_stem[path.stem] = path

    return files_by_stem


def build_source_pairs(raw_video_dir: Path, raw_audio_dir: Path) -> list[SourcePair]:
    videos = discover_media_files(raw_video_dir, SUPPORTED_VIDEO_EXTENSIONS)
    audios = discover_media_files(raw_audio_dir, SUPPORTED_AUDIO_EXTENSIONS)

    common_keys = sorted(set(videos) & set(audios))
    if not common_keys:
        raise ValueError(
            "No audio/video pairs found. Files must share the same stem, "
            "for example sample_001.mp4 and sample_001.wav."
        )

    unmatched_videos = sorted(set(videos) - set(audios))
    unmatched_audios = sorted(set(audios) - set(videos))
    if unmatched_videos:
        print(f"Skipping videos without matching audio: {', '.join(unmatched_videos)}")
    if unmatched_audios:
        print(f"Skipping audio without matching video: {', '.join(unmatched_audios)}")

    return [
        SourcePair(pair_key=key, video_path=videos[key], audio_path=audios[key])
        for key in common_keys
    ]


def split_pairs(
    pairs: list[SourcePair],
    train_ratio: float,
    rng: random.Random,
) -> dict[str, list[SourcePair]]:
    shuffled_pairs = pairs[:]
    rng.shuffle(shuffled_pairs)

    if len(shuffled_pairs) == 1:
        return {"train": shuffled_pairs, "val": []}

    train_count = round(len(shuffled_pairs) * train_ratio)
    train_count = max(1, min(train_count, len(shuffled_pairs) - 1))

    return {
        "train": shuffled_pairs[:train_count],
        "val": shuffled_pairs[train_count:],
    }


def probe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = run_command(command)
    payload = json.loads(result.stdout)
    duration = float(payload["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Invalid media duration for {path}: {duration}")
    return duration


def build_segment_starts(
    max_start: float,
    segment_count: int,
    rng: random.Random,
) -> list[float]:
    if segment_count <= 0:
        return []
    if max_start <= 0:
        return [0.0] * segment_count

    starts: list[float] = []
    bin_width = max_start / segment_count
    for index in range(segment_count):
        bin_start = bin_width * index
        bin_end = max_start if index == segment_count - 1 else bin_width * (index + 1)
        if bin_end <= bin_start:
            starts.append(bin_start)
            continue
        starts.append(rng.uniform(bin_start, bin_end))

    return sorted(starts)


def reset_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already contains files: {output_dir}. "
                "Use --overwrite to rebuild the prepared dataset."
            )

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / split / "videos").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "audio").mkdir(parents=True, exist_ok=True)


def export_video_segment(source_path: Path, destination_path: Path, start_time: float, duration: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(destination_path),
    ]
    run_command(command)


def export_audio_segment(
    source_path: Path,
    destination_path: Path,
    start_time: float,
    duration: float,
    sample_rate: int,
    channels: int,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(destination_path),
    ]
    run_command(command)


def prepare_split(
    split: str,
    pairs: list[SourcePair],
    output_dir: Path,
    segment_length: float,
    segments_per_pair: int,
    sample_rate: int,
    channels: int,
    rng: random.Random,
) -> list[PreparedSample]:
    samples: list[PreparedSample] = []
    video_dir = output_dir / split / "videos"
    audio_dir = output_dir / split / "audio"

    for pair in pairs:
        video_duration = probe_duration(pair.video_path)
        audio_duration = probe_duration(pair.audio_path)
        available_duration = min(video_duration, audio_duration)
        usable_duration = available_duration - segment_length

        if usable_duration < 0:
            print(
                f"Skipping pair '{pair.pair_key}' because the shared duration "
                f"({available_duration:.2f}s) is shorter than segment length ({segment_length:.2f}s)."
            )
            continue

        segment_starts = build_segment_starts(
            max_start=usable_duration,
            segment_count=segments_per_pair,
            rng=rng,
        )

        for segment_index, start_time in enumerate(segment_starts):
            sample_id = f"{pair.pair_key}_{segment_index:03d}"
            prepared_video_path = video_dir / f"{sample_id}.mp4"
            prepared_audio_path = audio_dir / f"{sample_id}.wav"
            end_time = start_time + segment_length

            export_video_segment(
                source_path=pair.video_path,
                destination_path=prepared_video_path,
                start_time=start_time,
                duration=segment_length,
            )
            export_audio_segment(
                source_path=pair.audio_path,
                destination_path=prepared_audio_path,
                start_time=start_time,
                duration=segment_length,
                sample_rate=sample_rate,
                channels=channels,
            )

            samples.append(
                PreparedSample(
                    sample_id=sample_id,
                    split=split,
                    pair_key=pair.pair_key,
                    start_time=round(start_time, 3),
                    end_time=round(end_time, 3),
                    duration=segment_length,
                    source_video_path=str(pair.video_path),
                    source_audio_path=str(pair.audio_path),
                    video_path=str(prepared_video_path),
                    audio_path=str(prepared_audio_path),
                )
            )

    return samples


def write_manifest(samples: list[PreparedSample], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="ascii") as manifest_file:
        for sample in samples:
            manifest_file.write(json.dumps(asdict(sample)) + "\n")


def write_summary(
    output_dir: Path,
    split_pairs_map: dict[str, list[SourcePair]],
    samples_by_split: dict[str, list[PreparedSample]],
    args: argparse.Namespace,
) -> None:
    summary = {
        "raw_video_dir": str(args.raw_video_dir),
        "raw_audio_dir": str(args.raw_audio_dir),
        "segment_length": args.segment_length,
        "segments_per_pair": args.segments_per_pair,
        "train_ratio": args.train_ratio,
        "audio_sample_rate": args.audio_sample_rate,
        "audio_channels": args.audio_channels,
        "seed": args.seed,
        "pairs": {split: len(pairs) for split, pairs in split_pairs_map.items()},
        "samples": {split: len(samples) for split, samples in samples_by_split.items()},
    }
    summary_path = output_dir / "manifests" / "summary.json"
    with summary_path.open("w", encoding="ascii") as summary_file:
        json.dump(summary, summary_file, indent=2)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.init:
        initialize_raw_directories(
            raw_video_dir=args.raw_video_dir,
            raw_audio_dir=args.raw_audio_dir,
        )
        print(f"Created raw video directory: {args.raw_video_dir}")
        print(f"Created raw audio directory: {args.raw_audio_dir}")
        return

    ensure_ffmpeg_tools_available()

    rng = random.Random(args.seed)
    pairs = build_source_pairs(args.raw_video_dir, args.raw_audio_dir)
    split_pairs_map = split_pairs(pairs=pairs, train_ratio=args.train_ratio, rng=rng)
    reset_output_dir(args.output_dir, overwrite=args.overwrite)

    samples_by_split: dict[str, list[PreparedSample]] = {}
    for split, split_pairs_list in split_pairs_map.items():
        split_samples = prepare_split(
            split=split,
            pairs=split_pairs_list,
            output_dir=args.output_dir,
            segment_length=args.segment_length,
            segments_per_pair=args.segments_per_pair,
            sample_rate=args.audio_sample_rate,
            channels=args.audio_channels,
            rng=rng,
        )
        samples_by_split[split] = split_samples
        write_manifest(
            samples=split_samples,
            manifest_path=args.output_dir / "manifests" / f"{split}.jsonl",
        )

    write_summary(
        output_dir=args.output_dir,
        split_pairs_map=split_pairs_map,
        samples_by_split=samples_by_split,
        args=args,
    )

    print(
        "Prepared dataset with "
        f"{len(samples_by_split['train'])} train samples and "
        f"{len(samples_by_split['val'])} validation samples."
    )
    print(f"Manifests written to: {args.output_dir / 'manifests'}")


if __name__ == "__main__":
    main()