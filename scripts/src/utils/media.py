from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.strip()
        stdout = error.stdout.strip()
        details = stderr or stdout or "No subprocess output captured."
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}") from error


def ensure_ffmpeg_tools_available() -> None:
    missing_tools = [
        tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None
    ]
    if missing_tools:
        joined = ", ".join(missing_tools)
        raise RuntimeError(f"Missing required system tools: {joined}.")


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
    available_duration: float,
    segment_length: float,
) -> list[float]:
    if segment_length <= 0 or available_duration < segment_length:
        return []

    segment_count = int(available_duration // segment_length)
    return [index * segment_length for index in range(segment_count)]


def export_video_segment(
    source_path: Path, destination_path: Path, start_time: float, duration: float
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


def split_video_into_segments(
    source_path: Path,
    output_dir: Path,
    segment_length: float,
    name_prefix: str | None = None,
    available_duration: float | None = None,
) -> list[Path]:
    """Split a video file into fixed-length segments written as MP4 files.

    Args:
        source_path: Path to the source video file.
        output_dir: Directory where segments will be written (created if missing).
        segment_length: Length of each segment in seconds.
        name_prefix: Optional file stem prefix; defaults to the source file stem.
        available_duration: Optional duration cap (seconds); defaults to the source duration.

    Returns:
        Sorted list of paths of the produced segment files.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = (
        available_duration if available_duration is not None else probe_duration(source_path)
    )
    starts = build_segment_starts(duration, segment_length)

    prefix = name_prefix if name_prefix is not None else source_path.stem
    produced: list[Path] = []
    for index, start_time in enumerate(starts):
        destination = output_dir / f"{prefix}_{index:03d}.mp4"
        export_video_segment(
            source_path=source_path,
            destination_path=destination,
            start_time=start_time,
            duration=segment_length,
        )
        produced.append(destination)
    return produced


def split_audio_into_segments(
    source_path: Path,
    output_dir: Path,
    segment_length: float,
    sample_rate: int,
    channels: int,
    name_prefix: str | None = None,
    available_duration: float | None = None,
) -> list[Path]:
    """Split an audio file into fixed-length WAV segments.

    Args:
        source_path: Path to the source audio file.
        output_dir: Directory where segments will be written (created if missing).
        segment_length: Length of each segment in seconds.
        sample_rate: Output sample rate (Hz).
        channels: Output channel count.
        name_prefix: Optional file stem prefix; defaults to the source file stem.
        available_duration: Optional duration cap (seconds); defaults to the source duration.

    Returns:
        Sorted list of paths of the produced segment files.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duration = (
        available_duration if available_duration is not None else probe_duration(source_path)
    )
    starts = build_segment_starts(duration, segment_length)

    prefix = name_prefix if name_prefix is not None else source_path.stem
    produced: list[Path] = []
    for index, start_time in enumerate(starts):
        destination = output_dir / f"{prefix}_{index:03d}.wav"
        export_audio_segment(
            source_path=source_path,
            destination_path=destination,
            start_time=start_time,
            duration=segment_length,
            sample_rate=sample_rate,
            channels=channels,
        )
        produced.append(destination)
    return produced
