#!/usr/bin/env python3
"""Video pre-extractor for ML datasets.

Recursively reads raw MP4 files and converts them into clips directly
usable by PyTorchVideo and a PyTorch training pipeline.

Usage:
    python video_preextractor.py --input ./raw_videos --output ./clips --clip_duration 2.0 --clip_stride 1.5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import BinaryIO, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI / argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_preextractor",
        description="Recursively extract clips from MP4 files for ML training.",
    )
    parser.add_argument("--input", required=True, help="Directory containing raw MP4 files.")
    parser.add_argument("--output", required=True, help="Directory for processed .pt clips.")
    parser.add_argument(
        "--clip_duration",
        type=float,
        default=2.0,
        help="Duration of each clip in seconds (default 2.0).",
    )
    parser.add_argument(
        "--clip_stride",
        type=float,
        default=1.5,
        help="Temporal stride between consecutive clips (default 1.5).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Target frame sampling rate (default 8).",
    )
    parser.add_argument(
        "--spatial_size",
        type=int,
        default=224,
        help="Target spatial dimension (square) (default 224).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing clips instead of resuming.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default INFO).",
    )
    return parser


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClipMetadata:
    """Metadata written to the JSONL manifest for each extracted clip."""

    source_path: str
    clip_path: str
    start_seconds: float
    end_seconds: float
    tensor_shape: list[int]
    video_width: int
    video_height: int
    video_fps: float
    duration_seconds: float
    num_clips: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


@dataclass
class ProcessingStats:
    """Aggregate counters reported at the end of a run."""

    total_videos: int = 0
    successful_videos: int = 0
    failed_videos: int = 0
    total_clips: int = 0
    skipped_clips: int = 0

    def summary(self) -> str:
        return (
            f"Processed {self.total_videos} videos: "
            f"{self.successful_videos} ok, {self.failed_videos} failed. "
            f"Total clips: {self.total_clips}, Skipped: {self.skipped_clips}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan_clips(video_duration: float, clip_duration: float, clip_stride: float) -> list[tuple[float, float]]:
    """Generate deterministic (start, end) pairs across a video.

    Returns empty list when the video is shorter than clip_duration.
    Clips are sampled starting at t=0 and advanced by clip_stride until the
    end of the video is reached.
    """
    if clip_duration <= 0 or clip_stride <= 0:
        return []
    if video_duration < clip_duration:
        return []
    clips: list[tuple[float, float]] = []
    t = 0.0
    while t + clip_duration <= video_duration + 1e-9:
        end = min(t + clip_duration, video_duration)
        clips.append((round(t, 6), round(end, 6)))
        t += clip_stride
    return clips


def _output_clip_path(source_path: Path, output_dir: Path, idx: int) -> Path:
    stem = source_path.stem
    relative = f"{stem}_clip{idx}.pt"
    return output_dir / relative


def _output_manifest_path(output_dir: Path) -> Path:
    return output_dir / "manifest.jsonl"


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}


# ---------------------------------------------------------------------------
# Video reader (with graceful fallback when pytorchvideo is unavailable)
# ---------------------------------------------------------------------------

def _make_reader(path: str):
    """Return a pytorchvideo EncodedVideo-compatible reader.

    Raises ImportError if pytorchvideo is not installed.
    """
    try:
        from pytorchvideo.data import EncodedVideo  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "pytorchvideo is required. Install with: pip install pytorchvideo"
        ) from exc
    return EncodedVideo(path, max_threads=4)  # type: ignore[no-any-return]


def _extract_frames(
    reader, start_frame: int, num_frames: int, classes: str = "all"
) -> tuple[list[int], list]:
    """Extract frames from a video. Returns (frame_indices, frames)."""
    return reader.get_clip(start_frame=start_frame, num_frames=num_frames, classes=classes)  # type: ignore[no-any-return]


def _normalize_frames(frames: list, mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                        std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> list:
    """Apply standard ImageNet normalization to list of PIL Images."""
    import torch
    from torchvision.transforms import functional as F  # type: ignore[import]

    tensors = []
    for frame in frames:
        img = F.to_tensor(frame)  # C x H x W in [0,1]
        for channel in range(3):
            img[channel] = (img[channel] - mean[channel]) / std[channel]
        tensors.append(img)
    return tensors


# ---------------------------------------------------------------------------
# Clip extraction worker
# ---------------------------------------------------------------------------

def _extract_video(
    source_path: Path,
    output_dir: Path,
    clip_duration: float,
    clip_stride: float,
    fps: int,
    spatial_size: int,
    overwrite: bool,
) -> list[ClipMetadata]:
    """Extract clips from a single video. Returns metadata for each clip."""
    results: list[ClipMetadata] = []

    if not source_path.exists():
        logger.error("Source path does not exist: %s", source_path)
        return [ClipMetadata(
            source_path=str(source_path),
            clip_path="",
            start_seconds=0.0,
            end_seconds=0.0,
            tensor_shape=[],
            video_width=0,
            video_height=0,
            video_fps=0.0,
            duration_seconds=0.0,
            num_clips=0,
            error=f"Source path does not exist: {source_path}",
        )]

    try:
        reader = _make_reader(str(source_path))
    except Exception as exc:
        logger.error("Failed to open video %s: %s", source_path, exc)
        return [ClipMetadata(
            source_path=str(source_path),
            clip_path="",
            start_seconds=0.0,
            end_seconds=0.0,
            tensor_shape=[],
            video_width=0,
            video_height=0,
            video_fps=0.0,
            duration_seconds=0.0,
            num_clips=0,
            error=f"Failed to open video: {exc}",
        )]

    try:
        video_duration = reader.video_duration
        video_fps_val = reader.video_fps
        video_width_val = reader.video_width
        video_height_val = reader.video_height
    except Exception as exc:
        logger.error("Failed to read video metadata for %s: %s", source_path, exc)
        reader.close()
        return [ClipMetadata(
            source_path=str(source_path),
            clip_path="",
            start_seconds=0.0,
            end_seconds=0.0,
            tensor_shape=[],
            video_width=0,
            video_height=0,
            video_fps=0.0,
            duration_seconds=0.0,
            num_clips=0,
            error=f"Failed to read video metadata: {exc}",
        )]

    clips = _plan_clips(video_duration, clip_duration, clip_stride)
    if not clips:
        logger.debug("Video %s too short (%.2fs) for clip_duration=%.2fs", source_path, video_duration, clip_duration)
        reader.close()
        return [ClipMetadata(
            source_path=str(source_path),
            clip_path="",
            start_seconds=0.0,
            end_seconds=0.0,
            tensor_shape=[],
            video_width=0,
            video_height=0,
            video_fps=video_fps_val,
            duration_seconds=video_duration,
            num_clips=0,
            error="Video too short for requested clip duration",
        )]

    num_frames_per_clip = int(fps * clip_duration)
    saved = 0
    skipped = 0

    for idx, (start_s, end_s) in enumerate(clips):
        # Check resume: skip if clip already exists and we're not overwriting
        clip_path = _output_clip_path(source_path, output_dir, idx)
        if clip_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            # Determine frame range
            start_frame = int(start_s * fps)
            # num_frames might exceed available frames; let get_clip handle it
            _, frames = _extract_frames(reader, start_frame, num_frames_per_clip)

            if not frames:
                skipped += 1
                continue

            # Resize each frame to spatial_size x spatial_size
            import torch
            from torchvision.transforms import functional as F  # type: ignore[import]

            normalized = []
            for frame in frames:
                img = F.resize(frame, spatial_size)
                img = F.to_tensor(img)
                # Standard normalize
                img[0] = (img[0] - 0.485) / 0.229
                img[1] = (img[1] - 0.456) / 0.224
                img[2] = (img[2] - 0.406) / 0.225
                normalized.append(img)

            tensor = torch.stack(normalized)  # C x T x H x W
            shape = list(tensor.shape)

            # Atomic write: write to temp file then rename
            fd, tmp_path = tempfile.mkstemp(suffix=".pt", dir=str(output_dir))
            try:
                torch.save(tensor, fd)
                os.replace(tmp_path, str(clip_path))
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            saved += 1
            results.append(ClipMetadata(
                source_path=str(source_path),
                clip_path=str(clip_path),
                start_seconds=start_s,
                end_seconds=end_s,
                tensor_shape=shape,
                video_width=video_width_val,
                video_height=video_height_val,
                video_fps=video_fps_val,
                duration_seconds=video_duration,
                num_clips=len(clips),
            ))

        except Exception as exc:
            logger.warning("Failed to extract clip from %s at (%.2f, %.2f): %s",
                           source_path, start_s, end_s, exc)
            skipped += 1
            results.append(ClipMetadata(
                source_path=str(source_path),
                clip_path=str(clip_path),
                start_seconds=start_s,
                end_seconds=end_s,
                tensor_shape=[],
                video_width=video_width_val,
                video_height=video_height_val,
                video_fps=video_fps_val,
                duration_seconds=video_duration,
                num_clips=len(clips),
                error=str(exc),
            ))

    reader.close()
    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> ProcessingStats:
    """Execute the full extraction pipeline. Returns processing stats."""
    stats = ProcessingStats()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect source videos (streaming, not loading all into memory)
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return stats

    video_paths = sorted(p for p in input_dir.rglob("*") if _is_video_file(p))
    stats.total_videos = len(video_paths)
    logger.info("Found %d video files in %s", stats.total_videos, input_dir)

    manifest_path = _output_manifest_path(output_dir)
    manifest_fh = None  # Open once and append

    for source_path in video_paths:
        logger.info("Processing %s", source_path)
        try:
            clip_metas = _extract_video(
                source_path=source_path,
                output_dir=output_dir,
                clip_duration=args.clip_duration,
                clip_stride=args.clip_stride,
                fps=args.fps,
                spatial_size=args.spatial_size,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            logger.error("Unhandled error processing %s: %s", source_path, exc)
            clip_metas = [ClipMetadata(
                source_path=str(source_path),
                clip_path="",
                start_seconds=0.0,
                end_seconds=0.0,
                tensor_shape=[],
                video_width=0,
                video_height=0,
                video_fps=0.0,
                duration_seconds=0.0,
                num_clips=0,
                error=f"Unhandled error: {exc}",
            )]

        # Track stats
        for meta in clip_metas:
            if meta.error:
                stats.failed_videos += 1
            else:
                stats.successful_videos += 1
            stats.skipped_clips += (1 if meta.clip_path == "" and meta.error else 0)
            stats.total_clips += (1 if meta.clip_path else 0)

        # Write to manifest
        if manifest_fh is None:
            manifest_fh = open(manifest_path, "a")

        for meta in clip_metas:
            manifest_fh.write(meta.to_jsonl() + "\n")
            logger.debug("Wrote manifest entry for %s", meta.clip_path or "(error)")

    if manifest_fh is not None:
        manifest_fh.close()

    logger.info(stats.summary())
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    stats = run(args)
    print("\n" + stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
