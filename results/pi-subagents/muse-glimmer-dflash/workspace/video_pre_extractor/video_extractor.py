"""
Video pre-extractor for PyTorchVideo training pipelines.

Lazy, production-ready pre-extraction:
- EncodedVideo for decoding
- Deterministic temporal sampling + resize/crop/normalization
- Atomic .pt writes, JSONL manifest, resume support
- Handles short/corrupt videos gracefully
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo

logger = logging.getLogger(__name__)


def build_transform(spatial_size: int, num_frames: int, mean: List[float], std: List[float]):
    """Build deterministic transform pipeline for C x T x H x W tensors."""
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean, std),
                ShortSideScale(size=spatial_size),
                CenterCropVideo(crop_size=(spatial_size, spatial_size)),
            ]
        ),
    )
    return transform


def plan_clip_starts(duration: float, clip_duration: float, stride: float) -> List[float]:
    """Return deterministic start times for clips covering the video."""
    if duration < clip_duration:
        return []
    starts = []
    t = 0.0
    # Use integer step to avoid floating errors
    while t + clip_duration <= duration + 1e-6:
        starts.append(t)
        t += stride
    return starts


def atomic_save(tensor: torch.Tensor, path: Path) -> None:
    """Write tensor atomically to avoid partial files on crash."""
    tmp = Path(tempfile.mktemp(dir=path.parent))
    torch.save(tensor, tmp)
    os.replace(tmp, path)


def process_video(
    video_path: Path,
    output_dir: Path,
    clip_duration: float,
    stride: float,
    target_fps: int,
    spatial_size: int,
    num_frames: int,
    mean: List[float],
    std: List[float],
    manifest_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Process a single video into clips. Returns summary dict."""
    video_name = video_path.name
    out_video_dir = output_dir / video_name.replace(".mp4", "")
    out_video_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "video": str(video_path),
        "clips_processed": 0,
        "clips_skipped": 0,
        "error": None,
    }

    try:
        video = EncodedVideo.from_path(str(video_path))
        duration = float(video.duration)
        logger.info("Processing %s, duration %.2f s", video_path, duration)

        starts = plan_clip_starts(duration, clip_duration, stride)
        if not starts:
            logger.warning("Video %s too short for clip duration %.2f", video_path, clip_duration)
            summary["error"] = "too_short"
            return summary

        transform = build_transform(spatial_size, num_frames, mean, std)

        for start in starts:
            end = start + clip_duration
            clip_idx = len(manifest_entries)  # simple incremental id
            clip_base = f"{video_name}_clip_{int(start)}"
            clip_path = out_video_dir / f"{clip_base}.pt"

            # Skip if already exists (resume)
            if clip_path.exists():
                summary["clips_skipped"] += 1
                continue

            try:
                clip_data = video.get_clip(start_sec=start, end_sec=end)
                # clip_data is dict with 'video' key
                transformed = transform(clip_data)
                video_tensor = transformed["video"]
                # Ensure layout C x T x H x W
                # EncodedVideo already returns this layout
                # Save atomically
                atomic_save(video_tensor, clip_path)

                entry = {
                    "source_path": str(video_path),
                    "clip_path": str(clip_path),
                    "start_sec": start,
                    "end_sec": end,
                    "tensor_shape": list(video_tensor.shape),
                    "video_metadata": {
                        "duration": duration,
                        "video_name": video_name,
                    },
                }
                manifest_entries.append(entry)
                summary["clips_processed"] += 1
            except Exception as e:
                logger.warning("Failed clip at %.2f for %s: %s", start, video_path, e)
                summary["clips_skipped"] += 1
                continue

        video.close()
    except Exception as e:
        logger.warning("Unreadable video %s: %s", video_path, e)
        summary["error"] = str(e)

    return summary


def find_mp4_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*.mp4"):
        if p.is_file():
            files.append(p)
    return files


def main_process(
    input_dir: Path,
    output_dir: Path,
    clip_duration: float,
    clip_stride: float,
    target_fps: int,
    spatial_size: int,
    overwrite: bool,
    resume: bool,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    manifest_entries: List[Dict[str, Any]] = []
    if resume and manifest_path.exists():
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    manifest_entries.append(json.loads(line))
        logger.info("Resumed from manifest with %d entries", len(manifest_entries))

    num_frames = max(1, int(round(clip_duration * target_fps)))
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    video_files = find_mp4_files(input_dir)
    logger.info("Found %d videos in %s", len(video_files), input_dir)

    summary_total = {"processed": 0, "skipped": 0, "errors": 0}

    for video_path in video_files:
        if overwrite:
            # Remove existing outputs for this video
            out_video_dir = output_dir / video_path.stem
            if out_video_dir.exists():
                import shutil
                shutil.rmtree(out_video_dir)
        summary = process_video(
            video_path,
            output_dir,
            clip_duration,
            clip_stride,
            target_fps,
            spatial_size,
            num_frames,
            mean,
            std,
            manifest_entries,
        )
        summary_total["processed"] += summary["clips_processed"]
        summary_total["skipped"] += summary["clips_skipped"]
        if summary["error"]:
            summary_total["errors"] += 1

    # Write manifest atomically
    tmp_manifest = Path(tempfile.mktemp(dir=output_dir))
    with open(tmp_manifest, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp_manifest, manifest_path)

    logger.info(
        "Done. Clips processed: %d, skipped: %d, errors: %d",
        summary_total["processed"],
        summary_total["skipped"],
        summary_total["errors"],
    )
    logger.info("Manifest written to %s", manifest_path)
