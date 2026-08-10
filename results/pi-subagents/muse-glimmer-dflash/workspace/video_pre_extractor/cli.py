#!/usr/bin/env python3
"""
CLI for video pre-extractor.
"""

import argparse
from pathlib import Path
from video_extractor import main_process


def parse_args():
    p = argparse.ArgumentParser(description="Video pre-extractor for PyTorchVideo")
    p.add_argument("--input-dir", required=True, type=str, help="Input directory with MP4 files")
    p.add_argument("--output-dir", required=True, type=str, help="Output directory for .pt clips")
    p.add_argument("--clip-duration", type=float, default=2.0, help="Clip duration in seconds")
    p.add_argument("--clip-stride", type=float, default=2.0, help="Stride between clips in seconds")
    p.add_argument("--target-fps", type=int, default=30, help="Target FPS for temporal subsampling")
    p.add_argument("--spatial-size", type=int, default=224, help="Spatial size for resize/crop")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p.add_argument("--resume", action="store_true", help="Resume from existing manifest")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    main_process(
        input_dir=input_dir,
        output_dir=output_dir,
        clip_duration=args.clip_duration,
        clip_stride=args.clip_stride,
        target_fps=args.target_fps,
        spatial_size=args.spatial_size,
        overwrite=args.overwrite,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
