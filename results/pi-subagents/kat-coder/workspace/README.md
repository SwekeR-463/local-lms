# Video Pre-Extractor

Recursively reads raw MP4 files and converts them into clips suitable for
PyTorchVideo / PyTorch training pipelines.

## Installation

```bash
pip install torch torchvision torchaudio
pip install pytorchvideo
pip install pytest
```

## CLI Usage

```bash
python video_preextractor.py \
    --input  /path/to/raw_videos \
    --output /path/to/extracted_clips \
    --clip-duration 2.0 \
    --clip-stride 1.0 \
    --target-fps 16 \
    --spatial-size 224 224 \
    --overwrite \
    --resume \
    --verbose
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Root directory containing raw MP4 files |
| `--output` | *(required)* | Directory for `.pt` clips and JSONL manifest |
| `--clip-duration` | `2.0` | Duration of each clip in seconds |
| `--clip-stride` | `1.0` | Temporal stride between clips in seconds |
| `--target-fps` | `16.0` | Target frames-per-second for output |
| `--spatial-size` | `224 224` | Output height and width |
| `--workers` | `1` | Number of worker threads |
| `--overwrite` | — | Overwrite existing clips |
| `--resume` | — | Skip already-extracted clips |
| `--log-file` | — | Optional log file path |
| `--verbose` | — | Enable debug-level logging |

## Manifest Format

Each line in `manifest.jsonl` is a JSON object:

```json
{
  "source_path": "/data/raw/video1.mp4",
  "clip_path": "video1_clip000000.pt",
  "start_ts": 0.0,
  "end_ts": 2.0,
  "tensor_shape": [3, 32, 224, 224],
  "fps": 16.0,
  "num_frames": 32,
  "spatial_size": [224, 224],
  "success": true
}
```

## Tests

```bash
pytest tests/test_video_preextractor.py -v
```
