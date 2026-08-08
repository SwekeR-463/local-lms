# Video Pre-Extractor

Recursively extracts clips from MP4 files for ML training pipelines using PyTorchVideo's `EncodedVideo` API.

## Installation

```bash
pip install torch torchvision pytorchvideo
# or mock pytorchvideo in tests (see tests/)
```

## CLI Usage

```bash
python video_preextractor.py \
    --input ./raw_videos \
    --output ./clips \
    --clip_duration 2.0 \
    --clip_stride 1.5 \
    --fps 8 \
    --spatial_size 224 \
    --workers 4 \
    --overwrite
```

### Options

| Flag           | Default | Description                                  |
|----------------|---------|----------------------------------------------|
| `--input`      | (req)   | Directory containing raw video files         |
| `--output`     | (req)   | Directory for `.pt` clips and manifest       |
| `--clip_duration` | 2.0 | Duration of each clip in seconds            |
| `--clip_stride`   | 1.5 | Temporal stride between clips              |
| `--fps`         | 8     | Target frame sampling rate                 |
| `--spatial_size`| 224   | Target square dimension for each frame     |
| `--workers`     | 4     | Number of parallel workers (reserved)      |
| `--overwrite`   | false | Overwrite existing clips instead of resume |
| `--log_level`   | INFO  | Logging verbosity                          |

## Output Format

- **Clips**: `./clips/<video_stem>_clip<N>.pt` — PyTorch tensors in `C x T x H x W` layout.
- **Manifest**: `./clips/manifest.jsonl` — JSONL with one line per clip:

```json
{
  "source_path": "/data/raw/video.mp4",
  "clip_path": "/clips/video_clip0.pt",
  "start_seconds": 0.0,
  "end_seconds": 2.0,
  "tensor_shape": [3, 16, 224, 224],
  "video_width": 1920,
  "video_height": 1080,
  "video_fps": 30.0,
  "duration_seconds": 10.0,
  "num_clips": 5
}
```

## Resume Behavior

By default, existing clips are preserved. Use `--overwrite` to regenerate all clips.

## Error Handling

- Short videos (duration < clip_duration): skipped with metadata recorded.
- Corrupt/unreadable videos: logged and continued.
- Atomic writes: temp file + `os.replace` ensures partial writes don't corrupt output.
- Streaming: videos are processed one at a time, not loaded into memory.

## Testing

```bash
pytest tests/test_video_preextractor.py -v
```
