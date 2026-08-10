# Video Pre-Extractor for PyTorchVideo

Production-ready pre-extraction of MP4 videos into PyTorch tensors `C x T x H x W` for PyTorchVideo training pipelines.

## Installation

```bash
pip install -r requirements.txt
```

Requires `pytorchvideo`, `torch`, `tqdm`.

## Usage

```bash
python cli.py --input-dir /data/raw --output-dir /data/preprocessed \
  --clip-duration 2.0 --clip-stride 2.0 --target-fps 30 --spatial-size 224 \
  --resume
```

Options:
- `--input-dir`: directory with MP4 files (recursive)
- `--output-dir`: where `.pt` clips and `manifest.jsonl` are written
- `--clip-duration`: seconds per clip
- `--clip-stride`: seconds between clip starts
- `--target-fps`: temporal subsampling target
- `--spatial-size`: resize/crop size
- `--overwrite`: reprocess videos
- `--resume`: load existing manifest and skip existing clips

## Output

- `output_dir/<video_stem>/..._clip_<start>.pt` – tensors saved atomically
- `output_dir/manifest.jsonl` – one JSON per line with source_path, clip_path, start/end, tensor_shape, video metadata

## Design notes

- Uses `pytorchvideo.data.EncodedVideo` for decoding
- Deterministic temporal sampling via `UniformTemporalSubsample`
- Standard ImageNet normalization, ShortSideScale + CenterCrop
- Resume + atomic writes, graceful handling of short/corrupt videos
- Lazy implementation, no unnecessary abstractions

## Testing

```bash
pytest tests/
```
