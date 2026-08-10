import json, tempfile, os, sys
sys.path.append("..")
from pathlib import Path
from video_extractor import atomic_save
import torch

def test_manifest_write_read():
    tmpdir = Path(tempfile.mkdtemp())
    manifest_path = tmpdir / "manifest.jsonl"
    entry = {
        "source_path": "/a.mp4",
        "clip_path": "/b.pt",
        "start_sec": 0.0,
        "end_sec": 2.0,
        "tensor_shape": [3, 16, 224, 224],
        "video_metadata": {"duration": 10.0, "video_name": "a.mp4"}
    }
    with open(manifest_path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    with open(manifest_path) as f:
        loaded = json.loads(next(f))
    assert loaded["source_path"] == "/a.mp4"
    assert loaded["tensor_shape"] == [3, 16, 224, 224]

def test_atomic_save():
    tmpdir = Path(tempfile.mkdtemp())
    p = tmpdir / "test.pt"
    t = torch.zeros(3, 4, 5, 6)
    atomic_save(t, p)
    loaded = torch.load(p)
    assert torch.equal(t, loaded)
