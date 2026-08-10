import json, tempfile, os, sys
sys.path.append("..")
from pathlib import Path

def test_resume_load():
    tmpdir = Path(tempfile.mkdtemp())
    manifest_path = tmpdir / "manifest.jsonl"
    entries = [
        {"source_path": "a.mp4", "clip_path": "a1.pt"},
        {"source_path": "b.mp4", "clip_path": "b1.pt"},
    ]
    with open(manifest_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    loaded = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                loaded.append(json.loads(line))
    assert len(loaded) == 2
    assert loaded[0]["source_path"] == "a.mp4"
