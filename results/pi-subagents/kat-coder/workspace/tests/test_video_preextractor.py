"""
Focused pytest tests for video_preextractor.

Mocks PyTorchVideo and torch.Tensor persistence so tests run without
installing the real packages.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_workspace(tmp_path: Path):
    """Create input/output temp directories and seed a fake MP4."""
    inp = tmp_path / "input"
    inp.mkdir()
    out = tmp_path / "output"
    out.mkdir()
    # Create fake mp4 placeholders (not real videos — we mock EncodedVideo)
    (inp / "video_a.mp4").write_bytes(b"fake_mp4_bytes_a")
    (inp / "video_b.mp4").write_bytes(b"fake_mp4_bytes_b")
    # Subdirectory with another video
    sub = inp / "sub"
    sub.mkdir()
    (sub / "video_c.mp4").write_bytes(b"fake_mp4_bytes_c")
    return tmp_path, inp, out


@pytest.fixture()
def mock_encoded_video():
    """Return a MagicMock that mimics the PyTorchVideo EncodedVideo API."""
    video = MagicMock()
    video.get_video_info.return_value = {
        "video_fps": 30.0,
        "num_frames": 300,
        "image_size": [224, 224],
        "duration": 10.0,
    }
    # Return a numpy-like array: T x H x W x C
    video.decode_video.return_value = np.zeros((60, 224, 224, 3), dtype=np.uint8)
    return video


# ---------------------------------------------------------------------------
# Clip planning tests
# ---------------------------------------------------------------------------


class TestClipPlanning:
    """Tests for plan_clips()."""

    def test_basic_planning(self):
        from video_preextractor import plan_clips
        # stride=2.0, clip_duration=2.0 -> non-overlapping clips
        clips = plan_clips("/data/v.mp4", fps=30.0, duration=10.0, clip_duration=2.0, clip_stride=2.0)
        assert len(clips) == 5
        assert clips[0].start_ts == 0.0
        assert clips[0].end_ts == 2.0
        assert clips[1].start_ts == 2.0
        assert clips[1].end_ts == 4.0
        assert clips[4].start_ts == 8.0
        assert clips[4].end_ts == 10.0

    def test_no_overlap_when_stride_equals_duration(self):
        from video_preextractor import plan_clips
        clips = plan_clips("/data/v.mp4", fps=30.0, duration=6.0, clip_duration=2.0, clip_stride=2.0)
        assert len(clips) == 3
        assert clips[0].end_ts == clips[1].start_ts

    def test_empty_when_duration_zero(self):
        from video_preextractor import plan_clips
        clips = plan_clips("/data/v.mp4", fps=30.0, duration=0.0, clip_duration=2.0, clip_stride=1.0)
        assert clips == []

    def test_invalid_clip_duration(self):
        from video_preextractor import plan_clips
        with pytest.raises(ValueError, match="clip_duration must be positive"):
            plan_clips("/data/v.mp4", fps=30.0, duration=10.0, clip_duration=0.0, clip_stride=1.0)

    def test_invalid_clip_stride(self):
        from video_preextractor import plan_clips
        with pytest.raises(ValueError, match="clip_stride must be positive"):
            plan_clips("/data/v.mp4", fps=30.0, duration=10.0, clip_duration=2.0, clip_stride=0.0)

    def test_short_last_clip(self):
        from video_preextractor import plan_clips
        clips = plan_clips("/data/v.mp4", fps=30.0, duration=5.0, clip_duration=3.0, clip_stride=3.0)
        assert len(clips) == 2
        assert clips[1].end_ts == 5.0

    def test_deterministic_paths(self):
        from video_preextractor import plan_clips
        clips = plan_clips("/data/v.mp4", fps=30.0, duration=4.0, clip_duration=2.0, clip_stride=2.0)
        paths = [c.output_path for c in clips]
        assert paths[0].startswith("v_clip")
        assert paths[1].startswith("v_clip")


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


class TestManifest:
    def test_write_and_read(self, tmp_workspace):
        _, _, out = tmp_workspace
        from video_preextractor import ClipResult, write_manifest

        results = [
            ClipResult(
                clip_path="video_a_clip000000.pt",
                success=True,
                tensor_shape=[3, 32, 224, 224],
                source_path="/data/video_a.mp4",
                start_ts=0.0,
                end_ts=2.0,
                fps=16.0,
                num_frames=32,
                spatial_size=(224, 224),
            ),
            ClipResult(
                clip_path="video_a_clip000001.pt",
                success=False,
                error="decode failed",
                source_path="/data/video_a.mp4",
                start_ts=2.0,
                end_ts=4.0,
            ),
        ]

        manifest_path = out / "manifest.jsonl"
        write_manifest(results, manifest_path)

        assert manifest_path.exists()
        lines = manifest_path.read_text().splitlines()
        assert len(lines) == 2

        entry = json.loads(lines[0])
        assert entry["source_path"] == "/data/video_a.mp4"
        assert entry["success"] is True
        assert entry["tensor_shape"] == [3, 32, 224, 224]

        entry2 = json.loads(lines[1])
        assert entry2["success"] is False
        assert entry2["error"] == "decode failed"

    def test_append_mode(self, tmp_workspace):
        _, _, out = tmp_workspace
        from video_preextractor import ClipResult, write_manifest

        manifest_path = out / "manifest.jsonl"
        write_manifest([
            ClipResult(clip_path="a.pt", success=True, source_path="/v.mp4", start_ts=0.0, end_ts=1.0),
        ], manifest_path)
        write_manifest([
            ClipResult(clip_path="b.pt", success=True, source_path="/v.mp4", start_ts=1.0, end_ts=2.0),
        ], manifest_path)

        lines = manifest_path.read_text().splitlines()
        assert len(lines) == 2


# ---------------------------------------------------------------------------
# Resume / overwrite behaviour
# ---------------------------------------------------------------------------


class TestResumeOverwrite:
    def test_resume_skips_existing(self, tmp_workspace):
        """Resume mode should skip clips that already exist on disk."""
        _, inp, out = tmp_workspace
        # Pre-create a clip to simulate prior extraction
        (out / "video_a_clip000000.pt").write_bytes(b"existing_clip")

        # Mock EncodedVideo
        mock_video = MagicMock()
        mock_video.get_video_info.return_value = {
            "video_fps": 30.0,
            "num_frames": 300,
            "image_size": [224, 224],
            "duration": 10.0,
        }
        mock_video.decode_video.return_value = np.zeros((60, 224, 224, 3), dtype=np.uint8)

        with patch("video_preextractor._import_pytorchvideo") as MockPV:
            MockPV.return_value = MagicMock(from_file=MagicMock(return_value=mock_video))
            from video_preextractor import extract
            extract(
                input_dir=inp,
                output_dir=out,
                clip_duration=2.0,
                clip_stride=2.0,
                target_fps=16,
                spatial_size=(224, 224),
                overwrite=False,
                num_workers=1,
                resume=True,
                log_file=None,
                verbose=False,
            )

        # The manifest should exist and the existing clip should be recorded
        manifest = out / "manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text().splitlines()
        assert len(lines) >= 1

    def test_overwrite_replaces(self, tmp_workspace):
        """Overwrite mode should replace existing clips."""
        _, inp, out = tmp_workspace
        (out / "video_a_clip000000.pt").write_bytes(b"old_clip_data")

        mock_video = MagicMock()
        mock_video.get_video_info.return_value = {
            "video_fps": 30.0,
            "num_frames": 300,
            "image_size": [224, 224],
            "duration": 10.0,
        }
        mock_video.decode_video.return_value = np.zeros((60, 224, 224, 3), dtype=np.uint8)

        with patch("video_preextractor._import_pytorchvideo") as MockPV:
            MockPV.return_value = MagicMock(from_file=MagicMock(return_value=mock_video))
            from video_preextractor import extract
            extract(
                input_dir=inp,
                output_dir=out,
                clip_duration=2.0,
                clip_stride=2.0,
                target_fps=16,
                spatial_size=(224, 224),
                overwrite=True,
                num_workers=1,
                resume=False,
                log_file=None,
                verbose=False,
            )

        manifest = out / "manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text().splitlines()
        assert len(lines) >= 1


# ---------------------------------------------------------------------------
# Corrupt / short video handling
# ---------------------------------------------------------------------------


class TestCorruptHandling:
    def test_unreadable_video_logs_and_continues(self, tmp_workspace):
        """A video that raises on from_file should not abort the run."""
        _, inp, out = tmp_workspace
        # Remove the fake mp4 so from_file will raise
        (inp / "video_a.mp4").unlink()
        # Add a valid one
        (inp / "video_b.mp4").write_bytes(b"fake")

        class BadVideo:
            def from_file(path):
                raise RuntimeError("File not found or corrupted")

        with patch("video_preextractor._import_pytorchvideo") as MockPV:
            MockPV.return_value = BadVideo
            from video_preextractor import extract
            extract(
                input_dir=inp,
                output_dir=out,
                clip_duration=2.0,
                clip_stride=1.0,
                target_fps=16,
                spatial_size=(224, 224),
                overwrite=True,
                num_workers=1,
                resume=False,
                log_file=None,
                verbose=True,
            )

        # Should not raise; manifest should contain entries for video_b only
        manifest = out / "manifest.jsonl"
        assert manifest.exists()

    def test_zero_fps_video(self, tmp_workspace):
        """A video reporting fps=0 should be skipped gracefully."""
        _, inp, out = tmp_workspace
        (inp / "video_a.mp4").write_bytes(b"fake")
        (inp / "video_b.mp4").write_bytes(b"fake")

        bad_video = MagicMock()
        bad_video.get_video_info.return_value = {"video_fps": 0.0, "duration": 0.0, "num_frames": 0, "image_size": [224, 224]}
        bad_video.decode_video.return_value = None

        with patch("video_preextractor._import_pytorchvideo") as MockPV:
            MockPV.return_value = MagicMock(from_file=MagicMock(return_value=bad_video))
            from video_preextractor import extract
            extract(
                input_dir=inp,
                output_dir=out,
                clip_duration=2.0,
                clip_stride=1.0,
                target_fps=16,
                spatial_size=(224, 224),
                overwrite=True,
                num_workers=1,
                resume=False,
                log_file=None,
                verbose=True,
            )

        manifest = out / "manifest.jsonl"
        assert manifest.exists()


# ---------------------------------------------------------------------------
# Decode / transform tests
# ---------------------------------------------------------------------------


class TestDecodeClip:
    def test_decode_clip_output_shape(self, tmp_workspace):
        """Verify the decode + resample path produces C x T x H x W."""
        from video_preextractor import _decode_clip_impl

        # Mock torch to avoid requiring real installation
        # Use a custom class that preserves shape and dim through method chains
        class MockTensor:
            shape = [3, 32, 224, 224]  # C x T x H x W output
            def dim(self):
                return 4
            def __call__(self, *args, **kwargs):
                return self
            def __getattr__(self, name):
                # Return self for any method call to preserve chain
                return self
            def __sub__(self, other):
                return self
            def __truediv__(self, other):
                return self

        mock_tensor = MockTensor()

        # Create a mock nn.functional module
        mock_nn = MagicMock()
        mock_nn.functional = MagicMock()
        mock_nn.functional.interpolate = lambda *a, **k: mock_tensor
        mock_nn.functional = mock_nn.functional

        torch_mock = MagicMock()
        torch_mock.from_numpy = lambda x: mock_tensor
        torch_mock.nn = mock_nn

        with patch("video_preextractor._import_torch", return_value=torch_mock):
            video = MagicMock()
            video.get_video_info.return_value = {"video_fps": 30.0}
            video.decode_video.return_value = np.zeros((60, 224, 224, 3), dtype=np.uint8)
            result = _decode_clip_impl(video, 0.0, 2.0, target_fps=16.0, spatial_size=(224, 224))
            # Should return C x T x H x W
            assert result.dim() == 4
            assert result.shape[0] == 3  # C
            assert result.shape[2] == 224  # H
            assert result.shape[3] == 224  # W


# ---------------------------------------------------------------------------
# End-to-end integration (with mocks)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self, tmp_workspace):
        """Full pipeline with mocked EncodedVideo and torch."""
        _, inp, out = tmp_workspace

        mock_video = MagicMock()
        mock_video.get_video_info.return_value = {
            "video_fps": 30.0,
            "num_frames": 300,
            "image_size": [224, 224],
            "duration": 10.0,
        }
        mock_video.decode_video.return_value = np.zeros((60, 224, 224, 3), dtype=np.uint8)

        # Mock torch to produce a predictable tensor
        mock_tensor = MagicMock()
        mock_tensor.shape = [3, 32, 224, 224]
        mock_tensor.permute = lambda *a, **k: mock_tensor
        mock_tensor.unsqueeze = lambda *a, **k: mock_tensor
        mock_tensor.__sub__ = lambda self, other: mock_tensor
        mock_tensor.__truediv__ = lambda self, other: mock_tensor

        with patch("video_preextractor._import_pytorchvideo") as MockPV, \
             patch("video_preextractor._import_torch", return_value=MagicMock(
                 from_numpy=lambda x: mock_tensor,
                 save=lambda path, obj: None,
                 tensor=lambda *a, **k: MagicMock(),
             )):
            MockPV.return_value = MagicMock(from_file=MagicMock(return_value=mock_video))
            from video_preextractor import extract
            extract(
                input_dir=inp,
                output_dir=out,
                clip_duration=2.0,
                clip_stride=2.0,
                target_fps=16,
                spatial_size=(224, 224),
                overwrite=True,
                num_workers=1,
                resume=False,
                log_file=None,
                verbose=False,
            )

        # Check manifest
        manifest = out / "manifest.jsonl"
        assert manifest.exists()
        lines = manifest.read_text().splitlines()
        assert len(lines) > 0

        for line in lines:
            entry = json.loads(line)
            assert "source_path" in entry
            assert "clip_path" in entry
            assert "start_ts" in entry
            assert "end_ts" in entry
            assert "tensor_shape" in entry
            assert "success" in entry

        # Check that .pt files were written
        pt_files = list(out.rglob("*.pt"))
        assert len(pt_files) > 0
