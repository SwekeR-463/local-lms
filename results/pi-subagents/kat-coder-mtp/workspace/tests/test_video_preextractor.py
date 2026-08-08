"""Focused pytest tests for video_preextractor.

Mock pytorchvideo boundaries since it may not be installed in the test environment.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
import video_preextractor as vpe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "clips"
    out.mkdir()
    return out


@pytest.fixture()
def tmp_input_dir(tmp_path: Path) -> Path:
    inp = tmp_path / "raw"
    inp.mkdir()
    return inp


# ---------------------------------------------------------------------------
# _plan_clips
# ---------------------------------------------------------------------------

class TestPlanClips:
    def test_basic(self):
        clips = vpe._plan_clips(10.0, 2.0, 1.5)
        assert len(clips) == 6
        assert clips[0] == (0.0, 2.0)
        assert clips[1] == (1.5, 3.5)
        assert clips[2] == (3.0, 5.0)
        assert clips[3] == (4.5, 6.5)
        assert clips[4] == (6.0, 8.0)
        assert clips[5] == (7.5, 9.5)

    def test_stride_exceeds_duration(self):
        clips = vpe._plan_clips(5.0, 3.0, 4.0)
        assert len(clips) == 1
        assert clips[0] == (0.0, 3.0)

    def test_video_shorter_than_clip(self):
        clips = vpe._plan_clips(1.0, 2.0, 1.5)
        assert clips == []

    def test_zero_duration(self):
        clips = vpe._plan_clips(5.0, 0.0, 1.5)
        assert clips == []

    def test_zero_stride(self):
        clips = vpe._plan_clips(5.0, 2.0, 0.0)
        assert clips == []

    def test_exact_fit(self):
        clips = vpe._plan_clips(4.0, 2.0, 2.0)
        assert len(clips) == 2
        assert clips[0] == (0.0, 2.0)
        assert clips[1] == (2.0, 4.0)


# ---------------------------------------------------------------------------
# ClipMetadata / manifest
# ---------------------------------------------------------------------------

class TestClipMetadata:
    def test_to_dict(self):
        meta = vpe.ClipMetadata(
            source_path="/a.mp4",
            clip_path="/out/a_clip0.pt",
            start_seconds=0.0,
            end_seconds=2.0,
            tensor_shape=[3, 16, 224, 224],
            video_width=1920,
            video_height=1080,
            video_fps=30.0,
            duration_seconds=10.0,
            num_clips=5,
        )
        d = meta.to_dict()
        assert d["source_path"] == "/a.mp4"
        assert d["tensor_shape"] == [3, 16, 224, 224]

    def test_to_jsonl(self):
        meta = vpe.ClipMetadata(
            source_path="/a.mp4",
            clip_path="/out/a_clip0.pt",
            start_seconds=0.0,
            end_seconds=2.0,
            tensor_shape=[3, 16, 224, 224],
            video_width=1920,
            video_height=1080,
            video_fps=30.0,
            duration_seconds=10.0,
            num_clips=5,
        )
        line = meta.to_jsonl()
        parsed = json.loads(line)
        assert parsed["clip_path"] == "/out/a_clip0.pt"

    def test_error_field(self):
        meta = vpe.ClipMetadata(
            source_path="/bad.mp4",
            clip_path="",
            start_seconds=0.0,
            end_seconds=0.0,
            tensor_shape=[],
            video_width=0,
            video_height=0,
            video_fps=0.0,
            duration_seconds=0.0,
            num_clips=0,
            error="some error",
        )
        assert meta.error == "some error"
        assert meta.to_dict()["error"] == "some error"


# ---------------------------------------------------------------------------
# Resume behavior
# ---------------------------------------------------------------------------

class TestResumeBehavior:
    def test_resume_skips_existing(self, tmp_output_dir: Path, tmp_input_dir: Path):
        """When --overwrite is False, existing clips should be preserved."""
        # Pre-create a clip
        clip_path = tmp_output_dir / "video1_clip0.pt"
        clip_path.write_bytes(b"existing")

        # Mock the video reader to produce one clip
        mock_reader = MagicMock()
        mock_reader.video_duration = 5.0
        mock_reader.video_fps = 8.0
        mock_reader.video_width = 224
        mock_reader.video_height = 224
        mock_reader.get_clip.return_value = ([0], [MagicMock()])

        with patch("video_preextractor._make_reader", return_value=mock_reader):
            metas = vpe._extract_video(
                source_path=tmp_input_dir / "video1.mp4",
                output_dir=tmp_output_dir,
                clip_duration=2.0,
                clip_stride=1.5,
                fps=8,
                spatial_size=224,
                overwrite=False,
            )

        # Should skip the existing clip
        assert all(m.clip_path == str(clip_path) for m in metas if m.clip_path)
        # The clip should still exist on disk
        assert clip_path.exists()

    def test_overwrite_replaces(self, tmp_output_dir: Path, tmp_input_dir: Path):
        """When --overwrite is True, existing clips should be replaced."""
        # Create the source video file so the path exists check passes
        (tmp_input_dir / "video1.mp4").write_bytes(b"\x00" * 100)
        clip_path = tmp_output_dir / "video1_clip0.pt"
        clip_path.write_bytes(b"old")

        mock_reader = MagicMock()
        mock_reader.video_duration = 5.0
        mock_reader.video_fps = 8.0
        mock_reader.video_width = 224
        mock_reader.video_height = 224
        mock_reader.get_clip.return_value = ([0], [MagicMock()])

        with patch("video_preextractor._make_reader", return_value=mock_reader):
            metas = vpe._extract_video(
                source_path=tmp_input_dir / "video1.mp4",
                output_dir=tmp_output_dir,
                clip_duration=2.0,
                clip_stride=1.5,
                fps=8,
                spatial_size=224,
                overwrite=True,
            )

        # Should have overwritten
        assert any(m.clip_path == str(clip_path) for m in metas if m.clip_path)
        assert clip_path.exists()


# ---------------------------------------------------------------------------
# Corrupt / unreadable input handling
# ---------------------------------------------------------------------------

class TestCorruptInput:
    def test_missing_source(self, tmp_output_dir: Path):
        """A non-existent source path should produce an error metadata entry."""
        metas = vpe._extract_video(
            source_path=Path("/nonexistent/path/video.mp4"),
            output_dir=tmp_output_dir,
            clip_duration=2.0,
            clip_stride=1.5,
            fps=8,
            spatial_size=224,
            overwrite=False,
        )
        assert len(metas) == 1
        assert metas[0].error is not None
        assert "does not exist" in metas[0].error

    def test_reader_failure(self, tmp_output_dir: Path, tmp_input_dir: Path):
        """A reader that raises on construction should be caught."""
        # Create the source file so the "does not exist" check doesn't short-circuit
        source = tmp_input_dir / "video.mp4"
        source.write_bytes(b"\x00" * 100)
        with patch("video_preextractor._make_reader", side_effect=RuntimeError("boom")):
            metas = vpe._extract_video(
                source_path=source,
                output_dir=tmp_output_dir,
                clip_duration=2.0,
                clip_stride=1.5,
                fps=8,
                spatial_size=224,
                overwrite=False,
            )
        assert len(metas) == 1
        assert metas[0].error is not None
        assert "boom" in metas[0].error

    def test_metadata_failure(self, tmp_output_dir: Path):
        """A reader that fails on video_duration should be caught."""
        mock_reader = MagicMock()
        mock_reader.video_duration = 5.0
        mock_reader.video_fps = 8.0
        mock_reader.video_width = 224
        mock_reader.video_height = 224
        mock_reader.get_clip.return_value = ([0], [MagicMock()])

        # Simulate failure when accessing metadata
        mock_reader_raises = MagicMock()
        mock_reader_raises.video_duration = 5.0
        mock_reader_raises.video_fps = 8.0
        mock_reader_raises.video_width = 224
        mock_reader_raises.video_height = 224
        mock_reader_raises.get_clip.side_effect = RuntimeError("metadata gone")
        mock_reader_raises.close = MagicMock()

        with patch("video_preextractor._make_reader", return_value=mock_reader_raises):
            metas = vpe._extract_video(
                source_path=Path("/fake/video.mp4"),
                output_dir=tmp_output_dir,
                clip_duration=2.0,
                clip_stride=1.5,
                fps=8,
                spatial_size=224,
                overwrite=False,
            )
        assert len(metas) == 1
        assert metas[0].error is not None


# ---------------------------------------------------------------------------
# Clip planning edge cases
# ---------------------------------------------------------------------------

class TestClipPlanning:
    def test_single_clip(self):
        clips = vpe._plan_clips(3.0, 3.0, 1.0)
        assert len(clips) == 1
        assert clips[0] == (0.0, 3.0)

    def test_multiple_overlapping(self):
        clips = vpe._plan_clips(10.0, 2.0, 0.5)
        assert len(clips) == 17  # 0, 0.5, 1.0, ..., 8.0
        # Verify deterministic ordering
        for i in range(len(clips) - 1):
            assert clips[i][0] < clips[i + 1][0]

    def test_very_short_video(self):
        clips = vpe._plan_clips(0.5, 2.0, 1.0)
        assert clips == []


# ---------------------------------------------------------------------------
# Is video file heuristic
# ---------------------------------------------------------------------------

class TestIsVideoFile:
    def test_mp4(self):
        assert vpe._is_video_file(Path("video.mp4")) is True

    def test_mkv(self):
        assert vpe._is_video_file(Path("video.mkv")) is True

    def test_txt(self):
        assert vpe._is_video_file(Path("readme.txt")) is False

    def test_nested(self):
        assert vpe._is_video_file(Path("sub/video.avi")) is True


# ---------------------------------------------------------------------------
# Processing stats summary
# ---------------------------------------------------------------------------

class TestProcessingStats:
    def test_summary(self):
        stats = vpe.ProcessingStats(total_videos=10, successful_videos=8,
                                    failed_videos=2, total_clips=50, skipped_clips=5)
        s = stats.summary()
        assert "10 videos" in s
        assert "8 ok" in s
        assert "2 failed" in s
        assert "50" in s
        assert "5" in s
