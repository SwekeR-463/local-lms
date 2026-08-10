import sys
sys.path.append("..")
from video_extractor import plan_clip_starts

def test_plan_clip_starts_normal():
    starts = plan_clip_starts(duration=10.0, clip_duration=2.0, stride=2.0)
    assert starts == [0.0, 2.0, 4.0, 6.0, 8.0]

def test_plan_clip_starts_short():
    starts = plan_clip_starts(duration=1.0, clip_duration=2.0, stride=1.0)
    assert starts == []

def test_plan_clip_starts_overlapping_stride():
    starts = plan_clip_starts(duration=5.0, clip_duration=2.0, stride=1.0)
    assert starts == [0.0, 1.0, 2.0, 3.0]
