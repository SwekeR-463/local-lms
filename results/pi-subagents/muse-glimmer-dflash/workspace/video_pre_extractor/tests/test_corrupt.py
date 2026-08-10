import sys
sys.path.append("..")
from video_extractor import plan_clip_starts

def test_corrupt_handling_short():
    # short video should yield no starts
    starts = plan_clip_starts(duration=0.5, clip_duration=2.0, stride=1.0)
    assert starts == []

def test_zero_duration():
    starts = plan_clip_starts(duration=0.0, clip_duration=1.0, stride=1.0)
    assert starts == []
