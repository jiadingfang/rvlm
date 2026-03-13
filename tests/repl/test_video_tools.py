"""Tests for video_utils and counting_tools using a synthetic video fixture (no API calls)."""

import numpy as np
import pytest

from benchmarks.videocount.eval import parse_count_from_response
from rvlm.utils.counting_tools import make_counting_tools
from rvlm.utils.video_utils import (
    get_video_metadata,
    resize_frames,
    sample_clip,
    sample_frames,
    split_into_segments,
)


class MockVLMClient:
    def __init__(self, responses):
        self.responses = iter(responses)

    def vlm_completion(self, frames, prompt, **kwargs):
        return next(self.responses)


# ── Video metadata & sampling ────────────────────────────────────────────────


class TestGetVideoMetadata:
    def test_get_video_metadata(self, synthetic_video_path):
        """Verify returns dict with expected keys and correct approximate values."""
        meta = get_video_metadata(synthetic_video_path)
        assert isinstance(meta, dict)
        expected_keys = {"duration", "fps", "total_frames", "width", "height", "format"}
        assert expected_keys == set(meta.keys())
        assert abs(meta["duration"] - 5.0) < 0.5
        assert meta["fps"] == 10.0
        assert meta["total_frames"] == 50
        assert meta["width"] == 320
        assert meta["height"] == 240
        assert meta["format"] == "mp4"


class TestSampleFrames:
    def test_sample_frames_uniform(self, synthetic_video_path):
        """Uniform sampling of 8 frames should return 8 RGB frames of (240, 320, 3)."""
        frames = sample_frames(synthetic_video_path, n=8, strategy="uniform")
        assert len(frames) == 8
        for f in frames:
            assert f.shape == (240, 320, 3)
            assert f.dtype == np.uint8

    def test_sample_frames_scene_change(self, synthetic_video_path):
        """Scene change sampling of 5 frames should return 5 frames."""
        frames = sample_frames(synthetic_video_path, n=5, strategy="scene_change")
        assert len(frames) == 5
        for f in frames:
            assert f.shape == (240, 320, 3)

    def test_sample_frames_with_range(self, synthetic_video_path):
        """Sampling with start_sec=1, end_sec=3 should return frames."""
        frames = sample_frames(
            synthetic_video_path, n=4, strategy="uniform", start_sec=1.0, end_sec=3.0
        )
        assert len(frames) == 4
        for f in frames:
            assert f.shape == (240, 320, 3)


class TestSampleClip:
    def test_sample_clip_range(self, synthetic_video_path):
        """start=0, end=2, fps=2 should return approximately 4 frames."""
        frames = sample_clip(synthetic_video_path, start_sec=0.0, end_sec=2.0, fps=2.0)
        assert len(frames) == 4

    def test_sample_clip_cap_64(self, synthetic_video_path):
        """fps * duration > 64 should raise ValueError."""
        with pytest.raises(ValueError, match="64"):
            sample_clip(synthetic_video_path, start_sec=0.0, end_sec=5.0, fps=20.0)


class TestSplitIntoSegments:
    def test_split_into_segments(self, synthetic_video_path):
        """segment_sec=2 on a 5s video should return ~3 segments covering full duration."""
        segments = split_into_segments(synthetic_video_path, segment_sec=2.0)
        assert len(segments) == 3
        assert segments[0][0] == 0.0
        assert abs(segments[-1][1] - 5.0) < 0.5
        # Segments should be contiguous
        for i in range(1, len(segments)):
            assert segments[i][0] == segments[i - 1][1]


class TestResizeFrames:
    def test_resize_frames(self, synthetic_video_path):
        """Resized frames should have longest side <= max_side."""
        frames = sample_frames(synthetic_video_path, n=3, strategy="uniform")
        resized = resize_frames(frames, max_side=128)
        assert len(resized) == 3
        for f in resized:
            assert max(f.shape[0], f.shape[1]) <= 128
            # Aspect ratio preserved: original is 320x240 -> 128x96
            assert f.shape[1] == 128
            assert f.shape[0] == 96


# ── Counting tools (mocked VLM) ─────────────────────────────────────────────


class TestVLMCount:
    def test_vlm_count_parses_integer(self):
        """vlm_count should parse an integer from VLM response."""
        client = MockVLMClient(["5\n"])
        tools = make_counting_tools(client)
        vlm_count = tools["vlm_count"]["tool"]
        result = vlm_count(frames=[np.zeros((10, 10, 3), dtype=np.uint8)], question="How many?")
        assert result == 5

    def test_vlm_count_raises_on_non_integer(self):
        """vlm_count should raise ValueError when VLM returns non-integer text."""
        client = MockVLMClient(["several"])
        tools = make_counting_tools(client)
        vlm_count = tools["vlm_count"]["tool"]
        with pytest.raises(ValueError, match="parseable integer"):
            vlm_count(frames=[np.zeros((10, 10, 3), dtype=np.uint8)], question="How many?")


class TestCountInSegments:
    def test_count_in_segments_max(self, synthetic_video_path):
        """With mock returning [3, 7, 5], max aggregation should return 7."""
        responses = ["3", "7", "5"]
        client = MockVLMClient(responses)
        tools = make_counting_tools(client)
        count_in_segments = tools["count_in_segments"]["tool"]
        result = count_in_segments(
            video_path=synthetic_video_path,
            question="How many objects?",
            segment_sec=2.0,
            frames_per_segment=2,
            aggregation="max",
        )
        assert result == 7

    def test_count_in_segments_sum(self, synthetic_video_path):
        """With mock returning [3, 7, 5], sum aggregation should return 15."""
        responses = ["3", "7", "5"]
        client = MockVLMClient(responses)
        tools = make_counting_tools(client)
        count_in_segments = tools["count_in_segments"]["tool"]
        result = count_in_segments(
            video_path=synthetic_video_path,
            question="How many objects?",
            segment_sec=2.0,
            frames_per_segment=2,
            aggregation="sum",
        )
        assert result == 15


# ── Eval parse function ──────────────────────────────────────────────────────


class TestParseCountFromResponse:
    def test_parse_count_from_response(self):
        """Test the eval.py parse function with various inputs."""
        assert parse_count_from_response("The answer is 42.") == 42
        assert parse_count_from_response("I see 3 dogs and 5 cats") == 5
        assert parse_count_from_response("7") == 7
        assert parse_count_from_response("There are about 12 items total") == 12

        with pytest.raises(ValueError, match="No integer found"):
            parse_count_from_response("no numbers here")

        with pytest.raises(ValueError, match="No integer found"):
            parse_count_from_response("")
