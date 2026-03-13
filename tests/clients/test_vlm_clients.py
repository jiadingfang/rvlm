"""Tests for vision_utils and VLM client content formatting (no real API calls)."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from rvlm.clients.vision_utils import (
    chunk_frames,
    frame_to_base64,
    frames_to_anthropic_content,
    frames_to_openai_content,
    resize_frame,
)


def _random_frame(h: int, w: int) -> np.ndarray:
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestFrameToBase64:
    def test_frame_to_base64_roundtrip(self):
        """Encode a random numpy array to base64, decode it, and verify dimensions match."""
        frame = _random_frame(48, 64)
        b64 = frame_to_base64(frame)
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        assert img.size == (64, 48)  # PIL returns (w, h)


class TestResizeFrame:
    def test_resize_frame_preserves_aspect(self):
        """640x480 with max_side=320 should produce 320x240."""
        frame = _random_frame(480, 640)
        resized = resize_frame(frame, max_side=320)
        assert resized.shape[0] == 240
        assert resized.shape[1] == 320

    def test_resize_frame_noop_small(self):
        """Frame already smaller than max_side should be returned unchanged."""
        frame = _random_frame(100, 100)
        resized = resize_frame(frame, max_side=512)
        assert resized is frame


class TestOpenAIContentBlock:
    def test_openai_content_block_format(self):
        """Verify structure matches OpenAI vision spec: image_url blocks + text block."""
        frames = [_random_frame(64, 64), _random_frame(64, 64)]
        content = frames_to_openai_content(frames, text="Describe this.", detail="low")

        assert len(content) == 3  # 2 images + 1 text

        for img_block in content[:2]:
            assert img_block["type"] == "image_url"
            assert "url" in img_block["image_url"]
            assert img_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
            assert img_block["image_url"]["detail"] == "low"

        text_block = content[-1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Describe this."


class TestAnthropicContentBlock:
    def test_anthropic_content_block_format(self):
        """Verify Anthropic format: image source blocks with base64 + text block."""
        frames = [_random_frame(64, 64)]
        content = frames_to_anthropic_content(frames, text="Count objects.")

        assert len(content) == 2  # 1 image + 1 text

        img_block = content[0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/jpeg"
        assert isinstance(img_block["source"]["data"], str)

        text_block = content[-1]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Count objects."

    def test_anthropic_rejects_over_20_frames(self):
        """ValueError should be raised when more than 20 frames are passed."""
        frames = [_random_frame(32, 32) for _ in range(21)]
        with pytest.raises(ValueError, match="max 20 images"):
            frames_to_anthropic_content(frames, text="Too many.")


class TestChunkFrames:
    def test_chunk_frames(self):
        """10 items with chunk size 3 should yield [[0,1,2],[3,4,5],[6,7,8],[9]]."""
        items = list(range(10))
        chunks = chunk_frames(items, max_per_chunk=3)
        assert chunks == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    def test_chunk_frames_exact_multiple(self):
        """6 items with chunk size 3 should yield [[0,1,2],[3,4,5]]."""
        items = list(range(6))
        chunks = chunk_frames(items, max_per_chunk=3)
        assert chunks == [[0, 1, 2], [3, 4, 5]]

    def test_chunk_frames_single(self):
        """2 items with chunk size 10 should yield [[0,1]]."""
        items = list(range(2))
        chunks = chunk_frames(items, max_per_chunk=10)
        assert chunks == [[0, 1]]
