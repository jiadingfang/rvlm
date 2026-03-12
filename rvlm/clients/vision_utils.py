"""Shared utilities for encoding video frames for VLM API calls."""

import base64
import io

import numpy as np
from PIL import Image


def frame_to_base64(frame: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    """Convert numpy HWC uint8 frame to base64-encoded JPEG/PNG string."""
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    save_kwargs = {"format": format}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode()


def resize_frame(frame: np.ndarray, max_side: int = 512) -> np.ndarray:
    """Resize frame so the longer side is at most max_side, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    img = Image.fromarray(frame)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)


def frames_to_openai_content(
    frames: list[np.ndarray],
    text: str,
    detail: str = "low",
    max_side: int = 512,
) -> list[dict]:
    """Build OpenAI-format content block list with images + text."""
    content = []
    for frame in frames:
        frame = resize_frame(frame, max_side)
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    content.append({"type": "text", "text": text})
    return content


def frames_to_anthropic_content(
    frames: list[np.ndarray],
    text: str,
    max_side: int = 512,
) -> list[dict]:
    """Build Anthropic-format content block list. Max 20 images per request."""
    if len(frames) > 20:
        raise ValueError(f"Anthropic supports max 20 images per request, got {len(frames)}")
    content = []
    for frame in frames:
        frame = resize_frame(frame, max_side)
        b64 = frame_to_base64(frame)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    content.append({"type": "text", "text": text})
    return content


def chunk_frames(frames: list, max_per_chunk: int) -> list[list]:
    """Split frames into chunks of at most max_per_chunk."""
    return [frames[i : i + max_per_chunk] for i in range(0, len(frames), max_per_chunk)]
