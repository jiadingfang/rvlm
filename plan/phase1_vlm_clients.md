# Phase 1: VLM Client Extension

## Goal

Extend the existing `BaseLM` client architecture to handle multimodal (image + text) inputs,
adding three closed-model VLM backends: GPT-4o, Gemini, and Claude.

The existing `completion(prompt: str | list[dict])` signature already accepts message lists
with image content blocks — the work here is thin wrappers that construct those blocks
correctly for each API and handle video-scale inputs (chunking, retries, rate limits).

## Architecture Decision

Do NOT create a separate `BaseVLM` class. Instead:
- Keep `BaseLM` as-is (image-capable APIs accept the same message list format)
- Add a shared `rvlm/clients/vision_utils.py` with frame-encoding helpers
- Each VLM client lives in `rvlm/clients/<provider>_vlm.py` and inherits `BaseLM`
- Register all clients in `rvlm/clients/__init__.py` under new string keys

This preserves full compatibility with the existing `RLM` class and `LMHandler`.

## File Structure

```
rvlm/
  clients/
    __init__.py          # extend existing registry with new keys
    vision_utils.py      # shared frame encoding helpers
    openai_vlm.py        # GPT-4o / GPT-4.1
    gemini_vlm.py        # Gemini 2.0 / 2.5 Flash
    anthropic_vlm.py     # Claude 3.7 Sonnet
```

## Tasks

### 1.1 — `rvlm/clients/vision_utils.py`

Shared utilities used by all three clients:

```python
import base64
import numpy as np
from PIL import Image

def frame_to_base64(frame: np.ndarray, format: str = "JPEG", quality: int = 85) -> str:
    """Convert numpy HWC uint8 frame to base64-encoded JPEG/PNG string."""

def frames_to_openai_content(
    frames: list[np.ndarray],
    text: str,
    detail: str = "low",
) -> list[dict]:
    """
    Build OpenAI-format content block list:
      [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": detail}}, ...]
      + [{"type": "text", "text": text}]
    Interleave frames and text at the end.
    """

def frames_to_anthropic_content(
    frames: list[np.ndarray],
    text: str,
) -> list[dict]:
    """
    Build Anthropic-format content block list:
      [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": ...}}, ...]
      + [{"type": "text", "text": text}]
    Max 20 images per request — raise ValueError if exceeded.
    """

def chunk_frames(frames: list, max_per_chunk: int) -> list[list]:
    """Split frames into chunks of at most max_per_chunk."""
```

### 1.2 — `rvlm/clients/openai_vlm.py` — GPT-4o / GPT-4.1

Wraps the existing `OpenAIClient` (or re-implements against the same `openai` SDK).

Key behaviors:
- `completion(messages)` passes through unchanged (text-only path)
- `vlm_completion(frames, prompt, detail="low")` builds image content blocks and calls the API
- Chunks frames into batches of ≤50 if needed (well under the 2000-image limit; keep
  small for cost control)
- `detail="low"` for coarse passes (85 tokens/image), `detail="high"` for close-up
  verification (~1700 tokens/image)
- Token tracking: image tokens = 85 (low) or ~1700 (high) per frame; add to usage

```python
class OpenAIVLMClient(BaseLM):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        detail: str = "low",   # default detail level for vlm_completion
        **kwargs,
    ): ...

    def completion(self, prompt: str | list[dict]) -> str:
        """Text-only path, identical to OpenAIClient."""

    def vlm_completion(
        self,
        frames: list[np.ndarray],
        prompt: str,
        detail: str | None = None,  # override instance default
    ) -> str:
        """Build multimodal message and call the chat completions API."""
```

### 1.3 — `rvlm/clients/gemini_vlm.py` — Gemini 2.0 / 2.5 Flash

Two input paths (choose at runtime based on video length):

**Path A — Native video upload** (videos ≤ ~1GB, duration ≤ 20 min):
- Upload the video file with `google.generativeai.upload_file(path, mime_type="video/mp4")`
- Pass the file object directly in the content; Gemini handles frame extraction internally
- Best for long videos; no manual frame sampling needed

**Path B — Frame images** (fallback, or when file upload is unavailable):
- Inline base64-encoded frames as image parts
- Same structure as OpenAI but using the Gemini `types.Part` format

```python
class GeminiVLMClient(BaseLM):
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ): ...

    def vlm_completion(
        self,
        frames: list[np.ndarray] | None = None,
        video_path: str | None = None,   # prefer this for long videos
        prompt: str = "",
    ) -> str:
        """
        If video_path provided: upload file and call with native video.
        If frames provided: encode as inline image parts.
        Exactly one of frames/video_path must be given.
        """

    def _upload_video(self, video_path: str):
        """Upload video via Files API; poll until state == ACTIVE."""
```

Note: Gemini natively samples frames from uploaded videos — no manual frame extraction
needed on the Python side. This is the preferred path for videos > 30s.

### 1.4 — `rvlm/clients/anthropic_vlm.py` — Claude 3.7 Sonnet

Anthropic has a 20-image limit per request and no native video support. Strategy:
- For videos requiring > 20 frames, the model must call `vlm_count` multiple times
  (handled at the REPL tools layer in Phase 2, not at the client layer)
- Client just handles the ≤ 20 frame case cleanly

```python
class AnthropicVLMClient(BaseLM):
    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ): ...

    def vlm_completion(
        self,
        frames: list[np.ndarray],
        prompt: str,
        system: str | None = None,
    ) -> str:
        """
        Raises ValueError if len(frames) > 20.
        Caller (REPL tools) is responsible for chunking.
        """
```

### 1.5 — Registry update in `rvlm/clients/__init__.py`

```python
from rvlm.clients.openai_vlm import OpenAIVLMClient
from rvlm.clients.gemini_vlm import GeminiVLMClient
from rvlm.clients.anthropic_vlm import AnthropicVLMClient

_VLM_REGISTRY = {
    "gpt4o": (OpenAIVLMClient, {"model_name": "gpt-4o"}),
    "gpt41": (OpenAIVLMClient, {"model_name": "gpt-4.1"}),
    "gemini-flash": (GeminiVLMClient, {"model_name": "gemini-2.0-flash"}),
    "gemini-pro": (GeminiVLMClient, {"model_name": "gemini-2.5-pro"}),
    "claude": (AnthropicVLMClient, {"model_name": "claude-sonnet-4-5"}),
}
```

Merge with the existing `_CLIENT_REGISTRY` from `rlm/clients/__init__.py` so `get_client()`
works for both text and VLM backends.

### 1.6 — Tests

`tests/clients/test_vlm_clients.py`:
- `test_frame_to_base64_roundtrip`: encode/decode a random numpy array
- `test_openai_content_block_format`: verify output structure matches OpenAI spec
- `test_anthropic_content_block_format`: verify Anthropic format
- `test_chunk_frames`: verify correct chunking at boundary cases
- `test_openai_vlm_client_mock`: mock the openai SDK, verify `vlm_completion` constructs
  the right request body and returns the parsed response string
- `test_gemini_vlm_client_mock`: mock `google.generativeai`, verify file upload path
  and frame path both work

All tests use mocked API responses — no real API calls in CI.

## Deliverables

- [ ] `rvlm/clients/vision_utils.py`
- [ ] `rvlm/clients/openai_vlm.py`
- [ ] `rvlm/clients/gemini_vlm.py`
- [ ] `rvlm/clients/anthropic_vlm.py`
- [ ] Registry updated in `rvlm/clients/__init__.py`
- [ ] Tests passing: `uv run pytest tests/clients/test_vlm_clients.py`
- [ ] Manual smoke test: call `vlm_completion` on a real video frame with each provider

## Dependencies

- `google-genai>=1.56.0` (already in `pyproject.toml`)
- `anthropic>=0.75.0` (already in `pyproject.toml`)
- `openai>=2.14.0` (already in `pyproject.toml`)
- `Pillow` (new, add to `video` optional extra)
