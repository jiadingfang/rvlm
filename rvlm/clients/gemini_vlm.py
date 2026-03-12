"""Gemini VLM client with native video upload and frame image paths."""

import base64
import os
import time
from typing import Any

import numpy as np

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rvlm.clients.vision_utils import frame_to_base64, resize_frame


class GeminiVLMClient(BaseLM):
    """VLM client for Gemini models (2.0 Flash, 2.5 Pro)."""

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, timeout=timeout, **kwargs)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.max_tokens = max_tokens
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def completion(self, prompt: str | dict[str, Any]) -> str:
        """Text-only completion."""
        from google.genai import types

        client = self._get_client()
        resp = client.models.generate_content(
            model=self.model_name,
            contents=prompt if isinstance(prompt, str) else str(prompt),
            config=types.GenerateContentConfig(max_output_tokens=self.max_tokens),
        )
        self._update_usage(resp)
        return resp.text.strip()

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError("Async completion not implemented for GeminiVLMClient")

    def vlm_completion(
        self,
        frames: list[np.ndarray] | None = None,
        video_path: str | None = None,
        prompt: str = "",
    ) -> str:
        """
        If video_path provided: upload file and call with native video.
        If frames provided: encode as inline image parts.
        """
        from google.genai import types

        if video_path is not None and frames is not None:
            raise ValueError("Provide exactly one of frames or video_path, not both")
        if video_path is None and frames is None:
            raise ValueError("Provide exactly one of frames or video_path")

        client = self._get_client()

        if video_path is not None:
            file_ref = self._upload_video(video_path)
            parts = [file_ref, types.Part.from_text(prompt)]
        else:
            parts = []
            for frame in frames:
                frame = resize_frame(frame)
                b64 = frame_to_base64(frame)
                img_bytes = base64.b64decode(b64)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
            parts.append(types.Part.from_text(prompt))

        resp = client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(max_output_tokens=self.max_tokens),
        )
        self._update_usage(resp)
        return resp.text.strip()

    def _upload_video(self, video_path: str):
        """Upload video via Files API; poll until state == ACTIVE."""
        client = self._get_client()
        file_ref = client.files.upload(file=video_path)
        while file_ref.state.name == "PROCESSING":
            time.sleep(2)
            file_ref = client.files.get(name=file_ref.name)
        if file_ref.state.name != "ACTIVE":
            raise RuntimeError(f"Video upload failed: state={file_ref.state.name}")
        return file_ref

    def _update_usage(self, resp):
        if resp.usage_metadata:
            self._last_input_tokens = resp.usage_metadata.prompt_token_count or 0
            self._last_output_tokens = resp.usage_metadata.candidates_token_count or 0
        else:
            self._last_input_tokens = 0
            self._last_output_tokens = 0
        self._total_input_tokens += self._last_input_tokens
        self._total_output_tokens += self._last_output_tokens

    def get_usage_summary(self) -> UsageSummary:
        model_usage = self.get_last_usage()
        return UsageSummary(model_usage_summaries={self.model_name: model_usage})

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            model_name=self.model_name,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
            total_input_tokens=self._total_input_tokens,
            total_output_tokens=self._total_output_tokens,
            num_calls=1,
        )
