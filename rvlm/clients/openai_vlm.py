"""GPT-4o / GPT-4.1 VLM client."""

import os
from typing import Any

import numpy as np

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rvlm.clients.vision_utils import frames_to_openai_content


class OpenAIVLMClient(BaseLM):
    """VLM client for OpenAI models (GPT-4o, GPT-4.1)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 1024,
        detail: str = "low",
        timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, timeout=timeout, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.max_tokens = max_tokens
        self.detail = detail
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def completion(self, prompt: str | dict[str, Any]) -> str:
        """Text-only completion."""
        client = self._get_client()
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        self._last_input_tokens = resp.usage.prompt_tokens
        self._last_output_tokens = resp.usage.completion_tokens
        self._total_input_tokens += self._last_input_tokens
        self._total_output_tokens += self._last_output_tokens
        return resp.choices[0].message.content.strip()

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError("Async completion not implemented for OpenAIVLMClient")

    def vlm_completion(
        self,
        frames: list[np.ndarray],
        prompt: str,
        detail: str | None = None,
    ) -> str:
        """Build multimodal message and call the chat completions API."""
        client = self._get_client()
        detail = detail or self.detail
        content = frames_to_openai_content(frames, prompt, detail=detail)
        messages = [{"role": "user", "content": content}]

        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        self._last_input_tokens = resp.usage.prompt_tokens
        self._last_output_tokens = resp.usage.completion_tokens
        self._total_input_tokens += self._last_input_tokens
        self._total_output_tokens += self._last_output_tokens
        return resp.choices[0].message.content.strip()

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
