"""Claude VLM client (max 20 images per request)."""

import os
from typing import Any

import numpy as np

from rlm.clients.base_lm import BaseLM
from rlm.core.types import ModelUsageSummary, UsageSummary
from rvlm.clients.vision_utils import frames_to_anthropic_content


class AnthropicVLMClient(BaseLM):
    """VLM client for Anthropic Claude models."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        **kwargs,
    ):
        super().__init__(model_name=model_name, timeout=timeout, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.api_key)
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

        resp = client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        self._last_input_tokens = resp.usage.input_tokens
        self._last_output_tokens = resp.usage.output_tokens
        self._total_input_tokens += self._last_input_tokens
        self._total_output_tokens += self._last_output_tokens
        return resp.content[0].text.strip()

    async def acompletion(self, prompt: str | dict[str, Any]) -> str:
        raise NotImplementedError("Async completion not implemented for AnthropicVLMClient")

    def vlm_completion(
        self,
        frames: list[np.ndarray],
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Multimodal completion. Raises ValueError if len(frames) > 20."""
        client = self._get_client()
        content = frames_to_anthropic_content(frames, prompt)
        messages = [{"role": "user", "content": content}]

        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        resp = client.messages.create(**kwargs)
        self._last_input_tokens = resp.usage.input_tokens
        self._last_output_tokens = resp.usage.output_tokens
        self._total_input_tokens += self._last_input_tokens
        self._total_output_tokens += self._last_output_tokens
        return resp.content[0].text.strip()

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
