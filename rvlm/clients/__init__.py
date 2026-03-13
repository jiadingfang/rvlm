from rvlm.clients.openai_vlm import OpenAIVLMClient
from rvlm.clients.gemini_vlm import GeminiVLMClient
from rvlm.clients.anthropic_vlm import AnthropicVLMClient

_VLM_REGISTRY = {
    "gpt-4o": (OpenAIVLMClient, {"model_name": "gpt-4o"}),
    "gpt-4.1": (OpenAIVLMClient, {"model_name": "gpt-4.1"}),
    "gpt-4.1-mini": (OpenAIVLMClient, {"model_name": "gpt-4.1-mini"}),
    "gemini-2.0-flash": (GeminiVLMClient, {"model_name": "gemini-2.0-flash"}),
    "gemini-2.5-flash": (GeminiVLMClient, {"model_name": "gemini-2.5-flash"}),
    "gemini-2.5-flash-preview-04-17": (GeminiVLMClient, {"model_name": "gemini-2.5-flash-preview-04-17"}),
    "gemini-2.5-pro": (GeminiVLMClient, {"model_name": "gemini-2.5-pro"}),
    "gemini-3-flash-preview": (GeminiVLMClient, {"model_name": "gemini-3-flash-preview"}),
    "claude-sonnet-4-5": (AnthropicVLMClient, {"model_name": "claude-sonnet-4-5"}),
    "claude-sonnet-4-6": (AnthropicVLMClient, {"model_name": "claude-sonnet-4-6"}),
}


def get_vlm_client(model_name: str, **kwargs):
    """Get a VLM client by model name."""
    if model_name not in _VLM_REGISTRY:
        raise ValueError(f"Unknown VLM model: {model_name}. Available: {list(_VLM_REGISTRY)}")
    cls, defaults = _VLM_REGISTRY[model_name]
    merged = {**defaults, **kwargs}
    return cls(**merged)
