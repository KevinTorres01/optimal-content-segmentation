from __future__ import annotations

import os

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import LLMConfig
from src.llm.deepseek_provider import DeepSeekEvaluator
from src.llm.fallback_provider import FallbackEvaluator
from src.llm.mistral_provider import MistralEvaluator
from src.llm.ollama_provider import OllamaEvaluator

_ONLINE_PROVIDERS: dict[str, type[BaseLLMEvaluator]] = {
    "mistral": MistralEvaluator,
    "deepseek": DeepSeekEvaluator,
}

_FALLBACK_DEFAULT = True
_TIMEOUT_DEFAULT = 15


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def get_llm_provider(config: LLMConfig) -> BaseLLMEvaluator | None:
    """Build the LLM evaluator described by an experiment's llm_evaluator block.

    Online providers (mistral, deepseek) are wrapped in a FallbackEvaluator
    that falls back to local Ollama on runtime failure, unless
    LLM_FALLBACK_ENABLED is false. This is the resilience strategy for
    intermittent connectivity. A missing API key is a setup error and is
    surfaced by the provider constructor rather than silently falling back.

    Args:
        config: The validated llm_evaluator config block.

    Returns:
        A ready evaluator, or None when provider is "none".

    Raises:
        ValueError: If the provider name is not recognised.
    """
    provider = config.provider

    if provider == "none":
        return None

    if provider == "ollama":
        return OllamaEvaluator(model=config.model)

    if provider in _ONLINE_PROVIDERS:
        kwargs: dict = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.model:
            kwargs["model"] = config.model
        primary = _ONLINE_PROVIDERS[provider](**kwargs)

        if not _env_bool("LLM_FALLBACK_ENABLED", _FALLBACK_DEFAULT):
            return primary

        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", str(_TIMEOUT_DEFAULT)))
        return FallbackEvaluator(
            primary=primary,
            fallback=OllamaEvaluator(),
            timeout=timeout,
        )

    raise ValueError(
        f"Unknown LLM provider: {provider}. "
        f"Available: mistral, deepseek, ollama, none"
    )
