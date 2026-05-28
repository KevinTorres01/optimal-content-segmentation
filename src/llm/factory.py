from __future__ import annotations

import os

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import LLMConfig
from src.llm.fallback_provider import FallbackEvaluator
from src.llm.groq_provider import GroqEvaluator
from src.llm.mistral_provider import MistralEvaluator

_PROVIDERS: dict[str, type[BaseLLMEvaluator]] = {
    "groq": GroqEvaluator,
    "mistral": MistralEvaluator,
}

_FALLBACK_DEFAULT = True
_TIMEOUT_DEFAULT = 30


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _build(provider: str, config: LLMConfig) -> BaseLLMEvaluator:
    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.model:
        kwargs["model"] = config.model
    return _PROVIDERS[provider](**kwargs)


def get_llm_provider(config: LLMConfig) -> BaseLLMEvaluator | None:
    """Build the LLM evaluator described by an experiment's llm_evaluator block.

    The resilience strategy is a Groq → Mistral chain: Groq (free tier) is the
    primary, and when LLM_FALLBACK_ENABLED is set it is wrapped in a
    FallbackEvaluator that falls back to Mistral on runtime failure. Mistral is
    the terminal provider (no further fallback). The fallback is only attached
    when a MISTRAL_API_KEY is configured; otherwise Groq runs alone. A missing
    key for the requested provider is a setup error surfaced by its constructor.

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

    if provider == "mistral":
        return _build("mistral", config)

    if provider == "groq":
        primary = _build("groq", config)
        if not _env_bool("LLM_FALLBACK_ENABLED", _FALLBACK_DEFAULT):
            return primary
        try:
            fallback: BaseLLMEvaluator = MistralEvaluator()
        except ValueError:
            # No Mistral key configured — run Groq without a fallback.
            return primary
        timeout = int(os.environ.get("LLM_TIMEOUT_SECONDS", str(_TIMEOUT_DEFAULT)))
        return FallbackEvaluator(primary=primary, fallback=fallback, timeout=timeout)

    raise ValueError(
        f"Unknown LLM provider: {provider}. Available: groq, mistral, none"
    )
