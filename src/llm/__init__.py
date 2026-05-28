from src.core.interfaces import BaseLLMEvaluator
from src.llm.mistral_provider import MistralEvaluator
from src.llm.deepseek_provider import DeepSeekEvaluator
from src.llm.ollama_provider import OllamaEvaluator
from src.llm.fallback_provider import FallbackEvaluator
from src.llm.factory import get_llm_provider

PROVIDER_REGISTRY: dict[str, type[BaseLLMEvaluator] | None] = {
    "mistral": MistralEvaluator,  # online primario (accesible desde Cuba)
    "deepseek": DeepSeekEvaluator,  # online secundario (accesible desde Cuba)
    "ollama": OllamaEvaluator,  # local fallback (sin internet)
    "none": None,
}

__all__ = [
    "BaseLLMEvaluator",
    "MistralEvaluator",
    "DeepSeekEvaluator",
    "OllamaEvaluator",
    "FallbackEvaluator",
    "get_llm_provider",
    "PROVIDER_REGISTRY",
]
