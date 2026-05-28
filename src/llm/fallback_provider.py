from __future__ import annotations

import concurrent.futures

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment
from src.llm.ollama_provider import OllamaEvaluator


class FallbackEvaluator(BaseLLMEvaluator):
    """Tries an online provider first; on failure, falls back to Ollama local.

    This is the recommended evaluator for use in Cuba where internet may be
    intermittent. The primary provider (Mistral or DeepSeek) is called with a
    configurable timeout. If it raises any exception or times out, the local
    Ollama evaluator is used instead and CohesionScore.used_fallback is set
    to True so the runner can log the fallback rate.
    """

    def __init__(
        self,
        primary: BaseLLMEvaluator,
        fallback: OllamaEvaluator,
        timeout: int = 15,
    ) -> None:
        """Initialize the fallback evaluator.

        Args:
            primary: Online LLM provider to try first.
            fallback: Local Ollama provider to use if primary fails.
            timeout: Seconds to wait for the primary before falling back.
        """
        self._primary = primary
        self._fallback = fallback
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return f"fallback({self._primary.provider_name}→{self._fallback.provider_name})"

    @property
    def model_name(self) -> str:
        return self._primary.model_name

    def score_segment(self, segment: Segment) -> CohesionScore:
        """Score one segment with automatic online→local fallback."""
        return self.score_segmentation([segment])[0]

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        """Score all segments; fall back per-segment on failure."""
        return [self._score_one(segment) for segment in segments]

    def _score_one(self, segment: Segment) -> CohesionScore:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._primary.score_segment, segment)
                return future.result(timeout=self._timeout)
        except Exception:
            pass

        try:
            score = self._fallback.score_segment(segment)
            return score.model_copy(update={"used_fallback": True})
        except Exception:
            # Both providers are unavailable (e.g. no internet AND no local
            # Ollama). Return a neutral score so a long experiment completes
            # instead of crashing; the rationale flags it for later filtering.
            return CohesionScore(
                segment_id=segment.segment_id,
                score=3,
                rationale="Both primary and fallback providers failed",
                provider=self.provider_name,
                model=self.model_name,
                used_fallback=True,
            )
