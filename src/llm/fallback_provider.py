from __future__ import annotations

import concurrent.futures

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment


class FallbackEvaluator(BaseLLMEvaluator):
    """Tries a primary provider first; on failure, falls back to a second one.

    Used as the resilience layer for intermittent connectivity: the primary
    (Groq) is called with a configurable timeout, and if it raises any
    exception or times out, the fallback (Mistral) is used instead and
    CohesionScore.used_fallback is set to True so the runner can log the rate.
    """

    def __init__(
        self,
        primary: BaseLLMEvaluator,
        fallback: BaseLLMEvaluator,
        timeout: int = 15,
    ) -> None:
        """Initialize the fallback evaluator.

        Args:
            primary: LLM provider to try first.
            fallback: LLM provider to use if the primary fails.
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
            # Both providers are unavailable (e.g. no internet at all). Return a
            # neutral score so a long experiment completes instead of crashing;
            # the rationale flags it for later filtering.
            return CohesionScore(
                segment_id=segment.segment_id,
                score=3,
                rationale="Both primary and fallback providers failed",
                provider=self.provider_name,
                model=self.model_name,
                used_fallback=True,
            )
