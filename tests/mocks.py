from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment


class MockLLMEvaluator(BaseLLMEvaluator):
    """Deterministic mock LLM evaluator for testing (no API calls).

    Always returns score=3 with a fixed rationale. Use this in all unit
    and integration tests to avoid real API calls and ensure determinism.
    """

    def score_segment(self, segment: Segment) -> CohesionScore:
        return CohesionScore(
            segment_id=segment.segment_id,
            score=3,
            rationale="Mock evaluation",
            provider="mock",
            model="mock",
        )

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        return [self.score_segment(s) for s in segments]

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"
