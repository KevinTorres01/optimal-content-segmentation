from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.models import CohesionScore, Document, Segment, SegmentationResult


class BaseSegmenter(ABC):
    """Abstract contract for all text segmentation algorithms."""

    @abstractmethod
    def segment(
        self, document: Document, max_segments: int | None = None
    ) -> SegmentationResult:
        """Segment a document into coherent parts.

        Args:
            document: The document to segment.
            max_segments: Optional upper bound on the number of segments.

        Returns:
            SegmentationResult with sorted boundary positions starting at 0.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used in config files and ALGORITHM_REGISTRY."""
        ...


class BaseLLMEvaluator(ABC):
    """Abstract contract for LLM-based semantic cohesion evaluators."""

    @abstractmethod
    def score_segment(self, segment: Segment) -> CohesionScore:
        """Score the semantic cohesion of a single segment on a 1–5 scale.

        Args:
            segment: A text segment to evaluate.

        Returns:
            CohesionScore with numeric score, rationale, and model metadata.
        """
        ...

    @abstractmethod
    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        """Score all segments in a segmentation result.

        Implementations should batch API calls where the provider supports it.

        Args:
            segments: List of segments to score.

        Returns:
            List of CohesionScore objects in the same order as the input.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier string used in config files (e.g. 'groq', 'mistral')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Specific model identifier (e.g. 'mistral-large-latest')."""
        ...
