from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator
from typing import Any


class Document(BaseModel):
    """A document represented as an ordered list of sentences."""

    doc_id: str
    sentences: list[str]

    @field_validator("sentences")
    @classmethod
    def sentences_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("Document must have at least one sentence")
        return v

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class Segment(BaseModel):
    """A contiguous slice of a document's sentences."""

    segment_id: str
    sentences: list[str]

    @field_validator("sentences")
    @classmethod
    def sentences_not_empty(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("Segment must have at least one sentence")
        return v

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class SegmentationResult(BaseModel):
    """Output of a segmentation algorithm: boundary positions + metadata."""

    doc_id: str
    # 0-based sentence indices marking the START of each segment.
    # boundaries[0] must always be 0 (start of document).
    boundaries: list[int]
    algorithm_name: str
    runtime_seconds: float

    @model_validator(mode="after")
    def validate_boundaries(self) -> SegmentationResult:
        if not self.boundaries:
            raise ValueError("boundaries must not be empty")
        if self.boundaries[0] != 0:
            raise ValueError("First boundary must be 0 (start of document)")
        if self.boundaries != sorted(self.boundaries):
            raise ValueError("boundaries must be in ascending order")
        return self

    def validate_against_document(self, document: Document) -> None:
        """Raise ValueError if any boundary is out of range for the given document."""
        for b in self.boundaries:
            if not (0 <= b < document.n_sentences):
                raise ValueError(
                    f"Boundary {b} is out of range for document with "
                    f"{document.n_sentences} sentences"
                )

    @property
    def n_segments(self) -> int:
        return len(self.boundaries)

    def to_segments(self, document: Document) -> list[Segment]:
        """Convert boundary positions into Segment objects."""
        self.validate_against_document(document)
        ends = self.boundaries[1:] + [document.n_sentences]
        return [
            Segment(
                segment_id=f"{self.doc_id}_seg{i}",
                sentences=document.sentences[start:end],
            )
            for i, (start, end) in enumerate(zip(self.boundaries, ends))
        ]


class CohesionScore(BaseModel):
    """LLM-generated semantic cohesion score for a single segment."""

    segment_id: str
    score: int          # 1–5 scale
    rationale: str
    provider: str       # e.g. "mistral", "deepseek", "ollama"
    model: str          # e.g. "mistral-large-latest"
    used_fallback: bool = False

    @field_validator("score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError(f"score must be between 1 and 5, got {v}")
        return v


class RangeConfig(BaseModel):
    """Integer range used in dataset configs."""

    min: int
    max: int

    @model_validator(mode="after")
    def min_lte_max(self) -> RangeConfig:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class LLMConfig(BaseModel):
    """LLM evaluator configuration block inside an experiment config."""

    provider: str           # "mistral" | "deepseek" | "ollama" | "none"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512


class AlgorithmConfig(BaseModel):
    """Single algorithm entry inside an experiment config."""

    name: str
    params: dict[str, Any] = {}


class DatasetRef(BaseModel):
    """Dataset reference inside an experiment config."""

    path: str
    split: str = "all"


class OutputConfig(BaseModel):
    """Output settings for experiment results."""

    path: str
    save_raw: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation settings: metrics and random seed."""

    metrics: list[str]
    random_seed: int


class ExperimentConfig(BaseModel):
    """Full experiment configuration loaded from a YAML file."""

    experiment_id: str
    description: str = ""
    dataset: DatasetRef
    algorithms: list[AlgorithmConfig]
    llm_evaluator: LLMConfig
    evaluation: EvaluationConfig
    output: OutputConfig
