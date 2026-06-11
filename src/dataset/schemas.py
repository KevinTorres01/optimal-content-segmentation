from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RangeConfig(BaseModel):
    """Inclusive integer range used by dataset configs (e.g., segments per doc)."""

    min: int
    max: int

    @model_validator(mode="after")
    def min_lte_max(self) -> RangeConfig:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    dataset_name: str
    n_documents: int
    segments_per_doc: RangeConfig
    sentences_per_segment: RangeConfig
    topic_source: Literal["synthetic_templates"] = "synthetic_templates"
    overlap_level: Literal["low", "medium", "high"] = "low"
    random_seed: int  # mandatory — Pydantic rejects configs missing this field
    language: Literal["es", "en"] = "es"


class WikipediaDatasetConfig(BaseModel):
    """Configuration for building a dataset from real Wikipedia articles.

    Sections in the article (level-2 ``==`` headings) provide the ground-truth
    boundaries. Sub-sections are folded into their parent so granularity matches
    the synthetic datasets.

    Real articles are much larger than the synthetic ``small`` dataset, so the
    loader truncates each article to ``max_segments_per_doc`` sections and
    ``max_sentences_per_segment`` sentences per section. Min values are applied
    *after* truncation and a document is dropped if it doesn't meet them.
    """

    dataset_name: str
    titles: list[str] = Field(..., min_length=1)
    language: Literal["es", "en"] = "es"

    min_sentences_per_doc: int = 10
    max_sentences_per_doc: int = 80
    min_segments_per_doc: int = 3
    max_segments_per_doc: int = 5
    max_sentences_per_segment: int = 8
    min_sentence_chars: int = 20

    request_delay_seconds: float = 0.5
