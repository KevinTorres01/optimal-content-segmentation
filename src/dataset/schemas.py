from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class RangeConfig(BaseModel):
    min: int
    max: int


class DatasetConfig(BaseModel):
    """Configuration for synthetic dataset generation."""

    dataset_name: str
    n_documents: int
    segments_per_doc: RangeConfig
    sentences_per_segment: RangeConfig
    topic_source: Literal["synthetic_templates"] = "synthetic_templates"
    overlap_level: Literal["low", "medium", "high"] = "low"
    random_seed: int      # mandatory — Pydantic rejects configs missing this field
    language: Literal["es", "en"] = "es"
