import pytest
from pydantic import ValidationError

from src.core.interfaces import BaseLLMEvaluator, BaseSegmenter
from src.core.models import (
    CohesionScore,
    Document,
    Segment,
    SegmentationResult,
)


# ── Document ──────────────────────────────────────────────────────────────────

def test_document_empty_sentences_raises() -> None:
    with pytest.raises(ValidationError):
        Document(doc_id="d1", sentences=[])


def test_document_n_sentences() -> None:
    doc = Document(doc_id="d1", sentences=["A.", "B.", "C."])
    assert doc.n_sentences == 3


def test_document_text_joins_sentences() -> None:
    doc = Document(doc_id="d1", sentences=["Hello.", "World."])
    assert doc.text == "Hello. World."


# ── Segment ───────────────────────────────────────────────────────────────────

def test_segment_empty_sentences_raises() -> None:
    with pytest.raises(ValidationError):
        Segment(segment_id="s1", sentences=[])


def test_segment_text_property() -> None:
    seg = Segment(segment_id="s1", sentences=["Hola.", "Mundo."])
    assert seg.text == "Hola. Mundo."


# ── SegmentationResult ────────────────────────────────────────────────────────

def test_segmentation_result_first_boundary_must_be_zero() -> None:
    with pytest.raises(ValidationError):
        SegmentationResult(
            doc_id="d1",
            boundaries=[3, 6],
            algorithm_name="test",
            runtime_seconds=0.1,
        )


def test_segmentation_result_boundaries_must_be_sorted() -> None:
    with pytest.raises(ValidationError):
        SegmentationResult(
            doc_id="d1",
            boundaries=[0, 6, 3],
            algorithm_name="test",
            runtime_seconds=0.1,
        )


def test_segmentation_result_empty_boundaries_raises() -> None:
    with pytest.raises(ValidationError):
        SegmentationResult(
            doc_id="d1",
            boundaries=[],
            algorithm_name="test",
            runtime_seconds=0.1,
        )


def test_segmentation_result_n_segments() -> None:
    result = SegmentationResult(
        doc_id="d1",
        boundaries=[0, 3, 6],
        algorithm_name="test",
        runtime_seconds=0.1,
    )
    assert result.n_segments == 3


def test_segmentation_result_to_segments() -> None:
    doc = Document(doc_id="d1", sentences=["A.", "B.", "C.", "D.", "E.", "F."])
    result = SegmentationResult(
        doc_id="d1",
        boundaries=[0, 3],
        algorithm_name="test",
        runtime_seconds=0.1,
    )
    segments = result.to_segments(doc)
    assert len(segments) == 2
    assert segments[0].sentences == ["A.", "B.", "C."]
    assert segments[1].sentences == ["D.", "E.", "F."]


def test_segmentation_result_boundary_out_of_range_raises() -> None:
    doc = Document(doc_id="d1", sentences=["A.", "B.", "C."])
    result = SegmentationResult(
        doc_id="d1",
        boundaries=[0, 10],
        algorithm_name="test",
        runtime_seconds=0.1,
    )
    with pytest.raises(ValueError, match="out of range"):
        result.validate_against_document(doc)


# ── CohesionScore ─────────────────────────────────────────────────────────────

def test_cohesion_score_invalid_score_raises() -> None:
    with pytest.raises(ValidationError):
        CohesionScore(
            segment_id="s1",
            score=6,
            rationale="too high",
            provider="mock",
            model="mock",
        )


def test_cohesion_score_zero_raises() -> None:
    with pytest.raises(ValidationError):
        CohesionScore(
            segment_id="s1",
            score=0,
            rationale="too low",
            provider="mock",
            model="mock",
        )


def test_cohesion_score_used_fallback_defaults_to_false() -> None:
    score = CohesionScore(
        segment_id="s1",
        score=3,
        rationale="ok",
        provider="mistral",
        model="mistral-large-latest",
    )
    assert score.used_fallback is False


# ── ABCs cannot be instantiated ───────────────────────────────────────────────

def test_base_segmenter_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseSegmenter()  # type: ignore[abstract]


def test_base_llm_evaluator_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseLLMEvaluator()  # type: ignore[abstract]
