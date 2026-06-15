import pytest

from src.algorithms.auto_k import AutoKResult, find_optimal_k
from src.core.models import Document
from src.experiments.runner import AUTO_K, _normalize_max_segments


@pytest.fixture
def three_topic_doc() -> Document:
    """9-sentence document with 3 clearly distinct topics (3 sentences each).

    Vocabulary repeats inside each block and barely overlaps across blocks, so
    the cohesion curve has a sharp elbow at k = 3.
    """
    return Document(
        doc_id="three_topics",
        sentences=[
            "El equipo de fútbol marcó un gol en el partido.",
            "El partido de fútbol terminó con tres goles del equipo local.",
            "El equipo celebró el gol decisivo del partido de fútbol.",
            "El software del programa fue actualizado en la computadora.",
            "El código del programa optimiza el software de la computadora.",
            "La computadora ejecuta el software gracias al nuevo código del programa.",
            "La célula contiene ADN que determina el gen biológico.",
            "El gen de la célula fue analizado mediante técnicas de biología del ADN.",
            "La biología estudia el ADN del gen dentro de la célula.",
        ],
    )


# ── find_optimal_k: contract ───────────────────────────────────────────────────


def test_returns_auto_k_result(three_topic_doc: Document) -> None:
    result = find_optimal_k(three_topic_doc)
    assert isinstance(result, AutoKResult)
    assert isinstance(result.k, int)
    assert isinstance(result.objectives, dict)
    assert result.rationale


def test_boundaries_populated(three_topic_doc: Document) -> None:
    """Boundaries must come back from the single DP pass — no second run needed."""
    result = find_optimal_k(three_topic_doc)
    assert isinstance(result.boundaries, list)
    assert len(result.boundaries) == result.k
    assert result.boundaries[0] == 0
    assert result.boundaries == sorted(result.boundaries)


def test_boundaries_match_dp_segmenter(three_topic_doc: Document) -> None:
    """Boundaries returned by auto-k must equal those a fresh DPSegmenter produces."""
    from src.algorithms.dynamic_programming import DPSegmenter

    auto = find_optimal_k(three_topic_doc)
    dp_result = DPSegmenter().segment(three_topic_doc, max_segments=auto.k)
    assert auto.boundaries == dp_result.boundaries


def test_k_within_explored_range(three_topic_doc: Document) -> None:
    result = find_optimal_k(three_topic_doc)
    explored = sorted(result.objectives)
    assert explored[0] <= result.k <= explored[-1]


def test_respects_k_max(three_topic_doc: Document) -> None:
    result = find_optimal_k(three_topic_doc, k_max=3)
    assert result.k <= 3
    assert max(result.objectives) <= 3


def test_picks_three_topics(three_topic_doc: Document) -> None:
    """The elbow should land on the true number of topics (3)."""
    result = find_optimal_k(three_topic_doc)
    assert result.k == 3


# ── find_optimal_k: edge cases ─────────────────────────────────────────────────


def test_single_sentence_document() -> None:
    doc = Document(doc_id="one", sentences=["Una sola oración."])
    result = find_optimal_k(doc)
    assert result.k == 1


def test_two_sentence_document() -> None:
    doc = Document(doc_id="two", sentences=["Primera.", "Segunda."])
    result = find_optimal_k(doc)
    assert result.k == 2


def test_k_min_clamped_to_k_max() -> None:
    """When k_min exceeds the feasible range it is clamped, not crashed."""
    doc = Document(doc_id="three", sentences=["Uno.", "Dos.", "Tres."])
    result = find_optimal_k(doc, k_min=10)
    assert 2 <= result.k <= doc.n_sentences


# ── runner: _normalize_max_segments ────────────────────────────────────────────


def test_normalize_none_is_none() -> None:
    assert _normalize_max_segments(None) is None


def test_normalize_int_passthrough() -> None:
    assert _normalize_max_segments(5) == 5


def test_normalize_auto_sentinel() -> None:
    assert _normalize_max_segments("auto") == AUTO_K
    assert _normalize_max_segments("AUTO") == AUTO_K
    assert _normalize_max_segments(" Auto ") == AUTO_K


def test_normalize_rejects_other_strings() -> None:
    with pytest.raises(ValueError):
        _normalize_max_segments("five")
