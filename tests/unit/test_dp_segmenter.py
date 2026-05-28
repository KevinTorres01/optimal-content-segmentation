import pytest

from src.algorithms.dynamic_programming import DPSegmenter
from src.core.models import Document, SegmentationResult


@pytest.fixture
def sports_tech_science_doc() -> Document:
    """9-sentence document with 3 clearly distinct topics.
    Sentences within each block intentionally repeat key vocabulary so that
    TF-IDF cosine similarity is high inside a block and low across blocks.
    """
    return Document(
        doc_id="test_doc",
        sentences=[
            # Sports block — repeated: fútbol, gol, partido, equipo
            "El equipo de fútbol marcó un gol en el partido.",
            "El partido de fútbol terminó con tres goles del equipo local.",
            "El equipo celebró el gol decisivo del partido de fútbol.",
            # Tech block — repeated: software, código, programa, computadora
            "El software del programa fue actualizado en la computadora.",
            "El código del programa optimiza el software de la computadora.",
            "La computadora ejecuta el software gracias al nuevo código del programa.",
            # Science block — repeated: célula, ADN, gen, biología
            "La célula contiene ADN que determina el gen biológico.",
            "El gen de la célula fue analizado mediante técnicas de biología del ADN.",
            "La biología estudia el ADN del gen dentro de la célula.",
        ],
    )


@pytest.fixture
def single_sentence_doc() -> Document:
    return Document(doc_id="single", sentences=["Una sola oración."])


@pytest.fixture
def segmenter() -> DPSegmenter:
    return DPSegmenter()


# ── Basic contract ────────────────────────────────────────────────────────────


def test_returns_segmentation_result(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc)
    assert isinstance(result, SegmentationResult)
    assert result.doc_id == sports_tech_science_doc.doc_id
    assert result.algorithm_name == "dynamic_programming"


def test_name_property(segmenter: DPSegmenter) -> None:
    assert segmenter.name == "dynamic_programming"


def test_first_boundary_is_zero(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc)
    assert result.boundaries[0] == 0


def test_boundaries_within_document(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc)
    n = sports_tech_science_doc.n_sentences
    assert all(0 <= b < n for b in result.boundaries)


def test_boundaries_are_sorted(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc)
    assert result.boundaries == sorted(result.boundaries)


def test_runtime_is_recorded(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc)
    assert result.runtime_seconds > 0


# ── max_segments ──────────────────────────────────────────────────────────────


def test_respects_max_segments(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc, max_segments=2)
    assert result.n_segments <= 2


def test_max_segments_one_returns_single_segment(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    result = segmenter.segment(sports_tech_science_doc, max_segments=1)
    assert result.boundaries == [0]
    assert result.n_segments == 1


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_single_sentence_document(
    segmenter: DPSegmenter, single_sentence_doc: Document
) -> None:
    result = segmenter.segment(single_sentence_doc)
    assert result.boundaries == [0]
    assert result.n_segments == 1


def test_detects_obvious_segments(
    segmenter: DPSegmenter, sports_tech_science_doc: Document
) -> None:
    """With 3 clearly distinct topic blocks of 3 sentences each,
    the algorithm should place boundaries near [0, 3, 6]."""
    result = segmenter.segment(sports_tech_science_doc, max_segments=3)
    assert result.n_segments == 3
    # Each boundary should be within 1 sentence of the true boundary
    expected = [0, 3, 6]
    for got, exp in zip(result.boundaries, expected):
        assert abs(got - exp) <= 1, (
            f"Expected boundary near {exp}, got {got}. "
            f"Full boundaries: {result.boundaries}"
        )
