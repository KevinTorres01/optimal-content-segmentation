"""Tests for the Wikipedia dataset loader.

The network layer (requests.Session) is mocked so tests do not hit Wikipedia.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.dataset.schemas import WikipediaDatasetConfig
from src.dataset.wikipedia_loader import WikipediaLoader


@pytest.fixture
def base_config() -> WikipediaDatasetConfig:
    return WikipediaDatasetConfig(
        dataset_name="test_wiki",
        titles=["Dummy"],
        language="es",
        min_sentences_per_doc=3,
        max_sentences_per_doc=40,
        min_segments_per_doc=2,
        max_segments_per_doc=5,
        max_sentences_per_segment=8,
        min_sentence_chars=10,
        request_delay_seconds=0.0,
    )


def _make_extract(
    *sections: tuple[str, list[str]], intro: list[str] | None = None
) -> str:
    """Build an extract string matching MediaWiki's exsectionformat=wiki output."""
    parts: list[str] = []
    if intro:
        parts.extend(intro)
    for heading, sentences in sections:
        parts.append(f"== {heading} ==")
        parts.extend(sentences)
    return "\n".join(parts)


# ── _split_top_level_sections ────────────────────────────────────────────────


def test_split_keeps_intro_separate(base_config: WikipediaDatasetConfig) -> None:
    loader = WikipediaLoader(base_config)
    extract = _make_extract(
        ("Historia", ["Línea uno.", "Línea dos."]),
        intro=["Primera oración de intro.", "Segunda oración de intro."],
    )
    intro, sections = loader._split_top_level_sections(extract)
    assert "Primera oración de intro." in intro
    assert len(sections) == 1
    assert sections[0][0] == "Historia"


def test_subsections_are_folded_into_parent(
    base_config: WikipediaDatasetConfig,
) -> None:
    """=== headings must not create new top-level segments."""
    loader = WikipediaLoader(base_config)
    extract = "\n".join(
        [
            "== Historia ==",
            "Primera frase de historia.",
            "=== Edad media ===",
            "Frase de la edad media.",
            "== Geografía ==",
            "Frase de geografía.",
        ]
    )
    _, sections = loader._split_top_level_sections(extract)
    assert [s[0] for s in sections] == ["Historia", "Geografía"]
    # Subsection heading itself is dropped but its body stays inside Historia.
    historia_body = sections[0][1]
    assert "Frase de la edad media." in historia_body
    assert "===" not in historia_body


# ── _split_sentences ─────────────────────────────────────────────────────────


def test_short_sentences_are_filtered(base_config: WikipediaDatasetConfig) -> None:
    """min_sentence_chars must drop fragments shorter than the threshold."""
    loader = WikipediaLoader(base_config)
    text = "Hola. Esta oración es lo suficientemente larga para pasar el filtro."
    sentences = loader._split_sentences(text)
    # "Hola." has only 5 chars (< 10) and must be dropped.
    assert all(len(s) >= base_config.min_sentence_chars for s in sentences)
    assert any("suficientemente larga" in s for s in sentences)


def test_split_on_punctuation_boundaries(
    base_config: WikipediaDatasetConfig,
) -> None:
    loader = WikipediaLoader(base_config)
    text = (
        "La primera oración termina aquí. "
        "La segunda oración empieza con mayúscula. "
        "Y la tercera también, claramente."
    )
    sentences = loader._split_sentences(text)
    assert len(sentences) == 3


# ── _extract_to_segments ─────────────────────────────────────────────────────


def test_skip_sections_are_dropped(base_config: WikipediaDatasetConfig) -> None:
    """Boilerplate sections like 'Véase también' must not generate boundaries."""
    loader = WikipediaLoader(base_config)
    extract = _make_extract(
        ("Historia", ["Frase larga sobre el tema histórico número uno."]),
        ("Véase también", ["Enlace ignorable largo número uno."]),
        ("Referencias", ["Cita bibliográfica que debe ser descartada."]),
        ("Geografía", ["Frase larga sobre la geografía del lugar."]),
    )
    sentences, boundaries = loader._extract_to_segments(extract)
    # Only Historia and Geografía should have produced segments.
    assert len(boundaries) == 2
    assert all(sentence for sentence in sentences)


def test_truncation_caps_segments_and_sentences(
    base_config: WikipediaDatasetConfig,
) -> None:
    cfg = base_config.model_copy(
        update={"max_segments_per_doc": 2, "max_sentences_per_segment": 2}
    )
    loader = WikipediaLoader(cfg)
    extract = _make_extract(
        (
            "Sección A",
            [
                "Una oración suficientemente larga número uno A.",
                "Una oración suficientemente larga número dos A.",
                "Una oración suficientemente larga número tres A.",
            ],
        ),
        (
            "Sección B",
            [
                "Una oración suficientemente larga número uno B.",
                "Una oración suficientemente larga número dos B.",
                "Una oración suficientemente larga número tres B.",
            ],
        ),
        (
            "Sección C",
            ["Una oración suficientemente larga de la sección C."],
        ),
    )
    sentences, boundaries = loader._extract_to_segments(extract)
    assert len(boundaries) == 2  # truncated to max_segments_per_doc
    # Each section is truncated to 2 sentences → 4 sentences total.
    assert len(sentences) == 4


# ── _passes_filters ──────────────────────────────────────────────────────────


def test_passes_filters_rejects_too_few_segments(
    base_config: WikipediaDatasetConfig,
) -> None:
    loader = WikipediaLoader(base_config)
    sentences = ["x" * 30] * 10
    # min_segments_per_doc=2, providing only 1 boundary → reject.
    assert loader._passes_filters(sentences, boundaries=[0]) is False


def test_passes_filters_accepts_in_range(
    base_config: WikipediaDatasetConfig,
) -> None:
    loader = WikipediaLoader(base_config)
    sentences = ["x" * 30] * 12
    assert loader._passes_filters(sentences, boundaries=[0, 6]) is True


def test_passes_filters_rejects_too_few_sentences(
    base_config: WikipediaDatasetConfig,
) -> None:
    loader = WikipediaLoader(base_config)
    # min_sentences_per_doc=3, only 2 provided.
    assert loader._passes_filters(["abc", "def"], boundaries=[0, 1]) is False


# ── load() integration with mocked MediaWiki API ─────────────────────────────


def _mock_response(extract: str) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"query": {"pages": {"1": {"extract": extract}}}}
    response.raise_for_status.return_value = None
    return response


def test_load_writes_documents_boundaries_and_metadata(
    tmp_path: Path, base_config: WikipediaDatasetConfig
) -> None:
    extract = _make_extract(
        (
            "Historia",
            [
                "Frase histórica larga número uno del artículo.",
                "Frase histórica larga número dos del artículo.",
            ],
        ),
        (
            "Geografía",
            [
                "Frase geográfica larga número uno del artículo.",
                "Frase geográfica larga número dos del artículo.",
            ],
        ),
        intro=["Introducción larga del artículo de prueba número uno."],
    )

    loader = WikipediaLoader(base_config)
    loader._session = MagicMock()
    loader._session.get.return_value = _mock_response(extract)

    meta = loader.load(tmp_path)

    assert meta.n_documents == 1
    assert meta.n_skipped == 0
    assert (tmp_path / "documents" / "doc_0001.txt").exists()
    boundary_file = tmp_path / "boundaries" / "doc_0001.json"
    assert boundary_file.exists()

    payload = json.loads(boundary_file.read_text(encoding="utf-8"))
    assert payload["boundaries"][0] == 0  # convention: first boundary is 0
    assert payload["source_title"] == "Dummy"
    assert payload["n_segments"] == len(payload["boundaries"])

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == "wikipedia"
    assert metadata["n_documents"] == 1


def test_load_skips_missing_articles(
    tmp_path: Path, base_config: WikipediaDatasetConfig
) -> None:
    """Articles flagged as 'missing' by the API must be counted as skipped."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"query": {"pages": {"-1": {"missing": ""}}}}
    response.raise_for_status.return_value = None

    loader = WikipediaLoader(base_config)
    loader._session = MagicMock()
    loader._session.get.return_value = response

    meta = loader.load(tmp_path)

    assert meta.n_documents == 0
    assert meta.n_skipped == 1
    assert list((tmp_path / "documents").glob("*.txt")) == []
