"""Unit tests for the cohesion backends (TF-IDF lexical vs SBERT semantic).

The SBERT tests are skipped automatically when ``sentence-transformers`` is not
installed or the model cannot be loaded (e.g. offline), so the default CI path
stays dependency-light.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms._cohesion import (
    SBERT_MODEL_NAME,
    build_cohesion_matrix,
    embed_sentences,
)
from src.core.models import Document

# Two sentences that mean the same thing but share no content words — only the
# stopwords "el"/"es". TF-IDF sees them as nearly unrelated; a semantic model
# should rate them very similar. This is the crux of the TF-IDF limitation the
# SBERT backend is meant to fix.
_SYN_A = "El automóvil es veloz."
_SYN_B = "El coche es rápido."


# ── TF-IDF backend (default, always available) ────────────────────────────────


def test_default_backend_is_tfidf_matrix_shape():
    sentences = ["Primera oración del texto.", "Segunda oración distinta.", "Tercera."]
    matrix = build_cohesion_matrix(sentences)
    assert matrix.shape == (3, 3)


def test_single_sentence_spans_are_zero():
    sentences = ["Una oración.", "Otra oración.", "Y una más."]
    matrix = build_cohesion_matrix(sentences)
    for i in range(len(sentences)):
        assert matrix[i][i] == 0.0


def test_embed_sentences_tfidf_returns_dense_2d():
    sentences = ["Hola mundo entero.", "Adiós mundo cruel."]
    vectors = embed_sentences(sentences, backend="tfidf")
    assert isinstance(vectors, np.ndarray)
    assert vectors.ndim == 2
    assert vectors.shape[0] == 2


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown cohesion backend"):
        embed_sentences(["x"], backend="word2vec")


def test_explicit_tfidf_matches_default():
    sentences = ["Tópico uno aquí.", "Tópico uno también.", "Tópico dos diferente."]
    default = build_cohesion_matrix(sentences)
    explicit = build_cohesion_matrix(sentences, backend="tfidf")
    np.testing.assert_array_equal(default, explicit)


# ── SBERT backend (optional dependency) ───────────────────────────────────────


@pytest.fixture(scope="module")
def sbert_available():
    """Skip the whole SBERT group if the model can't be loaded (offline/no dep)."""
    pytest.importorskip("sentence_transformers")
    try:
        embed_sentences(["prueba de carga"], backend="sbert")
    except Exception as exc:  # noqa: BLE001 - any load failure → skip, don't fail
        pytest.skip(f"SBERT model unavailable: {exc}")
    return True


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def test_sbert_embeddings_are_dense(sbert_available):
    vectors = embed_sentences([_SYN_A, _SYN_B], backend="sbert")
    assert vectors.ndim == 2
    assert vectors.shape[0] == 2
    # Dense embeddings: no all-zero rows and a fixed, modest dimensionality.
    assert vectors.shape[1] < 2000
    assert not np.any(np.all(vectors == 0.0, axis=1))


def test_sbert_relates_synonyms_better_than_tfidf(sbert_available):
    """Core demonstration of the improvement: synonyms with no shared words."""
    tfidf_vecs = embed_sentences([_SYN_A, _SYN_B], backend="tfidf")
    sbert_vecs = embed_sentences([_SYN_A, _SYN_B], backend="sbert")

    tfidf_sim = _cosine(tfidf_vecs[0], tfidf_vecs[1])
    sbert_sim = _cosine(sbert_vecs[0], sbert_vecs[1])

    # TF-IDF can only match the shared stopwords, so similarity stays low; the
    # semantic model recognises the paraphrase and rates them clearly closer.
    assert sbert_sim > tfidf_sim
    assert sbert_sim > 0.5


def test_sbert_backend_produces_valid_segmentation(sbert_available):
    from src.algorithms.dynamic_programming import DPSegmenter

    document = Document(
        doc_id="syn",
        sentences=[
            "El automóvil recorre la autopista.",
            "El coche acelera en la carretera.",
            "La cocinera prepara la cena.",
            "El chef hornea un pastel delicioso.",
        ],
    )
    result = DPSegmenter(cohesion_backend="sbert").segment(document, max_segments=2)
    assert result.boundaries[0] == 0
    assert all(0 <= b < document.n_sentences for b in result.boundaries)


def test_sbert_model_name_is_multilingual():
    assert "multilingual" in SBERT_MODEL_NAME
