from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Multilingual Sentence-BERT model used by the dense-embedding backend.
# Chosen for its Spanish support; it runs fully locally (no API call, so it is
# unaffected by the OFAC restrictions that block commercial LLM endpoints).
SBERT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Cohesion backends selectable per segmenter via the ``cohesion_backend`` param.
VALID_BACKENDS = ("tfidf", "sbert")


@lru_cache(maxsize=1)
def _load_sbert_model(model_name: str):
    """Load and memoise the SentenceTransformer model.

    The import and model load are deferred so the heavy ``sentence-transformers``
    dependency stays optional: code paths that use the default TF-IDF backend
    never touch it.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        A loaded ``SentenceTransformer`` instance (cached across calls).

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "The 'sbert' cohesion backend requires the optional dependency "
            "'sentence-transformers'. Install it with: "
            "pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def embed_sentences(sentences: list[str], backend: str = "tfidf") -> np.ndarray:
    """Embed sentences into dense vectors using the requested backend.

    Both backends return one row per sentence so downstream cosine-similarity
    code is identical regardless of the representation.

    Args:
        sentences: List of sentence strings.
        backend: ``"tfidf"`` (sparse lexical, default) or ``"sbert"`` (dense
            multilingual sentence embeddings).

    Returns:
        Dense array of shape (n_sentences, embedding_dim).

    Raises:
        ValueError: If ``backend`` is not a recognised value.
    """
    if backend == "tfidf":
        vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
        return vectorizer.fit_transform(sentences).toarray()
    if backend == "sbert":
        model = _load_sbert_model(SBERT_MODEL_NAME)
        return np.asarray(
            model.encode(list(sentences), convert_to_numpy=True), dtype=float
        )
    raise ValueError(
        f"Unknown cohesion backend {backend!r}; expected one of {VALID_BACKENDS}"
    )


def build_cohesion_matrix(sentences: list[str], backend: str = "tfidf") -> np.ndarray:
    """Length-weighted mean cosine similarity for all sentence spans.

    cohesion[i][j] = mean_pairwise_cosine_sim(sentences[i:j+1]) * (j-i+1) / n

    Single-sentence spans get 0.0. The length weighting prevents degenerate
    solutions where many single-sentence segments always appear "optimal".

    Args:
        sentences: List of sentence strings.
        backend: Sentence representation backend; see :func:`embed_sentences`.

    Returns:
        Upper-triangular cohesion matrix of shape (n, n).
    """
    n = len(sentences)
    vectors = embed_sentences(sentences, backend=backend)
    sim_full = cosine_similarity(vectors)

    cohesion = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            seg_len = j - i + 1
            if seg_len == 1:
                cohesion[i][j] = 0.0
            else:
                sub = sim_full[i : j + 1, i : j + 1]
                upper = sub[np.triu_indices(seg_len, k=1)]
                cohesion[i][j] = float(upper.mean()) * seg_len / n
    return cohesion
