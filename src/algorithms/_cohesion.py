from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_cohesion_matrix(sentences: list[str]) -> np.ndarray:
    """Length-weighted mean cosine similarity for all sentence spans.

    cohesion[i][j] = mean_pairwise_cosine_sim(sentences[i:j+1]) * (j-i+1) / n

    Single-sentence spans get 0.0. The length weighting prevents degenerate
    solutions where many single-sentence segments always appear "optimal".

    Args:
        sentences: List of sentence strings.

    Returns:
        Upper-triangular cohesion matrix of shape (n, n).
    """
    n = len(sentences)
    vectorizer = TfidfVectorizer(min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(sentences)
    sim_full = cosine_similarity(tfidf)

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
