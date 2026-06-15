from __future__ import annotations

from time import time

import numpy as np

from src.algorithms._cohesion import embed_sentences
from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult


class GreedySegmenter(BaseSegmenter):
    """TextTiling-inspired greedy segmentation heuristic.

    For each gap between consecutive sentences, computes the cosine similarity
    between the average TF-IDF vectors of the windows on each side. Boundaries
    are placed at the k-1 gap positions with the deepest similarity valleys
    (lowest block-to-block similarity).

    Time complexity: O(n * w) where w is the window size.
    """

    def __init__(self, window_size: int = 2, cohesion_backend: str = "tfidf") -> None:
        self._window_size = window_size
        self._cohesion_backend = cohesion_backend

    @property
    def name(self) -> str:
        return "greedy"

    def segment(
        self, document: Document, max_segments: int | None = None
    ) -> SegmentationResult:
        """Greedily segment a document using block-similarity valleys.

        Args:
            document: Document to segment.
            max_segments: Maximum number of segments (default: min(5, n)).

        Returns:
            SegmentationResult with boundary positions at similarity valleys.
        """
        start = time()
        n = document.n_sentences
        k = min(max_segments or 5, n)

        if n == 1 or k == 1:
            return SegmentationResult(
                doc_id=document.doc_id,
                boundaries=[0],
                algorithm_name=self.name,
                runtime_seconds=time() - start,
            )

        vectors = embed_sentences(document.sentences, backend=self._cohesion_backend)

        gap_sims = self._block_similarities(vectors, n)
        depth = self._depth_scores(gap_sims)

        n_splits = k - 1
        top_gaps = sorted(
            sorted(range(len(depth)), key=lambda i: depth[i], reverse=True)[:n_splits]
        )
        # Gap index i sits between sentence i and i+1 → boundary at i+1
        boundaries = [0] + [g + 1 for g in top_gaps]

        return SegmentationResult(
            doc_id=document.doc_id,
            boundaries=boundaries,
            algorithm_name=self.name,
            runtime_seconds=time() - start,
        )

    def _block_similarities(self, vectors: np.ndarray, n: int) -> list[float]:
        """Cosine similarity between left and right window vectors at each gap."""
        w = self._window_size
        sims: list[float] = []
        for gap in range(n - 1):
            left_vec = vectors[max(0, gap - w + 1) : gap + 1].mean(axis=0)
            right_vec = vectors[gap + 1 : min(n, gap + 1 + w)].mean(axis=0)
            denom = np.linalg.norm(left_vec) * np.linalg.norm(right_vec)
            sims.append(
                float(np.dot(left_vec, right_vec) / denom) if denom > 1e-10 else 0.0
            )
        return sims

    def _depth_scores(self, sims: list[float]) -> list[float]:
        """Valley depth at each gap: how far below its neighbours it sits."""
        m = len(sims)
        depth: list[float] = []
        for i in range(m):
            left_peak = max(sims[:i], default=sims[i])
            right_peak = max(sims[i + 1 :], default=sims[i])
            depth.append((left_peak - sims[i]) + (right_peak - sims[i]))
        return depth
