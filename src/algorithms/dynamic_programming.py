from __future__ import annotations

from time import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult


class DPSegmenter(BaseSegmenter):
    """Exact optimal segmentation via dynamic programming.

    Finds the partition of n sentences into k segments that minimises the
    total intra-segment heterogeneity, defined as:

        cost(a, b) = 1 - mean_pairwise_cosine_similarity(sentences[a:b])

    Similarity is computed on TF-IDF vectors of the sentences.

    Time complexity:  O(n^2 * k)
    Space complexity: O(n^2 + n * k)
    """

    @property
    def name(self) -> str:
        return "dynamic_programming"

    def segment(
        self, document: Document, max_segments: int | None = None
    ) -> SegmentationResult:
        """Segment document using exact DP optimisation.

        Args:
            document: Document to segment.
            max_segments: Maximum number of segments (default: min(5, n_sentences)).

        Returns:
            SegmentationResult with globally-optimal boundary positions.
        """
        start_time = time()
        n = document.n_sentences

        if n == 1:
            return SegmentationResult(
                doc_id=document.doc_id,
                boundaries=[0],
                algorithm_name=self.name,
                runtime_seconds=time() - start_time,
            )

        k = min(max_segments or 5, n)
        cohesion = self._build_cohesion_matrix(document.sentences)
        boundaries = self._run_dp(cohesion, n, k)

        return SegmentationResult(
            doc_id=document.doc_id,
            boundaries=boundaries,
            algorithm_name=self.name,
            runtime_seconds=time() - start_time,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _build_cohesion_matrix(self, sentences: list[str]) -> np.ndarray:
        """Compute cohesion[i][j] = length-weighted mean cosine similarity
        for the segment spanning sentences[i:j+1].

        Using mean_sim * (segment_length / n) rewards longer coherent segments
        and prevents the DP from degenerating to single-sentence partitions
        (which would have cohesion = 0 and always look "optimal" otherwise).

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
                    # Single sentence has no pairs → cohesion = 0.
                    # Length weighting naturally discourages this vs a longer
                    # segment with even modest internal similarity.
                    cohesion[i][j] = 0.0
                else:
                    sub = sim_full[i : j + 1, i : j + 1]
                    upper = sub[np.triu_indices(seg_len, k=1)]
                    mean_sim = float(upper.mean())
                    # Weight by relative segment length so longer coherent
                    # segments score better than many small ones.
                    cohesion[i][j] = mean_sim * seg_len / n
        return cohesion

    def _run_dp(self, cohesion: np.ndarray, n: int, k: int) -> list[int]:
        """Run the DP recurrence maximising total length-weighted cohesion.

        dp[i][j] = maximum cohesion of partitioning sentences[0:i] into j segments.

        Args:
            cohesion: Pre-computed cohesion matrix (n x n).
            n: Number of sentences.
            k: Number of segments.

        Returns:
            Sorted list of boundary indices starting with 0.
        """
        NEG_INF = float("-inf")
        dp = np.full((n + 1, k + 1), NEG_INF)
        split = np.full((n + 1, k + 1), -1, dtype=int)

        dp[0][0] = 0.0

        for j in range(1, k + 1):
            for i in range(j, n + 1):
                for i_prev in range(j - 1, i):
                    val = dp[i_prev][j - 1] + cohesion[i_prev][i - 1]
                    if val > dp[i][j]:
                        dp[i][j] = val
                        split[i][j] = i_prev

        # Backtrack from the best valid k (first from k downward that is finite)
        best_k = k
        for candidate in range(k, 0, -1):
            if dp[n][candidate] > NEG_INF:
                best_k = candidate
                break

        boundaries: list[int] = []
        pos, segs = n, best_k
        while segs > 0:
            prev = split[pos][segs]
            boundaries.append(prev)
            pos = prev
            segs -= 1

        boundaries.reverse()
        return boundaries
