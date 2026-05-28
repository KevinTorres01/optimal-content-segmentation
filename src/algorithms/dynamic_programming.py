from __future__ import annotations

from time import time

import numpy as np

from src.algorithms._cohesion import build_cohesion_matrix
from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult


class DPSegmenter(BaseSegmenter):
    """Exact optimal segmentation via dynamic programming.

    Finds the partition of n sentences into k segments that maximises the
    total length-weighted intra-segment cosine cohesion.

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
        cohesion = build_cohesion_matrix(document.sentences)
        boundaries = self._run_dp(cohesion, n, k)

        return SegmentationResult(
            doc_id=document.doc_id,
            boundaries=boundaries,
            algorithm_name=self.name,
            runtime_seconds=time() - start_time,
        )

    # ── private helpers ───────────────────────────────────────────────────────

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
