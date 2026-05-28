from __future__ import annotations

from itertools import combinations
from time import time

from src.algorithms._cohesion import build_cohesion_matrix
from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult

_MAX_SENTENCES = 15


class BruteForceSegmenter(BaseSegmenter):
    """Exact segmentation by exhaustive enumeration of all partitions.

    Enumerates all C(n-1, k-1) ways to place k-1 internal boundaries and
    picks the partition with the maximum total length-weighted cohesion.

    Only practical for small documents (n <= 15). Useful as a correctness
    reference for the DP and other algorithms.

    Time complexity:  O(C(n-1, k-1) * n)
    """

    @property
    def name(self) -> str:
        return "brute_force"

    def segment(
        self, document: Document, max_segments: int | None = None
    ) -> SegmentationResult:
        """Exhaustively find the globally optimal segmentation.

        Args:
            document: Document to segment. Must have <= 15 sentences.
            max_segments: Maximum number of segments (default: min(5, n)).

        Returns:
            SegmentationResult with globally-optimal boundary positions.

        Raises:
            ValueError: If the document has more than _MAX_SENTENCES sentences.
        """
        start = time()
        n = document.n_sentences

        if n > _MAX_SENTENCES:
            raise ValueError(
                f"BruteForceSegmenter supports at most {_MAX_SENTENCES} sentences; "
                f"document '{document.doc_id}' has {n}. Use dynamic_programming instead."
            )

        if n == 1:
            return SegmentationResult(
                doc_id=document.doc_id,
                boundaries=[0],
                algorithm_name=self.name,
                runtime_seconds=time() - start,
            )

        k = min(max_segments or 5, n)
        cohesion = build_cohesion_matrix(document.sentences)
        boundaries = self._exhaustive_search(cohesion, n, k)

        return SegmentationResult(
            doc_id=document.doc_id,
            boundaries=boundaries,
            algorithm_name=self.name,
            runtime_seconds=time() - start,
        )

    def _exhaustive_search(
        self, cohesion, n: int, k: int
    ) -> list[int]:
        """Enumerate all C(n-1, k-1) partitions and return the best."""
        best_score = float("-inf")
        best_boundaries: list[int] = [0]

        for internal in combinations(range(1, n), k - 1):
            boundaries = [0] + list(internal)
            ends = list(internal) + [n]
            score = sum(
                cohesion[b][e - 1] for b, e in zip(boundaries, ends)
            )
            if score > best_score:
                best_score = score
                best_boundaries = boundaries

        return best_boundaries
