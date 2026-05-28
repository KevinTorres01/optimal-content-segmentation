from __future__ import annotations

import math
import random
from time import time

from src.algorithms._cohesion import build_cohesion_matrix
from src.core.interfaces import BaseSegmenter
from src.core.models import Document, SegmentationResult


class SASegmenter(BaseSegmenter):
    """Simulated Annealing metaheuristic for text segmentation.

    Starts from a uniformly-spaced partition and iteratively perturbs one
    boundary by ±1 sentence. Improvements are always accepted; worse moves
    are accepted with probability exp(Δ/T) so the search can escape local
    optima. Temperature decreases geometrically each iteration.

    Uses the same length-weighted cohesion objective as DPSegmenter, so
    results are directly comparable.

    Time complexity: O(n^2 + n_iterations)
    """

    def __init__(
        self,
        n_iterations: int = 1000,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        random_seed: int | None = None,
    ) -> None:
        self._n_iterations = n_iterations
        self._initial_temp = initial_temp
        self._cooling_rate = cooling_rate
        self._random_seed = random_seed

    @property
    def name(self) -> str:
        return "simulated_annealing"

    def segment(
        self, document: Document, max_segments: int | None = None
    ) -> SegmentationResult:
        """Segment document using Simulated Annealing optimisation.

        Args:
            document: Document to segment.
            max_segments: Maximum number of segments (default: min(5, n)).

        Returns:
            SegmentationResult with near-optimal boundary positions.
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

        cohesion = build_cohesion_matrix(document.sentences)
        rng = random.Random(self._random_seed)
        boundaries = self._run_sa(cohesion, n, k, rng)

        return SegmentationResult(
            doc_id=document.doc_id,
            boundaries=boundaries,
            algorithm_name=self.name,
            runtime_seconds=time() - start,
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _total_cohesion(self, cohesion, boundaries: list[int], n: int) -> float:
        ends = boundaries[1:] + [n]
        return sum(cohesion[b][e - 1] for b, e in zip(boundaries, ends))

    def _run_sa(self, cohesion, n: int, k: int, rng: random.Random) -> list[int]:
        # Initialise with uniformly-spaced boundaries
        step = n // k
        current = [0] + sorted(
            {min(i * step, n - 1) for i in range(1, k)}
        )
        # Ensure exactly k boundaries (pad from remaining positions if needed)
        all_positions = set(range(n))
        while len(current) < k:
            candidates = sorted(all_positions - set(current))
            if not candidates:
                break
            current.append(rng.choice(candidates))
            current.sort()
        current = current[:k]

        current_score = self._total_cohesion(cohesion, current, n)
        best = list(current)
        best_score = current_score
        temp = self._initial_temp

        for _ in range(self._n_iterations):
            if len(current) < 2:
                break

            # Choose a random internal boundary to perturb
            idx = rng.randint(1, len(current) - 1)
            delta = rng.choice((-1, 1))
            new_pos = current[idx] + delta

            # Validity: must stay strictly between its neighbours
            prev = current[idx - 1]
            nxt = current[idx + 1] if idx + 1 < len(current) else n
            if not (prev < new_pos < nxt):
                temp *= self._cooling_rate
                continue

            candidate = list(current)
            candidate[idx] = new_pos
            candidate_score = self._total_cohesion(cohesion, candidate, n)

            d = candidate_score - current_score
            if d > 0 or rng.random() < math.exp(d / max(temp, 1e-10)):
                current = candidate
                current_score = candidate_score
                if current_score > best_score:
                    best = list(current)
                    best_score = current_score

            temp *= self._cooling_rate

        return best
