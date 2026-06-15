"""Automatic selection of the number of segments k.

The length-weighted cohesion objective grows monotonically with k (the
trivial maximum is k = n, one sentence per segment). To pick a meaningful
k automatically we apply the classical elbow (Kneedle) heuristic: compute
the DP-optimal objective J(k) for k = k_min .. k_max, then pick the k
where the marginal gain in cohesion drops most sharply.

References:
    Satopaa et al. (2011). Finding a "kneedle" in a haystack: detecting
    knee points in system behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.algorithms._cohesion import build_cohesion_matrix
from src.core.models import Document


@dataclass
class AutoKResult:
    """Outcome of an automatic k selection.

    Attributes:
        k: The selected number of segments.
        objectives: Map {k: J(k)} for the values explored.
        rationale: Short human-readable explanation of the decision.
        boundaries: Optimal boundary positions for the chosen k, recovered
            directly from the DP split table built during the sweep — no
            second DP pass needed.
    """

    k: int
    objectives: dict[int, float]
    rationale: str
    boundaries: list[int] = field(default_factory=list)


def _dp_sweep(
    cohesion: np.ndarray, n: int, k_max: int
) -> tuple[dict[int, float], np.ndarray]:
    """Run DP once, storing both objective values and split pointers.

    Returns J(exactly j) for every j in [1, k_max] AND the full split
    matrix needed to backtrack boundaries for any chosen k — so callers
    that also need a segmentation avoid a second DP pass.

    Args:
        cohesion: Pre-computed cohesion matrix (n × n).
        n: Number of sentences.
        k_max: Maximum number of segments to explore.

    Returns:
        objectives: dict {j: dp[n][j]} for j in [1, k_max].
        split: int array (n+1, k_max+1) of backtracking pointers.
    """
    NEG_INF = float("-inf")
    dp = np.full((n + 1, k_max + 1), NEG_INF)
    split = np.full((n + 1, k_max + 1), -1, dtype=np.intp)
    dp[0][0] = 0.0
    for j in range(1, k_max + 1):
        for i in range(j, n + 1):
            for i_prev in range(j - 1, i):
                val = dp[i_prev][j - 1] + cohesion[i_prev][i - 1]
                if val > dp[i][j]:
                    dp[i][j] = val
                    split[i][j] = i_prev
    objectives = {j: float(dp[n][j]) for j in range(1, k_max + 1) if dp[n][j] > NEG_INF}
    return objectives, split


def _backtrack(split: np.ndarray, n: int, k: int) -> list[int]:
    """Recover boundary positions from the split matrix in O(k·n) time."""
    boundaries: list[int] = []
    pos, segs = n, k
    while segs > 0:
        prev = int(split[pos][segs])
        boundaries.append(prev)
        pos = prev
        segs -= 1
    boundaries.reverse()
    return boundaries


def find_optimal_k(
    document: Document,
    k_min: int = 2,
    k_max: int | None = None,
) -> AutoKResult:
    """Pick k automatically using the elbow method on the DP-optimal objective.

    Runs a single DP sweep over all k in [k_min, k_max], stores the split
    matrix, chooses the elbow k, and recovers boundaries via backtracking —
    all in one pass. Callers that use DP as their segmentation algorithm
    can use ``result.boundaries`` directly and skip a second DP run.

    Args:
        document: Document to segment.
        k_min: Smallest k to consider (must be >= 2).
        k_max: Largest k to consider. Defaults to min(n - 1, max(5, ceil(√n))).
            The √n cap has a geometric justification: with k segments the
            average segment size is n/k sentences. Setting k = n/k gives
            k² = n → k = √n, the unique point where segment count equals
            average segment length. Beyond √n segments are shorter on average
            than the number of segments, signalling fragmentation; TF-IDF also
            becomes unreliable on very short segments. The max(5, …) floor is
            a practical guard for short documents (n < 25) where √n < 5 and
            the range [2, √n] would be too narrow for elbow detection.

    Returns:
        AutoKResult with the chosen k, the full {k: J(k)} curve, a short
        rationale string, and the optimal boundaries for the chosen k.
    """
    n = document.n_sentences

    if n <= 1:
        return AutoKResult(
            k=1, objectives={1: 0.0}, rationale="documento trivial", boundaries=[0]
        )
    if n == 2:
        return AutoKResult(
            k=2,
            objectives={2: 0.0},
            rationale="solo cabe k=2 con 2 oraciones",
            boundaries=[0, 1],
        )

    if k_max is None:
        k_max = min(n - 1, max(5, math.ceil(math.sqrt(n))))
    k_min = max(2, min(k_min, k_max))

    if k_min == k_max:
        cohesion = build_cohesion_matrix(document.sentences)
        _, split = _dp_sweep(cohesion, n, k_max)
        boundaries = _backtrack(split, n, k_min)
        return AutoKResult(
            k=k_min,
            objectives={k_min: 0.0},
            rationale=f"único candidato posible (k={k_min})",
            boundaries=boundaries,
        )

    cohesion = build_cohesion_matrix(document.sentences)
    full, split = _dp_sweep(cohesion, n, k_max)
    objectives = {k: full[k] for k in range(k_min, k_max + 1) if k in full}

    ks = np.array(sorted(objectives.keys()), dtype=float)
    js = np.array([objectives[int(k)] for k in ks], dtype=float)

    j_range = js.max() - js.min()
    if j_range == 0:
        chosen = int(ks[0])
        return AutoKResult(
            k=chosen,
            objectives=objectives,
            rationale="curva plana — sin gradiente para detectar codo",
            boundaries=_backtrack(split, n, chosen),
        )

    x_norm = (ks - ks.min()) / (ks.max() - ks.min())
    y_norm = (js - js.min()) / j_range

    # Kneedle: the elbow is the point of maximum vertical distance between the
    # normalised (concave) curve and the chord joining its endpoints — i.e. the
    # k at which adding another segment stops paying off in cohesion.
    distances = y_norm - x_norm
    best_idx = int(np.argmax(distances))

    chosen = int(ks[best_idx])
    rationale = f"codo (Kneedle) del objetivo en k∈[{int(ks.min())},{int(ks.max())}]"
    return AutoKResult(
        k=chosen,
        objectives=objectives,
        rationale=rationale,
        boundaries=_backtrack(split, n, chosen),
    )
