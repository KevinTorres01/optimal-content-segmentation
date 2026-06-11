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

from dataclasses import dataclass

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
    """

    k: int
    objectives: dict[int, float]
    rationale: str


def _dp_objective_per_k(
    cohesion: np.ndarray, n: int, k_max: int
) -> dict[int, float]:
    """Run DP once and return J(exactly j) for every j in [1, k_max].

    Reads dp[n][j] directly from the DP table, bypassing the public
    DPSegmenter API which collapses to the best j it finds. This is
    essential for auto-k: we need the true curve of exact-k optima,
    not the saturated "best so far" curve.
    """
    NEG_INF = float("-inf")
    dp = np.full((n + 1, k_max + 1), NEG_INF)
    dp[0][0] = 0.0
    for j in range(1, k_max + 1):
        for i in range(j, n + 1):
            for i_prev in range(j - 1, i):
                val = dp[i_prev][j - 1] + cohesion[i_prev][i - 1]
                if val > dp[i][j]:
                    dp[i][j] = val
    return {j: float(dp[n][j]) for j in range(1, k_max + 1) if dp[n][j] > NEG_INF}


def find_optimal_k(
    document: Document,
    k_min: int = 2,
    k_max: int | None = None,
) -> AutoKResult:
    """Pick k automatically using the elbow method on the DP-optimal objective.

    Runs exact DP for every k in [k_min, k_max], computes the optimal
    cohesion J(k), and selects the elbow — the point of maximum vertical
    distance between the normalised curve and the chord joining its
    endpoints.

    Args:
        document: Document to segment.
        k_min: Smallest k to consider (must be >= 2).
        k_max: Largest k to consider. Defaults to min(n - 1, max(5, n // 2)),
            which keeps the sweep cheap and ignores the trivial high-k tail.

    Returns:
        AutoKResult with the chosen k, the full {k: J(k)} curve, and a
        short rationale string.
    """
    n = document.n_sentences

    if n <= 1:
        return AutoKResult(k=1, objectives={1: 0.0}, rationale="documento trivial")
    if n == 2:
        return AutoKResult(
            k=2, objectives={2: 0.0}, rationale="solo cabe k=2 con 2 oraciones"
        )

    if k_max is None:
        k_max = min(n - 1, max(5, n // 2))
    k_min = max(2, min(k_min, k_max))

    if k_min == k_max:
        return AutoKResult(
            k=k_min,
            objectives={k_min: 0.0},
            rationale=f"único candidato posible (k={k_min})",
        )

    cohesion = build_cohesion_matrix(document.sentences)
    full = _dp_objective_per_k(cohesion, n, k_max)
    objectives = {k: full[k] for k in range(k_min, k_max + 1) if k in full}

    ks = np.array(sorted(objectives.keys()), dtype=float)
    js = np.array([objectives[int(k)] for k in ks], dtype=float)

    j_range = js.max() - js.min()
    if j_range == 0:
        return AutoKResult(
            k=int(ks[0]),
            objectives=objectives,
            rationale="curva plana — sin gradiente para detectar codo",
        )

    x_norm = (ks - ks.min()) / (ks.max() - ks.min())
    y_norm = (js - js.min()) / j_range

    distances = y_norm - x_norm
    best_idx = int(np.argmax(distances))

    # Refinement: if there are further "significant" jumps after the knee,
    # extend k. A jump counts as significant when it exceeds 5% of the total
    # gain across the curve. This rescues cases where the curve has two
    # distinct upward steps (e.g. multi-topic texts where TF-IDF separates
    # the dominant theme first and a finer split second).
    threshold = 0.05 * j_range
    for i in range(best_idx + 1, len(ks)):
        if js[i] - js[i - 1] >= threshold:
            best_idx = i

    chosen = int(ks[best_idx])
    rationale = (
        f"codo del objetivo en k∈[{int(ks.min())},{int(ks.max())}] "
        f"(Kneedle + refinamiento)"
    )
    return AutoKResult(k=chosen, objectives=objectives, rationale=rationale)
