from __future__ import annotations


def _boundaries_to_mass(boundaries: list[int], n_sentences: int) -> list[int]:
    """Convert boundary list to a segment-mass representation.

    Each position i in the returned list contains the segment index of
    sentence i (0-based). This is used internally by Pk and WindowDiff.

    Args:
        boundaries: Sorted boundary list starting with 0.
        n_sentences: Total number of sentences in the document.

    Returns:
        List of length n_sentences where each value is the segment index.
    """
    mass = [0] * n_sentences
    seg_idx = 0
    bound_set = set(boundaries)
    for i in range(n_sentences):
        if i > 0 and i in bound_set:
            seg_idx += 1
        mass[i] = seg_idx
    return mass


def _default_k(boundaries: list[int], n_sentences: int) -> int:
    """Compute standard window size k = n_sentences // (2 * n_segments)."""
    n_segments = len(boundaries)
    if n_segments == 0:
        return max(1, n_sentences // 2)
    return max(1, n_sentences // (2 * n_segments))


def compute_pk(
    reference: list[int],
    hypothesis: list[int],
    n_sentences: int,
    k: int | None = None,
) -> float:
    """Compute the Pk metric (Beeferman et al., 1999).

    Pk measures the probability that two sentences k positions apart are
    incorrectly classified as belonging to the same or different segments.
    Lower is better; 0 = perfect agreement with reference.

    Args:
        reference: Ground-truth boundary positions (sorted, starts at 0).
        hypothesis: Predicted boundary positions.
        n_sentences: Total number of sentences.
        k: Window size. Defaults to n_sentences // (2 * n_ref_segments).

    Returns:
        Pk score in [0, 1]. Returns 0.0 for single-sentence documents.
    """
    if n_sentences <= 1:
        return 0.0

    window = k if k is not None else _default_k(reference, n_sentences)
    ref_mass = _boundaries_to_mass(reference, n_sentences)
    hyp_mass = _boundaries_to_mass(hypothesis, n_sentences)

    errors = 0
    total = 0
    for i in range(n_sentences - window):
        j = i + window
        ref_same = ref_mass[i] == ref_mass[j]
        hyp_same = hyp_mass[i] == hyp_mass[j]
        if ref_same != hyp_same:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


def compute_window_diff(
    reference: list[int],
    hypothesis: list[int],
    n_sentences: int,
    k: int | None = None,
) -> float:
    """Compute the WindowDiff metric (Pevzner & Hearst, 2002).

    WindowDiff penalises both near-misses and false alarms by counting the
    absolute difference in the number of boundaries within each window.
    Lower is better; 0 = perfect agreement with reference.

    Args:
        reference: Ground-truth boundary positions (sorted, starts at 0).
        hypothesis: Predicted boundary positions.
        n_sentences: Total number of sentences.
        k: Window size. Defaults to n_sentences // (2 * n_ref_segments).

    Returns:
        WindowDiff score in [0, 1]. Returns 0.0 for single-sentence documents.
    """
    if n_sentences <= 1:
        return 0.0

    window = k if k is not None else _default_k(reference, n_sentences)
    ref_set = set(reference)
    hyp_set = set(hypothesis)

    errors = 0
    total = 0
    for i in range(n_sentences - window):
        j = i + window
        ref_count = sum(1 for b in ref_set if i < b <= j)
        hyp_count = sum(1 for b in hyp_set if i < b <= j)
        if ref_count != hyp_count:
            errors += 1
        total += 1

    return errors / total if total > 0 else 0.0


def compute_f1_boundary(
    reference: list[int],
    hypothesis: list[int],
    tolerance: int = 1,
) -> tuple[float, float, float]:
    """Compute precision, recall and F1 over individual boundary positions.

    A predicted boundary counts as correct if it is within ±tolerance
    sentences of a reference boundary. Each reference boundary can only
    be matched once (greedy left-to-right matching).

    The boundary at position 0 (start of document) is excluded from the
    comparison as it is trivially correct for all algorithms.

    Args:
        reference: Ground-truth boundary positions (sorted, starts at 0).
        hypothesis: Predicted boundary positions.
        tolerance: Allowed distance in sentences (default 1).

    Returns:
        Tuple of (precision, recall, f1), each in [0, 1].
    """
    ref_inner = [b for b in reference if b != 0]
    hyp_inner = [b for b in hypothesis if b != 0]

    if not ref_inner and not hyp_inner:
        return 1.0, 1.0, 1.0
    if not ref_inner:
        return 0.0, 1.0, 0.0
    if not hyp_inner:
        return 1.0, 0.0, 0.0

    matched_ref = set()
    true_positives = 0

    for h in hyp_inner:
        for r in ref_inner:
            if r in matched_ref:
                continue
            if abs(h - r) <= tolerance:
                true_positives += 1
                matched_ref.add(r)
                break

    precision = true_positives / len(hyp_inner) if hyp_inner else 0.0
    recall = true_positives / len(ref_inner) if ref_inner else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
