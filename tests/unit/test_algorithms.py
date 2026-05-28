"""Unit tests for BruteForceSegmenter, GreedySegmenter, and SASegmenter."""

import pytest

from src.algorithms.brute_force import BruteForceSegmenter, _MAX_SENTENCES
from src.algorithms.greedy import GreedySegmenter
from src.algorithms.simulated_annealing import SASegmenter
from src.core.models import Document, SegmentationResult


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_doc(sentences: list[str], doc_id: str = "test") -> Document:
    return Document(doc_id=doc_id, sentences=sentences)


# Three clearly distinct topics — 3 sentences each (9 total)
_SPORTS = [
    "The team scored three goals in the second half.",
    "The goalkeeper saved two penalty kicks.",
    "The coach praised the players after the match.",
]
_TECH = [
    "The new processor has eight cores and sixteen threads.",
    "RAM capacity doubled with the latest hardware revision.",
    "Benchmark results show a 40% speed improvement.",
]
_COOKING = [
    "Preheat the oven to 180 degrees Celsius.",
    "Mix flour, eggs, and butter in a large bowl.",
    "Bake for 25 minutes until golden brown.",
]
_MIXED = _SPORTS + _TECH + _COOKING  # 9 sentences, 3 natural segments


@pytest.fixture
def mixed_doc() -> Document:
    return _make_doc(_MIXED)


@pytest.fixture
def tiny_doc() -> Document:
    return _make_doc(["Only sentence."])


@pytest.fixture
def two_sentence_doc() -> Document:
    return _make_doc(["First sentence here.", "Second sentence here."])


# ── BruteForceSegmenter ───────────────────────────────────────────────────────


class TestBruteForce:
    def test_returns_correct_type(self, mixed_doc: Document) -> None:
        result = BruteForceSegmenter().segment(mixed_doc, max_segments=3)
        assert isinstance(result, SegmentationResult)

    def test_name_is_brute_force(self) -> None:
        assert BruteForceSegmenter().name == "brute_force"

    def test_first_boundary_is_zero(self, mixed_doc: Document) -> None:
        result = BruteForceSegmenter().segment(mixed_doc, max_segments=3)
        assert result.boundaries[0] == 0

    def test_respects_max_segments(self, mixed_doc: Document) -> None:
        for k in (2, 3, 4):
            result = BruteForceSegmenter().segment(mixed_doc, max_segments=k)
            assert result.n_segments <= k

    def test_single_sentence_returns_one_boundary(self, tiny_doc: Document) -> None:
        result = BruteForceSegmenter().segment(tiny_doc)
        assert result.boundaries == [0]

    def test_detects_first_topic_boundary(self, mixed_doc: Document) -> None:
        # Sports / Tech / Cooking split — the BF algorithm should at least
        # detect the sports-vs-rest boundary (sentences 0-2 vs 3+).
        result = BruteForceSegmenter().segment(mixed_doc, max_segments=3)
        assert result.n_segments == 3
        assert result.boundaries[1] == 3

    def test_matches_dp_on_small_doc(self, mixed_doc: Document) -> None:
        from src.algorithms.dynamic_programming import DPSegmenter

        bf = BruteForceSegmenter().segment(mixed_doc, max_segments=3)
        dp = DPSegmenter().segment(mixed_doc, max_segments=3)
        assert bf.boundaries == dp.boundaries

    def test_rejects_oversized_document(self) -> None:
        big = _make_doc([f"Sentence {i}." for i in range(_MAX_SENTENCES + 1)])
        with pytest.raises(ValueError, match="at most"):
            BruteForceSegmenter().segment(big)

    def test_runtime_is_positive(self, mixed_doc: Document) -> None:
        result = BruteForceSegmenter().segment(mixed_doc, max_segments=3)
        assert result.runtime_seconds >= 0.0


# ── GreedySegmenter ───────────────────────────────────────────────────────────


class TestGreedy:
    def test_returns_correct_type(self, mixed_doc: Document) -> None:
        result = GreedySegmenter().segment(mixed_doc, max_segments=3)
        assert isinstance(result, SegmentationResult)

    def test_name_is_greedy(self) -> None:
        assert GreedySegmenter().name == "greedy"

    def test_first_boundary_is_zero(self, mixed_doc: Document) -> None:
        result = GreedySegmenter().segment(mixed_doc, max_segments=3)
        assert result.boundaries[0] == 0

    def test_respects_max_segments(self, mixed_doc: Document) -> None:
        for k in (2, 3, 4):
            result = GreedySegmenter().segment(mixed_doc, max_segments=k)
            assert result.n_segments <= k

    def test_single_sentence_returns_one_boundary(self, tiny_doc: Document) -> None:
        result = GreedySegmenter().segment(tiny_doc)
        assert result.boundaries == [0]

    def test_boundaries_are_sorted_and_unique(self, mixed_doc: Document) -> None:
        result = GreedySegmenter().segment(mixed_doc, max_segments=3)
        assert result.boundaries == sorted(set(result.boundaries))

    def test_boundaries_within_document(self, mixed_doc: Document) -> None:
        n = mixed_doc.n_sentences
        result = GreedySegmenter().segment(mixed_doc, max_segments=3)
        assert all(0 <= b < n for b in result.boundaries)

    def test_runtime_is_positive(self, mixed_doc: Document) -> None:
        result = GreedySegmenter().segment(mixed_doc, max_segments=3)
        assert result.runtime_seconds >= 0.0

    def test_window_size_param_accepted(self, mixed_doc: Document) -> None:
        result = GreedySegmenter(window_size=3).segment(mixed_doc, max_segments=3)
        assert isinstance(result, SegmentationResult)


# ── SASegmenter ───────────────────────────────────────────────────────────────


class TestSA:
    def test_returns_correct_type(self, mixed_doc: Document) -> None:
        result = SASegmenter(random_seed=42).segment(mixed_doc, max_segments=3)
        assert isinstance(result, SegmentationResult)

    def test_name_is_simulated_annealing(self) -> None:
        assert SASegmenter().name == "simulated_annealing"

    def test_first_boundary_is_zero(self, mixed_doc: Document) -> None:
        result = SASegmenter(random_seed=42).segment(mixed_doc, max_segments=3)
        assert result.boundaries[0] == 0

    def test_respects_max_segments(self, mixed_doc: Document) -> None:
        for k in (2, 3, 4):
            result = SASegmenter(random_seed=0).segment(mixed_doc, max_segments=k)
            assert result.n_segments <= k

    def test_deterministic_with_seed(self, mixed_doc: Document) -> None:
        r1 = SASegmenter(random_seed=7).segment(mixed_doc, max_segments=3)
        r2 = SASegmenter(random_seed=7).segment(mixed_doc, max_segments=3)
        assert r1.boundaries == r2.boundaries

    def test_different_seeds_may_differ(self, mixed_doc: Document) -> None:
        # This is probabilistic — just verify both runs are valid
        r1 = SASegmenter(random_seed=1).segment(mixed_doc, max_segments=3)
        r2 = SASegmenter(random_seed=99).segment(mixed_doc, max_segments=3)
        assert isinstance(r1, SegmentationResult)
        assert isinstance(r2, SegmentationResult)

    def test_single_sentence_returns_one_boundary(self, tiny_doc: Document) -> None:
        result = SASegmenter(random_seed=0).segment(tiny_doc)
        assert result.boundaries == [0]

    def test_boundaries_within_document(self, mixed_doc: Document) -> None:
        n = mixed_doc.n_sentences
        result = SASegmenter(random_seed=42).segment(mixed_doc, max_segments=3)
        assert all(0 <= b < n for b in result.boundaries)

    def test_runtime_is_positive(self, mixed_doc: Document) -> None:
        result = SASegmenter(random_seed=0, n_iterations=100).segment(
            mixed_doc, max_segments=3
        )
        assert result.runtime_seconds >= 0.0
