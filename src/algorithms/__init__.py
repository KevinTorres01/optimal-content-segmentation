from src.core.interfaces import BaseSegmenter
from src.algorithms.brute_force import BruteForceSegmenter
from src.algorithms.dynamic_programming import DPSegmenter
from src.algorithms.greedy import GreedySegmenter
from src.algorithms.simulated_annealing import SASegmenter

ALGORITHM_REGISTRY: dict[str, type[BaseSegmenter]] = {
    "brute_force": BruteForceSegmenter,
    "dynamic_programming": DPSegmenter,
    "greedy": GreedySegmenter,
    "simulated_annealing": SASegmenter,
}

__all__ = [
    "BaseSegmenter",
    "ALGORITHM_REGISTRY",
    "BruteForceSegmenter",
    "DPSegmenter",
    "GreedySegmenter",
    "SASegmenter",
]
