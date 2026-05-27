from src.core.interfaces import BaseSegmenter
from src.algorithms.dynamic_programming import DPSegmenter

ALGORITHM_REGISTRY: dict[str, type[BaseSegmenter]] = {
    "dynamic_programming": DPSegmenter,
}

__all__ = ["BaseSegmenter", "ALGORITHM_REGISTRY"]
