from __future__ import annotations

import json
import os

import ollama as ollama_sdk

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment
from src.llm.prompts import COHESION_SCORING_PROMPT, parse_cohesion_response

_DEFAULT_MODEL = "llama3.2:3b"


class OllamaEvaluator(BaseLLMEvaluator):
    """LLM evaluator using a local Ollama model (always available, no internet needed).

    Setup:
        1. Install Ollama: https://ollama.ai/
        2. Pull the model: ollama pull llama3.2:3b
        3. Start the server: ollama serve

    Recommended model for CPU with 16GB RAM: llama3.2:3b (~3GB, ~8 tok/s)

    Set env vars (optional, these are the defaults):
        OLLAMA_BASE_URL=http://localhost:11434
        OLLAMA_MODEL=llama3.2:3b
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the Ollama evaluator.

        Args:
            model: Ollama model name (overrides OLLAMA_MODEL env var).
            base_url: Ollama server URL (overrides OLLAMA_BASE_URL env var).
        """
        self._model = model or os.environ.get("OLLAMA_MODEL") or _DEFAULT_MODEL
        self._base_url = (
            base_url or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_local(self) -> bool:
        return True

    def score_segment(self, segment: Segment) -> CohesionScore:
        """Score a single segment."""
        return self.score_segmentation([segment])[0]

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        """Score all segments sequentially using the local Ollama server."""
        return [self._score_one(segment) for segment in segments]

    def _score_one(self, segment: Segment) -> CohesionScore:
        prompt = COHESION_SCORING_PROMPT.format(segment_text=segment.text)
        try:
            client = ollama_sdk.Client(host=self._base_url)
            response = client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0},
                format="json",
            )
            raw = response.message.content or ""
            data = parse_cohesion_response(raw)
            return CohesionScore(
                segment_id=segment.segment_id,
                score=data["score"],
                rationale=data["rationale"],
                provider=self.provider_name,
                model=self.model_name,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return CohesionScore(
                segment_id=segment.segment_id,
                score=3,
                rationale="Error parsing LLM response",
                provider=self.provider_name,
                model=self.model_name,
            )
