from __future__ import annotations

import json
import os

from mistralai.client import Mistral

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment
from src.llm.prompts import COHESION_SCORING_PROMPT, parse_cohesion_response
from src.llm.rate_limit import call_with_retry, throttle_from_env

_DEFAULT_MODEL = "mistral-large-latest"


class MistralEvaluator(BaseLLMEvaluator):
    """LLM evaluator using Mistral AI models (French provider, accessible from Cuba).

    Obtain API key at: https://console.mistral.ai/
    Set env var: MISTRAL_API_KEY
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """Initialize the Mistral evaluator.

        Args:
            model: Mistral model ID (default: mistral-large-latest).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
        """
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Obtain your key at https://console.mistral.ai/"
            )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = Mistral(api_key=api_key)
        self._throttle = throttle_from_env()

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def model_name(self) -> str:
        return self._model

    def score_segment(self, segment: Segment) -> CohesionScore:
        """Score a single segment."""
        return self.score_segmentation([segment])[0]

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        """Score all segments sequentially (Mistral does not support batch requests)."""
        results: list[CohesionScore] = []
        for segment in segments:
            results.append(self._score_one(segment))
        return results

    def _score_one(self, segment: Segment) -> CohesionScore:
        prompt = COHESION_SCORING_PROMPT.format(segment_text=segment.text)
        try:
            self._throttle.wait()
            response = call_with_retry(
                lambda: self._client.chat.complete(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                )
            )
            raw = response.choices[0].message.content or ""
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
