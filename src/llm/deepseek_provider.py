from __future__ import annotations

import json
import os

from openai import OpenAI

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment
from src.llm.prompts import COHESION_SCORING_PROMPT, parse_cohesion_response
from src.llm.rate_limit import call_with_retry, throttle_from_env

_DEFAULT_MODEL = "deepseek-chat"
_BASE_URL = "https://api.deepseek.com"


class DeepSeekEvaluator(BaseLLMEvaluator):
    """LLM evaluator using DeepSeek models (Chinese provider, accessible from Cuba).

    DeepSeek exposes an OpenAI-compatible API.
    Obtain API key at: https://platform.deepseek.com/
    Set env var: DEEPSEEK_API_KEY
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """Initialize the DeepSeek evaluator.

        Args:
            model: DeepSeek model ID (default: deepseek-chat → DeepSeek-V3).
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens in the response.
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable is not set. "
                "Obtain your key at https://platform.deepseek.com/"
            )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key, base_url=_BASE_URL)
        self._throttle = throttle_from_env()

    @property
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def model_name(self) -> str:
        return self._model

    def score_segment(self, segment: Segment) -> CohesionScore:
        """Score a single segment."""
        return self.score_segmentation([segment])[0]

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        """Score all segments sequentially."""
        return [self._score_one(segment) for segment in segments]

    def _score_one(self, segment: Segment) -> CohesionScore:
        prompt = COHESION_SCORING_PROMPT.format(segment_text=segment.text)
        try:
            self._throttle.wait()
            response = call_with_retry(
                lambda: self._client.chat.completions.create(
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
