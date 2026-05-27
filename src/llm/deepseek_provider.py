from __future__ import annotations

import json
import os

from openai import OpenAI

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import CohesionScore, Segment
from src.llm.prompts import COHESION_SCORING_PROMPT

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
        max_tokens: int = 256,
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
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw.strip())
            return CohesionScore(
                segment_id=segment.segment_id,
                score=int(data["score"]),
                rationale=str(data["rationale"]),
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
