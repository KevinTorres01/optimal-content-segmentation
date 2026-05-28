import time

import pytest

from src.core.models import CohesionScore, LLMConfig, Segment
from src.llm import check as check_module
from src.llm.check import check_provider
from src.llm.factory import get_llm_provider
from src.llm.fallback_provider import FallbackEvaluator
from src.llm.ollama_provider import OllamaEvaluator
from src.llm.rate_limit import call_with_retry, is_rate_limit_error

# ── Test doubles ─────────────────────────────────────────────────────────────


class _FakeEvaluator:
    """Minimal evaluator double that returns a fixed score."""

    def __init__(self, provider: str = "fake", score: int = 5) -> None:
        self.provider_name = provider
        self.model_name = "fake-model"
        self._score = score

    def score_segment(self, segment: Segment) -> CohesionScore:
        return CohesionScore(
            segment_id=segment.segment_id,
            score=self._score,
            rationale="fake",
            provider=self.provider_name,
            model=self.model_name,
        )

    def score_segmentation(self, segments):
        return [self.score_segment(s) for s in segments]


class _FailingEvaluator(_FakeEvaluator):
    def score_segment(self, segment: Segment) -> CohesionScore:
        raise ConnectionError("simulated network failure")


class _SlowEvaluator(_FakeEvaluator):
    def score_segment(self, segment: Segment) -> CohesionScore:
        time.sleep(2)
        return super().score_segment(segment)


@pytest.fixture
def segment() -> Segment:
    return Segment(segment_id="d_seg0", sentences=["una frase.", "otra frase."])


# ── Factory ──────────────────────────────────────────────────────────────────


def test_provider_none_returns_none() -> None:
    assert get_llm_provider(LLMConfig(provider="none")) is None


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider(LLMConfig(provider="gpt-4"))


def test_ollama_returned_directly_not_wrapped() -> None:
    evaluator = get_llm_provider(LLMConfig(provider="ollama"))
    assert isinstance(evaluator, OllamaEvaluator)


def test_online_provider_wrapped_in_fallback_by_default(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-key")
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    evaluator = get_llm_provider(LLMConfig(provider="mistral"))
    assert isinstance(evaluator, FallbackEvaluator)
    assert evaluator.provider_name == "fallback(mistral→ollama)"


def test_online_provider_bare_when_fallback_disabled(monkeypatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "dummy-key")
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    evaluator = get_llm_provider(LLMConfig(provider="mistral"))
    assert not isinstance(evaluator, FallbackEvaluator)
    assert evaluator.provider_name == "mistral"


def test_missing_api_key_raises_not_silently_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        get_llm_provider(LLMConfig(provider="mistral"))


# ── Fallback behaviour ───────────────────────────────────────────────────────


def test_fallback_uses_primary_on_success(segment: Segment) -> None:
    fb = FallbackEvaluator(
        primary=_FakeEvaluator("mistral", score=5),
        fallback=_FakeEvaluator("ollama", score=1),
        timeout=5,
    )
    result = fb.score_segment(segment)
    assert result.score == 5
    assert result.used_fallback is False


def test_fallback_switches_to_local_on_error(segment: Segment) -> None:
    fb = FallbackEvaluator(
        primary=_FailingEvaluator("mistral"),
        fallback=_FakeEvaluator("ollama", score=2),
        timeout=5,
    )
    result = fb.score_segment(segment)
    assert result.score == 2
    assert result.provider == "ollama"
    assert result.used_fallback is True


def test_fallback_switches_on_timeout(segment: Segment) -> None:
    fb = FallbackEvaluator(
        primary=_SlowEvaluator("mistral"),
        fallback=_FakeEvaluator("ollama", score=3),
        timeout=1,
    )
    result = fb.score_segment(segment)
    assert result.provider == "ollama"
    assert result.used_fallback is True


# ── Connectivity check ───────────────────────────────────────────────────────


def test_check_unknown_provider_returns_false() -> None:
    assert check_provider("gpt-4") is False


def test_check_success(monkeypatch) -> None:
    monkeypatch.setitem(check_module._PROVIDERS, "fake", lambda: _FakeEvaluator())
    assert check_provider("fake") is True


def test_check_failure_on_exception(monkeypatch) -> None:
    monkeypatch.setitem(check_module._PROVIDERS, "fake", lambda: _FailingEvaluator())
    assert check_provider("fake") is False


# ── Rate-limit retry ─────────────────────────────────────────────────────────


def test_is_rate_limit_error_detects_429() -> None:
    assert is_rate_limit_error(Exception("Status 429. Rate limit exceeded"))
    assert is_rate_limit_error(Exception("rate_limited"))
    assert not is_rate_limit_error(Exception("invalid api key"))


def test_call_with_retry_succeeds_after_rate_limit() -> None:
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("Status 429: rate limit")
        return "ok"

    result = call_with_retry(flaky, max_retries=3, base_delay=0)
    assert result == "ok"
    assert calls["n"] == 3


def test_call_with_retry_reraises_non_rate_limit_immediately() -> None:
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("bad key")

    with pytest.raises(ValueError, match="bad key"):
        call_with_retry(boom, max_retries=3, base_delay=0)
    assert calls["n"] == 1  # no retries for non-rate-limit errors


def test_call_with_retry_gives_up_after_max_retries() -> None:
    calls = {"n": 0}

    def always_limited():
        calls["n"] += 1
        raise Exception("429 too many requests")

    with pytest.raises(Exception, match="429"):
        call_with_retry(always_limited, max_retries=2, base_delay=0)
    assert calls["n"] == 3  # 1 initial + 2 retries


# ── Fallback graceful degradation ────────────────────────────────────────────


def test_fallback_degrades_when_both_fail(segment: Segment) -> None:
    fb = FallbackEvaluator(
        primary=_FailingEvaluator("mistral"),
        fallback=_FailingEvaluator("ollama"),
        timeout=5,
    )
    result = fb.score_segment(segment)
    assert result.score == 3
    assert result.used_fallback is True
    assert "failed" in result.rationale.lower()
