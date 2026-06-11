import json

import pytest
import yaml

from src.core.models import CohesionScore, Segment
from src.dataset.generator import DatasetGenerator
from src.dataset.schemas import DatasetConfig
from src.experiments import runner as runner_module
from src.experiments.runner import run_experiment
from src.llm.fallback_provider import NEUTRAL_RATIONALE
from tests.mocks import MockLLMEvaluator


@pytest.fixture
def llm_env(tmp_path, monkeypatch):
    """Run a tiny experiment with the LLM evaluator mocked (no API calls)."""
    dataset_dir = tmp_path / "data" / "small"
    cfg = DatasetConfig(
        dataset_name="small",
        n_documents=3,
        segments_per_doc={"min": 2, "max": 2},
        sentences_per_segment={"min": 3, "max": 4},
        topic_source="synthetic_templates",
        overlap_level="low",
        random_seed=7,
        language="es",
    )
    DatasetGenerator(cfg).generate(dataset_dir)

    results_dir = tmp_path / "results" / "exp_llm"
    config = {
        "experiment_id": "exp_llm",
        "description": "LLM integration test",
        "dataset": {"path": str(dataset_dir)},
        "algorithms": [{"name": "dynamic_programming", "params": {"max_segments": 2}}],
        "llm_evaluator": {"provider": "mistral", "model": "mistral-large-latest"},
        "evaluation": {"metrics": ["pk", "llm_score"], "random_seed": 7},
        "output": {"path": str(results_dir), "save_raw": True},
    }
    config_path = tmp_path / "exp_llm.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    # Replace the real factory so no network/API key is needed.
    monkeypatch.setattr(
        runner_module, "get_llm_provider", lambda llm_config: MockLLMEvaluator()
    )
    run_experiment(config_path)
    return results_dir


def test_llm_score_is_populated(llm_env) -> None:
    results = json.loads((llm_env / "results.json").read_text(encoding="utf-8"))
    assert results, "expected at least one result entry"
    for entry in results:
        assert entry["llm_score"] is not None
        assert entry["llm_score"] == 3.0  # MockLLMEvaluator always returns 3
        assert entry["llm_used_fallback"] == 0.0
        # Mock never goes through the neutral path.
        assert entry["llm_n_segments"] == entry["n_pred_segments"]
        assert entry["llm_n_neutral"] == 0


def test_summary_includes_llm_score(llm_env) -> None:
    import pandas as pd

    df = pd.read_csv(llm_env / "summary.csv")
    assert "llm_score" in df.columns


class _NeutralEvaluator:
    """Evaluator that always returns a neutral fallback score (both providers failed)."""

    provider_name = "fallback(groq→mistral)"
    model_name = "mock"

    def score_segment(self, segment: Segment) -> CohesionScore:
        return CohesionScore(
            segment_id=segment.segment_id,
            score=3,
            rationale=NEUTRAL_RATIONALE,
            provider=self.provider_name,
            model=self.model_name,
            used_fallback=True,
        )

    def score_segmentation(self, segments: list[Segment]) -> list[CohesionScore]:
        return [self.score_segment(s) for s in segments]


def test_runner_counts_neutral_fallback_scores(tmp_path, monkeypatch) -> None:
    """When every score falls to the neutral path, results.json must reflect it."""
    dataset_dir = tmp_path / "data" / "small"
    cfg = DatasetConfig(
        dataset_name="small",
        n_documents=2,
        segments_per_doc={"min": 2, "max": 2},
        sentences_per_segment={"min": 3, "max": 4},
        topic_source="synthetic_templates",
        overlap_level="low",
        random_seed=7,
        language="es",
    )
    DatasetGenerator(cfg).generate(dataset_dir)

    results_dir = tmp_path / "results" / "exp_neutral"
    config = {
        "experiment_id": "exp_neutral",
        "description": "Neutral fallback audit test",
        "dataset": {"path": str(dataset_dir)},
        "algorithms": [{"name": "dynamic_programming", "params": {"max_segments": 2}}],
        "llm_evaluator": {"provider": "mistral", "model": "mistral-large-latest"},
        "evaluation": {"metrics": ["llm_score"], "random_seed": 7},
        "output": {"path": str(results_dir), "save_raw": True},
    }
    config_path = tmp_path / "exp_neutral.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    monkeypatch.setattr(
        runner_module, "get_llm_provider", lambda llm_config: _NeutralEvaluator()
    )
    run_experiment(config_path)

    results = json.loads((results_dir / "results.json").read_text(encoding="utf-8"))
    assert results
    for entry in results:
        assert entry["llm_used_fallback"] == 1.0
        assert entry["llm_n_neutral"] == entry["llm_n_segments"]
        assert entry["llm_n_neutral"] == entry["n_pred_segments"]
