import json

import pytest
import yaml

from src.dataset.generator import DatasetGenerator
from src.dataset.schemas import DatasetConfig
from src.experiments import runner as runner_module
from src.experiments.runner import run_experiment
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
        "dataset": {"path": str(dataset_dir), "split": "all"},
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


def test_summary_includes_llm_score(llm_env) -> None:
    import pandas as pd

    df = pd.read_csv(llm_env / "summary.csv")
    assert "llm_score" in df.columns
