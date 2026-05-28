import json

import pandas as pd
import pytest
import yaml

from src.dataset.generator import DatasetGenerator
from src.dataset.schemas import DatasetConfig
from src.experiments.runner import run_experiment


@pytest.fixture(scope="module")
def smoke_env(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Create a complete smoke-test environment in a temporary directory."""
    base = tmp_path_factory.mktemp("smoke")

    # Generate a small dataset
    dataset_dir = base / "data" / "small"
    cfg = DatasetConfig(
        dataset_name="small",
        n_documents=5,
        segments_per_doc={"min": 2, "max": 3},
        sentences_per_segment={"min": 3, "max": 5},
        topic_source="synthetic_templates",
        overlap_level="low",
        random_seed=42,
        language="es",
    )
    DatasetGenerator(cfg).generate(dataset_dir)

    # Write a smoke_test config pointing to the temp dataset
    results_dir = base / "results" / "smoke_test"
    config = {
        "experiment_id": "smoke_test",
        "description": "Integration smoke test",
        "dataset": {"path": str(dataset_dir), "split": "all"},
        "algorithms": [{"name": "dynamic_programming", "params": {"max_segments": 3}}],
        "llm_evaluator": {"provider": "none", "temperature": 0.0, "max_tokens": 512},
        "evaluation": {
            "metrics": ["windowdiff", "pk", "f1_boundary"],
            "random_seed": 42,
        },
        "output": {"path": str(results_dir), "save_raw": True},
    }
    config_path = base / "smoke_test.yaml"
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    # Run the experiment
    run_experiment(config_path)

    return {"results_dir": results_dir, "n_docs": 5}


def test_output_files_exist(smoke_env: dict) -> None:
    results_dir = smoke_env["results_dir"]
    assert (results_dir / "results.json").exists()
    assert (results_dir / "summary.csv").exists()
    assert (results_dir / "run_metadata.json").exists()


def test_results_json_has_correct_document_count(smoke_env: dict) -> None:
    results_file = smoke_env["results_dir"] / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8"))
    assert len(results) == smoke_env["n_docs"]


def test_results_json_has_required_fields(smoke_env: dict) -> None:
    results_file = smoke_env["results_dir"] / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8"))
    required = {
        "doc_id",
        "algorithm",
        "pk",
        "windowdiff",
        "f1_boundary",
        "runtime_seconds",
    }
    for entry in results:
        assert required.issubset(
            entry.keys()
        ), f"Missing fields in result entry: {required - entry.keys()}"


def test_metrics_are_finite_and_in_range(smoke_env: dict) -> None:
    results_file = smoke_env["results_dir"] / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8"))
    for entry in results:
        assert 0.0 <= entry["pk"] <= 1.0, f"pk={entry['pk']} out of [0,1]"
        assert (
            0.0 <= entry["windowdiff"] <= 1.0
        ), f"wd={entry['windowdiff']} out of [0,1]"
        assert (
            0.0 <= entry["f1_boundary"] <= 1.0
        ), f"f1={entry['f1_boundary']} out of [0,1]"
        assert entry["runtime_seconds"] >= 0.0


def test_summary_csv_has_algorithm_column(smoke_env: dict) -> None:
    df = pd.read_csv(smoke_env["results_dir"] / "summary.csv")
    assert "algorithm" in df.columns
    assert "dynamic_programming" in df["algorithm"].values


def test_run_metadata_has_seed(smoke_env: dict) -> None:
    meta = json.loads(
        (smoke_env["results_dir"] / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert "random_seed" in meta
    assert meta["random_seed"] == 42
