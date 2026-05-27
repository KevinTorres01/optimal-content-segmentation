import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.dataset.generator import DatasetGenerator
from src.dataset.schemas import DatasetConfig


@pytest.fixture
def base_config() -> dict:
    return {
        "dataset_name": "test",
        "n_documents": 5,
        "segments_per_doc": {"min": 2, "max": 4},
        "sentences_per_segment": {"min": 3, "max": 5},
        "topic_source": "synthetic_templates",
        "overlap_level": "low",
        "random_seed": 42,
        "language": "es",
    }


@pytest.fixture
def config(base_config: dict) -> DatasetConfig:
    return DatasetConfig.model_validate(base_config)


# ── Schema validation ─────────────────────────────────────────────────────────

def test_missing_random_seed_raises() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(
            dataset_name="test",
            n_documents=5,
            segments_per_doc={"min": 2, "max": 4},
            sentences_per_segment={"min": 3, "max": 5},
            # random_seed intentionally missing
        )


def test_invalid_overlap_level_raises(base_config: dict) -> None:
    base_config["overlap_level"] = "extreme"
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(base_config)


def test_invalid_language_raises(base_config: dict) -> None:
    base_config["language"] = "fr"
    with pytest.raises(ValidationError):
        DatasetConfig.model_validate(base_config)


# ── Generation correctness ────────────────────────────────────────────────────

def test_generates_correct_number_of_documents(
    config: DatasetConfig, tmp_path: Path
) -> None:
    gen = DatasetGenerator(config)
    gen.generate(tmp_path)
    docs = list((tmp_path / "documents").glob("*.txt"))
    assert len(docs) == config.n_documents


def test_boundary_files_match_document_files(
    config: DatasetConfig, tmp_path: Path
) -> None:
    gen = DatasetGenerator(config)
    gen.generate(tmp_path)
    doc_ids = {p.stem for p in (tmp_path / "documents").glob("*.txt")}
    boundary_ids = {p.stem for p in (tmp_path / "boundaries").glob("*.json")}
    assert doc_ids == boundary_ids


def test_boundaries_are_valid(config: DatasetConfig, tmp_path: Path) -> None:
    gen = DatasetGenerator(config)
    gen.generate(tmp_path)
    for boundary_file in (tmp_path / "boundaries").glob("*.json"):
        data = json.loads(boundary_file.read_text(encoding="utf-8"))
        assert data["boundaries"][0] == 0, "First boundary must be 0"
        assert data["boundaries"] == sorted(data["boundaries"]), "Must be sorted"
        doc_file = tmp_path / "documents" / f"{boundary_file.stem}.txt"
        n_sentences = len(doc_file.read_text(encoding="utf-8").splitlines())
        for b in data["boundaries"]:
            assert 0 <= b < n_sentences, f"Boundary {b} out of range [0, {n_sentences})"


def test_metadata_json_created(config: DatasetConfig, tmp_path: Path) -> None:
    gen = DatasetGenerator(config)
    gen.generate(tmp_path)
    meta_file = tmp_path / "metadata.json"
    assert meta_file.exists()
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["n_documents"] == config.n_documents
    assert meta["generation_seed"] == config.random_seed


def test_deterministic_with_seed(config: DatasetConfig, tmp_path: Path) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    DatasetGenerator(config).generate(out1)
    DatasetGenerator(config).generate(out2)

    docs1 = sorted((out1 / "documents").glob("*.txt"))
    docs2 = sorted((out2 / "documents").glob("*.txt"))
    assert len(docs1) == len(docs2)
    for f1, f2 in zip(docs1, docs2):
        assert f1.read_text() == f2.read_text(), (
            f"Non-deterministic output between runs for {f1.name}"
        )


def test_english_language_generates_documents(
    base_config: dict, tmp_path: Path
) -> None:
    base_config["language"] = "en"
    config = DatasetConfig.model_validate(base_config)
    gen = DatasetGenerator(config)
    gen.generate(tmp_path)
    docs = list((tmp_path / "documents").glob("*.txt"))
    assert len(docs) == config.n_documents
