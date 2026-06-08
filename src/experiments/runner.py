from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.algorithms import ALGORITHM_REGISTRY
from src.core.config import load_config
from src.core.models import Document
from src.evaluation.metrics import (
    compute_f1_boundary,
    compute_pk,
    compute_window_diff,
)
from src.llm import get_llm_provider

console = Console()


def _load_dataset(dataset_path: Path) -> list[tuple[Document, list[int]]]:
    """Load all documents and their ground-truth boundaries from dataset_path.

    Args:
        dataset_path: Directory with documents/ and boundaries/ subdirs.

    Returns:
        List of (Document, boundaries) pairs.
    """
    docs_dir = dataset_path / "documents"
    bounds_dir = dataset_path / "boundaries"

    if not docs_dir.exists():
        raise FileNotFoundError(f"Dataset documents directory not found: {docs_dir}")
    if not bounds_dir.exists():
        raise FileNotFoundError(f"Dataset boundaries directory not found: {bounds_dir}")

    dataset: list[tuple[Document, list[int]]] = []
    for doc_file in sorted(docs_dir.glob("*.txt")):
        doc_id = doc_file.stem
        boundary_file = bounds_dir / f"{doc_id}.json"
        if not boundary_file.exists():
            console.print(f"[yellow]Warning: no boundary file for {doc_id}, skipping")
            continue

        sentences = doc_file.read_text(encoding="utf-8").splitlines()
        boundary_data = json.loads(boundary_file.read_text(encoding="utf-8"))
        boundaries: list[int] = boundary_data["boundaries"]

        dataset.append((Document(doc_id=doc_id, sentences=sentences), boundaries))

    return dataset


def run_experiment(config_path: Path) -> None:
    """Run a full experiment: load config, dataset, algorithms, compute metrics.

    Results are saved to the output directory specified in the config:
        results.json     — per-document metrics
        summary.csv      — aggregated metrics per algorithm
        run_metadata.json — config snapshot, timestamps, library versions

    Args:
        config_path: Path to experiment YAML config file.
    """
    config = load_config(config_path)
    console.rule(f"[bold blue]Experiment: {config.experiment_id}")
    console.print(f"[dim]{config.description}")

    dataset_path = Path(config.dataset.path)
    output_path = Path(config.output.path)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\nLoading dataset from [cyan]{dataset_path}[/]...")
    dataset = _load_dataset(dataset_path)
    console.print(f"  Loaded {len(dataset)} documents")

    try:
        llm_evaluator = get_llm_provider(config.llm_evaluator)
    except ValueError as exc:
        console.print(f"[red]{exc}")
        sys.exit(1)
    if llm_evaluator is not None:
        console.print(
            f"  LLM evaluator: [cyan]{llm_evaluator.provider_name}[/] / "
            f"[cyan]{llm_evaluator.model_name}[/]"
        )

    all_results: list[dict] = []
    start_time = datetime.now(timezone.utc)

    for algo_config in config.algorithms:
        algo_cls = ALGORITHM_REGISTRY.get(algo_config.name)
        if algo_cls is None:
            console.print(
                f"[red]Unknown algorithm: {algo_config.name}. "
                f"Available: {list(ALGORITHM_REGISTRY.keys())}"
            )
            continue

        params = dict(algo_config.params)
        max_segments = params.pop("max_segments", None)
        # Default any unset random_seed to the experiment-wide seed so SA (and any
        # future stochastic algorithm) stays reproducible even when the YAML omits it.
        constructor_params = inspect.signature(algo_cls).parameters
        if "random_seed" in constructor_params and "random_seed" not in params:
            params["random_seed"] = config.evaluation.random_seed
        try:
            algorithm = algo_cls(**params)
        except TypeError as exc:
            console.print(
                f"[red]Invalid params for {algo_config.name}: {exc}[/]"
            )
            continue
        console.print(f"\nRunning algorithm: [bold]{algo_config.name}[/]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Segmenting {len(dataset)} documents...", total=len(dataset)
            )

            for document, gt_boundaries in dataset:
                result = algorithm.segment(document, max_segments=max_segments)

                n = document.n_sentences
                pk = compute_pk(gt_boundaries, result.boundaries, n)
                wd = compute_window_diff(gt_boundaries, result.boundaries, n)
                _, _, f1 = compute_f1_boundary(gt_boundaries, result.boundaries)

                doc_result: dict = {
                    "experiment_id": config.experiment_id,
                    "algorithm": algo_config.name,
                    "doc_id": document.doc_id,
                    "n_sentences": n,
                    "n_ref_segments": len(gt_boundaries),
                    "n_pred_segments": result.n_segments,
                    "boundaries_ref": gt_boundaries,
                    "boundaries_pred": result.boundaries,
                    "pk": round(pk, 4),
                    "windowdiff": round(wd, 4),
                    "f1_boundary": round(f1, 4),
                    "runtime_seconds": round(result.runtime_seconds, 4),
                    "llm_score": None,
                    "llm_used_fallback": None,
                }

                if llm_evaluator is not None:
                    segments = result.to_segments(document)
                    scores = llm_evaluator.score_segmentation(segments)
                    avg_llm = sum(s.score for s in scores) / len(scores)
                    fallback_rate = sum(s.used_fallback for s in scores) / len(scores)
                    doc_result["llm_score"] = round(avg_llm, 2)
                    doc_result["llm_used_fallback"] = round(fallback_rate, 2)

                all_results.append(doc_result)
                progress.advance(task)

    # ── Save raw results ───────────────────────────────────────────────────────
    if config.output.save_raw:
        raw_path = output_path / "results.json"
        raw_path.write_text(
            json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ── Aggregate and save summary ─────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    numeric_cols = ["pk", "windowdiff", "f1_boundary", "runtime_seconds"]
    if "llm_score" in df.columns and df["llm_score"].notna().any():
        numeric_cols.append("llm_score")

    summary = df.groupby("algorithm")[numeric_cols].mean().round(4).reset_index()
    summary_path = output_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    # ── Save run metadata ──────────────────────────────────────────────────────
    metadata = {
        "experiment_id": config.experiment_id,
        "started_at": start_time.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.evaluation.random_seed,
        "n_documents": len(dataset),
        "algorithms": [a.name for a in config.algorithms],
        "llm_provider": config.llm_evaluator.provider,
        "config_path": str(config_path),
    }
    (output_path / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # ── Print summary table ────────────────────────────────────────────────────
    table = Table(title=f"Results: {config.experiment_id}", show_lines=True)
    for col in summary.columns:
        table.add_column(col, style="cyan" if col == "algorithm" else "")
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row])

    console.print()
    console.print(table)
    console.print(f"\n[green]✓ Results saved to [bold]{output_path}[/]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a segmentation experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()
    run_experiment(Path(args.config))
