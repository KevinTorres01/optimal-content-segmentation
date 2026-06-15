"""Comparación de backends de cohesión: TF-IDF léxico vs SBERT semántico.

Mide el impacto de reemplazar la representación de oraciones (TF-IDF disperso)
por embeddings densos multilingües (Sentence-BERT) manteniendo intactos los
algoritmos de segmentación. El backend se selecciona por construcción vía el
parámetro ``cohesion_backend`` de cada segmentador.

Para cada (dataset × algoritmo × backend) se promedian F1-Boundary, Pk,
WindowDiff y el tiempo de ejecución sobre todos los documentos, y se reporta
el delta SBERT − TF-IDF.

Salida:
  results/exp_cohesion_backend/
    summary.csv   — una fila por (dataset, algoritmo, backend)
    deltas.csv    — delta SBERT − TF-IDF por (dataset, algoritmo)
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from time import time

from rich.console import Console
from rich.table import Table

from src.algorithms.dynamic_programming import DPSegmenter
from src.algorithms.greedy import GreedySegmenter
from src.algorithms.simulated_annealing import SASegmenter
from src.core.models import Document
from src.evaluation.metrics import (
    compute_f1_boundary,
    compute_pk,
    compute_window_diff,
)

# Datasets a evaluar: Wikipedia (texto natural, donde TF-IDF se degrada) y small
# (sintético controlado, para verificar que la adición no perjudica el caso ideal).
DATASETS = {
    "wikipedia": Path("data/wikipedia/"),
    "small": Path("data/small/"),
}
BACKENDS = ("tfidf", "sbert")
MAX_SEGMENTS = 5
OUTPUT_PATH = Path("results/exp_cohesion_backend/")

console = Console()


def _make_segmenters(backend: str) -> dict[str, object]:
    """Instancia los tres algoritmos escalables con el backend dado.

    SA usa la mejor configuración del análisis de sensibilidad (Exp. 5):
    T0=0.5, α=0.995, n_iter=2000, semilla fija para reproducibilidad.
    """
    return {
        "dynamic_programming": DPSegmenter(cohesion_backend=backend),
        "greedy": GreedySegmenter(window_size=2, cohesion_backend=backend),
        "simulated_annealing": SASegmenter(
            n_iterations=2000,
            initial_temp=0.5,
            cooling_rate=0.995,
            random_seed=42,
            cohesion_backend=backend,
        ),
    }


def load_dataset(path: Path) -> list[tuple[Document, list[int]]]:
    docs_dir = path / "documents"
    bounds_dir = path / "boundaries"
    dataset: list[tuple[Document, list[int]]] = []
    for doc_file in sorted(docs_dir.glob("*.txt")):
        doc_id = doc_file.stem
        bf = bounds_dir / f"{doc_id}.json"
        if not bf.exists():
            continue
        sentences = doc_file.read_text(encoding="utf-8").splitlines()
        boundaries = json.loads(bf.read_text(encoding="utf-8"))["boundaries"]
        dataset.append((Document(doc_id=doc_id, sentences=sentences), boundaries))
    return dataset


def _evaluate(
    segmenter: object, dataset: list[tuple[Document, list[int]]]
) -> dict[str, float]:
    f1s: list[float] = []
    pks: list[float] = []
    wds: list[float] = []
    rts: list[float] = []
    for document, gt in dataset:
        t0 = time()
        result = segmenter.segment(document, max_segments=MAX_SEGMENTS)  # type: ignore[attr-defined]
        rts.append(time() - t0)
        n = document.n_sentences
        pks.append(compute_pk(gt, result.boundaries, n))
        wds.append(compute_window_diff(gt, result.boundaries, n))
        _, _, f1 = compute_f1_boundary(gt, result.boundaries)
        f1s.append(f1)
    return {
        "f1_mean": statistics.mean(f1s),
        "pk_mean": statistics.mean(pks),
        "windowdiff_mean": statistics.mean(wds),
        "runtime_mean_ms": statistics.mean(rts) * 1000,
    }


def run_comparison() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    console.rule("[bold blue]Comparación de backends de cohesión — TF-IDF vs SBERT")

    rows: list[dict] = []
    for ds_name, ds_path in DATASETS.items():
        dataset = load_dataset(ds_path)
        console.print(f"\n[bold]{ds_name}[/] — {len(dataset)} documentos")
        for backend in BACKENDS:
            console.print(f"  backend = {backend} …")
            segmenters = _make_segmenters(backend)
            for algo_name, segmenter in segmenters.items():
                metrics = _evaluate(segmenter, dataset)
                rows.append(
                    {
                        "dataset": ds_name,
                        "algorithm": algo_name,
                        "backend": backend,
                        **{k: round(v, 4) for k, v in metrics.items()},
                    }
                )

    # ── Guardar summary ───────────────────────────────────────────────────────
    summary_path = OUTPUT_PATH / "summary.csv"
    fieldnames = list(rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\n[green]✓ Summary → {summary_path}")

    # ── Deltas SBERT − TF-IDF ─────────────────────────────────────────────────
    indexed = {(r["dataset"], r["algorithm"], r["backend"]): r for r in rows}
    delta_rows: list[dict] = []
    for ds_name in DATASETS:
        for algo_name in ("dynamic_programming", "greedy", "simulated_annealing"):
            tf = indexed[(ds_name, algo_name, "tfidf")]
            sb = indexed[(ds_name, algo_name, "sbert")]
            delta_rows.append(
                {
                    "dataset": ds_name,
                    "algorithm": algo_name,
                    "f1_tfidf": tf["f1_mean"],
                    "f1_sbert": sb["f1_mean"],
                    "f1_delta": round(sb["f1_mean"] - tf["f1_mean"], 4),
                    "pk_tfidf": tf["pk_mean"],
                    "pk_sbert": sb["pk_mean"],
                    "pk_delta": round(sb["pk_mean"] - tf["pk_mean"], 4),
                    "wd_tfidf": tf["windowdiff_mean"],
                    "wd_sbert": sb["windowdiff_mean"],
                    "wd_delta": round(sb["windowdiff_mean"] - tf["windowdiff_mean"], 4),
                }
            )

    deltas_path = OUTPUT_PATH / "deltas.csv"
    with open(deltas_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(delta_rows)
    console.print(f"[green]✓ Deltas → {deltas_path}")

    # ── Tabla de deltas ───────────────────────────────────────────────────────
    table = Table(
        title="Δ SBERT − TF-IDF (F1↑ mejor positivo; Pk/WD↓ mejor negativo)",
        show_lines=True,
    )
    table.add_column("Dataset", style="cyan")
    table.add_column("Algoritmo", style="cyan")
    table.add_column("F1 TF-IDF", style="dim")
    table.add_column("F1 SBERT", style="green")
    table.add_column("Δ F1", style="bold green")
    table.add_column("Δ Pk", style="yellow")
    table.add_column("Δ WD", style="red")
    for r in delta_rows:
        table.add_row(
            r["dataset"],
            r["algorithm"],
            f"{r['f1_tfidf']:.4f}",
            f"{r['f1_sbert']:.4f}",
            f"{r['f1_delta']:+.4f}",
            f"{r['pk_delta']:+.4f}",
            f"{r['wd_delta']:+.4f}",
        )
    console.print()
    console.print(table)
    console.print(f"\n[green]✓ Resultados en {OUTPUT_PATH}")


if __name__ == "__main__":
    run_comparison()
