"""Análisis de sensibilidad de hiperparámetros del Recocido Simulado.

Ejecuta un grid de experimentos sobre tres parámetros del SA:
  - T0 (temperatura inicial): [0.5, 1.0, 2.0]
  - alpha (tasa de enfriamiento): [0.990, 0.995, 0.999]
  - n_iter (número de iteraciones): [500, 1000, 2000]

Cada configuración se corre con N_SEEDS semillas distintas (réplicas).
Se calculan medias, desviaciones estándar e intervalos de confianza al 95 %
usando la distribución t de Student (n pequeño).

Salida:
  results/exp_sa_sensitivity/
    raw_results.json   — cada fila: config × semilla × documento
    summary.csv        — media ± IC95 por configuración
    best_config.json   — la configuración con mayor F1 medio
"""

from __future__ import annotations

import json
import math
import statistics
from itertools import product
from pathlib import Path
from time import time

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from src.algorithms._cohesion import build_cohesion_matrix
from src.algorithms.simulated_annealing import SASegmenter
from src.core.models import Document
from src.evaluation.metrics import (
    compute_f1_boundary,
    compute_pk,
    compute_window_diff,
)

# ── Hiperparámetros del grid ──────────────────────────────────────────────────
T0_VALUES = [0.5, 1.0, 2.0]
ALPHA_VALUES = [0.990, 0.995, 0.999]
N_ITER_VALUES = [500, 1000, 2000]

# Semillas para las réplicas (30 réplicas → IC confiable para n pequeño)
N_SEEDS = 30
SEEDS = list(range(N_SEEDS))

MAX_SEGMENTS = 5
DATASET_PATH = Path("data/small/")
OUTPUT_PATH = Path("results/exp_sa_sensitivity/")

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def t_critical_95(n: int) -> float:
    """Valor crítico t para IC 95 % con n-1 grados de libertad (aproximación)."""
    # Tabla simplificada; para n >= 30 ≈ 2.045, para n < 30 usamos valores estándar
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 29: 2.045,
    }
    df = n - 1
    if df <= 0:
        return float("inf")
    # Devolver el valor más cercano disponible o 2.045 para df >= 29
    keys = sorted(t_table.keys())
    for k in keys:
        if df <= k:
            return t_table[k]
    return 2.045


def confidence_interval_95(values: list[float]) -> tuple[float, float]:
    """Retorna (media, margen_IC95) para una lista de valores."""
    n = len(values)
    if n < 2:
        return (values[0] if values else 0.0, 0.0)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    margin = t_critical_95(n) * std / math.sqrt(n)
    return mean, margin


def load_dataset(path: Path) -> list[tuple[Document, list[int]]]:
    """Carga documentos y fronteras de referencia."""
    docs_dir = path / "documents"
    bounds_dir = path / "boundaries"
    dataset = []
    for doc_file in sorted(docs_dir.glob("*.txt")):
        doc_id = doc_file.stem
        bf = bounds_dir / f"{doc_id}.json"
        if not bf.exists():
            continue
        sentences = doc_file.read_text(encoding="utf-8").splitlines()
        boundaries = json.loads(bf.read_text(encoding="utf-8"))["boundaries"]
        dataset.append((Document(doc_id=doc_id, sentences=sentences), boundaries))
    return dataset


# ── Núcleo del experimento ────────────────────────────────────────────────────

def run_sensitivity() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    console.rule("[bold blue]Análisis de Sensibilidad — Recocido Simulado")
    console.print(f"  Grid: {len(T0_VALUES)}×{len(ALPHA_VALUES)}×{len(N_ITER_VALUES)} = "
                  f"{len(T0_VALUES)*len(ALPHA_VALUES)*len(N_ITER_VALUES)} configuraciones")
    console.print(f"  Réplicas por configuración: {N_SEEDS} semillas")
    console.print(f"  Dataset: {DATASET_PATH}\n")

    dataset = load_dataset(DATASET_PATH)
    console.print(f"  Documentos cargados: {len(dataset)}")

    configs = list(product(T0_VALUES, ALPHA_VALUES, N_ITER_VALUES))
    total_runs = len(configs) * N_SEEDS * len(dataset)

    raw_results: list[dict] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Procesando...", total=total_runs)

        for t0, alpha, n_iter in configs:
            config_label = f"T0={t0} α={alpha} iter={n_iter}"

            for seed in SEEDS:
                segmenter = SASegmenter(
                    n_iterations=n_iter,
                    initial_temp=t0,
                    cooling_rate=alpha,
                    random_seed=seed,
                )

                for document, gt_boundaries in dataset:
                    t_start = time()
                    result = segmenter.segment(document, max_segments=MAX_SEGMENTS)
                    runtime = time() - t_start

                    n = document.n_sentences
                    pk = compute_pk(gt_boundaries, result.boundaries, n)
                    wd = compute_window_diff(gt_boundaries, result.boundaries, n)
                    _, _, f1 = compute_f1_boundary(gt_boundaries, result.boundaries)

                    raw_results.append({
                        "t0": t0,
                        "alpha": alpha,
                        "n_iter": n_iter,
                        "config_label": config_label,
                        "seed": seed,
                        "doc_id": document.doc_id,
                        "pk": round(pk, 4),
                        "windowdiff": round(wd, 4),
                        "f1_boundary": round(f1, 4),
                        "runtime_seconds": round(runtime, 6),
                    })
                    progress.advance(task)

    # ── Guardar raw ───────────────────────────────────────────────────────────
    raw_path = OUTPUT_PATH / "raw_results.json"
    raw_path.write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]✓ Raw results → {raw_path}")

    # ── Agregar por configuración ─────────────────────────────────────────────
    # Agrupamos: para cada config, promediamos sobre todos los documentos y semillas
    from collections import defaultdict
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw_results:
        key = (r["t0"], r["alpha"], r["n_iter"])
        grouped[key].append(r)

    summary_rows = []
    for (t0, alpha, n_iter), rows in sorted(grouped.items()):
        pks = [r["pk"] for r in rows]
        wds = [r["windowdiff"] for r in rows]
        f1s = [r["f1_boundary"] for r in rows]
        rts = [r["runtime_seconds"] for r in rows]

        pk_mean, pk_ci = confidence_interval_95(pks)
        wd_mean, wd_ci = confidence_interval_95(wds)
        f1_mean, f1_ci = confidence_interval_95(f1s)
        rt_mean, rt_ci = confidence_interval_95(rts)

        summary_rows.append({
            "t0": t0,
            "alpha": alpha,
            "n_iter": n_iter,
            "config_label": f"T0={t0} α={alpha} iter={n_iter}",
            "n_replicas": len(SEEDS),
            "n_observations": len(rows),
            "pk_mean": round(pk_mean, 4),
            "pk_ci95": round(pk_ci, 4),
            "windowdiff_mean": round(wd_mean, 4),
            "windowdiff_ci95": round(wd_ci, 4),
            "f1_mean": round(f1_mean, 4),
            "f1_ci95": round(f1_ci, 4),
            "runtime_mean_ms": round(rt_mean * 1000, 3),
            "runtime_ci95_ms": round(rt_ci * 1000, 3),
        })

    # Ordenar por F1 descendente
    summary_rows.sort(key=lambda x: x["f1_mean"], reverse=True)

    # Guardar CSV
    import csv
    summary_path = OUTPUT_PATH / "summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    console.print(f"[green]✓ Summary → {summary_path}")

    # Guardar mejor configuración
    best = summary_rows[0]
    best_path = OUTPUT_PATH / "best_config.json"
    best_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[green]✓ Best config → {best_path}")

    # ── Imprimir tabla resumen (top 10) ───────────────────────────────────────
    table = Table(
        title="Top 10 configuraciones por F1-Boundary (media ± IC 95 %)",
        show_lines=True,
    )
    table.add_column("T₀", style="cyan")
    table.add_column("α", style="cyan")
    table.add_column("iter", style="cyan")
    table.add_column("F1 ↑", style="green")
    table.add_column("IC 95 % F1", style="dim green")
    table.add_column("Pk ↓", style="yellow")
    table.add_column("IC 95 % Pk", style="dim yellow")
    table.add_column("WD ↓", style="red")
    table.add_column("RT (ms)", style="dim")

    for row in summary_rows[:10]:
        table.add_row(
            str(row["t0"]),
            str(row["alpha"]),
            str(row["n_iter"]),
            f"{row['f1_mean']:.4f}",
            f"±{row['f1_ci95']:.4f}",
            f"{row['pk_mean']:.4f}",
            f"±{row['pk_ci95']:.4f}",
            f"{row['windowdiff_mean']:.4f}",
            f"{row['runtime_mean_ms']:.1f}",
        )

    console.print()
    console.print(table)
    console.print(f"\n[bold green]Mejor configuración:[/] "
                  f"T₀={best['t0']}, α={best['alpha']}, iter={best['n_iter']} "
                  f"→ F1={best['f1_mean']:.4f} ± {best['f1_ci95']:.4f}")
    console.print(f"\n[green]✓ Todos los resultados guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    run_sensitivity()
