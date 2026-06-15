"""Interactive CLI to segment an arbitrary text with a chosen algorithm.

Run with no arguments to launch the interactive guided mode:

    python -m src.demo

Or pass flags for one-shot, non-interactive use:

    python -m src.demo --file my_text.txt --algorithm dynamic_programming -k 4
    cat my_text.txt | python -m src.demo --algorithm greedy -k 3
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from src.algorithms import ALGORITHM_REGISTRY
from src.algorithms.auto_k import AutoKResult, find_optimal_k
from src.core.models import Document, SegmentationResult

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])")

_ALGORITHM_INFO: dict[str, dict[str, str]] = {
    "brute_force": {
        "label": "Fuerza bruta",
        "blurb": "Exacto. Solo práctico hasta 15 oraciones.",
    },
    "dynamic_programming": {
        "label": "Programación dinámica",
        "blurb": "Exacto y eficiente. Recomendado por defecto.",
    },
    "greedy": {
        "label": "Greedy (TextTiling)",
        "blurb": "Heurística rápida basada en valles de similitud.",
    },
    "simulated_annealing": {
        "label": "Recocido simulado",
        "blurb": "Metaheurística estocástica con criterio de Metropolis.",
    },
}

_EXAMPLE_TEXT = (
    "El gato duerme en el sofá. "
    "Los felinos suelen descansar muchas horas al día. "
    "Un gato adulto puede dormir hasta 16 horas. "
    "Por otro lado, la economía cubana enfrenta retos importantes. "
    "La inflación afecta el poder adquisitivo. "
    "El gobierno busca nuevas medidas. "
    "En cuanto a la programación, Python es muy popular. "
    "Los desarrolladores valoran su sintaxis clara. "
    "Existen muchas bibliotecas para ciencia de datos."
)


def split_into_sentences(text: str, min_chars: int = 5) -> list[str]:
    """Split a raw text block into clean sentences."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    sentences: list[str] = []
    for paragraph in paragraphs:
        for raw in _SENTENCE_SPLIT_RE.split(paragraph):
            clean = raw.strip()
            if len(clean) >= min_chars:
                sentences.append(clean)
    return sentences


def parse_max_segments(raw: str) -> int | str:
    """Parse the -k flag: an integer, or the literal 'auto' for elbow selection."""
    if raw.strip().lower() == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"-k expects an integer or 'auto', got: {raw!r}"
        ) from exc


def parse_param(raw: str) -> tuple[str, int | float | str]:
    """Parse a --param key=value pair, coercing numeric values."""
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"--param expects key=value, got: {raw!r}")
    key, value = raw.split("=", 1)
    key, value = key.strip(), value.strip()
    for caster in (int, float):
        try:
            return key, caster(value)
        except ValueError:
            continue
    return key, value


def render_result(
    console: Console,
    document: Document,
    result: SegmentationResult,
) -> None:
    """Pretty-print the segmentation to the terminal."""
    segments = result.to_segments(document)

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Algoritmo:", result.algorithm_name)
    header.add_row("Oraciones:", str(document.n_sentences))
    header.add_row("Segmentos:", str(result.n_segments))
    header.add_row("Fronteras:", str(result.boundaries))
    header.add_row("Tiempo:", f"{result.runtime_seconds:.4f} s")
    console.print(header)
    console.print()

    for i, segment in enumerate(segments):
        start = result.boundaries[i]
        end = (
            result.boundaries[i + 1] - 1
            if i + 1 < len(result.boundaries)
            else document.n_sentences - 1
        )
        title = f"Segmento {i + 1}  (oraciones {start}–{end})"
        body = "\n".join(
            f"  [{start + j}] {s}" for j, s in enumerate(segment.sentences)
        )
        console.print(
            Panel(body, title=title, title_align="left", border_style="green")
        )


# --------------------------------------------------------------------------- #
# Interactive mode
# --------------------------------------------------------------------------- #


def _prompt_text_source(console: Console) -> str:
    """Ask the user to type or paste the text to segment."""
    console.print("[bold]Escribe o pega el texto que quieres segmentar.[/bold]")
    console.print(
        "[dim]Cuando termines, pulsa [bold]Ctrl+D[/bold] (Linux/macOS) "
        "o [bold]Ctrl+Z + Enter[/bold] (Windows).[/dim]"
    )
    console.print(
        "[dim]Atajos: escribe [bold]:ejemplo[/bold] para usar un texto de "
        "demostración, o [bold]:archivo <ruta>[/bold] para leer un fichero.[/dim]\n"
    )
    raw = sys.stdin.read().strip()

    if not raw:
        console.print("[yellow]No se recibió texto. Usando el ejemplo.[/yellow]")
        return _EXAMPLE_TEXT

    if raw.lower() == ":ejemplo":
        console.print("[dim]Usando texto de ejemplo (gatos / economía / Python).[/dim]")
        return _EXAMPLE_TEXT

    if raw.lower().startswith(":archivo"):
        parts = raw.split(maxsplit=1)
        if len(parts) == 2:
            path = Path(parts[1]).expanduser()
            if path.is_file():
                return path.read_text(encoding="utf-8")
            console.print(
                f"[red]No existe el archivo: {path}. Usando el ejemplo.[/red]"
            )
            return _EXAMPLE_TEXT
        console.print("[red]Falta la ruta tras :archivo. Usando el ejemplo.[/red]")
        return _EXAMPLE_TEXT

    return raw


def _prompt_algorithm(console: Console) -> str:
    """Show a numbered menu and return the chosen algorithm key."""
    console.print("\n[bold]¿Qué algoritmo quieres usar?[/bold]")
    keys = list(ALGORITHM_REGISTRY.keys())
    for i, key in enumerate(keys, start=1):
        info = _ALGORITHM_INFO[key]
        console.print(
            f"  [cyan]{i}[/cyan]) [bold]{info['label']}[/bold] "
            f"[dim]({key})[/dim] — {info['blurb']}"
        )
    idx = IntPrompt.ask(
        "Elige",
        choices=[str(i) for i in range(1, len(keys) + 1)],
        default=2,  # dynamic_programming
    )
    return keys[idx - 1]


def _print_auto_k_curve(console: Console, auto: AutoKResult) -> None:
    """Show the J(k) curve and which k was picked."""
    console.print("\n[dim]Auto-k explorando objetivo J(k) por DP exacto…[/dim]")
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", justify="right")
    table.add_column(justify="right")
    table.add_column()
    for k, j in sorted(auto.objectives.items()):
        marker = " [bold green]← elegido[/bold green]" if k == auto.k else ""
        table.add_row(f"k={k}", f"J={j:.4f}", marker)
    console.print(table)
    console.print(
        f"[green]✓[/green] k = [bold]{auto.k}[/bold]  " f"[dim]({auto.rationale})[/dim]"
    )


def _prompt_k(console: Console, document: Document, last_k: int | None) -> int:
    """Ask for k, defaulting to automatic elbow-based selection."""
    n = document.n_sentences
    if n <= 2:
        console.print(f"[dim]Solo {n} oración(es) → k = {n}[/dim]")
        return n

    default = str(last_k) if last_k else "auto"
    console.print(
        "\n[bold]¿Cuántos segmentos (k)?[/bold]  "
        "[dim]Enter para auto (método del codo), o introduce un entero.[/dim]"
    )
    while True:
        raw = Prompt.ask("k", default=default).strip().lower()
        if raw in ("auto", ""):
            auto = find_optimal_k(document)
            _print_auto_k_curve(console, auto)
            return auto.k
        try:
            k = int(raw)
            if 1 <= k <= n:
                return k
            console.print(f"[red]k debe estar entre 1 y {n}.[/red]")
        except ValueError:
            console.print("[red]Introduce un entero o 'auto'.[/red]")


def _prompt_algorithm_params(
    console: Console, algorithm: str
) -> dict[str, int | float | str]:
    """Ask for algorithm-specific hyperparameters (with sensible defaults)."""
    params: dict[str, int | float | str] = {}
    if algorithm == "greedy":
        params["window_size"] = IntPrompt.ask("Tamaño de ventana", default=2)
    elif algorithm == "simulated_annealing":
        params["n_iterations"] = IntPrompt.ask("Iteraciones", default=2000)
        seed_str = Prompt.ask(
            "Semilla aleatoria (entero, o vacío para no fijar)", default="42"
        )
        if seed_str.strip():
            try:
                params["random_seed"] = int(seed_str)
            except ValueError:
                console.print("[yellow]Semilla inválida, se ignora.[/yellow]")
    return params


def _prompt_next_action(console: Console) -> str:
    """After showing a result, ask what to do next."""
    console.print("\n[bold]¿Qué quieres hacer ahora?[/bold]")
    console.print("  [cyan]1[/cyan]) Probar otro algoritmo sobre el mismo texto")
    console.print("  [cyan]2[/cyan]) Cambiar el número de segmentos (k)")
    console.print("  [cyan]3[/cyan]) Cargar otro texto")
    console.print("  [cyan]4[/cyan]) Salir")
    return Prompt.ask("Elige", choices=["1", "2", "3", "4"], default="4")


def run_interactive(console: Console) -> int:
    """Guided interactive flow: pick text, algorithm, k, params, see result."""
    console.print(
        Panel.fit(
            "[bold cyan]Aplicación de Segmentación Óptima de Textos[/bold cyan]\n\n"
            "Divide un texto en segmentos semánticamente coherentes usando\n"
            "uno de los 4 algoritmos del proyecto (fuerza bruta, programación\n"
            "dinámica, greedy o recocido simulado).\n\n"
            "[dim]Modo demo: sin métricas, sin LLM — inspección cualitativa.[/dim]",
            border_style="cyan",
            title="✦ Optimal Content Segmentation ✦",
            title_align="center",
        )
    )
    console.print()

    document: Document | None = None
    last_k: int | None = None

    while True:
        if document is None:
            raw_text = _prompt_text_source(console)
            sentences = split_into_sentences(raw_text)
            if not sentences:
                console.print("[red]No se detectaron oraciones. Reintenta.[/red]\n")
                continue
            console.print(
                f"\n[green]✓[/green] Detectadas [bold]{len(sentences)}[/bold] oraciones."
            )
            document = Document(doc_id="demo", sentences=sentences)
            last_k = None

        algorithm = _prompt_algorithm(console)
        k = _prompt_k(console, document, last_k)
        last_k = k

        if algorithm == "brute_force" and document.n_sentences > 15:
            console.print(
                f"[yellow]Aviso:[/yellow] fuerza bruta solo es práctica con "
                f"≤ 15 oraciones; tienes {document.n_sentences}. Puede tardar mucho."
            )
            if Prompt.ask("¿Continuar?", choices=["s", "n"], default="n") == "n":
                continue

        params = _prompt_algorithm_params(console, algorithm)
        segmenter = ALGORITHM_REGISTRY[algorithm](**params)

        console.print("\n[dim]Calculando…[/dim]\n")
        result = segmenter.segment(document, max_segments=k)
        render_result(console, document, result)

        action = _prompt_next_action(console)
        if action == "1":
            continue
        if action == "2":
            continue  # k will be re-prompted at the top of the loop
        if action == "3":
            document = None
            last_k = None
            console.print()
            continue
        console.print("\n[cyan]¡Hasta la próxima![/cyan]")
        return 0


# --------------------------------------------------------------------------- #
# Non-interactive mode (flags)
# --------------------------------------------------------------------------- #


def _read_text_from_flags(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit(
        "No input text. Pass --text, --file <path>, or pipe text via stdin "
        "(or run without flags for interactive mode)."
    )


def run_one_shot(console: Console, args: argparse.Namespace) -> int:
    """One-shot non-interactive run driven entirely by CLI flags."""
    text = _read_text_from_flags(args)
    sentences = split_into_sentences(text)
    if not sentences:
        console.print("[red]No sentences detected in the input text.[/red]")
        return 1

    document = Document(doc_id=args.doc_id, sentences=sentences)
    segmenter_cls = ALGORITHM_REGISTRY[args.algorithm]
    try:
        segmenter = segmenter_cls(**dict(args.param))
    except TypeError as exc:
        console.print(f"[red]Invalid params for {args.algorithm}: {exc}[/red]")
        return 2

    if args.algorithm == "brute_force" and document.n_sentences > 15:
        console.print(
            f"[yellow]Warning:[/yellow] brute_force is only practical for "
            f"≤ 15 sentences; got {document.n_sentences}."
        )

    max_segments = args.max_segments
    if max_segments == "auto":
        # Forward the chosen cohesion backend so auto-k's DP sweep uses the same
        # sentence representation the segmenter does (default TF-IDF otherwise).
        cohesion_backend = dict(args.param).get("cohesion_backend", "tfidf")
        auto = find_optimal_k(document, cohesion_backend=cohesion_backend)
        _print_auto_k_curve(console, auto)
        # Reuse boundaries from the auto-k sweep when DP is the chosen algorithm —
        # they were computed in the same pass, so no second triple-loop needed.
        if args.algorithm == "dynamic_programming" and auto.boundaries:
            from src.core.models import SegmentationResult

            result = SegmentationResult(
                doc_id=args.doc_id,
                boundaries=auto.boundaries,
                algorithm_name=segmenter.name,
                runtime_seconds=0.0,
            )
            render_result(console, document, result)
            return 0
        max_segments = auto.k

    result = segmenter.segment(document, max_segments=max_segments)
    render_result(console, document, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.demo",
        description=(
            "Segment an arbitrary text. Run with no arguments for interactive mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--text", help="Inline text to segment.")
    src_group.add_argument("--file", help="Path to a UTF-8 text file.")
    parser.add_argument(
        "-a",
        "--algorithm",
        choices=sorted(ALGORITHM_REGISTRY.keys()),
        help="Segmentation algorithm to run.",
    )
    parser.add_argument(
        "-k",
        "--max-segments",
        type=parse_max_segments,
        default=None,
        metavar="N|auto",
        help=(
            "Number of segments: an integer, or 'auto' to pick k automatically "
            "with the elbow method (default: min(5, n_sentences))."
        ),
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=parse_param,
        metavar="key=value",
        help="Extra constructor arg for the algorithm (repeatable).",
    )
    parser.add_argument("--doc-id", default="demo", help="Document identifier.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    interactive = (
        args.text is None
        and args.file is None
        and args.algorithm is None
        and sys.stdin.isatty()
    )
    if interactive:
        try:
            return run_interactive(console)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[cyan]Interrumpido. ¡Hasta la próxima![/cyan]")
            return 0

    if args.algorithm is None:
        args.algorithm = "dynamic_programming"
    return run_one_shot(console, args)


if __name__ == "__main__":
    raise SystemExit(main())
