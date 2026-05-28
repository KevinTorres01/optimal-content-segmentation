from __future__ import annotations

import time
from typing import Callable

from rich.console import Console

from src.core.interfaces import BaseLLMEvaluator
from src.core.models import Segment
from src.llm.deepseek_provider import DeepSeekEvaluator
from src.llm.mistral_provider import MistralEvaluator
from src.llm.ollama_provider import OllamaEvaluator

console = Console()

_PROVIDERS: dict[str, Callable[..., BaseLLMEvaluator]] = {
    "mistral": MistralEvaluator,
    "deepseek": DeepSeekEvaluator,
    "ollama": OllamaEvaluator,
}

# A short, obviously-cohesive segment so a healthy model returns a high score.
_PROBE = Segment(
    segment_id="connectivity_probe",
    sentences=[
        "El equipo ganó el partido de fútbol.",
        "El delantero marcó dos goles en el segundo tiempo.",
    ],
)

_PARSE_ERROR_RATIONALE = "Error parsing LLM response"


def check_provider(provider: str, model: str | None = None) -> bool:
    """Run a single scoring call against one provider and report the outcome.

    This is meant as a fast pre-flight check before launching an experiment,
    especially useful on intermittent connections where you want to know the
    online provider is reachable before committing to a full run.

    Args:
        provider: One of "mistral", "deepseek", "ollama".
        model: Optional model override.

    Returns:
        True if the provider returned a well-formed score, False otherwise.
    """
    if provider not in _PROVIDERS:
        console.print(
            f"[red]Unknown provider: {provider}. " f"Available: {', '.join(_PROVIDERS)}"
        )
        return False

    try:
        evaluator = (
            _PROVIDERS[provider](model=model) if model else _PROVIDERS[provider]()
        )
    except ValueError as exc:
        console.print(f"[red]✗ {provider}: not configured[/]\n  {exc}")
        return False

    console.print(
        f"Probing [cyan]{evaluator.provider_name}[/] / "
        f"[cyan]{evaluator.model_name}[/]..."
    )
    start = time.perf_counter()
    try:
        score = evaluator.score_segment(_PROBE)
    except Exception as exc:  # connection, auth, timeout — surfaced by the SDK
        elapsed = time.perf_counter() - start
        console.print(
            f"[red]✗ {provider}: unreachable[/] (after {elapsed:.1f}s)\n"
            f"  {type(exc).__name__}: {exc}"
        )
        return False
    elapsed = time.perf_counter() - start

    if score.rationale == _PARSE_ERROR_RATIONALE:
        console.print(
            f"[yellow]⚠ {provider}: reached the model but the response was not "
            f"valid JSON[/] ({elapsed:.1f}s)"
        )
        return False

    console.print(
        f"[green]✓ {provider}: OK[/] ({elapsed:.1f}s) — "
        f"score={score.score}, rationale={score.rationale!r}"
    )
    return True


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Check that an LLM provider is reachable and responding."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=list(_PROVIDERS),
        help="Provider to probe.",
    )
    parser.add_argument("--model", default=None, help="Optional model override.")
    args = parser.parse_args()

    ok = check_provider(args.provider, args.model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
