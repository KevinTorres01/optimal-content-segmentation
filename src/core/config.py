from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from src.core.models import ExperimentConfig


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated ExperimentConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is malformed or fails Pydantic validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        # Re-raise with a human-readable message listing missing/invalid fields
        errors = "; ".join(
            f"{' -> '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"Invalid experiment config at {path}: {errors}") from exc
