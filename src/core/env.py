from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Project root is two levels up from this file: src/core/env.py -> src/ -> root/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load environment variables from the project-root .env file.

    Existing environment variables are not overridden, so values set
    explicitly in the shell take precedence over .env. Missing .env is a
    no-op, which keeps tests and CI working without a secrets file.
    """
    load_dotenv(_PROJECT_ROOT / ".env", override=False)
