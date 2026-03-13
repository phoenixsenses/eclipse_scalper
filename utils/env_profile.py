from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def env_truthy(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def current_env_profile(default: str = "") -> str:
    return str(os.getenv("SCALPER_ENV_PROFILE", default) or default).strip().lower()


def paper_profile_active() -> bool:
    return current_env_profile() == "paper" or env_truthy("SCALPER_DRY_RUN", False)


def load_dotenv_best_effort(*, root: Optional[Path] = None, cwd_fallback: bool = False) -> str:
    """
    Load environment variables with a consistent profile contract.

    Priority:
    1. .env.paper
    2. .env
    3. generic dotenv discovery (optional)
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return ""

    try:
        base = Path(root) if root is not None else Path.cwd()
        env_paper = base / ".env.paper"
        env_default = base / ".env"
        if env_paper.exists():
            load_dotenv(dotenv_path=env_paper, override=False)
            os.environ.setdefault("SCALPER_ENV_PROFILE", "paper")
            return ".env.paper"
        if env_default.exists():
            load_dotenv(dotenv_path=env_default, override=False)
            return ".env"
        if cwd_fallback:
            load_dotenv(override=False)
    except Exception:
        return ""
    return ""
