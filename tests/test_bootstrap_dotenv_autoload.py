from __future__ import annotations

import os
from pathlib import Path
import shutil
import uuid

try:
    from execution import bootstrap as bs
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import bootstrap as bs


def test_bootstrap_dotenv_prefers_env_paper(monkeypatch) -> None:
    tmp = Path("localtests") / f"dotenv_boot_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        env_paper = tmp / ".env.paper"
        env_default = tmp / ".env"
        env_paper.write_text("ZZ_BOOTSTRAP_DOTENV_TEST=paper\n", encoding="utf-8")
        env_default.write_text("ZZ_BOOTSTRAP_DOTENV_TEST=default\n", encoding="utf-8")
        monkeypatch.chdir(tmp)
        monkeypatch.delenv("ZZ_BOOTSTRAP_DOTENV_TEST", raising=False)
        monkeypatch.delenv("SCALPER_ENV_PROFILE", raising=False)
        src = bs._load_dotenv_best_effort()
        assert src == ".env.paper"
        assert os.getenv("ZZ_BOOTSTRAP_DOTENV_TEST") == "paper"
        assert os.getenv("SCALPER_ENV_PROFILE") == "paper"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
