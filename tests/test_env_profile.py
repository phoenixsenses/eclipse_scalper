from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from utils import env_profile


def test_load_dotenv_best_effort_sets_paper_profile(monkeypatch) -> None:
    tmp = Path("localtests") / f"env_profile_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        (tmp / ".env.paper").write_text("ZZ_ENV_PROFILE_TEST=paper\n", encoding="utf-8")
        monkeypatch.delenv("ZZ_ENV_PROFILE_TEST", raising=False)
        monkeypatch.delenv("SCALPER_ENV_PROFILE", raising=False)
        src = env_profile.load_dotenv_best_effort(root=tmp)
        assert src == ".env.paper"
        assert os.getenv("ZZ_ENV_PROFILE_TEST") == "paper"
        assert os.getenv("SCALPER_ENV_PROFILE") == "paper"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
