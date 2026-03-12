from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

try:
    from execution import bootstrap as bs
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution import bootstrap as bs


def test_write_startup_manifest_persists_runtime_contract(monkeypatch) -> None:
    tmp = (Path("localtests") / f"startup_manifest_{uuid.uuid4().hex[:8]}").resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        monkeypatch.setenv("SCALPER_ENV_PROFILE", "paper")
        monkeypatch.setenv("SCALPER_DRY_RUN", "1")
        monkeypatch.setenv("PAPER_EXECUTION_MODE", "router_blocked")
        monkeypatch.setenv("BINANCE_TESTNET", "1")
        monkeypatch.setenv("BINANCE_API_KEY", "")
        monkeypatch.setenv("BINANCE_API_SECRET", "")

        bot = SimpleNamespace(
            cfg=SimpleNamespace(EXCHANGE="binance", DEFAULT_TYPE="future", ACTIVE_SYMBOLS=["ETHUSDT"]),
            state=SimpleNamespace(run_context={}),
        )

        bs._write_startup_manifest(bot, dotenv_source=".env.paper")

        manifest_path = tmp / "logs" / "paper_startup_manifest.json"
        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert payload["entrypoint"] == "execution.bootstrap"
        assert payload["dotenv_source"] == ".env.paper"
        assert payload["env_profile"] == "paper"
        assert payload["dry_run"] is True
        assert payload["binance_testnet"] is True
        assert payload["private_api_key_present"] is False
        assert bot.state.run_context["startup_manifest_path"] == "logs/paper_startup_manifest.json"
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
