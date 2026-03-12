from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from tools import preflight_check as pf


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"preflight_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def _mk_db(path: Path, ts_ms: int) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE mark_prices (ts INTEGER, price REAL)")
        conn.execute("INSERT INTO mark_prices(ts,price) VALUES (?,?)", (int(ts_ms), 100.0))
        conn.commit()
    finally:
        conn.close()


def test_preflight_passes_with_fresh_db(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        db = tmp / "micro.db"
        now_ms = int(time.time() * 1000.0)
        _mk_db(db, now_ms)
        out_json = tmp / "reports" / "preflight.json"
        out_md = tmp / "reports" / "preflight.md"

        monkeypatch.setenv("SCALPER_DRY_RUN", "1")
        monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT")
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            pf,
            "analyze_research_fitness",
            lambda **kwargs: (
                seen.update(kwargs),
                {
                    "status": "pass",
                    "db_ready": True,
                    "warnings": [],
                    "failures": [],
                    "contract": {"tier": "trade_plus_liq_mark_proxy"},
                },
            )[1],
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--trade-db",
                str(tmp / "paper.db"),
                "--max-db-stale-sec",
                "3600",
                "--min-free-gb",
                "0",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert pf.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert seen["symbols"] == ["ETHUSDT"]
        assert payload["run_summary"]["run_type"] == "preflight_check"
        assert out_md.exists()
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_warns_on_data_research_fitness(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        db = tmp / "micro.db"
        now_ms = int(time.time() * 1000.0)
        _mk_db(db, now_ms)
        out_json = tmp / "reports" / "preflight.json"
        out_md = tmp / "reports" / "preflight.md"

        monkeypatch.setenv("SCALPER_DRY_RUN", "1")
        monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT")
        monkeypatch.setattr(
            pf,
            "analyze_research_fitness",
            lambda **kwargs: {
                "status": "warn",
                "db_ready": True,
                "warnings": ["no_spread:ETHUSDT"],
                "failures": [],
                "contract": {"tier": "trade_plus_liq_mark_proxy"},
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--max-db-stale-sec",
                "3600",
                "--min-free-gb",
                "0",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert pf.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ok"] is True
        assert payload["checks"]["data_research_fitness_symbols"] == ["ETHUSDT"]
        assert payload["checks"]["data_research_fitness_status"] == "warn"
        assert payload["checks"]["data_research_fitness_summary"] == "1 warning(s), no failures"
        assert payload["data_research_fitness_summary"]["warning_summary"] == ["spread not computable for ETHUSDT"]
        assert any("Data research fitness warn" in item for item in payload["warnings"])
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


def test_preflight_fails_without_paper_profile(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        db = tmp / "micro.db"
        now_ms = int(time.time() * 1000.0)
        _mk_db(db, now_ms)
        out_json = tmp / "reports" / "preflight.json"
        out_md = tmp / "reports" / "preflight.md"

        monkeypatch.delenv("SCALPER_ENV_PROFILE", raising=False)
        monkeypatch.delenv("SCALPER_DRY_RUN", raising=False)
        monkeypatch.setenv("ACTIVE_SYMBOLS", "ETHUSDT")
        monkeypatch.setattr(
            pf,
            "analyze_research_fitness",
            lambda **kwargs: {
                "status": "pass",
                "db_ready": True,
                "warnings": [],
                "failures": [],
                "contract": {"tier": "trade_plus_liq_mark_proxy"},
            },
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--db",
                str(db),
                "--max-db-stale-sec",
                "3600",
                "--min-free-gb",
                "0",
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
        )
        assert pf.main() == 1
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["ok"] is False
        assert payload["checks"]["SCALPER_ENV_PROFILE"] == ""
        assert any("SCALPER_ENV_PROFILE" in item for item in payload["failures"])
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)
