from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app
from dashboard.backend import data_sources as ds


def test_overview_normalizes_numeric_regime_ts(monkeypatch) -> None:
    logs_dir = Path("tmp/tests/dashboard_overview_api/logs")
    if logs_dir.parent.exists():
        shutil.rmtree(logs_dir.parent, ignore_errors=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    regimes_file = logs_dir / "regime_transitions.jsonl"
    regimes_file.write_text(
        "\n".join(
            [
                json.dumps({"ts": 1771391020, "symbol": "ETHUSDT", "effective_regime": "trending"}),
                json.dumps({"ts": 1771391035000, "symbol": "BTCUSDT", "effective_regime": "ranging"}),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ds, "LOGS_DIR", logs_dir)

    client = TestClient(app)
    response = client.get("/api/overview")
    assert response.status_code == 200, response.text

    payload = response.json()
    recent = payload.get("recent_regimes", [])
    assert len(recent) == 2
    assert all(isinstance(item.get("ts"), str) for item in recent)
    assert all("T" in item["ts"] for item in recent)


def test_health_overall_stats_reads_data_research_fitness(monkeypatch) -> None:
    logs_dir = Path("tmp/tests/dashboard_overview_health/logs")
    if logs_dir.parent.exists():
        shutil.rmtree(logs_dir.parent, ignore_errors=True)
    (logs_dir / "health").mkdir(parents=True, exist_ok=True)
    (logs_dir / "health" / "overall.json").write_text(
        json.dumps(
            {
                "ts_utc": "2026-03-10T00:00:00+00:00",
                "mode": "paper",
                "state": "ok",
                "components": {
                    "collector": {
                        "connected": True,
                        "reconnects_last_5m": 1,
                        "errors_last_5m": 0,
                    },
                    "data_research_fitness": {
                        "status": "warning",
                        "connected": True,
                        "detail": "fitness_status=warn warnings=1 failures=0",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ds, "LOGS_DIR", logs_dir)
    payload = ds._health_overall_stats()
    assert payload["collector_connected"] is True
    assert payload["data_research_fitness_status"] == "warning"
    assert payload["data_research_fitness_connected"] is True


def test_overview_includes_health_overall_summary(monkeypatch) -> None:
    logs_dir = Path("tmp/tests/dashboard_overview_with_fitness/logs")
    if logs_dir.parent.exists():
        shutil.rmtree(logs_dir.parent, ignore_errors=True)
    (logs_dir / "health").mkdir(parents=True, exist_ok=True)
    (logs_dir / "health" / "overall.json").write_text(
        json.dumps(
            {
                "components": {
                    "collector": {"connected": True},
                    "data_research_fitness": {
                        "status": "warning",
                        "connected": True,
                        "detail": "fitness_status=warn warnings=1 failures=0",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ds, "LOGS_DIR", logs_dir)

    payload = ds.build_overview()
    assert payload["health_overall"]["data_research_fitness_status"] == "warning"
    assert payload["health_overall"]["data_research_fitness_connected"] is True
