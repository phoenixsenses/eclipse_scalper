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
