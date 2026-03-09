from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app
from dashboard.backend import data_sources as ds


def test_read_research_events_picks_latest_daily_report(monkeypatch) -> None:
    reports_dir = Path("tmp/tests/dashboard_research_events_api/reports")
    if reports_dir.parent.exists():
        shutil.rmtree(reports_dir.parent, ignore_errors=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    older = reports_dir / "DAILY_2026-03-12.json"
    newer = reports_dir / "DAILY_2026-03-13.json"
    older.write_text(
        json.dumps(
            {
                "report_date": "2026-03-12",
                "headline": {
                    "event_lanes": "BLOCKED",
                    "regime_recovery_prep": "WATCH",
                    "pocket_promotion_checklist": "WATCH",
                },
            }
        ),
        encoding="utf-8",
    )
    newer.write_text(
        json.dumps(
            {
                "report_date": "2026-03-13",
                "headline": {
                    "event_lanes": "CLEAR",
                    "regime_recovery_prep": "HOLD",
                    "pocket_promotion_checklist": "INCOMPLETE",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ds, "REPORTS_DIR", reports_dir)

    payload = ds.read_research_events()
    daily = payload["daily_report"]
    assert daily["report_date"] == "2026-03-13"
    assert daily["headline"]["event_lanes"] == "CLEAR"
    assert daily["_meta"]["exists"] is True
    assert str(newer) == daily["_meta"]["path"]


def test_overview_exposes_daily_research_report(monkeypatch) -> None:
    reports_dir = Path("tmp/tests/dashboard_research_events_api_overview/reports")
    logs_dir = Path("tmp/tests/dashboard_research_events_api_overview/logs")
    state_dir = Path("tmp/tests/dashboard_research_events_api_overview/state")
    data_dir = Path("tmp/tests/dashboard_research_events_api_overview/data")
    root = reports_dir.parent
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "DAILY_2026-03-13.json").write_text(
        json.dumps(
            {
                "report_date": "2026-03-13",
                "headline": {
                    "event_lanes": "CLEAR",
                    "regime_recovery_prep": "HOLD",
                    "pocket_promotion_checklist": "INCOMPLETE",
                },
                "event_lane": {"status": "CLEAR", "summary": "gate=ALLOWED"},
                "recovery": {"status": "HOLD", "summary": "guard_mode=ORANGE"},
                "promotion": {"status": "INCOMPLETE", "summary": "missing promotion artifacts=5"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ds, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(ds, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ds, "STATE_DIR", state_dir)
    monkeypatch.setattr(ds, "DATA_DIR", data_dir)

    client = TestClient(app)
    response = client.get("/api/overview")
    assert response.status_code == 200, response.text

    payload = response.json()
    daily = payload["research_events"]["daily_report"]
    assert daily["report_date"] == "2026-03-13"
    assert daily["headline"]["event_lanes"] == "CLEAR"
    assert daily["headline"]["regime_recovery_prep"] == "HOLD"
    assert daily["headline"]["pocket_promotion_checklist"] == "INCOMPLETE"
