from __future__ import annotations

import json
import shutil
from pathlib import Path

from dashboard.backend import control_actions as ca


def test_list_incidents_includes_data_research_fitness(monkeypatch) -> None:
    tmp = Path("tmp/tests/control_actions_incidents")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    reports_dir = tmp / "reports"
    logs_dir = tmp / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "DATA_RESEARCH_FITNESS.json").write_text(
        json.dumps(
            {
                "status": "warn",
                "warnings": ["no_spread:ETHUSDT"],
                "failures": [],
                "operator_summary": {
                    "headline": "1 warning(s), no failures",
                    "operator_action": "continue with caution; review degraded feature coverage",
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(ca, "REPO_ROOT", tmp)
    monkeypatch.setattr(ca, "LOGS_DIR", logs_dir)
    monkeypatch.setattr(ca, "_INCIDENT_STATE_PATH", logs_dir / "debug_incident_state.json")
    monkeypatch.setattr(ca, "_SESSION_DIR", logs_dir / "debug_sessions")

    rows = ca.list_incidents(limit=20)
    assert rows
    top = rows[0]
    assert top["incident_id"] == "data_research_fitness"
    assert top["type"] == "data_research_fitness"
    assert top["level"] == "WARNING"
    assert "continue with caution" in top["detail"]
