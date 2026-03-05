from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app


def test_runbook_endpoint_returns_session(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setattr(
        "dashboard.backend.app.run_runbook",
        lambda actions=None: {
            "session_id": "session_1",
            "started_ts": 1.0,
            "ended_ts": 2.0,
            "duration_sec": 1.0,
            "ok": True,
            "failed_action": None,
            "steps": [],
            "incident_hint": None,
        },
    )
    client = TestClient(app)
    response = client.post("/api/debug/runbook", json={"actions": ["validate_env"]})
    assert response.status_code == 200
    assert response.json()["session_id"] == "session_1"


def test_sessions_list_and_detail(monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard.backend.app.list_runbook_sessions",
        lambda limit=30: [
            {
                "session_id": "session_1",
                "started_ts": 1.0,
                "ended_ts": 2.0,
                "duration_sec": 1.0,
                "ok": False,
                "failed_action": "preflight_check",
                "incident_type": "data_freshness",
                "tag": "data",
                "note_preview": "db stale",
            }
        ],
    )
    monkeypatch.setattr(
        "dashboard.backend.app.get_runbook_session",
        lambda session_id: {
            "session_id": session_id,
            "started_ts": 1.0,
            "ended_ts": 2.0,
            "duration_sec": 1.0,
            "ok": False,
            "failed_action": "preflight_check",
            "steps": [],
            "incident_hint": {
                "type": "data_freshness",
                "title": "Data freshness issue",
                "detail": "x",
                "file": "microstructure_collector.log",
                "query": "stale",
                "level": "WARNING",
            },
            "log_snippets": [],
            "context": {},
            "tag": "data",
            "note": "db stale",
            "updated_ts": 2.0,
        },
    )

    client = TestClient(app)
    list_resp = client.get("/api/debug/sessions")
    assert list_resp.status_code == 200
    assert list_resp.json()[0]["session_id"] == "session_1"

    detail_resp = client.get("/api/debug/sessions/session_1")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["failed_action"] == "preflight_check"


def test_runbook_from_incident_patch_and_timeline(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setattr(
        "dashboard.backend.app.run_runbook_with_context",
        lambda actions=None, context=None: {
            "session_id": "session_2",
            "started_ts": 10.0,
            "ended_ts": 12.0,
            "duration_sec": 2.0,
            "ok": True,
            "failed_action": None,
            "steps": [],
            "incident_hint": None,
            "log_snippets": [],
            "context": context or {},
            "tag": "",
            "note": "",
            "updated_ts": 12.0,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.update_runbook_session",
        lambda session_id, tag=None, note=None: {
            "session_id": session_id,
            "started_ts": 10.0,
            "ended_ts": 12.0,
            "duration_sec": 2.0,
            "ok": True,
            "failed_action": None,
            "steps": [],
            "incident_hint": None,
            "log_snippets": [],
            "context": {},
            "tag": tag,
            "note": note,
            "updated_ts": 13.0,
        },
    )
    monkeypatch.setattr("dashboard.backend.app.get_runbook_session", lambda session_id: {"session_id": session_id, "steps": []})
    monkeypatch.setattr(
        "dashboard.backend.app.build_session_timeline",
        lambda session: [{"ts": 11.0, "kind": "step_end", "title": "x", "detail": "", "status": "ok", "action": "validate_env"}],
    )

    client = TestClient(app)
    run_resp = client.post(
        "/api/debug/runbook/from-incident",
        json={"file": "paper_trading.log", "query": "RequestTimeout", "level": "WARNING", "actions": ["validate_env"]},
    )
    assert run_resp.status_code == 200
    assert run_resp.json()["context"]["source"] == "dashboard_incident"

    patch_resp = client.patch("/api/debug/sessions/session_2", json={"tag": "network", "note": "timeout burst"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["tag"] == "network"

    timeline_resp = client.get("/api/debug/sessions/session_2/timeline")
    assert timeline_resp.status_code == 200
    assert timeline_resp.json()[0]["kind"] == "step_end"


def test_incident_inbox_and_policy_endpoints(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setattr(
        "dashboard.backend.app.list_incidents",
        lambda limit=50: [
            {
                "incident_id": "session_3",
                "session_id": "session_3",
                "ts": 100.0,
                "type": "exchange_timeout",
                "title": "Exchange timeout",
                "level": "WARNING",
                "file": "paper_trading.log",
                "query": "RequestTimeout",
                "status": "new",
                "snoozed_until": None,
                "muted": False,
                "failed_action": "preflight_check",
            }
        ],
    )
    monkeypatch.setattr(
        "dashboard.backend.app.update_incident",
        lambda incident_id, action, incident_type=None, snooze_minutes=60: {
            "ok": True,
            "incident_id": incident_id,
            "action": action,
            "incident_type": incident_type,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.bulk_update_incidents",
        lambda action, incident_type=None, status_scope="active": {
            "ok": True,
            "updated": 3,
            "action": action,
            "incident_type": incident_type,
            "status_scope": status_scope,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.preview_bulk_update_incidents",
        lambda incident_type=None, status_scope="active": {
            "ok": True,
            "eligible": 4,
            "incident_type": incident_type,
            "status_scope": status_scope,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.undo_last_incident_action",
        lambda: {"ok": True, "reason": "restored"},
    )
    monkeypatch.setattr(
        "dashboard.backend.app.read_incident_audit",
        lambda limit=50: [{"ts": 1.0, "operator": "local", "kind": "bulk", "action": "ack", "updated": 3}],
    )
    monkeypatch.setattr(
        "dashboard.backend.app.run_runbook_for_incident",
        lambda incident_id: {
            "session_id": "session_9",
            "started_ts": 1.0,
            "ended_ts": 2.0,
            "duration_sec": 1.0,
            "ok": True,
            "failed_action": None,
            "steps": [],
            "incident_hint": None,
            "log_snippets": [],
            "context": {"incident_id": incident_id},
            "tag": "",
            "note": "",
            "updated_ts": 2.0,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.get_auto_runbook_policy",
        lambda: {"enabled": False, "min_level": "WARNING", "cooldown_sec": 900, "last_run_ts_by_type": {}},
    )
    monkeypatch.setattr(
        "dashboard.backend.app.set_auto_runbook_policy",
        lambda enabled=None, min_level=None, cooldown_sec=None: {
            "enabled": bool(enabled),
            "min_level": min_level or "WARNING",
            "cooldown_sec": cooldown_sec or 900,
            "last_run_ts_by_type": {},
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.run_auto_runbook_once",
        lambda: {"ran": True, "reason": "executed", "incident_id": "session_3", "session_id": "session_9"},
    )
    monkeypatch.setattr(
        "dashboard.backend.app.get_macro_preset",
        lambda: {
            "preset": "full",
            "ackFiltered": True,
            "autoRun": True,
            "exportMd": True,
            "refresh": True,
            "owner": "qa",
            "updated_ts": 123.0,
        },
    )
    monkeypatch.setattr(
        "dashboard.backend.app.set_macro_preset",
        lambda preset=None, ack_filtered=None, auto_run=None, export_md=None, refresh=None, owner=None: {
            "preset": preset or "custom",
            "ackFiltered": bool(ack_filtered),
            "autoRun": bool(auto_run),
            "exportMd": bool(export_md),
            "refresh": bool(refresh),
            "owner": owner or "local",
            "updated_ts": 456.0,
        },
    )
    client = TestClient(app)
    inc_resp = client.get("/api/debug/incidents")
    assert inc_resp.status_code == 200
    assert inc_resp.json()[0]["incident_id"] == "session_3"

    patch_resp = client.patch("/api/debug/incidents/session_3", json={"action": "ack"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["ok"] is True
    bulk_resp = client.post("/api/debug/incidents/bulk", json={"action": "resolve", "incident_type": "exchange_timeout", "status_scope": "active"})
    assert bulk_resp.status_code == 200
    assert bulk_resp.json()["updated"] == 3
    preview_resp = client.post("/api/debug/incidents/bulk/preview", json={"incident_type": "exchange_timeout", "status_scope": "active"})
    assert preview_resp.status_code == 200
    assert preview_resp.json()["eligible"] == 4
    undo_resp = client.post("/api/debug/incidents/undo")
    assert undo_resp.status_code == 200
    assert undo_resp.json()["ok"] is True
    audit_resp = client.get("/api/debug/incidents/audit")
    assert audit_resp.status_code == 200
    assert audit_resp.json()[0]["kind"] == "bulk"

    run_resp = client.post("/api/debug/incidents/session_3/runbook")
    assert run_resp.status_code == 200
    assert run_resp.json()["context"]["incident_id"] == "session_3"

    pol_get = client.get("/api/debug/incidents-policy")
    assert pol_get.status_code == 200
    pol_patch = client.patch("/api/debug/incidents-policy", json={"enabled": True, "min_level": "ERROR", "cooldown_sec": 1200})
    assert pol_patch.status_code == 200
    assert pol_patch.json()["enabled"] is True

    auto_resp = client.post("/api/debug/incidents/auto-run")
    assert auto_resp.status_code == 200
    assert auto_resp.json()["ran"] is True
    macro_get = client.get("/api/debug/macro-preset")
    assert macro_get.status_code == 200
    macro_patch = client.patch(
        "/api/debug/macro-preset",
        json={"preset": "quick", "ackFiltered": True, "autoRun": False, "exportMd": False, "refresh": True},
        headers={"X-Operator": "alice"},
    )
    assert macro_patch.status_code == 200
    assert macro_patch.json()["preset"] == "quick"
    assert macro_patch.json()["owner"] == "alice"


def test_auto_run_idempotency_replay(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setenv("DASHBOARD_CONTROL_ROLE", "admin")
    monkeypatch.setenv("DASHBOARD_IDEMPOTENCY_ENABLED", "1")
    calls = {"n": 0}

    def _run():
        calls["n"] += 1
        return {"ran": True, "reason": "executed", "incident_id": "session_x", "session_id": f"session_{calls['n']}"}

    monkeypatch.setattr("dashboard.backend.app.run_auto_runbook_once", _run)
    client = TestClient(app)
    h = {"X-Idempotency-Key": "same-key"}
    r1 = client.post("/api/debug/incidents/auto-run", headers=h)
    r2 = client.post("/api/debug/incidents/auto-run", headers=h)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert calls["n"] == 1
    assert r1.json()["session_id"] == r2.json()["session_id"]
