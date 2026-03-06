from __future__ import annotations

from pathlib import Path
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.backend.app import app


def test_debug_actions_list(monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard.backend.app.list_actions",
        lambda: [{"action": "preflight_check", "description": "x", "timeout_sec": 60}],
    )
    client = TestClient(app)
    response = client.get("/api/debug/actions")
    assert response.status_code == 200
    payload = response.json()
    assert payload and payload[0]["action"] == "preflight_check"


def test_debug_run_rejects_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: False)
    client = TestClient(app)
    response = client.post("/api/debug/run", json={"action": "preflight_check"})
    assert response.status_code == 403


def test_debug_run_executes_action(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setenv("DASHBOARD_CONTROL_ROLE", "operator")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "dashboard.backend.app.append_incident_audit",
        lambda payload: seen.update(payload),
    )
    monkeypatch.setattr(
        "dashboard.backend.app.run_action",
        lambda action: {
            "action": action,
            "ok": True,
            "exit_code": 0,
            "duration_sec": 0.12,
            "output": "ok",
            "started_ts": 1.0,
            "ended_ts": 1.12,
        },
    )
    client = TestClient(app)
    response = client.post(
        "/api/debug/run",
        json={"action": "preflight_check"},
        headers={"X-Operator": "alice"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "preflight_check"
    assert seen.get("operator") == "alice"


def test_debug_run_rejects_for_viewer_role(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setenv("DASHBOARD_CONTROL_ROLE", "viewer")
    client = TestClient(app)
    response = client.post("/api/debug/run", json={"action": "preflight_check"})
    assert response.status_code == 403


def test_debug_run_respects_x_role_strict_mode(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setenv("DASHBOARD_CONTROL_ROLE", "admin")
    monkeypatch.setenv("DASHBOARD_STRICT_HEADER_ROLE", "1")
    client = TestClient(app)
    response = client.post(
        "/api/debug/run",
        json={"action": "preflight_check"},
        headers={"X-Role": "viewer"},
    )
    assert response.status_code == 403


def test_debug_run_requires_api_key_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr("dashboard.backend.app.control_enabled", lambda: True)
    monkeypatch.setenv("DASHBOARD_CONTROL_ROLE", "operator")
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret")
    monkeypatch.setattr(
        "dashboard.backend.app.run_action",
        lambda action: {
            "action": action,
            "ok": True,
            "exit_code": 0,
            "duration_sec": 0.1,
            "output": "ok",
            "started_ts": 1.0,
            "ended_ts": 1.1,
        },
    )
    client = TestClient(app)
    r1 = client.post("/api/debug/run", json={"action": "preflight_check"})
    assert r1.status_code == 401
    r2 = client.post(
        "/api/debug/run",
        json={"action": "preflight_check"},
        headers={"X-Api-Key": "secret"},
    )
    assert r2.status_code == 200


def test_security_audit_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "dashboard.backend.app._read_security_audit",
        lambda limit=100: [{"ts": 1.0, "kind": "auth_failed", "path": "/api/debug/run"}],
    )
    client = TestClient(app)
    r = client.get("/api/debug/security-audit")
    assert r.status_code == 200
    assert r.json()[0]["kind"] == "auth_failed"
