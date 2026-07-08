"""BATCH-OPERATOR-HOST-HEALTH-AND-RESTART-READINESS-DASHBOARD-V1 -- focused
tests for the host-health integration into tools/s34_live_chart.py (the
":5050 Eclipse S34 Control" dashboard, the operator's actual daily-use
surface). `s34_live_chart.py` uses sibling-relative imports
(`import s34_current_prediction_card`), so it must be imported with
`tools/` on `sys.path`, matching how the module itself is normally run.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import s34_live_chart as CHART  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_host_health_cache():
    """Each test gets a cold cache so TTL behavior is deterministic."""
    CHART._HOST_HEALTH_CACHE = (0.0, {})
    CHART._HOST_HEALTH_RAM_HISTORY.clear()
    CHART._HOST_HEALTH_COMMIT_HISTORY.clear()
    yield
    CHART._HOST_HEALTH_CACHE = (0.0, {})
    CHART._HOST_HEALTH_RAM_HISTORY.clear()
    CHART._HOST_HEALTH_COMMIT_HISTORY.clear()


def test_host_health_payload_available_and_shaped():
    payload = CHART.host_health_payload()
    assert payload["available"] is True
    assert payload["state"] in {"HOST_RESTART_GREEN", "HOST_RESTART_YELLOW", "HOST_RESTART_RED", "HOST_RESTART_UNKNOWN"}
    assert payload["no_automatic_action"] is True
    assert "recommended_action" in payload
    assert "reasons" in payload and isinstance(payload["reasons"], list)
    assert "observations" in payload
    assert "observation_timestamp" in payload
    assert payload["d_drive_intervention_free_gb"] == 800.0


def test_host_health_payload_cached_within_ttl():
    a = CHART.host_health_payload()
    b = CHART.host_health_payload()
    assert a is b, "second call within TTL must return the cached object, not recompute"


def test_host_health_payload_recomputes_after_ttl_expiry(monkeypatch):
    a = CHART.host_health_payload()
    # Force the cache to look stale without sleeping the full 20s TTL.
    cached_ts, cached_payload = CHART._HOST_HEALTH_CACHE
    CHART._HOST_HEALTH_CACHE = (cached_ts - CHART._HOST_HEALTH_CACHE_TTL_SEC - 1.0, cached_payload)
    b = CHART.host_health_payload()
    assert b is not a


def test_host_health_payload_fails_closed_on_collector_exception(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated observation failure")
    monkeypatch.setattr(CHART, "collect_host_observation", None, raising=False)

    import ami.host_health.observation as OBS
    monkeypatch.setattr(OBS, "collect_host_observation", boom)

    payload = CHART.host_health_payload()
    assert payload["available"] is False
    assert payload["state"] == "HOST_RESTART_UNKNOWN"
    assert "error" in payload


def test_build_payload_includes_host_health():
    payload = CHART.build_payload()
    assert "host_health" in payload
    assert payload["host_health"].get("available") is True


def test_host_health_maintains_bounded_ram_history():
    for _ in range(3):
        CHART.host_health_payload()
        CHART._HOST_HEALTH_CACHE = (0.0, {})  # force recompute each call
    assert len(CHART._HOST_HEALTH_RAM_HISTORY) <= 200
    assert len(CHART._HOST_HEALTH_COMMIT_HISTORY) <= 200


def test_html_contains_host_health_panel_and_renderer():
    html = CHART.HTML
    assert "PC / Host health" in html
    assert 'id="host-health-body"' in html
    assert "renderHostHealth" in html
    assert "renderHostHealth(payload)" in html


def test_html_host_health_panel_never_calls_restart_or_shutdown():
    html = CHART.HTML
    lowered = html.lower()
    for token in ("shutdown.exe", "restart-computer", "stop-computer", "/api/restart", "/api/shutdown"):
        assert token not in lowered


def test_no_automatic_action_flag_always_true_in_payload():
    payload = CHART.host_health_payload()
    assert payload.get("no_automatic_action") is True
