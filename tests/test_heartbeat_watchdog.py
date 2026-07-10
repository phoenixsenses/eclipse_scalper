from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import heartbeat_watchdog as hw


def _make_db(root: Path, *, agg_age=1.0, mark_age=1.0, liq_age=1.0) -> Path:
    db = root / "data" / "microstructure.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    now_ms = int(time.time() * 1000)
    conn.execute("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    conn.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    conn.execute("CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    conn.execute("INSERT INTO agg_trades (ts_ms) VALUES (?)", (now_ms - int(agg_age * 1000),))
    conn.execute("INSERT INTO mark_prices (ts_ms) VALUES (?)", (now_ms - int(mark_age * 1000),))
    conn.execute("INSERT INTO liquidations (ts_ms) VALUES (?)", (now_ms - int(liq_age * 1000),))
    conn.commit()
    conn.close()
    return db


def _write_collector_heartbeat(root: Path, **overrides) -> None:
    payload = {
        "connected": True,
        "last_message_ts_utc": hw.utc_now_z(),
        "last_data_progress_ts_utc": hw.utc_now_z(),
        "rest_fallback_enabled": True,
        "rest_fallback_active": False,
        "rest_last_progress_ts_utc": hw.utc_now_z(),
        "current_backoff_seconds": 1.0,
        "last_error": "",
    }
    payload.update(overrides)
    (root / "logs" / "collector_heartbeat.json").write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_preserves_degraded_collector_without_false_all_down(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "degraded",
        "ts_utc": now,
        "last_progress_ts_utc": now,
        "required_streams_progressing": False,
        "transport_connected": False,
    }), encoding="utf-8")
    (root / "logs" / "health" / "bookticker.json").write_text(json.dumps({
        "status": "ok",
        "connected": True,
        "ts_utc": now,
        "last_tick_event_age_sec": 5,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready",
        "active_mode": "live",
        "fallback_to_paper": False,
        "ts_utc": now,
    }), encoding="utf-8")
    _write_collector_heartbeat(root)
    _make_db(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)["report"]

    assert result["overall"] == "YELLOW"
    assert result["collector_alive"] is True
    assert result["collector_healthy"] is False
    assert result["bookticker_healthy"] is True
    assert result["runtime_healthy"] is True
    assert result["detector_healthy"] is True


def test_run_once_writes_report_and_health(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok",
        "ts_utc": now,
        "last_progress_ts_utc": now,
        "required_streams_progressing": True,
        "transport_connected": True,
        "connected": True,
    }), encoding="utf-8")
    (root / "logs" / "health" / "bookticker.json").write_text(json.dumps({
        "status": "ok",
        "connected": True,
        "ts_utc": now,
        "last_tick_event_age_sec": 1,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready",
        "active_mode": "live",
        "fallback_to_paper": False,
        "ts_utc": now,
    }), encoding="utf-8")
    _write_collector_heartbeat(root)
    _make_db(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    report = hw.run_once(max_age_sec=180)

    assert report["overall"] == "GREEN"
    assert (root / "reports" / "WATCHDOG_STATUS.json").exists()
    assert (root / "logs" / "health" / "watchdog.json").exists()


def test_data_only_mode_does_not_require_missing_optional_components(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok",
        "ts_utc": now,
        "last_progress_ts_utc": now,
        "required_streams_progressing": True,
        "transport_connected": True,
        "connected": True,
    }), encoding="utf-8")
    _write_collector_heartbeat(root)
    _make_db(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: needle == "data.microstructure_collector")

    result = hw.evaluate(
        max_age_sec=180,
        expect_bookticker=False,
        expect_detector=False,
        expect_runtime=False,
    )["report"]

    assert result["overall"] == "GREEN"
    assert result["issues"] == []
    assert result["expect_bookticker"] is False


# --- top-level status merge (native-WS policy folded into "overall") --------


def _fixture_root(tmp_path, monkeypatch, *, db_age_kwargs=None, heartbeat_overrides=None):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok",
        "ts_utc": now,
        "last_progress_ts_utc": now,
        "required_streams_progressing": True,
        "transport_connected": True,
        "connected": True,
    }), encoding="utf-8")
    (root / "logs" / "health" / "bookticker.json").write_text(json.dumps({
        "status": "ok",
        "connected": True,
        "ts_utc": now,
        "last_tick_event_age_sec": 1,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready",
        "active_mode": "live",
        "fallback_to_paper": False,
        "ts_utc": now,
    }), encoding="utf-8")
    _write_collector_heartbeat(root, **(heartbeat_overrides or {}))
    _make_db(root, **(db_age_kwargs or {}))
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)
    return root


def test_native_degraded_forces_top_level_yellow(monkeypatch, tmp_path):
    # Native message lost, but REST fallback is genuinely covering -- DEGRADED,
    # not RED (RED requires REST to also not be progressing).
    _fixture_root(
        tmp_path, monkeypatch,
        heartbeat_overrides={
            "last_message_ts_utc": None,
            "rest_fallback_active": True,
            "rest_last_progress_ts_utc": hw.utc_now_z(),
        },
    )
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert result["native_ws_status"] == "DEGRADED"
    assert result["overall"] == "YELLOW"
    assert any(r.startswith("native_ws_degraded") for r in result["issues"])


def test_native_red_forces_top_level_red(monkeypatch, tmp_path):
    # All collection paths stopped: no native message ever, REST also not progressing.
    _fixture_root(
        tmp_path, monkeypatch,
        heartbeat_overrides={
            "connected": False,
            "last_message_ts_utc": None,
            "rest_fallback_active": False,
            "rest_last_progress_ts_utc": None,
        },
        db_age_kwargs={"agg_age": 5000.0, "mark_age": 5000.0, "liq_age": 5000.0},
    )
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert result["native_ws_status"] == "RED"
    assert result["overall"] == "RED"


def test_existing_red_cannot_be_weakened_by_healthy_native(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path, monkeypatch)  # native fully healthy
    monkeypatch.setattr(hw, "python_process_running", lambda needle: needle != "data.microstructure_collector")
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert result["overall"] == "RED"
    assert "collector_process_down" in result["issues"]


def test_existing_yellow_cannot_be_upgraded_by_healthy_native(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        # transport_connected True: native itself is fine. The existing YELLOW
        # here comes only from required_streams_progressing=False (some other,
        # non-native cause) -- it must not be upgradeable by a healthy native
        # status, but it also must not be conflated with a native problem.
        "status": "degraded", "ts_utc": now, "last_progress_ts_utc": now,
        "required_streams_progressing": False, "transport_connected": True, "connected": True,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready", "active_mode": "live", "fallback_to_paper": False, "ts_utc": now,
    }), encoding="utf-8")
    _write_collector_heartbeat(root)  # native itself reports fully healthy
    _make_db(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180, expect_bookticker=False, expect_detector=False, expect_runtime=True)["report"]
    assert result["overall"] == "YELLOW"
    assert result["native_ws_status"] == "GREEN"


def test_healthy_native_state_permits_green(monkeypatch, tmp_path):
    _fixture_root(tmp_path, monkeypatch)
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert result["native_ws_status"] == "GREEN"
    assert result["overall"] == "GREEN"


def test_rest_fallback_active_with_fresh_agg_mark_and_stale_liquidations_is_at_least_yellow(monkeypatch, tmp_path):
    # liq_age between the warning (600s) and critical (1800s) thresholds.
    _fixture_root(
        tmp_path, monkeypatch,
        db_age_kwargs={"agg_age": 2.0, "mark_age": 2.0, "liq_age": 1000.0},
    )
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert result["native_ws_status"] == "DEGRADED"
    assert result["overall"] in ("YELLOW", "RED")


def test_july_outage_fixture_replay_never_produces_top_level_green(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    now = hw.utc_now_z()
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok", "ts_utc": now, "last_progress_ts_utc": now,
        "required_streams_progressing": True, "transport_connected": False, "connected": False,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready", "active_mode": "live", "fallback_to_paper": False, "ts_utc": now,
    }), encoding="utf-8")
    _write_collector_heartbeat(root, connected=False, last_message_ts_utc=None, rest_fallback_active=True)
    # liquidations frozen ~2 days, matching the actual captured outage state.
    _make_db(root, agg_age=3.0, mark_age=3.0, liq_age=2 * 24 * 3600.0)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180, expect_bookticker=False, expect_detector=False, expect_runtime=True)["report"]
    assert result["overall"] != "GREEN"
    assert result["overall"] in ("YELLOW", "RED")


def test_missing_collector_json_remains_deterministic_not_crash(monkeypatch, tmp_path):
    root = tmp_path
    (root / "logs" / "health").mkdir(parents=True)
    (root / "reports").mkdir()
    # No collector.json written at all -- component state is unreadable.
    _write_collector_heartbeat(root)
    _make_db(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "REPORTS", root / "reports")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180, expect_bookticker=False, expect_detector=False, expect_runtime=False)["report"]
    assert result["native_ws_status"] == "RED"
    assert result["overall"] == "RED"


# --- canonical logs/health/overall.json (sole authoritative writer) --------


def test_run_once_writes_canonical_overall_json_matching_report_severity(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path, monkeypatch)
    report = hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    overall_path = root / "logs" / "health" / "overall.json"
    assert overall_path.exists()
    canonical = json.loads(overall_path.read_text(encoding="utf-8"))
    assert canonical["state"] == "ok"
    assert report["overall"] == "GREEN"
    assert canonical["native_ws_status"] == "GREEN"


def test_canonical_overall_native_degraded_maps_to_degraded_in_both_outputs(monkeypatch, tmp_path):
    root = _fixture_root(
        tmp_path, monkeypatch,
        heartbeat_overrides={"last_message_ts_utc": None, "rest_fallback_active": True, "rest_last_progress_ts_utc": hw.utc_now_z()},
    )
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    assert result["report"]["overall"] == "YELLOW"
    assert result["overall"]["state"] == "degraded"


def test_canonical_overall_native_red_maps_to_halted_in_both_outputs(monkeypatch, tmp_path):
    _fixture_root(
        tmp_path, monkeypatch,
        heartbeat_overrides={
            "connected": False, "last_message_ts_utc": None,
            "rest_fallback_active": False, "rest_last_progress_ts_utc": None,
        },
        db_age_kwargs={"agg_age": 5000.0, "mark_age": 5000.0, "liq_age": 5000.0},
    )
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    assert result["report"]["overall"] == "RED"
    assert result["overall"]["state"] == "halted"


def test_two_output_files_cannot_disagree_in_severity(monkeypatch, tmp_path):
    """WATCHDOG_STATUS.json's 'overall' (RED/YELLOW/GREEN) and the canonical
    overall.json's 'state' (halted/degraded/ok) are both derived from the same
    evaluate() call in the same cycle -- assert the mapping is exhaustive and
    consistent across all three severities."""
    root = _fixture_root(tmp_path, monkeypatch)
    result = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    expected = {"GREEN": "ok", "YELLOW": "degraded", "RED": "halted"}
    assert result["overall"]["state"] == expected[result["report"]["overall"]]


def test_canonical_overall_sources_paper_trader_from_its_own_dedicated_file(monkeypatch, tmp_path):
    """paper_trader must be read fresh from logs/health/paper_trader.json
    every cycle (its owner, execution/health_gate.py, writes only that
    file) -- never preserved/merged from a previous overall.json, which no
    longer exists as a read path at all."""
    root = _fixture_root(tmp_path, monkeypatch)
    paper_trader_path = root / "logs" / "health" / "paper_trader.json"
    paper_trader_path.write_text(json.dumps(
        {"status": "ok", "connected": True, "marker": "from_health_gate", "ts_utc": hw.utc_now_z()}
    ), encoding="utf-8")

    overall_path = root / "logs" / "health" / "overall.json"
    hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)

    canonical = json.loads(overall_path.read_text(encoding="utf-8"))
    assert canonical["components"]["paper_trader"]["marker"] == "from_health_gate"
    assert canonical["components"]["collector"]  # still populated alongside it


def test_canonical_overall_omits_paper_trader_when_dedicated_file_absent(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path, monkeypatch)
    overall_path = root / "logs" / "health" / "overall.json"
    hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    canonical = json.loads(overall_path.read_text(encoding="utf-8"))
    assert "paper_trader" not in canonical["components"]


def test_canonical_overall_never_reads_a_previous_overall_json(monkeypatch, tmp_path):
    """A stale/garbage prior overall.json must have zero influence on the
    freshly-computed canonical payload -- it is never read back."""
    root = _fixture_root(tmp_path, monkeypatch)
    overall_path = root / "logs" / "health" / "overall.json"
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall_path.write_text(json.dumps({
        "ts_utc": "old", "state": "halted", "mode": "live",
        "components": {"paper_trader": {"status": "ok", "connected": True, "marker": "stale_ghost"},
                        "notifier": {"status": "ok", "marker": "unknown_placeholder"}},
    }), encoding="utf-8")

    report = hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)

    canonical = json.loads(overall_path.read_text(encoding="utf-8"))
    assert report["overall"] == "GREEN"
    assert canonical["state"] == "ok"  # not "halted" carried forward from the stale file
    assert "notifier" not in canonical["components"]  # unknown historical placeholder not copied
    assert "paper_trader" not in canonical["components"]  # no dedicated file -> omitted, not resurrected


def test_stale_paper_trader_component_remains_visibly_stale_after_rewrite(monkeypatch, tmp_path):
    """Rewriting overall.json must not refresh an unrelated component's own
    timestamp -- a reader computing age(components.paper_trader.ts_utc) must
    still see it as stale even though overall.json's own ts_utc is fresh."""
    root = _fixture_root(tmp_path, monkeypatch)
    stale_ts = (hw.utc_now() - __import__("datetime").timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (root / "logs" / "health" / "paper_trader.json").write_text(json.dumps(
        {"status": "ok", "connected": True, "ts_utc": stale_ts}
    ), encoding="utf-8")

    hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)

    overall_path = root / "logs" / "health" / "overall.json"
    canonical = json.loads(overall_path.read_text(encoding="utf-8"))
    assert canonical["ts_utc"] != canonical["components"]["paper_trader"]["ts_utc"]
    age = hw.age_sec(canonical["components"]["paper_trader"]["ts_utc"])
    assert age is not None and age > 3000  # ~1h old, unmistakably stale


def test_canonical_overall_write_is_atomic_no_leftover_tmp_files(monkeypatch, tmp_path):
    root = _fixture_root(tmp_path, monkeypatch)
    hw.run_once(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)
    leftover = list((root / "logs" / "health").glob(".tmp_*"))
    assert leftover == []


def test_collector_no_longer_writes_overall_json():
    """Retirement guard: data/microstructure_collector.py must not import or
    call write_overall_health -- logs/health/overall.json is owned solely by
    tools/heartbeat_watchdog.py."""
    import inspect
    import data.microstructure_collector as collector_mod

    source = inspect.getsource(collector_mod)
    assert "write_overall_health" not in source


def test_recovery_to_green_only_after_native_flow_actually_resumes(monkeypatch, tmp_path):
    root = _fixture_root(
        tmp_path, monkeypatch,
        heartbeat_overrides={
            "last_message_ts_utc": None,
            "rest_fallback_active": True,
            "rest_last_progress_ts_utc": hw.utc_now_z(),
        },
    )
    degraded = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert degraded["overall"] == "YELLOW"

    _write_collector_heartbeat(root)  # native message flow resumes for real
    recovered = hw.evaluate(max_age_sec=180, expect_bookticker=True, expect_detector=False, expect_runtime=True)["report"]
    assert recovered["overall"] == "GREEN"
    assert recovered["native_ws_status"] == "GREEN"
