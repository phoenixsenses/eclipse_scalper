"""I/O, writer-ownership, DB-safety, bounded-query, evaluation-mode, error-
propagation, and historical-replay tests for
tools/liquidation_silence_detector.py (v2 corrective).

All production-DB reads use data/microstructure.db strictly `mode=ro` and
are scoped to short historical windows via the same bounded, indexed
`ORDER BY ts_ms ... LIMIT 1` pattern the detector itself uses -- never a
full-table scan. Every write in this file targets an isolated tmp/scratch
path; nothing here ever writes to the real logs/health/ directory.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import heartbeat_watchdog as hw
from tools.health_state import write_component_health
from tools.liquidation_silence_detector import (
    DEFAULT_DB_PATH,
    ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH,
    ERR_DB_LOCKED,
    ERR_DB_PERMISSION,
    ERR_DB_SCHEMA_MISMATCH,
    ERR_DB_TABLE_MISSING,
    ERR_HISTORICAL_EVIDENCE_REQUIRED,
    ERR_LIVE_MODE_TIMESTAMP_MISMATCH,
    MODE_HISTORICAL_REPLAY,
    MODE_LIVE,
    _classify_sqlite_error,
    _open_ro,
    discover_tracked_symbols,
    evaluate_once,
    read_collector_component_alive,
    read_control_freshness,
    read_last_liquidation_ts,
    run_once,
)
from tools.liquidation_silence_policy import (
    STATUS_HEALTHY,
    STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
    STATUS_UNKNOWN,
)

REAL_DB_AVAILABLE = DEFAULT_DB_PATH.exists()
requires_real_db = pytest.mark.skipif(not REAL_DB_AVAILABLE, reason="data/microstructure.db not present in this environment")

# Explicit historical-evidence paths (nonexistent, so no present-day live
# file can leak into a historical evaluation). Used by every synthetic /
# replay evaluate_once call so it runs under the safe explicit mode.
_HIST_PATHS = dict(
    evaluation_mode=MODE_HISTORICAL_REPLAY,
    pid_meta_path=Path("/nonexistent/pid_meta.json"),
    overall_path=Path("/nonexistent/overall.json"),
    collector_component_path=Path("/nonexistent/collector.json"),
)


def _iso_to_ts(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp()


# --- symbol universe discovery (incl. normalization / dedup) -----------------

def test_discover_tracked_symbols_from_canonical_config(tmp_path):
    meta = tmp_path / "collector_supervisor.json"
    # Deliberately includes a UTF-8 BOM, matching the real
    # logs/pids/collector_supervisor.json this reads from in production.
    meta.write_bytes(b"\xef\xbb\xbf" + json.dumps({"symbols": "BTCUSDT,ETHUSDT,SOLUSDT"}).encode("utf-8"))
    result = discover_tracked_symbols(meta)
    assert result["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert result["source"] == "canonical_runtime_config"


def test_discover_tracked_symbols_deduplicates_and_normalizes(tmp_path):
    meta = tmp_path / "collector_supervisor.json"
    meta.write_text(json.dumps({"symbols": " btcusdt , BTCUSDT ,ethusdt,"}), encoding="utf-8")
    result = discover_tracked_symbols(meta)
    assert result["symbols"] == ["BTCUSDT", "ETHUSDT"]


def test_discover_tracked_symbols_falls_back_when_config_missing(tmp_path):
    result = discover_tracked_symbols(tmp_path / "does_not_exist.json")
    assert result["source"] == "fallback_default"
    assert result["symbols"]


def test_discover_tracked_symbols_falls_back_when_config_corrupt(tmp_path):
    meta = tmp_path / "collector_supervisor.json"
    meta.write_text("{not valid json", encoding="utf-8")
    result = discover_tracked_symbols(meta)
    assert result["source"] == "fallback_default"


def test_discover_tracked_symbols_respects_arbitrary_universe(tmp_path):
    meta = tmp_path / "collector_supervisor.json"
    meta.write_text(json.dumps({"symbols": "ETHUSDT"}), encoding="utf-8")
    result = discover_tracked_symbols(meta)
    assert result["symbols"] == ["ETHUSDT"]


# --- writer ownership / atomicity -------------------------------------------

def test_write_creates_only_its_own_component_file(tmp_path):
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["liquidation_silence.json"]


def test_never_writes_overall_json(tmp_path):
    existing_overall = tmp_path / "overall.json"
    existing_overall.write_text(json.dumps({"state": "ok", "sentinel": "untouched"}), encoding="utf-8")
    before = existing_overall.read_bytes()
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    after = existing_overall.read_bytes()
    assert before == after
    assert (tmp_path / "liquidation_silence.json").exists()


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []


def test_write_overwrites_corrupt_predecessor(tmp_path):
    target = tmp_path / "liquidation_silence.json"
    target.write_bytes(b"\x00\x01not-json-garbage{{{")
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["component"] == "liquidation_silence"


def test_component_output_carries_ts_utc_for_staleness_grading(tmp_path):
    now_ts = 1_800_000_000.0
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=now_ts, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    reloaded = json.loads((tmp_path / "liquidation_silence.json").read_text(encoding="utf-8"))
    assert "ts_utc" in reloaded
    later = now_ts + 1000.0
    fresh = hw.component_fresh(reloaded, max_age_sec=1500, now=dt.datetime.fromtimestamp(later, dt.timezone.utc))
    stale = hw.component_fresh(reloaded, max_age_sec=500, now=dt.datetime.fromtimestamp(later, dt.timezone.utc))
    assert fresh is True
    assert stale is False


# --- deterministic serialization --------------------------------------------

def test_run_once_dry_run_does_not_write(tmp_path):
    payload = run_once(db_path=tmp_path / "nonexistent.db", health_root=tmp_path, now_ts=1_800_000_000.0, dry_run=True, **_HIST_PATHS)
    assert payload["status"] in {STATUS_HEALTHY, STATUS_UNKNOWN}
    assert not (tmp_path / "liquidation_silence.json").exists()


def test_two_runs_same_inputs_produce_byte_identical_component_payload(tmp_path):
    p1 = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    p2 = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    p1.pop("detector_runtime_sec")
    p2.pop("detector_runtime_sec")
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# --- evaluation-mode contamination safety (CORRECTION 3) ---------------------

def test_live_mode_with_wallclock_now_ts_is_allowed(tmp_path):
    # Real now_ts + LIVE mode + default paths must NOT be rejected.
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=time.time(), evaluation_mode=MODE_LIVE)
    assert payload["evaluation_mode"] == MODE_LIVE
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_LIVE_MODE_TIMESTAMP_MISMATCH not in codes


def test_live_mode_with_historical_now_ts_and_default_paths_is_rejected(tmp_path):
    # Historical now_ts under LIVE mode -> fail-visible UNKNOWN, live default
    # files never consulted.
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_700_000_000.0, evaluation_mode=MODE_LIVE)
    assert payload["status"] == STATUS_UNKNOWN
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_LIVE_MODE_TIMESTAMP_MISMATCH in codes


def test_historical_mode_with_default_live_paths_is_rejected(tmp_path):
    # HISTORICAL_REPLAY may not read the live default health/pid files; this
    # is the structural guard that stops present-day evidence contaminating a
    # historical (April/May) replay.
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_700_000_000.0, evaluation_mode=MODE_HISTORICAL_REPLAY)
    assert payload["status"] == STATUS_UNKNOWN
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_HISTORICAL_EVIDENCE_REQUIRED in codes


def test_historical_mode_with_explicit_fixture_paths_is_allowed(tmp_path):
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_700_000_000.0, **_HIST_PATHS)
    assert payload["evaluation_mode"] == MODE_HISTORICAL_REPLAY
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_HISTORICAL_EVIDENCE_REQUIRED not in codes


def test_historical_replay_component_ts_after_now_ts_is_temporal_mismatch(tmp_path):
    # Explicit collector fixture whose ts_utc is AFTER the historical now_ts
    # must be rejected as a temporal mismatch, not counted as fresh evidence.
    now_ts = _iso_to_ts("2026-05-15T12:00:00")
    collector = tmp_path / "collector.json"
    collector.write_text(json.dumps({"ts_utc": "2026-07-11T00:00:00Z"}), encoding="utf-8")
    alive, err = read_collector_component_alive(collector, now_ts=now_ts)
    assert alive is None
    assert err == ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH


def test_present_day_collector_file_does_not_influence_historical_replay(tmp_path):
    # End-to-end: a present-day collector.json cannot sneak into an April/May
    # replay because (a) default-path reads are refused in historical mode,
    # and (b) an explicit future-dated fixture is temporally rejected.
    now_ts = _iso_to_ts("2026-05-15T12:00:00")
    collector = tmp_path / "collector.json"
    collector.write_text(json.dumps({"ts_utc": "2026-07-11T00:00:00Z", "state": "ok"}), encoding="utf-8")
    payload = evaluate_once(
        db_path=tmp_path / "nonexistent.db",
        pid_meta_path=Path("/nonexistent/pid.json"),
        overall_path=Path("/nonexistent/overall.json"),
        collector_component_path=collector,
        now_ts=now_ts,
        evaluation_mode=MODE_HISTORICAL_REPLAY,
    )
    assert payload["collector_component_alive"] is None
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_CONTROL_COMPONENT_TEMPORAL_MISMATCH in codes


# --- controlled activation: wired into OPTIONAL_COMPONENT_FILES + folded ----
# into top-level severity via compose_with_overall_severity (see
# tools/heartbeat_watchdog.py's evaluate(), the block right after the
# native-WS-aware merge). The two tests below lock in the two halves of
# that intentional behavior: (1) with the detector's own file absent, the
# mere presence of "liquidation_silence" in OPTIONAL_COMPONENT_FILES must
# still be a pure no-op (severity defaults to GREEN, compose is inert) --
# this preserves the original safety invariant this test module locked in
# before activation, just no longer via "the key isn't registered at all".
# (2) once the detector's file is present with a non-GREEN severity, it now
# genuinely participates in the canonical overall/state, which is the new,
# deliberate behavior this activation batch introduces.

def _write_healthy_control_components(root: Path) -> None:
    """Populates the minimal set of control-plane files/DB tables that make
    tools.heartbeat_watchdog.evaluate() resolve to a GREEN baseline on its
    own, independent of liquidation_silence, so that any deviation from
    GREEN in the tests below is attributable solely to the
    liquidation_silence fold under test."""
    now = hw.utc_now_z()
    log_health = root / "logs" / "health"
    log_health.mkdir(parents=True, exist_ok=True)
    (log_health / "collector.json").write_text(json.dumps({
        "status": "ok", "ts_utc": now, "last_progress_ts_utc": now,
        "required_streams_progressing": True, "transport_connected": True,
        "connected": True,
    }), encoding="utf-8")
    (log_health / "bookticker.json").write_text(json.dumps({
        "status": "ok", "connected": True, "ts_utc": now, "last_tick_event_age_sec": 1,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready", "active_mode": "paper", "fallback_to_paper": False, "ts_utc": now,
    }), encoding="utf-8")
    (root / "logs" / "collector_heartbeat.json").write_text(json.dumps({
        "connected": True, "last_message_ts_utc": now, "last_data_progress_ts_utc": now,
        "rest_fallback_enabled": True, "rest_fallback_active": False,
        "rest_last_progress_ts_utc": now, "current_backoff_seconds": 1.0, "last_error": "",
    }), encoding="utf-8")
    db = root / "data" / "microstructure.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    now_ms = int(time.time() * 1000)
    con.execute("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("INSERT INTO agg_trades (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.execute("INSERT INTO mark_prices (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.execute("INSERT INTO liquidations (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.commit()
    con.close()


def test_liquidation_silence_now_registered_in_optional_component_files():
    """Locks in that this activation batch did in fact add
    "liquidation_silence" to OPTIONAL_COMPONENT_FILES -- the precondition
    both tests below depend on."""
    assert "liquidation_silence" in hw.OPTIONAL_COMPONENT_FILES
    assert hw.OPTIONAL_COMPONENT_FILES["liquidation_silence"] == "liquidation_silence.json"


def test_absent_component_file_still_has_zero_effect_on_canonical_overall(monkeypatch, tmp_path):
    """Even though "liquidation_silence" is now a registered optional
    component, a cycle where the detector has never run (its dedicated
    logs/health/liquidation_silence.json is simply absent) must still be a
    complete no-op: no entry in overall.json's components[], and no
    influence at all on the composed top-level GREEN/ok state. This is the
    half of the pre-activation invariant that must survive activation."""
    root = tmp_path
    _write_healthy_control_components(root)
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert not (root / "logs" / "health" / "liquidation_silence.json").exists()
    assert result["report"]["overall"] == "GREEN"
    assert "liquidation_silence" not in result["overall"]["components"]
    assert result["overall"]["state"] == "ok"


def test_present_red_component_file_now_forces_canonical_overall_red(monkeypatch, tmp_path):
    """The new, intentional half of activation: once the detector's own
    file is present with severity RED, tools.heartbeat_watchdog.evaluate()'s
    compose_with_overall_severity fold must genuinely raise the canonical
    overall/state to RED/halted -- even though every other control-plane
    input is healthy -- and surface a liquidation_silence_red reason code."""
    root = tmp_path
    _write_healthy_control_components(root)
    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["liquidation_transport_outage"]},
        root=root / "logs" / "health",
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] == "RED"
    assert result["overall"]["state"] == "halted"
    assert any(str(r).startswith("liquidation_silence_red") for r in result["report"]["issues"])
    assert result["overall"]["components"]["liquidation_silence"]["severity"] == "RED"


@pytest.mark.parametrize("severity_in", ["YELLOW", "UNKNOWN"])
def test_present_yellow_component_file_raises_canonical_overall_to_yellow(monkeypatch, tmp_path, severity_in):
    """Symmetric coverage for the fold's non-critical branch, which the RED
    test above does not exercise: per compose_with_overall_severity's own
    rank map (_OVERALL_RANK in tools/liquidation_silence_policy.py), YELLOW
    and UNKNOWN both rank=1 and must raise an otherwise-GREEN canonical
    overall to YELLOW/"degraded" (never RED/"halted") and surface a
    liquidation_silence_<severity> warning reason code -- mirroring the
    sibling native_ws fold's DEGRADED/YELLOW integration coverage in
    tests/test_heartbeat_watchdog.py (see
    test_canonical_overall_native_degraded_maps_to_degraded_in_both_outputs,
    which asserts the same result["overall"]["state"] == "degraded" string).
    This locks in the equivalent depth for the liquidation_silence fold."""
    root = tmp_path
    _write_healthy_control_components(root)
    write_component_health(
        "liquidation_silence",
        {"severity": severity_in, "reason_codes": ["liquidation_silence_test_reason"]},
        root=root / "logs" / "health",
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] == "YELLOW"
    assert result["overall"]["state"] == "degraded"
    assert any(
        str(r).startswith(f"liquidation_silence_{severity_in.lower()}")
        for r in result["report"]["issues"]
    )
    assert result["overall"]["components"]["liquidation_silence"]["severity"] == severity_in


# --- freshness gate (scheduling-readiness corrective, 2026-07-11) -----------
# A future scheduled/periodic wrapper (see
# reports/research/s34/LIQUIDATION_SILENCE_DETECTOR_PERIODIC_SCHEDULING_DESIGN.md
# Finding 1) can die or stall; without a staleness gate on this specific
# fold, a frozen severity value would be represented as current forever.
# These tests lock in hw.LIQUIDATION_SILENCE_MAX_AGE_SEC-gated behavior,
# reusing the exact same hw.component_fresh() pattern already governing
# collector/bookticker/runtime -- no new freshness model introduced.


def test_watchdog_cadence_constant_is_the_scheduler_wrapper_constant_not_a_copy():
    """CORRECTIVE (independent re-review finding F2): an earlier version
    declared hw.LIQUIDATION_SILENCE_SCHEDULER_CADENCE_SEC=300 as its own
    independent literal, duplicating tools/liquidation_silence_scheduler.py's
    DEFAULT_CADENCE_SEC=300 with no enforced link between them -- either
    could be changed alone and silently desynchronize the freshness budget
    from the cadence it's supposed to be derived from. hw now imports the
    scheduler module's constant directly (an alias, not a copy), so this is
    the SAME object, not merely an equal value -- `is`, not just `==`,
    proving structural impossibility of drift rather than just its current
    absence."""
    from tools import liquidation_silence_scheduler as sched

    assert hw.LIQUIDATION_SILENCE_SCHEDULER_CADENCE_SEC is sched.DEFAULT_CADENCE_SEC
    assert hw.LIQUIDATION_SILENCE_MAX_AGE_SEC == sched.DEFAULT_CADENCE_SEC * 3


def _iso_offset(seconds: float) -> str:
    """A ts_utc string `seconds` in the past relative to real wall-clock
    now, in the exact format hw.utc_now_z()/detector payloads use."""
    return (hw.utc_now() - dt.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_fresh_liquidation_silence_artifact_is_folded_normally(monkeypatch, tmp_path):
    """Case 1: an artifact well inside the freshness budget is folded exactly
    as before this correction -- its own severity/reason_codes pass through
    verbatim."""
    root = tmp_path
    _write_healthy_control_components(root)
    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["liquidation_transport_outage"], "ts_utc": _iso_offset(10)},
        root=root / "logs" / "health",
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] == "RED"
    assert any(str(r).startswith("liquidation_silence_red") for r in result["report"]["issues"])
    assert result["overall"]["components"]["liquidation_silence"]["severity"] == "RED"


def test_artifact_older_than_freshness_threshold_is_not_folded_as_current(monkeypatch, tmp_path):
    """Case 2: the central corrective behavior. A RED artifact written well
    past hw.LIQUIDATION_SILENCE_MAX_AGE_SEC must NOT force top-level RED --
    it is treated as UNKNOWN (stale), which raises to YELLOW/degraded, never
    RED/halted and never a false GREEN."""
    root = tmp_path
    _write_healthy_control_components(root)
    stale_age = hw.LIQUIDATION_SILENCE_MAX_AGE_SEC + 60
    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["liquidation_transport_outage"], "ts_utc": _iso_offset(stale_age)},
        root=root / "logs" / "health",
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] != "RED"
    assert result["overall"]["state"] != "halted"
    assert result["report"]["overall"] == "YELLOW"
    assert result["overall"]["state"] == "degraded"
    assert any(str(r) == "liquidation_silence_unknown:STALE_ARTIFACT" for r in result["report"]["issues"])


def test_missing_evaluated_at_utc_is_handled_safely(monkeypatch, tmp_path):
    """Case 3: an artifact present but with no ts_utc field at all (a
    malformed writer, or a partially-written file) must not crash and must
    not be trusted as fresh -- treated as stale/UNKNOWN, same as an
    old-but-present timestamp."""
    root = tmp_path
    _write_healthy_control_components(root)
    log_health = root / "logs" / "health"
    log_health.mkdir(parents=True, exist_ok=True)
    (log_health / "liquidation_silence.json").write_text(
        json.dumps({"severity": "RED", "reason_codes": ["x"]}), encoding="utf-8"
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", log_health)
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] == "YELLOW"
    assert result["overall"]["state"] == "degraded"
    assert any(str(r) == "liquidation_silence_unknown:STALE_ARTIFACT" for r in result["report"]["issues"])


def test_malformed_evaluated_at_utc_is_handled_safely(monkeypatch, tmp_path):
    """Case 4: an unparseable ts_utc string must not crash evaluate() and
    must not be trusted as fresh."""
    root = tmp_path
    _write_healthy_control_components(root)
    log_health = root / "logs" / "health"
    log_health.mkdir(parents=True, exist_ok=True)
    (log_health / "liquidation_silence.json").write_text(
        json.dumps({"severity": "RED", "reason_codes": ["x"], "ts_utc": "not-a-timestamp"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", log_health)
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert result["report"]["overall"] == "YELLOW"
    assert result["overall"]["state"] == "degraded"
    assert any(str(r) == "liquidation_silence_unknown:STALE_ARTIFACT" for r in result["report"]["issues"])


def test_freshness_boundary_is_deterministic(monkeypatch, tmp_path):
    """Case 5: comfortably inside hw.LIQUIDATION_SILENCE_MAX_AGE_SEC is
    fresh; comfortably past it is stale (a small margin on both sides of the
    literal boundary avoids sub-second flakiness from the gap between this
    test computing ts_utc and evaluate()'s own internal utc_now() call --
    component_fresh()'s own boundary itself is exactly `<=`, unchanged)."""
    root = tmp_path
    _write_healthy_control_components(root)
    log_health = root / "logs" / "health"

    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["x"], "ts_utc": _iso_offset(hw.LIQUIDATION_SILENCE_MAX_AGE_SEC - 2)},
        root=log_health,
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", log_health)
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)
    inside_budget = hw.evaluate(max_age_sec=180)
    assert inside_budget["report"]["overall"] == "RED"

    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["x"], "ts_utc": _iso_offset(hw.LIQUIDATION_SILENCE_MAX_AGE_SEC + 2)},
        root=log_health,
    )
    past_budget = hw.evaluate(max_age_sec=180)
    assert past_budget["report"]["overall"] != "RED"
    assert any(str(r) == "liquidation_silence_unknown:STALE_ARTIFACT" for r in past_budget["report"]["issues"])


def _write_healthy_control_components_at(root: Path, now_dt: "dt.datetime") -> None:
    """Same shape as _write_healthy_control_components, but every
    freshness-bearing field (JSON ts_utc AND DB row ts_ms) is anchored to an
    explicit `now_dt` rather than real wall-clock time, and DB setup is
    idempotent (CREATE TABLE IF NOT EXISTS + DELETE + reinsert) -- used only
    by test_fresh_output_becomes_stale_as_time_advances so a second call at
    a later synthetic `now_dt` can re-freshen every OTHER component in
    lockstep with a monkeypatched hw.utc_now, without colliding on a second
    sqlite CREATE TABLE call."""
    ts = now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    log_health = root / "logs" / "health"
    log_health.mkdir(parents=True, exist_ok=True)
    (log_health / "collector.json").write_text(json.dumps({
        "status": "ok", "ts_utc": ts, "last_progress_ts_utc": ts,
        "required_streams_progressing": True, "transport_connected": True,
        "connected": True,
    }), encoding="utf-8")
    (log_health / "bookticker.json").write_text(json.dumps({
        "status": "ok", "connected": True, "ts_utc": ts, "last_tick_event_age_sec": 1,
    }), encoding="utf-8")
    (root / "logs" / "runtime_launcher_status.json").write_text(json.dumps({
        "status": "ready", "active_mode": "paper", "fallback_to_paper": False, "ts_utc": ts,
    }), encoding="utf-8")
    (root / "logs" / "collector_heartbeat.json").write_text(json.dumps({
        "connected": True, "last_message_ts_utc": ts, "last_data_progress_ts_utc": ts,
        "rest_fallback_enabled": True, "rest_fallback_active": False,
        "rest_last_progress_ts_utc": ts, "current_backoff_seconds": 1.0, "last_error": "",
    }), encoding="utf-8")
    db = root / "data" / "microstructure.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    now_ms = int(now_dt.timestamp() * 1000)
    con.execute("CREATE TABLE IF NOT EXISTS agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("CREATE TABLE IF NOT EXISTS liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    con.execute("DELETE FROM agg_trades")
    con.execute("DELETE FROM mark_prices")
    con.execute("DELETE FROM liquidations")
    con.execute("INSERT INTO agg_trades (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.execute("INSERT INTO mark_prices (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.execute("INSERT INTO liquidations (ts_ms) VALUES (?)", (now_ms - 1000,))
    con.commit()
    con.close()


def test_fresh_output_becomes_stale_as_time_advances(monkeypatch, tmp_path):
    """Case 6: the SAME liquidation_silence artifact (fixed evaluated_at_utc,
    never rewritten again), evaluated at two different wall-clock instants --
    the first inside the freshness budget, the second past it -- transitions
    from folded-as-current to folded-as-stale. Models a scheduler wrapper
    that died silently: the file it last wrote does not change, but the
    world around it keeps moving. Every OTHER control-plane file (and the DB
    control-stream rows) is re-stamped fresh at each instant (simulating
    collector/bookticker/runtime/native-WS staying genuinely alive and
    continuously refreshed, same as real operation) so this isolates the
    liquidation_silence fold's own freshness gate from the unrelated
    component_fresh() call sites the other components already use --
    otherwise a global clock jump this large would independently push
    native_ws/collector/bookticker into their own staleness/RED,
    confounding the assertion."""
    root = tmp_path
    log_health = root / "logs" / "health"
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", log_health)
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    t0 = dt.datetime(2026, 7, 11, 12, 0, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(hw, "utc_now", lambda: t0)
    _write_healthy_control_components_at(root, t0)
    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["x"], "ts_utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ")},
        root=log_health,
    )
    still_fresh = hw.evaluate(max_age_sec=180)
    assert still_fresh["report"]["overall"] == "RED"

    t1 = t0 + dt.timedelta(seconds=hw.LIQUIDATION_SILENCE_MAX_AGE_SEC + 30)
    monkeypatch.setattr(hw, "utc_now", lambda: t1)
    _write_healthy_control_components_at(root, t1)  # re-freshens collector/bookticker/runtime/DB at t1
    # liquidation_silence.json intentionally NOT rewritten -- still stamped t0.
    now_stale = hw.evaluate(max_age_sec=180)
    assert now_stale["report"]["overall"] != "RED"
    assert any(str(r) == "liquidation_silence_unknown:STALE_ARTIFACT" for r in now_stale["report"]["issues"])


def test_collector_bookticker_runtime_component_fresh_unaffected(monkeypatch, tmp_path):
    """Case 7: this correction touches only the liquidation_silence fold's
    freshness gate; collector/bookticker/runtime's existing
    component_fresh(payload, max_age_sec, now) call sites and their
    max_age_sec parameter are untouched -- a stale collector.json must still
    behave exactly as before (collector_degraded), independent of
    liquidation_silence, which stays absent -> GREEN no-op (the preserved
    absent-file invariant)."""
    root = tmp_path
    _write_healthy_control_components(root)
    stale_collector_ts = _iso_offset(400)
    (root / "logs" / "health" / "collector.json").write_text(json.dumps({
        "status": "ok", "ts_utc": stale_collector_ts, "last_progress_ts_utc": stale_collector_ts,
        "required_streams_progressing": True, "transport_connected": True, "connected": True,
    }), encoding="utf-8")
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", root / "logs" / "health")
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)

    result = hw.evaluate(max_age_sec=180)

    assert not (root / "logs" / "health" / "liquidation_silence.json").exists()
    assert "collector_degraded" in result["report"]["issues"]
    assert "liquidation_silence" not in result["overall"]["components"]


def test_fresh_warning_and_critical_semantics_unchanged(monkeypatch, tmp_path):
    """Case 8: a FRESH detector artifact's WARNING(YELLOW)/CRITICAL(RED)
    verdicts fold exactly as before this correction -- the freshness gate
    only changes behavior for STALE artifacts, never for fresh ones,
    preserving the accepted §107-109 detector severity semantics."""
    root = tmp_path
    _write_healthy_control_components(root)
    log_health = root / "logs" / "health"

    write_component_health(
        "liquidation_silence",
        {"severity": "YELLOW", "reason_codes": ["ALL_SYMBOL_SILENCE_BEYOND_WARNING"], "ts_utc": _iso_offset(5)},
        root=log_health,
    )
    monkeypatch.setattr(hw, "ROOT", root)
    monkeypatch.setattr(hw, "LOG_HEALTH", log_health)
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)
    warning_result = hw.evaluate(max_age_sec=180)
    assert warning_result["report"]["overall"] == "YELLOW"
    assert any(
        str(r).startswith("liquidation_silence_yellow:ALL_SYMBOL_SILENCE_BEYOND_WARNING")
        for r in warning_result["report"]["issues"]
    )

    write_component_health(
        "liquidation_silence",
        {"severity": "RED", "reason_codes": ["ALL_SYMBOL_SILENCE_BEYOND_CRITICAL"], "ts_utc": _iso_offset(5)},
        root=log_health,
    )
    critical_result = hw.evaluate(max_age_sec=180)
    assert critical_result["report"]["overall"] == "RED"
    assert any(
        str(r).startswith("liquidation_silence_red:ALL_SYMBOL_SILENCE_BEYOND_CRITICAL")
        for r in critical_result["report"]["issues"]
    )


# --- read-only / no-mutation guarantees --------------------------------------

def _make_fixture_db(path: Path) -> None:
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
        "symbol TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL, quantity REAL NOT NULL, "
        "notional REAL NOT NULL, trade_time_ms INTEGER NOT NULL)"
    )
    cur.execute("CREATE INDEX idx_liq_ts ON liquidations(ts_ms)")
    cur.execute("CREATE INDEX idx_liq_symbol_ts ON liquidations(symbol, ts_ms)")
    cur.execute(
        "CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
        "symbol TEXT NOT NULL, mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)"
    )
    cur.execute("CREATE INDEX idx_mark_ts ON mark_prices(ts_ms)")
    cur.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    cur.execute(
        "CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, "
        "symbol TEXT NOT NULL, price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL, "
        "is_buyer_maker INTEGER NOT NULL)"
    )
    cur.execute("CREATE INDEX idx_trade_ts ON agg_trades(ts_ms)")
    cur.execute("CREATE INDEX idx_trade_symbol_ts ON agg_trades(symbol, ts_ms)")

    base_ms = 1_800_000_000_000
    rows = []
    # 50k unrelated-symbol rows (proves bounded queries never scan these)
    for i in range(50_000):
        rows.append((base_ms - i * 1000, f"UNRELATED{i % 400}USDT", "SELL", 1.0, 1.0, 1.0, base_ms - i * 1000))
    # a handful of tracked-symbol rows, most recent last
    for i, sym in enumerate(("BTCUSDT", "ETHUSDT", "SOLUSDT")):
        rows.append((base_ms - (5 - i) * 1000, sym, "SELL", 1.0, 1.0, 1.0, base_ms))
    cur.executemany(
        "INSERT INTO liquidations (ts_ms, symbol, side, price, quantity, notional, trade_time_ms) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    cur.executemany(
        "INSERT INTO mark_prices (ts_ms, symbol, mark_price) VALUES (?,?,?)",
        [(base_ms - i, "BTCUSDT", 100.0) for i in range(0, 3000, 3)],
    )
    cur.executemany(
        "INSERT INTO agg_trades (ts_ms, symbol, price, quantity, notional, is_buyer_maker) VALUES (?,?,?,?,?,?)",
        [(base_ms - i, "BTCUSDT", 100.0, 1.0, 100.0, 0) for i in range(0, 3000, 3)],
    )
    con.commit()
    con.close()


def test_production_db_connection_uses_read_only_mode_and_rejects_writes(tmp_path):
    db_path = tmp_path / "fixture.db"
    _make_fixture_db(db_path)
    conn = _open_ro(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO liquidations (ts_ms, symbol, side, price, quantity, notional, trade_time_ms) VALUES (0,'X','SELL',1,1,1,0)")
    finally:
        conn.close()


def test_evaluation_never_mutates_the_fixture_database(tmp_path):
    db_path = tmp_path / "fixture.db"
    _make_fixture_db(db_path)
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    evaluate_once(db_path=db_path, now_ts=1_800_000_010.0, **_HIST_PATHS)
    read_last_liquidation_ts(db_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    read_control_freshness(db_path, now_ts=1_800_000_010.0)
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert before == after


def test_healthy_fixture_evaluation_has_null_error_field(tmp_path):
    db_path = tmp_path / "fixture.db"
    _make_fixture_db(db_path)
    payload = evaluate_once(db_path=db_path, now_ts=1_800_000_010.0, **_HIST_PATHS)
    assert payload["status"] == STATUS_HEALTHY
    assert payload["error"] is None
    assert payload["complete_symbol_evidence"] is True


# --- structured error propagation (CORRECTION 6) -----------------------------

def test_missing_db_table_surfaces_error_code(tmp_path):
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE unrelated (x INTEGER)")
    con.commit()
    con.close()
    payload = evaluate_once(db_path=db_path, now_ts=1_800_000_010.0, **_HIST_PATHS)
    codes = [e["code"] for e in (payload["error"] or [])]
    assert ERR_DB_TABLE_MISSING in codes
    assert payload["status"] == STATUS_UNKNOWN


def test_schema_mismatch_surfaces_error_code(tmp_path):
    db_path = tmp_path / "badschema.db"
    con = sqlite3.connect(str(db_path))
    # liquidations table exists but lacks the ts_ms column
    con.execute("CREATE TABLE liquidations (symbol TEXT, notional REAL)")
    con.execute("CREATE TABLE mark_prices (ts_ms INTEGER, symbol TEXT)")
    con.execute("CREATE TABLE agg_trades (ts_ms INTEGER, symbol TEXT)")
    con.commit()
    con.close()
    liq, errors = read_last_liquidation_ts(db_path, ["BTCUSDT"], now_ts=1_800_000_010.0)
    codes = [e["code"] for e in errors]
    assert ERR_DB_SCHEMA_MISMATCH in codes


def test_db_locked_surfaces_error_code(tmp_path):
    db_path = tmp_path / "locked.db"
    _make_fixture_db(db_path)
    writer = sqlite3.connect(str(db_path), timeout=1)
    try:
        writer.execute("BEGIN EXCLUSIVE")
        writer.execute("INSERT INTO liquidations (ts_ms, symbol, side, price, quantity, notional, trade_time_ms) VALUES (1,'Z','SELL',1,1,1,1)")
        # reader (mode=ro) cannot acquire a shared lock -> DB_LOCKED after busy timeout
        _liq, errors = read_last_liquidation_ts(db_path, ["BTCUSDT"], now_ts=1_800_000_010.0)
        codes = [e["code"] for e in errors]
        assert ERR_DB_LOCKED in codes
    finally:
        writer.rollback()
        writer.close()


def test_classify_permission_error_maps_to_permission_code():
    # A permission-style OperationalError message classifies deterministically.
    exc = sqlite3.OperationalError("unable to open database file")
    assert _classify_sqlite_error(exc) == ERR_DB_PERMISSION


def test_output_write_failure_is_not_silently_swallowed(tmp_path):
    # health_root pointed at a regular file -> the component write must raise,
    # not silently no-op (output permission/write failure surfaces).
    not_a_dir = tmp_path / "blocker"
    not_a_dir.write_text("i am a file", encoding="utf-8")
    with pytest.raises(OSError):
        run_once(db_path=tmp_path / "nonexistent.db", health_root=not_a_dir, now_ts=1_800_000_000.0, **_HIST_PATHS)


# --- bounded-query behavior --------------------------------------------------

def test_query_plan_uses_index_search_not_full_table_scan(tmp_path):
    db_path = tmp_path / "fixture.db"
    _make_fixture_db(db_path)
    conn = _open_ro(db_path)
    try:
        cur = conn.cursor()
        cur.execute("EXPLAIN QUERY PLAN SELECT ts_ms FROM liquidations WHERE symbol = ? ORDER BY ts_ms DESC LIMIT 1", ("BTCUSDT",))
        plan = " ".join(str(row) for row in cur.fetchall())
        assert "idx_liq_symbol_ts" in plan
        assert "SCAN liquidations USING INTEGER PRIMARY KEY" not in plan
    finally:
        conn.close()


def test_bounded_query_against_50k_unrelated_rows_is_fast(tmp_path):
    db_path = tmp_path / "fixture.db"
    _make_fixture_db(db_path)
    t0 = time.time()
    result, errors = read_last_liquidation_ts(db_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    elapsed = time.time() - t0
    assert elapsed < 2.0
    assert errors == []
    assert all(v is not None for v in result.values())


# --- historical replay against real production data (safe explicit mode) -----

@requires_real_db
def test_replay_confirmed_april_june_outage_classifies_transport_outage():
    """2026-04-27T14:24:51Z..2026-06-06T17:47:05Z is a confirmed, forensically
    documented all-tracked-symbol liquidation outage during which
    mark_prices/agg_trades kept advancing (REST-covered). Evaluated deep
    inside the gap (2026-05-15), point-in-time, HISTORICAL_REPLAY mode with
    no live overall.json/collector.json consulted -- must classify as
    LIQUIDATION_TRANSPORT_OUTAGE / RED."""
    now_ts = _iso_to_ts("2026-05-15T12:00:00")
    payload = evaluate_once(db_path=DEFAULT_DB_PATH, now_ts=now_ts, **_HIST_PATHS)
    assert payload["evaluation_mode"] == MODE_HISTORICAL_REPLAY
    assert payload["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert payload["severity"] == "RED"
    assert payload["complete_symbol_evidence"] is True
    assert payload["control_stream_ages_sec"]["mark_prices"] < 300
    assert payload["control_stream_ages_sec"]["agg_trades"] < 300


@requires_real_db
def test_replay_july_routed_endpoint_outage_also_classifies_transport_outage():
    """Separate, shorter, mechanistically distinct incident
    (2026-07-06T10:06:39Z..2026-07-10T11:24:37Z, routed-WS-endpoint
    regression) -- also REST-covered on mark/agg, also must classify as
    LIQUIDATION_TRANSPORT_OUTAGE once past the frozen critical threshold."""
    now_ts = _iso_to_ts("2026-07-08T12:00:00")
    payload = evaluate_once(db_path=DEFAULT_DB_PATH, now_ts=now_ts, **_HIST_PATHS)
    assert payload["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert payload["severity"] == "RED"


@requires_real_db
def test_replay_may3_diagnostic_instant_is_not_false_transport_outage():
    """2026-05-21T16:30Z inside outage #1 had a control-stream staleness blip;
    the detector must refuse to over-claim a liquidation-specific transport
    outage when controls are also stale (CONTROL_STREAMS_STALE / UNKNOWN)."""
    now_ts = _iso_to_ts("2026-05-21T16:30:00")
    payload = evaluate_once(db_path=DEFAULT_DB_PATH, now_ts=now_ts, **_HIST_PATHS)
    assert payload["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE


@requires_real_db
@pytest.mark.parametrize(
    "probe_iso",
    [
        "2026-06-11T00:00:00",
        "2026-06-13T12:00:00",
        "2026-06-30T06:00:00",
        "2026-07-02T18:00:00",
        "2026-07-10T18:00:00",
        "2026-07-11T06:00:00",
    ],
)
def test_replay_healthy_periods_never_produce_false_transport_outage(probe_iso):
    """Six probe points spread across the three known-healthy calibration
    windows (post-2026-06-06 architecture, July 6-10 outage excluded).
    None may classify as LIQUIDATION_TRANSPORT_OUTAGE."""
    now_ts = _iso_to_ts(probe_iso)
    payload = evaluate_once(db_path=DEFAULT_DB_PATH, now_ts=now_ts, **_HIST_PATHS)
    assert payload["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE
