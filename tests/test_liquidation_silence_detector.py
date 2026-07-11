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


# --- disabled-by-default integration ----------------------------------------

def test_disabled_integration_has_zero_effect_on_canonical_overall(tmp_path):
    """Proves the detector's component file is invisible to
    tools.heartbeat_watchdog.build_canonical_overall (unmodified,
    real production module) unless/until a future controlled-activation
    batch explicitly adds "liquidation_silence" to OPTIONAL_COMPONENT_FILES
    -- which this batch does not do."""
    log_health = tmp_path / "logs" / "health"
    log_health.mkdir(parents=True)
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0, **_HIST_PATHS)
    write_component_health("liquidation_silence", payload, root=log_health)

    assert "liquidation_silence" not in hw.OPTIONAL_COMPONENT_FILES

    overall_without = hw.build_canonical_overall(
        overall="GREEN", issues=[], collector_component={}, bookticker_component={},
        native_ws_policy={"status": "GREEN", "reasons": [], "native_websocket": {}, "rest_fallback": {}, "source_freshness": {}, "thresholds": {}},
        runtime_mode="paper", now_iso="2026-07-11T00:00:00Z", log_health=log_health,
    )
    assert "liquidation_silence" not in overall_without["components"]
    assert overall_without["state"] == "ok"


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
