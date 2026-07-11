"""I/O, writer-ownership, DB-safety, bounded-query, and historical-replay
tests for tools/liquidation_silence_detector.py.

All production-DB reads use data/microstructure.db strictly `mode=ro` and
are scoped to short historical windows via the same bounded, indexed
`ORDER BY ts_ms ... LIMIT 1` / `WHERE ts_ms BETWEEN` pattern the detector
itself uses -- never a full-table scan. Every write in this file targets an
isolated tmp/scratch path; nothing here ever writes to the real
logs/health/ directory.
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
    _open_ro,
    discover_tracked_symbols,
    evaluate_once,
    read_control_freshness,
    read_last_liquidation_ts,
    run_once,
)
from tools.liquidation_silence_policy import (
    STATUS_HEALTHY,
    STATUS_LIQUIDATION_TRANSPORT_OUTAGE,
)

REAL_DB_AVAILABLE = DEFAULT_DB_PATH.exists()
requires_real_db = pytest.mark.skipif(not REAL_DB_AVAILABLE, reason="data/microstructure.db not present in this environment")


def _iso_to_ms(iso: str) -> int:
    return int(dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def _iso_to_ts(iso: str) -> float:
    return dt.datetime.fromisoformat(iso).replace(tzinfo=dt.timezone.utc).timestamp()


# --- symbol universe discovery (test 22) ------------------------------------

def test_discover_tracked_symbols_from_canonical_config(tmp_path):
    meta = tmp_path / "collector_supervisor.json"
    # Deliberately includes a UTF-8 BOM, matching the real
    # logs/pids/collector_supervisor.json this reads from in production.
    meta.write_bytes(b"\xef\xbb\xbf" + json.dumps({"symbols": "BTCUSDT,ETHUSDT,SOLUSDT"}).encode("utf-8"))
    result = discover_tracked_symbols(meta)
    assert result["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert result["source"] == "canonical_runtime_config"


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


# --- writer ownership / atomicity (tests 13, 14, 16, 21) ---------------------

def test_write_creates_only_its_own_component_file(tmp_path):
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["liquidation_silence.json"]


def test_never_writes_overall_json(tmp_path):
    existing_overall = tmp_path / "overall.json"
    existing_overall.write_text(json.dumps({"state": "ok", "sentinel": "untouched"}), encoding="utf-8")
    before = existing_overall.read_bytes()
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    after = existing_overall.read_bytes()
    assert before == after
    assert (tmp_path / "liquidation_silence.json").exists()


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]
    assert leftovers == []


def test_write_overwrites_corrupt_predecessor(tmp_path):
    target = tmp_path / "liquidation_silence.json"
    target.write_bytes(b"\x00\x01not-json-garbage{{{")
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["component"] == "liquidation_silence"


def test_component_output_carries_ts_utc_for_staleness_grading(tmp_path):
    now_ts = 1_800_000_000.0
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=now_ts)
    write_component_health("liquidation_silence", payload, root=tmp_path)
    reloaded = json.loads((tmp_path / "liquidation_silence.json").read_text(encoding="utf-8"))
    assert "ts_utc" in reloaded
    later = now_ts + 1000.0
    fresh = hw.component_fresh(reloaded, max_age_sec=1500, now=dt.datetime.fromtimestamp(later, dt.timezone.utc))
    stale = hw.component_fresh(reloaded, max_age_sec=500, now=dt.datetime.fromtimestamp(later, dt.timezone.utc))
    assert fresh is True
    assert stale is False


# --- deterministic serialization (test 15) -----------------------------------

def test_run_once_dry_run_does_not_write(tmp_path):
    payload = run_once(db_path=tmp_path / "nonexistent.db", health_root=tmp_path, now_ts=1_800_000_000.0, dry_run=True)
    assert payload["status"] in {STATUS_HEALTHY, "UNKNOWN_INSUFFICIENT_EVIDENCE"}
    assert not (tmp_path / "liquidation_silence.json").exists()


def test_two_runs_same_inputs_produce_byte_identical_component_payload(tmp_path):
    p1 = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    p2 = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    p1.pop("detector_runtime_sec")
    p2.pop("detector_runtime_sec")
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


# --- disabled-by-default integration (test 19) -------------------------------

def test_disabled_integration_has_zero_effect_on_canonical_overall(tmp_path):
    """Proves the detector's component file is invisible to
    tools.heartbeat_watchdog.build_canonical_overall (unmodified,
    real production module) unless/until a future controlled-activation
    batch explicitly adds "liquidation_silence" to OPTIONAL_COMPONENT_FILES
    -- which this batch does not do."""
    log_health = tmp_path / "logs" / "health"
    log_health.mkdir(parents=True)
    payload = evaluate_once(db_path=tmp_path / "nonexistent.db", now_ts=1_800_000_000.0)
    write_component_health("liquidation_silence", payload, root=log_health)

    assert "liquidation_silence" not in hw.OPTIONAL_COMPONENT_FILES

    overall_without = hw.build_canonical_overall(
        overall="GREEN", issues=[], collector_component={}, bookticker_component={},
        native_ws_policy={"status": "GREEN", "reasons": [], "native_websocket": {}, "rest_fallback": {}, "source_freshness": {}, "thresholds": {}},
        runtime_mode="paper", now_iso="2026-07-11T00:00:00Z", log_health=log_health,
    )
    assert "liquidation_silence" not in overall_without["components"]
    assert overall_without["state"] == "ok"


# --- read-only / no-mutation guarantees (tests 17, 18) -----------------------

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
    evaluate_once(db_path=db_path, now_ts=1_800_000_010.0)
    read_last_liquidation_ts(db_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    read_control_freshness(db_path, now_ts=1_800_000_010.0)
    after = hashlib.sha256(db_path.read_bytes()).hexdigest()
    assert before == after


# --- bounded-query behavior (test 23) ----------------------------------------

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
    result = read_last_liquidation_ts(db_path, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    elapsed = time.time() - t0
    assert elapsed < 2.0
    assert all(v is not None for v in result.values())


# --- historical replay against real production data (tests 24, 25) ----------

@requires_real_db
def test_replay_confirmed_april_june_outage_classifies_transport_outage():
    """2026-04-27T14:24:51Z..2026-06-06T17:47:05Z is a confirmed, forensically
    documented all-tracked-symbol liquidation outage (see
    reports/research/s34/S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md)
    during which mark_prices/agg_trades kept advancing (REST-covered).
    Evaluated deep inside the gap (2026-05-15), point-in-time, with no
    live overall.json/collector.json consulted (paths point at nonexistent
    files) -- this must classify as LIQUIDATION_TRANSPORT_OUTAGE / RED."""
    now_ts = _iso_to_ts("2026-05-15T12:00:00")
    payload = evaluate_once(
        db_path=DEFAULT_DB_PATH,
        pid_meta_path=Path("/nonexistent/pid_meta.json"),
        overall_path=Path("/nonexistent/overall.json"),
        collector_component_path=Path("/nonexistent/collector.json"),
        now_ts=now_ts,
    )
    assert payload["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert payload["severity"] == "RED"
    assert payload["control_stream_ages_sec"]["mark_prices"] < 300
    assert payload["control_stream_ages_sec"]["agg_trades"] < 300


@requires_real_db
def test_replay_july_routed_endpoint_outage_also_classifies_transport_outage():
    """Separate, shorter, mechanistically distinct incident
    (2026-07-06T10:06:39Z..2026-07-10T11:24:37Z, routed-WS-endpoint
    regression) -- also REST-covered on mark/agg, also must classify as
    LIQUIDATION_TRANSPORT_OUTAGE once past the frozen critical threshold."""
    now_ts = _iso_to_ts("2026-07-08T12:00:00")
    payload = evaluate_once(
        db_path=DEFAULT_DB_PATH,
        pid_meta_path=Path("/nonexistent/pid_meta.json"),
        overall_path=Path("/nonexistent/overall.json"),
        collector_component_path=Path("/nonexistent/collector.json"),
        now_ts=now_ts,
    )
    assert payload["status"] == STATUS_LIQUIDATION_TRANSPORT_OUTAGE
    assert payload["severity"] == "RED"


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
    None may classify as LIQUIDATION_TRANSPORT_OUTAGE -- the frozen
    critical threshold was calibrated to carry ~2.87x margin over the
    observed historical maximum all-symbol silence age (2508.9s) in
    exactly this population."""
    now_ts = _iso_to_ts(probe_iso)
    payload = evaluate_once(
        db_path=DEFAULT_DB_PATH,
        pid_meta_path=Path("/nonexistent/pid_meta.json"),
        overall_path=Path("/nonexistent/overall.json"),
        collector_component_path=Path("/nonexistent/collector.json"),
        now_ts=now_ts,
    )
    assert payload["status"] != STATUS_LIQUIDATION_TRANSPORT_OUTAGE
