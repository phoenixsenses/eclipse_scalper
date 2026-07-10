from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.native_ws_health_policy import (
    AGG_TRADES_STALE,
    ALL_COLLECTION_PATHS_STOPPED,
    COLLECTOR_PROCESS_MISSING,
    HEALTH_STATE_UNREADABLE,
    LIQUIDATIONS_CRITICALLY_STALE,
    LIQUIDATIONS_STALE,
    LIQUIDATIONS_WARNING_AGE_SEC,
    NATIVE_WS_DISCONNECTED,
    NATIVE_WS_MESSAGE_STALE,
    NATIVE_WS_MESSAGE_WARNING_AGE_SEC,
    RECONNECT_BACKOFF_ACTIVE,
    REST_FALLBACK_ACTIVE,
    STATUS_DEGRADED,
    STATUS_GREEN,
    STATUS_RED,
    evaluate_policy,
    read_source_freshness,
)

NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _iso(seconds_ago: float) -> str:
    return (NOW - timedelta(seconds=seconds_ago)).isoformat()


def _fresh_sources(agg_age=1.0, mark_age=1.0, liq_age=5.0):
    return {
        "agg_trades": {"last_ts_ms": 1, "age_sec": agg_age, "error": None},
        "mark_prices": {"last_ts_ms": 1, "age_sec": mark_age, "error": None},
        "liquidations": {"last_ts_ms": 1, "age_sec": liq_age, "error": None},
    }


def _healthy_heartbeat(message_age=1.0, backoff=1.0, rest_active=False, rest_age=1.0):
    return {
        "connected": True,
        "last_message_ts_utc": _iso(message_age),
        "last_data_progress_ts_utc": _iso(message_age),
        "rest_fallback_enabled": True,
        "rest_fallback_active": rest_active,
        "rest_last_progress_ts_utc": _iso(rest_age),
        "current_backoff_seconds": backoff,
        "last_error": "",
    }


def _healthy_component():
    return {"status": "ok", "connected": True, "transport_connected": True}


# --- required scenarios -----------------------------------------------------


def test_healthy_native_ws_and_fresh_sources_is_green():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_GREEN
    assert result["reasons"] == []


def test_native_disconnected_rest_active_agg_mark_fresh_is_degraded():
    hb = _healthy_heartbeat(message_age=500.0, rest_active=True, rest_age=2.0)
    hb["connected"] = False
    result = evaluate_policy(
        collector_heartbeat=hb,
        collector_component={"status": "degraded", "connected": False, "transport_connected": False},
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert NATIVE_WS_DISCONNECTED in result["reasons"]
    assert REST_FALLBACK_ACTIVE in result["reasons"]


def test_native_message_stale_while_transport_flag_still_connected_is_degraded():
    # transport_connected still True, but no application frame in a long time --
    # handshake success must not be treated as healthy data flow. Age chosen
    # between the warning and critical thresholds (currently 60s / 180s) so
    # this exercises DEGRADED specifically, not RED.
    hb = _healthy_heartbeat(message_age=100.0, rest_active=False)
    result = evaluate_policy(
        collector_heartbeat=hb,
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert NATIVE_WS_MESSAGE_STALE in result["reasons"]


def test_liquidations_stale_while_agg_and_mark_fresh_is_degraded():
    # Between warning (600s) and critical (1800s) so this exercises DEGRADED.
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(liq_age=1000.0),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert LIQUIDATIONS_STALE in result["reasons"]
    assert AGG_TRADES_STALE not in result["reasons"]


def test_all_sources_stale_is_red():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=5000.0, rest_active=True, rest_age=5000.0),
        collector_component={"status": "degraded", "connected": True, "transport_connected": True},
        collector_process_alive=True,
        source_freshness=_fresh_sources(agg_age=5000.0, mark_age=5000.0, liq_age=5000.0),
        now=NOW,
    )
    assert result["status"] == STATUS_RED
    assert ALL_COLLECTION_PATHS_STOPPED in result["reasons"] or LIQUIDATIONS_CRITICALLY_STALE in result["reasons"]


def test_collector_process_missing_is_red():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=False,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_RED
    assert result["reasons"] == [COLLECTOR_PROCESS_MISSING]


def test_corrupt_or_missing_required_state_is_red():
    result = evaluate_policy(
        collector_heartbeat=None,
        collector_component={},
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_RED
    assert result["reasons"] == [HEALTH_STATE_UNREADABLE]


def test_brief_rest_fallback_within_grace_period_is_not_flagged():
    # native message age just over stall timeout but well within the short
    # REST-fallback grace period -- a single reconnect blip, not a real problem.
    hb = _healthy_heartbeat(message_age=50.0, rest_active=True, rest_age=1.0)
    result = evaluate_policy(
        collector_heartbeat=hb,
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert REST_FALLBACK_ACTIVE not in result["reasons"]
    assert result["status"] == STATUS_GREEN


def test_recovery_from_degraded_to_green_after_native_frames_resume():
    degraded = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=500.0, rest_active=True, rest_age=2.0),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert degraded["status"] == STATUS_DEGRADED

    recovered = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=1.0, rest_active=False),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert recovered["status"] == STATUS_GREEN


# --- boundary tests: immediately below, at, and above each threshold ---------


def test_native_message_age_boundary_just_below_warning_is_green():
    age = NATIVE_WS_MESSAGE_WARNING_AGE_SEC - 1.0
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=age),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_GREEN


def test_native_message_age_boundary_at_warning_is_degraded():
    age = NATIVE_WS_MESSAGE_WARNING_AGE_SEC
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=age),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert NATIVE_WS_MESSAGE_STALE in result["reasons"]


def test_native_message_age_boundary_just_above_warning_is_degraded():
    age = NATIVE_WS_MESSAGE_WARNING_AGE_SEC + 1.0
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(message_age=age),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert NATIVE_WS_MESSAGE_STALE in result["reasons"]


def test_native_message_stale_past_stall_timeout_with_rest_not_yet_critical_is_degraded():
    # Required scenario: transport flags still true, native age past the allowed
    # post-stall window, REST fallback has not yet become critical.
    hb = _healthy_heartbeat(message_age=NATIVE_WS_MESSAGE_WARNING_AGE_SEC + 5.0, rest_active=False)
    result = evaluate_policy(
        collector_heartbeat=hb,
        collector_component=_healthy_component(),  # transport_connected still True
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert NATIVE_WS_MESSAGE_STALE in result["reasons"]


def test_liquidations_age_boundary_just_below_warning_is_green():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(liq_age=LIQUIDATIONS_WARNING_AGE_SEC - 1.0),
        now=NOW,
    )
    assert result["status"] == STATUS_GREEN


def test_liquidations_age_boundary_at_warning_is_degraded():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(liq_age=LIQUIDATIONS_WARNING_AGE_SEC),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert LIQUIDATIONS_STALE in result["reasons"]


def test_liquidations_age_boundary_just_above_warning_is_degraded():
    result = evaluate_policy(
        collector_heartbeat=_healthy_heartbeat(),
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(liq_age=LIQUIDATIONS_WARNING_AGE_SEC + 1.0),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert LIQUIDATIONS_STALE in result["reasons"]


def test_reconnect_backoff_active_is_degraded_even_if_connected_flag_true():
    hb = _healthy_heartbeat(message_age=1.0, backoff=32.0)
    result = evaluate_policy(
        collector_heartbeat=hb,
        collector_component=_healthy_component(),
        collector_process_alive=True,
        source_freshness=_fresh_sources(),
        now=NOW,
    )
    assert result["status"] == STATUS_DEGRADED
    assert RECONNECT_BACKOFF_ACTIVE in result["reasons"]


# --- source-freshness reader: no full-table scan -----------------------------


def test_read_source_freshness_uses_bounded_rowid_tail_not_full_scan(tmp_path):
    db = tmp_path / "micro.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE agg_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    conn.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    conn.execute("CREATE TABLE liquidations (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER)")
    now_ms = int(time.time() * 1000)
    # A large row count so a full-table MAX(ts_ms) scan would be measurably slow;
    # ORDER BY rowid DESC LIMIT 1 must stay fast regardless of table size.
    rows = [(now_ms - (50_000 - i) * 1000,) for i in range(50_000)]
    conn.executemany("INSERT INTO agg_trades (ts_ms) VALUES (?)", rows)
    conn.executemany("INSERT INTO mark_prices (ts_ms) VALUES (?)", rows)
    conn.executemany("INSERT INTO liquidations (ts_ms) VALUES (?)", rows)
    conn.commit()
    conn.close()

    start = time.time()
    result = read_source_freshness(db, now=time.time())
    elapsed = time.time() - start

    assert elapsed < 2.0, f"read_source_freshness took {elapsed:.2f}s -- looks like a full scan, not a rowid-tail read"
    assert result["agg_trades"]["age_sec"] is not None and result["agg_trades"]["age_sec"] < 5.0
    assert result["mark_prices"]["age_sec"] is not None
    assert result["liquidations"]["age_sec"] is not None

    # The function body's executed query must use the bounded rowid-tail form.
    # (Checked on the function body specifically, not the docstring, which
    # documents the anti-pattern by name for future readers.)
    import inspect
    import tools.native_ws_health_policy as policy_mod

    source = inspect.getsource(policy_mod.read_source_freshness)
    body = source.split('"""', 2)[-1]  # drop the docstring
    assert "MAX(ts_ms)" not in body
    assert "ORDER BY rowid DESC LIMIT 1" in body


def test_read_source_freshness_handles_missing_db_gracefully(tmp_path):
    result = read_source_freshness(tmp_path / "does_not_exist.db", now=time.time())
    for table in ("agg_trades", "mark_prices", "liquidations"):
        assert result[table]["age_sec"] is None


# --- historical outage replay -------------------------------------------------
# Fixture values below are the actual captured heartbeat/component/DB state from
# the 2026-07-06..2026-07-10 routed-endpoint outage (see
# reports/research/s34/LIQUIDATION_TRANSPORT_RESTORED_2026-07-10.md), snapshotted
# at 2026-07-08 (roughly the midpoint of the outage), to prove the old
# watchdog_overall=GREEN reading for this exact state would not recur.


def test_july_outage_state_replay_would_not_be_green():
    replay_now = datetime(2026, 7, 8, 12, 0, 0, tzinfo=timezone.utc)
    outage_heartbeat = {
        "connected": False,
        "last_message_ts_utc": None,
        "last_data_progress_ts_utc": "2026-07-08T11:59:55.000000+00:00",
        "rest_fallback_enabled": True,
        "rest_fallback_active": True,
        "rest_last_progress_ts_utc": "2026-07-08T11:59:55.000000+00:00",
        "current_backoff_seconds": 60.0,
        "last_error": "connection_error:RuntimeError: stall_timeout_no_messages>45s",
    }
    outage_component = {
        "status": "ok",  # the actual, misleading value the old collector-side logic wrote
        "connected": False,
        "transport_connected": False,
        "required_streams_progressing": True,
        "liquidation_transport_available": False,
    }
    # agg_trades/mark_prices kept fresh by REST; liquidations frozen since
    # 2026-07-06T10:06:39Z -- ~2 days old by this replay point.
    outage_sources = {
        "agg_trades": {"last_ts_ms": 1, "age_sec": 3.0, "error": None},
        "mark_prices": {"last_ts_ms": 1, "age_sec": 4.0, "error": None},
        "liquidations": {"last_ts_ms": 1, "age_sec": 2 * 24 * 3600.0, "error": None},
    }

    result = evaluate_policy(
        collector_heartbeat=outage_heartbeat,
        collector_component=outage_component,
        collector_process_alive=True,
        source_freshness=outage_sources,
        now=replay_now,
    )

    assert result["status"] != STATUS_GREEN
    assert result["status"] == STATUS_RED
    assert LIQUIDATIONS_CRITICALLY_STALE in result["reasons"]
