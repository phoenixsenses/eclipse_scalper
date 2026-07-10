"""OD-004-FOLLOWUP bd_first_buy50 activation-boundary test matrix (in-memory
SQLite only). Drives the REAL `advance_positions` function (unmodified
import) with `log_event` monkeypatched to a no-op collector so nothing is
ever written to the live production ledger
(reports/shadow/s34_state_machine_shadow.jsonl).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.s34_realtime_shadow_runner as m  # noqa: E402

SYM = "ETHUSDT"  # hardcoded inside advance_positions' BUYF branch
T0 = 1_800_000_000_000


def make_db(liq_rows, mark_span, mark_price=100.0):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE liquidations (symbol TEXT, side TEXT, ts_ms INTEGER, price REAL, notional REAL);"
        "CREATE TABLE mark_prices  (symbol TEXT, ts_ms INTEGER, mark_price REAL);"
        "CREATE TABLE agg_trades   (symbol TEXT, ts_ms INTEGER, price REAL, qty REAL, notional REAL, is_buyer_maker INTEGER);"
        "CREATE TABLE book_ticker  (symbol TEXT, ts_ms INTEGER, bid_qty REAL, ask_qty REAL);"
    )
    for ts, notional in liq_rows:
        conn.execute("INSERT INTO liquidations VALUES (?,?,?,?,?)", (SYM, "BUY", ts, mark_price, notional))
    for ts in range(mark_span[0], mark_span[1] + 60_000, 5_000):
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (SYM, ts, mark_price))
    conn.commit()
    return conn


def make_pos(entry_ts_ms: int, entry_price: float = 1800.0, exit_due_ms: int | None = None):
    return {
        "id": "TESTPOS1",
        "signal": "BUY_FADE_SHORT_H45_SL75",
        "direction": "SHORT",
        "status": "OPEN_BUY_FADE",
        "anchor_ts_ms": entry_ts_ms,
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "buy_state": "SILENCE",  # skip the PENDING/NOISY sub-branch entirely
        "sl_bps": 999.0,  # effectively disable SL so it never interferes with the observer test
        "exit_due_ms": exit_due_ms if exit_due_ms is not None else entry_ts_ms + 10_000_000_000,  # far future
    }


def run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=None):
    monkeypatch.setattr(m, "log_event", lambda rec: None)  # never touch the live ledger file
    if process_start_ms is not None:
        monkeypatch.setattr(m, "_PROCESS_START_MS", process_start_ms)
    state = {"positions": {pos["id"]: pos}}
    m.advance_positions(conn, state, now_ms)
    return state["positions"].get(pos["id"], pos)


ACT_MS = m.BD_FIRST_BUY50_ACTIVATION_MS


def test_entry_plus_30m_before_activation_event_after_activation_is_selected(monkeypatch):
    """entry+30m falls before activation; a qualifying BUY event exists both
    before and after activation -> only the AT-OR-AFTER-activation one may
    be selected (never the earlier one)."""
    entry_ts = ACT_MS - 3_600_000  # entry 1h before activation -> entry+30m also before activation
    event_before = ACT_MS - 600_000  # 10min before activation: must be ignored
    event_after = ACT_MS + 900_000   # 15min after activation: eligible
    conn = make_db(
        [(event_before, 60_000.0), (event_after, 60_000.0)],
        (entry_ts, event_after + 60_000),
    )
    pos = make_pos(entry_ts)
    now_ms = event_after + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["hypothetical_exit_ts_ms"] == event_after
    conn.close()


def test_entry_plus_30m_after_activation_uses_entry_plus_30m_normally(monkeypatch):
    entry_ts = ACT_MS + 3_600_000  # position opens well after activation
    event_ts = entry_ts + 1_800_000 + 300_000  # 5min after entry+30m
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 60_000))
    pos = make_pos(entry_ts)
    now_ms = event_ts + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["hypothetical_exit_ts_ms"] == event_ts
    conn.close()


def test_event_one_ms_before_activation_is_ignored(monkeypatch):
    entry_ts = ACT_MS - 3_600_000
    event_ts = ACT_MS - 1  # 1ms before activation
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, ACT_MS + 60_000))
    pos = make_pos(entry_ts)
    now_ms = ACT_MS + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is False
    assert bobs["hypothetical_exit_ts_ms"] is None
    conn.close()


def test_event_exactly_at_activation_is_eligible(monkeypatch):
    entry_ts = ACT_MS - 3_600_000
    event_ts = ACT_MS  # exactly at activation
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, ACT_MS + 60_000))
    pos = make_pos(entry_ts)
    now_ms = ACT_MS + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["hypothetical_exit_ts_ms"] == event_ts
    conn.close()


def test_event_exactly_at_entry_plus_30m_is_eligible(monkeypatch):
    entry_ts = ACT_MS + 3_600_000  # after activation, so entry+30m is the binding bound
    event_ts = entry_ts + 1_800_000  # exactly entry+30m
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 60_000))
    pos = make_pos(entry_ts)
    now_ms = event_ts + 1_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["hypothetical_exit_ts_ms"] == event_ts
    conn.close()


def test_restart_after_qualifying_post_activation_event_is_idempotent(monkeypatch):
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 120_000))
    pos = make_pos(entry_ts)
    now_ms = event_ts + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    assert pos["bd_first_buy50_observer"]["triggered"] is True
    first_bobs = dict(pos["bd_first_buy50_observer"])
    # "restart": a later tick with a NEW process_start_ms, same persisted pos dict
    later_now_ms = now_ms + 60_000
    pos = run_tick(monkeypatch, conn, pos, later_now_ms, process_start_ms=later_now_ms - 1_000)
    assert pos["bd_first_buy50_observer"] == first_bobs  # untouched, no re-trigger
    conn.close()


def test_reconstructed_after_restart_marked_true_for_historical_event(monkeypatch):
    """The market event happened before THIS process instance started
    (simulated by setting _PROCESS_START_MS after event_ts) -> must be
    flagged reconstructed, while retaining its true historical timestamp."""
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 120_000))
    pos = make_pos(entry_ts)
    now_ms = event_ts + 3_600_000  # detected a full hour later (e.g. after a restart)
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=event_ts + 1_800_000)  # process started AFTER event_ts
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["hypothetical_exit_ts_ms"] == event_ts  # historical timestamp retained
    assert bobs["detected_at_ts_ms"] == now_ms  # detection time recorded separately
    assert bobs["reconstructed_after_restart"] is True
    conn.close()


def test_not_reconstructed_when_detected_live(monkeypatch):
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 60_000))
    pos = make_pos(entry_ts)
    now_ms = event_ts + 5_000  # detected almost immediately
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=entry_ts - 1_000)  # process predates the event
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["reconstructed_after_restart"] is False
    conn.close()


def test_no_qualifying_event_leaves_observer_untriggered(monkeypatch):
    entry_ts = ACT_MS + 3_600_000
    conn = make_db([], (entry_ts, entry_ts + 3_600_000))
    pos = make_pos(entry_ts)
    now_ms = entry_ts + 2_000_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is False
    conn.close()


def test_non_buyf_position_is_never_given_an_observer(monkeypatch):
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 60_000))
    pos = make_pos(entry_ts)
    pos["signal"] = "SHORT_NOISY"
    pos["status"] = "OPEN"
    now_ms = event_ts + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    assert "bd_first_buy50_observer" not in pos
    conn.close()


def test_canonical_time_exit_and_baseline_unchanged_by_observer(monkeypatch):
    """Baseline TIME_EXIT close logic (pre-existing, untouched) must still
    fire on schedule regardless of observer state, and must populate the
    observer's baseline fields independently from its own hypothetical exit."""
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    exit_due_ms = entry_ts + 2_700_000  # 45m hold, before "now"
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, exit_due_ms + 60_000))
    pos = make_pos(entry_ts, exit_due_ms=exit_due_ms)
    now_ms = exit_due_ms + 10_000  # observer triggers first, then TIME_EXIT closes the position
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    assert pos["status"] == "CLOSED_TIME_EXIT"  # canonical baseline exit unaffected
    bobs = pos["bd_first_buy50_observer"]
    assert bobs["triggered"] is True
    assert bobs["baseline_exit_ts_ms"] == pos["exit_ts_ms"]
    assert bobs["baseline_net_bps"] == pos["net_bps"]
    assert bobs["delta_vs_baseline_bps"] == round(bobs["hypothetical_exit_net_bps"] - pos["net_bps"], 2)
    conn.close()


def test_observer_cannot_mutate_position_lifecycle_fields(monkeypatch):
    entry_ts = ACT_MS + 3_600_000
    event_ts = entry_ts + 1_800_000 + 300_000
    conn = make_db([(event_ts, 60_000.0)], (entry_ts, event_ts + 60_000))
    pos = make_pos(entry_ts)
    frozen_entry_price = pos["entry_price"]
    frozen_exit_due = pos["exit_due_ms"]
    now_ms = event_ts + 30_000
    pos = run_tick(monkeypatch, conn, pos, now_ms, process_start_ms=now_ms - 1_000)
    assert pos["bd_first_buy50_observer"]["triggered"] is True
    assert pos["status"] == "OPEN_BUY_FADE"  # never closed by the observer
    assert pos["entry_price"] == frozen_entry_price
    assert pos["exit_due_ms"] == frozen_exit_due
    assert "exit_price" not in pos  # canonical exit fields absent -- position never closed
    conn.close()
