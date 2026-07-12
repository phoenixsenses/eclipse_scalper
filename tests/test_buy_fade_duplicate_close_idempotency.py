"""F-DUPCLOSE-01 corrective test matrix (in-memory SQLite + temp state/ledger
files only). Drives the REAL `advance_positions` / `load_state` /
`load_closed_ids_from_ledger` functions (unmodified import) with `log_event`
monkeypatched to a collector so nothing is ever written to the live
production ledger (reports/shadow/s34_state_machine_shadow.jsonl), and
`STATE_PATH`/`LEDGER_PATH` monkeypatched to pytest tmp_path files for the
persistence-reconciliation tests.

Background: a confirmed defect allowed the same logical position to be
CLOSE-logged more than once when the shadow-runner process restarted between
a durable ledger CLOSE write (log_event) and that cycle's state.json
persistence (save_state, which only happens once per run_once() cycle).
Evidence: D:\\eclipse_scalper_research\\buy_fade_h45_sl75_reconstruction_20260712T151830Z\\
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.s34_realtime_shadow_runner as m  # noqa: E402

SYM = "ETHUSDT"


def make_db(mark_span, mark_price=100.0, price_path=None):
    """price_path: optional list of (ts_ms, mark_price) overriding the flat price."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE liquidations (symbol TEXT, side TEXT, ts_ms INTEGER, price REAL, notional REAL);"
        "CREATE TABLE mark_prices  (symbol TEXT, ts_ms INTEGER, mark_price REAL);"
    )
    if price_path:
        for ts, px in price_path:
            conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (SYM, ts, px))
    else:
        for ts in range(mark_span[0], mark_span[1] + 60_000, 5_000):
            conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (SYM, ts, mark_price))
    conn.commit()
    return conn


def make_pos(pid, entry_ts_ms, entry_price=1800.0, exit_due_ms=None, sl_bps=999.0, signal="BUY_FADE_SHORT_H45_SL75"):
    return {
        "id": pid,
        "signal": signal,
        "direction": "SHORT",
        "status": "OPEN_BUY_FADE",
        "anchor_ts_ms": entry_ts_ms,
        "entry_ts_ms": entry_ts_ms,
        "entry_price": entry_price,
        "buy_state": "SILENCE",
        "sl_bps": sl_bps,
        "exit_due_ms": exit_due_ms if exit_due_ms is not None else entry_ts_ms + 10_000_000_000,
    }


def run_tick(monkeypatch, conn, state, now_ms, ledger_sink=None):
    if ledger_sink is None:
        ledger_sink = []
    monkeypatch.setattr(m, "log_event", lambda rec: ledger_sink.append(rec))
    m.advance_positions(conn, state, now_ms)
    return ledger_sink


# ── 1. one position emits exactly one CLOSE ─────────────────────────────────

def test_one_position_emits_exactly_one_close_on_time_exit(monkeypatch):
    entry_ts = 1_800_000_000_000
    exit_due = entry_ts + 2_700_000
    conn = make_db((entry_ts, exit_due + 60_000))
    pos = make_pos("P1", entry_ts, exit_due_ms=exit_due)
    state = {"positions": {"P1": pos}}
    ledger = run_tick(monkeypatch, conn, state, exit_due + 10_000)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 1
    assert "P1" not in state["positions"]  # removed after close
    conn.close()


def test_one_position_emits_exactly_one_close_on_stop_loss(monkeypatch):
    entry_ts = 1_800_000_000_000
    # price path: flat then jumps past the 75bps stop shortly after entry
    price_path = [(entry_ts, 1800.0), (entry_ts + 60_000, 1800.0 * 1.02)]
    conn = make_db(None, price_path=price_path)
    pos = make_pos("P1", entry_ts, entry_price=1800.0, sl_bps=75.0, exit_due_ms=entry_ts + 2_700_000)
    state = {"positions": {"P1": pos}}
    ledger = run_tick(monkeypatch, conn, state, entry_ts + 60_000)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 1
    assert closes[0]["close_reason"] == "BUY_FADE_SL75"
    conn.close()


# ── 2. repeated close invocation emits no duplicate ─────────────────────────

def test_repeated_close_invocation_same_tick_state_emits_no_duplicate(monkeypatch):
    """Calling advance_positions() a second time with the SAME in-memory
    state (post-close) must not re-close -- the position is gone from
    state["positions"], and even if manually re-inserted with its closed
    status, the closed_ids guard blocks a second log_event."""
    entry_ts = 1_800_000_000_000
    exit_due = entry_ts + 2_700_000
    conn = make_db((entry_ts, exit_due + 120_000))
    pos = make_pos("P1", entry_ts, exit_due_ms=exit_due)
    state = {"positions": {"P1": pos}}
    ledger = run_tick(monkeypatch, conn, state, exit_due + 10_000)
    assert len(ledger) == 1

    # simulate a bug that re-inserts the already-closed position into the
    # active set (defense-in-depth: closed_ids must still block it)
    reinserted = dict(pos)
    reinserted["status"] = "OPEN_BUY_FADE"  # as if it were never closed
    state["positions"]["P1"] = reinserted
    run_tick(monkeypatch, conn, state, exit_due + 20_000, ledger_sink=ledger)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 1, "closed_ids guard must block a second CLOSE for the same id"
    conn.close()


# ── 3. repeated advance cycle after close emits no duplicate (regression lock) ──

def test_repeated_advance_cycle_after_close_emits_no_duplicate(monkeypatch):
    entry_ts = 1_800_000_000_000
    exit_due = entry_ts + 2_700_000
    conn = make_db((entry_ts, exit_due + 300_000))
    pos = make_pos("P1", entry_ts, exit_due_ms=exit_due)
    state = {"positions": {"P1": pos}}
    ledger = []
    for tick in range(5):
        run_tick(monkeypatch, conn, state, exit_due + 10_000 + tick * 5_000, ledger_sink=ledger)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 1
    conn.close()


# ── 4/5. stop and time-exit closes are each idempotent (covered above); this
#         adds an explicit stop-then-further-ticks check ──────────────────

def test_stop_loss_close_idempotent_across_further_ticks(monkeypatch):
    entry_ts = 1_800_000_000_000
    price_path = [(entry_ts, 1800.0)] + [(entry_ts + i * 5_000, 1800.0 * 1.02) for i in range(1, 10)]
    conn = make_db(None, price_path=price_path)
    pos = make_pos("P1", entry_ts, entry_price=1800.0, sl_bps=75.0, exit_due_ms=entry_ts + 2_700_000)
    state = {"positions": {"P1": pos}}
    ledger = []
    for tick in range(1, 6):
        run_tick(monkeypatch, conn, state, entry_ts + tick * 5_000, ledger_sink=ledger)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 1
    conn.close()


# ── 6. restart/reloaded state does not re-close an already-closed position ──

def test_load_state_purges_position_already_closed_in_ledger(tmp_path, monkeypatch):
    """The confirmed defect: state.json (stale, pre-crash) still shows a
    position OPEN, but the ledger (durable, written before the crash) already
    has its CLOSE row. load_state() must reconcile: drop the stale open
    position and populate closed_ids from ledger truth."""
    state_path = tmp_path / "state.json"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(m, "STATE_PATH", state_path)
    monkeypatch.setattr(m, "LEDGER_PATH", ledger_path)

    stale_state = {
        "positions": {
            "P1": {"id": "P1", "signal": "BUY_FADE_SHORT_H45_SL75", "status": "OPEN_BUY_FADE"},
            "P2": {"id": "P2", "signal": "BUY_FADE_SHORT_H45_SL75", "status": "OPEN_BUY_FADE"},
        },
        "processed": {}, "pnl": {},
    }
    state_path.write_text(json.dumps(stale_state), encoding="utf-8")
    ledger_path.write_text(
        json.dumps({"event": "CLOSE", "id": "P1", "net_bps": 5.0}) + "\n",
        encoding="utf-8",
    )

    loaded = m.load_state()
    assert "P1" not in loaded["positions"], "already-closed position must be purged on reload"
    assert "P2" in loaded["positions"], "distinct, still-open position must be unaffected"
    assert loaded["closed_ids"].get("P1") is True
    assert "P2" not in loaded["closed_ids"]


def test_load_state_reconciled_position_cannot_be_re_closed(tmp_path, monkeypatch):
    """End-to-end restart simulation: after load_state() reconciliation, even
    if the (purged) position were somehow re-added to the active set by a
    caller, advance_positions() must not emit a second CLOSE for it."""
    state_path = tmp_path / "state.json"
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(m, "STATE_PATH", state_path)
    monkeypatch.setattr(m, "LEDGER_PATH", ledger_path)

    entry_ts = 1_800_000_000_000
    exit_due = entry_ts + 2_700_000
    stale_state = {
        "positions": {
            "P1": make_pos("P1", entry_ts, exit_due_ms=exit_due),
        },
        "processed": {}, "pnl": {},
    }
    state_path.write_text(json.dumps(stale_state), encoding="utf-8")
    ledger_path.write_text(
        json.dumps({"event": "CLOSE", "id": "P1", "net_bps": 5.0}) + "\n",
        encoding="utf-8",
    )

    loaded = m.load_state()
    assert "P1" not in loaded["positions"]

    # simulate a caller re-adding the position despite the reconciliation
    loaded["positions"]["P1"] = make_pos("P1", entry_ts, exit_due_ms=exit_due)
    conn = make_db((entry_ts, exit_due + 60_000))
    ledger = run_tick(monkeypatch, conn, loaded, exit_due + 10_000)
    closes = [r for r in ledger if r.get("event") == "CLOSE" and r.get("id") == "P1"]
    assert len(closes) == 0, "closed_ids (from ledger reconciliation) must block the re-close"
    conn.close()


# ── 7. distinct positions still produce distinct CLOSE rows ────────────────

def test_distinct_positions_produce_distinct_close_rows(monkeypatch):
    entry_ts = 1_800_000_000_000
    exit_due = entry_ts + 2_700_000
    conn = make_db((entry_ts, exit_due + 60_000))
    pos1 = make_pos("P1", entry_ts, exit_due_ms=exit_due)
    pos2 = make_pos("P2", entry_ts, exit_due_ms=exit_due)
    state = {"positions": {"P1": pos1, "P2": pos2}}
    ledger = run_tick(monkeypatch, conn, state, exit_due + 10_000)
    closed_ids = sorted(r["id"] for r in ledger if r.get("event") == "CLOSE")
    assert closed_ids == ["P1", "P2"]
    conn.close()


# ── 8. pre-existing duplicate historical rows are not mutated ──────────────

def test_load_closed_ids_from_ledger_does_not_mutate_existing_duplicate_rows(tmp_path, monkeypatch):
    """Historical evidence: 6 anchors were duplicate-closed 2-3x each in the
    real ledger before this corrective. load_closed_ids_from_ledger() must be
    strictly read-only -- byte-identical file content before and after -- and
    must correctly report the id as closed despite the duplicate rows."""
    ledger_path = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(m, "LEDGER_PATH", ledger_path)

    rows = [
        {"event": "OPEN", "id": "BUYF:123:H45SL75"},
        {"event": "CLOSE", "id": "BUYF:123:H45SL75", "net_bps": 35.96},
        {"event": "CLOSE", "id": "BUYF:123:H45SL75", "net_bps": 35.49},
        {"event": "CLOSE", "id": "BUYF:123:H45SL75", "net_bps": 34.79},
        {"event": "CLOSE", "id": "BUYF:999:H45SL75", "net_bps": 1.0},
    ]
    content = "\n".join(json.dumps(r) for r in rows) + "\n"
    ledger_path.write_text(content, encoding="utf-8")
    before_bytes = ledger_path.read_bytes()

    closed_ids = m.load_closed_ids_from_ledger()

    after_bytes = ledger_path.read_bytes()
    assert before_bytes == after_bytes, "ledger file must not be rewritten or modified"
    assert closed_ids == {"BUYF:123:H45SL75", "BUYF:999:H45SL75"}


def test_load_closed_ids_from_ledger_missing_file_returns_empty_set(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "LEDGER_PATH", tmp_path / "does_not_exist.jsonl")
    assert m.load_closed_ids_from_ledger() == set()
