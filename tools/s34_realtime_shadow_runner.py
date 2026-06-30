"""S34 Real-Time Shadow Runner — paper trades all validated signals, no live orders.

Polls every 5s. For each new ETH SELL cascade ≥200K:
  1. LONG_SILENCE  : enter LONG immediately, exit early if noisy, extend 4h if silence
  2. SHORT_NOISY   : wait for first ETH SELL propagation ≥50K in 1-30min, enter SHORT 2h
  3. SHORT_NEITHER : if BTC SELL ≥500K detected in 30min window, enter SHORT 2h

All positions are paper-only (no exchange calls). Outcomes tracked vs mark price history.

Ledger : reports/shadow/s34_realtime_shadow.jsonl
State  : reports/shadow/s34_realtime_shadow_state.json

Usage:
    python tools/s34_realtime_shadow_runner.py [--once] [--db data/microstructure.db]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DEFAULT_DB    = ROOT / "data" / "microstructure.db"
LEDGER_PATH   = ROOT / "reports" / "shadow" / "s34_realtime_shadow.jsonl"
STATE_PATH    = ROOT / "reports" / "shadow" / "s34_realtime_shadow_state.json"
PID_PATH      = ROOT / "logs" / "pids" / "s34_realtime_shadow_runner.pid"

# Signal parameters — must match research findings exactly
ETH_THRESH        = 200_000.0   # anchor cascade threshold
PROP_THRESH       = 50_000.0    # follow-on / propagation threshold
BTC_THRESH        = 500_000.0   # BTC cascade threshold for neither_silence
SYNC_WIN_MS       = 10 * 60_000 # sync_k window (prior 10min)
SIL_LO_MS         = 60_000      # silence window low bound (skip ultra-early)
SIL_HI_MS         = 30 * 60_000 # silence confirmed after 30min
NOISY_WIN_HI_MS   = 30 * 60_000 # noisy detection window top
HORIZON_LONG_MS   = 4 * 3600_000
HORIZON_SHORT_MS  = 2 * 3600_000
FEE_BPS           = 5.0

BUCKET_SEC        = 300
MIN_GAP_SEC       = 900
ACCEL_WIN_SEC     = 30
LOOKBACK_SEC      = 3 * 3600    # how far back to look for fresh anchors
SIGNAL_FRESH_MS   = 120_000     # max age for new anchor to trigger
POLL_SEC          = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def ts_to_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ── DB helpers ────────────────────────────────────────────────────────────────

def _scalar(conn: sqlite3.Connection, sql: str, params: tuple) -> float:
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row else 0.0

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, lo, hi))

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def mark_at(conn, sym, ts_ms):
    row = conn.execute(
        "SELECT price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, ts_ms)).fetchone()
    return float(row[0]) if row else None

def prior_bps(conn, sym, ts_ms, lookback_ms):
    p0 = mark_at(conn, sym, ts_ms - lookback_ms)
    p1 = mark_at(conn, sym, ts_ms)
    if p0 and p1 and p0 > 0:
        return (p1 - p0) / p0 * 10_000.0
    return 0.0


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(conn, ts_ms: int, sync_k: float, n2h: int) -> int:
    """Score 0-6 at cascade time (no silence gate — added separately when confirmed)."""
    hour   = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    b4h    = prior_bps(conn, "BTCUSDT", ts_ms, 4 * 3600_000)
    book   = book_features_at(conn, "ETHUSDT", ts_ms, 30)
    vdep   = float(book.get("vdepth_bps") or 0) if book else 0.0
    sess_us = 13 <= hour < 21
    return sum([
        int(n2h >= 3),
        int(b4h < 0),
        int(vdep >= 30),
        int(sess_us),
        int(sync_k >= 200_000),
    ])  # silence adds +1 later = max 6


# ── State / ledger ────────────────────────────────────────────────────────────

def load_state() -> dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return {"positions": {}, "processed": {}, "pnl": {}}
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d.setdefault("positions", {})
    d.setdefault("processed", {})
    d.setdefault("pnl", {})
    return d

def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

def log_event(rec: dict[str, Any]) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")


# ── Per-anchor signal evaluation ──────────────────────────────────────────────

def open_signals_for_anchor(conn, anchor_ts_ms: int, anchor_price: float,
                             running_notional: float, now_ms: int, state: dict) -> list[dict]:
    ts = anchor_ts_ms
    base_id = f"SHD:{ts}"
    new_pos = []

    # Features knowable at cascade time
    sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts)
             + liq_sum(conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts))
    n2h    = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
    book   = book_features_at(conn, "ETHUSDT", ts, 30)
    bid_dep = float(book.get("bid_depth_usd") or 0) if book else 0.0
    score  = compute_score(conn, ts, sync_k, n2h)
    weekday = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday() < 5

    # 1 — LONG_SILENCE: enter all ≥200K events immediately; manage on silence/noisy
    lid = f"{base_id}:LS"
    if lid not in state["processed"]:
        pos = {
            "id": lid, "signal": "LONG_SILENCE", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": now_ms, "entry_price": anchor_price,
            "exit_due_ms": ts + HORIZON_LONG_MS,
            "status": "MONITORING",   # waiting for silence confirmation
            "sil_check_ms": ts + SIL_HI_MS,
            "sync_k": sync_k, "n2h": n2h, "bid_dep": bid_dep,
            "score": score, "weekday": weekday,
            "running_notional": running_notional,
            "opened_utc": utc_now(),
        }
        new_pos.append(pos)
        state["processed"][lid] = utc_now()

    # 2 — SHORT_NEITHER: ETH cascade + BTC cascade → SHORT immediately
    nid = f"{base_id}:SN"
    if nid not in state["processed"]:
        btc_now = liq_max(conn, "BTCUSDT", "SELL", ts, now_ms)
        if btc_now >= BTC_THRESH:
            pos = {
                "id": nid, "signal": "SHORT_NEITHER", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": now_ms, "entry_price": anchor_price,
                "exit_due_ms": now_ms + HORIZON_SHORT_MS,
                "status": "OPEN",
                "btc_max": btc_now,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][nid] = utc_now()
        elif now_ms < ts + SIL_HI_MS:
            # Queue for BTC monitoring
            pos = {
                "id": nid, "signal": "SHORT_NEITHER", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_BTC",
                "sil_check_ms": ts + SIL_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][nid] = utc_now()

    # 3 — SHORT_NOISY: first ETH SELL propagation ≥50K in 1-30min window
    snid = f"{base_id}:SP"
    if snid not in state["processed"] and now_ms < ts + NOISY_WIN_HI_MS + 5_000:
        prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                               ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH)
        if prop_ts is not None:
            px = mark_at(conn, "ETHUSDT", prop_ts) or anchor_price
            pos = {
                "id": snid, "signal": "SHORT_NOISY", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": prop_ts, "entry_price": px,
                "exit_due_ms": prop_ts + HORIZON_SHORT_MS,
                "status": "OPEN",
                "prop_delay_ms": prop_ts - ts,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][snid] = utc_now()
        elif now_ms < ts + NOISY_WIN_HI_MS:
            pos = {
                "id": snid, "signal": "SHORT_NOISY", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": None, "entry_price": None,
                "exit_due_ms": None,
                "status": "MONITORING_PROP",
                "sil_check_ms": ts + NOISY_WIN_HI_MS,
                "sync_k": sync_k, "n2h": n2h, "score": score,
                "opened_utc": utc_now(),
            }
            new_pos.append(pos)
            state["processed"][snid] = utc_now()

    return new_pos


# ── Position management ────────────────────────────────────────────────────────

def advance_positions(conn, state: dict, now_ms: int) -> None:
    closed = []
    for pid, pos in list(state["positions"].items()):
        sig    = pos["signal"]
        status = pos["status"]
        ts     = int(pos["anchor_ts_ms"])

        # ── LONG_SILENCE monitoring ───────────────────────────────────────────
        if sig == "LONG_SILENCE" and status == "MONITORING":
            # Ultra-early (<60s): abort
            ultra = liq_cnt(conn, "ETHUSDT", "SELL", ts, ts + SIL_LO_MS, PROP_THRESH)
            if ultra > 0:
                _close_pos(pos, conn, now_ms, "ABORT_ULTRA_EARLY")
                log_event({**pos, "event": "CLOSE"})
                closed.append(pid)
                print(f"{utc_now()} [SHD] LONG_SILENCE ABORT ultra-early id={pid}")
                continue
            # Noisy (1-30min cascade): close LONG early
            noisy_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                                    ts + SIL_LO_MS, min(now_ms, ts + SIL_HI_MS), PROP_THRESH)
            if noisy_ts is not None:
                _close_pos(pos, conn, now_ms, "NOISY_EARLY_EXIT")
                log_event({**pos, "event": "CLOSE"})
                closed.append(pid)
                print(f"{utc_now()} [SHD] LONG_SILENCE NOISY_EXIT net={pos.get('net_bps','?')} id={pid}")
                continue
            # Silence confirmed (30min window passed)
            if now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "OPEN_SILENCE"
                pos["silence_confirmed_utc"] = utc_now()
                pos["score"] = pos.get("score", 0) + 1   # +1 for silence gate
                pos["exit_due_ms"] = ts + HORIZON_LONG_MS
                log_event({**pos, "event": "SILENCE_CONFIRMED"})
                print(f"{utc_now()} [SHD] LONG_SILENCE SILENCE confirmed score={pos['score']} id={pid}")

        # ── SHORT_NEITHER BTC monitoring ──────────────────────────────────────
        elif sig == "SHORT_NEITHER" and status == "MONITORING_BTC":
            btc_max = liq_max(conn, "BTCUSDT", "SELL", ts, now_ms)
            if btc_max >= BTC_THRESH:
                px = mark_at(conn, "ETHUSDT", now_ms) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = now_ms
                pos["entry_price"] = px
                pos["exit_due_ms"] = now_ms + HORIZON_SHORT_MS
                pos["btc_max"] = btc_max
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_NEITHER OPEN id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + SIL_HI_MS)):
                pos["status"] = "EXPIRED_NO_BTC"
                closed.append(pid)

        # ── SHORT_NOISY propagation monitoring ────────────────────────────────
        elif sig == "SHORT_NOISY" and status == "MONITORING_PROP":
            prop_ts = liq_first_ts(conn, "ETHUSDT", "SELL",
                                   ts + SIL_LO_MS, min(now_ms, ts + NOISY_WIN_HI_MS), PROP_THRESH)
            if prop_ts is not None:
                px = mark_at(conn, "ETHUSDT", prop_ts) or 0.0
                pos["status"] = "OPEN"
                pos["entry_ts_ms"] = prop_ts
                pos["entry_price"] = px
                pos["exit_due_ms"] = prop_ts + HORIZON_SHORT_MS
                pos["prop_delay_ms"] = prop_ts - ts
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] SHORT_NOISY OPEN delay={prop_ts-ts:.0f}ms id={pid}")
            elif now_ms >= int(pos.get("sil_check_ms", ts + NOISY_WIN_HI_MS)):
                # Window passed, no propagation → it's a silence event, not noisy
                pos["status"] = "EXPIRED_SILENCE"
                closed.append(pid)

        # ── Time exit for all OPEN positions ──────────────────────────────────
        if pos.get("status") in {"OPEN", "OPEN_SILENCE"}:
            exit_due = pos.get("exit_due_ms")
            if exit_due and now_ms >= int(exit_due):
                _close_pos(pos, conn, now_ms, "TIME_EXIT")
                log_event({**pos, "event": "CLOSE"})
                closed.append(pid)
                print(f"{utc_now()} [SHD] {sig} TIME_EXIT net={pos.get('net_bps','?'):.1f}bps id={pid}")

    for pid in closed:
        state["positions"].pop(pid, None)


def _close_pos(pos: dict, conn, now_ms: int, reason: str) -> None:
    entry_px = float(pos.get("entry_price") or 0)
    exit_px  = mark_at(conn, "ETHUSDT", now_ms) or entry_px
    if entry_px > 0:
        raw_move = (exit_px - entry_px) / entry_px * 10_000.0
        outcome  = -raw_move if pos["direction"] == "SHORT" else raw_move
    else:
        outcome = 0.0
    pos["status"]      = f"CLOSED_{reason}"
    pos["close_reason"]= reason
    pos["exit_price"]  = exit_px
    pos["exit_ts_ms"]  = now_ms
    pos["closed_utc"]  = utc_now()
    pos["outcome_bps"] = round(outcome, 2)
    pos["net_bps"]     = round(outcome - FEE_BPS, 2)


# ── Stats ─────────────────────────────────────────────────────────────────────

def refresh_pnl(state: dict) -> None:
    if not LEDGER_PATH.exists():
        return
    by_sig: dict[str, list[float]] = {}
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "CLOSE":
                sig = r.get("signal", "?")
                net = r.get("net_bps")
                if net is not None:
                    by_sig.setdefault(sig, []).append(float(net))
    pnl: dict[str, Any] = {}
    for sig, vals in by_sig.items():
        wins = sum(1 for v in vals if v > 0)
        pnl[sig] = {
            "n": len(vals),
            "wins": wins,
            "wr": round(wins / len(vals), 3) if vals else 0,
            "total_net": round(sum(vals), 1),
            "avg_net": round(sum(vals) / len(vals), 1) if vals else 0,
        }
    pnl["updated_utc"] = utc_now()
    state["pnl"] = pnl


# ── Main ──────────────────────────────────────────────────────────────────────

def run_once(db_path: Path, state: dict) -> None:
    now_ms = int(time.time() * 1000)
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        # 1. Advance existing positions
        advance_positions(conn, state, now_ms)

        # 2. Detect fresh anchors
        start_ms = now_ms - LOOKBACK_SEC * 1000
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        anchors = reconstruct_anchors(
            liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
            thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_WIN_SEC,
        )
        marks = load_mark_index(conn, "ETHUSDT")
        for anchor in anchors:
            age_ms = now_ms - int(anchor.anchor_ts_ms)
            if not (0 <= age_ms <= SIGNAL_FRESH_MS):
                continue
            mark = marks.at_or_after(int(anchor.anchor_ts_ms))
            if not mark:
                continue
            anchor_price = float(mark[1])
            new_pos = open_signals_for_anchor(
                conn, int(anchor.anchor_ts_ms), anchor_price,
                float(anchor.running_notional), now_ms, state,
            )
            for pos in new_pos:
                state["positions"][pos["id"]] = pos
                log_event({**pos, "event": "OPEN"})
                print(f"{utc_now()} [SHD] OPENED {pos['signal']} status={pos['status']} id={pos['id']}")

    # 3. Refresh PnL stats
    refresh_pnl(state)
    save_state(state)


def print_status(state: dict) -> None:
    open_pos = {k: v for k, v in state["positions"].items()
                if v.get("status") not in {"EXPIRED_NO_BTC", "EXPIRED_SILENCE"}}
    print(f"\n=== Shadow Paper Status  {utc_now()} ===")
    print(f"Open positions: {len(open_pos)}")
    for pid, pos in open_pos.items():
        print(f"  {pos['signal']:20s} {pos['status']:20s}  score={pos.get('score',0)}"
              f"  sync_k={pos.get('sync_k',0):.0f}")
    pnl = state.get("pnl", {})
    if pnl:
        print("\nSignal P&L:")
        for sig, d in pnl.items():
            if sig == "updated_utc":
                continue
            print(f"  {sig:20s}  N={d['n']:4d}  WR={d['wr']:.1%}  total={d['total_net']:+.0f}bps"
                  f"  avg={d['avg_net']:+.1f}bps")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 Real-Time Shadow Runner")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=POLL_SEC)
    args = parser.parse_args()

    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    state = load_state()
    print(f"{utc_now()} S34 Real-Time Shadow Runner started  db={args.db}")
    print(f"  Signals: LONG_SILENCE, SHORT_NEITHER, SHORT_NOISY")
    print(f"  Ledger : {LEDGER_PATH}")
    print(f"  State  : {STATE_PATH}")

    if args.once:
        run_once(args.db, state)
        print_status(state)
        return 0

    while True:
        try:
            run_once(args.db, state)
        except Exception as exc:  # noqa: BLE001
            print(f"{utc_now()} ERROR: {type(exc).__name__}: {exc}")
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
