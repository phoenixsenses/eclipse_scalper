"""
One-shot backfill for the S34 State Machine shadow ledger.

Simulates LONG_SILENCE + SHORT_NEITHER outcomes for every ETH SELL >=200K
cascade since a given start date, then appends OPEN + CLOSE records to the
same ledger used by s34_realtime_shadow_runner.py.

Run once after the shadow runner starts fresh (0 trades). It won't re-process
cascades that are already in state["processed"].

Usage:
    python tools/backfill_s34_state_machine_shadow.py [--days N] [--db PATH]
"""

from __future__ import annotations

import argparse
import json
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
    reconstruct_anchors,
)

DEFAULT_DB   = ROOT / "data" / "microstructure.db"
LEDGER_PATH  = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
STATE_PATH   = ROOT / "reports" / "shadow" / "s34_state_machine_shadow_state.json"
RULE_NAME    = "S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3"

ETH_THRESH      = 200_000.0
PROP_THRESH     = 50_000.0
BTC_THRESH      = 1_000_000.0
SYNC_WIN_MS     = 10 * 60_000
SIL_LO_MS       = 60_000
SIL_HI_MS       = 30 * 60_000
HORIZON_LONG_MS = 4 * 3600_000
HORIZON_SHORT_MS= 2 * 3600_000
FEE_BPS         = 5.0
BUCKET_SEC      = 300
MIN_GAP_SEC     = 900
ACCEL_WIN_SEC   = 30


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def mark_at_or_after(conn: sqlite3.Connection, sym: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (sym, int(ts_ms)),
    ).fetchone()
    return float(row[0]) if row else None


def mark_at_or_before(conn: sqlite3.Connection, sym: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (sym, int(ts_ms)),
    ).fetchone()
    return float(row[0]) if row else None


def mark_at(conn: sqlite3.Connection, sym: str, ts_ms: int) -> float | None:
    return mark_at_or_after(conn, sym, ts_ms) or mark_at_or_before(conn, sym, ts_ms)


def liq_sum(conn: sqlite3.Connection, sym: str, side: str, lo: int, hi: int) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, int(lo), int(hi)),
    ).fetchone()
    return float(row[0] or 0.0) if row else 0.0


def liq_cnt(conn: sqlite3.Connection, sym: str, side: str, lo: int, hi: int, thr: float) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, int(lo), int(hi), float(thr)),
    ).fetchone()
    return int(row[0] or 0) if row else 0


def liq_max(conn: sqlite3.Connection, sym: str, side: str, lo: int, hi: int) -> float:
    row = conn.execute(
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
        (sym, side, int(lo), int(hi)),
    ).fetchone()
    return float(row[0] or 0.0) if row else 0.0


def liq_first_ts(conn: sqlite3.Connection, sym: str, side: str, lo: int, hi: int, thr: float) -> int | None:
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, int(lo), int(hi), float(thr)),
    ).fetchone()
    return int(row[0]) if row else None


def mark_ret_bps(conn: sqlite3.Connection, sym: str, t0: int, t1: int) -> float | None:
    p0 = mark_at(conn, sym, t0)
    p1 = mark_at(conn, sym, t1)
    if not p0 or not p1 or p0 <= 0:
        return None
    return (p1 - p0) / p0 * 10_000.0


def session_name(ts_ms: int) -> str:
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if h < 7:   return "ASIA"
    if h < 13:  return "EUROPE"
    if h < 21:  return "US"
    return "OFF"




def is_bull_pullback(conn: sqlite3.Connection, ts_ms: int) -> bool:
    eth1h = mark_ret_bps(conn, "ETHUSDT", ts_ms - 3600_000, ts_ms)
    btc4h = mark_ret_bps(conn, "BTCUSDT", ts_ms - 4 * 3600_000, ts_ms)
    if eth1h is None or btc4h is None:
        return False
    return eth1h > 20.0 and btc4h > 50.0


def pnl_bps(entry_px: float, exit_px: float, direction: str) -> float:
    if entry_px <= 0:
        return 0.0
    raw = (exit_px - entry_px) / entry_px * 10_000.0
    return -raw if direction == "SHORT" else raw


def load_state() -> dict:
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


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_ledger(rec: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def simulate_cascade(conn: sqlite3.Connection, ts: int, running_notional: float, state: dict) -> dict[str, Any]:
    """Simulate both LONG_SILENCE and SHORT_NEITHER for one cascade. Returns summary."""
    base_id = f"BF:{ts}"
    lid  = f"{base_id}:LS"
    nid  = f"{base_id}:SN"

    sync_k = liq_sum(conn, "BTCUSDT", "SELL", ts - SYNC_WIN_MS, ts) + \
             liq_sum(conn, "SOLUSDT", "SELL", ts - SYNC_WIN_MS, ts)
    n2h = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2 * 3600_000, ts - 1000, PROP_THRESH)
    btc4h = mark_ret_bps(conn, "BTCUSDT", ts - 4 * 3600_000, ts)
    sess  = session_name(ts)
    dt    = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
    dow   = dt.weekday()
    bull  = is_bull_pullback(conn, ts)

    # vdepth skipped in backfill (no marks index per-cascade); score conservative by 0-1
    score = sum([
        int(n2h >= 3),
        int(btc4h is not None and btc4h < 0.0),
        0,
        int(sess == "US"),
        int(sync_k >= 200_000.0),
    ])
    long_score    = score + 1
    long_eligible  = (not bull) and sess != "EUROPE" and dow not in {0, 2} and long_score >= 3
    short_eligible = (not bull) and dow != 6 and score >= 3

    anchor_px = mark_at(conn, "ETHUSDT", ts) or 0.0
    results: list[dict] = []

    # ── LONG_SILENCE ────────────────────────────────────────────────────────────
    if long_eligible and lid not in state["processed"] and anchor_px > 0:
        open_rec = {
            "id": lid, "signal": "LONG_SILENCE", "direction": "LONG",
            "anchor_ts_ms": ts, "entry_ts_ms": ts, "entry_price": anchor_px,
            "exit_due_ms": ts + HORIZON_LONG_MS,
            "status": "MONITORING",
            "sil_check_ms": ts + SIL_HI_MS,
            "sync_k": round(sync_k, 0), "n2h": n2h, "score": score,
            "long_score": long_score, "dow": dow, "session": sess,
            "bull_pullback": bull, "rule_name": RULE_NAME,
            "running_notional": running_notional, "backfill": True,
            "opened_utc": utc(ts), "event": "OPEN",
        }
        append_ledger(open_rec)
        state["processed"][lid] = utc(ts)

        # Matches live executor manage_pending logic exactly:
        # - Skip first 60s (SIL_LO_MS) — executor does `if hi < ts+SIL_LO_MS: continue`
        # - After 60s: ETH follow-on >= PROP_THRESH -> NOISY_EARLY_EXIT
        # - After 30min (SIL_HI_MS): SILENCE CONFIRMED -> hold to 4h
        # No ABORT_ULTRA_EARLY — live executor has no such check
        noisy_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)
        if noisy_ts is not None:
            exit_ts = noisy_ts
            reason = "NOISY_EARLY_EXIT"
        else:
            # silence confirmed — hold to T+4h
            exit_ts = ts + HORIZON_LONG_MS
            reason = "TIME_EXIT"
            open_rec["status"] = "OPEN_SILENCE"
            open_rec["silence_confirmed_utc"] = utc(ts + SIL_HI_MS)

        exit_px  = mark_at(conn, "ETHUSDT", exit_ts) or anchor_px
        outcome  = pnl_bps(anchor_px, exit_px, "LONG")
        close_rec = {
            **open_rec,
            "event": "CLOSE",
            "status": f"CLOSED_{reason}",
            "close_reason": reason,
            "exit_price": exit_px, "exit_ts_ms": exit_ts,
            "closed_utc": utc(exit_ts),
            "outcome_bps": round(outcome, 2),
            "net_bps": round(outcome - FEE_BPS, 2),
        }
        append_ledger(close_rec)
        results.append({"signal": "LONG_SILENCE", "net_bps": close_rec["net_bps"], "reason": reason})

    # ── SHORT_NEITHER ────────────────────────────────────────────────────────────
    if short_eligible and nid not in state["processed"] and anchor_px > 0:
        btc_max = liq_max(conn, "BTCUSDT", "SELL", ts + SIL_LO_MS, ts + SIL_HI_MS)
        if btc_max >= BTC_THRESH:
            btc_ts = liq_first_ts(conn, "BTCUSDT", "SELL", ts + SIL_LO_MS, ts + SIL_HI_MS, BTC_THRESH)
            entry_ts  = btc_ts if btc_ts else ts + SIL_LO_MS
            entry_px  = mark_at(conn, "ETHUSDT", entry_ts) or anchor_px
            exit_ts   = entry_ts + HORIZON_SHORT_MS
            exit_px   = mark_at(conn, "ETHUSDT", exit_ts) or entry_px
            outcome   = pnl_bps(entry_px, exit_px, "SHORT")

            open_rec = {
                "id": nid, "signal": "SHORT_NEITHER", "direction": "SHORT",
                "anchor_ts_ms": ts, "entry_ts_ms": entry_ts, "entry_price": entry_px,
                "exit_due_ms": exit_ts, "status": "OPEN",
                "btc_max": btc_max,
                "sync_k": round(sync_k, 0), "n2h": n2h, "score": score,
                "dow": dow, "session": sess, "bull_pullback": bull,
                "rule_name": RULE_NAME, "backfill": True,
                "opened_utc": utc(entry_ts), "event": "OPEN",
            }
            close_rec = {
                **open_rec,
                "event": "CLOSE",
                "status": "CLOSED_TIME_EXIT",
                "close_reason": "TIME_EXIT",
                "exit_price": exit_px, "exit_ts_ms": exit_ts,
                "closed_utc": utc(exit_ts),
                "outcome_bps": round(outcome, 2),
                "net_bps": round(outcome - FEE_BPS, 2),
            }
            append_ledger(open_rec)
            append_ledger(close_rec)
            state["processed"][nid] = utc(ts)
            results.append({"signal": "SHORT_NEITHER", "net_bps": close_rec["net_bps"], "reason": "TIME_EXIT"})
        else:
            state["processed"][nid] = utc(ts)

    return {
        "ts": ts, "anchor_utc": utc(ts), "score": score, "long_eligible": long_eligible,
        "short_eligible": short_eligible, "session": sess, "dow": dow,
        "results": results,
    }


def refresh_pnl(state: dict) -> None:
    if not LEDGER_PATH.exists():
        return
    by_sig: dict[str, list[float]] = {}
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("event") != "CLOSE":
                continue
            sig = rec.get("signal")
            net = rec.get("net_bps")
            if sig and net is not None:
                by_sig.setdefault(sig, []).append(float(net))
    pnl: dict[str, Any] = {}
    for sig, vals in by_sig.items():
        wins = [v for v in vals if v > 0]
        pnl[sig] = {
            "n": len(vals),
            "wr": round(len(wins) / len(vals), 4) if vals else 0.0,
            "total_net": round(sum(vals), 1),
            "avg_net": round(sum(vals) / len(vals), 1) if vals else 0.0,
            "win_avg": round(sum(wins) / len(wins), 1) if wins else None,
        }
    pnl["updated_utc"] = datetime.now(timezone.utc).isoformat()
    state["pnl"] = pnl


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill S34 state machine shadow ledger")
    parser.add_argument("--days", type=int, default=35, help="How many days back to process (default: 35)")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - args.days * 86400 * 1000

    print(f"Backfill S34 State Machine Shadow")
    print(f"  DB: {args.db}")
    print(f"  From: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).isoformat()}")
    print(f"  To:   {datetime.fromtimestamp(now_ms/1000, tz=timezone.utc).isoformat()}")

    state = load_state()
    already_processed = len(state["processed"])
    print(f"  Already processed: {already_processed} signals")

    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=BUCKET_SEC,
            min_gap_sec=MIN_GAP_SEC,
            thresholds=(ETH_THRESH,),
            accel_window_sec=ACCEL_WIN_SEC,
        )
        print(f"  Cascades found: {len(anchors)}")

        long_results: list[float] = []
        short_results: list[float] = []
        skipped = 0

        for i, anchor in enumerate(anchors):
            ts = int(anchor.anchor_ts_ms)
            base_id = f"BF:{ts}"
            # Skip if future (shouldn't happen) or if already processed
            if ts > now_ms - HORIZON_LONG_MS:
                # skip very recent ones that haven't had time to close
                skipped += 1
                continue

            summary = simulate_cascade(conn, ts, float(anchor.running_notional), state)
            for r in summary["results"]:
                if r["signal"] == "LONG_SILENCE":
                    long_results.append(r["net_bps"])
                elif r["signal"] == "SHORT_NEITHER":
                    short_results.append(r["net_bps"])

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(anchors)}] ts={datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime('%m/%d %H:%M')} "
                      f"ls_n={len(long_results)} sn_n={len(short_results)}")

    print(f"\n=== Backfill Complete ===")
    print(f"Skipped (too recent): {skipped}")
    if long_results:
        wins = [v for v in long_results if v > 0]
        print(f"LONG_SILENCE  n={len(long_results)} wr={len(wins)/len(long_results):.1%} "
              f"avg={sum(long_results)/len(long_results):.1f}bps total={sum(long_results):.0f}bps")
    if short_results:
        wins = [v for v in short_results if v > 0]
        print(f"SHORT_NEITHER n={len(short_results)} wr={len(wins)/len(short_results):.1%} "
              f"avg={sum(short_results)/len(short_results):.1f}bps total={sum(short_results):.0f}bps")

    refresh_pnl(state)
    save_state(state)
    print(f"State saved -> {STATE_PATH}")
    print(f"Ledger: {LEDGER_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
