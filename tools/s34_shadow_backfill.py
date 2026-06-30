"""Backfill the real-time shadow bucket from historical NAV_EVENTS + DB.

Reads S34_NAVIGATION_EVENTS.jsonl (2006 events) and uses the liquidations DB
to compute silence gate classification for each event, then writes closed
trades to the shadow ledger. This makes the SBUCK panel show real P&L history.

One-shot historical replay — does NOT touch live orders.
Run once to populate:
    python tools/s34_shadow_backfill.py
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV_EVENTS  = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
LEDGER_PATH = ROOT / "reports" / "shadow" / "s34_realtime_shadow.jsonl"
STATE_PATH  = ROOT / "reports" / "shadow" / "s34_realtime_shadow_state.json"
DEFAULT_DB  = ROOT / "data" / "microstructure.db"

HOLDOUT_FRAC    = 0.30
FEE_BPS         = 5.0
SIL_GATE_LO_MS  = 60_000
SIL_GATE_HI_MS  = 30 * 60_000
BTC_THRESH      = 500_000.0
PROP_THRESH     = 50_000.0
LIVE_THRESH     = 200_000.0
SYNC_WIN_MS     = 10 * 60_000


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_events():
    rows = []
    with NAV_EVENTS.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def load_liq_arrays(conn, sym, side):
    rows = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND side=? ORDER BY ts_ms",
        (sym, side)
    ).fetchall()
    ts  = [int(r[0]) for r in rows]
    not_ = [float(r[1]) for r in rows]
    return ts, not_


def win_sum(ts, vals, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(vals[i] for i in range(a, b))


def win_max(ts, vals, lo, hi):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return max((vals[i] for i in range(a, b)), default=0.0)


def win_cnt(ts, vals, lo, hi, thr):
    a = bisect.bisect_left(ts, lo)
    b = bisect.bisect_right(ts, hi)
    return sum(1 for i in range(a, b) if vals[i] >= thr)


def classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not) -> dict:
    """Compute silence gate classification directly from DB arrays."""
    ts  = int(row["signal_ts_ms"])
    thr = float(row.get("threshold_usd") or 0)
    net2 = float(row.get("net_2h_bps") or "nan")
    net4v = row.get("net_4h_bps")
    net4 = float(net4v) if net4v is not None else net2

    # ETH silence: no follow-on >= 50K in 1min–30min
    n_prop  = win_cnt(eth_ts, eth_not, ts + SIL_GATE_LO_MS, ts + SIL_GATE_HI_MS, PROP_THRESH)
    sil_eth = n_prop == 0

    # BTC silence: no BTC SELL >= 500K in 1min–30min
    max_btc = win_max(btc_ts, btc_not, ts + SIL_GATE_LO_MS, ts + SIL_GATE_HI_MS)
    sil_btc = max_btc < BTC_THRESH

    # Bull check (from NAV_EVENTS tags)
    tags = row.get("tags") or []
    bull = "BULL_PULLBACK" in tags

    # Features for score
    b4h = float(row.get("btc4h_bps") or 0)
    vd  = float(row.get("vdepth_bps") or 0)
    bid = float(row.get("bid_depth_usd") or 0)
    ts_dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
    hour  = ts_dt.hour
    day   = ts_dt.weekday()
    sess_us = 13 <= hour < 21
    weekday = day < 5
    sync_k = (win_sum(btc_ts, btc_not, ts - SYNC_WIN_MS, ts)
             + win_sum(sol_ts, sol_not, ts - SYNC_WIN_MS, ts))
    n2h   = win_cnt(eth_ts, eth_not, ts - 2*3600_000, ts - 1000, PROP_THRESH)

    score = sum([
        int(sil_eth),
        int(n2h >= 3),
        int(b4h < 0),
        int(vd >= 30),
        int(sess_us),
        int(sync_k >= 200_000),
    ])

    return {
        "ts": ts, "thr": thr, "net2": net2, "net4": net4,
        "sil_eth": sil_eth, "sil_btc": sil_btc, "bull": bull,
        "score": score, "bid": bid, "weekday": weekday,
        "sync_k": sync_k, "n2h": n2h,
    }


def make_trade(eid, signal, direction, ts_ms, entry_price, outcome_bps, is_holdout, label="TIME_EXIT", score=0, sil=None):
    net = outcome_bps - FEE_BPS
    return {
        "id": eid, "signal": signal, "direction": direction,
        "anchor_ts_ms": ts_ms, "entry_ts_ms": ts_ms,
        "entry_price": entry_price,
        "status": f"CLOSED_{label}",
        "close_reason": label,
        "exit_price": None,
        "outcome_bps": round(outcome_bps, 2),
        "fee_bps": FEE_BPS,
        "net_bps": round(net, 2),
        "score": score,
        "silence_confirmed": sil,
        "is_holdout": is_holdout,
        "source": "BACKFILL",
        "event": "CLOSE",
    }


def main():
    events = load_events()
    if not events:
        print("No events found in NAV_EVENTS")
        return 1

    n_total = len(events)
    n_cal   = int(n_total * (1 - HOLDOUT_FRAC))

    print(f"Loaded {n_total} events  Cal={n_cal}  Hold={n_total-n_cal}")

    # Load liquidation arrays from DB for silence gate computation
    print("Loading liquidation arrays from DB...")
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq_arrays(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq_arrays(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq_arrays(conn, "SOLUSDT", "SELL")
    print(f"  ETH SELL: {len(eth_ts)} rows  BTC SELL: {len(btc_ts)}  SOL SELL: {len(sol_ts)}")

    # Remove all existing backfill entries so we can rewrite cleanly
    existing_ids: set[str] = set()
    non_backfill_lines: list[str] = []
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("source") == "BACKFILL":
                        existing_ids.add(r.get("id", ""))
                    else:
                        non_backfill_lines.append(line)
                except Exception:
                    non_backfill_lines.append(line)
    print(f"Existing backfill entries to replace: {len(existing_ids)}")
    # Rewrite keeping only non-backfill lines
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text("".join(non_backfill_lines), encoding="utf-8")

    written = 0
    written_ids: set[str] = set()

    with LEDGER_PATH.open("a", encoding="utf-8") as ledger:
        for i, row in enumerate(events):
            c = classify(row, eth_ts, eth_not, btc_ts, btc_not, sol_ts, sol_not)
            ts       = c["ts"]
            net2     = c["net2"]
            net4     = c["net4"]
            is_hold  = i >= n_cal

            if not math.isfinite(net2):
                continue
            if c["thr"] < LIVE_THRESH:
                continue

            base_id = f"SHD:{ts}"

            # 1. LONG_SILENCE
            if c["sil_eth"]:
                eid = f"{base_id}:LS"
                if eid not in written_ids:
                    outcome = net4  # silence events hold 4h
                    rec = make_trade(eid, "LONG_SILENCE", "LONG", ts,
                                     0.0, outcome, is_hold, "TIME_EXIT",
                                     c["score"], sil=True)
                    ledger.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written_ids.add(eid); written += 1

            # 2. SHORT_NEITHER (ETH noisy AND BTC noisy)
            if not c["sil_eth"] and not c["sil_btc"] and not c["bull"]:
                eid = f"{base_id}:SN"
                if eid not in written_ids:
                    outcome = -net2  # SHORT 2h
                    rec = make_trade(eid, "SHORT_NEITHER", "SHORT", ts,
                                     0.0, outcome, is_hold, "TIME_EXIT",
                                     c["score"])
                    ledger.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written_ids.add(eid); written += 1

            # 3. SHORT_NOISY (ETH noisy, not bull — broader than neither_silence)
            if not c["sil_eth"] and not c["bull"]:
                eid = f"{base_id}:SP"
                if eid not in written_ids:
                    outcome = -net2  # SHORT 2h
                    rec = make_trade(eid, "SHORT_NOISY", "SHORT", ts,
                                     0.0, outcome, is_hold, "TIME_EXIT",
                                     c["score"])
                    ledger.write(json.dumps(rec, ensure_ascii=True) + "\n")
                    written_ids.add(eid); written += 1

    print(f"Wrote {written} trades to {LEDGER_PATH}")

    # Recompute state PnL from ledger
    by_sig: dict = {}
    split_stats: dict = {"cal": {}, "hold": {}}
    with LEDGER_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") == "CLOSE":
                sig = r.get("signal", "?")
                net = r.get("net_bps")
                is_h = r.get("is_holdout", False)
                if net is not None and math.isfinite(float(net)):
                    v = float(net)
                    by_sig.setdefault(sig, []).append(v)
                    sk = "hold" if is_h else "cal"
                    split_stats[sk].setdefault(sig, []).append(v)

    print("\n=== Shadow Bucket P&L ===")
    print(f"{'Signal':<25} {'N':>5} {'WR':>7} {'Total bps':>10} {'Avg bps':>8} {'Cal WR':>8} {'Hold WR':>8}")
    print("-" * 80)
    for sig, vals in sorted(by_sig.items()):
        wins = sum(1 for v in vals if v > 0)
        wr   = wins / len(vals) if vals else 0
        cv   = split_stats["cal"].get(sig, [])
        hv   = split_stats["hold"].get(sig, [])
        cwr  = sum(1 for v in cv if v > 0) / len(cv) if cv else 0
        hwr  = sum(1 for v in hv if v > 0) / len(hv) if hv else 0
        print(f"{sig:<25} {len(vals):>5} {wr:>7.1%} {sum(vals):>+10.0f} {sum(vals)/len(vals):>+8.1f}  {cwr:>7.1%}  {hwr:>7.1%}")

    # Update state with pnl summary
    state: dict = {"positions": {}, "processed": {}, "pnl": {}}
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    pnl_summary: dict = {}
    for sig, vals in by_sig.items():
        wins = sum(1 for v in vals if v > 0)
        pnl_summary[sig] = {
            "n": len(vals),
            "wins": wins,
            "wr": round(wins/len(vals), 3) if vals else 0,
            "total_net": round(sum(vals), 1),
            "avg_net": round(sum(vals)/len(vals), 1) if vals else 0,
        }
    pnl_summary["updated_utc"] = utc_now()
    state["pnl"] = pnl_summary
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nState updated: {STATE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
