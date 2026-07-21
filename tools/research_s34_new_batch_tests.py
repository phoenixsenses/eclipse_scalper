"""S34 New Batch Tests — 13 targeted tests on backfill ledger + DB.

Tests:
  T01 LONG score gate (long_score ≥3 vs ≥4 vs ≥5)
  T02 LONG n2h hard gate (≥2 vs ≥3 vs ≥4)
  T03 LONG session breakdown
  T04 LONG DOW breakdown
  T05 LONG monthly stability
  T06 LONG cascade spacing filter (<2h vs ≥2h since last event)
  T07 LONG sync_k hard gate (200K / 500K / 1M)
  T08 LONG exit-reason split (TIME_EXIT vs NOISY_EARLY_EXIT)
  T09 SHORT score gate (≥3 vs ≥4)
  T10 SHORT session breakdown (with vs without Europe)
  T11 SHORT monthly stability
  T12 [DB] BTC 4h hard gate for LONG (require btc4h_bps < 0)
  T13 [DB] SIL_LO=3min — how many NOISY events fired in [60s,180s]?

Usage: python tools/research_s34_new_batch_tests.py
"""
from __future__ import annotations
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"
FEE_BPS = 5.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "total": None}
    wins = sum(1 for v in vals if v > 0)
    return {
        "n": len(vals),
        "wr": round(wins / len(vals), 3),
        "avg": round(sum(vals) / len(vals), 1),
        "total": round(sum(vals), 0),
    }

def pct(v) -> str:
    if v is None: return "  -  "
    return f"{v*100:5.1f}%"

def fmt(v, digits=1) -> str:
    if v is None: return "   -   "
    return f"{v:+{7+digits}.{digits}f}"

def hdr(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def row(label: str, s: dict, note: str = "") -> None:
    if s["n"] == 0:
        print(f"  {label:<28s}  N=  0  ----  ------  ------  {note}")
        return
    print(f"  {label:<28s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

# ── Load ledger ───────────────────────────────────────────────────────────────

def load_close_records() -> tuple[list[dict], list[dict]]:
    longs, shorts = [], []
    if not LEDGER.exists():
        return longs, shorts
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("event") != "CLOSE":
                continue
            net = r.get("net_bps")
            if net is None:
                continue
            r["_net"] = float(net)
            if r.get("direction") == "LONG":
                longs.append(r)
            elif r.get("direction") == "SHORT":
                shorts.append(r)
    return longs, shorts

# ── DB helpers ────────────────────────────────────────────────────────────────

def btc4h_bps_at(conn: sqlite3.Connection, ts_ms: int) -> float | None:
    lo = ts_ms - 4 * 3600_000
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (lo,)
    ).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (ts_ms,)
    ).fetchone()
    if not a or not b or float(a[0] or 0) <= 0:
        return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def eth_sell_first_after(conn: sqlite3.Connection, ts_ms: int, lo_offset_ms: int, hi_offset_ms: int, thresh: float) -> int | None:
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms + lo_offset_ms, ts_ms + hi_offset_ms, thresh)
    ).fetchone()
    return int(row[0]) if row else None

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    longs, shorts = load_close_records()
    print(f"\nLoaded: {len(longs)} LONG CLOSE, {len(shorts)} SHORT CLOSE records")

    longs.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))
    shorts.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))

    # ── T01 LONG score gate ────────────────────────────────────────────────────
    hdr("T01 · LONG long_score threshold (currently ≥3)")
    for thr in [3, 4, 5]:
        sub = [r["_net"] for r in longs if (r.get("long_score") or 0) >= thr]
        row(f"long_score >= {thr}", stat(sub))

    # ── T02 LONG n2h hard gate ────────────────────────────────────────────────
    hdr("T02 · LONG n2h hard gate (currently ≥2 for score component)")
    for thr in [0, 1, 2, 3, 4]:
        sub = [r["_net"] for r in longs if (r.get("n2h") or 0) >= thr]
        row(f"n2h >= {thr}", stat(sub))

    # ── T03 LONG session breakdown ────────────────────────────────────────────
    hdr("T03 · LONG session breakdown")
    for sess in ["US", "ASIA", "EUROPE", "OFF"]:
        sub = [r["_net"] for r in longs if r.get("session") == sess]
        row(sess, stat(sub))

    # ── T04 LONG DOW breakdown ────────────────────────────────────────────────
    hdr("T04 · LONG DOW breakdown (0=Mon … 6=Sun)")
    dow_names = {0:"Mon(0)", 1:"Tue(1)", 2:"Wed(2)", 3:"Thu(3)", 4:"Fri(4)", 5:"Sat(5)", 6:"Sun(6)"}
    for d in range(7):
        sub = [r["_net"] for r in longs if r.get("dow") == d]
        note = " ← EXCLUDED" if d in {0, 2} else ""
        row(dow_names[d], stat(sub), note)

    # ── T05 LONG monthly stability ────────────────────────────────────────────
    hdr("T05 · LONG monthly stability")
    months: dict[str, list[float]] = {}
    for r in longs:
        try:
            dt = datetime.fromtimestamp(int(r["anchor_ts_ms"]) / 1000, tz=timezone.utc)
            key = dt.strftime("%Y-%m")
        except Exception:
            key = "?"
        months.setdefault(key, []).append(r["_net"])
    for m in sorted(months):
        row(m, stat(months[m]))

    # ── T06 LONG cascade spacing filter ──────────────────────────────────────
    hdr("T06 · LONG cascade spacing filter (skip if prev event <2h ago)")
    GAP_MS = 2 * 3600_000
    with_gap, too_close = [], []
    prev_ts = None
    for r in longs:
        ts = int(r.get("anchor_ts_ms") or 0)
        gap_ok = (prev_ts is None) or (ts - prev_ts >= GAP_MS)
        if gap_ok:
            with_gap.append(r["_net"])
        else:
            too_close.append(r["_net"])
        prev_ts = ts
    row("gap >= 2h (pass)", stat(with_gap))
    row("gap <  2h (filtered)", stat(too_close))

    # ── T07 LONG sync_k hard gate ─────────────────────────────────────────────
    hdr("T07 · LONG sync_k hard gate (BTC+SOL prior 10min)")
    for thr_k in [0, 200, 500, 1000]:
        sub = [r["_net"] for r in longs if (r.get("sync_k") or 0) >= thr_k * 1000]
        row(f"sync_k >= {thr_k}K", stat(sub))

    # ── T08 LONG exit-reason split ────────────────────────────────────────────
    hdr("T08 · LONG exit reason split (TIME_EXIT = hold-4h policy)")
    for reason in ["TIME_EXIT", "NOISY_EARLY_EXIT"]:
        sub = [r["_net"] for r in longs if r.get("close_reason") == reason]
        row(reason, stat(sub))
    # Combined: what if ALL longs ran to time exit?
    # We can't know the 4h outcome for noisy exits without DB, but show the split clearly
    print(f"\n  NOTE: Forward behavior = TIME_EXIT only (noisy exit removed from live+shadow)")

    # ── T09 SHORT score gate ──────────────────────────────────────────────────
    hdr("T09 · SHORT score threshold (currently ≥3)")
    for thr in [3, 4]:
        sub = [r["_net"] for r in shorts if (r.get("score") or 0) >= thr]
        row(f"score >= {thr}", stat(sub))

    # ── T10 SHORT session breakdown ───────────────────────────────────────────
    hdr("T10 · SHORT session breakdown (Europe filtered since this session)")
    for sess in ["US", "ASIA", "EUROPE", "OFF"]:
        sub = [r["_net"] for r in shorts if r.get("session") == sess]
        note = " ← NOW FILTERED" if sess == "EUROPE" else ""
        row(sess, stat(sub), note)
    print()
    non_europe = [r["_net"] for r in shorts if r.get("session") != "EUROPE"]
    row("non-EUROPE total", stat(non_europe))

    # ── T11 SHORT monthly stability ───────────────────────────────────────────
    hdr("T11 · SHORT monthly stability")
    s_months: dict[str, list[float]] = {}
    for r in shorts:
        try:
            dt = datetime.fromtimestamp(int(r["anchor_ts_ms"]) / 1000, tz=timezone.utc)
            key = dt.strftime("%Y-%m")
        except Exception:
            key = "?"
        s_months.setdefault(key, []).append(r["_net"])
    for m in sorted(s_months):
        row(m, stat(s_months[m]))

    # ── T12 [DB] BTC 4h hard gate for LONG ───────────────────────────────────
    hdr("T12 · [DB] BTC 4h hard gate for LONG (btc4h_bps < 0)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=5) as conn:
            below, above = [], []
            missing = 0
            for r in longs:
                ts = int(r.get("anchor_ts_ms") or 0)
                b4h = btc4h_bps_at(conn, ts)
                if b4h is None:
                    missing += 1
                    continue
                r["_btc4h"] = b4h
                if b4h < 0:
                    below.append(r["_net"])
                else:
                    above.append(r["_net"])
            row("btc4h_bps < 0 (gate pass)", stat(below), f"(missing={missing})")
            row("btc4h_bps >= 0 (gate fail)", stat(above))
            # Score ≥4 AND btc4h<0
            score4_and_bear = [r["_net"] for r in longs
                               if r.get("_btc4h") is not None and r["_btc4h"] < 0
                               and (r.get("long_score") or 0) >= 4]
            row("long_score>=4 AND btc4h<0", stat(score4_and_bear))
            # Monthly for btc4h<0 gate
            print()
            b4h_months: dict[str, list[float]] = {}
            for r in longs:
                if r.get("_btc4h") is None or r["_btc4h"] >= 0:
                    continue
                try:
                    dt = datetime.fromtimestamp(int(r["anchor_ts_ms"]) / 1000, tz=timezone.utc)
                    key = dt.strftime("%Y-%m")
                except Exception:
                    key = "?"
                b4h_months.setdefault(key, []).append(r["_net"])
            print("  btc4h<0 gate monthly breakdown:")
            for m in sorted(b4h_months):
                row(f"  {m}", stat(b4h_months[m]))
    except Exception as exc:
        print(f"  [DB ERROR] {exc}")

    # ── T13 [DB] SIL_LO=3min impact ───────────────────────────────────────────
    hdr("T13 · [DB] SIL_LO=3min — noisy events rescued by wider tail gap")
    # Current SIL_LO=60s. Ask: for NOISY_EARLY_EXIT events, did the noisy trigger
    # fire in the first 3 minutes [T+60s, T+180s]? If yes → moving to 3min would
    # reclassify these as silence (no follow-on in [T+180s, T+30min]).
    # We then need to check whether [T+180s, T+30min] is actually silence.
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=5) as conn:
            noisy_exits = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]
            in_tail = []      # noisy trigger in [60s, 180s] — tail contamination
            true_noisy = []   # noisy trigger after 180s — real follow-on
            tail_still_noisy = 0   # in_tail events that still have follow-on in [180s, 30min]
            tail_would_silence = 0  # in_tail events with no follow-on in [180s, 30min]
            THRESH = 50_000.0
            SIL_HI_MS = 30 * 60_000
            for r in noisy_exits:
                ts = int(r.get("anchor_ts_ms") or 0)
                # Check if noisy trigger was in [60s, 180s]
                early = eth_sell_first_after(conn, ts, 60_000, 180_000, THRESH)
                if early is not None:
                    in_tail.append(r)
                    # Would [180s, 30min] also be noisy?
                    late = eth_sell_first_after(conn, ts, 180_000, SIL_HI_MS, THRESH)
                    if late is not None:
                        tail_still_noisy += 1
                    else:
                        tail_would_silence += 1
                else:
                    true_noisy.append(r)

            noisy_n = len(noisy_exits)
            print(f"  NOISY_EARLY_EXIT total     : {noisy_n}")
            print(f"  Trigger in [60s,180s] (tail): {len(in_tail)}  ({100*len(in_tail)/max(noisy_n,1):.0f}%)")
            print(f"    → still noisy in [180s,30m]: {tail_still_noisy}")
            print(f"    → would be SILENCE at 3min : {tail_would_silence}")
            print(f"  Trigger after 180s (real)   : {len(true_noisy)}")
            print()
            print(f"  Implication: SIL_LO=3min would rescue {tail_would_silence} events")
            print(f"  (They'd hold to T+4h instead of early exit)")
            if tail_would_silence > 0:
                rescued = [r["_net"] for r in in_tail]
                row("  Tail events (actual early exit net)", stat(rescued))
                print(f"  NOTE: Forward 4h outcome for these {tail_would_silence} events needs DB mark prices")

    except Exception as exc:
        print(f"  [DB ERROR] {exc}")

    print()
    print("=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
