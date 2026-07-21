"""S34 Round-2 Tests — Forward-relevant deep dive after batch-1 findings.

Key context from batch-1:
  - TIME_EXIT LONG: N=72, WR=68.1%, avg=+47.9 bps (forward baseline — noisy exit removed)
  - NOISY_EARLY_EXIT: 145 trades dragging total avg to -1.5 bps (no longer happening)
  - US session LONG: avg=-7.7 bps (but mixed with noisy exits — need to re-check)
  - sync_k HIGH hurts LONG (backwards hypothesis)
  - SHORT score>=4: WR=80%, avg=+116.6 bps
  - n2h>=3 adds +6.1 bps to LONG avg

Tests:
  A1-A6  TIME_EXIT only breakdown (forward-relevant population)
  B1-B5  Combined gate scenarios for LONG
  C1-C5  SHORT deep dive (btc_max, n2h, sync_k, entry delay, score+session)
  D1-D2  April regime detection (DB: btc7d trend)
  E1-E2  Score component deconstruction
  F1     Permutation null check on TIME_EXIT subset

Usage: python tools/research_s34_round2_tests.py
"""
from __future__ import annotations
import json
import math
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"
FEE_BPS = 5.0
RANDOM_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "total": None}
    wins = sum(1 for v in vals if v > 0)
    return {"n": len(vals), "wr": round(wins/len(vals),3),
            "avg": round(sum(vals)/len(vals),1), "total": round(sum(vals),0)}

def pct(v) -> str:
    return "  -  " if v is None else f"{v*100:5.1f}%"

def fmt(v, d=1) -> str:
    return "   -   " if v is None else f"{v:+{7+d}.{d}f}"

def hdr(title: str) -> None:
    print(); print("=" * 64); print(f"  {title}"); print("=" * 64)

def row(label: str, s: dict, note: str = "") -> None:
    if s["n"] == 0:
        print(f"  {label:<32s}  N=  0  -----  ------  ------  {note}")
        return
    print(f"  {label:<32s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def month_of(r: dict) -> str:
    try:
        return datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
    except Exception:
        return "?"

# ── DB helpers ────────────────────────────────────────────────────────────────

def btc_ret_at(conn, ts_ms: int, lookback_ms: int) -> float | None:
    lo = ts_ms - lookback_ms
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (lo,)).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0) <= 0:
        return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def btc_first_after(conn, ts_ms: int, lo_off: int, hi_off: int, thresh: float):
    r = conn.execute(
        "SELECT ts_ms, notional FROM liquidations "
        "WHERE symbol='BTCUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND notional>=? "
        "ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms+lo_off, ts_ms+hi_off, thresh)).fetchone()
    return (int(r[0]), float(r[1])) if r else None

# ── Load ledger ───────────────────────────────────────────────────────────────

def load_records():
    longs, shorts = [], []
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            if r.get("event") != "CLOSE": continue
            net = r.get("net_bps")
            if net is None: continue
            r["_net"] = float(net)
            r["_month"] = month_of(r)
            if r.get("direction") == "LONG":
                longs.append(r)
            elif r.get("direction") == "SHORT":
                shorts.append(r)
    longs.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))
    shorts.sort(key=lambda r: int(r.get("anchor_ts_ms") or 0))
    return longs, shorts

# ── SECTION A: TIME_EXIT only (forward-relevant) ──────────────────────────────

def section_a(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]
    ne = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]

    hdr("A0 · Recap: TIME_EXIT vs NOISY populations")
    row("TIME_EXIT  (forward baseline)", stat([r["_net"] for r in te]))
    row("NOISY_EARLY_EXIT (no longer fires)", stat([r["_net"] for r in ne]))

    hdr("A1 · TIME_EXIT LONG by session")
    for sess in ["US","ASIA","OFF","EUROPE"]:
        sub = [r["_net"] for r in te if r.get("session")==sess]
        row(sess, stat(sub))

    hdr("A2 · TIME_EXIT LONG by DOW")
    dnames = {0:"Mon(0)",1:"Tue(1)",2:"Wed(2)",3:"Thu(3)",4:"Fri(4)",5:"Sat(5)",6:"Sun(6)"}
    for d in range(7):
        sub = [r["_net"] for r in te if r.get("dow")==d]
        note = " ← EXCLUDED" if d in {0,2} else ""
        row(dnames[d], stat(sub), note)

    hdr("A3 · TIME_EXIT LONG by month")
    months: dict[str, list] = {}
    for r in te:
        months.setdefault(r["_month"],[]).append(r["_net"])
    for m in sorted(months):
        row(m, stat(months[m]))

    hdr("A4 · TIME_EXIT LONG by n2h gate")
    for thr in [0,1,2,3,4]:
        sub = [r["_net"] for r in te if (r.get("n2h") or 0) >= thr]
        row(f"n2h >= {thr}", stat(sub))

    hdr("A5 · TIME_EXIT LONG by sync_k (revisit: does low sync = better?)")
    buckets = [(0,200,"sync < 200K"),(200,500,"sync 200-500K"),(500,1000,"sync 500K-1M"),
               (1000,999999,"sync >= 1M")]
    for lo,hi,label in buckets:
        sub = [r["_net"] for r in te if lo*1000 <= (r.get("sync_k") or 0) < hi*1000]
        row(label, stat(sub))
    # Hard gate variations
    print()
    for thr in [0,200,500,1000]:
        sub = [r["_net"] for r in te if (r.get("sync_k") or 0) >= thr*1000]
        row(f"sync >= {thr}K", stat(sub))

    hdr("A6 · TIME_EXIT LONG by long_score")
    for thr in [3,4,5]:
        sub = [r["_net"] for r in te if (r.get("long_score") or 0) >= thr]
        row(f"long_score >= {thr}", stat(sub))

# ── SECTION B: Combined gates ─────────────────────────────────────────────────

def section_b(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("B1 · Combined LONG gates (TIME_EXIT only — forward scenario)")
    combos = [
        ("All TIME_EXIT (baseline)",
         lambda r: True),
        ("!US session",
         lambda r: r.get("session") != "US"),
        ("n2h >= 3",
         lambda r: (r.get("n2h") or 0) >= 3),
        ("!US AND n2h>=3",
         lambda r: r.get("session") != "US" and (r.get("n2h") or 0) >= 3),
        ("!US AND long_score>=4",
         lambda r: r.get("session") != "US" and (r.get("long_score") or 0) >= 4),
        ("!US AND n2h>=3 AND long_score>=4",
         lambda r: r.get("session")!="US" and (r.get("n2h") or 0)>=3 and (r.get("long_score") or 0)>=4),
        ("ASIA or OFF only",
         lambda r: r.get("session") in {"ASIA","OFF"}),
        ("ASIA/OFF AND n2h>=3",
         lambda r: r.get("session") in {"ASIA","OFF"} and (r.get("n2h") or 0)>=3),
        ("Fri only",
         lambda r: r.get("dow") == 4),
        ("!Sat AND !Tue",
         lambda r: r.get("dow") not in {1,5}),
        ("!Sat AND !Tue AND !US",
         lambda r: r.get("dow") not in {1,5} and r.get("session") != "US"),
        ("sync_k < 200K",
         lambda r: (r.get("sync_k") or 0) < 200_000),
        ("sync_k<200K AND !US AND n2h>=3",
         lambda r: (r.get("sync_k") or 0)<200_000 and r.get("session")!="US" and (r.get("n2h") or 0)>=3),
    ]
    for label, fn in combos:
        sub = [r["_net"] for r in te if fn(r)]
        row(label, stat(sub))

    hdr("B2 · US session LONG breakdown — noisy vs time exit")
    us_te = [r for r in longs if r.get("session")=="US" and r.get("close_reason")=="TIME_EXIT"]
    us_ne = [r for r in longs if r.get("session")=="US" and r.get("close_reason")=="NOISY_EARLY_EXIT"]
    row("US TIME_EXIT", stat([r["_net"] for r in us_te]))
    row("US NOISY_EARLY_EXIT", stat([r["_net"] for r in us_ne]))
    print()
    # US TIME_EXIT by n2h
    for thr in [0,2,3,4]:
        sub = [r["_net"] for r in us_te if (r.get("n2h") or 0) >= thr]
        row(f"  US TIME_EXIT n2h>={thr}", stat(sub))

# ── SECTION C: SHORT deep dive ────────────────────────────────────────────────

def section_c(shorts: list[dict]) -> None:
    hdr("C1 · SHORT by btc_max magnitude (BTC SELL that confirmed entry)")
    buckets = [(0,500,"<500K"),(500,1000,"500K-1M"),(1000,2000,"1M-2M"),
               (2000,5000,"2M-5M"),(5000,9999999,">=5M")]
    for lo,hi,label in buckets:
        sub = [r["_net"] for r in shorts
               if lo*1000 <= (r.get("btc_max") or 0) < hi*1000]
        row(f"btc_max {label}", stat(sub))

    hdr("C2 · SHORT by n2h at cascade (prior ETH SELLs in 2h window)")
    for thr in [0,1,2,3,4]:
        sub = [r["_net"] for r in shorts if (r.get("n2h") or 0) >= thr]
        row(f"n2h >= {thr}", stat(sub))

    hdr("C3 · SHORT by sync_k")
    buckets_s = [(0,200,"sync < 200K"),(200,500,"200K-500K"),
                 (500,1000,"500K-1M"),(1000,9999,">=1M")]
    for lo,hi,label in buckets_s:
        sub = [r["_net"] for r in shorts
               if lo*1000 <= (r.get("sync_k") or 0) < hi*1000]
        row(label, stat(sub))

    hdr("C4 · SHORT entry delay (anchor_ts_ms -> entry_ts_ms)")
    delays: list[tuple[float, float]] = []
    for r in shorts:
        ts = r.get("anchor_ts_ms"); et = r.get("entry_ts_ms")
        if ts and et:
            delay_min = (int(et) - int(ts)) / 60_000
            delays.append((delay_min, r["_net"]))
    if delays:
        for lo_m, hi_m, label in [(0,5,"delay <5 min"),(5,10,"5-10 min"),
                                   (10,20,"10-20 min"),(20,30,"20-30 min")]:
            sub = [net for d,net in delays if lo_m <= d < hi_m]
            row(label, stat(sub))
        avg_delay = sum(d for d,_ in delays)/len(delays)
        print(f"\n  Mean entry delay: {avg_delay:.1f} min  (N={len(delays)})")

    hdr("C5 · SHORT score>=4 by session")
    score4 = [r for r in shorts if (r.get("score") or 0) >= 4]
    for sess in ["US","ASIA","EUROPE","OFF"]:
        sub = [r["_net"] for r in score4 if r.get("session")==sess]
        row(f"score>=4 + {sess}", stat(sub))
    row("score>=4 total", stat([r["_net"] for r in score4]))
    print()
    row("score>=4 + !EUROPE", stat([r["_net"] for r in score4 if r.get("session")!="EUROPE"]))

    hdr("C6 · SHORT combined: score>=4 AND various gates")
    combos = [
        ("score>=4 (baseline)", lambda r: (r.get("score") or 0)>=4),
        ("score>=4 AND !EUROPE", lambda r: (r.get("score") or 0)>=4 and r.get("session")!="EUROPE"),
        ("score>=4 AND n2h>=3", lambda r: (r.get("score") or 0)>=4 and (r.get("n2h") or 0)>=3),
        ("score>=4 AND btc_max>=2M", lambda r: (r.get("score") or 0)>=4 and (r.get("btc_max") or 0)>=2_000_000),
        ("score>=3 AND n2h>=3", lambda r: (r.get("score") or 0)>=3 and (r.get("n2h") or 0)>=3),
        ("score>=3 AND btc_max>=2M", lambda r: (r.get("score") or 0)>=3 and (r.get("btc_max") or 0)>=2_000_000),
    ]
    for label, fn in combos:
        sub = [r["_net"] for r in shorts if fn(r)]
        row(label, stat(sub))

# ── SECTION D: April regime detection (DB) ────────────────────────────────────

def section_d(longs: list[dict], shorts: list[dict]) -> None:
    hdr("D1 · [DB] April vs non-April: cascade-time features")
    all_events = longs + shorts
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=5) as conn:
            april, non_april = [], []
            btc7d_april, btc7d_other = [], []
            btc4h_april, btc4h_other = [], []
            for r in all_events:
                ts = int(r.get("anchor_ts_ms") or 0)
                b4h  = btc_ret_at(conn, ts, 4*3600_000)
                b7d  = btc_ret_at(conn, ts, 7*24*3600_000)
                r["_btc4h"] = b4h
                r["_btc7d"] = b7d
                is_april = r["_month"] == "2026-04"
                net = r["_net"]
                if is_april:
                    april.append(r)
                    if b4h is not None: btc4h_april.append(b4h)
                    if b7d is not None: btc7d_april.append(b7d)
                else:
                    non_april.append(r)
                    if b4h is not None: btc4h_other.append(b4h)
                    if b7d is not None: btc7d_other.append(b7d)

            def _avg(lst): return round(sum(lst)/len(lst),1) if lst else None
            def _pct_neg(lst): return round(sum(1 for x in lst if x<0)/len(lst)*100,0) if lst else None

            print(f"  April events:     N={len(april)}")
            print(f"    btc4h avg:      {_avg(btc4h_april)} bps  "
                  f"(pct<0: {_pct_neg(btc4h_april)}%)")
            print(f"    btc7d avg:      {_avg(btc7d_april)} bps  "
                  f"(pct<0: {_pct_neg(btc7d_april)}%)")
            print(f"  Non-April events: N={len(non_april)}")
            print(f"    btc4h avg:      {_avg(btc4h_other)} bps  "
                  f"(pct<0: {_pct_neg(btc4h_other)}%)")
            print(f"    btc7d avg:      {_avg(btc7d_other)} bps  "
                  f"(pct<0: {_pct_neg(btc7d_other)}%)")

            hdr("D2 · [DB] BTC 7-day trend gate as regime filter")
            # If btc7d is strongly negative (bear trend), does it predict bad outcomes?
            for thr in [-500, -300, -100, 0, 100]:
                longs_below = [r["_net"] for r in longs
                               if r.get("_btc7d") is not None and r["_btc7d"] <= thr]
                longs_above = [r["_net"] for r in longs
                               if r.get("_btc7d") is not None and r["_btc7d"] > thr]
                note = f"(below={len(longs_below)}, above={len(longs_above)})"
                row(f"LONG btc7d <= {thr} bps", stat(longs_below), note)
                row(f"LONG btc7d >  {thr} bps", stat(longs_above))
                print()

            hdr("D3 · [DB] BTC 7d trend gate for TIME_EXIT LONGs only")
            te = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
            for thr in [-300, -100, 0, 100]:
                sub = [r["_net"] for r in te
                       if r.get("_btc7d") is not None and r["_btc7d"] > thr]
                row(f"TIME_EXIT + btc7d > {thr} bps", stat(sub))

    except Exception as exc:
        print(f"  [DB ERROR] {exc}")

# ── SECTION E: Score component deconstruction ─────────────────────────────────

def section_e(longs: list[dict]) -> None:
    te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]

    hdr("E1 · Score component analysis (TIME_EXIT only)")
    # Components knowable from backfill fields:
    # n2h>=3 → score +1
    # sync_k>=200K → score +1
    # Session US (13-21h) → score +1
    # btc4h<0 → score +1 (not in ledger, skip)
    # vdepth>=30 → not in ledger, skip
    # silence → score +1 (all TIME_EXIT got silence or ran to 4h)

    components = [
        ("n2h>=3 component ON",  lambda r: (r.get("n2h") or 0) >= 3),
        ("n2h>=3 component OFF", lambda r: (r.get("n2h") or 0) < 3),
        ("sync>=200K comp ON",   lambda r: (r.get("sync_k") or 0) >= 200_000),
        ("sync>=200K comp OFF",  lambda r: (r.get("sync_k") or 0) < 200_000),
        ("US session comp ON",   lambda r: r.get("session") == "US"),
        ("US session comp OFF",  lambda r: r.get("session") != "US"),
        ("long_score==3 (min)",  lambda r: (r.get("long_score") or 0) == 3),
        ("long_score==4",        lambda r: (r.get("long_score") or 0) == 4),
        ("long_score==5 (max)",  lambda r: (r.get("long_score") or 0) == 5),
    ]
    for label, fn in components:
        sub = [r["_net"] for r in te if fn(r)]
        row(label, stat(sub))

    hdr("E2 · Sync_k direction check (TIME_EXIT): is LOW sync better?")
    # Hypothesis from batch-1: HIGH sync hurts LONG
    # Test: low sync = ETH-idiosyncratic cascade = no broad selling = bounce
    te_low  = [r["_net"] for r in te if (r.get("sync_k") or 0) < 200_000]
    te_high = [r["_net"] for r in te if (r.get("sync_k") or 0) >= 200_000]
    row("sync_k < 200K (low)", stat(te_low), "<-- ETH-idiosyncratic")
    row("sync_k >= 200K (high)", stat(te_high), "<-- broad BTC+SOL selling")

    hdr("E3 · Running notional at cascade (bigger cascade = better?)")
    buckets = [(200,300,"200-300K"),(300,500,"300-500K"),
               (500,1000,"500K-1M"),(1000,9999,">=1M")]
    for lo,hi,label in buckets:
        sub = [r["_net"] for r in te
               if lo*1000 <= (r.get("running_notional") or 0) < hi*1000]
        row(f"notional {label}", stat(sub))

# ── SECTION F: Permutation null on TIME_EXIT ─────────────────────────────────

def section_f(longs: list[dict]) -> None:
    te_nets = [r["_net"] for r in longs if r.get("close_reason") == "TIME_EXIT"]
    hdr("F1 · Permutation null check on TIME_EXIT LONGs")
    if len(te_nets) < 10:
        print("  Too few samples"); return
    rng = random.Random(RANDOM_SEED)
    observed_avg = sum(te_nets) / len(te_nets)
    observed_wr  = sum(1 for v in te_nets if v > 0) / len(te_nets)
    n_perm = 10_000
    n_beats_avg = 0; n_beats_wr = 0
    for _ in range(n_perm):
        shuffled = [v * rng.choice([-1,1]) for v in te_nets]
        if sum(shuffled)/len(shuffled) >= observed_avg: n_beats_avg += 1
        if sum(1 for v in shuffled if v>0)/len(shuffled) >= observed_wr: n_beats_wr += 1
    p_avg = n_beats_avg / n_perm
    p_wr  = n_beats_wr  / n_perm
    print(f"  TIME_EXIT LONG: N={len(te_nets)}, avg={observed_avg:+.1f} bps, WR={observed_wr:.1%}")
    print(f"  Permutation p(avg)  = {p_avg:.4f}  {'PASS <0.05' if p_avg<0.05 else 'FAIL'}")
    print(f"  Permutation p(WR)   = {p_wr:.4f}  {'PASS <0.05' if p_wr<0.05 else 'FAIL'}")

    # Also test best combined gate from B1
    hdr("F2 · Permutation null: ASIA/OFF AND n2h>=3 (TIME_EXIT)")
    filtered = [r["_net"] for r in longs
                if r.get("close_reason")=="TIME_EXIT"
                and r.get("session") in {"ASIA","OFF"}
                and (r.get("n2h") or 0) >= 3]
    if len(filtered) < 5:
        print(f"  N={len(filtered)} too small")
        return
    obs_avg = sum(filtered)/len(filtered)
    obs_wr  = sum(1 for v in filtered if v>0)/len(filtered)
    nb_avg = 0; nb_wr = 0
    for _ in range(n_perm):
        shuffled = [v * rng.choice([-1,1]) for v in filtered]
        if sum(shuffled)/len(shuffled) >= obs_avg: nb_avg += 1
        if sum(1 for v in shuffled if v>0)/len(shuffled) >= obs_wr: nb_wr += 1
    p_avg2 = nb_avg/n_perm; p_wr2 = nb_wr/n_perm
    print(f"  ASIA/OFF+n2h>=3: N={len(filtered)}, avg={obs_avg:+.1f} bps, WR={obs_wr:.1%}")
    print(f"  Permutation p(avg)  = {p_avg2:.4f}  {'PASS <0.05' if p_avg2<0.05 else 'FAIL'}")
    print(f"  Permutation p(WR)   = {p_wr2:.4f}  {'PASS <0.05' if p_wr2<0.05 else 'FAIL'}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\nLoading backfill ledger...")
    longs, shorts = load_records()
    te = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
    print(f"LONG: {len(longs)} total  ({len(te)} TIME_EXIT, "
          f"{len(longs)-len(te)} NOISY_EARLY_EXIT)")
    print(f"SHORT: {len(shorts)} total")

    section_a(longs)
    section_b(longs)
    section_c(shorts)
    section_d(longs, shorts)
    section_e(longs)
    section_f(longs)

    print(); print("=" * 64); print("  ALL ROUND-2 TESTS COMPLETE"); print("=" * 64)

if __name__ == "__main__":
    main()
