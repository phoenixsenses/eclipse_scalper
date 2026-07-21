"""S34 Round-4 Tests — Sections O through W.

  O  Mon/Wed re-eval with new gates (TIME_EXIT + sync<200K)
  P  Session breakdown in new-gate population
  Q  btc7d<0 marginal benefit after sync<200K
  R  n2h>=4 + sync<200K combined gate
  S  Trade frequency projection with new gates
  T  SIL_LO=3min: monthly frequency + expected contribution
  U  score==5 burial: how many blocked by sync<200K?
  V  btc4h<0 in new-gate population
  W  Triple best: sync<200K + n2h>=4 + btc7d<0

Usage: python tools/research_s34_round4_tests.py
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"

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

def hdr(t):
    print(); print("="*66); print(f"  {t}"); print("="*66)

def row(label, s, note=""):
    if s["n"] == 0:
        print(f"  {label:<36s}  N=  0  -----  ------  ------  {note}"); return
    print(f"  {label:<36s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def load_records():
    longs, shorts = [], []
    if not LEDGER.exists():
        return longs, shorts
    with LEDGER.open(encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            if r.get("event") != "CLOSE": continue
            net = r.get("net_bps")
            if net is None: continue
            r["_net"] = float(net)
            if r.get("direction") == "LONG": longs.append(r)
            elif r.get("direction") == "SHORT": shorts.append(r)
    return longs, shorts

def btc7d_bps_at(conn, ts_ms):
    lo = ts_ms - 7*24*3600_000
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1", (lo,)
    ).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)
    ).fetchone()
    if not a or not b or float(a[0] or 0) <= 0: return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def btc4h_bps_at(conn, ts_ms):
    lo = ts_ms - 4*3600_000
    a = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1", (lo,)
    ).fetchone()
    b = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)
    ).fetchone()
    if not a or not b or float(a[0] or 0) <= 0: return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def mark_at(conn, ts_ms, offset_ms):
    t = ts_ms + offset_ms
    r = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (t,)
    ).fetchone()
    return float(r[0]) if r else None

def main():
    longs, shorts = load_records()
    print(f"Loading backfill ledger...")
    print(f"LONG: {len(longs)} ({sum(1 for r in longs if r.get('close_reason')=='TIME_EXIT')} TIME_EXIT, "
          f"{sum(1 for r in longs if r.get('close_reason')=='NOISY_EARLY_EXIT')} NOISY)")
    print(f"SHORT: {len(shorts)}")

    # New-gate population
    te_low = [r for r in longs
              if r.get("close_reason") == "TIME_EXIT"
              and float(r.get("sync_k") or 0) < 200_000]
    te_vals = [r["_net"] for r in te_low]

    # ── O · Mon/Wed re-eval with new gates ────────────────────────────────────
    hdr("O1 · Mon/Wed re-eval (TIME_EXIT + sync<200K population)")
    dow_names = {0:"Mon(0)", 1:"Tue(1)", 2:"Wed(2)", 3:"Thu(3)", 4:"Fri(4)", 5:"Sat(5)", 6:"Sun(6)"}
    for d in range(7):
        sub = [r["_net"] for r in te_low if r.get("dow") == d]
        note = " <- EXCLUDED (current)" if d in {0,2} else ""
        row(dow_names[d], stat(sub), note)
    print()
    excluded = [r["_net"] for r in te_low if r.get("dow") in {0,2}]
    allowed  = [r["_net"] for r in te_low if r.get("dow") not in {0,2}]
    row("Mon+Wed excluded (current rule)", stat(excluded))
    row("Other days (current pass)", stat(allowed))

    hdr("O2 · Mon/Wed: all TIME_EXIT (any sync) for context")
    all_te = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]
    for d in range(7):
        sub = [r["_net"] for r in all_te if r.get("dow") == d]
        note = " <- EXCLUDED" if d in {0,2} else ""
        row(dow_names[d], stat(sub), note)

    # ── P · Session breakdown in new-gate population ──────────────────────────
    hdr("P1 · Session breakdown (TIME_EXIT + sync<200K)")
    for sess in ["US","ASIA","EUROPE","OFF"]:
        sub = [r["_net"] for r in te_low if r.get("session") == sess]
        row(sess, stat(sub))
    non_us = [r["_net"] for r in te_low if r.get("session") != "US"]
    us_only = [r["_net"] for r in te_low if r.get("session") == "US"]
    print()
    row("!US total", stat(non_us))
    row("US total", stat(us_only))

    hdr("P2 · Would adding !US filter help? (new-gate population)")
    # Current: all sessions pass (except EUROPE already blocked)
    # Proposal: also block US
    row("sync<200K all sessions (current)", stat(te_vals))
    row("sync<200K + !US (proposed)", stat(non_us))
    us_blocked_impact = (sum(non_us) - sum(te_vals)) if te_vals else 0
    print(f"\n  Blocking US would remove {len(us_only)} trades, change total from "
          f"{sum(te_vals):+.0f} to {sum(non_us):+.0f} bps")

    # ── Q · btc7d<0 marginal benefit after sync<200K ─────────────────────────
    hdr("Q1 · btc7d<0 on new-gate population (loading from DB...)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
            btc7d_pass, btc7d_fail, missing = [], [], 0
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                b7 = btc7d_bps_at(conn, ts)
                if b7 is None:
                    missing += 1; continue
                r["_btc7d"] = b7
                if b7 < 0: btc7d_pass.append(r["_net"])
                else: btc7d_fail.append(r["_net"])
            row("sync<200K baseline (new gate)", stat(te_vals), f"(total={len(te_vals)})")
            row("+ btc7d<0 (add regime gate)", stat(btc7d_pass), f"(missing={missing})")
            row("btc7d>=0 (excluded by regime)", stat(btc7d_fail))
            print(f"\n  Adding btc7d<0 removes {len(btc7d_fail)} events, "
                  f"delta avg = {stat(btc7d_pass)['avg'] or 0:.1f} vs {stat(te_vals)['avg'] or 0:.1f} bps")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── R · n2h>=4 + sync<200K ────────────────────────────────────────────────
    hdr("R1 · n2h gate within new-gate population (TIME_EXIT + sync<200K)")
    for thr in [2, 3, 4, 5]:
        sub = [r["_net"] for r in te_low if (r.get("n2h") or 0) >= thr]
        row(f"n2h >= {thr}", stat(sub))
    print()
    # Session breakdown for n2h>=4 within new gates
    hdr("R2 · n2h>=4 + sync<200K by session")
    n2h4_low = [r for r in te_low if (r.get("n2h") or 0) >= 4]
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in n2h4_low if r.get("session") == sess]
        row(sess, stat(sub))
    row("all sessions", stat([r["_net"] for r in n2h4_low]))

    # ── S · Trade frequency projection ───────────────────────────────────────
    hdr("S1 · Trade frequency: new gates vs old (by month)")
    # Count TIME_EXIT events per month at different gate levels
    months_all: dict[str,int] = {}
    months_te: dict[str,int] = {}
    months_te_low: dict[str,int] = {}
    for r in longs:
        ts = int(r.get("anchor_ts_ms") or 0)
        try: mk = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m")
        except: mk = "?"
        months_all[mk] = months_all.get(mk, 0) + 1
        if r.get("close_reason") == "TIME_EXIT":
            months_te[mk] = months_te.get(mk, 0) + 1
            if float(r.get("sync_k") or 0) < 200_000:
                months_te_low[mk] = months_te_low.get(mk, 0) + 1
    print(f"  {'Month':<10} {'All LONGs':>10} {'TIME_EXIT':>10} {'TE+sync<200K':>13}")
    for m in sorted(months_all):
        print(f"  {m:<10} {months_all.get(m,0):>10} {months_te.get(m,0):>10} {months_te_low.get(m,0):>13}")
    all_months = sorted(months_te_low)
    if all_months:
        total_months = len(all_months)
        total_trades = sum(months_te_low.values())
        print(f"\n  Avg trades/month (new gates): {total_trades/total_months:.1f}")
        print(f"  Avg trades/month (TIME_EXIT all): {sum(months_te.values())/len(months_te):.1f}")
        print(f"  Avg trades/month (all LONGs): {sum(months_all.values())/len(months_all):.1f}")

    # ── T · SIL_LO=3min frequency ────────────────────────────────────────────
    hdr("T1 · SIL_LO=3min rescue: monthly frequency + expected gain")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
            noisy = [r for r in longs if r.get("close_reason") == "NOISY_EARLY_EXIT"]
            rescued_by_month: dict[str, list[float]] = {}
            for r in noisy:
                ts = int(r.get("anchor_ts_ms") or 0)
                early = conn.execute(
                    "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
                    "AND ts_ms>=? AND ts_ms<? AND notional>=50000 ORDER BY ts_ms ASC LIMIT 1",
                    (ts+60_000, ts+180_000)
                ).fetchone()
                if early is None: continue
                late = conn.execute(
                    "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
                    "AND ts_ms>=? AND ts_ms<? AND notional>=50000 ORDER BY ts_ms ASC LIMIT 1",
                    (ts+180_000, ts+30*60_000)
                ).fetchone()
                if late is not None: continue  # still noisy — wouldn't be rescued
                try: mk = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m")
                except: mk = "?"
                rescued_by_month.setdefault(mk, []).append(r["_net"])
            print(f"  Rescued events by month (SIL_LO=3min):")
            for m in sorted(rescued_by_month):
                vals = rescued_by_month[m]
                print(f"  {m}: {len(vals)} events rescued  (actual early exit avg={sum(vals)/len(vals):+.1f} bps)")
            total_r = sum(len(v) for v in rescued_by_month.values())
            n_months = len(rescued_by_month)
            print(f"\n  Total rescued: {total_r} over {n_months} months = {total_r/max(n_months,1):.1f}/month avg")
            print(f"  Each rescued event gains ~+96.3 bps (M1 finding)")
            print(f"  Expected monthly gain from SIL_LO change: ~{total_r/max(n_months,1)*96.3:.0f} bps/month")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── U · score==5 burial ───────────────────────────────────────────────────
    hdr("U1 · score==5 events: how many now blocked by sync<200K gate?")
    all_te_longs = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]
    score5 = [r for r in all_te_longs if (r.get("long_score") or 0) >= 5]
    score5_blocked = [r for r in score5 if float(r.get("sync_k") or 0) >= 200_000]
    score5_pass    = [r for r in score5 if float(r.get("sync_k") or 0) < 200_000]
    print(f"  score==5 events in TIME_EXIT: {len(score5)}")
    print(f"  sync>=200K (blocked by new gate): {len(score5_blocked)} ({100*len(score5_blocked)/max(len(score5),1):.0f}%)")
    print(f"  sync<200K (pass new gate): {len(score5_pass)}")
    row("score5 blocked (sync>=200K)", stat([r["_net"] for r in score5_blocked]))
    row("score5 pass (sync<200K)", stat([r["_net"] for r in score5_pass]))
    row("score5 all TIME_EXIT", stat([r["_net"] for r in score5]))
    # Show the sync distribution of score5 events
    print("\n  sync_k distribution for score==5 TIME_EXIT events:")
    for r in sorted(score5, key=lambda x: float(x.get("sync_k") or 0)):
        sk = float(r.get("sync_k") or 0)
        sess = r.get("session","?")
        net = r["_net"]
        flag = "BLOCKED" if sk >= 200_000 else "pass"
        print(f"    sync={sk/1000:.0f}K  sess={sess}  net={net:+.1f} bps  [{flag}]")

    # ── V · btc4h<0 in new-gate population ───────────────────────────────────
    hdr("V1 · btc4h<0 gate in new-gate population (TIME_EXIT + sync<200K)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
            b4h_pass, b4h_fail, missing = [], [], 0
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                b4h = btc4h_bps_at(conn, ts)
                if b4h is None: missing += 1; continue
                if b4h < 0: b4h_pass.append(r["_net"])
                else: b4h_fail.append(r["_net"])
            row("sync<200K baseline", stat(te_vals))
            row("+ btc4h<0", stat(b4h_pass), f"(missing={missing})")
            row("btc4h>=0 (excluded)", stat(b4h_fail))
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── W · Triple best combo ─────────────────────────────────────────────────
    hdr("W1 · Triple combo: sync<200K + n2h>=4 + btc7d<0")
    # Need btc7d loaded — reuse from Q section
    te_with_b7 = [r for r in te_low if r.get("_btc7d") is not None]
    if len(te_with_b7) < 5:
        print("  btc7d not available — skipping (run Q section first)")
    else:
        combos = [
            ("sync<200K (baseline)", lambda r: True),
            ("+ n2h>=4",             lambda r: (r.get("n2h") or 0) >= 4),
            ("+ btc7d<0",            lambda r: r.get("_btc7d", 0) < 0),
            ("+ n2h>=4 + btc7d<0",  lambda r: (r.get("n2h") or 0) >= 4 and r.get("_btc7d", 0) < 0),
        ]
        for label, fn in combos:
            sub = [r["_net"] for r in te_with_b7 if fn(r)]
            row(label, stat(sub))

    hdr("W2 · Triple combo by session")
    triple = [r for r in te_with_b7 if (r.get("n2h") or 0) >= 4 and r.get("_btc7d", 0) < 0] if te_with_b7 else []
    if triple:
        for sess in ["US","ASIA","OFF"]:
            sub = [r["_net"] for r in triple if r.get("session") == sess]
            row(sess, stat(sub))
        row("all sessions", stat([r["_net"] for r in triple]))
        print(f"\n  N={len(triple)} events pass triple gate")
        # Monthly
        m_triple: dict[str,list] = {}
        for r in triple:
            try: mk = datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
            except: mk="?"
            m_triple.setdefault(mk,[]).append(r["_net"])
        print("  By month:")
        for m in sorted(m_triple):
            v = m_triple[m]
            print(f"    {m}: N={len(v)}  avg={sum(v)/len(v):+.1f} bps")
    else:
        print("  [not enough data]")

    hdr("W3 · What's the best gating combo? Summary table")
    gates = [
        ("baseline TIME_EXIT",          [r["_net"] for r in all_te_longs]),
        ("+ sync<200K",                 te_vals),
        ("+ sync<200K + n2h>=4",        [r["_net"] for r in te_low if (r.get("n2h") or 0) >= 4]),
        ("+ sync<200K + btc7d<0",       [r["_net"] for r in te_with_b7 if r.get("_btc7d",0) < 0] if te_with_b7 else []),
        ("+ sync<200K + n2h>=4 + b7d<0",[r["_net"] for r in te_with_b7 if (r.get("n2h") or 0) >= 4 and r.get("_btc7d",0) < 0] if te_with_b7 else []),
        ("+ sync<200K + !US",           [r["_net"] for r in te_low if r.get("session") != "US"]),
        ("+ sync<200K + n2h>=4 + !US",  [r["_net"] for r in te_low if (r.get("n2h") or 0) >= 4 and r.get("session") != "US"]),
    ]
    for lbl, vals in gates:
        row(lbl, stat(vals))

    print()
    print("="*66)
    print("  ALL ROUND-4 TESTS COMPLETE")
    print("="*66)

if __name__ == "__main__":
    main()
