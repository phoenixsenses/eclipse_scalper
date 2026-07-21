"""S34 Round-5 Tests — Sections X through AE.

  X1  US losers autopsy: n2h / dow / vdepth / btc4h within US+sync<200K
  X2  US vs ASIA/OFF at different n2h thresholds — does n2h rescue US?
  Y1  Monthly stability in new-gate population (sync<200K TIME_EXIT)
  Y2  April isolation — sync<200K April vs non-April
  Z1  SHORT score>=4 deep dive: session / month / n2h
  Z2  SHORT frequency: score>=4 trades per month
  AA1 Cascade size gate: >=200K vs >=300K vs >=500K on TIME_EXIT
  AB1 vdepth threshold sweep in new-gate population
  AC1 btc7d multi-threshold sweep in new-gate population
  AD1 n2h>=5 deep dive: session / month / DOW
  AE1 Score revision: drop sync_k component, find optimal threshold

Usage: python tools/research_s34_round5_tests.py
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"

def stat(vals):
    if not vals: return {"n": 0, "wr": None, "avg": None, "total": None}
    wins = sum(1 for v in vals if v > 0)
    return {"n": len(vals), "wr": round(wins/len(vals), 3),
            "avg": round(sum(vals)/len(vals), 1), "total": round(sum(vals), 0)}

def pct(v): return "  -  " if v is None else f"{v*100:5.1f}%"
def fmt(v, d=1): return "   -   " if v is None else f"{v:+{7+d}.{d}f}"
def hdr(t): print(); print("="*68); print(f"  {t}"); print("="*68)
def row(label, s, note=""):
    if s["n"] == 0:
        print(f"  {label:<38s}  N=  0  -----  ------  ------  {note}"); return
    print(f"  {label:<38s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def load_records():
    longs, shorts = [], []
    if not LEDGER.exists(): return longs, shorts
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
    a = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1", (lo,)).fetchone()
    b = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0) <= 0: return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def btc4h_bps_at(conn, ts_ms):
    lo = ts_ms - 4*3600_000
    a = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1", (lo,)).fetchone()
    b = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0) <= 0: return None
    return (float(b[0]) - float(a[0])) / float(a[0]) * 10_000.0

def main():
    longs, shorts = load_records()
    print(f"Loading backfill ledger...")
    print(f"LONG: {len(longs)} ({sum(1 for r in longs if r.get('close_reason')=='TIME_EXIT')} TIME_EXIT, "
          f"{sum(1 for r in longs if r.get('close_reason')=='NOISY_EARLY_EXIT')} NOISY)")
    print(f"SHORT: {len(shorts)}")

    # Core filtered populations
    te_all = [r for r in longs if r.get("close_reason") == "TIME_EXIT"]
    te_low = [r for r in te_all if float(r.get("sync_k") or 0) < 200_000]   # new gate pop
    te_us  = [r for r in te_low if r.get("session") == "US"]
    te_non_us = [r for r in te_low if r.get("session") != "US"]
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

    # ── X1 · US losers autopsy ────────────────────────────────────────────────
    hdr("X1 · US losers autopsy (US + sync<200K + TIME_EXIT, N=21)")
    print(f"  US sync<200K TIME_EXIT:  N={len(te_us)}, WR={stat([r['_net'] for r in te_us])['wr']*100:.0f}%, avg={stat([r['_net'] for r in te_us])['avg']:+.1f} bps")

    print("\n  --- by n2h ---")
    for thr in [3, 4, 5]:
        sub = [r["_net"] for r in te_us if (r.get("n2h") or 0) >= thr]
        row(f"US n2h>={thr}", stat(sub))

    print("\n  --- by DOW ---")
    for d in range(7):
        sub = [r["_net"] for r in te_us if r.get("dow") == d]
        if sub: row(f"US {dow_names[d]}", stat(sub))

    print("\n  --- by long_score ---")
    for sc in [3, 4, 5]:
        sub = [r["_net"] for r in te_us if (r.get("long_score") or 0) == sc]
        row(f"US long_score=={sc}", stat(sub))

    print("\n  --- US winners vs losers: n2h distribution ---")
    us_wins  = [r for r in te_us if r["_net"] > 0]
    us_loses = [r for r in te_us if r["_net"] <= 0]
    if us_wins:
        n2h_w = [r.get("n2h") or 0 for r in us_wins]
        n2h_l = [r.get("n2h") or 0 for r in us_loses]
        print(f"  Winners (N={len(us_wins)})  avg n2h={sum(n2h_w)/len(n2h_w):.1f}  "
              f"n2h vals={sorted(n2h_w)}")
        print(f"  Losers  (N={len(us_loses)}) avg n2h={sum(n2h_l)/len(n2h_l) if n2h_l else 0:.1f}  "
              f"n2h vals={sorted(n2h_l)}")

    # ── X2 · Does n2h rescue US? ──────────────────────────────────────────────
    hdr("X2 · n2h threshold: US vs ASIA+OFF (in sync<200K TIME_EXIT)")
    for thr in [3, 4, 5]:
        us_sub   = [r["_net"] for r in te_us if (r.get("n2h") or 0) >= thr]
        nUS_sub  = [r["_net"] for r in te_non_us if (r.get("n2h") or 0) >= thr]
        all_sub  = [r["_net"] for r in te_low if (r.get("n2h") or 0) >= thr]
        row(f"n2h>={thr} US", stat(us_sub))
        row(f"n2h>={thr} ASIA+OFF", stat(nUS_sub))
        row(f"n2h>={thr} all", stat(all_sub))
        print()

    # ── Y1 · Monthly stability new-gate population ────────────────────────────
    hdr("Y1 · Monthly stability (TIME_EXIT + sync<200K)")
    months: dict[str, list] = {}
    for r in te_low:
        try: mk = datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
        except: mk = "?"
        months.setdefault(mk, []).append(r["_net"])
    for m in sorted(months):
        row(m, stat(months[m]))
    row("ALL", stat([r["_net"] for r in te_low]))

    # ── Y2 · April isolation ──────────────────────────────────────────────────
    hdr("Y2 · April isolation in new-gate population")
    april = [r for r in te_low if "2026-04" in datetime.fromtimestamp(
        int(r.get("anchor_ts_ms") or 0)/1000, tz=timezone.utc).strftime("%Y-%m")]
    non_april = [r for r in te_low if r not in april]
    row("April (sync<200K TIME_EXIT)", stat([r["_net"] for r in april]))
    row("Non-April", stat([r["_net"] for r in non_april]))
    if april:
        print(f"\n  April events:")
        for r in sorted(april, key=lambda x: x.get("anchor_ts_ms") or 0):
            print(f"    n2h={r.get('n2h')}  sess={r.get('session')}  sync={float(r.get('sync_k') or 0)/1000:.0f}K  net={r['_net']:+.1f} bps")

    # ── Z1 · SHORT score>=4 deep dive ─────────────────────────────────────────
    hdr("Z1 · SHORT score>=4 breakdown (session / month / n2h)")
    s4 = [r for r in shorts if (r.get("score") or 0) >= 4]
    print(f"  score>=4 total: N={len(s4)}")

    print("\n  --- by session ---")
    for sess in ["US","ASIA","EUROPE","OFF"]:
        sub = [r["_net"] for r in s4 if r.get("session") == sess]
        if sub: row(f"  {sess}", stat(sub))

    print("\n  --- by month ---")
    s4_months: dict[str,list] = {}
    for r in s4:
        try: mk = datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
        except: mk = "?"
        s4_months.setdefault(mk, []).append(r["_net"])
    for m in sorted(s4_months):
        row(f"  {m}", stat(s4_months[m]))

    print("\n  --- by n2h ---")
    for thr in [2, 3, 4]:
        sub = [r["_net"] for r in s4 if (r.get("n2h") or 0) >= thr]
        row(f"  n2h>={thr}", stat(sub))

    print("\n  --- score>=4 vs score>=5 ---")
    row("score>=4", stat([r["_net"] for r in s4]))
    row("score>=5", stat([r["_net"] for r in shorts if (r.get("score") or 0) >= 5]))
    row("score==4 exactly", stat([r["_net"] for r in s4 if (r.get("score") or 0) == 4]))

    # ── Z2 · SHORT frequency ──────────────────────────────────────────────────
    hdr("Z2 · SHORT trade frequency per month (score>=4)")
    s4_freq: dict[str,int] = {}
    all_s_freq: dict[str,int] = {}
    for r in shorts:
        try: mk = datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
        except: mk = "?"
        all_s_freq[mk] = all_s_freq.get(mk, 0) + 1
        if (r.get("score") or 0) >= 4:
            s4_freq[mk] = s4_freq.get(mk, 0) + 1
    print(f"  {'Month':<10} {'All SHORTs':>12} {'score>=4':>10}")
    for m in sorted(all_s_freq):
        print(f"  {m:<10} {all_s_freq.get(m,0):>12} {s4_freq.get(m,0):>10}")
    if s4_freq:
        print(f"\n  Avg SHORTs/month (score>=4): {sum(s4_freq.values())/len(s4_freq):.1f}")

    # ── AA1 · Cascade size gate ───────────────────────────────────────────────
    hdr("AA1 · Cascade notional threshold sweep (TIME_EXIT, from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
            # For each TIME_EXIT LONG, check the actual cascade notional at anchor_ts_ms
            # We look for the ETH SELL liq that triggered (at or just before anchor)
            te_notionals = []
            for r in te_all:
                ts = int(r.get("anchor_ts_ms") or 0)
                # Largest ETH SELL liq in [ts-10s, ts+10s]
                result = conn.execute(
                    "SELECT MAX(notional) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
                    "AND ts_ms>=? AND ts_ms<=?",
                    (ts-10_000, ts+10_000)
                ).fetchone()
                notional = float(result[0] or 0) if result and result[0] else None
                if notional:
                    r["_cascade_notional"] = notional
                    te_notionals.append(r)

            print(f"  Matched cascade notional for {len(te_notionals)}/{len(te_all)} TIME_EXIT events")
            for thresh_k in [200, 300, 500, 750, 1000]:
                sub = [r["_net"] for r in te_notionals if r.get("_cascade_notional", 0) >= thresh_k*1000]
                row(f"notional >= {thresh_k}K", stat(sub))
            # Distribution
            print("\n  Notional distribution of TIME_EXIT events:")
            bands = [(200,300),(300,500),(500,750),(750,1000),(1000,2000),(2000,1e9)]
            for lo, hi in bands:
                sub = [r["_net"] for r in te_notionals
                       if lo*1000 <= r.get("_cascade_notional",0) < hi*1000]
                lbl = f"{lo}K-{int(hi/1000) if hi < 1e9 else '∞'}M"
                row(lbl, stat(sub))
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── AB1 · vdepth threshold sweep ─────────────────────────────────────────
    hdr("AB1 · vdepth threshold sweep (TIME_EXIT + sync<200K)")
    # vdepth_bps field (bid depth ≥ vdepth bps means deep book)
    # Actually check if vdepth is stored — it may be as part of score components
    # Score component: vdepth>=30. Let's look for vdepth_bps in the records
    has_vdepth = [r for r in te_low if r.get("vdepth_bps") is not None]
    print(f"  Records with vdepth_bps: {len(has_vdepth)}/{len(te_low)}")
    if has_vdepth:
        for thr in [20, 30, 40, 50, 60]:
            sub = [r["_net"] for r in has_vdepth if float(r.get("vdepth_bps") or 0) >= thr]
            row(f"vdepth >= {thr}", stat(sub))
    else:
        # Fall back to score component: vdepth>=30 is already baked into base_score
        # Check base_score breakdown as proxy
        print("  vdepth_bps not in ledger — using long_score as proxy")
        for sc in [3, 4, 5]:
            sub = [r["_net"] for r in te_low if (r.get("long_score") or 0) == sc]
            row(f"long_score=={sc}", stat(sub))

    # ── AC1 · btc7d multi-threshold sweep in new-gate population ─────────────
    hdr("AC1 · btc7d threshold sweep (TIME_EXIT + sync<200K, from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=10) as conn:
            te_low_b7 = []
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                b7 = btc7d_bps_at(conn, ts)
                if b7 is not None:
                    r["_btc7d"] = b7
                    te_low_b7.append(r)
            print(f"  btc7d loaded for {len(te_low_b7)}/{len(te_low)} new-gate events")
            for thresh in [-500, -300, -200, -100, -50, 0, 100, 300]:
                sub = [r["_net"] for r in te_low_b7 if r["_btc7d"] < thresh]
                fail = [r["_net"] for r in te_low_b7 if r["_btc7d"] >= thresh]
                row(f"btc7d < {thresh:+4d}", stat(sub))
            print()
            # Best split
            print("  Optimal split search:")
            best_avg, best_thresh = 0, 0
            for t in range(-600, 400, 50):
                sub = [r["_net"] for r in te_low_b7 if r["_btc7d"] < t]
                if len(sub) >= 8:
                    s = stat(sub)
                    if s["avg"] and s["avg"] > best_avg:
                        best_avg = s["avg"]
                        best_thresh = t
            print(f"  Best threshold: btc7d < {best_thresh:+d} bps  "
                  f"=> {stat([r['_net'] for r in te_low_b7 if r['_btc7d'] < best_thresh])}")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── AD1 · n2h>=5 deep dive ────────────────────────────────────────────────
    hdr("AD1 · n2h>=5 deep dive (TIME_EXIT + sync<200K)")
    n2h5 = [r for r in te_low if (r.get("n2h") or 0) >= 5]
    print(f"  n2h>=5 events: {len(n2h5)}")
    row("n2h>=5 all", stat([r["_net"] for r in n2h5]))

    print("\n  --- by session ---")
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in n2h5 if r.get("session") == sess]
        if sub: row(f"  n2h>=5 {sess}", stat(sub))

    print("\n  --- by month ---")
    n5_months: dict[str,list] = {}
    for r in n2h5:
        try: mk = datetime.fromtimestamp(int(r["anchor_ts_ms"])/1000, tz=timezone.utc).strftime("%Y-%m")
        except: mk="?"
        n5_months.setdefault(mk,[]).append(r["_net"])
    for m in sorted(n5_months):
        row(f"  {m}", stat(n5_months[m]))

    print("\n  --- by DOW ---")
    for d in range(7):
        sub = [r["_net"] for r in n2h5 if r.get("dow") == d]
        if sub: row(f"  {dow_names[d]}", stat(sub))

    print("\n  --- individual events ---")
    for r in sorted(n2h5, key=lambda x: x.get("anchor_ts_ms") or 0):
        ts = int(r.get("anchor_ts_ms") or 0)
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  {dt}  n2h={r.get('n2h')}  sess={r.get('session')}  sync={float(r.get('sync_k') or 0)/1000:.0f}K  net={r['_net']:+.1f} bps")

    # ── AE1 · Score revision: drop sync_k component ───────────────────────────
    hdr("AE1 · Score revision: remove sync_k from score (TIME_EXIT all)")
    # Current score = n2h>=3 + btc4h<0 + vdepth>=30 + sess==US + sync_k>=200K  (max 5)
    # long_score = base_score + 1 (silence)
    # Revised: score_no_sync = n2h>=3 + btc4h<0 + vdepth>=30 + sess==US  (max 4)
    # Can reconstruct from long_score: score_no_sync = long_score - silence(1) - sync_component
    # sync_component = 1 if sync_k>=200K else 0
    # So revised_long_score = long_score - int(sync_k>=200K)

    print("  Reconstructing revised score (drop sync_k component)...")
    for r in te_all:
        sk = float(r.get("sync_k") or 0)
        ls = int(r.get("long_score") or 0)
        r["_rev_ls"] = ls - int(sk >= 200_000)  # drop sync component

    print("\n  Current long_score gate vs revised (no-sync) gate — TIME_EXIT all:")
    print(f"  {'Gate':<38} {'N':>4}  {'WR':>7}  {'avg':>10}  {'tot':>10}")
    for thr in [3, 4]:
        cur = [r["_net"] for r in te_all if (r.get("long_score") or 0) >= thr]
        rev = [r["_net"] for r in te_all if r["_rev_ls"] >= thr]
        row(f"current long_score>={thr}", stat(cur))
        row(f"revised no-sync>={thr}", stat(rev))
        print()

    print("  Revised score distribution (no-sync):")
    for sc in range(1, 6):
        cur_n = sum(1 for r in te_all if (r.get("long_score") or 0) == sc)
        rev_n = sum(1 for r in te_all if r["_rev_ls"] == sc)
        sub   = [r["_net"] for r in te_all if r["_rev_ls"] == sc]
        row(f"rev_ls=={sc} (orig had {cur_n})", stat(sub))

    print("\n  Best revised gate per sync zone:")
    # Low sync (new gate already): does revised score change threshold?
    for thr in [2, 3]:
        rev_low = [r["_net"] for r in te_all if r["_rev_ls"] >= thr and float(r.get("sync_k") or 0) < 200_000]
        rev_hi  = [r["_net"] for r in te_all if r["_rev_ls"] >= thr and float(r.get("sync_k") or 0) >= 200_000]
        row(f"rev>={thr} + sync<200K", stat(rev_low))
        row(f"rev>={thr} + sync>=200K", stat(rev_hi))
        print()

    print()
    print("="*68)
    print("  ALL ROUND-5 TESTS COMPLETE")
    print("="*68)

if __name__ == "__main__":
    main()
