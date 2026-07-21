"""S34 Round-6 Tests — Sections BA through BI.

  BA  April exclusion: full impact + forward frequency
  BB  Cascade notional upper limit <500K: full impact
  BC  US Tuesday exclusion: worth it?
  BD  long_score==3 paradox in sync<200K population
  BE  SHORT BTC confirm size: larger = better?
  BF  SHORT btc7d: regime filter for SHORTs
  BG  US hour-of-day analysis (which hour band best)
  BH  Combined gate projection: April excl + notional<500K + sync<200K
  BI  n2h>=5 vs n2h>=3 frequency/quality trade-off

Usage: python tools/research_s34_round6_tests.py
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
LEDGER  = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DB_PATH = ROOT / "data" / "microstructure.db"

def stat(vals):
    if not vals: return {"n":0,"wr":None,"avg":None,"total":None}
    wins = sum(1 for v in vals if v > 0)
    return {"n":len(vals),"wr":round(wins/len(vals),3),
            "avg":round(sum(vals)/len(vals),1),"total":round(sum(vals),0)}

def pct(v):  return "  -  " if v is None else f"{v*100:5.1f}%"
def fmt(v,d=1): return "   -   " if v is None else f"{v:+{7+d}.{d}f}"
def hdr(t): print(); print("="*68); print(f"  {t}"); print("="*68)
def row(label, s, note=""):
    if s["n"]==0:
        print(f"  {label:<38s}  N=  0  -----  ------  ------  {note}"); return
    print(f"  {label:<38s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def month_key(ts_ms):
    try: return datetime.fromtimestamp(int(ts_ms)/1000,tz=timezone.utc).strftime("%Y-%m")
    except: return "?"

def hour_utc(ts_ms):
    try: return datetime.fromtimestamp(int(ts_ms)/1000,tz=timezone.utc).hour
    except: return -1

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
            r["_month"] = month_key(r.get("anchor_ts_ms",0))
            r["_hour"]  = hour_utc(r.get("anchor_ts_ms",0))
            if r.get("direction") == "LONG": longs.append(r)
            elif r.get("direction") == "SHORT": shorts.append(r)
    return longs, shorts

def btc7d_bps_at(conn, ts_ms):
    lo = ts_ms - 7*24*3600_000
    a = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",(lo,)).fetchone()
    b = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0)<=0: return None
    return (float(b[0])-float(a[0]))/float(a[0])*10_000.0

def cascade_notional_at(conn, ts_ms):
    r = conn.execute(
        "SELECT MAX(notional) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<=?", (ts_ms-10_000, ts_ms+10_000)
    ).fetchone()
    return float(r[0]) if r and r[0] else None

def btc_confirm_notional(conn, ts_ms, lo_ms=60_000, hi_ms=30*60_000):
    r = conn.execute(
        "SELECT MAX(notional) FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<=?", (ts_ms+lo_ms, ts_ms+hi_ms)
    ).fetchone()
    return float(r[0]) if r and r[0] else None

def main():
    longs, shorts = load_records()
    print(f"Loading backfill ledger...")
    print(f"LONG: {len(longs)} ({sum(1 for r in longs if r.get('close_reason')=='TIME_EXIT')} TIME_EXIT, "
          f"{sum(1 for r in longs if r.get('close_reason')=='NOISY_EARLY_EXIT')} NOISY)")
    print(f"SHORT: {len(shorts)}")

    te_all = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
    te_low = [r for r in te_all if float(r.get("sync_k") or 0) < 200_000]
    dow_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

    # ── BA · April exclusion ──────────────────────────────────────────────────
    hdr("BA1 · April exclusion: full impact (TIME_EXIT + sync<200K)")
    april    = [r for r in te_low if r["_month"] == "2026-04"]
    non_apr  = [r for r in te_low if r["_month"] != "2026-04"]
    row("with April (current)", stat([r["_net"] for r in te_low]))
    row("without April (proposed)", stat([r["_net"] for r in non_apr]))
    row("April only", stat([r["_net"] for r in april]))
    print(f"\n  April events removed: {len(april)}")
    print(f"  Non-April avg improvement: "
          f"{stat([r['_net'] for r in non_apr])['avg']:+.1f} vs "
          f"{stat([r['_net'] for r in te_low])['avg']:+.1f} bps")

    hdr("BA2 · April exclusion: monthly frequency impact")
    months_te_low: dict[str,int] = {}
    for r in te_low: months_te_low[r["_month"]] = months_te_low.get(r["_month"],0)+1
    total = sum(months_te_low.values())
    non_apr_total = sum(v for k,v in months_te_low.items() if k != "2026-04")
    n_months = len([k for k in months_te_low if k != "2026-04"])
    print(f"  Current:  {total} trades over {len(months_te_low)} months = {total/len(months_te_low):.1f}/month")
    print(f"  Excl Apr: {non_apr_total} trades over {n_months} months = {non_apr_total/max(n_months,1):.1f}/month")
    print(f"  April is {100*len(april)/max(total,1):.0f}% of all new-gate trades")
    print(f"  Monthly freq drop: {total/len(months_te_low):.1f} -> {non_apr_total/max(n_months,1):.1f} trades/month")

    hdr("BA3 · April pattern: is it calendar month or BTC regime?")
    # All TIME_EXIT (any sync) in April vs others
    row("April TIME_EXIT all sync",     stat([r["_net"] for r in te_all if r["_month"]=="2026-04"]))
    row("Non-April TIME_EXIT all sync", stat([r["_net"] for r in te_all if r["_month"]!="2026-04"]))
    row("April sync<200K", stat([r["_net"] for r in april]))
    print("\n  April events detail (sync<200K):")
    for r in sorted(april, key=lambda x: x.get("anchor_ts_ms") or 0):
        print(f"    n2h={r.get('n2h'):2d}  sess={r.get('session')}  "
              f"sync={float(r.get('sync_k') or 0)/1000:.0f}K  net={r['_net']:+.1f} bps")

    # ── BB · Cascade notional <500K filter ───────────────────────────────────
    hdr("BB1 · Cascade notional <500K filter (TIME_EXIT + sync<200K, from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            te_low_notional = []
            missing = 0
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                n = cascade_notional_at(conn, ts)
                if n is None: missing += 1; continue
                r["_notional"] = n
                te_low_notional.append(r)
            print(f"  Matched: {len(te_low_notional)}/{len(te_low)} (missing={missing})")
            all_vals = [r["_net"] for r in te_low_notional]
            lt500   = [r["_net"] for r in te_low_notional if r["_notional"] < 500_000]
            gt500   = [r["_net"] for r in te_low_notional if r["_notional"] >= 500_000]
            lt300   = [r["_net"] for r in te_low_notional if r["_notional"] < 300_000]
            row("all (baseline sync<200K)", stat(all_vals))
            row("notional < 300K", stat(lt300))
            row("notional < 500K (proposed)", stat(lt500))
            row("notional >= 500K (blocked)", stat(gt500))
            print(f"\n  Events blocked by <500K filter: {len(gt500)}")
            print(f"  Avg improvement: {stat(lt500)['avg']:+.1f} vs {stat(all_vals)['avg']:+.1f} bps")
            # Notional distribution
            print("\n  Notional bands in sync<200K TIME_EXIT:")
            bands = [(0,200),(200,300),(300,500),(500,750),(750,9999)]
            for lo,hi in bands:
                sub = [r["_net"] for r in te_low_notional
                       if lo*1000 <= r["_notional"] < hi*1000]
                lbl = f"{lo}K-{hi}K" if hi<9999 else f"{lo}K+"
                row(lbl, stat(sub))
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── BC · US Tuesday exclusion ─────────────────────────────────────────────
    hdr("BC1 · US Tuesday exclusion (in sync<200K TIME_EXIT)")
    te_us = [r for r in te_low if r.get("session")=="US"]
    us_tue    = [r["_net"] for r in te_us if r.get("dow")==1]
    us_no_tue = [r["_net"] for r in te_us if r.get("dow")!=1]
    all_no_tue = [r["_net"] for r in te_low if r.get("dow")!=1]
    row("US all (baseline)", stat([r["_net"] for r in te_us]))
    row("US excl Tue", stat(us_no_tue))
    row("US Tue only", stat(us_tue))
    print()
    row("All sessions all (baseline)", stat([r["_net"] for r in te_low]))
    row("All sessions excl Tue", stat(all_no_tue))
    print(f"\n  US Tue events: {len(us_tue)}, removing saves "
          f"{sum(us_tue):+.0f} bps total, "
          f"avg change {stat(us_no_tue)['avg']:+.1f} vs {stat([r['_net'] for r in te_us])['avg']:+.1f} bps")

    hdr("BC2 · DOW breakdown for non-US (ASIA+OFF, sync<200K)")
    te_non_us = [r for r in te_low if r.get("session")!="US"]
    for d in range(7):
        sub = [r["_net"] for r in te_non_us if r.get("dow")==d]
        if sub: row(f"ASIA+OFF {dow_names[d]}", stat(sub))

    # ── BD · long_score==3 paradox ────────────────────────────────────────────
    hdr("BD1 · long_score==3 paradox: why better than score==4 in sync<200K?")
    ls3 = [r for r in te_low if (r.get("long_score") or 0)==3]
    ls4 = [r for r in te_low if (r.get("long_score") or 0)==4]
    print(f"  long_score==3: N={len(ls3)}, WR={stat([r['_net'] for r in ls3])['wr']*100:.0f}%, avg={stat([r['_net'] for r in ls3])['avg']:+.1f} bps")
    print(f"  long_score==4: N={len(ls4)}, WR={stat([r['_net'] for r in ls4])['wr']*100:.0f}%, avg={stat([r['_net'] for r in ls4])['avg']:+.1f} bps")

    # Compare features between score==3 and score==4 events
    def avg_feat(recs, feat):
        vals = [float(r.get(feat) or 0) for r in recs]
        return sum(vals)/len(vals) if vals else 0

    print("\n  Feature comparison (score==3 vs score==4):")
    for feat in ["n2h","sync_k"]:
        v3 = avg_feat(ls3, feat)
        v4 = avg_feat(ls4, feat)
        scale = 1000 if feat=="sync_k" else 1
        unit = "K" if feat=="sync_k" else ""
        print(f"  {feat:<12} score==3: {v3/scale:.1f}{unit}   score==4: {v4/scale:.1f}{unit}")

    def savg(vals): return f"{stat(vals)['avg']:+.1f}" if vals else "-"

    print("\n  Session breakdown:")
    for sess in ["US","ASIA","OFF"]:
        s3 = [r["_net"] for r in ls3 if r.get("session")==sess]
        s4 = [r["_net"] for r in ls4 if r.get("session")==sess]
        if s3 or s4:
            print(f"  {sess}: score==3 N={len(s3)} avg={savg(s3)}   "
                  f"score==4 N={len(s4)} avg={savg(s4)}")

    print("\n  n2h breakdown:")
    for thr in [3, 4, 5]:
        s3 = [r["_net"] for r in ls3 if (r.get("n2h") or 0)>=thr]
        s4 = [r["_net"] for r in ls4 if (r.get("n2h") or 0)>=thr]
        print(f"  n2h>={thr}: score==3 N={len(s3)} avg={savg(s3)}   "
              f"score==4 N={len(s4)} avg={savg(s4)}")

    print("\n  Month breakdown:")
    for m in sorted(set(r["_month"] for r in te_low)):
        s3 = [r["_net"] for r in ls3 if r["_month"]==m]
        s4 = [r["_net"] for r in ls4 if r["_month"]==m]
        print(f"  {m}: score==3 N={len(s3)} avg={savg(s3)}   "
              f"score==4 N={len(s4)} avg={savg(s4)}")

    # ── BE · SHORT BTC confirm size ───────────────────────────────────────────
    hdr("BE1 · SHORT BTC confirm size → better outcome? (from DB)")
    s4_shorts = [r for r in shorts if (r.get("score") or 0)>=4]
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in s4_shorts:
                ts = int(r.get("anchor_ts_ms") or 0)
                btc_n = btc_confirm_notional(conn, ts)
                r["_btc_confirm"] = btc_n
            matched = [r for r in s4_shorts if r.get("_btc_confirm") is not None]
            print(f"  BTC confirm notional matched: {len(matched)}/{len(s4_shorts)}")
            for thresh_m in [1.0, 2.0, 3.0, 5.0]:
                sub = [r["_net"] for r in matched if r["_btc_confirm"] >= thresh_m*1_000_000]
                row(f"BTC confirm >= {thresh_m:.0f}M", stat(sub))
            print("\n  BTC confirm size bands:")
            bands = [(1,2),(2,3),(3,5),(5,10),(10,999)]
            for lo,hi in bands:
                sub = [r["_net"] for r in matched
                       if lo*1_000_000 <= r["_btc_confirm"] < hi*1_000_000]
                row(f"BTC {lo}M-{hi}M", stat(sub))
            # Distribution
            if matched:
                sorted_btc = sorted(r["_btc_confirm"]/1_000_000 for r in matched)
                print(f"\n  BTC confirm sizes (M): {[round(x,1) for x in sorted_btc]}")
                print(f"  Median: {sorted_btc[len(sorted_btc)//2]:.1f}M")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── BF · SHORT btc7d ─────────────────────────────────────────────────────
    hdr("BF1 · SHORT btc7d: does regime matter for SHORTs? (score>=4)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in s4_shorts:
                ts = int(r.get("anchor_ts_ms") or 0)
                b7 = btc7d_bps_at(conn, ts)
                r["_btc7d"] = b7
            matched7 = [r for r in s4_shorts if r.get("_btc7d") is not None]
            print(f"  btc7d loaded for {len(matched7)}/{len(s4_shorts)} score>=4 SHORTs")
            row("score>=4 baseline", stat([r["_net"] for r in matched7]))
            row("btc7d > 0 (BTC rising)", stat([r["_net"] for r in matched7 if r["_btc7d"]>0]))
            row("btc7d < 0 (BTC falling)", stat([r["_net"] for r in matched7 if r["_btc7d"]<0]))
            row("btc7d > 100", stat([r["_net"] for r in matched7 if r["_btc7d"]>100]))
            row("btc7d > 200", stat([r["_net"] for r in matched7 if r["_btc7d"]>200]))
            # Distribution
            if matched7:
                vals7 = sorted(r["_btc7d"] for r in matched7)
                print(f"\n  btc7d distribution for score>=4 SHORTs:")
                print(f"  min={vals7[0]:+.0f}  median={vals7[len(vals7)//2]:+.0f}  max={vals7[-1]:+.0f} bps")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── BG · US hour-of-day ───────────────────────────────────────────────────
    hdr("BG1 · US hour-of-day analysis (sync<200K TIME_EXIT, US session)")
    # US session = roughly 13:00-21:00 UTC
    te_us_all = [r for r in te_low if r.get("session")=="US"]
    print(f"  US sync<200K TIME_EXIT: N={len(te_us_all)}")
    print("\n  --- by UTC hour ---")
    for h in range(13, 23):
        sub = [r["_net"] for r in te_us_all if r["_hour"]==h]
        if sub: row(f"  hour {h:02d}:00 UTC", stat(sub))

    print("\n  --- by hour band ---")
    bands_h = [("early US (13-16)", range(13,17)),
               ("mid US (16-19)",   range(16,20)),
               ("late US (19-22)",  range(19,23))]
    for label, hrs in bands_h:
        sub = [r["_net"] for r in te_us_all if r["_hour"] in hrs]
        row(label, stat(sub))

    # All sessions hour
    hdr("BG2 · All sessions hour-of-day (sync<200K TIME_EXIT)")
    for h in range(0, 24):
        sub = [r["_net"] for r in te_low if r["_hour"]==h]
        if sub: row(f"  hour {h:02d}:00 UTC", stat(sub))

    # ── BH · Combined gate projection ────────────────────────────────────────
    hdr("BH1 · Combined gate: sync<200K + April excl + notional<500K")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in te_low:
                if r.get("_notional") is None:
                    ts = int(r.get("anchor_ts_ms") or 0)
                    r["_notional"] = cascade_notional_at(conn, ts)
            # All gates applied
            gate1 = [r for r in te_low]  # sync<200K (base)
            gate2 = [r for r in gate1 if r["_month"] != "2026-04"]  # + no April
            gate3 = [r for r in gate2 if r.get("_notional") is None or r["_notional"] < 500_000]  # + notional<500K
            gate4 = [r for r in gate3 if r.get("dow") not in {1}]  # + no US Tue (tentative)

            row("gate1: sync<200K", stat([r["_net"] for r in gate1]))
            row("gate2: + no April", stat([r["_net"] for r in gate2]))
            row("gate3: + notional<500K", stat([r["_net"] for r in gate3]))
            row("gate4: + no US Tue", stat([r["_net"] for r in gate4]))

            # Monthly breakdown for gate3 (the recommended combined)
            print("\n  Gate3 monthly breakdown (sync<200K + no April + notional<500K):")
            m_g3: dict[str,list] = {}
            for r in gate3: m_g3.setdefault(r["_month"],[]).append(r["_net"])
            for m in sorted(m_g3):
                v = m_g3[m]
                row(f"  {m}", stat(v))
            total_g3 = sum(len(v) for v in m_g3.values())
            print(f"\n  Gate3 total: {total_g3} trades over {len(m_g3)} months = {total_g3/len(m_g3):.1f}/month")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── BI · n2h>=5 vs n2h>=3 trade-off ──────────────────────────────────────
    hdr("BI1 · n2h threshold frequency/quality trade-off (sync<200K TIME_EXIT)")
    configs = [
        ("n2h>=2 (permissive)", lambda r: (r.get("n2h") or 0)>=2),
        ("n2h>=3 (current)", lambda r: (r.get("n2h") or 0)>=3),
        ("n2h>=4", lambda r: (r.get("n2h") or 0)>=4),
        ("n2h>=5 (strict)", lambda r: (r.get("n2h") or 0)>=5),
    ]
    print(f"  {'Config':<30} {'N':>4}  {'WR':>7}  {'avg':>10}  {'tot':>10}  {'N/mo':>6}")
    n_months_data = len(set(r["_month"] for r in te_low))
    for label, fn in configs:
        sub = [r["_net"] for r in te_low if fn(r)]
        s = stat(sub)
        nmo = len(sub)/max(n_months_data,1)
        row(f"{label}", s, f"~{nmo:.1f}/mo")

    hdr("BI2 · Expected annual P&L at different n2h gates (assume 40x, $11 notional)")
    # At 40x leverage, $11 notional, 1 bps = $0.011 profit
    notional = 11.0
    for label, fn in configs:
        sub = [r["_net"] for r in te_low if fn(r)]
        if not sub: continue
        s = stat(sub)
        nmo = len(sub)/max(n_months_data,1)
        annual_trades = nmo * 12
        avg_bps = s["avg"] or 0
        annual_bps = annual_trades * avg_bps
        annual_usd = annual_bps * notional / 10_000
        fee_drag = annual_trades * 5.0  # 5 bps fee round-trip assumed (already in net_bps, shown for info)
        print(f"  {label:<28} {nmo:.1f}/mo  {annual_trades:.0f} trades/yr  "
              f"avg {avg_bps:+.1f} bps  est ~{annual_bps:+.0f} bps/yr")

    print()
    print("="*68)
    print("  ALL ROUND-6 TESTS COMPLETE")
    print("="*68)

if __name__ == "__main__":
    main()
