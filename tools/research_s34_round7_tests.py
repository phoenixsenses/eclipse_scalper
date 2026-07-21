"""S34 Round-7 Tests — Sections CA through CJ.

  CA  US 13-14 UTC exclusion: full impact on new-gate population
  CB  sync_k=0 events: pure ETH idiosyncratic deep dive
  CC  Individual score components as standalone predictors
  CD  April excl vs btc7d<0: which gate subsumes the other?
  CE  Weekend (Sat+Sun) deep dive in new-gate population
  CF  SHORT BTC confirm: timing x size interaction
  CG  n2h extreme (>=10, >=20) profile
  CH  ETH 4h trend at cascade time for LONG (from DB)
  CI  US Tuesday with hour filter — salvageable?
  CJ  btc7d<-100 as hard gate on top of sync<200K

Usage: python tools/research_s34_round7_tests.py
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
        print(f"  {label:<40s}  N=  0  -----  ------  ------  {note}"); return
    print(f"  {label:<40s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
          f"avg={fmt(s['avg'])} bps  tot={fmt(s['total'],0)} bps  {note}")

def savg(vals): return f"{stat(vals)['avg']:+.1f}" if vals else "-"
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
            r["_net"]   = float(net)
            r["_month"] = month_key(r.get("anchor_ts_ms",0))
            r["_hour"]  = hour_utc(r.get("anchor_ts_ms",0))
            if r.get("direction") == "LONG":  longs.append(r)
            elif r.get("direction") == "SHORT": shorts.append(r)
    return longs, shorts

def btc7d_at(conn, ts_ms):
    lo = ts_ms - 7*24*3600_000
    a = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",(lo,)).fetchone()
    b = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0)<=0: return None
    return (float(b[0])-float(a[0]))/float(a[0])*10_000.0

def eth4h_at(conn, ts_ms):
    lo = ts_ms - 4*3600_000
    a = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",(lo,)).fetchone()
    b = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",(ts_ms,)).fetchone()
    if not a or not b or float(a[0] or 0)<=0: return None
    return (float(b[0])-float(a[0]))/float(a[0])*10_000.0

def btc_confirm_info(conn, ts_ms, lo_ms=60_000, hi_ms=30*60_000):
    r = conn.execute(
        "SELECT ts_ms, notional FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<=? AND notional>=1000000 ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms+lo_ms, ts_ms+hi_ms)
    ).fetchone()
    if not r: return None, None
    delay_ms = int(r[0]) - ts_ms
    return delay_ms, float(r[1])

def main():
    longs, shorts = load_records()
    print(f"Loading backfill ledger...")
    print(f"LONG: {len(longs)} ({sum(1 for r in longs if r.get('close_reason')=='TIME_EXIT')} TIME_EXIT, "
          f"{sum(1 for r in longs if r.get('close_reason')=='NOISY_EARLY_EXIT')} NOISY)")
    print(f"SHORT: {len(shorts)}")

    te_all = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
    te_low = [r for r in te_all if float(r.get("sync_k") or 0) < 200_000]
    s4     = [r for r in shorts if (r.get("score") or 0) >= 4]

    # ── CA · US 13-14 UTC exclusion ──────────────────────────────────────────
    hdr("CA1 · US 13-14 UTC exclusion (sync<200K TIME_EXIT)")
    te_us = [r for r in te_low if r.get("session")=="US"]
    early_us  = [r for r in te_us if r["_hour"] in {13,14}]
    late_us   = [r for r in te_us if r["_hour"] not in {13,14}]
    non_us    = [r for r in te_low if r.get("session")!="US"]
    all_excl  = late_us + non_us

    row("US 13-14 UTC (early, weak)", stat([r["_net"] for r in early_us]))
    row("US 15-22 UTC (rest)", stat([r["_net"] for r in late_us]))
    row("non-US sessions", stat([r["_net"] for r in non_us]))
    print()
    row("baseline (all sync<200K)", stat([r["_net"] for r in te_low]))
    row("excl US 13-14 UTC", stat([r["_net"] for r in all_excl]))
    print(f"\n  Removing US 13-14 UTC: {len(early_us)} events, "
          f"avg delta: {savg([r['_net'] for r in all_excl])} vs {savg([r['_net'] for r in te_low])} bps")

    hdr("CA2 · US hour breakdown: can 13-14 UTC be rescued by n2h?")
    for h in [13,14]:
        for thr in [3,5]:
            sub = [r["_net"] for r in te_us if r["_hour"]==h and (r.get("n2h") or 0)>=thr]
            row(f"hour {h:02d} n2h>={thr}", stat(sub))
        sub_all = [r["_net"] for r in te_us if r["_hour"]==h]
        row(f"hour {h:02d} all", stat(sub_all))
        print()

    # ── CB · sync_k=0 deep dive ───────────────────────────────────────────────
    hdr("CB1 · sync_k=0 events: pure ETH idiosyncratic")
    sk0     = [r for r in te_low if float(r.get("sync_k") or 0) == 0]
    sk_low  = [r for r in te_low if 0 < float(r.get("sync_k") or 0) < 50_000]
    sk_mid  = [r for r in te_low if 50_000 <= float(r.get("sync_k") or 0) < 100_000]
    sk_high = [r for r in te_low if 100_000 <= float(r.get("sync_k") or 0) < 200_000]
    row("sync==0 (pure ETH)", stat([r["_net"] for r in sk0]))
    row("sync 0-50K", stat([r["_net"] for r in [r for r in te_low if float(r.get("sync_k") or 0)<50_000]]))
    row("sync 50-100K", stat([r["_net"] for r in sk_mid]))
    row("sync 100-200K", stat([r["_net"] for r in sk_high]))
    row("all sync<200K (baseline)", stat([r["_net"] for r in te_low]))

    hdr("CB2 · sync_k=0 by session and n2h")
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in sk0 if r.get("session")==sess]
        if sub: row(f"sync=0 {sess}", stat(sub))
    print()
    for thr in [3,5]:
        sub = [r["_net"] for r in sk0 if (r.get("n2h") or 0)>=thr]
        row(f"sync=0 n2h>={thr}", stat(sub))
    # List individual events
    print(f"\n  sync=0 individual events (N={len(sk0)}):")
    for r in sorted(sk0, key=lambda x: x.get("anchor_ts_ms") or 0):
        dt = datetime.fromtimestamp(int(r.get("anchor_ts_ms") or 0)/1000,tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {dt}  n2h={r.get('n2h'):2d}  sess={r.get('session')}  net={r['_net']:+.1f} bps")

    # ── CC · Individual score components ─────────────────────────────────────
    hdr("CC1 · Individual score components as standalone predictors (TIME_EXIT all)")
    # Reconstruct each component from records
    # Score = n2h>=3 + btc4h<0 + vdepth>=30 + sess==US + sync_k>=200K
    # We have: n2h, sync_k, session from ledger
    # btc4h and vdepth we can infer from long_score - silence(1) - known components
    # But let's just use what we have directly
    def has_n2h3(r):   return (r.get("n2h") or 0) >= 3
    def has_sync(r):   return float(r.get("sync_k") or 0) >= 200_000
    def is_us(r):      return r.get("session") == "US"
    # For btc4h and vdepth we need DB — do a separate section

    print("  Using ledger-available features only (n2h, sync_k, session):")
    row("n2h>=3 only",          stat([r["_net"] for r in te_all if has_n2h3(r)]))
    row("n2h<3 only",           stat([r["_net"] for r in te_all if not has_n2h3(r)]))
    row("sync_k>=200K only",    stat([r["_net"] for r in te_all if has_sync(r)]))
    row("sync_k<200K only",     stat([r["_net"] for r in te_all if not has_sync(r)]))
    row("sess==US only",        stat([r["_net"] for r in te_all if is_us(r)]))
    row("sess!=US only",        stat([r["_net"] for r in te_all if not is_us(r)]))

    hdr("CC2 · Component interactions (2x2)")
    row("n2h>=3 AND sync<200K",   stat([r["_net"] for r in te_all if has_n2h3(r) and not has_sync(r)]))
    row("n2h>=3 AND sync>=200K",  stat([r["_net"] for r in te_all if has_n2h3(r) and has_sync(r)]))
    row("n2h<3  AND sync<200K",   stat([r["_net"] for r in te_all if not has_n2h3(r) and not has_sync(r)]))
    row("n2h<3  AND sync>=200K",  stat([r["_net"] for r in te_all if not has_n2h3(r) and has_sync(r)]))
    print()
    row("n2h>=3 AND !US",         stat([r["_net"] for r in te_all if has_n2h3(r) and not is_us(r)]))
    row("n2h>=3 AND US",          stat([r["_net"] for r in te_all if has_n2h3(r) and is_us(r)]))
    row("sync<200K AND !US",      stat([r["_net"] for r in te_all if not has_sync(r) and not is_us(r)]))
    row("sync<200K AND US",       stat([r["_net"] for r in te_all if not has_sync(r) and is_us(r)]))

    # ── CD · April excl vs btc7d<0 overlap ───────────────────────────────────
    hdr("CD1 · April vs btc7d<0: which gate subsumes the other? (from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                r["_btc7d"] = btc7d_at(conn, ts)

            april     = [r for r in te_low if r["_month"]=="2026-04"]
            non_apr   = [r for r in te_low if r["_month"]!="2026-04"]
            btc_neg   = [r for r in te_low if (r.get("_btc7d") or 0) < 0]
            btc_pos   = [r for r in te_low if (r.get("_btc7d") or 1) >= 0]

            print("  Overlap analysis:")
            apr_btcneg = [r for r in april if (r.get("_btc7d") or 0) < 0]
            apr_btcpos = [r for r in april if (r.get("_btc7d") or 1) >= 0]
            print(f"  April total: {len(april)}")
            print(f"  April with btc7d<0: {len(apr_btcneg)} ({100*len(apr_btcneg)/max(len(april),1):.0f}%)")
            print(f"  April with btc7d>=0: {len(apr_btcpos)} ({100*len(apr_btcpos)/max(len(april),1):.0f}%)")
            print(f"\n  => btc7d<0 blocks {len(apr_btcpos)}/{len(april)} April events automatically")
            print(f"  => {len(apr_btcneg)} April events SURVIVE btc7d<0 filter")
            print()

            row("April excl only",      stat([r["_net"] for r in non_apr]))
            row("btc7d<0 only",         stat([r["_net"] for r in btc_neg]))
            row("April excl + btc7d<0", stat([r["_net"] for r in btc_neg if r["_month"]!="2026-04"]))
            row("baseline sync<200K",   stat([r["_net"] for r in te_low]))
            print()
            # Which is the better single gate?
            print("  Single gate comparison (what each alone does):")
            row("+ April excl",    stat([r["_net"] for r in non_apr]))
            row("+ btc7d<0",       stat([r["_net"] for r in btc_neg]))
            # April-surviving btc7d<0 events
            if apr_btcneg:
                print(f"\n  April events that survive btc7d<0 filter ({len(apr_btcneg)}):")
                for r in apr_btcneg:
                    print(f"    n2h={r.get('n2h')}  sess={r.get('session')}  btc7d={r['_btc7d']:+.0f}  net={r['_net']:+.1f} bps")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── CE · Weekend deep dive ────────────────────────────────────────────────
    hdr("CE1 · Weekend (Sat+Sun) deep dive (sync<200K TIME_EXIT)")
    weekend  = [r for r in te_low if r.get("dow") in {5,6}]
    weekday  = [r for r in te_low if r.get("dow") not in {5,6}]
    sat      = [r for r in te_low if r.get("dow")==5]
    sun      = [r for r in te_low if r.get("dow")==6]
    row("Saturday", stat([r["_net"] for r in sat]))
    row("Sunday",   stat([r["_net"] for r in sun]))
    row("Weekend total", stat([r["_net"] for r in weekend]))
    row("Weekday total", stat([r["_net"] for r in weekday]))

    hdr("CE2 · Weekend by session and n2h")
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in weekend if r.get("session")==sess]
        if sub: row(f"Weekend {sess}", stat(sub))
    print()
    for thr in [3,4,5]:
        sub = [r["_net"] for r in weekend if (r.get("n2h") or 0)>=thr]
        row(f"Weekend n2h>={thr}", stat(sub))
    print(f"\n  Weekend events detail:")
    for r in sorted(weekend, key=lambda x: x.get("anchor_ts_ms") or 0):
        dt = datetime.fromtimestamp(int(r.get("anchor_ts_ms") or 0)/1000,tz=timezone.utc).strftime("%Y-%m-%d %a")
        print(f"    {dt}  n2h={r.get('n2h'):2d}  sess={r.get('session')}  "
              f"sync={float(r.get('sync_k') or 0)/1000:.0f}K  net={r['_net']:+.1f} bps")

    # ── CF · SHORT BTC confirm timing x size ─────────────────────────────────
    hdr("CF1 · SHORT BTC confirm: timing x size interaction (score>=4, from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in s4:
                ts = int(r.get("anchor_ts_ms") or 0)
                delay, notional = btc_confirm_info(conn, ts)
                r["_btc_delay_min"] = delay/60_000 if delay else None
                r["_btc_confirm_m"] = notional/1_000_000 if notional else None

            matched = [r for r in s4 if r.get("_btc_delay_min") is not None]
            print(f"  BTC confirm info matched: {len(matched)}/{len(s4)}")

            # Timing buckets
            print("\n  --- by timing ---")
            for lo,hi in [(0,5),(5,10),(10,20),(20,30)]:
                sub = [r["_net"] for r in matched if lo <= r["_btc_delay_min"] < hi]
                row(f"  BTC confirm {lo}-{hi} min", stat(sub))

            # Size buckets
            print("\n  --- by size ---")
            for lo,hi in [(1,2),(2,3),(3,5),(5,99)]:
                sub = [r["_net"] for r in matched if lo <= (r["_btc_confirm_m"] or 0) < hi]
                row(f"  BTC {lo}M-{hi}M", stat(sub))

            # Combined: timing x size
            print("\n  --- timing x size combos ---")
            row("early (<10min) + large (>=2M)",
                stat([r["_net"] for r in matched
                      if r["_btc_delay_min"]<10 and (r["_btc_confirm_m"] or 0)>=2]))
            row("late (>=10min) + large (>=2M)",
                stat([r["_net"] for r in matched
                      if r["_btc_delay_min"]>=10 and (r["_btc_confirm_m"] or 0)>=2]))
            row("early (<10min) + small (<2M)",
                stat([r["_net"] for r in matched
                      if r["_btc_delay_min"]<10 and (r["_btc_confirm_m"] or 0)<2]))
            row("late (>=10min) + small (<2M)",
                stat([r["_net"] for r in matched
                      if r["_btc_delay_min"]>=10 and (r["_btc_confirm_m"] or 0)<2]))

            # Full table
            print("\n  Individual SHORTs (score>=4):")
            for r in sorted(matched, key=lambda x: x.get("anchor_ts_ms") or 0):
                dt = datetime.fromtimestamp(int(r.get("anchor_ts_ms") or 0)/1000,tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"    {dt}  delay={r['_btc_delay_min']:.0f}min  "
                      f"btc={r['_btc_confirm_m']:.1f}M  net={r['_net']:+.1f} bps")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── CG · n2h extreme ─────────────────────────────────────────────────────
    hdr("CG1 · n2h extreme values (sync<200K TIME_EXIT)")
    for thr in [5, 8, 10, 15, 20]:
        sub = [r["_net"] for r in te_low if (r.get("n2h") or 0) >= thr]
        row(f"n2h >= {thr}", stat(sub))
    print("\n  n2h distribution:")
    from collections import Counter
    n2h_counts = Counter((r.get("n2h") or 0) for r in te_low)
    for n in sorted(n2h_counts):
        sub = [r["_net"] for r in te_low if (r.get("n2h") or 0) == n]
        print(f"  n2h={n:2d}: N={len(sub)}  avg={savg(sub)} bps")

    hdr("CG2 · n2h extreme by session")
    for thr in [10, 15]:
        for sess in ["US","ASIA","OFF"]:
            sub = [r["_net"] for r in te_low if (r.get("n2h") or 0)>=thr and r.get("session")==sess]
            if sub: row(f"n2h>={thr} {sess}", stat(sub))

    # ── CH · ETH 4h trend at cascade ─────────────────────────────────────────
    hdr("CH1 · ETH 4h trend at cascade time (from DB, sync<200K TIME_EXIT)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            eth4h_pass, eth4h_fail, missing = [], [], 0
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                e4h = eth4h_at(conn, ts)
                if e4h is None: missing += 1; continue
                r["_eth4h"] = e4h
                if e4h < 0:   eth4h_pass.append(r)
                else:          eth4h_fail.append(r)

            print(f"  ETH 4h loaded: {len(eth4h_pass)+len(eth4h_fail)}/{len(te_low)} (missing={missing})")
            row("ETH 4h < 0 (ETH falling — cascade into downtrend)",
                stat([r["_net"] for r in eth4h_pass]))
            row("ETH 4h >= 0 (ETH rising — cascade after rally)",
                stat([r["_net"] for r in eth4h_fail]))
            # Threshold sweep
            print("\n  ETH 4h threshold sweep:")
            for thr in [-200,-100,-50,-20,0,20,50]:
                sub = [r["_net"] for r in te_low
                       if r.get("_eth4h") is not None and r["_eth4h"] < thr]
                row(f"ETH 4h < {thr:+d} bps", stat(sub))
            # Combine with n2h
            print()
            row("ETH4h<0 + n2h>=3",
                stat([r["_net"] for r in eth4h_pass if (r.get("n2h") or 0)>=3]))
            row("ETH4h<0 + n2h>=5",
                stat([r["_net"] for r in eth4h_pass if (r.get("n2h") or 0)>=5]))
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── CI · US Tuesday with hour filter ─────────────────────────────────────
    hdr("CI1 · US Tuesday: salvageable with hour filter?")
    us_tue = [r for r in te_us if r.get("dow")==1]
    print(f"  US Tue total: N={len(us_tue)}, avg={savg([r['_net'] for r in us_tue])}")
    print("\n  US Tue by hour:")
    for h in range(13,23):
        sub = [r["_net"] for r in us_tue if r["_hour"]==h]
        if sub: row(f"  US Tue hour {h:02d}:00", stat(sub))
    print("\n  US Tue excl 13-14 UTC:")
    sub_excl = [r["_net"] for r in us_tue if r["_hour"] not in {13,14}]
    row("  US Tue excl 13-14", stat(sub_excl))
    sub_early = [r["_net"] for r in us_tue if r["_hour"] in {13,14}]
    row("  US Tue 13-14 only", stat(sub_early))
    print(f"\n  US Tue individual events:")
    for r in sorted(us_tue, key=lambda x: x.get("anchor_ts_ms") or 0):
        dt = datetime.fromtimestamp(int(r.get("anchor_ts_ms") or 0)/1000,tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {dt} {r['_hour']:02d}:xx UTC  n2h={r.get('n2h'):2d}  sync={float(r.get('sync_k') or 0)/1000:.0f}K  net={r['_net']:+.1f} bps")

    # ── CJ · btc7d<-100 as hard gate ─────────────────────────────────────────
    hdr("CJ1 · btc7d<-100 vs btc7d<0: incremental benefit (sync<200K, from DB)")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            # btc7d already loaded in CD section
            te_low_b7 = [r for r in te_low if r.get("_btc7d") is not None]
            if len(te_low_b7) < 5:
                for r in te_low:
                    ts = int(r.get("anchor_ts_ms") or 0)
                    if r.get("_btc7d") is None:
                        r["_btc7d"] = btc7d_at(conn, ts)
                te_low_b7 = [r for r in te_low if r.get("_btc7d") is not None]

            print(f"  btc7d available: {len(te_low_b7)}/{len(te_low)}")
            row("sync<200K baseline", stat([r["_net"] for r in te_low_b7]))
            row("+ btc7d < 0",        stat([r["_net"] for r in te_low_b7 if r["_btc7d"]<0]))
            row("+ btc7d < -50",      stat([r["_net"] for r in te_low_b7 if r["_btc7d"]<-50]))
            row("+ btc7d < -100",     stat([r["_net"] for r in te_low_b7 if r["_btc7d"]<-100]))
            row("+ btc7d < -200",     stat([r["_net"] for r in te_low_b7 if r["_btc7d"]<-200]))
            # Threshold removed events and their quality
            print()
            b7_0    = [r for r in te_low_b7 if r["_btc7d"]>=0]
            b7_0_100= [r for r in te_low_b7 if 0<=r["_btc7d"]<100]
            b7_50   = [r for r in te_low_b7 if -50<=r["_btc7d"]<0]
            row("removed by btc7d>=0 (18 events)",   stat([r["_net"] for r in b7_0]))
            row("removed by btc7d 0-100",             stat([r["_net"] for r in b7_0_100]))
            row("removed by btc7d -50 to 0",          stat([r["_net"] for r in b7_50]))
            print()
            # Combined with April excl
            row("btc7d<0 + no April",
                stat([r["_net"] for r in te_low_b7
                      if r["_btc7d"]<0 and r["_month"]!="2026-04"]))
            row("btc7d<-100 + no April",
                stat([r["_net"] for r in te_low_b7
                      if r["_btc7d"]<-100 and r["_month"]!="2026-04"]))
            # Note: April btc7d<0 count
            apr_b7neg = [r for r in te_low_b7 if r["_month"]=="2026-04" and r["_btc7d"]<0]
            print(f"\n  April events with btc7d<0: {len(apr_b7neg)} "
                  f"(btc7d<0 alone leaves {len(apr_b7neg)} April events in)")
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    print()
    print("="*68)
    print("  ALL ROUND-7 TESTS COMPLETE")
    print("="*68)

if __name__ == "__main__":
    main()
