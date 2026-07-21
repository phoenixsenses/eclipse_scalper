"""S34 Round-8 Tests — Sections DA through DJ.

  DA  Full pipeline: sync<200K + btc7d<0 + US excl 13-14 UTC combined
  DB  sync_k lower bound: sync=0 terrible, add sync>0/5K/10K?
  DC  n2h=4 anomaly deep dive: why worse than n2h=3?
  DD  SHORT btc4h<0 filter (from DB)
  DE  SHORT June concentration: structural or coincidence?
  DF  BTC 4h standalone gate in sync<200K population (from DB)
  DG  Cascade spacing: >=2h/4h/8h since last cascade
  DH  SHORT minimum BTC confirm delay filter (>=5min, >=10min)
  DI  ETH 4h gate frequency: how many trades blocked per month?
  DJ  Final ranking table: all gate combos ranked

Usage: python tools/research_s34_round8_tests.py
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

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
def hdr(t): print(); print("="*70); print(f"  {t}"); print("="*70)
def row(label, s, note=""):
    if s["n"]==0:
        print(f"  {label:<42s}  N=  0  -----  ------  ------  {note}"); return
    print(f"  {label:<42s}  N={s['n']:4d}  WR={pct(s['wr'])}  "
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

def btc4h_at(conn, ts_ms):
    lo = ts_ms - 4*3600_000
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

def btc_confirm_delay(conn, ts_ms):
    r = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' "
        "AND ts_ms>=? AND ts_ms<=? AND notional>=1000000 ORDER BY ts_ms ASC LIMIT 1",
        (ts_ms+60_000, ts_ms+30*60_000)
    ).fetchone()
    return (int(r[0])-ts_ms)/60_000 if r else None

def main():
    longs, shorts = load_records()
    print(f"Loading backfill ledger...")
    print(f"LONG: {len(longs)} ({sum(1 for r in longs if r.get('close_reason')=='TIME_EXIT')} TIME_EXIT, "
          f"{sum(1 for r in longs if r.get('close_reason')=='NOISY_EARLY_EXIT')} NOISY)")
    print(f"SHORT: {len(shorts)}")

    te_all  = [r for r in longs if r.get("close_reason")=="TIME_EXIT"]
    te_low  = [r for r in te_all  if float(r.get("sync_k") or 0) < 200_000]
    s4      = [r for r in shorts  if (r.get("score") or 0) >= 4]
    n_months = len(set(r["_month"] for r in te_low))

    # ── Pre-load DB features for all te_low events ────────────────────────────
    print("  Pre-loading DB features (btc7d, btc4h, eth4h)...")
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=15) as conn:
            for r in te_low:
                ts = int(r.get("anchor_ts_ms") or 0)
                r["_btc7d"] = btc7d_at(conn, ts)
                r["_btc4h"] = btc4h_at(conn, ts)
                r["_eth4h"] = eth4h_at(conn, ts)
            for r in s4:
                ts = int(r.get("anchor_ts_ms") or 0)
                r["_btc4h"] = btc4h_at(conn, ts)
                r["_btc7d"] = btc7d_at(conn, ts)
                r["_delay"] = btc_confirm_delay(conn, ts)
        print(f"  DB features loaded for {len(te_low)} LONG + {len(s4)} SHORT events")
    except Exception as e:
        print(f"  [DB ERROR during pre-load] {e}")

    te_low_b7 = [r for r in te_low if r.get("_btc7d") is not None]

    # ── DA · Full pipeline combined gate ──────────────────────────────────────
    hdr("DA1 · Full pipeline: all gates combined (TIME_EXIT + sync<200K base)")
    def is_early_us(r): return r.get("session")=="US" and r["_hour"] in {13,14}
    def btc7d_neg(r):   return (r.get("_btc7d") or 1) < 0
    def sync_low(r):    return float(r.get("sync_k") or 0) < 200_000
    def no_april(r):    return r["_month"] != "2026-04"

    gates = [
        ("sync<200K (base)",                 [r for r in te_low]),
        ("+ excl US 13-14 UTC",              [r for r in te_low if not is_early_us(r)]),
        ("+ btc7d<0",                        [r for r in te_low_b7 if btc7d_neg(r)]),
        ("+ excl US 13-14 + btc7d<0",        [r for r in te_low_b7 if not is_early_us(r) and btc7d_neg(r)]),
        ("+ btc7d<0 + ETH4h<0",             [r for r in te_low_b7 if btc7d_neg(r) and (r.get("_eth4h") or 0)<0]),
        ("+ excl US 13-14 + btc7d<0 + ETH4h<-50",
             [r for r in te_low_b7 if not is_early_us(r) and btc7d_neg(r) and (r.get("_eth4h") or 0)<-50]),
    ]
    for label, pop in gates:
        row(label, stat([r["_net"] for r in pop]))

    hdr("DA2 · Full pipeline monthly stability")
    best_pop = [r for r in te_low_b7 if not is_early_us(r) and btc7d_neg(r)]
    m_best: dict[str,list] = {}
    for r in best_pop: m_best.setdefault(r["_month"],[]).append(r["_net"])
    for m in sorted(m_best):
        row(f"  {m}", stat(m_best[m]))
    row("ALL", stat([r["_net"] for r in best_pop]))
    total = sum(len(v) for v in m_best.values())
    print(f"\n  Full pipeline: {total} trades over {len(m_best)} months = {total/max(len(m_best),1):.1f}/month")

    # ── DB · sync_k lower bound ───────────────────────────────────────────────
    hdr("DB1 · sync_k lower bound: exclude ultra-low sync (TIME_EXIT + sync<200K)")
    for lo in [0, 1000, 5000, 10000, 20000]:
        sub = [r["_net"] for r in te_low if float(r.get("sync_k") or 0) >= lo]
        row(f"sync >= {lo//1000}K", stat(sub))
    print()
    # Show the worst performers by sync band
    print("  sync_k bands within <200K:")
    bands = [(0,1),(1,5),(5,10),(10,20),(20,50),(50,100),(100,150),(150,200)]
    for lo,hi in bands:
        sub = [r["_net"] for r in te_low
               if lo*1000 <= float(r.get("sync_k") or 0) < hi*1000]
        row(f"  sync {lo}K-{hi}K", stat(sub))

    # ── DC · n2h=4 anomaly ────────────────────────────────────────────────────
    hdr("DC1 · n2h=4 anomaly: why worse than n2h=3? (sync<200K TIME_EXIT)")
    n2h3_events = [r for r in te_low if (r.get("n2h") or 0)==3]
    n2h4_events = [r for r in te_low if (r.get("n2h") or 0)==4]
    print(f"  n2h==3: N={len(n2h3_events)}, avg={savg([r['_net'] for r in n2h3_events])} bps")
    print(f"  n2h==4: N={len(n2h4_events)}, avg={savg([r['_net'] for r in n2h4_events])} bps")

    print("\n  n2h==3 vs n2h==4 feature comparison:")
    for feat, scale, unit in [("sync_k",1000,"K"),("long_score",1,"")]:
        v3 = [float(r.get(feat) or 0) for r in n2h3_events]
        v4 = [float(r.get(feat) or 0) for r in n2h4_events]
        print(f"  {feat}: n2h=3 avg={sum(v3)/len(v3)/scale:.1f}{unit}   "
              f"n2h=4 avg={sum(v4)/len(v4)/scale:.1f}{unit}")

    print("\n  n2h==3 session breakdown:")
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in n2h3_events if r.get("session")==sess]
        if sub: print(f"    {sess}: N={len(sub)} avg={savg(sub)} bps")

    print("\n  n2h==4 session breakdown:")
    for sess in ["US","ASIA","OFF"]:
        sub = [r["_net"] for r in n2h4_events if r.get("session")==sess]
        if sub: print(f"    {sess}: N={len(sub)} avg={savg(sub)} bps")

    print("\n  n2h==3 month breakdown:")
    m3 = defaultdict(list)
    for r in n2h3_events: m3[r["_month"]].append(r["_net"])
    for m in sorted(m3): print(f"    {m}: N={len(m3[m])} avg={savg(m3[m])} bps")

    print("\n  n2h==4 month breakdown:")
    m4 = defaultdict(list)
    for r in n2h4_events: m4[r["_month"]].append(r["_net"])
    for m in sorted(m4): print(f"    {m}: N={len(m4[m])} avg={savg(m4[m])} bps")

    print("\n  n2h==4 individual events:")
    for r in sorted(n2h4_events, key=lambda x: x.get("anchor_ts_ms") or 0):
        dt = datetime.fromtimestamp(int(r.get("anchor_ts_ms") or 0)/1000,tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {dt}  sess={r.get('session')}  sync={float(r.get('sync_k') or 0)/1000:.0f}K  "
              f"ls={r.get('long_score')}  net={r['_net']:+.1f} bps")

    # ── DD · SHORT btc4h<0 ────────────────────────────────────────────────────
    hdr("DD1 · SHORT btc4h filter (score>=4, from DB)")
    s4_b4 = [r for r in s4 if r.get("_btc4h") is not None]
    print(f"  btc4h loaded: {len(s4_b4)}/{len(s4)}")
    row("score>=4 baseline", stat([r["_net"] for r in s4_b4]))
    row("btc4h < 0 (BTC falling 4h)", stat([r["_net"] for r in s4_b4 if r["_btc4h"]<0]))
    row("btc4h >= 0 (BTC rising 4h)", stat([r["_net"] for r in s4_b4 if r["_btc4h"]>=0]))
    print()
    for thr in [-200,-100,-50,0,50,100]:
        sub = [r["_net"] for r in s4_b4 if r["_btc4h"]<thr]
        row(f"btc4h < {thr:+d}", stat(sub))

    hdr("DD2 · SHORT btc4h x btc7d (score>=4)")
    s4_both = [r for r in s4 if r.get("_btc4h") is not None and r.get("_btc7d") is not None]
    print(f"  Both loaded: {len(s4_both)}")
    row("btc4h<0 AND btc7d<0", stat([r["_net"] for r in s4_both if r["_btc4h"]<0 and r["_btc7d"]<0]))
    row("btc4h<0 AND btc7d>0", stat([r["_net"] for r in s4_both if r["_btc4h"]<0 and r["_btc7d"]>=0]))
    row("btc4h>=0 AND btc7d<0", stat([r["_net"] for r in s4_both if r["_btc4h"]>=0 and r["_btc7d"]<0]))
    row("btc4h>=0 AND btc7d>=0",stat([r["_net"] for r in s4_both if r["_btc4h"]>=0 and r["_btc7d"]>=0]))

    # ── DE · SHORT June concentration ─────────────────────────────────────────
    hdr("DE1 · SHORT June concentration: structural analysis")
    print(f"  All SHORTs by month:")
    m_all_s: dict[str,list] = {}
    for r in shorts: m_all_s.setdefault(r["_month"],[]).append(r["_net"])
    for m in sorted(m_all_s):
        row(f"  {m} (all score)", stat(m_all_s[m]))
    print()
    print(f"  score>=4 SHORTs by month:")
    m_s4: dict[str,list] = {}
    for r in s4: m_s4.setdefault(r["_month"],[]).append(r["_net"])
    for m in sorted(m_s4):
        row(f"  {m} score>=4", stat(m_s4[m]))

    print("\n  June score>=4 SHORTs — what makes June different?")
    jun_s4 = [r for r in s4 if r["_month"]=="2026-06"]
    pre_s4 = [r for r in s4 if r["_month"]!="2026-06"]
    print(f"  June N={len(jun_s4)}, avg={savg([r['_net'] for r in jun_s4])}")
    print(f"  Pre-June N={len(pre_s4)}, avg={savg([r['_net'] for r in pre_s4])}")
    if jun_s4:
        n2h_j = [r.get("n2h") or 0 for r in jun_s4]
        sc_j  = [r.get("score") or 0 for r in jun_s4]
        print(f"  June avg n2h={sum(n2h_j)/len(n2h_j):.1f}  avg score={sum(sc_j)/len(sc_j):.1f}")

    # ── DF · BTC 4h standalone in sync<200K ──────────────────────────────────
    hdr("DF1 · BTC 4h gate standalone (sync<200K TIME_EXIT, from DB)")
    te_b4 = [r for r in te_low if r.get("_btc4h") is not None]
    print(f"  btc4h loaded: {len(te_b4)}/{len(te_low)}")
    row("sync<200K baseline", stat([r["_net"] for r in te_b4]))
    row("btc4h < 0",          stat([r["_net"] for r in te_b4 if r["_btc4h"]<0]))
    row("btc4h >= 0",         stat([r["_net"] for r in te_b4 if r["_btc4h"]>=0]))
    print()
    for thr in [-100,-50,-20,0,20,50]:
        sub = [r["_net"] for r in te_b4 if r["_btc4h"]<thr]
        row(f"btc4h < {thr:+d}", stat(sub))
    print()
    # btc4h x btc7d
    row("btc4h<0 AND btc7d<0",
        stat([r["_net"] for r in te_low_b7 if (r.get("_btc4h") or 1)<0 and r["_btc7d"]<0]))
    row("btc4h<0 AND btc7d>=0",
        stat([r["_net"] for r in te_low_b7 if (r.get("_btc4h") or 1)<0 and r["_btc7d"]>=0]))

    # ── DG · Cascade spacing ──────────────────────────────────────────────────
    hdr("DG1 · Cascade spacing: time since last cascade (sync<200K TIME_EXIT)")
    # Sort all longs by timestamp, compute gap to prior event
    all_longs_sorted = sorted(longs, key=lambda x: int(x.get("anchor_ts_ms") or 0))
    prev_ts = None
    for r in all_longs_sorted:
        ts = int(r.get("anchor_ts_ms") or 0)
        r["_gap_h"] = (ts - prev_ts) / 3_600_000 if prev_ts else 999
        prev_ts = ts

    te_low_gap = [r for r in te_low if r.get("_gap_h") is not None]
    print(f"  Gap data available: {len(te_low_gap)}/{len(te_low)}")
    row("all (baseline)", stat([r["_net"] for r in te_low_gap]))
    for g in [1,2,4,6,8,12]:
        sub = [r["_net"] for r in te_low_gap if r["_gap_h"] >= g]
        row(f"gap >= {g:2d}h", stat(sub))
    print()
    # Distribution
    print("  Gap distribution:")
    bands = [(0,1),(1,2),(2,4),(4,8),(8,12),(12,24),(24,999)]
    for lo,hi in bands:
        sub = [r["_net"] for r in te_low_gap if lo <= r["_gap_h"] < hi]
        lbl = f"gap {lo}-{hi}h" if hi<999 else f"gap >{lo}h"
        row(lbl, stat(sub))

    # ── DH · SHORT min delay filter ───────────────────────────────────────────
    hdr("DH1 · SHORT minimum BTC confirm delay (score>=4, from DB)")
    s4_d = [r for r in s4 if r.get("_delay") is not None]
    print(f"  Delay loaded: {len(s4_d)}/{len(s4)}")
    row("score>=4 all (baseline)", stat([r["_net"] for r in s4_d]))
    for min_d in [2, 5, 10, 15]:
        sub = [r["_net"] for r in s4_d if r["_delay"] >= min_d]
        row(f"delay >= {min_d} min", stat(sub))
    print()
    # Delay x size
    try:
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro",uri=True,timeout=10) as conn:
            for r in s4_d:
                ts = int(r.get("anchor_ts_ms") or 0)
                res = conn.execute(
                    "SELECT notional FROM liquidations WHERE symbol='BTCUSDT' AND side='SELL' "
                    "AND ts_ms>=? AND ts_ms<=? AND notional>=1000000 ORDER BY ts_ms ASC LIMIT 1",
                    (ts+60_000, ts+30*60_000)
                ).fetchone()
                r["_btc_n_m"] = float(res[0])/1_000_000 if res else None
        s4_dm = [r for r in s4_d if r.get("_btc_n_m") is not None]
        print("  delay x size combos:")
        row("delay>=5 + BTC>=2M",  stat([r["_net"] for r in s4_dm if r["_delay"]>=5  and r["_btc_n_m"]>=2]))
        row("delay>=10 + BTC>=2M", stat([r["_net"] for r in s4_dm if r["_delay"]>=10 and r["_btc_n_m"]>=2]))
        row("delay>=5 + BTC<2M",   stat([r["_net"] for r in s4_dm if r["_delay"]>=5  and r["_btc_n_m"]<2]))
        row("no delay filter + BTC>=2M", stat([r["_net"] for r in s4_dm if r["_btc_n_m"]>=2]))
    except Exception as e:
        print(f"  [DB ERROR] {e}")

    # ── DI · ETH 4h gate frequency analysis ───────────────────────────────────
    hdr("DI1 · ETH 4h gate: monthly frequency impact on sync<200K population")
    te_e4 = [r for r in te_low if r.get("_eth4h") is not None]
    print(f"  ETH4h loaded: {len(te_e4)}/{len(te_low)}")
    m_all: dict[str,list]   = defaultdict(list)
    m_e4neg: dict[str,list] = defaultdict(list)
    m_e4n100: dict[str,list]= defaultdict(list)
    for r in te_e4:
        m_all[r["_month"]].append(r["_net"])
        if (r.get("_eth4h") or 0) < 0:    m_e4neg[r["_month"]].append(r["_net"])
        if (r.get("_eth4h") or 0) < -100: m_e4n100[r["_month"]].append(r["_net"])

    print(f"\n  {'Month':<10} {'All':>5} {'ETH4h<0':>9} {'ETH4h<-100':>12}")
    for m in sorted(m_all):
        print(f"  {m:<10} {len(m_all[m]):>5} "
              f"{len(m_e4neg.get(m,[]))*'':>0}{len(m_e4neg.get(m,[])):>9} "
              f"{len(m_e4n100.get(m,[])):>12}")
    t_all  = sum(len(v) for v in m_all.values())
    t_neg  = sum(len(v) for v in m_e4neg.values())
    t_n100 = sum(len(v) for v in m_e4n100.values())
    nm = len(m_all)
    print(f"\n  ETH4h<0 : {t_neg}/{t_all} events pass ({100*t_neg/max(t_all,1):.0f}%), {t_neg/nm:.1f}/month")
    print(f"  ETH4h<-100: {t_n100}/{t_all} events pass ({100*t_n100/max(t_all,1):.0f}%), {t_n100/nm:.1f}/month")
    print()
    row("ETH4h<0",    stat([r["_net"] for r in te_e4 if (r.get("_eth4h") or 0)<0]))
    row("ETH4h<-50",  stat([r["_net"] for r in te_e4 if (r.get("_eth4h") or 0)<-50]))
    row("ETH4h<-100", stat([r["_net"] for r in te_e4 if (r.get("_eth4h") or 0)<-100]))

    # ── DJ · Final ranking table ───────────────────────────────────────────────
    hdr("DJ1 · Final gate ranking table (all combos, sync<200K TIME_EXIT base)")
    def gate_stat(pop): return stat([r["_net"] for r in pop])

    # Build all combos using pre-loaded features
    combos = [
        ("baseline TIME_EXIT all",          te_all),
        ("sync<200K (live gate)",           te_low),
        ("sync<200K + no US13-14",          [r for r in te_low if not is_early_us(r)]),
        ("sync<200K + btc7d<0",             [r for r in te_low_b7 if r["_btc7d"]<0]),
        ("sync<200K + btc4h<0",             [r for r in te_low if (r.get("_btc4h") or 1)<0]),
        ("sync<200K + ETH4h<0",             [r for r in te_low if (r.get("_eth4h") or 1)<0]),
        ("sync<200K + ETH4h<-50",           [r for r in te_low if (r.get("_eth4h") or 1)<-50]),
        ("sync<200K + n2h>=5",              [r for r in te_low if (r.get("n2h") or 0)>=5]),
        ("sync<200K + !US",                 [r for r in te_low if r.get("session")!="US"]),
        ("sync<200K + btc7d<0 + no US13-14",[r for r in te_low_b7 if r["_btc7d"]<0 and not is_early_us(r)]),
        ("sync<200K + btc7d<0 + ETH4h<0",  [r for r in te_low_b7 if r["_btc7d"]<0 and (r.get("_eth4h") or 1)<0]),
        ("sync<200K + btc7d<0 + n2h>=5",   [r for r in te_low_b7 if r["_btc7d"]<0 and (r.get("n2h") or 0)>=5]),
        ("sync<200K + !US + btc7d<0",      [r for r in te_low_b7 if r.get("session")!="US" and r["_btc7d"]<0]),
        ("sync<200K + !US + n2h>=5",       [r for r in te_low if r.get("session")!="US" and (r.get("n2h") or 0)>=5]),
        ("sync<200K+btc7d<0+no13-14+n2h>=5",[r for r in te_low_b7 if r["_btc7d"]<0 and not is_early_us(r) and (r.get("n2h") or 0)>=5]),
    ]
    nm_data = len(set(r["_month"] for r in te_low))
    print(f"  {'Gate':<42} {'N':>4}  {'WR':>7}  {'avg':>8}  {'N/mo':>5}")
    for label, pop in combos:
        s = gate_stat(pop)
        nmo = s["n"] / max(nm_data,1)
        if s["n"] == 0:
            print(f"  {label:<42}  N=  0  -----  ------  -----"); continue
        print(f"  {label:<42}  N={s['n']:4d}  WR={pct(s['wr'])}  "
              f"avg={fmt(s['avg'])} bps  {nmo:.1f}/mo")

    hdr("DJ2 · SHORT gate ranking")
    short_combos = [
        ("score>=3 (historical)",   [r for r in shorts]),
        ("score>=4 (live gate)",    s4),
        ("score>=4 + BTC>=2M",      [r for r in s4 if (r.get("_btc_n_m") or 0)>=2]),
        ("score>=4 + delay>=5min",  [r for r in s4 if (r.get("_delay") or 0)>=5]),
        ("score>=4 + delay>=10min", [r for r in s4 if (r.get("_delay") or 0)>=10]),
        ("score>=4 + btc4h<0",      [r for r in s4 if (r.get("_btc4h") or 1)<0]),
        ("score>=4 + btc7d<0",      [r for r in s4 if (r.get("_btc7d") or 1)<0]),
        ("score>=4 + BTC>=2M + delay>=5", [r for r in s4 if (r.get("_btc_n_m") or 0)>=2 and (r.get("_delay") or 0)>=5]),
    ]
    nm_s = len(set(r["_month"] for r in s4))
    for label, pop in short_combos:
        s = gate_stat(pop)
        nmo = s["n"] / max(nm_s, 1)
        if s["n"] == 0:
            print(f"  {label:<42}  N=  0  -----  ------  -----"); continue
        print(f"  {label:<42}  N={s['n']:4d}  WR={pct(s['wr'])}  "
              f"avg={fmt(s['avg'])} bps  {nmo:.1f}/mo")

    print()
    print("="*70)
    print("  ALL ROUND-8 TESTS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
