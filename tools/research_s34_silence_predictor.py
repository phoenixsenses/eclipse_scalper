"""S34 T0-Knowable Hold Predictor.

Bulgu (önceki gauntlet): cascade → LONG 6h TUT (erken çıkış yok) gerçek edge ama
tüm cascade'lere girmek tail-ağır ve no-overlap'ta anlamsız. "silence" T+30'da
bilinir (lookahead). SORU: T0'da (cascade anında) bilinebilir bir feature ile
hangi cascade'in 6h tutunca KAZANACAĞINI tahmin edebilir miyiz?

Başarırsa: yüksek frekans + yüksek WR + kontrollü tail = aranan canlı aday.

Disiplin:
  - Hedef: hold-6h profit (erken çıkış YOK, tradeable).
  - SADECE T0-knowable feature (gelecek bilgisi yok).
  - Kronolojik 70/30 holdout: eşik TRAIN'de seçilir, TEST'te raporlanır.
  - Basit 1-2 feature (overfit yasak). MC + no-overlap + cost final.

Bölümler:
  S1  Feature screening — her feature tercile, TRAIN-best bin, TEST lift
  S2  Tek-feature filtre — en iyi feature gate, holdout + no-overlap
  S3  2-feature combo — en iyi ikililer
  S4  Silence-prediction — feature'lar silence'ı öngörüyor mu (precision/recall)
  S5  FINAL — en iyi predictor: freq/WR/TOTAL/tail/holdout/no-overlap/cost/MC/verdict

Çıktı:
  reports/research/s34/S34_SILENCE_PREDICTOR.json
  reports/research/s34/S34_SILENCE_PREDICTOR.md
"""
from __future__ import annotations

import bisect
import json
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations, load_mark_index, reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_SILENCE_PREDICTOR.json"
OUT_MD   = OUT_DIR / "S34_SILENCE_PREDICTOR.md"

PROP_THRESH =  50_000.0
LOOKBACK_MS = 400 * 24 * 3600_000
FEE_BPS     = 5.0
MC_ITER     = 1000
HOLD_MS     = 6 * 3600_000
TOTAL_MONTHS = 4.5
THRESHOLDS  = [150_000, 200_000]
PRIMARY_THR = 150_000
TRAIN_FRAC  = 0.70

random.seed(42)

# --- DB helpers ------------------------------------------------------------

def _scalar(conn, sql, p=()):
    r = conn.execute(sql, p).fetchone()
    return float(r[0]) if r and r[0] is not None else 0.0

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn, "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<? AND notional>=?", (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? "
        "AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn, "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? "
        "AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    r = conn.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(r[0]) if r else None

def mark_bps(conn, sym, ts, lb):
    r0 = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts-lb)).fetchone()
    r1 = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0])-float(r0[0]))/float(r0[0])*10_000.0
    return 0.0

def ofi_pre(conn, sym, lo, hi):
    r = conn.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),"
        "SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END) FROM agg_trades "
        "WHERE symbol=? AND ts_ms>=? AND ts_ms<?", (sym, lo, hi)).fetchone()
    if not r or r[0] is None: return None
    b, s = float(r[0]), float(r[1]); t = b+s
    return (b-s)/t if t > 0 else 0.0

def session_name(ts):
    h = datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour
    return "EUROPE" if 7<=h<13 else ("US" if 13<=h<21 else "OFF")
def dow_of(ts):  return datetime.fromtimestamp(ts/1000, tz=timezone.utc).weekday()
def hour_of(ts): return datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour

def load_vol(conn):
    rows = conn.execute("SELECT ts_ms, rv_5m, vol_decile FROM vol_state WHERE symbol='ETHUSDT' "
        "ORDER BY ts_ms").fetchall()
    return [r[0] for r in rows], rows

def long_gross(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts); r1 = marks.at_or_before(ts+hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0: return None
    return (float(r1[1])-float(r0[1]))/float(r0[1])*10_000.0

# --- stats -----------------------------------------------------------------

def _mc_p(vals, avg):
    if len(vals) < 4: return None
    rng = random.Random(0)
    ct = sum(1 for _ in range(MC_ITER)
             if sum(rng.choice([-1,1])*abs(v) for v in vals)/len(vals) >= avg)
    return round(ct/MC_ITER, 3)

def wf_folds(vals, k=5):
    n = len(vals)
    if n < k: return None
    return "%d/%d" % (sum(1 for i in range(k) if sum(vals[i*n//k:(i+1)*n//k])>0), k)

def stats(gvals, label="", months=None, fee=FEE_BPS):
    m = months or TOTAL_MONTHS
    if not gvals: return {"label": label, "n": 0}
    net = [v-fee for v in gvals]; n = len(net); wins = sum(1 for v in net if v>0)
    sv = sorted(net); avg = sum(net)/n
    cum=0; peak=0; mdd=0
    for v in net: cum+=v; peak=max(peak,cum); mdd=min(mdd,cum-peak)
    cut = max(1, int(n*0.7)); ho = net[cut:]
    return {"label": label, "n": n, "wr": round(100*wins/n,1), "avg": round(avg,1),
            "total": round(sum(net),0), "per_month": round(n/m,1),
            "pnl_per_month": round(sum(net)/m,0), "worst": round(sv[0],1),
            "tail_n": sum(1 for v in net if v<-100), "mdd": round(mdd,0),
            "mc_p": _mc_p(net, avg), "wf": wf_folds(net),
            "ho_avg": round(sum(ho)/len(ho),1) if ho else None,
            "ho_wr": round(100*sum(1 for v in ho if v>0)/len(ho),1) if ho else None}

def no_overlap(pairs, hold_ms):
    busy=-1; out=[]
    for ts,v in sorted(pairs):
        if ts>=busy: out.append(v); busy=ts+hold_ms
    return out

def pstat(k, v):
    if not v or v.get("n",0)==0: print("    %-34s N=0" % k[:34]); return
    print("    %-34s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s tail=%-2d mc_p=%s wf=%s" % (
        k[:34], v["n"], v.get("per_month",0), str(v["wr"])+"%", str(v["avg"])+"bps",
        str(v.get("total")), v.get("tail_n",0), v.get("mc_p","?"), v.get("wf")))

# --- build -----------------------------------------------------------------

FEATS = ["rn","dominance","accel","liq_count","prebuildup","n2h","density24",
         "sync_k","btc_conc_pre","be_ratio_pre","btc5m","btc4h","btc7d","btc3d",
         "eth_drop5m","vdepth","ofi_pre","rv5m","vol_decile","hour"]

def build(conn, marks, vol_ts, vol_rows, thr, now, start):
    liqs = load_liquidations(conn,"ETHUSDT","SELL",start,now)
    ancs = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                               thresholds=(float(thr),), accel_window_sec=30)
    evs = []
    print("  thr=%dK anchors=%d" % (thr//1000, len(ancs)))
    for anc in ancs:
        ts = int(anc.anchor_ts_ms); rn = float(anc.running_notional)
        if rn < thr or marks.at_or_after(ts) is None: continue
        btc4h = mark_bps(conn,"BTCUSDT",ts,4*3600_000)
        eth1h = mark_bps(conn,"ETHUSDT",ts,3600_000)
        bull  = eth1h>20 and btc4h>50
        sess  = session_name(ts)
        if bull or sess=="EUROPE": continue   # temel evren (not bull, not EU)
        btc7d = mark_bps(conn,"BTCUSDT",ts,7*24*3600_000)
        if not (btc4h<0 or btc7d<0): continue  # regime (temel filtre — bilinir)
        sync_k = (liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
                + liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts))
        btc_conc_pre = liq_max(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
        book = book_features_at(conn,"ETHUSDT",ts,30) or {}
        vi = bisect.bisect_right(vol_ts, ts)-1
        rv5m = float(vol_rows[vi][1]) if vi>=0 and vol_rows[vi][1] is not None else None
        vd   = int(vol_rows[vi][2]) if vi>=0 and vol_rows[vi][2] is not None else None
        noisy_ts = liq_first_ts(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP_THRESH)
        ev = {
            "ts": ts,
            "rn": rn,
            "dominance": float(anc.running_single_liq_dominance),
            "accel": float(anc.running_accel),
            "liq_count": float(anc.running_liq_count),
            "prebuildup": float(liq_cnt(conn,"ETHUSDT","SELL",ts-30*60_000,ts-1000,PROP_THRESH)),
            "n2h": float(liq_cnt(conn,"ETHUSDT","SELL",ts-2*3600_000,ts-1000,PROP_THRESH)),
            "density24": float(liq_cnt(conn,"ETHUSDT","SELL",ts-24*3600_000,ts-300_000,thr)),
            "sync_k": sync_k,
            "btc_conc_pre": btc_conc_pre,
            "be_ratio_pre": (btc_conc_pre/rn) if rn>0 else 0.0,
            "btc5m": mark_bps(conn,"BTCUSDT",ts,5*60_000),
            "btc4h": btc4h, "btc7d": btc7d,
            "btc3d": mark_bps(conn,"BTCUSDT",ts,3*24*3600_000),
            "eth_drop5m": mark_bps(conn,"ETHUSDT",ts,5*60_000),
            "vdepth": float(book.get("vdepth_bps") or 0.0),
            "ofi_pre": ofi_pre(conn,"ETHUSDT",ts-5*60_000,ts),
            "rv5m": rv5m, "vol_decile": vd,
            "hour": float(hour_of(ts)),
            "sess": sess, "dow": dow_of(ts),
            "silent": noisy_ts is None,
        }
        ev["l6h"] = long_gross(marks, ts, HOLD_MS)
        if ev["l6h"] is None: continue
        # ofi_pre / rv5m None olabilir → 0'a çek (feature taraması için)
        if ev["ofi_pre"] is None: ev["ofi_pre"] = 0.0
        if ev["rv5m"] is None: ev["rv5m"] = 0.0
        if ev["vol_decile"] is None: ev["vol_decile"] = -1
        evs.append(ev)
    evs.sort(key=lambda e: e["ts"])
    return evs

def split(evs):
    cut = int(len(evs)*TRAIN_FRAC)
    return evs[:cut], evs[cut:]

# --- S1 feature screening --------------------------------------------------

def terciles(vals):
    s = sorted(vals); n = len(s)
    return s[n//3], s[2*n//3]

def run_S1(train, test, months):
    print("\n=== S1: Feature screening (TRAIN-best tercile → TEST) ===")
    R = {}; ranking = []
    for f in FEATS:
        tv = [ev[f] for ev in train if ev.get(f) is not None]
        if len(set(tv)) < 3: continue
        q1, q2 = terciles(tv)
        # bins
        def binof(x):
            if x < q1: return "lo"
            if x < q2: return "mid"
            return "hi"
        # TRAIN bin avgs (net) → en iyi bin
        tr_bins = {"lo": [], "mid": [], "hi": []}
        for ev in train: tr_bins[binof(ev[f])].append(ev["l6h"]-FEE_BPS)
        best_bin = max(tr_bins, key=lambda b: (sum(tr_bins[b])/len(tr_bins[b])) if tr_bins[b] else -1e9)
        # TEST stats of TRAIN-best bin
        te_best = [ev["l6h"] for ev in test if binof(ev[f]) == best_bin]
        te_all  = [ev["l6h"] for ev in test]
        s_best = stats(te_best, "%s bin=%s (TRAIN-best) TEST" % (f, best_bin), months)
        s_all  = stats(te_all, "TEST all", months)
        lift = (s_best.get("avg") or 0) - (s_all.get("avg") or 0)
        R["S1_%s" % f] = {"best_bin": best_bin, "q1": round(q1,4), "q2": round(q2,4),
                          "test_best": s_best, "test_all_avg": s_all.get("avg"),
                          "lift": round(lift,1)}
        ranking.append((f, best_bin, lift, s_best))
        print("    %-14s TRAIN-best=%-4s TEST: N=%-3d WR=%-6s avg=%-8s lift=%+.1f mc_p=%s" % (
            f, best_bin, s_best.get("n",0), str(s_best.get("wr"))+"%",
            str(s_best.get("avg"))+"bps", lift, s_best.get("mc_p")))
    ranking.sort(key=lambda x: -x[2])
    print("  --- En iyi lift (OOS): ---")
    for f, b, lift, s in ranking[:6]:
        print("    %-14s bin=%-4s lift=%+.1f  (TEST WR=%s avg=%s N=%d)" % (
            f, b, lift, str(s.get("wr"))+"%", str(s.get("avg")), s.get("n",0)))
    R["_ranking"] = [(f,b,lift) for f,b,lift,_ in ranking]
    return R, ranking

# --- S2 single-feature filter (full period + holdout) ----------------------

def bin_gate(f, best_bin, q1, q2):
    if best_bin == "lo":  return lambda ev: ev[f] < q1
    if best_bin == "hi":  return lambda ev: ev[f] >= q2
    return lambda ev: q1 <= ev[f] < q2

def run_S2(evs, train, test, ranking, months):
    print("\n=== S2: Tek-feature filtre (holdout + no-overlap) ===")
    R = {}
    for f, best_bin, lift, _ in ranking[:5]:
        tv = [ev[f] for ev in train if ev.get(f) is not None]
        q1, q2 = terciles(tv)
        g = bin_gate(f, best_bin, q1, q2)
        # TEST (OOS)
        te = [ev["l6h"] for ev in test if g(ev)]
        R["S2_%s_TEST" % f] = stats(te, "%s %s TEST" % (f, best_bin), months*(1-TRAIN_FRAC))
        # full period (referans)
        full = [ev["l6h"] for ev in evs if g(ev)]
        R["S2_%s_FULL" % f] = stats(full, "%s %s FULL" % (f, best_bin), months)
        # no-overlap full
        nov = no_overlap([(ev["ts"], ev["l6h"]) for ev in evs if g(ev)], HOLD_MS)
        s = stats(nov, "%s %s no-overlap" % (f, best_bin), months)
        s["per_month"] = round(len(nov)/months, 1)
        R["S2_%s_NOOV" % f] = s
        pstat("S2_%s_TEST" % f, R["S2_%s_TEST" % f])
        pstat("S2_%s_FULL" % f, R["S2_%s_FULL" % f])
        pstat("S2_%s_NOOV" % f, s)
    return R

# --- S3 two-feature combos -------------------------------------------------

def run_S3(evs, train, test, ranking, months):
    print("\n=== S3: 2-feature combo (top feature'lardan) ===")
    R = {}
    top = ranking[:5]
    cuts = {}
    for f, bb, _, _ in top:
        tv = [ev[f] for ev in train if ev.get(f) is not None]
        cuts[f] = (terciles(tv), bb)
    import itertools
    for (f1,b1,_,_),(f2,b2,_,_) in itertools.combinations(top, 2):
        (q1a,q2a),bb1 = cuts[f1]; (q1b,q2b),bb2 = cuts[f2]
        g1 = bin_gate(f1, bb1, q1a, q2a); g2 = bin_gate(f2, bb2, q1b, q2b)
        gate = lambda ev, a=g1, b=g2: a(ev) and b(ev)
        te   = [ev["l6h"] for ev in test if gate(ev)]
        full = [ev["l6h"] for ev in evs if gate(ev)]
        nov  = no_overlap([(ev["ts"], ev["l6h"]) for ev in evs if gate(ev)], HOLD_MS)
        s_full = stats(full, "%s&%s FULL" % (f1,f2), months)
        s_test = stats(te, "%s&%s TEST" % (f1,f2), months*(1-TRAIN_FRAC))
        s_nov  = stats(nov, "%s&%s noov" % (f1,f2), months); s_nov["per_month"]=round(len(nov)/months,1)
        R["S3_%s_%s_FULL" % (f1,f2)] = s_full
        R["S3_%s_%s_TEST" % (f1,f2)] = s_test
        R["S3_%s_%s_NOOV" % (f1,f2)] = s_nov
        if s_full.get("n",0) >= 15:
            print("    %-24s FULL N=%-3d WR=%-6s avg=%-7s | TEST N=%-2d avg=%-7s | noov /mo=%.1f" % (
                ("%s&%s"%(f1,f2))[:24], s_full["n"], str(s_full["wr"])+"%", str(s_full["avg"]),
                s_test.get("n",0), str(s_test.get("avg")), s_nov.get("per_month",0)))
    return R

# --- S4 silence prediction -------------------------------------------------

def run_S4(train, test, ranking, months):
    print("\n=== S4: Feature'lar SILENCE'ı öngörüyor mu? ===")
    R = {}
    base_sil = sum(1 for ev in test if ev["silent"]) / max(1, len(test))
    print("  TEST silence base-rate: %.1f%%" % (100*base_sil))
    R["_base_silence"] = round(100*base_sil, 1)
    for f, best_bin, _, _ in ranking[:6]:
        tv = [ev[f] for ev in train if ev.get(f) is not None]
        q1, q2 = terciles(tv)
        g = bin_gate(f, best_bin, q1, q2)
        sub = [ev for ev in test if g(ev)]
        if not sub: continue
        prec = sum(1 for ev in sub if ev["silent"]) / len(sub)
        R["S4_%s" % f] = {"n": len(sub), "silence_rate": round(100*prec,1),
                          "lift_vs_base": round(100*(prec-base_sil),1)}
        print("    %-14s bin=%-4s → silence-rate=%.1f%% (base %.1f%%, lift %+.1f)" % (
            f, best_bin, 100*prec, 100*base_sil, 100*(prec-base_sil)))
    return R

# --- S5 FINAL --------------------------------------------------------------

def run_S5(ev_by_thr, marks, best_gate_builder, months):
    print("\n=== S5: FINAL predictor scorecard ===")
    R = {}
    for thr in THRESHOLDS:
        evs = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        train, test = split(evs)
        gate, desc = best_gate_builder(train)
        full = [ev["l6h"] for ev in evs if gate(ev)]
        te   = [ev["l6h"] for ev in test if gate(ev)]
        base = [ev["l6h"] for ev in evs]
        R["S5_%s_base_all" % lbl] = stats(base, "%s tüm cascade (baz)" % lbl, months)
        R["S5_%s_pred_FULL" % lbl] = stats(full, "%s predictor FULL [%s]" % (lbl, desc), months)
        R["S5_%s_pred_TEST" % lbl] = stats(te, "%s predictor TEST-OOS" % lbl, months*(1-TRAIN_FRAC))
        nov = no_overlap([(ev["ts"], ev["l6h"]) for ev in evs if gate(ev)], HOLD_MS)
        s = stats(nov, "%s predictor no-overlap" % lbl, months); s["per_month"]=round(len(nov)/months,1)
        s["net10_pnl_mo"] = round(sum(x-10 for x in nov)/months,0) if nov else 0
        R["S5_%s_pred_NOOV" % lbl] = s
        pstat("S5_%s_base_all" % lbl, R["S5_%s_base_all" % lbl])
        pstat("S5_%s_pred_FULL" % lbl, R["S5_%s_pred_FULL" % lbl])
        pstat("S5_%s_pred_TEST" % lbl, R["S5_%s_pred_TEST" % lbl])
        pstat("S5_%s_pred_NOOV" % lbl, s)
        print("        desc=%s  no-ov /ay=%.1f pnl/mo@10=%s" % (desc, s.get("per_month",0), str(s.get("net10_pnl_mo"))))
    return R

# --- S6 FINAL config lock (hour=hi + tail mgmt + combos) -------------------

def hold_gross_stop(marks, ts, hold_ms, stop_bps):
    r0 = marks.at_or_after(ts)
    if not r0 or float(r0[1]) <= 0: return None
    entry = float(r0[1]); stop_px = entry*(1-stop_bps/10_000.0)
    for _, px in marks.slice_range(r0[0], ts+hold_ms):
        if float(px) <= stop_px: return -float(stop_bps)
    r1 = marks.at_or_before(ts+hold_ms)
    if not r1: return None
    return (float(r1[1])-entry)/entry*10_000.0

def run_S6(ev_by_thr, marks, months):
    print("\n=== S6: FINAL config lock (hour=hi + tail mgmt) ===")
    R = {}
    evs = ev_by_thr[PRIMARY_THR]
    train, _ = split(evs)
    tv = [ev["hour"] for ev in train]
    q1, q2 = terciles(tv)
    hours_hi = sorted(set(int(ev["hour"]) for ev in evs if ev["hour"] >= q2))
    print("  hour=hi eşiği: hour >= %.0f  → saatler: %s" % (q2, hours_hi))
    R["_hours_hi"] = hours_hi
    ghi = lambda ev: ev["hour"] >= q2

    for thr in THRESHOLDS:
        e = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        sub = [ev for ev in e if ghi(ev)]
        # stop sweep no-overlap
        for sb in [None, 200, 150]:
            pairs = []
            for ev in sub:
                gg = ev["l6h"] if sb is None else hold_gross_stop(marks, ev["ts"], HOLD_MS, float(sb))
                if gg is not None: pairs.append((ev["ts"], gg))
            nov = no_overlap(pairs, HOLD_MS)
            s = stats(nov, "%s hour=hi stop=%s noov" % (lbl, sb or "none"), months)
            s["per_month"] = round(len(nov)/months, 1)
            s["net10_pnl_mo"] = round(sum(x-10 for x in nov)/months, 0) if nov else 0
            R["S6_%s_stop%s_noov" % (lbl, sb or "none")] = s
            pstat("S6_%s_stop%s_noov" % (lbl, sb or "none"), s)
            print("        /ay=%.1f pnl/mo@10=%s mdd=%s" % (
                s.get("per_month",0), str(s.get("net10_pnl_mo")), str(s.get("mdd"))))
        # full raw (fee5) for reference
        R["S6_%s_full" % lbl] = stats([ev["l6h"] for ev in sub], "%s hour=hi FULL" % lbl, months)
        pstat("S6_%s_full" % lbl, R["S6_%s_full" % lbl])

    # combo: hour=hi + sync_k mid (ikisi de OOS-validated) — küçük N uyarısı
    tvs = [ev["sync_k"] for ev in train]
    s1, s2 = terciles(tvs)
    gsync = lambda ev: s1 <= ev["sync_k"] < s2
    sub2 = [ev for ev in evs if ghi(ev) and gsync(ev)]
    nov2 = no_overlap([(ev["ts"], ev["l6h"]) for ev in sub2], HOLD_MS)
    s = stats(nov2, "hour=hi & sync_k=mid noov", months); s["per_month"]=round(len(nov2)/months,1)
    R["S6_combo_hour_sync_noov"] = s
    pstat("S6_combo_hour_sync_noov", s)
    return R

# --- Markdown --------------------------------------------------------------

def _row(k, v):
    if not isinstance(v, dict) or v.get("n",0)==0:
        return None
    return "| %s | %d | %.1f | %.1f%% | %+.1f | %s | %d | %s | %s |" % (
        k, v["n"], v.get("per_month",0), v["wr"], v["avg"], v.get("total"),
        v.get("tail_n",0), v.get("mc_p","?"), v.get("wf","-"))

def make_md(sections, meta, ranking):
    L = ["# S34 T0-Knowable Hold Predictor", "",
         "> Hedef: T0'da bilinen feature ile hangi cascade'in 6h tutunca kazanacağını tahmin.",
         "> Erken çıkış YOK. Kronolojik 70/30 holdout (eşik TRAIN, rapor TEST).",
         "> Evren: not bull + not EU + regime(btc4h<0 OR btc7d<0). %.1f ay, FEE=%dbps." % (
             meta["months"], int(FEE_BPS)),
         "> Tarih: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"), ""]
    L += ["## Feature Lift Sıralaması (OOS)", "", "| Feature | best-bin | OOS lift (bps) |", "|---|---|---:|"]
    for f, b, lift in ranking[:12]:
        L.append("| %s | %s | %+.1f |" % (f, b, lift))
    L.append("")
    hdr = "| Konfig | N | /ay | WR | Avg | TOTAL | Tail | mc_p | WF |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    titles = {"S2":"S2) Tek-Feature Filtre","S3":"S3) 2-Feature Combo","S5":"S5) FINAL Predictor"}
    for sec in ["S2","S3","S5"]:
        Rs = sections.get(sec, {})
        L += ["## %s" % titles[sec], "", hdr, sep]
        for k, v in Rs.items():
            row = _row(k, v)
            if row: L.append(row)
        L.append("")
    L += ["---","*Script: tools/research_s34_silence_predictor.py*"]
    return "\n".join(L)

# --- main ------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 T0 Hold Predictor ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp()*1000); start = now - LOOKBACK_MS
        marks = load_mark_index(conn, "ETHUSDT")
        vol_ts, vol_rows = load_vol(conn)
        ev_by_thr = {}
        for thr in THRESHOLDS:
            print("Building thr=%dK ..." % (thr//1000))
            ev_by_thr[thr] = build(conn, marks, vol_ts, vol_rows, thr, now, start)
            print("  usable events=%d" % len(ev_by_thr[thr]))

        evs = ev_by_thr[PRIMARY_THR]
        span = [e["ts"] for e in evs]
        TOTAL_MONTHS = max(1.0, (span[-1]-span[0])/86_400_000/30.0); months = TOTAL_MONTHS
        print("  %.2f ay, primary events=%d" % (months, len(evs)))
        train, test = split(evs)
        print("  train=%d test=%d" % (len(train), len(test)))

        sections = {}
        s1, ranking = run_S1(train, test, months); sections["S1"] = s1
        sections["S2"] = run_S2(evs, train, test, ranking, months)
        sections["S3"] = run_S3(evs, train, test, ranking, months)
        sections["S4"] = run_S4(train, test, ranking, months)

        # best gate builder: en yüksek OOS lift tek feature (+ ikinci feature combine)
        def best_gate_builder(train_evs):
            tv = [ev[ranking[0][0]] for ev in train_evs if ev.get(ranking[0][0]) is not None]
            q1, q2 = terciles(tv)
            f, bb = ranking[0][0], ranking[0][1]
            return bin_gate(f, bb, q1, q2), "%s=%s" % (f, bb)
        sections["S5"] = run_S5(ev_by_thr, marks, best_gate_builder, months)
        sections["S6"] = run_S6(ev_by_thr, marks, months)

    meta = {"months": round(months,2),
            "n_by_thr": {("%dK"%(t//1000)): len(ev_by_thr[t]) for t in THRESHOLDS}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON,"w",encoding="utf-8") as f:
        json.dump({"sections": sections, "meta": meta, "ranking": ranking[:12] if ranking else []},
                  f, indent=2, default=str)
    with open(OUT_MD,"w",encoding="utf-8") as f:
        f.write(make_md(sections, meta, [(a,b,c) for a,b,c,_ in ranking] if ranking else []))
    print("\nJSON: %s\nMD:   %s\nDone." % (OUT_JSON, OUT_MD))


if __name__ == "__main__":
    main()
