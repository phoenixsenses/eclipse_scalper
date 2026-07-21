"""S34 Silence-Core FINAL Gauntlet + Falsification.

Ana alfa (attribution ile bulundu): SILENCE LONG = ETH SELL cascade + ilk 30dk
follow-on likidasyon YOK → LONG. Bu script onu canlıya hazır hâle getirir ve
CIDDI şekilde çürütmeye çalışır.

KRİTİK realizm sorunu: "silence" ancak T+30dk sonra bilinir → T0'da silence'a
bahis yapmak lookahead. Gerçek tradeable mekanizma = provisional giriş + noisy
gelince erken çıkış (canlının zaten yaptığı). Bunu modelliyoruz.

Bölümler:
  R  REALISM     — ideal-silence (lookahead) vs provisional-early-exit (gerçek)
                    vs confirm-T30 vs T15-bounce. Eşik 100/150/200K.
  T  TAIL MGMT   — stop {none,200,150,100} × veto {be<2, notUS1314, notSat, vol}
  N  NO-OVERLAP  — tek-pozisyon gerçekçi frekans + TOTAL
  C  COST        — fee 5/8/10/15 + slippage; nerede ölür
  F  FALSIFY     — random-entry kontrolü (downtrend beta), silence-window
                    duyarlılığı, zaman-split, tek-ay-çıkar, top3-removed,
                    regime-gerekli mi
  Z  FINAL       — önerilen config scorecard + verdict

Çıktı:
  reports/research/s34/S34_SILENCE_CORE_FINAL.json
  reports/research/s34/S34_SILENCE_CORE_FINAL.md
"""
from __future__ import annotations

import bisect
import json
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_SILENCE_CORE_FINAL.json"
OUT_MD   = OUT_DIR / "S34_SILENCE_CORE_FINAL.md"

PROP_THRESH =  50_000.0
LOOKBACK_MS = 400 * 24 * 3600_000
FEE_BPS     = 5.0
MC_ITER     = 1000
TOTAL_MONTHS = 4.5
THRESHOLDS  = [100_000, 150_000, 200_000]
PRIMARY_THR = 150_000   # denge noktası (frekans + robustluk)

random.seed(42)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<? AND notional>=?", (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? "
        "AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1",
        (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def mark_bps(conn, sym, ts_ms, lookback_ms):
    r0 = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms - lookback_ms)).fetchone()
    r1 = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0]) - float(r0[0])) / float(r0[0]) * 10_000.0
    return 0.0

def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= h < 13: return "EUROPE"
    if 13 <= h < 21: return "US"
    return "OFF"

def dow_of(ts_ms):  return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).weekday()
def hour_of(ts_ms): return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).hour
def month_of(ts_ms):
    d = datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc); return "%04d-%02d" % (d.year, d.month)

def compute_score(conn, ts, sync_k, n2h):
    b4h  = mark_bps(conn, "BTCUSDT", ts, 4*3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    return (int(n2h>=3)+int(b4h<0)+int(vdep>=30)+int(13<=hour<21)+int(sync_k>=200_000))

def load_vol_ts(conn):
    rows = conn.execute("SELECT ts_ms, vol_decile FROM vol_state WHERE symbol='ETHUSDT' "
                        "ORDER BY ts_ms").fetchall()
    return [r[0] for r in rows], rows

# ---------------------------------------------------------------------------
# Outcomes (gross, fee ayrı)
# ---------------------------------------------------------------------------

def long_gross(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts); r1 = marks.at_or_before(ts+hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0: return None
    return (float(r1[1])-float(r0[1]))/float(r0[1])*10_000.0

def provisional_gross(marks, ts, noisy_ts, hold_ms, stop_bps=None):
    """Enter T0; noisy 30dk içinde gelirse noisy_ts'de erken çık; yoksa hold. Opsiyonel stop."""
    r0 = marks.at_or_after(ts)
    if not r0 or float(r0[1]) <= 0: return None
    entry = float(r0[1])
    exit_ts = ts + hold_ms
    if noisy_ts is not None and noisy_ts < exit_ts:
        exit_ts = noisy_ts
    if stop_bps is not None:
        stop_px = entry*(1.0-stop_bps/10_000.0)
        for _, px in marks.slice_range(r0[0], exit_ts):
            if float(px) <= stop_px: return -float(stop_bps)
    r1 = marks.at_or_before(exit_ts)
    if not r1: return None
    return (float(r1[1])-entry)/entry*10_000.0

def confirm_gross(marks, ts, hold_ms):
    """Silence T+30'da confirm → T+30'da gir, hold_ms tut (T+30'dan)."""
    return long_gross(marks, ts + 30*60_000, hold_ms)

def t15_bounce_gross(marks, ts, hold_ms):
    """T+15'te gir eğer mark(T+15)>=mark(T0); exit T0+hold."""
    r0 = marks.at_or_after(ts)
    m15 = marks.at_or_before(ts + 15*60_000)
    if not r0 or not m15 or float(r0[1]) <= 0: return None
    if float(m15[1]) < float(r0[1]): return None
    return long_gross(marks, ts + 15*60_000, hold_ms - 15*60_000)

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

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

def stats(gross_vals, label="", months=None, fee=FEE_BPS):
    m = months or TOTAL_MONTHS
    if not gross_vals: return {"label": label, "n": 0}
    net = [v-fee for v in gross_vals]
    n = len(net); wins = sum(1 for v in net if v>0); sv = sorted(net)
    avg = sum(net)/n; total = sum(net)
    cut = max(1, int(n*0.7)); ho = net[cut:]
    # max drawdown & loss streak (kronolojik)
    cum = 0.0; peak = 0.0; mdd = 0.0; streak = 0; wstreak = 0
    for v in net:
        cum += v; peak = max(peak, cum); mdd = min(mdd, cum-peak)
        if v < 0: streak += 1; wstreak = max(wstreak, streak)
        else: streak = 0
    return {
        "label": label, "n": n,
        "wr": round(100*wins/n, 1), "avg": round(avg, 1),
        "total": round(total, 0), "per_month": round(n/m, 1),
        "pnl_per_month": round(total/m, 0),
        "worst": round(sv[0], 1), "tail_n": sum(1 for v in net if v<-100),
        "mdd": round(mdd, 0), "loss_streak": wstreak,
        "mc_p": _mc_p(net, avg), "wf": wf_folds(net),
        "ho_n": len(ho),
        "ho_avg": round(sum(ho)/len(ho), 1) if ho else None,
        "ho_wr": round(100*sum(1 for v in ho if v>0)/len(ho), 1) if ho else None,
    }

def no_overlap(pairs, hold_ms):
    busy = -1; out = []
    for ts, val in sorted(pairs):
        if ts >= busy: out.append(val); busy = ts + hold_ms
    return out

def pstat(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-36s N=0" % k[:36]); return
    print("    %-36s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-7s tail=%-2d mdd=%-7s mc_p=%s wf=%s" % (
        k[:36], v["n"], v.get("per_month",0), str(v["wr"])+"%", str(v["avg"])+"bps",
        str(v.get("total")), v.get("tail_n",0), str(v.get("mdd")), v.get("mc_p","?"), v.get("wf")))

# ---------------------------------------------------------------------------
# Build events per threshold
# ---------------------------------------------------------------------------

def build_events(conn, marks, vol_ts, vol_rows, thr, now_ms, start_ms):
    liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
    ancs = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                               thresholds=(float(thr),), accel_window_sec=30)
    ts_list = sorted(int(a.anchor_ts_ms) for a in ancs)
    events = []
    print("  thr=%dK: %d anchors" % (thr//1000, len(ancs)))
    for i, anc in enumerate(ancs):
        ts = int(anc.anchor_ts_ms); rn = float(anc.running_notional)
        if rn < thr or marks.at_or_after(ts) is None:
            continue
        sync_k = (liq_sum(conn,"BTCUSDT","SELL",ts-10*60_000,ts)
                + liq_sum(conn,"SOLUSDT","SELL",ts-10*60_000,ts))
        n2h  = liq_cnt(conn,"ETHUSDT","SELL",ts-2*3600_000,ts-1000,PROP_THRESH)
        score= compute_score(conn, ts, sync_k, n2h)
        btc4h= mark_bps(conn,"BTCUSDT",ts,4*3600_000)
        btc7d= mark_bps(conn,"BTCUSDT",ts,7*24*3600_000)
        eth1h= mark_bps(conn,"ETHUSDT",ts,3600_000)
        bull = eth1h>20 and btc4h>50
        sess = session_name(ts); dow = dow_of(ts); hour = hour_of(ts)
        blocked = (sess=="US" and hour in {13,14})
        noisy_ts = liq_first_ts(conn,"ETHUSDT","SELL",ts+60_000,ts+30*60_000,PROP_THRESH)
        noisy = noisy_ts is not None
        btc_conc = liq_max(conn,"BTCUSDT","SELL",ts-10*60_000,ts+10*60_000)
        be_ratio = (btc_conc/rn) if rn>0 else 0.0
        vi = bisect.bisect_right(vol_ts, ts)-1
        vd = int(vol_rows[vi][1]) if vi>=0 and vol_rows[vi][1] is not None else None
        ev = {
            "ts": ts, "rn": rn, "sync_k": sync_k, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "bull": bull, "sess": sess,
            "dow": dow, "hour": hour, "blocked": blocked,
            "noisy": noisy, "noisy_ts": noisy_ts, "be_ratio": be_ratio, "vd": vd,
        }
        ev["l4h"]  = long_gross(marks, ts, 4*3600_000)
        ev["l6h"]  = long_gross(marks, ts, 6*3600_000)
        ev["prov4"] = provisional_gross(marks, ts, noisy_ts, 4*3600_000)
        ev["prov6"] = provisional_gross(marks, ts, noisy_ts, 6*3600_000)
        ev["conf4"] = confirm_gross(marks, ts, 4*3600_000)
        ev["t15_6"] = t15_bounce_gross(marks, ts, 6*3600_000)
        events.append(ev)
    events.sort(key=lambda e: e["ts"])
    return events

def regime(ev): return ev["btc4h"] < 0 or ev["btc7d"] < 0
def base(ev):   return (not ev["bull"] and ev["sess"] != "EUROPE")

# ---------------------------------------------------------------------------
# R: REALISM
# ---------------------------------------------------------------------------

def run_R(ev_by_thr, months):
    print("\n=== R: REALISM (lookahead vs tradeable) ===")
    R = {}
    for thr in THRESHOLDS:
        evs = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        g = lambda ev: base(ev) and regime(ev)
        # ideal-silence lookahead (silence subset, hold 6h)
        R["R_%s_ideal_sil6" % lbl] = stats(
            [ev["l6h"] for ev in evs if g(ev) and not ev["noisy"] and ev.get("l6h") is not None],
            "%s ideal-silence(lookahead) 6h" % lbl, months)
        # hold-all 6h (tüm cascade, ERKEN ÇIKMA yok, silence şartı yok) — tradeable
        R["R_%s_holdall6" % lbl] = stats(
            [ev["l6h"] for ev in evs if g(ev) and ev.get("l6h") is not None],
            "%s hold-all 6h (erken çıkma yok)" % lbl, months)
        # provisional (tüm cascade, erken çıkış) hold 6h — GERÇEK
        R["R_%s_prov6" % lbl] = stats(
            [ev["prov6"] for ev in evs if g(ev) and ev.get("prov6") is not None],
            "%s provisional-early-exit 6h" % lbl, months)
        R["R_%s_prov4" % lbl] = stats(
            [ev["prov4"] for ev in evs if g(ev) and ev.get("prov4") is not None],
            "%s provisional-early-exit 4h" % lbl, months)
        # confirm T+30 (silence subset, enter T+30)
        R["R_%s_confirm" % lbl] = stats(
            [ev["conf4"] for ev in evs if g(ev) and not ev["noisy"] and ev.get("conf4") is not None],
            "%s confirm-T30 enter 4h" % lbl, months)
        # T+15 bounce (silence-partial, enter T+15)
        R["R_%s_t15" % lbl] = stats(
            [ev["t15_6"] for ev in evs if g(ev) and not ev["noisy"] and ev.get("t15_6") is not None],
            "%s T+15 bounce 6h" % lbl, months)
        for k in ["R_%s_ideal_sil6","R_%s_holdall6","R_%s_prov6","R_%s_prov4","R_%s_confirm","R_%s_t15"]:
            pstat(k % lbl, R[k % lbl])
    return R

# ---------------------------------------------------------------------------
# T: TAIL MANAGEMENT (primary threshold, provisional 6h)
# ---------------------------------------------------------------------------

def run_T(ev_by_thr, marks, months):
    print("\n=== T: TAIL MGMT (thr=%dK, provisional 6h) ===" % (PRIMARY_THR//1000))
    R = {}
    evs = ev_by_thr[PRIMARY_THR]
    g = lambda ev: base(ev) and regime(ev)
    sub = [ev for ev in evs if g(ev)]

    # stop sweep (provisional + stop)
    for sb in [None, 200, 150, 100]:
        vals = []
        for ev in sub:
            gg = provisional_gross(marks, ev["ts"], ev["noisy_ts"], 6*3600_000,
                                   stop_bps=(float(sb) if sb else None))
            if gg is not None: vals.append(gg)
        R["T_stop_%s" % (sb or "none")] = stats(vals, "stop=%s" % (sb or "none"), months)
        pstat("T_stop_%s" % (sb or "none"), R["T_stop_%s" % (sb or "none")])

    # veto layers (provisional 6h no stop)
    vetos = [
        ("V_none",      lambda ev: True),
        ("V_be2",       lambda ev: ev["be_ratio"] < 2.0),
        ("V_notUS1314", lambda ev: not ev["blocked"]),
        ("V_notSat",    lambda ev: ev["dow"] != 5),
        ("V_vol_ok",    lambda ev: ev["vd"] is None or ev["vd"] <= 8),
        ("V_be2_US_Sat",lambda ev: ev["be_ratio"]<2.0 and not ev["blocked"] and ev["dow"]!=5),
    ]
    for name, vf in vetos:
        vals = [ev["prov6"] for ev in sub if vf(ev) and ev.get("prov6") is not None]
        R["T_%s" % name] = stats(vals, name, months)
        pstat("T_%s" % name, R["T_%s" % name])

    # best combo: veto + stop150
    vals = []
    for ev in sub:
        if ev["be_ratio"]<2.0 and not ev["blocked"] and ev["dow"]!=5:
            gg = provisional_gross(marks, ev["ts"], ev["noisy_ts"], 6*3600_000, stop_bps=150.0)
            if gg is not None: vals.append(gg)
    R["T_veto_stop150"] = stats(vals, "veto(be2+US+Sat)+stop150", months)
    pstat("T_veto_stop150", R["T_veto_stop150"])
    return R

# ---------------------------------------------------------------------------
# N: NO-OVERLAP realistic frequency
# ---------------------------------------------------------------------------

def run_N(ev_by_thr, months):
    print("\n=== N: NO-OVERLAP gerçekçi frekans ===")
    R = {}
    for thr in THRESHOLDS:
        evs = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        g = lambda ev: base(ev) and regime(ev)
        for hk, hold, hl in [("prov6", 6*3600_000, "6h"), ("prov4", 4*3600_000, "4h")]:
            pairs = [(ev["ts"], ev[hk]) for ev in evs if g(ev) and ev.get(hk) is not None]
            raw = stats([v for _, v in pairs], "%s %s raw" % (lbl, hl), months)
            nov = no_overlap(pairs, hold)
            s = stats(nov, "%s %s no-overlap" % (lbl, hl), months)
            s["per_month"] = round(len(nov)/months, 1)
            s["pnl_per_month"] = round(sum(x-FEE_BPS for x in nov)/months, 0) if nov else 0
            R["N_%s_%s_raw" % (lbl, hl)] = raw
            R["N_%s_%s_noov" % (lbl, hl)] = s
            pstat("N_%s_%s_raw" % (lbl, hl), raw)
            pstat("N_%s_%s_noov" % (lbl, hl), s)
    return R

# ---------------------------------------------------------------------------
# C: COST / slippage
# ---------------------------------------------------------------------------

def run_C(ev_by_thr, months):
    print("\n=== C: COST + slippage ===")
    R = {}
    evs = ev_by_thr[PRIMARY_THR]
    g = lambda ev: base(ev) and regime(ev)
    vals = [ev["prov6"] for ev in evs if g(ev) and ev.get("prov6") is not None]
    for fee in [5, 8, 10, 15, 20]:
        R["C_fee%d" % fee] = stats(vals, "%dK prov6 fee=%d" % (PRIMARY_THR//1000, fee), months, fee=float(fee))
        pstat("C_fee%d" % fee, R["C_fee%d" % fee])
    return R

# ---------------------------------------------------------------------------
# F: FALSIFICATION
# ---------------------------------------------------------------------------

def run_F(conn, ev_by_thr, marks, months, now_ms, start_ms):
    print("\n=== F: FALSIFICATION (çürütme) ===")
    R = {}
    evs = ev_by_thr[PRIMARY_THR]
    g = lambda ev: base(ev) and regime(ev)

    # F1: random-entry kontrolü (aynı rejim, cascade OLMAYAN rastgele zaman) → edge ~0 olmalı
    print("  F1: random-entry (downtrend beta) kontrolü")
    lo = marks.ts[0]; hi = marks.ts[-1] - 6*3600_000
    rng = random.Random(7)
    rvals = []
    tries = 0
    while len(rvals) < 300 and tries < 4000:
        tries += 1
        rts = rng.randint(lo, hi)
        b4 = mark_bps(conn,"BTCUSDT",rts,4*3600_000)
        b7 = mark_bps(conn,"BTCUSDT",rts,7*24*3600_000)
        e1 = mark_bps(conn,"ETHUSDT",rts,3600_000)
        if (e1>20 and b4>50) or session_name(rts)=="EUROPE": continue
        if not (b4<0 or b7<0): continue
        gg = long_gross(marks, rts, 6*3600_000)
        if gg is not None: rvals.append(gg)
    R["F1_random_regime"] = stats(rvals, "RANDOM entry same-regime 6h (kontrol)", months)
    pstat("F1_random_regime", R["F1_random_regime"])
    R["F1_cascade_prov6"] = stats([ev["prov6"] for ev in evs if g(ev) and ev.get("prov6") is not None],
                                  "CASCADE prov6 (karşılaştırma)", months)
    pstat("F1_cascade_prov6", R["F1_cascade_prov6"])

    # F2: silence-window duyarlılığı (15/30/45m) + prop threshold (50/100K)
    print("  F2: silence-window duyarlılığı")
    for win_m in [15, 30, 45]:
        for prop in [50_000, 100_000]:
            vals = []
            for ev in evs:
                if not g(ev): continue
                # yeniden hesap: bu pencere/eşikte noisy var mı?
                nts = liq_first_ts(conn,"ETHUSDT","SELL",ev["ts"]+60_000,
                                   ev["ts"]+win_m*60_000, float(prop))
                gg = provisional_gross(marks, ev["ts"], nts, 6*3600_000)
                if gg is not None: vals.append(gg)
            k = "F2_win%d_prop%dK" % (win_m, prop//1000)
            R[k] = stats(vals, "silence win=%dm prop=%dK prov6" % (win_m, prop//1000), months)
            pstat(k, R[k])

    # F3: zaman-split (ilk yarı vs ikinci yarı)
    print("  F3: zaman-split degradation")
    sub = [ev for ev in evs if g(ev) and ev.get("prov6") is not None]
    if sub:
        mid = sub[len(sub)//2]["ts"]
        R["F3_first_half"]  = stats([ev["prov6"] for ev in sub if ev["ts"] < mid], "ilk yarı", months/2)
        R["F3_second_half"] = stats([ev["prov6"] for ev in sub if ev["ts"] >= mid], "ikinci yarı", months/2)
        pstat("F3_first_half", R["F3_first_half"])
        pstat("F3_second_half", R["F3_second_half"])

    # F4: tek-ay-çıkar (bir aya bağımlı mı?)
    print("  F4: single-month removed")
    by_m = defaultdict(list)
    for ev in sub: by_m[month_of(ev["ts"])].append(ev["prov6"])
    allv = [ev["prov6"] for ev in sub]
    base_total = sum(v-FEE_BPS for v in allv)
    for mo in sorted(by_m):
        rest = [v for ev in sub for v in [ev["prov6"]] if month_of(ev["ts"]) != mo]
        s = stats(rest, "− %s" % mo, months)
        R["F4_minus_%s" % mo] = s
        print("    − %s: N=%d TOTAL=%s (base=%d) mc_p=%s" % (
            mo, s.get("n",0), s.get("total"), int(base_total), s.get("mc_p")))

    # F5: top-3 winner removed
    srt = sorted(allv, reverse=True)
    R["F5_top3_removed"] = stats(srt[3:], "top-3 winner removed", months)
    pstat("F5_top3_removed", R["F5_top3_removed"])

    # F6: regime gerekli mi? (regime YOK)
    ng = lambda ev: base(ev)
    R["F6_no_regime"] = stats([ev["prov6"] for ev in evs if ng(ev) and ev.get("prov6") is not None],
                              "regime YOK prov6", months)
    pstat("F6_no_regime", R["F6_no_regime"])
    return R

# ---------------------------------------------------------------------------
# Z: FINAL scorecard
# ---------------------------------------------------------------------------

def run_Z(ev_by_thr, marks, months):
    print("\n=== Z: FINAL scorecard ===")
    R = {}
    # aday: primary thr, provisional 6h, veto(be2+US+Sat)+stop150, no-overlap
    for thr in THRESHOLDS:
        evs = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        g = lambda ev: (base(ev) and regime(ev) and ev["be_ratio"]<2.0
                        and not ev["blocked"] and ev["dow"]!=5)
        pairs = []
        for ev in evs:
            if not g(ev): continue
            gg = provisional_gross(marks, ev["ts"], ev["noisy_ts"], 6*3600_000, stop_bps=150.0)
            if gg is not None: pairs.append((ev["ts"], gg))
        raw = stats([v for _, v in pairs], "%s FINAL raw" % lbl, months)
        nov = no_overlap(pairs, 6*3600_000)
        s = stats(nov, "%s FINAL no-overlap" % lbl, months)
        s["per_month"] = round(len(nov)/months, 1)
        s["pnl_per_month"] = round(sum(x-FEE_BPS for x in nov)/months, 0) if nov else 0
        s["net10_avg"] = round(sum(x-10 for x in nov)/len(nov), 1) if nov else None
        R["Z_%s_raw" % lbl] = raw
        R["Z_%s_noov" % lbl] = s
        pstat("Z_%s_raw" % lbl, raw)
        pstat("Z_%s_noov" % lbl, s)
        print("        no-overlap /ay=%.1f  net@10bps avg=%s  mdd=%s  streak=%s" % (
            s.get("per_month",0), str(s.get("net10_avg")), str(s.get("mdd")), str(s.get("loss_streak"))))
    return R

# ---------------------------------------------------------------------------
# H: HOLD-ALL (düzeltilmiş çekirdek) — stop + no-overlap + cost + falsify
# ---------------------------------------------------------------------------

def hold_gross(marks, ts, hold_ms, stop_bps=None):
    r0 = marks.at_or_after(ts)
    if not r0 or float(r0[1]) <= 0: return None
    entry = float(r0[1])
    if stop_bps is not None:
        stop_px = entry*(1.0-stop_bps/10_000.0)
        for _, px in marks.slice_range(r0[0], ts+hold_ms):
            if float(px) <= stop_px: return -float(stop_bps)
    r1 = marks.at_or_before(ts+hold_ms)
    if not r1: return None
    return (float(r1[1])-entry)/entry*10_000.0

def run_H(ev_by_thr, marks, months):
    print("\n=== H: HOLD-ALL çekirdek (erken çıkış YOK) — deploy testi ===")
    R = {}
    g = lambda ev: base(ev) and regime(ev)
    for thr in THRESHOLDS:
        evs = ev_by_thr[thr]; lbl = "%dK" % (thr//1000)
        sub = [ev for ev in evs if g(ev)]
        # stop sweep (hold-all 6h + stop)
        for sb in [None, 200, 150, 100]:
            pairs = []
            for ev in sub:
                gg = hold_gross(marks, ev["ts"], 6*3600_000, stop_bps=(float(sb) if sb else None))
                if gg is not None: pairs.append((ev["ts"], gg))
            raw = stats([v for _, v in pairs], "%s holdall6 stop=%s" % (lbl, sb or "none"), months)
            R["H_%s_stop%s" % (lbl, sb or "none")] = raw
            pstat("H_%s_stop%s" % (lbl, sb or "none"), raw)
        # no-overlap (stop150) + cost
        pairs = []
        for ev in sub:
            gg = hold_gross(marks, ev["ts"], 6*3600_000, stop_bps=150.0)
            if gg is not None: pairs.append((ev["ts"], gg))
        nov = no_overlap(pairs, 6*3600_000)
        s = stats(nov, "%s holdall6 stop150 no-overlap" % lbl, months)
        s["per_month"] = round(len(nov)/months, 1)
        s["pnl_per_month"] = round(sum(x-FEE_BPS for x in nov)/months, 0) if nov else 0
        s["net10_pnl_mo"] = round(sum(x-10 for x in nov)/months, 0) if nov else 0
        R["H_%s_stop150_noov" % lbl] = s
        pstat("H_%s_stop150_noov" % lbl, s)
        print("        no-ov /ay=%.1f  pnl/mo@5=%s  pnl/mo@10=%s  mdd=%s  streak=%s" % (
            s.get("per_month",0), str(s.get("pnl_per_month")), str(s.get("net10_pnl_mo")),
            str(s.get("mdd")), str(s.get("loss_streak"))))

    # Falsify hold-all (primary thr): time-split + top3 + no-regime
    print("  Falsify hold-all (thr=%dK, stop150):" % (PRIMARY_THR//1000))
    sub = [ev for ev in ev_by_thr[PRIMARY_THR] if g(ev)]
    vals = []
    for ev in sub:
        gg = hold_gross(marks, ev["ts"], 6*3600_000, stop_bps=150.0)
        if gg is not None: vals.append((ev["ts"], gg))
    vals.sort()
    only = [v for _, v in vals]
    mid = vals[len(vals)//2][0]
    R["H_fals_first"]  = stats([v for t, v in vals if t < mid], "ilk yarı stop150", months/2)
    R["H_fals_second"] = stats([v for t, v in vals if t >= mid], "ikinci yarı stop150", months/2)
    R["H_fals_top3"]   = stats(sorted(only, reverse=True)[3:], "top-3 removed stop150", months)
    ng = [ev for ev in ev_by_thr[PRIMARY_THR] if base(ev)]
    nrv = []
    for ev in ng:
        gg = hold_gross(marks, ev["ts"], 6*3600_000, stop_bps=150.0)
        if gg is not None: nrv.append(gg)
    R["H_fals_no_regime"] = stats(nrv, "regime YOK stop150", months)
    for k in ["H_fals_first","H_fals_second","H_fals_top3","H_fals_no_regime"]:
        pstat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _row(k, v):
    if not v or v.get("n",0)==0:
        return "| %s | 0 | - | - | - | - | - | - | - | - | - |" % k
    return "| %s | %d | %.1f | %.1f%% | %+.1f | %s | %s | %d | %s | %s | %s |" % (
        k, v["n"], v.get("per_month",0), v["wr"], v["avg"], v.get("total"),
        v.get("pnl_per_month"), v.get("tail_n",0), v.get("mdd"), v.get("mc_p","?"), v.get("wf","-"))

def make_md(sections, meta):
    L = ["# S34 Silence-Core FINAL Gauntlet + Falsification", "",
         "> Ana alfa: **SILENCE LONG** (ETH SELL cascade + ilk 30dk follow-on YOK → LONG).",
         "> Realizm: silence T+30'da bilinir → gerçek mekanizma **provisional giriş + noisy'de erken çıkış**.",
         "> Evren: eşik başına ayrı, %.1f ay, FEE=%dbps (aksi belirtilmedikçe)." % (meta["months"], int(FEE_BPS)),
         "> Tarih: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"), "",
         "Kolon: N, /ay, WR, Avg, TOTAL, pnl/mo, Tail(<-100), MDD, mc_p, WF.", ""]
    titles = {
        "R": "R) Realizm — Lookahead vs Tradeable Giriş",
        "T": "T) Tail Yönetimi — Stop × Veto",
        "N": "N) No-Overlap Gerçekçi Frekans",
        "C": "C) Maliyet / Slippage Stresi",
        "F": "F) Falsification (Çürütme Testleri)",
        "H": "H) HOLD-ALL Düzeltilmiş Çekirdek (deploy testi)",
        "Z": "Z) FINAL Scorecard (provisional/veto — referans)",
    }
    hdr = "| Konfig | N | /ay | WR | Avg | TOTAL | pnl/mo | Tail | MDD | mc_p | WF |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for sec in ["R","T","N","C","F","H","Z"]:
        Rs = sections.get(sec, {})
        L += ["## %s" % titles[sec], "", hdr, sep]
        for k, v in Rs.items():
            if isinstance(v, dict):
                L.append(_row(k, v))
        L.append("")
    L += ["---", "*Script: tools/research_s34_silence_core_final.py*"]
    return "\n".join(L)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== S34 Silence-Core FINAL ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp()*1000)
        start_ms = now_ms - LOOKBACK_MS
        print("Loading marks + vol ...")
        marks = load_mark_index(conn, "ETHUSDT")
        vol_ts, vol_rows = load_vol_ts(conn)

        ev_by_thr = {}
        for thr in THRESHOLDS:
            print("Building events thr=%dK ..." % (thr//1000))
            ev_by_thr[thr] = build_events(conn, marks, vol_ts, vol_rows, thr, now_ms, start_ms)

        span = sorted(ev["ts"] for ev in ev_by_thr[PRIMARY_THR])
        span_days = (span[-1]-span[0])/86_400_000 if len(span)>1 else 30
        TOTAL_MONTHS = max(1.0, span_days/30.0)
        months = TOTAL_MONTHS
        print("  %.0f gün = %.2f ay" % (span_days, months))

        sections = {}
        sections["R"] = run_R(ev_by_thr, months)
        sections["T"] = run_T(ev_by_thr, marks, months)
        sections["N"] = run_N(ev_by_thr, months)
        sections["C"] = run_C(ev_by_thr, months)
        sections["F"] = run_F(conn, ev_by_thr, marks, months, now_ms, start_ms)
        sections["H"] = run_H(ev_by_thr, marks, months)
        sections["Z"] = run_Z(ev_by_thr, marks, months)

    meta = {"months": round(months, 2),
            "n_by_thr": {("%dK" % (t//1000)): len(ev_by_thr[t]) for t in THRESHOLDS}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sections": sections, "meta": meta}, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(sections, meta))
    print("\nJSON: %s\nMD:   %s\nDone." % (OUT_JSON, OUT_MD))


if __name__ == "__main__":
    main()
