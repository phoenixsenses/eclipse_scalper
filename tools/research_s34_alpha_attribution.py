"""S34 Alpha Attribution — "Ana alfa ne?" sorusuna veriyle cevap.

Soru: Sistemin tüm route'ları (silence LONG, echo, double-cascade, SHORT_NOISY)
ETH SELL likidasyon cascade → LONG reversion çekirdeği üstüne kurulu. Ama:
  - Toplam net PnL'i asıl HANGİ konfigürasyon taşıyor?
  - Hangi filtre para getiriyor, hangisi sadece frekans kesiyor (over-filter)?
  - Ana alfayı büyütmek için doğru frekans/eşik noktası nerede?

Metrik: TOTAL net bps (dönem toplamı) = /ay × avg × ay. WR değil — toplam para.

Bölümler:
  A  RAW base rate — tüm cascade universe, filtresiz (çekirdek edge + hold seçimi)
  B  Eşik taraması — 100/150/200/300K, toplam-PnL frontier (daha çok trade?)
  C  Filtre attribution — her filtrenin marjinal toplam-PnL + frekans katkısı
  D  Çekirdek konfig frontier — toplam net PnL'e göre sıralı, robustness kolonlu
  E  Portföy stack — core LONG + SHORT_NOISY + echo: toplam PnL ve overlap

Çıktı:
  reports/research/s34/S34_ALPHA_ATTRIBUTION.json
  reports/research/s34/S34_ALPHA_ATTRIBUTION.md
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
    load_liquidations,
    load_mark_index,
    reconstruct_anchors,
)
from tools.research_s34_wave_absorption import book_features_at

DB_PATH  = ROOT / "data" / "microstructure.db"
OUT_DIR  = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_ALPHA_ATTRIBUTION.json"
OUT_MD   = OUT_DIR / "S34_ALPHA_ATTRIBUTION.md"

ETH_THRESH  = 200_000.0
PROP_THRESH =  50_000.0
LOOKBACK_MS = 400 * 24 * 3600_000
FEE_BPS     = 5.0
MC_ITER     = 2000
TOTAL_MONTHS = 4.5

random.seed(42)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _scalar(conn, sql, params=()):
    row = conn.execute(sql, params).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0

def liq_cnt(conn, sym, side, lo, hi, thr):
    return int(_scalar(conn,
        "SELECT COUNT(*) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?",
        (sym, side, lo, hi, thr)))

def liq_sum(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(SUM(notional),0) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_max(conn, sym, side, lo, hi):
    return _scalar(conn,
        "SELECT COALESCE(MAX(notional),0) FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (sym, side, lo, hi))

def liq_first_ts(conn, sym, side, lo, hi, thr):
    row = conn.execute(
        "SELECT ts_ms FROM liquidations "
        "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?"
        " ORDER BY ts_ms ASC LIMIT 1", (sym, side, lo, hi, thr)).fetchone()
    return int(row[0]) if row else None

def mark_bps(conn, sym, ts_ms, lookback_ms):
    r0 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms - lookback_ms)).fetchone()
    r1 = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? "
        "ORDER BY ts_ms DESC LIMIT 1", (sym, ts_ms)).fetchone()
    if r0 and r1 and float(r0[0]) > 0:
        return (float(r1[0]) - float(r0[0])) / float(r0[0]) * 10_000.0
    return 0.0

def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    if 7 <= h < 13:
        return "EUROPE"
    if 13 <= h < 21:
        return "US"
    return "OFF"

def dow_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).weekday()

def hour_of(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour

def compute_score(conn, ts, sync_k, n2h):
    b4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30)
            + int(13 <= hour < 21) + int(sync_k >= 200_000))

def echo_check(ts_list, ts, lo_min, hi_min):
    lo_ms = ts - hi_min * 60_000
    hi_ms = ts - lo_min * 60_000
    lo_i = bisect.bisect_left(ts_list, lo_ms)
    hi_i = bisect.bisect_left(ts_list, hi_ms)
    return any(ts_list[i] != ts for i in range(lo_i, hi_i))

# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def long_gross(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return (float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000.0

def short_gross(marks, ts, hold_ms):
    r0 = marks.at_or_after(ts)
    r1 = marks.at_or_before(ts + hold_ms)
    if not r0 or not r1 or float(r0[1]) <= 0:
        return None
    return -(float(r1[1]) - float(r0[1])) / float(r0[1]) * 10_000.0

# ---------------------------------------------------------------------------
# Stats — TOTAL net PnL öne çıkar
# ---------------------------------------------------------------------------

def _mc_p(vals, avg):
    if len(vals) < 4:
        return None
    rng = random.Random(0)
    ct = sum(1 for _ in range(MC_ITER)
             if sum(rng.choice([-1, 1]) * abs(v) for v in vals) / len(vals) >= avg)
    return round(ct / MC_ITER, 3)

def wf_folds(vals, k=5):
    n = len(vals)
    if n < k:
        return None
    pos = sum(1 for i in range(k) if sum(vals[i*n//k:(i+1)*n//k]) > 0)
    return "%d/%d" % (pos, k)

def stats(gross_vals, label="", months=None, fee=FEE_BPS):
    m = months or TOTAL_MONTHS
    if not gross_vals:
        return {"label": label, "n": 0}
    net = [v - fee for v in gross_vals]
    n = len(net)
    wins = sum(1 for v in net if v > 0)
    sv = sorted(net)
    avg = sum(net) / n
    total = sum(net)
    cut = max(1, int(n * 0.7))
    ho = net[cut:]
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "total": round(total, 0),            # ← ana metrik
        "per_month": round(n / m, 1),
        "pnl_per_month": round(total / m, 0),  # ← aylık ortalama net bps
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in net if v < -100),
        "mc_p": _mc_p(net, avg),
        "wf": wf_folds(net),
        "ho_avg": round(sum(ho) / len(ho), 1) if ho else None,
    }

def pstat(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-34s N=0" % k[:34])
        return
    print("    %-34s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOTAL=%-7s pnl/mo=%-6s tail=%-2d mc_p=%s wf=%s" % (
        k[:34], v["n"], v.get("per_month", 0), str(v["wr"]) + "%",
        str(v["avg"]) + "bps", str(v.get("total")), str(v.get("pnl_per_month")),
        v.get("tail_n", 0), v.get("mc_p", "?"), v.get("wf")))

# ---------------------------------------------------------------------------
# Build 200K events
# ---------------------------------------------------------------------------

def build_events(conn, anchors, marks_eth):
    events = []
    ts_list = sorted(int(a.anchor_ts_ms) for a in anchors)
    total = len(anchors)
    print("  Building %d events ..." % total)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print("    [%d/%d]" % (i + 1, total))
        ts = int(anc.anchor_ts_ms)
        rn = float(anc.running_notional)
        if marks_eth.at_or_after(ts) is None:
            continue
        sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts)
                + liq_sum(conn, "SOLUSDT", "SELL", ts - 10*60_000, ts))
        n2h    = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
        score  = compute_score(conn, ts, sync_k, n2h)
        btc4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        btc7d  = mark_bps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000)
        eth1h  = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        bull   = eth1h > 20.0 and btc4h > 50.0
        sess   = session_name(ts)
        dow    = dow_of(ts)
        hour   = hour_of(ts)
        blocked = (sess == "US" and hour in {13, 14})
        prebuildup = liq_cnt(conn, "ETHUSDT", "SELL", ts - 30*60_000, ts - 1000, PROP_THRESH)
        noisy_ts = liq_first_ts(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 30*60_000, PROP_THRESH)
        noisy = noisy_ts is not None
        btc_5d_30m = liq_max(conn, "BTCUSDT", "SELL", ts + 5*60_000, ts + 30*60_000)
        echo_30_90 = echo_check(ts_list, ts, 30, 90)
        ev = {
            "ts": ts, "rn": rn, "sync_k": sync_k, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "bull": bull, "sess": sess,
            "dow": dow, "hour": hour, "blocked": blocked,
            "prebuildup": prebuildup, "noisy": noisy, "noisy_ts": noisy_ts,
            "btc_5d_30m": btc_5d_30m, "echo_30_90": echo_30_90,
        }
        ev["l2h"] = long_gross(marks_eth, ts, 2*3600_000)
        ev["l3h"] = long_gross(marks_eth, ts, 3*3600_000)
        ev["l4h"] = long_gross(marks_eth, ts, 4*3600_000)
        ev["l6h"] = long_gross(marks_eth, ts, 6*3600_000)
        ev["sn_2h"] = short_gross(marks_eth, noisy_ts, 2*3600_000) if noisy_ts else None
        events.append(ev)
    events.sort(key=lambda e: e["ts"])
    return events

# ---------------------------------------------------------------------------
# A: RAW base rate (filtresiz)
# ---------------------------------------------------------------------------

def run_A(events, months):
    print("\n=== A: RAW base rate (tüm cascade, filtresiz) ===")
    R = {}
    for hk in ["l2h", "l3h", "l4h", "l6h"]:
        v = [ev[hk] for ev in events if ev.get(hk) is not None]
        R["A_raw_LONG_%s" % hk] = stats(v, "RAW LONG %s (tüm cascade)" % hk, months)
        pstat("A_raw_LONG_%s" % hk, R["A_raw_LONG_%s" % hk])
    # minimal gate: not bull, not EU
    for hk in ["l4h", "l6h"]:
        v = [ev[hk] for ev in events
             if not ev["bull"] and ev["sess"] != "EUROPE" and ev.get(hk) is not None]
        R["A_min_LONG_%s" % hk] = stats(v, "not bull/EU LONG %s" % hk, months)
        pstat("A_min_LONG_%s" % hk, R["A_min_LONG_%s" % hk])
    # silence only
    v = [ev["l4h"] for ev in events
         if not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"] and ev.get("l4h") is not None]
    R["A_silence_LONG_4h"] = stats(v, "not bull/EU + silence LONG 4h", months)
    pstat("A_silence_LONG_4h", R["A_silence_LONG_4h"])
    return R

# ---------------------------------------------------------------------------
# B: Eşik taraması — toplam-PnL frontier
# ---------------------------------------------------------------------------

def run_B(conn, marks_eth, now_ms, start_ms, months):
    print("\n=== B: Eşik taraması (toplam-PnL frontier) ===")
    R = {}
    liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
    for thr in [100_000, 150_000, 200_000, 300_000]:
        lbl = "%dK" % (thr // 1000)
        ancs = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                   thresholds=(float(thr),), accel_window_sec=30)
        vals_min, vals_sil, vals_sil_reg = [], [], []
        for anc in ancs:
            ts = int(anc.anchor_ts_ms)
            if float(anc.running_notional) < thr:
                continue
            if marks_eth.at_or_after(ts) is None:
                continue
            btc4h = mark_bps(conn, "BTCUSDT", ts, 4*3600_000)
            btc7d = mark_bps(conn, "BTCUSDT", ts, 7*24*3600_000)
            eth1h = mark_bps(conn, "ETHUSDT", ts, 3600_000)
            bull  = eth1h > 20 and btc4h > 50
            sess  = session_name(ts)
            if bull or sess == "EUROPE":
                continue
            noisy = liq_first_ts(conn, "ETHUSDT", "SELL", ts+60_000, ts+30*60_000, PROP_THRESH) is not None
            l4 = long_gross(marks_eth, ts, 4*3600_000)
            if l4 is None:
                continue
            vals_min.append(l4)
            if not noisy:
                vals_sil.append(l4)
                if btc4h < 0 or btc7d < 0:
                    vals_sil_reg.append(l4)
        R["B_%s_min" % lbl]     = stats(vals_min, "%s not bull/EU LONG 4h" % lbl, months)
        R["B_%s_sil" % lbl]     = stats(vals_sil, "%s +silence" % lbl, months)
        R["B_%s_sil_reg" % lbl] = stats(vals_sil_reg, "%s +silence+regime" % lbl, months)
        for k in ["B_%s_min" % lbl, "B_%s_sil" % lbl, "B_%s_sil_reg" % lbl]:
            pstat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# C: Filtre attribution — marjinal toplam-PnL katkısı
# ---------------------------------------------------------------------------

def run_C(events, months):
    print("\n=== C: Filtre attribution (200K, LONG 4h) ===")
    R = {}

    # Ortak yardımcı filtreler
    F = {
        "not_bull":    lambda ev: not ev["bull"],
        "not_EU":      lambda ev: ev["sess"] != "EUROPE",
        "silence":     lambda ev: not ev["noisy"],
        "regime":      lambda ev: ev["btc4h"] < 0 or ev["btc7d"] < 0,
        "sync200":     lambda ev: ev["sync_k"] < 200_000,
        "score3":      lambda ev: ev["score"] >= 3,
        "not_US1314":  lambda ev: not ev["blocked"],
        "not_MonWed":  lambda ev: ev["dow"] not in {0, 2},
        "echo3090":    lambda ev: ev["echo_30_90"],
        "prebuild":    lambda ev: ev["prebuildup"] > 0,
    }

    # (1) Tekli filtre: her filtre TEK BAŞINA (LONG 4h) toplam-PnL
    print("  (1) Tekil filtre toplam-PnL:")
    for name, fn in F.items():
        v = [ev["l4h"] for ev in events if fn(ev) and ev.get("l4h") is not None]
        R["C1_%s" % name] = stats(v, "solo %s" % name, months)
        pstat("C1_%s" % name, R["C1_%s" % name])

    # (2) Kümülatif ekleme sırası (kalite artışı) — her adımda toplam-PnL & frekans
    print("  (2) Kümülatif stack (ekledikçe):")
    order = ["not_bull", "not_EU", "silence", "regime", "sync200",
             "not_US1314", "not_MonWed", "score3"]
    active = []
    for name in order:
        active.append(name)
        def gate(ev, act=list(active)):
            return all(F[a](ev) for a in act)
        v = [ev["l4h"] for ev in events if gate(ev) and ev.get("l4h") is not None]
        R["C2_%02d_%s" % (len(active), name)] = stats(v, "+".join(active), months)
        pstat("C2_%02d_+%s" % (len(active), name), R["C2_%02d_%s" % (len(active), name)])

    # (3) Leave-one-out: çekirdek stack'ten her filtreyi ÇIKAR — kaybettiğimiz/kazandığımız
    print("  (3) Leave-one-out (çekirdek = not_bull+not_EU+silence+regime):")
    core = ["not_bull", "not_EU", "silence", "regime"]
    def core_gate(ev):
        return all(F[a](ev) for a in core)
    v_core = [ev["l4h"] for ev in events if core_gate(ev) and ev.get("l4h") is not None]
    R["C3_core"] = stats(v_core, "CORE not_bull+EU+silence+regime", months)
    pstat("C3_core", R["C3_core"])
    for name in core:
        rest = [a for a in core if a != name]
        def gate(ev, r=rest):
            return all(F[a](ev) for a in r)
        v = [ev["l4h"] for ev in events if gate(ev) and ev.get("l4h") is not None]
        R["C3_minus_%s" % name] = stats(v, "core − %s" % name, months)
        pstat("C3_minus_%s" % name, R["C3_minus_%s" % name])
    return R

# ---------------------------------------------------------------------------
# D: Çekirdek konfig frontier (toplam-PnL sıralı)
# ---------------------------------------------------------------------------

def run_D(events, months):
    print("\n=== D: Çekirdek konfig frontier ===")
    R = {}
    def g(ev, **k):
        if k.get("not_bull") and ev["bull"]: return False
        if k.get("not_EU") and ev["sess"] == "EUROPE": return False
        if k.get("silence") and ev["noisy"]: return False
        if k.get("regime") and not (ev["btc4h"] < 0 or ev["btc7d"] < 0): return False
        if k.get("sync") and not ev["sync_k"] < 200_000: return False
        if k.get("not_US") and ev["blocked"]: return False
        if k.get("not_MW") and ev["dow"] in {0, 2}: return False
        if k.get("score3") and ev["score"] < 3: return False
        return True

    configs = {
        "D_min":                 dict(not_bull=1, not_EU=1),
        "D_silence":             dict(not_bull=1, not_EU=1, silence=1),
        "D_sil_regime":          dict(not_bull=1, not_EU=1, silence=1, regime=1),
        "D_sil_reg_US_MW":       dict(not_bull=1, not_EU=1, silence=1, regime=1, not_US=1, not_MW=1),
        "D_sil_reg_US_MW_sync":  dict(not_bull=1, not_EU=1, silence=1, regime=1, not_US=1, not_MW=1, sync=1),
        "D_full_live":           dict(not_bull=1, not_EU=1, silence=1, regime=1, not_US=1, not_MW=1, sync=1, score3=1),
        "D_no_silence":          dict(not_bull=1, not_EU=1, regime=1, not_US=1, not_MW=1),
        "D_no_regime":           dict(not_bull=1, not_EU=1, silence=1, not_US=1, not_MW=1),
    }
    for name, kw in configs.items():
        for hk, hl in [("l4h", "4h"), ("l6h", "6h")]:
            v = [ev[hk] for ev in events if g(ev, **kw) and ev.get(hk) is not None]
            R["%s_%s" % (name, hl)] = stats(v, "%s %s" % (name, hl), months)
            pstat("%s_%s" % (name, hl), R["%s_%s" % (name, hl)])
    return R

# ---------------------------------------------------------------------------
# E: Portföy stack (non-overlapping)
# ---------------------------------------------------------------------------

def run_E(events, months):
    print("\n=== E: Portföy stack (core LONG + SHORT_NOISY + echo) ===")
    R = {}

    core_long = lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
                            and (ev["btc4h"] < 0 or ev["btc7d"] < 0)
                            and not ev["blocked"] and ev["dow"] not in {0, 2})
    short_noisy = lambda ev: (ev["noisy"] and ev.get("noisy_ts") is not None
                              and ev["btc_5d_30m"] >= 1_000_000
                              and not ev["bull"] and ev["sess"] != "EUROPE" and ev["dow"] != 6)
    echo_long = lambda ev: core_long(ev) and ev["echo_30_90"]

    # Legs: (gate, out_key, hold_ms)
    legs = {
        "core_long":   (core_long,   "l4h", 4*3600_000),
        "short_noisy": (short_noisy, "sn_2h", 2*3600_000),
        "echo_long":   (echo_long,   "l4h", 4*3600_000),
    }
    # tekil
    for name, (gate, key, _) in legs.items():
        v = [ev[key] for ev in events if gate(ev) and ev.get(key) is not None]
        R["E_%s" % name] = stats(v, name, months)
        pstat("E_%s" % name, R["E_%s" % name])

    # kombinasyonlar (non-overlapping, zaman sıralı; entry ts = anchor veya noisy_ts)
    def portfolio(leg_names):
        busy = -1
        out = []
        rows = []
        for ev in events:
            for nm in leg_names:
                gate, key, hold = legs[nm]
                if gate(ev) and ev.get(key) is not None:
                    ent = ev["noisy_ts"] if nm == "short_noisy" else ev["ts"]
                    rows.append((ent, ev[key], hold))
                    break
        rows.sort()
        for ent, val, hold in rows:
            if ent >= busy:
                out.append(val)
                busy = ent + hold
        return out

    for name, legn in [
        ("E_long_short",      ["core_long", "short_noisy"]),
        ("E_long_echo",       ["core_long", "echo_long"]),
        ("E_all_three",       ["core_long", "short_noisy", "echo_long"]),
    ]:
        v = portfolio(legn)
        s = stats(v, name + " (non-overlap)", months)
        s["per_month"] = round(len(v) / months, 1)
        s["pnl_per_month"] = round(sum(x - FEE_BPS for x in v) / months, 0) if v else 0
        R[name] = s
        pstat(name, s)
    return R

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _row(k, v):
    if not v or v.get("n", 0) == 0:
        return "| %s | 0 | - | - | - | - | - | - | - | - |" % k
    return "| %s | %d | %.1f | %.1f%% | %+.1f | %s | %s | %d | %s | %s |" % (
        k, v["n"], v.get("per_month", 0), v["wr"], v["avg"],
        v.get("total"), v.get("pnl_per_month"), v.get("tail_n", 0),
        v.get("mc_p", "?"), v.get("wf", "-"))

def make_md(sections, meta):
    L = ["# S34 Alpha Attribution — Ana Alfa Nerede?", "",
         "> Soru: Toplam net PnL'i hangi konfigürasyon taşıyor? Hangi filtre para, hangisi over-filter?",
         "> Metrik: **TOTAL** = dönem toplam net bps; **pnl/mo** = aylık ortalama net bps.",
         "> Evren: %d ETH SELL 200K anchor, %.1f ay, FEE=%dbps." % (
             meta["n_events"], meta["months"], int(FEE_BPS)),
         "> Tarih: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"), "",
         "Kolon: N, /ay, WR, Avg, **TOTAL**, **pnl/mo**, Tail, mc_p, WF.", ""]
    titles = {
        "A": "A) RAW Base Rate (filtresiz cascade çekirdeği)",
        "B": "B) Eşik Taraması (toplam-PnL frontier)",
        "C": "C) Filtre Attribution (marjinal katkı)",
        "D": "D) Çekirdek Konfig Frontier",
        "E": "E) Portföy Stack (non-overlapping)",
    }
    hdr = "| Konfig | N | /ay | WR | Avg | TOTAL | pnl/mo | Tail | mc_p | WF |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for sec in ["A", "B", "C", "D", "E"]:
        Rs = sections.get(sec, {})
        L += ["## %s" % titles[sec], "", hdr, sep]
        for k, v in Rs.items():
            L.append(_row(k, v))
        L.append("")
    L += ["---", "*Script: tools/research_s34_alpha_attribution.py*"]
    return "\n".join(L)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    print("=== S34 Alpha Attribution ===")
    with sqlite3.connect("file:%s?mode=ro" % DB_PATH, uri=True) as conn:
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("PRAGMA temp_store=MEMORY")
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ms = now_ms - LOOKBACK_MS

        print("Loading ETH SELL liqs ...")
        liqs = load_liquidations(conn, "ETHUSDT", "SELL", start_ms, now_ms)
        print("Reconstructing 200K anchors ...")
        anchors = reconstruct_anchors(liqs, bucket_sec=300, min_gap_sec=900,
                                      thresholds=(ETH_THRESH,), accel_window_sec=30)
        print("  anchors: %d" % len(anchors))
        span = sorted(int(a.anchor_ts_ms) for a in anchors)
        span_days = (span[-1] - span[0]) / 86_400_000 if len(span) > 1 else 30
        TOTAL_MONTHS = max(1.0, span_days / 30.0)
        months = TOTAL_MONTHS
        print("  %.0f gün = %.2f ay" % (span_days, months))

        marks_eth = load_mark_index(conn, "ETHUSDT")
        events = build_events(conn, anchors, marks_eth)
        print("  events: %d" % len(events))

        sections = {}
        sections["A"] = run_A(events, months)
        sections["C"] = run_C(events, months)
        sections["D"] = run_D(events, months)
        sections["E"] = run_E(events, months)
        sections["B"] = run_B(conn, marks_eth, now_ms, start_ms, months)

    meta = {"n_events": len(events), "months": round(months, 2)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sections": sections, "meta": meta}, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(sections, meta))
    print("\nJSON: %s\nMD:   %s" % (OUT_JSON, OUT_MD))

    # Özet: en yüksek toplam-PnL konfigler
    print("\n=== ÖZET: en yüksek TOTAL net PnL (N>=15, mc_p<=0.05, tail<=2) ===")
    allr = {}
    for R in sections.values():
        allr.update(R)
    good = [(k, v) for k, v in allr.items()
            if v and v.get("n", 0) >= 15 and v.get("mc_p") is not None
            and v["mc_p"] <= 0.05 and v.get("tail_n", 99) <= 2]
    good.sort(key=lambda x: -(x[1].get("total") or 0))
    for k, v in good[:15]:
        pstat(k, v)
    print("Done.")


if __name__ == "__main__":
    main()
