"""S34 Echo Live-Readiness Gauntlet.

Aday: echo + regime (btc4h<0 OR btc7d<0), silence, LONG.
  - echo_30_90 + regime  : ~7/ay, WR 81%, tail 0 (kalite)
  - echo_30_120 + regime : ~9.7/ay, WR 75%, tail 2 (denge)

Canlıya almadan önce cevaplanması gereken sorular:
  ENTRY  Giriş mekanizması: T0 vs T+15 bounce-confirm vs sabit gecikme.
         (Live executor T0 provisional yerine T+15 bounce kullanıyor.)
  HOLD   Çıkış/hold taraması 2/3/4/6h.
  TAIL   Tail adli analizi + tek-veto denemeleri (Sat / be_ratio / vol).
  COST   Fee duyarlılığı 5/8/10/15 bps + no-overlap gerçekçi /ay.
  REGIME Rejim bacağı ayrıştırma + ay-ay stabilite.
  FINAL  Live-readiness scorecard.

Çıktı:
  reports/research/s34/S34_ECHO_LIVE_GAUNTLET.json
  reports/research/s34/S34_ECHO_LIVE_GAUNTLET.md
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
OUT_JSON = OUT_DIR / "S34_ECHO_LIVE_GAUNTLET.json"
OUT_MD   = OUT_DIR / "S34_ECHO_LIVE_GAUNTLET.md"

ETH_THRESH  = 200_000.0
PROP_THRESH =  50_000.0
LOOKBACK_MS = 400 * 24 * 3600_000
FEE_BPS     = 5.0
MC_ITER     = 2000
HOLD_MS     = 4 * 3600_000
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

def month_of(ts_ms):
    d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return "%04d-%02d" % (d.year, d.month)

def iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

def compute_score(conn, ts, sync_k, n2h):
    b4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
    book = book_features_at(conn, "ETHUSDT", ts, 30)
    vdep = float(book.get("vdepth_bps") or 0) if book else 0.0
    hour = hour_of(ts)
    return (int(n2h >= 3) + int(b4h < 0) + int(vdep >= 30)
            + int(13 <= hour < 21) + int(sync_k >= 200_000))

def load_vol_state(conn):
    return conn.execute(
        "SELECT ts_ms, rv_5m, vol_decile, high_vol_alert FROM vol_state "
        "WHERE symbol='ETHUSDT' ORDER BY ts_ms").fetchall()

def echo_check(ts_list, ts, lo_min, hi_min):
    lo_ms = ts - hi_min * 60_000
    hi_ms = ts - lo_min * 60_000
    lo_i = bisect.bisect_left(ts_list, lo_ms)
    hi_i = bisect.bisect_left(ts_list, hi_ms)
    return any(ts_list[i] != ts for i in range(lo_i, hi_i))

# ---------------------------------------------------------------------------
# Gross returns (fee ayrı uygulanır)
# ---------------------------------------------------------------------------

def gross(marks, entry_ts, exit_ts, stop_bps=None):
    r0 = marks.at_or_after(entry_ts)
    if not r0 or float(r0[1]) <= 0:
        return None
    entry = float(r0[1])
    if stop_bps is not None:
        stop_px = entry * (1.0 - stop_bps / 10_000.0)
        for _, px in marks.slice_range(r0[0], exit_ts):
            if float(px) <= stop_px:
                return -float(stop_bps)
    r1 = marks.at_or_before(exit_ts)
    if not r1:
        return None
    return (float(r1[1]) - entry) / entry * 10_000.0

def mark_at(marks, ts):
    r = marks.at_or_before(ts)
    return float(r[1]) if r else None

# ---------------------------------------------------------------------------
# Stats
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
    pos = sum(1 for i in range(k)
              if sum(vals[i * n // k:(i + 1) * n // k]) > 0)
    return "%d/%d" % (pos, k)

def stats(vals, label="", months=None, fee=FEE_BPS):
    """vals = GROSS bps listesi (zaman sıralı). fee net için düşülür."""
    m = months or TOTAL_MONTHS
    if not vals:
        return {"label": label, "n": 0}
    net = [v - fee for v in vals]
    n = len(net)
    wins = sum(1 for v in net if v > 0)
    sv = sorted(net)
    t3 = sorted(net, reverse=True)
    t3r = sum(t3[3:]) if len(t3) > 3 else sum(t3)
    avg = sum(net) / n
    cut = max(1, int(n * 0.7))
    ho = net[cut:]
    return {
        "label": label, "n": n,
        "wr": round(100 * wins / n, 1),
        "avg": round(avg, 1),
        "sum": round(sum(net), 1),
        "t3r": round(t3r, 1),
        "worst": round(sv[0], 1),
        "tail_n": sum(1 for v in net if v < -100),
        "per_month": round(n / m, 1),
        "mc_p": _mc_p(net, avg),
        "wf": wf_folds(net),
        "ho_n": len(ho),
        "ho_avg": round(sum(ho) / len(ho), 1) if ho else None,
        "ho_wr": round(100 * sum(1 for v in ho if v > 0) / len(ho), 1) if ho else None,
    }

def no_overlap(pairs, hold_ms=HOLD_MS):
    busy = -1
    out = []
    for ts, val in sorted(pairs):
        if ts >= busy:
            out.append(val)
            busy = ts + hold_ms
    return out

def pstat(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-38s N=0" % k[:38])
        return
    print("    %-38s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s worst=%-7s tail=%-2d mc_p=%s wf=%s" % (
        k[:38], v["n"], v.get("per_month", 0), str(v["wr"]) + "%",
        str(v["avg"]) + "bps", str(v.get("worst")), v.get("tail_n", 0),
        v.get("mc_p", "?"), v.get("wf")))

# ---------------------------------------------------------------------------
# Build events
# ---------------------------------------------------------------------------

def build_events(conn, anchors, marks_eth, vol_rows):
    events = []
    ts_list = sorted(int(a.anchor_ts_ms) for a in anchors)
    vol_ts = [r[0] for r in vol_rows]
    total = len(anchors)
    print("  Building %d events ..." % total)
    for i, anc in enumerate(anchors):
        if (i + 1) % 100 == 0:
            print("    [%d/%d]" % (i + 1, total))
        ts = int(anc.anchor_ts_ms)
        rn = float(anc.running_notional)
        p0 = marks_eth.at_or_after(ts)
        if p0 is None:
            continue

        sync_k = (liq_sum(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts)
                + liq_sum(conn, "SOLUSDT", "SELL", ts - 10*60_000, ts))
        n2h    = liq_cnt(conn, "ETHUSDT", "SELL", ts - 2*3600_000, ts - 1000, PROP_THRESH)
        score  = compute_score(conn, ts, sync_k, n2h)
        btc4h  = mark_bps(conn, "BTCUSDT", ts, 4 * 3600_000)
        btc7d  = mark_bps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000)
        btc3d  = mark_bps(conn, "BTCUSDT", ts, 3 * 24 * 3600_000)
        eth1h  = mark_bps(conn, "ETHUSDT", ts, 3600_000)
        bull   = eth1h > 20.0 and btc4h > 50.0
        sess   = session_name(ts)
        dow    = dow_of(ts)
        hour   = hour_of(ts)
        blocked = (sess == "US" and hour in {13, 14})
        prebuildup = liq_cnt(conn, "ETHUSDT", "SELL", ts - 30*60_000, ts - 1000, PROP_THRESH)
        noisy = liq_first_ts(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 30*60_000, PROP_THRESH) is not None
        btc_conc = liq_max(conn, "BTCUSDT", "SELL", ts - 10*60_000, ts + 10*60_000)
        be_ratio = (btc_conc / rn) if rn > 0 else 0.0
        vs_idx = bisect.bisect_right(vol_ts, ts) - 1
        vd_now = int(vol_rows[vs_idx][2]) if vs_idx >= 0 and vol_rows[vs_idx][2] is not None else None

        echo_30_90  = echo_check(ts_list, ts, 30, 90)
        echo_30_120 = echo_check(ts_list, ts, 30, 120)

        events.append({
            "ts": ts, "rn": rn, "sync_k": sync_k, "score": score,
            "btc4h": btc4h, "btc7d": btc7d, "btc3d": btc3d,
            "bull": bull, "sess": sess, "dow": dow, "hour": hour,
            "blocked": blocked, "prebuildup": prebuildup, "noisy": noisy,
            "be_ratio": be_ratio, "vd_now": vd_now,
            "echo_30_90": echo_30_90, "echo_30_120": echo_30_120,
        })
    events.sort(key=lambda e: e["ts"])

    print("  Computing entry/hold/stop grosses ...")
    for ev in events:
        ts = ev["ts"]
        mark0 = mark_at(marks_eth, ts)
        # T0 hold sweep (gross)
        ev["g_t0_2h"] = gross(marks_eth, ts, ts + 2*3600_000)
        ev["g_t0_3h"] = gross(marks_eth, ts, ts + 3*3600_000)
        ev["g_t0_4h"] = gross(marks_eth, ts, ts + 4*3600_000)
        ev["g_t0_6h"] = gross(marks_eth, ts, ts + 6*3600_000)
        # T0 + stop150 (4h)
        ev["g_t0_4h_s150"] = gross(marks_eth, ts, ts + 4*3600_000, stop_bps=150.0)
        # fixed delay entries, hold 4h from entry
        for dm in (5, 10, 15, 30):
            ev["g_d%d_4h" % dm] = gross(marks_eth, ts + dm*60_000, ts + dm*60_000 + 4*3600_000)
        # T+15 bounce confirm: enter T+15 only if mark(T+15) >= mark(T0); exit anchor+4h
        m15 = mark_at(marks_eth, ts + 15*60_000)
        confirm = (mark0 is not None and m15 is not None and m15 >= mark0)
        ev["t15_confirm"] = confirm
        ev["g_t15_4h"] = gross(marks_eth, ts + 15*60_000, ts + 4*3600_000) if confirm else None
        ev["g_t15_4h_s150"] = gross(marks_eth, ts + 15*60_000, ts + 4*3600_000, stop_bps=150.0) if confirm else None
    return events

# ---------------------------------------------------------------------------
# Candidate gates
# ---------------------------------------------------------------------------

def regime(ev):
    return ev["btc4h"] < 0 or ev["btc7d"] < 0

def cand_9090(ev):   # echo_30_90 + regime
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
            and ev["dow"] not in {0, 2} and ev["echo_30_90"] and regime(ev))

def cand_30120(ev):  # echo_30_120 + regime
    return (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
            and ev["dow"] not in {0, 2} and ev["echo_30_120"] and regime(ev))

def gvals(events, gate, key):
    return [ev[key] for ev in events if gate(ev) and ev.get(key) is not None]

def gpairs(events, gate, key):
    return [(ev["ts"], ev[key]) for ev in events if gate(ev) and ev.get(key) is not None]

# ---------------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------------

def run_ENTRY(events, months):
    print("\n=== ENTRY: giriş mekanizması ===")
    R = {}
    for cname, gate in [("e3090", cand_9090), ("e30120", cand_30120)]:
        R["ENTRY_%s_T0" % cname]  = stats(gvals(events, gate, "g_t0_4h"), "%s T0 4h" % cname, months)
        pstat("ENTRY_%s_T0" % cname, R["ENTRY_%s_T0" % cname])
        for dm in (5, 10, 15, 30):
            k = "ENTRY_%s_d%d" % (cname, dm)
            R[k] = stats(gvals(events, gate, "g_d%d_4h" % dm), "%s delay%dm 4h" % (cname, dm), months)
            pstat(k, R[k])
        # T+15 bounce confirm (live mekanizması)
        conf_gate = lambda ev, g=gate: g(ev) and ev.get("t15_confirm")
        R["ENTRY_%s_t15conf" % cname] = stats(gvals(events, conf_gate, "g_t15_4h"),
                                              "%s T+15 bounce-confirm" % cname, months)
        pstat("ENTRY_%s_t15conf" % cname, R["ENTRY_%s_t15conf" % cname])
        # confirm oranı
        tot = sum(1 for ev in events if gate(ev))
        con = sum(1 for ev in events if gate(ev) and ev.get("t15_confirm"))
        print("      %s confirm oranı: %d/%d (%.0f%%)" % (cname, con, tot, 100*con/max(1,tot)))
    return R

# ---------------------------------------------------------------------------
# HOLD
# ---------------------------------------------------------------------------

def run_HOLD(events, months):
    print("\n=== HOLD: çıkış süresi taraması ===")
    R = {}
    for cname, gate in [("e3090", cand_9090), ("e30120", cand_30120)]:
        for hk, hl in [("g_t0_2h","2h"), ("g_t0_3h","3h"), ("g_t0_4h","4h"), ("g_t0_6h","6h")]:
            k = "HOLD_%s_%s" % (cname, hl)
            R[k] = stats(gvals(events, gate, hk), "%s hold %s" % (cname, hl), months)
            pstat(k, R[k])
    return R

# ---------------------------------------------------------------------------
# TAIL forensics
# ---------------------------------------------------------------------------

def run_TAIL(events, months):
    print("\n=== TAIL: adli analiz + tek-veto ===")
    R = {}
    losers = []
    for ev in events:
        if cand_30120(ev) and ev.get("g_t0_4h") is not None:
            net = ev["g_t0_4h"] - FEE_BPS
            if net < -100:
                losers.append({
                    "ts": iso(ev["ts"]), "net": round(net, 1), "dow": ev["dow"],
                    "sess": ev["sess"], "hour": ev["hour"], "btc4h": round(ev["btc4h"], 1),
                    "btc7d": round(ev["btc7d"], 1), "prebuildup": ev["prebuildup"],
                    "rn": round(ev["rn"], 0), "sync_k": round(ev["sync_k"], 0),
                    "be_ratio": round(ev["be_ratio"], 2), "vd_now": ev["vd_now"],
                    "echo_30_90": ev["echo_30_90"],
                })
    R["_losers"] = losers
    print("  echo_30_120+regime tail (<-100 net) event sayısı: %d" % len(losers))
    for lo in losers:
        print("    %s net=%s dow=%d sess=%s btc4h=%s btc7d=%s pre=%d be=%s vd=%s e3090=%s" % (
            lo["ts"], lo["net"], lo["dow"], lo["sess"], lo["btc4h"], lo["btc7d"],
            lo["prebuildup"], lo["be_ratio"], lo["vd_now"], lo["echo_30_90"]))

    # tek-veto denemeleri (echo_30_120+regime üstünde)
    vetos = [
        ("V_base",       cand_30120),
        ("V_no_sat",     lambda ev: cand_30120(ev) and ev["dow"] != 5),
        ("V_be_lt2",     lambda ev: cand_30120(ev) and ev["be_ratio"] < 2.0),
        ("V_be_lt1",     lambda ev: cand_30120(ev) and ev["be_ratio"] < 1.0),
        ("V_prebuild",   lambda ev: cand_30120(ev) and ev["prebuildup"] > 0),
        ("V_vol_low",    lambda ev: cand_30120(ev) and (ev.get("vd_now") is None or ev["vd_now"] <= 6)),
        ("V_not_us1314", lambda ev: cand_30120(ev) and not ev["blocked"]),
        ("V_echo3090",   lambda ev: cand_30120(ev) and ev["echo_30_90"]),
        ("V_btc4h_hard", lambda ev: cand_30120(ev) and ev["btc4h"] < 0),
    ]
    for name, g in vetos:
        R[name] = stats(gvals(events, g, "g_t0_4h"), name, months)
        pstat(name, R[name])
    return R

# ---------------------------------------------------------------------------
# COST
# ---------------------------------------------------------------------------

def run_COST(events, months):
    print("\n=== COST: fee duyarlılığı + no-overlap ===")
    R = {}
    for cname, gate, key in [
        ("e3090_T0", cand_9090, "g_t0_4h"),
        ("e30120_T0", cand_30120, "g_t0_4h"),
        ("e30120_t15", lambda ev: cand_30120(ev) and ev.get("t15_confirm"), "g_t15_4h"),
    ]:
        g = gvals(events, gate, key)
        for fee in (5, 8, 10, 15):
            k = "COST_%s_fee%d" % (cname, fee)
            R[k] = stats(g, "%s fee=%d" % (cname, fee), months, fee=float(fee))
            pstat(k, R[k])
        # no-overlap gerçekçi
        pairs = gpairs(events, gate, key)
        nov = no_overlap(pairs)
        s = stats(nov, "%s no-overlap fee5" % cname, months)
        s["per_month"] = round(len(nov) / months, 1)
        R["COST_%s_noov" % cname] = s
        pstat("COST_%s_noov" % cname, s)
    return R

# ---------------------------------------------------------------------------
# REGIME decomposition + month stability
# ---------------------------------------------------------------------------

def run_REGIME(events, months):
    print("\n=== REGIME: bacak ayrıştırma + ay stabilite ===")
    R = {}
    base = lambda ev: (not ev["bull"] and ev["sess"] != "EUROPE" and not ev["noisy"]
                       and ev["dow"] not in {0, 2} and ev["echo_30_120"])
    legs = [
        ("REG_none",     base),
        ("REG_btc4h",    lambda ev: base(ev) and ev["btc4h"] < 0),
        ("REG_btc7d",    lambda ev: base(ev) and ev["btc7d"] < 0),
        ("REG_btc3d",    lambda ev: base(ev) and ev["btc3d"] < 0),
        ("REG_or4h7d",   lambda ev: base(ev) and (ev["btc4h"] < 0 or ev["btc7d"] < 0)),
        ("REG_and4h7d",  lambda ev: base(ev) and ev["btc4h"] < 0 and ev["btc7d"] < 0),
        ("REG_4h_only",  lambda ev: base(ev) and ev["btc4h"] < 0 and ev["btc7d"] >= 0),
        ("REG_7d_only",  lambda ev: base(ev) and ev["btc7d"] < 0 and ev["btc4h"] >= 0),
    ]
    for name, g in legs:
        R[name] = stats(gvals(events, g, "g_t0_4h"), name, months)
        pstat(name, R[name])

    # ay-ay stabilite (echo_30_120+regime)
    by_month = defaultdict(list)
    for ev in events:
        if cand_30120(ev) and ev.get("g_t0_4h") is not None:
            by_month[month_of(ev["ts"])].append(ev["g_t0_4h"] - FEE_BPS)
    print("  Ay-ay (echo_30_120+regime):")
    month_tbl = {}
    for mo in sorted(by_month):
        vv = by_month[mo]
        wr = round(100 * sum(1 for v in vv if v > 0) / len(vv), 1)
        av = round(sum(vv) / len(vv), 1)
        month_tbl[mo] = {"n": len(vv), "wr": wr, "avg": av, "sum": round(sum(vv), 1)}
        print("    %s N=%-3d WR=%-6s avg=%-8s sum=%s" % (mo, len(vv), str(wr)+"%", str(av)+"bps", round(sum(vv),1)))
    R["_month_stability"] = month_tbl
    return R

# ---------------------------------------------------------------------------
# FINAL scorecard
# ---------------------------------------------------------------------------

def run_FINAL(events, months):
    print("\n=== FINAL: live-readiness scorecard ===")
    R = {}
    cands = [
        ("F_e3090_T0",      cand_9090,  "g_t0_4h"),
        ("F_e3090_t15",     lambda ev: cand_9090(ev) and ev.get("t15_confirm"),  "g_t15_4h"),
        ("F_e30120_T0",     cand_30120, "g_t0_4h"),
        ("F_e30120_t15",    lambda ev: cand_30120(ev) and ev.get("t15_confirm"), "g_t15_4h"),
        ("F_e30120_T0_s150", cand_30120, "g_t0_4h_s150"),
        ("F_e30120_prebuild", lambda ev: cand_30120(ev) and ev["prebuildup"] > 0, "g_t0_4h"),
    ]
    for name, gate, key in cands:
        s = stats(gvals(events, gate, key), name, months)
        pairs = gpairs(events, gate, key)
        nov = no_overlap(pairs)
        s["noov_n"] = len(nov)
        s["noov_per_month"] = round(len(nov) / months, 1)
        s["noov_wr"] = round(100 * sum(1 for v in nov if (v - FEE_BPS) > 0) / len(nov), 1) if nov else None
        # verdict
        verdict = "RESEARCH_ONLY"
        if (s.get("n", 0) >= 20 and s.get("wr", 0) >= 72 and s.get("tail_n", 99) == 0
                and s.get("mc_p", 1) is not None and s.get("mc_p", 1) <= 0.05
                and s.get("wf") in ("5/5", "4/5")):
            verdict = "PAPER_CANDIDATE"
        elif (s.get("n", 0) >= 15 and s.get("wr", 0) >= 70 and s.get("tail_n", 99) <= 2
              and s.get("mc_p", 1) is not None and s.get("mc_p", 1) <= 0.05):
            verdict = "SHADOW_ONLY"
        s["verdict"] = verdict
        R[name] = s
        pstat(name, s)
        print("        no-overlap: N=%d /mo=%.1f WR=%s  -> %s" % (
            s["noov_n"], s["noov_per_month"], str(s["noov_wr"]) + "%", verdict))
    return R

# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _row(k, v):
    if not v or v.get("n", 0) == 0:
        return "| %s | 0 | - | - | - | - | - | - | - |" % k
    return "| %s | %d | %.1f | %.1f%% | %+.1f | %+.1f | %d | %s | %s |" % (
        k, v["n"], v.get("per_month", 0), v["wr"], v["avg"],
        v.get("worst", 0), v.get("tail_n", 0), v.get("mc_p", "?"), v.get("wf", "-"))

def make_md(sections, meta):
    L = ["# S34 Echo Live-Readiness Gauntlet", "",
         "> Aday: **echo + regime** (btc4h<0 OR btc7d<0), silence LONG.",
         "> Evren: %d anchor, %.1f ay, FEE=%dbps (aksi belirtilmedikçe), hold=4h." % (
             meta["n_events"], meta["months"], int(FEE_BPS)),
         "> Tarih: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d"),
         "", "Kolon: N, /ay, WR, Avg(net), Worst, Tail(<-100), mc_p, WF.", ""]
    titles = {
        "ENTRY": "1) Giriş Mekanizması (T0 / gecikme / T+15 bounce-confirm)",
        "HOLD":  "2) Çıkış Süresi Taraması",
        "TAIL":  "3) Tail Adli Analizi + Tek-Veto",
        "COST":  "4) Fee Duyarlılığı + No-Overlap",
        "REGIME":"5) Rejim Bacağı + Ay Stabilite",
        "FINAL": "6) Live-Readiness Scorecard",
    }
    hdr = "| Test | N | /ay | WR | Avg | Worst | Tail | mc_p | WF |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    for sec in ["ENTRY", "HOLD", "TAIL", "COST", "REGIME", "FINAL"]:
        Rs = sections.get(sec, {})
        L += ["## %s" % titles[sec], "", hdr, sep]
        for k, v in Rs.items():
            if k.startswith("_"):
                continue
            L.append(_row(k, v))
        L.append("")
        if sec == "TAIL" and Rs.get("_losers"):
            L += ["**Tail eventleri (echo_30_120+regime, net<-100):**", "",
                  "| Zaman | net | dow | sess | btc4h | btc7d | pre | be | vd | e3090 |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
            for lo in Rs["_losers"]:
                L.append("| %s | %s | %d | %s | %s | %s | %d | %s | %s | %s |" % (
                    lo["ts"], lo["net"], lo["dow"], lo["sess"], lo["btc4h"],
                    lo["btc7d"], lo["prebuildup"], lo["be_ratio"], lo["vd_now"], lo["echo_30_90"]))
            L.append("")
        if sec == "REGIME" and Rs.get("_month_stability"):
            L += ["**Ay-ay (echo_30_120+regime):**", "", "| Ay | N | WR | avg | sum |",
                  "|---|---:|---:|---:|---:|"]
            for mo, mv in Rs["_month_stability"].items():
                L.append("| %s | %d | %.1f%% | %+.1f | %+.1f |" % (
                    mo, mv["n"], mv["wr"], mv["avg"], mv["sum"]))
            L.append("")
        if sec == "FINAL":
            L += ["**No-overlap + verdict:**", "", "| Aday | noov N | noov /ay | noov WR | verdict |",
                  "|---|---:|---:|---:|---|"]
            for k, v in Rs.items():
                if k.startswith("_") or not v or v.get("n", 0) == 0:
                    continue
                L.append("| %s | %d | %.1f | %s | %s |" % (
                    k, v.get("noov_n", 0), v.get("noov_per_month", 0),
                    str(v.get("noov_wr")) + "%", v.get("verdict", "-")))
            L.append("")
    L += ["---", "*Script: tools/research_s34_echo_live_gauntlet.py*"]
    return "\n".join(L)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TOTAL_MONTHS
    print("=== S34 Echo Live-Readiness Gauntlet ===")
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
        vol_rows = load_vol_state(conn)
        events = build_events(conn, anchors, marks_eth, vol_rows)
        print("  events: %d" % len(events))

        sections = {}
        sections["ENTRY"]  = run_ENTRY(events, months)
        sections["HOLD"]   = run_HOLD(events, months)
        sections["TAIL"]   = run_TAIL(events, months)
        sections["COST"]   = run_COST(events, months)
        sections["REGIME"] = run_REGIME(events, months)
        sections["FINAL"]  = run_FINAL(events, months)

    meta = {"n_events": len(events), "months": round(months, 2)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"sections": sections, "meta": meta}, f, indent=2, default=str)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(make_md(sections, meta))
    print("\nJSON: %s\nMD:   %s" % (OUT_JSON, OUT_MD))
    print("Done.")


if __name__ == "__main__":
    main()
