"""S34 Refined Recipe Gauntlet — bulgulari birlestir + acik sorular + veri sagligi.

R1 dow max-stat MC: Pazartesi etkisi dow-fishing mi? (500 permutasyon, max-stat)
R2 tarif merdiveni (kumulatif): base s7>=2 -> +notMon -> score9(tsl+two_sided)>=4
   -> +funding<0 boost degil VETO test -> pt200 exit -> double-trigger sizing
R3 portfoy: rafine LONG + SHORT 13-17 confirm (noov, tek slot)
R4 veri sagligi: mark/liq/book tazeligi, mark gap taramasi (son 7 gun)

hour17 200K composite baz. FEE=5. Esikler onceki TRAIN'den sabit. MC=500.
Cikti: reports/research/s34/S34_REFINED_RECIPE.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_REFINED_RECIPE.json"; OM = OUT / "S34_REFINED_RECIPE.md"
PROP = 50_000.0; LB = 400 * 24 * 3600_000; FEE = 5.0; MC = 500; HOLD = 6 * 3600_000
TM = 4.5; TRAIN = 0.70
CT = {"sync": 0.5421, "rv": 0.0304, "d24": 5.0, "be_lo": 0.2195, "be_hi": 2.0,
      "shelf": 2_775_000.0, "whale": 6440.0, "tsl": 114.65, "two_sided": 68_425.0}
random.seed(42)


def _s(c, q, p=()):
    r = c.execute(q, p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lmax(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lcnt(c, s, sd, lo, hi, t): return int(_s(c, "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?", (s, sd, lo, hi, t)))
def lfirst(c, s, sd, lo, hi, t):
    r = c.execute("SELECT ts_ms FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=? ORDER BY ts_ms ASC LIMIT 1", (s, sd, lo, hi, t)).fetchone()
    return int(r[0]) if r else None
def mbps(c, s, ts, lb):
    a = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts - lb)).fetchone()
    b = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts)).fetchone()
    return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4 if a and b and float(a[0]) > 0 else None
def rv5(c, ts):
    r = c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def ofir(c, lo, hi):
    r = c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or r[0] is None: return None, None
    b, se = float(r[0]), float(r[1]); t = b + se
    whale = (float(r[2]) / int(r[3])) if r[3] else None
    return ((b - se) / t if t > 0 else 0.0), whale
def fund_rate(c, ts):
    r = c.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def nextfund(c, ts):
    r = c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
def dowf(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday()
def sxn(ts):
    h = hod(ts); return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")
def ep(m, ts):
    r = m.at_or_after(ts); return (int(r[0]), float(r[1])) if r and float(r[1]) > 0 else None
def lret(m, ts, hold):
    e = ep(m, ts)
    if not e: return None
    r = m.at_or_before(ts + hold); return (float(r[1]) - e[1]) / e[1] * 1e4 if r else None
def sret(m, ts, hold):
    e = ep(m, ts)
    if not e: return None
    r = m.at_or_before(ts + hold); return -(float(r[1]) - e[1]) / e[1] * 1e4 if r else None


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)
def mdd(vals):
    eq = 0.0; peak = 0.0; d = 0.0
    for v in vals:
        eq += v; peak = max(peak, eq); d = min(d, eq - peak)
    return round(d, 1)
def stat(g, label="", months=None, fee=FEE):
    m = months or TM
    if not g: return {"label": label, "n": 0}
    net = [x - fee for x in g]; n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    dd = mdd(net); tot = sum(net)
    return {"label": label, "n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(tot, 0), "per_month": round(n / m, 1), "worst": round(min(net), 1),
            "tail_n": sum(1 for x in net if x <= -100), "mdd": dd,
            "risk_adj": round(tot / max(abs(dd), 50.0), 2), "mc_p": mcp(net, a)}
def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-42s N=0" % k[:42]); return
    print("    %-42s N=%-4d /mo=%-5.1f WR=%-6s avg=%-7s TOT=%-8s worst=%-8s mdd=%-8s RA=%-6s mc=%s"
          % (k[:42], v["n"], v.get("per_month", 0), str(v["wr"]) + "%", str(v["avg"]),
             str(v.get("total")), str(v.get("worst")), str(v.get("mdd")), str(v.get("risk_adj")), v.get("mc_p", "?")))


def feats(conn, m, ts, rn):
    sk = lsum(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts)
    of, whale = ofir(conn, ts - 5 * 60_000, ts); e = ep(m, ts)
    shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, e[1] * 0.98, e[1])) if e else 0
    return {"sync": sk / rn if rn > 0 else 0, "rv": rv5(conn, ts),
            "d24": lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000),
            "ofi": of, "be": lmax(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / rn if rn > 0 else 0,
            "shelf": shelf, "whale": whale}
def hits(f):
    return {"sync": f["sync"] >= CT["sync"], "rv": f["rv"] is not None and f["rv"] >= CT["rv"],
            "d24": f["d24"] >= CT["d24"], "ofi": f["ofi"] is not None and f["ofi"] >= 0,
            "be": CT["be_lo"] <= f["be"] < CT["be_hi"],
            "shelf": f["shelf"] >= CT["shelf"], "whale_lo": f["whale"] is not None and f["whale"] < CT["whale"]}
def score7(f):
    h = hits(f); return sum(1 for k in ("sync", "rv", "d24", "ofi", "be", "shelf") if h[k])


def build(conn, m, now, start):
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(200_000.0,), accel_window_sec=30)
    ev = []; prev_ts = None
    for a in ancs:
        ts = int(a.anchor_ts_ms); rn = float(a.running_notional)
        if rn < 200_000 or m.at_or_after(ts) is None: continue
        b4 = mbps(conn, "BTCUSDT", ts, 4 * 3600_000) or 0
        b7 = mbps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0
        gate = not (((mbps(conn, "ETHUSDT", ts, 3600_000) or 0) > 20 and b4 > 50)
                    or sxn(ts) == "EUROPE" or not (b4 < 0 or b7 < 0) or hod(ts) < 17)
        tsl = (ts - prev_ts) / 60_000 if prev_ts is not None else None
        prev_ts = ts
        if not gate: continue
        f = feats(conn, m, ts, rn); nf = nextfund(conn, ts)
        m2f = ((nf - ts) / 60_000) if nf else None
        y = lret(m, ts, HOLD)
        if y is None: continue
        ev.append({"ts": ts, "rn": rn, "f": f, "h": hits(f), "s7": score7(f), "b7": b7,
                   "dow": dowf(ts), "veto": (m2f is not None and m2f < 60), "y": y,
                   "tsl": tsl, "two_sided": lsum(conn, "ETHUSDT", "BUY", ts - 3600_000, ts),
                   "funding": fund_rate(conn, ts)})
    ev.sort(key=lambda x: x["ts"])
    return ev


def s9(e):
    s = e["s7"]
    if e["tsl"] is not None and e["tsl"] < CT["tsl"]: s += 1
    if e["two_sided"] >= CT["two_sided"]: s += 1
    return s


def run_R1(ev):
    print("\n=== R1: dow max-stat MC (Pazartesi gercek mi?) ===")
    R = {}
    net = [(e["dow"], e["y"] - FEE) for e in ev]
    obs = {}
    for d in range(7):
        g = [v for dd, v in net if dd == d]
        if len(g) >= 5: obs[d] = sum(g) / len(g)
    mon = obs.get(0)
    base_avg = sum(v for _, v in net) / len(net)
    obs_dev = abs(mon - base_avg) if mon is not None else 0
    rng = random.Random(7); cnt = 0
    dows = [d for d, _ in net]; vals = [v for _, v in net]
    for _ in range(500):
        rng.shuffle(dows)
        mx = 0
        for d in range(7):
            g = [v for dd, v in zip(dows, vals) if dd == d]
            if len(g) >= 5: mx = max(mx, abs(sum(g) / len(g) - base_avg))
        if mx >= obs_dev: cnt += 1
    R["R1"] = {"monday_avg": round(mon, 1), "base_avg": round(base_avg, 1),
               "obs_dev": round(obs_dev, 1), "maxstat_p": round(cnt / 500, 3)}
    print(f"    Monday avg={mon:.1f} base={base_avg:.1f} dev={obs_dev:.1f} max-stat p={cnt/500:.3f}")
    cut = int(len(ev) * TRAIN); te = ev[cut:]
    R["R1_mon_TEST"] = stat([e["y"] for e in te if e["dow"] == 0], "Mon TEST", TM * (1 - TRAIN))
    ps("R1_mon_TEST", R["R1_mon_TEST"])
    return R


def run_R2(ev, m):
    print("\n=== R2: tarif merdiveni (noov, kumulatif) ===")
    R = {}
    cut = int(len(ev) * TRAIN); cut_ts = ev[cut]["ts"] if cut < len(ev) else 0
    tem = TM * (1 - TRAIN)

    def noov_ev(evs):
        busy = -1; o = []
        for e in evs:
            if e["ts"] >= busy: o.append(e); busy = e["ts"] + HOLD
        return o

    def pt_ret(e, target=200):
        e0 = ep(m, e["ts"])
        if not e0: return e["y"]
        for pts, px in m.slice_range(e0[0], e["ts"] + HOLD):
            if (px - e0[1]) / e0[1] * 1e4 >= target: return float(target)
        return e["y"]

    steps = {
        "L0_base_s2": lambda e: e["s7"] >= 2 and not e["veto"],
        "L1_+notMon": lambda e: e["s7"] >= 2 and not e["veto"] and e["dow"] != 0,
        "L2_score9_ge4": lambda e: s9(e) >= 4 and not e["veto"] and e["dow"] != 0,
        "L3_+fund_neg": lambda e: s9(e) >= 4 and not e["veto"] and e["dow"] != 0
                                  and (e["funding"] is not None and e["funding"] < 0),
    }
    for name, cond in steps.items():
        sub = noov_ev([e for e in ev if cond(e)])
        R[name] = stat([e["y"] for e in sub], name, TM); ps(name, R[name])
        R[name + "_TEST"] = stat([e["y"] for e in sub if e["ts"] >= cut_ts], name + " TEST", tem)
        ps(name + "_TEST", R[name + "_TEST"])
    # pt200 exit on L2
    sub = noov_ev([e for e in ev if steps["L2_score9_ge4"](e)])
    R["L4_L2_pt200"] = stat([pt_ret(e) for e in sub], "L2+pt200", TM); ps("L4_L2_pt200", R["L4_L2_pt200"])
    # funding<0 as extra score (10. sinyal) instead of hard filter
    def s10(e):
        return s9(e) + (1 if (e["funding"] is not None and e["funding"] < 0) else 0)
    for K in (4, 5):
        sub = noov_ev([e for e in ev if s10(e) >= K and not e["veto"] and e["dow"] != 0])
        R[f"L5_score10_ge{K}"] = stat([e["y"] for e in sub], f"s10>={K}", TM); ps(f"L5_score10_ge{K}", R[f"L5_score10_ge{K}"])
        R[f"L5_score10_ge{K}_TEST"] = stat([e["y"] for e in sub if e["ts"] >= cut_ts], f"s10>={K} TEST", tem)
        ps(f"L5_score10_ge{K}_TEST", R[f"L5_score10_ge{K}_TEST"])
    # sizing: L2 universe, double-trigger (rv+shelf 2u, +whale 3u), weighted
    sub = noov_ev([e for e in ev if steps["L2_score9_ge4"](e)])
    wp = []
    for e in sub:
        u = 3.0 if (e["h"]["rv"] and e["h"]["shelf"] and e["h"]["whale_lo"]) else \
            (2.0 if (e["h"]["rv"] and e["h"]["shelf"]) else 1.0)
        wp.append(u * (e["y"] - FEE))
    if wp:
        R["L6_sizing"] = {"n": len(wp), "w_total": round(sum(wp), 0), "w_mdd": mdd(wp),
                          "w_worst": round(min(wp), 1),
                          "risk_adj": round(sum(wp) / max(abs(mdd(wp)), 50.0), 2)}
        print("    L6_sizing (L2+double-trigger)             N=%-3d wTOT=%-8s wMDD=%-8s RA=%s"
              % (len(wp), R["L6_sizing"]["w_total"], R["L6_sizing"]["w_mdd"], R["L6_sizing"]["risk_adj"]))
    return R


def run_R3(conn, m, ev, now, start):
    print("\n=== R3: portfoy — rafine LONG + SHORT 13-17 (tek slot) ===")
    R = {}
    longs = [(e["ts"], e["y"], HOLD) for e in ev if s9(e) >= 4 and not e["veto"] and e["dow"] != 0]
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(200_000.0,), accel_window_sec=30)
    shorts = []
    for a in ancs:
        ts = int(a.anchor_ts_ms); rn = float(a.running_notional)
        if rn < 200_000 or m.at_or_after(ts) is None: continue
        b4 = mbps(conn, "BTCUSDT", ts, 4 * 3600_000) or 0
        if ((mbps(conn, "ETHUSDT", ts, 3600_000) or 0) > 20 and b4 > 50) or sxn(ts) == "EUROPE": continue
        nt = lfirst(conn, "ETHUSDT", "SELL", ts + 60_000, ts + 30 * 60_000, PROP)
        if nt is None or not (13 <= hod(nt) < 17): continue
        conf = lfirst(conn, "BTCUSDT", "SELL", nt + 5 * 60_000, nt + 30 * 60_000, 1_000_000.0)
        if conf is None: continue
        y = sret(m, conf, 180 * 60_000)
        if y is not None: shorts.append((conf, y, 180 * 60_000))
    R["R3_long_only"] = stat([v for _, v, _ in longs], "refined LONG", TM); ps("R3_long_only", R["R3_long_only"])
    R["R3_short_only"] = stat([v for _, v, _ in shorts], "SHORT 13-17", TM); ps("R3_short_only", R["R3_short_only"])
    busy = -1; combo = []
    for tsx, v, hold in sorted(longs + shorts):
        if tsx >= busy: combo.append(v); busy = tsx + hold
    R["R3_portfolio"] = stat(combo, "portfolio", TM); ps("R3_portfolio", R["R3_portfolio"])
    return R


def run_R4(conn):
    print("\n=== R4: veri sagligi ===")
    R = {}
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    for t, sym in (("mark_prices", "ETHUSDT"), ("liquidations", "ETHUSDT"), ("book_ticker", "ETHUSDT"),
                   ("agg_trades", "ETHUSDT"), ("vol_state", "ETHUSDT")):
        r = conn.execute(f"SELECT MAX(ts_ms) FROM {t} WHERE symbol=?", (sym,)).fetchone()
        age = (now - int(r[0])) / 60_000 if r and r[0] else None
        R[f"R4_{t}_age_min"] = round(age, 1) if age is not None else None
        print(f"    {t:18s} last age = {age:.1f} min" if age is not None else f"    {t}: NO DATA")
    # mark gap taramasi son 7 gun (>120s gap)
    rows = conn.execute("SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms",
                        (now - 7 * 24 * 3600_000,)).fetchall()
    gaps = []
    for i in range(1, len(rows)):
        d = (int(rows[i][0]) - int(rows[i - 1][0])) / 1000
        if d > 120: gaps.append((int(rows[i - 1][0]), round(d, 0)))
    R["R4_mark_gaps_7d"] = {"count": len(gaps), "max_gap_s": max((g[1] for g in gaps), default=0)}
    print(f"    mark gaps(>120s) last7d: {len(gaps)}  max={R['R4_mark_gaps_7d']['max_gap_s']}s")
    return R


def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Refined Recipe Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
        m = load_mark_index(conn, "ETHUSDT")
        print("build..."); ev = build(conn, m, now, start)
        span = [e["ts"] for e in ev]; TM = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R = {}
        R["R1"] = run_R1(ev)
        R["R2"] = run_R2(ev, m)
        R["R3"] = run_R3(conn, m, ev, now, start)
        R["R4"] = run_R4(conn)
    meta = {"n": len(ev), "months": round(TM, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps({"results": R, "meta": meta}, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Refined Recipe Gauntlet", "",
             f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for q, sec in R.items():
        lines += [f"## {q}", ""]
        for k, v in sec.items():
            if isinstance(v, dict) and v.get("n", 0) > 0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s worst=%s mdd=%s RA=%s mc_p=%s"
                             % (k, v["n"], v.get("per_month", 0), v["wr"], v["avg"], v.get("total"),
                                v.get("worst"), v.get("mdd"), v.get("risk_adj"), v.get("mc_p", "?")))
            elif isinstance(v, dict):
                lines.append(f"- **{k}**: {json.dumps(v, default=str)}")
        lines.append("")
    lines += ["---", "*Script: tools/research_s34_refined_recipe_gauntlet.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
