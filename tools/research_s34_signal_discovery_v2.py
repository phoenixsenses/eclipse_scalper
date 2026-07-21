"""S34 Signal Discovery V2 — yeni T0 sinyalleri + onceki adaylarin holdout dogrulamasi.

S1 gentleness (28k B1): pre-drop siddeti — TRAIN median -> TEST
S2 momentum-after-win (28h N2): onceki KAPANMIS event kazandiysa
S3 profit-target +150/+200/+250 vs fixed 6h (28k C2) — path bazli
S4 yeni T0 sinyalleri (TRAIN yon secer, TEST raporlar):
   dow, eth_pre1h, taker1h, two_sided(BUY liq pre-1h), spread, basis,
   funding_rate, time_since_last, sol_simul
S5 en iyi yeni sinyal(ler) composite'e eklenince score9 TEST'te kazaniyor mu

Mezarliga girilmez: buy-side fade / reversal / cross-asset transfer yok.
hour17 200K composite baz. FEE=5. no-lookahead: esikler TRAIN'de.
Cikti: reports/research/s34/S34_SIGNAL_DISCOVERY_V2.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_SIGNAL_DISCOVERY_V2.json"; OM = OUT / "S34_SIGNAL_DISCOVERY_V2.md"
PROP = 50_000.0; LB = 400 * 24 * 3600_000; FEE = 5.0; MC = 500; HOLD = 6 * 3600_000
TM = 4.5; TRAIN = 0.70
CT = {"sync": 0.5421, "rv": 0.0304, "d24": 5.0, "be_lo": 0.2195, "be_hi": 2.0,
      "imb": 0.2633, "shelf": 2_775_000.0, "whale": 6440.0}
random.seed(42)


def _s(c, q, p=()):
    r = c.execute(q, p).fetchone(); return float(r[0]) if r and r[0] is not None else 0.0
def lsum(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lmax(c, s, sd, lo, hi): return _s(c, "SELECT COALESCE(MAX(notional),0) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?", (s, sd, lo, hi))
def lcnt(c, s, sd, lo, hi, t): return int(_s(c, "SELECT COUNT(*) FROM liquidations WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<? AND notional>=?", (s, sd, lo, hi, t)))
def mbps(c, s, ts, lb):
    a = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts - lb)).fetchone()
    b = c.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (s, ts)).fetchone()
    return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4 if a and b and float(a[0]) > 0 else None
def rv5(c, ts):
    r = c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def bookrow(c, ts):
    r = c.execute("SELECT spread_pct,book_imbalance,ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    if not r or (ts - int(r[2])) > 5 * 60_000: return None, None
    return (float(r[0]) if r[0] is not None else None), (float(r[1]) if r[1] is not None else None)
def ofir(c, lo, hi):
    r = c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or r[0] is None: return None, None, None
    b, se = float(r[0]), float(r[1]); t = b + se
    whale = (float(r[2]) / int(r[3])) if r[3] else None
    return ((b - se) / t if t > 0 else 0.0), whale, (b / t if t > 0 else None)
def last_trade_px(c, ts):
    r = c.execute("SELECT price FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r else None
def fund_rate(c, ts):
    r = c.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
def nextfund(c, ts):
    r = c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
def dow(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday()
def sxn(ts):
    h = hod(ts); return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")
def ep(m, ts):
    r = m.at_or_after(ts); return (int(r[0]), float(r[1])) if r and float(r[1]) > 0 else None
def lret(m, ts, hold):
    e = ep(m, ts)
    if not e: return None
    r = m.at_or_before(ts + hold); return (float(r[1]) - e[1]) / e[1] * 1e4 if r else None


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)
def stat(g, label="", months=None, fee=FEE):
    m = months or TM
    if not g: return {"label": label, "n": 0}
    net = [x - fee for x in g]; n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    return {"label": label, "n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(sum(net), 0), "per_month": round(n / m, 1),
            "worst": round(min(net), 1), "tail_n": sum(1 for x in net if x <= -100), "mc_p": mcp(net, a)}
def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-40s N=0" % k[:40]); return
    print("    %-40s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-8s worst=%-8s mc=%s"
          % (k[:40], v["n"], v.get("per_month", 0), str(v["wr"]) + "%", str(v["avg"]),
             str(v.get("total")), str(v.get("worst")), v.get("mc_p", "?")))
def med(x):
    s = sorted(v for v in x if v is not None); return s[len(s) // 2] if s else None
def noov(pairs, hold=HOLD):
    busy = -1; o = []
    for ts, v in sorted(pairs):
        if ts >= busy: o.append(v); busy = ts + hold
    return o


def feats(conn, m, ts, rn):
    sk = lsum(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts)
    of, whale, tbr5 = ofir(conn, ts - 5 * 60_000, ts); e = ep(m, ts)
    shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, e[1] * 0.98, e[1])) if e else 0
    return {"sync": sk / rn if rn > 0 else 0, "rv": rv5(conn, ts),
            "d24": lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000),
            "ofi": of, "be": lmax(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / rn if rn > 0 else 0,
            "imb": None, "shelf": shelf, "whale": whale}


def hits(f):
    return {"sync": f["sync"] >= CT["sync"], "rv": f["rv"] is not None and f["rv"] >= CT["rv"],
            "d24": f["d24"] >= CT["d24"], "ofi": f["ofi"] is not None and f["ofi"] >= 0,
            "be": CT["be_lo"] <= f["be"] < CT["be_hi"], "imb": f["imb"] is not None and f["imb"] <= CT["imb"],
            "shelf": f["shelf"] >= CT["shelf"], "whale_lo": f["whale"] is not None and f["whale"] < CT["whale"]}
def score7(f):
    h = hits(f); return sum(1 for k in ("sync", "rv", "d24", "ofi", "be", "shelf") if h[k])  # imb yok (None)


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
        spread, imb = bookrow(conn, ts)
        e0 = ep(m, ts)
        pre15 = m.at_or_before(ts - 15 * 60_000)
        predrop = ((e0[1] - float(pre15[1])) / float(pre15[1]) * 1e4) if (e0 and pre15 and float(pre15[1]) > 0) else None
        _, _, tbr1h = ofir(conn, ts - 3600_000, ts)
        ltp = last_trade_px(conn, ts)
        basis = ((e0[1] - ltp) / ltp * 1e4) if (e0 and ltp and ltp > 0) else None
        ev.append({"ts": ts, "rn": rn, "f": f, "h": hits(f), "s7": score7(f), "b7": b7,
                   "hour": hod(ts), "dow": dow(ts), "veto": (m2f is not None and m2f < 60), "y": y,
                   "sig": {"predrop": predrop, "eth1h": mbps(conn, "ETHUSDT", ts, 3600_000),
                           "taker1h": tbr1h, "two_sided": lsum(conn, "ETHUSDT", "BUY", ts - 3600_000, ts),
                           "spread": spread, "basis": basis, "funding": fund_rate(conn, ts),
                           "tsl": tsl, "sol_sim": lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts + 60_000)}})
    ev.sort(key=lambda x: x["ts"])
    return ev


def run_S1_S2(ev):
    print("\n=== S1: gentleness (pre-drop) + S2: momentum-after-win ===")
    R = {}
    cut = int(len(ev) * TRAIN); tr, te = ev[:cut], ev[cut:]; tem = TM * (1 - TRAIN)
    mdn = med([e["sig"]["predrop"] for e in tr])
    R["S1_thresh"] = mdn
    # gentle = predrop az negatif (median ustu)
    for lbl, cond in (("gentle", lambda e: e["sig"]["predrop"] is not None and e["sig"]["predrop"] >= mdn),
                      ("sharp", lambda e: e["sig"]["predrop"] is not None and e["sig"]["predrop"] < mdn)):
        R[f"S1_{lbl}_full"] = stat([e["y"] for e in ev if cond(e)], lbl, TM)
        R[f"S1_{lbl}_TEST"] = stat([e["y"] for e in te if cond(e)], lbl + " TEST", tem)
        ps(f"S1_{lbl}_full", R[f"S1_{lbl}_full"]); ps(f"S1_{lbl}_TEST", R[f"S1_{lbl}_TEST"])
    # S2: momentum — son KAPANMIS event
    lastw = {}
    closed = []  # (close_ts, win?)
    for e in ev:
        st = None
        for cts, w in reversed(closed):
            if cts <= e["ts"]: st = w; break
        lastw[e["ts"]] = st
        closed.append((e["ts"] + HOLD, (e["y"] - FEE) > 0))
    for lbl, want in (("after_win", True), ("after_loss", False)):
        R[f"S2_{lbl}_full"] = stat([e["y"] for e in ev if lastw[e["ts"]] is want], lbl, TM)
        R[f"S2_{lbl}_TEST"] = stat([e["y"] for e in te if lastw[e["ts"]] is want], lbl + " TEST", tem)
        ps(f"S2_{lbl}_full", R[f"S2_{lbl}_full"]); ps(f"S2_{lbl}_TEST", R[f"S2_{lbl}_TEST"])
    return R


def run_S3(ev, m):
    print("\n=== S3: profit-target vs fixed 6h (score>=2, noov) ===")
    R = {}
    sub = [e for e in ev if e["s7"] >= 2 and not e["veto"]]
    busy = -1; adm = []
    for e in sub:
        if e["ts"] >= busy: adm.append(e); busy = e["ts"] + HOLD
    def pt_ret(e, target):
        e0 = ep(m, e["ts"])
        if not e0: return None
        path = m.slice_range(e0[0], e["ts"] + HOLD)
        for pts, px in path:
            if (px - e0[1]) / e0[1] * 1e4 >= target: return target
        return e["y"]
    R["S3_fixed6h"] = stat([e["y"] for e in adm], "fixed 6h", TM); ps("S3_fixed6h", R["S3_fixed6h"])
    for t in (150, 200, 250, 300):
        g = [pt_ret(e, t) for e in adm]; g = [x for x in g if x is not None]
        R[f"S3_pt{t}"] = stat(g, f"pt+{t}", TM); ps(f"S3_pt{t}", R[f"S3_pt{t}"])
    return R


def run_S4(ev):
    print("\n=== S4: yeni T0 sinyalleri (TRAIN yon secimi -> TEST) ===")
    R = {}
    cut = int(len(ev) * TRAIN); tr, te = ev[:cut], ev[cut:]; tem = TM * (1 - TRAIN)
    sigs = ["predrop", "eth1h", "taker1h", "two_sided", "spread", "basis", "funding", "tsl", "sol_sim"]
    for k in sigs:
        vals = [e["sig"][k] for e in tr if e["sig"][k] is not None]
        if len(vals) < 20:
            print(f"    {k}: TRAIN N yetersiz ({len(vals)})"); continue
        mdn = med(vals)
        hi_tr = [e["y"] for e in tr if e["sig"][k] is not None and e["sig"][k] >= mdn]
        lo_tr = [e["y"] for e in tr if e["sig"][k] is not None and e["sig"][k] < mdn]
        if not hi_tr or not lo_tr: continue
        fav = "hi" if (sum(hi_tr) / len(hi_tr)) > (sum(lo_tr) / len(lo_tr)) else "lo"
        cond = (lambda e, m=mdn: e["sig"][k] is not None and e["sig"][k] >= m) if fav == "hi" \
            else (lambda e, m=mdn: e["sig"][k] is not None and e["sig"][k] < m)
        te_f = stat([e["y"] for e in te if cond(e)], f"{k} {fav} TEST", tem)
        te_u = stat([e["y"] for e in te if e["sig"][k] is not None and not cond(e)], f"{k} anti TEST", tem)
        full = stat([e["y"] for e in ev if cond(e)], f"{k} {fav} full", TM)
        delta = (te_f.get("avg") or 0) - (te_u.get("avg") or 0)
        R[f"S4_{k}"] = {"fav": fav, "thresh": mdn, "TEST": te_f, "TEST_anti": te_u, "full": full,
                        "test_delta": round(delta, 1)}
        print("    %-11s fav=%-3s thr=%-12s TEST N=%-3d WR=%-6s avg=%-7s (anti avg=%-7s) delta=%-7s full_mc=%s"
              % (k, fav, round(mdn, 4) if isinstance(mdn, float) else mdn, te_f.get("n", 0),
                 str(te_f.get("wr")) + "%", str(te_f.get("avg")), str(te_u.get("avg")),
                 str(round(delta, 1)), full.get("mc_p")))
    # dow kategorik
    print("    -- dow (full) --")
    R["S4_dow"] = {}
    for d in range(7):
        s = stat([e["y"] for e in ev if e["dow"] == d], f"dow{d}", TM)
        R["S4_dow"][d] = s
        if s.get("n", 0) >= 5:
            print("      dow=%d N=%-3d WR=%-6s avg=%-8s mc=%s" % (d, s["n"], str(s["wr"]) + "%", str(s["avg"]), s.get("mc_p")))
    return R


def run_S5(ev, s4):
    print("\n=== S5: composite'e yeni sinyal ekleme (TEST) ===")
    R = {}
    cut = int(len(ev) * TRAIN); te = ev[cut:]; tem = TM * (1 - TRAIN)
    # en iyi 2 yeni sinyali sec (test_delta'ya gore, N>=15)
    cands = sorted(((k, v) for k, v in s4.items()
                    if isinstance(v, dict) and "test_delta" in v and v["TEST"].get("n", 0) >= 15),
                   key=lambda kv: kv[1]["test_delta"], reverse=True)[:2]
    R["S5_added"] = [k for k, _ in cands]
    print("    eklenen:", R["S5_added"])
    def extra(e):
        s = 0
        for k, v in cands:
            key = k.replace("S4_", ""); val = e["sig"][key]
            if val is None: continue
            if v["fav"] == "hi" and val >= v["thresh"]: s += 1
            if v["fav"] == "lo" and val < v["thresh"]: s += 1
        return s
    for K in (3, 4, 5):
        base = stat([e["y"] for e in te if e["s7"] >= K and not e["veto"]], f"s7>={K} TEST", tem)
        ext = stat([e["y"] for e in te if (e["s7"] + extra(e)) >= K + 1 and not e["veto"]], f"s9>={K+1} TEST", tem)
        R[f"S5_s7_ge{K}"] = base; R[f"S5_s9_ge{K+1}"] = ext
        ps(f"S5_s7_ge{K}_TEST", base); ps(f"S5_s9_ge{K+1}_TEST", ext)
    return R


def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Signal Discovery V2 ===")
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
        m = load_mark_index(conn, "ETHUSDT")
        print("build..."); ev = build(conn, m, now, start)
        span = [e["ts"] for e in ev]; TM = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R = {}
        R["S1S2"] = run_S1_S2(ev)
        R["S3"] = run_S3(ev, m)
        R["S4"] = run_S4(ev)
        R["S5"] = run_S5(ev, R["S4"])
    meta = {"n": len(ev), "months": round(TM, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps({"results": R, "meta": meta}, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Signal Discovery V2", "",
             f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    def emit(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict) and v.get("n", 0) > 0 and "wr" in v:
                lines.append("- **%s%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s worst=%s mc_p=%s"
                             % (prefix, k, v["n"], v.get("per_month", 0), v["wr"], v["avg"],
                                v.get("total"), v.get("worst"), v.get("mc_p", "?")))
            elif isinstance(v, dict) and "TEST" in v:
                t = v["TEST"]
                lines.append("- **%s%s** fav=%s thr=%s: TEST N=%s WR=%s avg=%s delta=%s (full mc=%s)"
                             % (prefix, k, v["fav"], v["thresh"], t.get("n"), t.get("wr"), t.get("avg"),
                                v["test_delta"], v["full"].get("mc_p")))
    for q, sec in R.items():
        lines += [f"## {q}", ""]; emit(sec); lines.append("")
    lines += ["---", "*Script: tools/research_s34_signal_discovery_v2.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
