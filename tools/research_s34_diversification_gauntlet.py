"""S34 Diversification Gauntlet — sizing + interaction + premium sleeve.

A  feature interaction (ikili+uclu) holdout + no-overlap + mdd + risk_adj
B  conviction-weighted sizing politikalari (flat/score/sleeve/premium/deep7d)
C  interaction-tetikli sleeve (iki-sinyal-birden = ekstra birim)
D  15x guvenlik: worst weighted loss + hesap-sim (compound, %10 unit)

hour17 200K composite baz (fixed CT esikleri — TRAIN'de secilmisti).
FEE=5. Cikti: reports/research/s34/S34_DIVERSIFICATION.json + .md
"""
from __future__ import annotations
import itertools, json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_DIVERSIFICATION.json"; OM = OUT / "S34_DIVERSIFICATION.md"
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
def book_imb(c, ts):
    r = c.execute("SELECT book_imbalance,ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None and (ts - int(r[1])) <= 5 * 60_000 else None
def ofir(c, lo, hi):
    r = c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or r[0] is None: return None, None
    b, se = float(r[0]), float(r[1]); t = b + se
    whale = (float(r[2]) / int(r[3])) if r[3] else None
    return ((b - se) / t if t > 0 else 0.0), whale
def nextfund(c, ts):
    r = c.execute("SELECT next_funding_time_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND next_funding_time_ms IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return int(r[0]) if r and r[0] else None
def hod(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
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


def mdd(vals):
    eq = 0.0; peak = 0.0; d = 0.0
    for v in vals:
        eq += v; peak = max(peak, eq); d = min(d, eq - peak)
    return round(d, 1)


def stat(g, label="", months=None, fee=FEE):
    m = months or TM
    if not g: return {"label": label, "n": 0}
    net = [x - fee for x in g]; n = len(net); w = sum(1 for x in net if x > 0)
    a = sum(net) / n; dd = mdd(net)
    tot = sum(net)
    return {"label": label, "n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(tot, 0), "per_month": round(n / m, 1), "worst": round(min(net), 1),
            "tail_n": sum(1 for x in net if x <= -100), "mdd": dd,
            "risk_adj": round(tot / max(abs(dd), 50.0), 2), "mc_p": mcp(net, a)}


def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-38s N=0" % k[:38]); return
    print("    %-38s N=%-4d /mo=%-5.1f WR=%-6s avg=%-7s TOT=%-8s worst=%-8s mdd=%-8s RA=%-6s mc=%s"
          % (k[:38], v["n"], v.get("per_month", 0), str(v["wr"]) + "%", str(v["avg"]),
             str(v.get("total")), str(v.get("worst")), str(v.get("mdd")),
             str(v.get("risk_adj")), v.get("mc_p", "?")))


def noov(pairs, hold=HOLD):
    busy = -1; o = []
    for ts, v in sorted(pairs):
        if ts >= busy: o.append((ts, v)); busy = ts + hold
    return o


def feats(conn, m, ts, rn):
    sk = lsum(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts)
    of, whale = ofir(conn, ts - 5 * 60_000, ts); e = ep(m, ts)
    shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, e[1] * 0.98, e[1])) if e else 0
    return {"sync": sk / rn if rn > 0 else 0, "rv": rv5(conn, ts),
            "d24": lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000),
            "ofi": of, "be": lmax(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / rn if rn > 0 else 0,
            "imb": book_imb(conn, ts), "shelf": shelf, "whale": whale}


def hits(f):
    return {"sync": f["sync"] >= CT["sync"], "rv": f["rv"] is not None and f["rv"] >= CT["rv"],
            "d24": f["d24"] >= CT["d24"], "ofi": f["ofi"] is not None and f["ofi"] >= 0,
            "be": CT["be_lo"] <= f["be"] < CT["be_hi"], "imb": f["imb"] is not None and f["imb"] <= CT["imb"],
            "shelf": f["shelf"] >= CT["shelf"], "whale_lo": f["whale"] is not None and f["whale"] < CT["whale"]}


def score7(f):
    h = hits(f); return sum(1 for k in ("sync", "rv", "d24", "ofi", "be", "imb", "shelf") if h[k])


def build(conn, m, now, start):
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(200_000.0,), accel_window_sec=30)
    ev = []
    for a in ancs:
        ts = int(a.anchor_ts_ms); rn = float(a.running_notional)
        if rn < 200_000 or m.at_or_after(ts) is None: continue
        b4 = mbps(conn, "BTCUSDT", ts, 4 * 3600_000) or 0
        b7 = mbps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0
        if ((mbps(conn, "ETHUSDT", ts, 3600_000) or 0) > 20 and b4 > 50) or sxn(ts) == "EUROPE" \
                or not (b4 < 0 or b7 < 0) or hod(ts) < 17:
            continue
        f = feats(conn, m, ts, rn); nf = nextfund(conn, ts)
        m2f = ((nf - ts) / 60_000) if nf else None
        y = lret(m, ts, HOLD)
        if y is None: continue
        ev.append({"ts": ts, "rn": rn, "f": f, "h": hits(f), "s7": score7(f), "b7": b7,
                   "hour": hod(ts), "veto": (m2f is not None and m2f < 60), "y": y})
    ev.sort(key=lambda x: x["ts"])
    return ev


def run_A(ev):
    print("\n=== A: interaction ikili+uclu (holdout + noov) ===")
    R = {}
    n = len(ev); cut = int(n * TRAIN); te = ev[cut:]; tem = TM * (1 - TRAIN)
    keys = ["sync", "rv", "shelf", "be", "whale_lo"]
    combos = list(itertools.combinations(keys, 2)) + list(itertools.combinations(keys, 3))
    for c in combos:
        name = "+".join(c)
        sub = [e for e in ev if all(e["h"].get(k) for k in c) and not e["veto"]]
        if len(sub) < 8: continue
        R[f"A_{name}_full"] = stat([e["y"] for e in sub], name, TM)
        R[f"A_{name}_TEST"] = stat([e["y"] for e in te if all(e["h"].get(k) for k in c) and not e["veto"]], name + " TEST", tem)
        nv = noov([(e["ts"], e["y"]) for e in sub])
        R[f"A_{name}_noov"] = stat([v for _, v in nv], name + " noov", TM)
        ps(f"A_{name}_full", R[f"A_{name}_full"]); ps(f"A_{name}_TEST", R[f"A_{name}_TEST"])
    # min3 (sync/shelf/whale herhangi 2 / 3)
    def min3(e): return sum(1 for k in ("sync", "shelf", "whale_lo") if e["h"].get(k))
    for K in (2, 3):
        sub = [e for e in ev if min3(e) >= K and not e["veto"]]
        R[f"A_min3_ge{K}_full"] = stat([e["y"] for e in sub], f"min3>={K}", TM)
        R[f"A_min3_ge{K}_TEST"] = stat([e["y"] for e in te if min3(e) >= K and not e["veto"]], f"min3>={K} TEST", tem)
        nv = noov([(e["ts"], e["y"]) for e in sub])
        R[f"A_min3_ge{K}_noov"] = stat([v for _, v in nv], f"min3>={K} noov", TM)
        ps(f"A_min3_ge{K}_full", R[f"A_min3_ge{K}_full"]); ps(f"A_min3_ge{K}_noov", R[f"A_min3_ge{K}_noov"])
    return R


def _apply_policy(seq, unit_fn):
    """seq: chronological no-overlap [(ts, net_bps, e)] -> weighted stats."""
    wpnl = []; units = []
    for ts, net, e in seq:
        u = unit_fn(e)
        units.append(u); wpnl.append(u * net)
    if not wpnl: return {"n": 0}
    tot = sum(wpnl); dd = mdd(wpnl); tu = sum(units)
    worst = min(wpnl)
    # hesap-sim: unit basina %10 notional, compound (1x bps pass-through)
    eqx = 1.0; peak = 1.0; mdd_pct = 0.0
    for (ts, net, e), u in zip(seq, units):
        eqx *= (1.0 + u * 0.10 * net / 1e4 * 15.0)  # 15x kaldiracli notional
        peak = max(peak, eqx); mdd_pct = min(mdd_pct, eqx / peak - 1.0)
    return {"n": len(wpnl), "units": tu, "w_total": round(tot, 0),
            "per_unit_avg": round(tot / tu, 1) if tu else None,
            "w_worst": round(worst, 1), "w_mdd": dd,
            "risk_adj": round(tot / max(abs(dd), 50.0), 2),
            "acct15x_final": round(eqx, 3), "acct15x_mdd_pct": round(100 * mdd_pct, 1),
            "max_unit": max(units)}


def run_BC(ev):
    print("\n=== B/C: sizing politikalari (no-overlap, score>=2 admit) ===")
    R = {}
    admitted = [e for e in ev if e["s7"] >= 2 and not e["veto"]]
    nv = noov([(e["ts"], e) for e in admitted])
    seq = [(ts, e["y"] - FEE, e) for ts, e in nv]
    def min3(e): return sum(1 for k in ("sync", "shelf", "whale_lo") if e["h"].get(k))
    pols = {
        "B_flat_1u": lambda e: 1.0,
        "B_unit_eq_score": lambda e: float(e["s7"]),
        "B_sleeve_123": lambda e: 1.0 if e["s7"] <= 3 else (2.0 if e["s7"] == 4 else 3.0),
        "B_premium_only_s4_2u": lambda e: 2.0 if e["s7"] >= 4 else 0.0,
        "B_min3_sleeve": lambda e: 1.0 if min3(e) < 2 else (2.0 if min3(e) == 2 else 3.0),
        "B_deep7d_s4_boost": lambda e: (2.0 if (e["b7"] < -300 and e["s7"] >= 4) else 1.0),
        "C_rv_shelf_trigger": lambda e: 2.0 if (e["h"]["rv"] and e["h"]["shelf"]) else 1.0,
        "C_rv_whale_trigger": lambda e: 2.0 if (e["h"]["rv"] and e["h"]["whale_lo"]) else 1.0,
        "C_double_trigger_3u": lambda e: 3.0 if (e["h"]["rv"] and e["h"]["shelf"] and e["h"]["whale_lo"]) else (2.0 if (e["h"]["rv"] and e["h"]["shelf"]) else 1.0),
    }
    for name, fn in pols.items():
        r = _apply_policy([s for s in seq if fn(s[2]) > 0], fn)
        R[name] = r
        if r.get("n"):
            print("    %-26s N=%-3d units=%-5.0f wTOT=%-8s perU=%-6s wWorst=%-7s wMDD=%-8s RA=%-6s acct15x=%-6s mddPct=%s%%"
                  % (name, r["n"], r["units"], r["w_total"], r["per_unit_avg"], r["w_worst"],
                     r["w_mdd"], r["risk_adj"], r["acct15x_final"], r["acct15x_mdd_pct"]))
    # holdout: ayni politikalar sadece TEST bolumunde
    cut_ts = admitted[int(len(admitted) * TRAIN)]["ts"] if admitted else 0
    seq_te = [s for s in seq if s[0] >= cut_ts]
    for name, fn in pols.items():
        r = _apply_policy([s for s in seq_te if fn(s[2]) > 0], fn)
        R[name + "_TEST"] = r
    print("    -- TEST split ozeti --")
    for name in pols:
        r = R[name + "_TEST"]
        if r.get("n"):
            print("    %-26s N=%-3d wTOT=%-8s perU=%-6s wMDD=%-8s RA=%s"
                  % (name + "_TEST", r["n"], r["w_total"], r["per_unit_avg"], r["w_mdd"], r["risk_adj"]))
    return R


def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Diversification Gauntlet ===")
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
        m = load_mark_index(conn, "ETHUSDT")
        print("build..."); ev = build(conn, m, now, start)
        span = [e["ts"] for e in ev]; TM = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R = {}
        R["A"] = run_A(ev)
        R["BC"] = run_BC(ev)
    meta = {"n": len(ev), "months": round(TM, 2)}
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps({"results": R, "meta": meta}, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Diversification Gauntlet", "",
             f"> hour17 200K composite {len(ev)} event {TM:.1f} ay. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for q, sec in R.items():
        lines += [f"## {q}", ""]
        for k, v in sec.items():
            if isinstance(v, dict) and v.get("n", 0) > 0 and "wr" in v:
                lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s worst=%s mdd=%s RA=%s mc_p=%s"
                             % (k, v["n"], v.get("per_month", 0), v["wr"], v["avg"], v.get("total"),
                                v.get("worst"), v.get("mdd"), v.get("risk_adj"), v.get("mc_p", "?")))
            elif isinstance(v, dict) and v.get("n", 0) > 0 and "w_total" in v:
                lines.append("- **%s**: N=%d units=%s wTOT=%s perU=%s wWorst=%s wMDD=%s RA=%s acct15x=%s (mdd %s%%)"
                             % (k, v["n"], v.get("units"), v["w_total"], v["per_unit_avg"],
                                v["w_worst"], v["w_mdd"], v["risk_adj"], v.get("acct15x_final"), v.get("acct15x_mdd_pct")))
        lines.append("")
    lines += ["---", "*Script: tools/research_s34_diversification_gauntlet.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
