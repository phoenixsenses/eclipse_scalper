"""S34 Pre-Cascade Predictor — Faz 3: durust tradeable EV.

Zaman cizgisi 5dk adimlarla taranir (book kapsami, 11 Nis ->). Her ornekte
SADECE o ana kadar bilinen ozellikler: rv, ret_1h, spot-basis, OFI-10m,
cross-liq stress. Cascade bilgisi girise ASLA sizmez:
- Tetik esikleri TRAIN doneminin tercile'lerinden.
- Her tetik islem sayilir (cascade gelmese de) -> false-positive maliyeti EV'de.
- Aktif cascade donemi (anchor sonrasi 30dk) 'pre' orneklerinden haric.

Olcumler (TEST doneminde):
- P(cascade<=10m | tetik) vs taban orani (lift)
- LONG@tetik 6h hold EV (bounce'i erken yakalama)
- SHORT@tetik 30dk cover EV (cascade dususunu hasat)
- 30dk cooldown (no-overlap)

Cikti: reports/research/s34/S34_PRECASCADE.json + .md
"""
from __future__ import annotations
import bisect, json, math, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"
OJ = ROOT / "reports" / "research" / "s34" / "S34_PRECASCADE.json"
OM = ROOT / "reports" / "research" / "s34" / "S34_PRECASCADE.md"
STEP = 5 * 60_000; FEE = 5.0; MC = 500; TRAIN = 0.70; HOLD6 = 6 * 3600_000
COOLDOWN = 30 * 60_000
random.seed(42)


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)


def stat(g):
    net = [x - FEE for x in g if x is not None]
    if not net: return {"n": 0}
    n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    return {"n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(sum(net), 0), "worst": round(min(net), 1), "mc_p": mcp(net, a)}


def rv_proxy(m, ts):
    px = []
    for k in range(5, -1, -1):
        r = m.at_or_before(ts - k * 60_000)
        if r is None: return None
        px.append(float(r[1]))
    rets = [math.log(px[i + 1] / px[i]) for i in range(5) if px[i] > 0]
    return math.sqrt(sum(x * x for x in rets)) if len(rets) == 5 else None


def pct(vals, p):
    s = sorted(v for v in vals if v is not None)
    return s[int(p * (len(s) - 1))] if s else None


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Pre-Cascade Predictor (Faz 3) ===")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    now = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    bs = conn.execute("SELECT MIN(ts_ms) FROM book_ticker WHERE symbol='ETHUSDT'").fetchone()[0]
    start = int(bs) + 3600_000
    m = load_mark_index(conn, "ETHUSDT")
    ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start - 3600_000, now),
                               bucket_sec=300, min_gap_sec=900, thresholds=(100_000.0,), accel_window_sec=30)
    anc_ts = sorted(int(a.anchor_ts_ms) for a in ancs if float(a.running_notional) >= 100_000)
    print(f"  anchor: {len(anc_ts)}, tarama: {datetime.fromtimestamp(start/1000, tz=timezone.utc):%m-%d} -> simdi, adim 5dk")

    def next_anchor_within(ts, win):
        i = bisect.bisect_right(anc_ts, ts)
        return i < len(anc_ts) and anc_ts[i] <= ts + win

    def last_anchor_before(ts):
        i = bisect.bisect_right(anc_ts, ts) - 1
        return anc_ts[i] if i >= 0 else None

    samples = []
    ts = start
    n_scan = 0
    while ts < now - HOLD6:
        n_scan += 1
        la = last_anchor_before(ts)
        if la is not None and ts - la < COOLDOWN:  # aktif cascade donemi degil
            ts += STEP; continue
        rv = rv_proxy(m, ts)
        e0 = m.at_or_before(ts); e1h = m.at_or_before(ts - 3600_000)
        ret1h = ((float(e0[1]) - float(e1h[1])) / float(e1h[1]) * 1e4) if (e0 and e1h and float(e1h[1]) > 0) else None
        sp = conn.execute("SELECT spot_price, ts_ms FROM spot_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
        basis = ((float(e0[1]) - float(sp[0])) / float(sp[0]) * 1e4) if (e0 and sp and sp[0] and (ts - int(sp[1])) <= 10 * 60_000) else None
        o = conn.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), SUM(notional) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (ts - 600_000, ts)).fetchone()
        ofi = ((2 * float(o[0]) - float(o[1])) / float(o[1])) if (o and o[1] and float(o[1]) > 0) else None
        srow = conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol IN ('ETHUSDT','BTCUSDT') AND side='SELL' AND ts_ms>=? AND ts_ms<?", (ts - 600_000, ts)).fetchone()
        stress = float(srow[0]) if srow else 0.0
        y6 = None
        if e0:
            x = m.at_or_before(ts + HOLD6)
            y6 = ((float(x[1]) - float(e0[1])) / float(e0[1]) * 1e4) if x else None
        y30s = None
        if e0:
            x = m.at_or_before(ts + 1800_000)
            y30s = (-(float(x[1]) - float(e0[1])) / float(e0[1]) * 1e4) if x else None
        samples.append({"ts": ts, "rv": rv, "ret1h": ret1h, "basis": basis, "ofi": ofi,
                        "stress": stress, "casc10": int(next_anchor_within(ts, 600_000)),
                        "casc30": int(next_anchor_within(ts, 1800_000)),
                        "y6_long": y6, "y30_short": y30s})
        if len(samples) % 4000 == 0: print(f"    ornek {len(samples)}")
        ts += STEP
    print(f"  ornek: {len(samples)} (aktif-cascade haric)")

    cut = int(len(samples) * TRAIN); tr, te = samples[:cut], samples[cut:]
    thr = {"rv": pct([s["rv"] for s in tr], 0.66), "ret1h": pct([s["ret1h"] for s in tr], 0.33),
           "basis": pct([s["basis"] for s in tr], 0.33), "ofi": pct([s["ofi"] for s in tr], 0.33),
           "stress": pct([s["stress"] for s in tr], 0.66)}
    print(f"  esikler (TRAIN tercile): { {k: (round(v,6) if isinstance(v,float) else v) for k,v in thr.items()} }")

    def score(s):
        sc = 0
        if s["rv"] is not None and thr["rv"] is not None and s["rv"] >= thr["rv"]: sc += 1
        if s["ret1h"] is not None and thr["ret1h"] is not None and s["ret1h"] <= thr["ret1h"]: sc += 1
        if s["basis"] is not None and thr["basis"] is not None and s["basis"] <= thr["basis"]: sc += 1
        if s["ofi"] is not None and thr["ofi"] is not None and s["ofi"] <= thr["ofi"]: sc += 1
        if s["stress"] >= (thr["stress"] or 0): sc += 1
        return sc

    R = {"meta": {"n_samples": len(samples), "n_anchors": len(anc_ts)},
         "thresholds": {k: v for k, v in thr.items()}}
    base10 = sum(s["casc10"] for s in te) / len(te); base30 = sum(s["casc30"] for s in te) / len(te)
    R["base_rate_TEST"] = {"casc10": round(100 * base10, 2), "casc30": round(100 * base30, 2)}
    print(f"\n  TEST taban orani: cascade<=10m {100*base10:.2f}%  <=30m {100*base30:.2f}%")
    print("\n=== Tetik degerlendirme (TEST, 30dk cooldown) ===")
    for K in (3, 4, 5):
        trig = [s for s in te if score(s) >= K]
        # cooldown no-overlap
        busy = -1; tt = []
        for s in trig:
            if s["ts"] >= busy: tt.append(s); busy = s["ts"] + COOLDOWN
        if not tt:
            print(f"  K>={K}: tetik yok"); R[f"K{K}"] = {"n_trig": 0}; continue
        p10 = sum(s["casc10"] for s in tt) / len(tt); p30 = sum(s["casc30"] for s in tt) / len(tt)
        ev_long = stat([s["y6_long"] for s in tt])
        ev_short = stat([s["y30_short"] for s in tt])
        per_day = len(tt) / max(1.0, (te[-1]["ts"] - te[0]["ts"]) / 86_400_000)
        R[f"K{K}"] = {"n_trig": len(tt), "per_day": round(per_day, 1),
                      "p_casc10": round(100 * p10, 1), "lift10": round(p10 / base10, 1) if base10 else None,
                      "p_casc30": round(100 * p30, 1), "lift30": round(p30 / base30, 1) if base30 else None,
                      "LONG_6h": ev_long, "SHORT_30m": ev_short}
        print("  K>=%d: tetik=%-4d (%.1f/gun)  P(casc10)=%.1f%% lift=%.1fx  P(casc30)=%.1f%% lift=%.1fx"
              % (K, len(tt), per_day, 100 * p10, p10 / base10 if base10 else 0, 100 * p30, p30 / base30 if base30 else 0))
        print("        LONG@tetik 6h : N=%-4d WR=%-6s avg=%-7s TOT=%-8s mc=%s"
              % (ev_long.get("n", 0), str(ev_long.get("wr")) + "%", ev_long.get("avg"), ev_long.get("total"), ev_long.get("mc_p")))
        print("        SHORT@tetik30m: N=%-4d WR=%-6s avg=%-7s TOT=%-8s mc=%s"
              % (ev_short.get("n", 0), str(ev_short.get("wr")) + "%", ev_short.get("avg"), ev_short.get("total"), ev_short.get("mc_p")))
    conn.close()
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Pre-Cascade Predictor (Faz 3)", "",
             f"> {len(samples)} ornek (5dk adim, aktif-cascade haric), {len(anc_ts)} anchor. "
             f"{datetime.now(timezone.utc):%Y-%m-%d}", "",
             f"- TEST taban orani: casc<=10m {R['base_rate_TEST']['casc10']}%, <=30m {R['base_rate_TEST']['casc30']}%", ""]
    for K in (3, 4, 5):
        v = R.get(f"K{K}", {})
        if v.get("n_trig"):
            L, S = v["LONG_6h"], v["SHORT_30m"]
            lines.append("- **K>=%d**: %d tetik (%.1f/gun) P(casc10)=%s%% lift=%sx | LONG6h avg=%s mc=%s | SHORT30m avg=%s mc=%s"
                         % (K, v["n_trig"], v["per_day"], v["p_casc10"], v["lift10"],
                            L.get("avg"), L.get("mc_p"), S.get("avg"), S.get("mc_p")))
    lines += ["", "---", "*Script: tools/s34_precascade_predictor.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
