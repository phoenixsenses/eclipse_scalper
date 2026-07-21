"""S34 rv_5m Staleness Fix + Re-Validation.

PROBLEM: vol_state tablosu 2026-06-05 19:00'da durdu (producer silinmis).
rv_5m sorgusu 'son satir' dondugu icin 5 Haziran sonrasi TUM event'ler
bayat rv=0.253 aldi -> rv hit her zaman True -> composite skorlar sisti.
TEST split (son %30) tam bu doneme denk geliyor.

BU SCRIPT:
1. rv_proxy'yi mark_prices'tan hesaplar (5m pencerede 1m log-return RMS)
2. Bayat olmayan donemde vol_state rv ile korelasyonunu olcer (sanity)
3. Bayat donemde stale-hit vs proxy-hit farkini sayar (etki)
4. Duzeltilmis rv ile ana sonuclari yeniden kosar:
   - composite s7>=3/4 (full + TEST)
   - interaction rv+shelf / rv+whale_lo / rv+shelf+whale_lo (full + TEST)
   - L1 tarifi (rv'siz — kontrol, degismemeli)
Cikti: reports/research/s34/S34_RV_STALE_FIX.json + .md
"""
from __future__ import annotations
import json, math, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_RV_STALE_FIX.json"; OM = OUT / "S34_RV_STALE_FIX.md"
LB = 400 * 24 * 3600_000; FEE = 5.0; MC = 500; HOLD = 6 * 3600_000; TM = 4.5; TRAIN = 0.70
STALE_TS = None  # runtime'da bulunur
CT = {"sync": 0.5421, "d24": 5.0, "be_lo": 0.2195, "be_hi": 2.0,
      "shelf": 2_775_000.0, "whale": 6440.0}
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
def rv_stale(c, ts):
    r = c.execute("SELECT rv_5m FROM vol_state WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    return float(r[0]) if r and r[0] is not None else None
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
def dowf(ts): return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).weekday()
def sxn(ts):
    h = hod(ts); return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")
def ep(m, ts):
    r = m.at_or_after(ts); return (int(r[0]), float(r[1])) if r and float(r[1]) > 0 else None
def lret(m, ts, hold):
    e = ep(m, ts)
    if not e: return None
    r = m.at_or_before(ts + hold); return (float(r[1]) - e[1]) / e[1] * 1e4 if r else None


def rv_proxy(m, ts):
    """Mark-price bazli rv: 5m pencerede 1m log-return RMS (fraction)."""
    px = []
    for k in range(5, -1, -1):
        r = m.at_or_before(ts - k * 60_000)
        if r is None: return None
        px.append(float(r[1]))
    rets = [math.log(px[i + 1] / px[i]) for i in range(5) if px[i] > 0]
    if len(rets) < 5: return None
    return math.sqrt(sum(x * x for x in rets))


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
            "worst": round(min(net), 1), "mc_p": mcp(net, a)}
def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-44s N=0" % k[:44]); return
    print("    %-44s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-8s mc=%s"
          % (k[:44], v["n"], v.get("per_month", 0), str(v["wr"]) + "%", str(v["avg"]),
             str(v.get("total")), v.get("mc_p", "?")))
def med(x):
    s = sorted(v for v in x if v is not None); return s[len(s) // 2] if s else None


def main():
    global TM, STALE_TS
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 rv Stale Fix Validation ===")
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
        r = conn.execute("SELECT MAX(ts_ms) FROM vol_state WHERE symbol='ETHUSDT'").fetchone()
        STALE_TS = int(r[0])
        print(f"  vol_state son satir: {datetime.fromtimestamp(STALE_TS/1000, tz=timezone.utc)}")
        m = load_mark_index(conn, "ETHUSDT")
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
            y = lret(m, ts, HOLD)
            if y is None: continue
            of, whale = ofir(conn, ts - 5 * 60_000, ts); e0 = ep(m, ts)
            shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, e0[1] * 0.98, e0[1])) if e0 else 0
            sk = lsum(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts)
            nf = nextfund(conn, ts); m2f = ((nf - ts) / 60_000) if nf else None
            ev.append({"ts": ts, "y": y, "dow": dowf(ts), "veto": (m2f is not None and m2f < 60),
                       "rvs": rv_stale(conn, ts), "rvp": rv_proxy(m, ts),
                       "h": {"sync": (sk / rn if rn > 0 else 0) >= CT["sync"],
                             "d24": lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000) >= CT["d24"],
                             "ofi": of is not None and of >= 0,
                             "be": CT["be_lo"] <= (lmax(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / rn if rn > 0 else 0) < CT["be_hi"],
                             "shelf": shelf >= CT["shelf"],
                             "whale_lo": whale is not None and whale < CT["whale"]}})
        ev.sort(key=lambda x: x["ts"])
        span = [e["ts"] for e in ev]; TM = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        R = {}
        # 1) korelasyon (bayat olmayan donem)
        fresh = [e for e in ev if e["ts"] <= STALE_TS and e["rvs"] is not None and e["rvp"] is not None]
        if len(fresh) >= 10:
            xs = [e["rvs"] for e in fresh]; ys = [e["rvp"] for e in fresh]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
            corr = cov / (sx * sy) if sx > 0 and sy > 0 else None
            R["corr_fresh"] = {"n": len(fresh), "pearson": round(corr, 3) if corr is not None else None}
            print(f"  taze donem korelasyon (vol_state vs proxy): r={corr:.3f} N={len(fresh)}")
        # 2) etki: bayat donemde hit farki
        cut = int(len(ev) * TRAIN)
        tr = ev[:cut]
        thr_p = med([e["rvp"] for e in tr])
        R["proxy_thr_train_median"] = thr_p
        stale_ev = [e for e in ev if e["ts"] > STALE_TS]
        stale_hit = sum(1 for e in stale_ev if e["rvs"] is not None and e["rvs"] >= 0.0304)
        proxy_hit = sum(1 for e in stale_ev if e["rvp"] is not None and e["rvp"] >= thr_p)
        flips = sum(1 for e in stale_ev
                    if (e["rvs"] is not None and e["rvs"] >= 0.0304) != (e["rvp"] is not None and e["rvp"] >= thr_p))
        R["stale_impact"] = {"stale_events": len(stale_ev), "stale_rv_hits": stale_hit,
                             "proxy_rv_hits": proxy_hit, "flipped": flips}
        print(f"  bayat donem: {len(stale_ev)} event, stale-hit={stale_hit}, proxy-hit={proxy_hit}, flip={flips}")
        # 3) duzeltilmis sonuclar
        te = ev[cut:]; tem = TM * (1 - TRAIN)
        def s7(e, use_proxy):
            rvh = (e["rvp"] is not None and e["rvp"] >= thr_p) if use_proxy \
                else (e["rvs"] is not None and e["rvs"] >= 0.0304)
            return sum([e["h"]["sync"], rvh, e["h"]["d24"], e["h"]["ofi"], e["h"]["be"], e["h"]["shelf"]])
        def rvh_p(e): return e["rvp"] is not None and e["rvp"] >= thr_p
        def rvh_s(e): return e["rvs"] is not None and e["rvs"] >= 0.0304
        print("\n  -- duzeltilmis (proxy) vs bayat (stale) --")
        for K in (3, 4):
            for lbl, up in (("stale", False), ("proxy", True)):
                key = f"s7_ge{K}_{lbl}"
                R[key] = stat([e["y"] for e in ev if s7(e, up) >= K and not e["veto"]], key, TM)
                R[key + "_TEST"] = stat([e["y"] for e in te if s7(e, up) >= K and not e["veto"]], key + " TEST", tem)
                ps(key, R[key]); ps(key + "_TEST", R[key + "_TEST"])
        for lbl, rvh in (("stale", rvh_s), ("proxy", rvh_p)):
            for combo, cond in (("rv+shelf", lambda e, rv=rvh: rv(e) and e["h"]["shelf"]),
                                ("rv+whale", lambda e, rv=rvh: rv(e) and e["h"]["whale_lo"]),
                                ("rv+shelf+whale", lambda e, rv=rvh: rv(e) and e["h"]["shelf"] and e["h"]["whale_lo"])):
                key = f"I_{combo}_{lbl}"
                R[key] = stat([e["y"] for e in ev if cond(e) and not e["veto"]], key, TM)
                R[key + "_TEST"] = stat([e["y"] for e in te if cond(e) and not e["veto"]], key + " TEST", tem)
                ps(key, R[key]); ps(key + "_TEST", R[key + "_TEST"])
        # L1 kontrol (rv'siz)
        busy = -1; sub = []
        for e in ev:
            if s7(e, True) >= 2 and not e["veto"] and e["dow"] != 0 and e["ts"] >= busy:
                sub.append(e); busy = e["ts"] + HOLD
        R["L1_notMon_proxy"] = stat([e["y"] for e in sub], "L1 proxy s>=2", TM); ps("L1_notMon_proxy", R["L1_notMon_proxy"])
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps({"results": R, "meta": {"n": len(ev), "months": round(TM, 2),
                                                     "stale_ts": STALE_TS}}, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 rv Stale Fix Validation", "",
             f"> vol_state {datetime.fromtimestamp(STALE_TS/1000, tz=timezone.utc):%Y-%m-%d %H:%M} sonrasi bayat. "
             f"{len(ev)} event {TM:.1f} ay. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for k, v in R.items():
        if isinstance(v, dict) and v.get("n", 0) > 0 and "wr" in v:
            lines.append("- **%s**: N=%d WR=%.1f%% avg=%+.1f TOT=%s mc_p=%s"
                         % (k, v["n"], v["wr"], v["avg"], v.get("total"), v.get("mc_p", "?")))
        elif isinstance(v, dict):
            lines.append(f"- **{k}**: {json.dumps(v, default=str)}")
        else:
            lines.append(f"- **{k}**: {v}")
    lines += ["", "---", "*Script: tools/research_s34_rv_stale_fix_validation.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
