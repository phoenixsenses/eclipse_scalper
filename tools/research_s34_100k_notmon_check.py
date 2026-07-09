"""S34 100K route + notMonday + proxy-rv hizli kontrol.

Soru: Pazartesi blogu ve duzeltilmis rv, 100K frekans-genisletme route'unda
da tutuyor mu? (100-200K mini + hour17 + regime + composite)
Cikti: reports/research/s34/S34_100K_NOTMON.json + .md
"""
from __future__ import annotations
import json, math, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_liquidations, load_mark_index, reconstruct_anchors
from ami.storage import production as PR
from ami.storage import research_reader as RR

DB = ROOT / "data" / "microstructure.db"; OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "S34_100K_NOTMON.json"; OM = OUT / "S34_100K_NOTMON.md"
LB = 400 * 24 * 3600_000; FEE = 5.0; MC = 500; HOLD = 6 * 3600_000; TM = 4.5; TRAIN = 0.70
CT = {"sync": 0.5421, "rvp": 0.0026337, "d24": 5.0, "be_lo": 0.2195, "be_hi": 2.0,
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
def ofir(c, lo, hi):
    """Direct-SQL oracle -- kept as the parity reference for
    `window_agg_trades_ofi_whale` below (BATCH-STORAGE-ROTATION-RETENTION-
    SECOND-RESEARCH-CONSUMER-INTEGRATION-V1). No longer called by main();
    the reader-backed path is used instead."""
    r = c.execute("SELECT SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END),SUM(notional),COUNT(*) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?", (lo, hi)).fetchone()
    if not r or r[0] is None: return None, None
    b, se = float(r[0]), float(r[1]); t = b + se
    whale = (float(r[2]) / int(r[3])) if r[3] else None
    return ((b - se) / t if t > 0 else 0.0), whale


def window_agg_trades_ofi_whale(root, symbol, lo, hi):
    """Reader-backed replacement for `ofir` -- fetches raw (notional,
    is_buyer_maker) rows over [lo, hi) via the unified research reader
    (transparently archive/SQLite/hybrid) and reduces them in Python,
    replicating `ofir`'s SQL field-by-field (buy/sell notional summed
    independently from the raw per-row total, matching the original
    4-column SELECT) rather than assuming buy+sell==total."""
    plan = RR.plan_read(root, table="agg_trades", symbol=symbol, start_ms=lo, end_ms=hi)
    result = RR.execute_read(plan, columns=("notional", "is_buyer_maker"))
    buy = se = total = 0.0
    count = 0
    for notional, is_buyer_maker in result.iter_rows():
        count += 1
        total += notional
        if is_buyer_maker == 0:
            buy += notional
        elif is_buyer_maker == 1:
            se += notional
    if count == 0:
        return None, None
    t = buy + se
    whale = total / count
    return ((buy - se) / t if t > 0 else 0.0), whale
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
def mdd(vals):
    eq = 0.0; peak = 0.0; d = 0.0
    for v in vals:
        eq += v; peak = max(peak, eq); d = min(d, eq - peak)
    return round(d, 1)
def stat(g, label="", months=None, fee=FEE):
    m = months or TM
    if not g: return {"label": label, "n": 0}
    net = [x - fee for x in g]; n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    dd = mdd(net)
    return {"label": label, "n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
            "total": round(sum(net), 0), "per_month": round(n / m, 1),
            "worst": round(min(net), 1), "mdd": dd, "mc_p": mcp(net, a)}
def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-40s N=0" % k[:40]); return
    print("    %-40s N=%-4d /mo=%-5.1f WR=%-6s avg=%-8s TOT=%-8s worst=%-8s mdd=%-8s mc=%s"
          % (k[:40], v["n"], v.get("per_month", 0), str(v["wr"]) + "%", str(v["avg"]),
             str(v.get("total")), str(v.get("worst")), str(v.get("mdd")), v.get("mc_p", "?")))


def main():
    global TM
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 100K + notMon + proxy-rv check ===")
    root, _root_source = PR.resolve_production_root()
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA cache_size=-200000"); conn.execute("PRAGMA temp_store=MEMORY")
        now = int(datetime.now(tz=timezone.utc).timestamp() * 1000); start = now - LB
        m = load_mark_index(conn, "ETHUSDT")
        ancs = reconstruct_anchors(load_liquidations(conn, "ETHUSDT", "SELL", start, now),
                                   bucket_sec=300, min_gap_sec=900, thresholds=(100_000.0,), accel_window_sec=30)
        ev = []
        for a in ancs:
            ts = int(a.anchor_ts_ms); rn = float(a.running_notional)
            if rn < 100_000 or m.at_or_after(ts) is None: continue
            b4 = mbps(conn, "BTCUSDT", ts, 4 * 3600_000) or 0
            b7 = mbps(conn, "BTCUSDT", ts, 7 * 24 * 3600_000) or 0
            if ((mbps(conn, "ETHUSDT", ts, 3600_000) or 0) > 20 and b4 > 50) or sxn(ts) == "EUROPE" \
                    or not (b4 < 0 or b7 < 0) or hod(ts) < 17:
                continue
            y = lret(m, ts, HOLD)
            if y is None: continue
            of, whale = window_agg_trades_ofi_whale(root, "ETHUSDT", ts - 5 * 60_000, ts); e0 = ep(m, ts)
            shelf = _s(conn, "SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<? AND price>=? AND price<=?", (ts - 24 * 3600_000, ts, e0[1] * 0.98, e0[1])) if e0 else 0
            sk = lsum(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) + lsum(conn, "SOLUSDT", "SELL", ts - 10 * 60_000, ts)
            rvp = rv_proxy(m, ts)
            nf = nextfund(conn, ts); m2f = ((nf - ts) / 60_000) if nf else None
            s = sum([(sk / rn if rn > 0 else 0) >= CT["sync"],
                     rvp is not None and rvp >= CT["rvp"],
                     lcnt(conn, "ETHUSDT", "SELL", ts - 24 * 3600_000, ts - 300_000, 200_000) >= CT["d24"],
                     of is not None and of >= 0,
                     CT["be_lo"] <= (lmax(conn, "BTCUSDT", "SELL", ts - 10 * 60_000, ts) / rn if rn > 0 else 0) < CT["be_hi"],
                     shelf >= CT["shelf"],
                     whale is not None and whale < CT["whale"]])
            ev.append({"ts": ts, "y": y, "dow": dowf(ts), "s": s,
                       "veto": (m2f is not None and m2f < 60)})
        ev.sort(key=lambda x: x["ts"])
        span = [e["ts"] for e in ev]; TM = max(1.0, (span[-1] - span[0]) / 86_400_000 / 30.0)
        print(f"  events={len(ev)} months={TM:.2f}")
        cut = int(len(ev) * TRAIN); te = ev[cut:]; tem = TM * (1 - TRAIN)
        R = {}
        def noov_y(evs):
            busy = -1; o = []
            for e in evs:
                if e["ts"] >= busy: o.append(e["y"]); busy = e["ts"] + HOLD
            return o
        for name, cond in (
                ("all_s3", lambda e: e["s"] >= 3 and not e["veto"]),
                ("notMon_s3", lambda e: e["s"] >= 3 and not e["veto"] and e["dow"] != 0),
                ("notMon_s2", lambda e: e["s"] >= 2 and not e["veto"] and e["dow"] != 0),
                ("mon_only_s2", lambda e: e["s"] >= 2 and not e["veto"] and e["dow"] == 0)):
            R[name] = stat([e["y"] for e in ev if cond(e)], name, TM)
            R[name + "_TEST"] = stat([e["y"] for e in te if cond(e)], name + " TEST", tem)
            R[name + "_noov"] = stat(noov_y([e for e in ev if cond(e)]), name + " noov", TM)
            ps(name, R[name]); ps(name + "_TEST", R[name + "_TEST"]); ps(name + "_noov", R[name + "_noov"])
    OUT.mkdir(parents=True, exist_ok=True)
    OJ.write_text(json.dumps({"results": R, "meta": {"n": len(ev), "months": round(TM, 2)}},
                             indent=2, default=str), encoding="utf-8")
    lines = ["# S34 100K + notMonday + proxy-rv", "",
             f"> 100K mini universe {len(ev)} event {TM:.1f} ay. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for k, v in R.items():
        if isinstance(v, dict) and v.get("n", 0) > 0:
            lines.append("- **%s**: N=%d /ay=%.1f WR=%.1f%% avg=%+.1f TOT=%s worst=%s mdd=%s mc_p=%s"
                         % (k, v["n"], v.get("per_month", 0), v["wr"], v["avg"], v.get("total"),
                            v.get("worst"), v.get("mdd"), v.get("mc_p", "?")))
    lines += ["", "---", "*Script: tools/research_s34_100k_notmon_check.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
