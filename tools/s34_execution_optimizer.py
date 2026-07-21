"""S34 Execution Optimizer — Faz 4: gated event'lerde giris/cikis mekanigi.

E1 Fill gercekciligi: mark-varsayimi vs ASK'ten market girisi (spread maliyeti)
E2 Limit giris -10/-20/-30bps: L1 bid path'ten durust fill; DOLMAYANLAR EV'de
   (Q2 survivorship dersi) — EV/sinyal karsilastirmasi
E3 VWAP giris (ilk 5dk agg_trades VWAP) vs T0 market
E4 Dinamik TP: k x rv_bps (k=1/2/3) vs sabit 100/200/300 vs yok — mark path
E5 Vol-olcekli stop: m x rv_bps (m=1/2/3) vs sabit -150/-300 vs yok
E6 En iyi TP x stop kombinasyonu, TRAIN'de sec TEST'te raporla

Evren: store'daki GATED (hour17+regime) event'ler. Ana DB salt-okunur (path'ler).
Cikti: reports/research/s34/S34_EXECUTION_OPT.json + .md
"""
from __future__ import annotations
import json, random, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import load_mark_index

STORE = ROOT / "reports" / "research" / "s34" / "mechanism_store.sqlite"
DB = ROOT / "data" / "microstructure.db"
OJ = ROOT / "reports" / "research" / "s34" / "S34_EXECUTION_OPT.json"
OM = ROOT / "reports" / "research" / "s34" / "S34_EXECUTION_OPT.md"
FEE = 5.0; MC = 500; TRAIN = 0.70; HOLD6 = 6 * 3600_000; LIMIT_WIN = 15 * 60_000
random.seed(42)


def mcp(v, a):
    if len(v) < 4: return None
    r = random.Random(0)
    ct = sum(1 for _ in range(MC) if sum(r.choice([-1, 1]) * abs(x) for x in v) / len(v) >= a)
    return round(ct / MC, 3)


def stat(g, n_signals=None):
    net = [x - FEE for x in g if x is not None]
    if not net: return {"n": 0}
    n = len(net); w = sum(1 for x in net if x > 0); a = sum(net) / n
    out = {"n": n, "wr": round(100 * w / n, 1), "avg": round(a, 1),
           "total": round(sum(net), 0), "worst": round(min(net), 1), "mc_p": mcp(net, a)}
    if n_signals:
        out["ev_per_signal"] = round(sum(net) / n_signals, 1)
        out["fill_rate"] = round(100 * n / n_signals, 1)
    return out


def ps(k, v):
    if not v or v.get("n", 0) == 0:
        print("    %-38s N=0" % k[:38]); return
    ex = ""
    if "ev_per_signal" in v:
        ex = " EV/sig=%-6s fill=%s%%" % (v["ev_per_signal"], v["fill_rate"])
    print("    %-38s N=%-4d WR=%-6s avg=%-8s TOT=%-8s worst=%-8s mc=%s%s"
          % (k[:38], v["n"], str(v["wr"]) + "%", str(v["avg"]), str(v.get("total")),
             str(v.get("worst")), v.get("mc_p", "?"), ex))


def bt_at(c, ts, after=True):
    if after:
        r = c.execute("SELECT ts_ms, bid_price, ask_price, mid_price FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? ORDER BY ts_ms LIMIT 1", (ts,)).fetchone()
    else:
        r = c.execute("SELECT ts_ms, bid_price, ask_price, mid_price FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1", (ts,)).fetchone()
    if not r or abs(int(r[0]) - ts) > 60_000: return None
    return {"ts": int(r[0]), "bid": float(r[1]), "ask": float(r[2]), "mid": float(r[3])}


def limit_fill(c, t0, limit_px):
    r = c.execute("SELECT ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=? AND bid_price<=? ORDER BY ts_ms LIMIT 1",
                  (t0, t0 + LIMIT_WIN, limit_px)).fetchone()
    return int(r[0]) if r else None


def vwap5(c, t0):
    r = c.execute("SELECT SUM(price*notional), SUM(notional) FROM agg_trades WHERE symbol='ETHUSDT' AND ts_ms>? AND ts_ms<=?",
                  (t0, t0 + 300_000)).fetchone()
    if not r or not r[1] or float(r[1]) <= 0: return None
    return float(r[0]) / float(r[1])


def path_exit(m, entry_ts, entry_px, tp_bps=None, sl_bps=None, hold=HOLD6):
    """Mark path uzerinde ilk-dokunma TP/SL, yoksa time-exit. Net bps (fee'siz)."""
    path = m.slice_range(entry_ts, entry_ts + hold)
    for pts, px in path:
        ret = (float(px) - entry_px) / entry_px * 1e4
        if sl_bps is not None and ret <= -sl_bps: return -sl_bps
        if tp_bps is not None and ret >= tp_bps: return tp_bps
    if path:
        return (float(path[-1][1]) - entry_px) / entry_px * 1e4
    return None


def main():
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    print("=== S34 Execution Optimizer (Faz 4) ===")
    st = sqlite3.connect(f"file:{STORE}?mode=ro", uri=True); st.row_factory = sqlite3.Row
    ev = [dict(r) for r in st.execute("SELECT ts_ms, px_rv, y_6h FROM events WHERE is_event=1 AND gated=1.0 AND y_6h IS NOT NULL ORDER BY ts_ms")]
    st.close()
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    m = load_mark_index(conn, "ETHUSDT")
    print(f"  gated event: {len(ev)}")
    cut = int(len(ev) * TRAIN); cut_ts = ev[cut]["ts_ms"] if cut < len(ev) else 0
    R = {"meta": {"n": len(ev)}}

    # entry hazirliklari
    for e in ev:
        ts = e["ts_ms"]
        q = bt_at(conn, ts, after=True)
        e["q0"] = q
        mk = m.at_or_after(ts)
        e["mk0"] = float(mk[1]) if mk else None
        e["rv_bps"] = (e["px_rv"] * 1e4) if e.get("px_rv") else None

    # E1 — mark vs ask girisi (exit: 6h sonunda bid'e sat)
    print("\n=== E1: fill gercekciligi (mark vs ask->bid) ===")
    g_mark, g_ask = [], []
    for e in ev:
        if not e["q0"] or not e["mk0"]: continue
        xq = bt_at(conn, e["ts_ms"] + HOLD6, after=False)
        xm = m.at_or_before(e["ts_ms"] + HOLD6)
        if not xq or not xm: continue
        g_mark.append((float(xm[1]) - e["mk0"]) / e["mk0"] * 1e4)
        g_ask.append((xq["bid"] - e["q0"]["ask"]) / e["q0"]["ask"] * 1e4)
    R["E1_mark"] = stat(g_mark); R["E1_ask_bid"] = stat(g_ask)
    ps("E1 mark->mark (varsayim)", R["E1_mark"]); ps("E1 ask->bid (gercekci)", R["E1_ask_bid"])
    if R["E1_mark"].get("n") and R["E1_ask_bid"].get("n"):
        R["E1_spread_cost_bps"] = round(R["E1_mark"]["avg"] - R["E1_ask_bid"]["avg"], 1)
        print(f"    spread+quote maliyeti ~ {R['E1_spread_cost_bps']} bps/trade")

    # E2 — limit giris (durust: dolmayan sinyaller EV'de)
    print("\n=== E2: limit giris -10/-20/-30bps (15dk pencere, dolmayan=0) ===")
    for xb in (10, 20, 30):
        fills = []
        for e in ev:
            if not e["q0"]: continue
            lp = e["q0"]["mid"] * (1 - xb / 1e4)
            ft = limit_fill(conn, e["ts_ms"], lp)
            if ft is None: continue
            xq = bt_at(conn, ft + HOLD6, after=False)
            if not xq: continue
            fills.append((xq["bid"] - lp) / lp * 1e4)
        R[f"E2_limit{xb}"] = stat(fills, n_signals=len([e for e in ev if e["q0"]]))
        ps(f"E2 limit -{xb}bps", R[f"E2_limit{xb}"])
    n_sig = len([e for e in ev if e["q0"]])
    if R["E1_ask_bid"].get("n"):
        R["E1_ask_bid"]["ev_per_signal"] = round((R["E1_ask_bid"]["avg"] * R["E1_ask_bid"]["n"]) / n_sig, 1)
        print(f"    kiyas: market EV/sinyal = {R['E1_ask_bid']['ev_per_signal']} bps")

    # E3 — VWAP 5dk girisi
    print("\n=== E3: VWAP-5m girisi ===")
    g_vwap = []
    for e in ev:
        vp = vwap5(conn, e["ts_ms"])
        if vp is None: continue
        xq = bt_at(conn, e["ts_ms"] + 300_000 + HOLD6, after=False)
        if not xq: continue
        g_vwap.append((xq["bid"] - vp) / vp * 1e4)
    R["E3_vwap5"] = stat(g_vwap); ps("E3 vwap-5m -> bid", R["E3_vwap5"])

    # E4/E5 — dinamik TP + vol stop (mark path, entry=mk0; izole cikis etkisi)
    print("\n=== E4/E5: TP x SL taramasi (mark path) ===")
    tps = [("none", None)] + [(f"fix{t}", float(t)) for t in (100, 200, 300)] + \
          [(f"rv{k}", ("rv", k)) for k in (1, 2, 3)]
    sls = [("none", None)] + [(f"fix{s}", float(s)) for s in (150, 300)] + \
          [(f"rv{mm}", ("rv", mm)) for mm in (1, 2, 3)]
    grid = {}
    for tn, tv in tps:
        for sn, sv in sls:
            g_tr, g_te = [], []
            for e in ev:
                if not e["mk0"]: continue
                tp = None if tv is None else (tv if not isinstance(tv, tuple) else (e["rv_bps"] * tv[1] if e["rv_bps"] else None))
                sl = None if sv is None else (sv if not isinstance(sv, tuple) else (e["rv_bps"] * sv[1] if e["rv_bps"] else None))
                r = path_exit(m, e["ts_ms"], e["mk0"], tp, sl)
                if r is None: continue
                (g_te if e["ts_ms"] >= cut_ts else g_tr).append(r)
            grid[(tn, sn)] = {"TRAIN": stat(g_tr), "TEST": stat(g_te)}
    # TRAIN'de en iyi 5 (avg) + baseline
    ranked = sorted(grid.items(), key=lambda kv: kv[1]["TRAIN"].get("avg") or -999, reverse=True)
    base = grid[("none", "none")]
    print("    baseline (none,none)  TRAIN avg=%s  TEST avg=%s worst=%s"
          % (base["TRAIN"].get("avg"), base["TEST"].get("avg"), base["TEST"].get("worst")))
    R["E45_baseline"] = base
    R["E45_top"] = []
    for (tn, sn), v in ranked[:5]:
        R["E45_top"].append({"tp": tn, "sl": sn, "TRAIN": v["TRAIN"], "TEST": v["TEST"]})
        print("    tp=%-6s sl=%-6s TRAIN avg=%-7s | TEST N=%-3d WR=%-6s avg=%-7s worst=%s mc=%s"
              % (tn, sn, v["TRAIN"].get("avg"), v["TEST"].get("n", 0),
                 str(v["TEST"].get("wr")) + "%", v["TEST"].get("avg"), v["TEST"].get("worst"), v["TEST"].get("mc_p")))
    # rv-stop tek basina etkisi (tail sorusu)
    for sn in ("rv2", "rv3", "fix300"):
        v = grid[("none", sn)]
        print("    sl-only %-6s TEST avg=%-7s worst=%-8s (baseline worst=%s)"
              % (sn, v["TEST"].get("avg"), v["TEST"].get("worst"), base["TEST"].get("worst")))
        R[f"E5_slonly_{sn}"] = v

    conn.close()
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# S34 Execution Optimizer (Faz 4)", "",
             f"> gated event={len(ev)}, TRAIN/TEST 70/30. {datetime.now(timezone.utc):%Y-%m-%d}", ""]
    for k in ("E1_mark", "E1_ask_bid", "E2_limit10", "E2_limit20", "E2_limit30", "E3_vwap5"):
        v = R.get(k)
        if v and v.get("n"):
            ex = f" EV/sig={v.get('ev_per_signal')} fill={v.get('fill_rate')}%" if "ev_per_signal" in v else ""
            lines.append("- **%s**: N=%d WR=%s%% avg=%+.1f worst=%s mc=%s%s"
                         % (k, v["n"], v["wr"], v["avg"], v.get("worst"), v.get("mc_p"), ex))
    lines += ["", "## TP x SL (TRAIN-sirali top5, TEST raporu)", ""]
    for t in R.get("E45_top", []):
        te = t["TEST"]
        lines.append("- tp=%s sl=%s: TEST N=%s WR=%s%% avg=%+.1f worst=%s mc=%s"
                     % (t["tp"], t["sl"], te.get("n"), te.get("wr"), te.get("avg") or 0, te.get("worst"), te.get("mc_p")))
    b = R["E45_baseline"]["TEST"]
    lines += ["", f"- baseline(none,none): TEST N={b.get('n')} avg={b.get('avg')} worst={b.get('worst')}",
              "", "---", "*Script: tools/s34_execution_optimizer.py*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nJSON:{OJ}\nMD:  {OM}\nDone.")


if __name__ == "__main__":
    main()
