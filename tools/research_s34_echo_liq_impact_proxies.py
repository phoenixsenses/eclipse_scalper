"""
research_s34_echo_liq_impact_proxies.py — DESCRIPTIVE characterization of impact/liquidity
regime proxies at echo-anchor T0 (read-only, OD-029 SAFE, NO outcome mining).

WHY: docs/ECHO_SIGNAL_DEV_INDICATORS.md lists Kyle-lambda / Amihud (#15) and jump-flag (#19)
as `o candidate` regime indicators to potentially enrich the echo_30_90+regime lead. Before
wiring anything into the forward ledger (research_s34_echo_forward_ledger.py), this tool answers
the FEASIBILITY / STRUCTURE question on the (burned) historical echo population:
    - are the four proxies computable at anchor T0 given book_ticker coverage?
    - what are their distributions across echo anchors?
    - are they redundant with each other and with the existing regime gate?

DISCIPLINE (hard):
    - Anchor population = IDENTICAL frozen params to the echo ledger (200K/300s/900s/30s), ETHUSDT SELL.
    - Every proxy is CAUSAL: uses only data <= T0 (liq-impact markout uses t+delta <= T0).
    - NO conditioning on outcome / net_bps / forward return. Any predictive/threshold claim is
      FORWARD-ONLY per OD-029 and the microstructure_indicators.py frame. This file characterizes
      structure; it does NOT select or validate alpha.
    - Cross-tab with the existing regime gate (btc4h<0 OR btc7d<0) is T0-knowable, not outcome.

Kyle-lambda / Amihud / RV formulas mirror tools/microstructure_indicators.py exactly (same window,
same signed-volume convention) so numbers reconcile. Bipower/jump (BNS) and liq-per-impact are new.

Read-only (mode=ro, query_only=1). Writes ONLY its report .md + .json under reports/research/s34/.
Deterministic: no randomness, no wall-clock in any computed value (only a metadata gen-stamp).
"""
from __future__ import annotations
import sqlite3, json, math, sys, argparse, bisect, statistics as st, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors, load_liquidations  # noqa: E402

DB_URI = f"file:{ROOT / 'data' / 'microstructure.db'}?mode=ro"
OUTDIR = ROOT / "reports" / "research" / "s34"

# ---- FROZEN anchor params: MUST match echo_forward_ledger §1 (identical echo population) ----
SYMBOL = "ETHUSDT"
ETH_THRESH = 200_000.0
BUCKET_SEC, MIN_GAP_SEC, ACCEL_SEC = 300, 900, 30
# ---- proxy windows ----
MICRO_WIN_MIN = 15                # microstructure window (match microstructure_indicators default)
LIQ_IMPACT_WIN_MS = 30 * 60_000   # trailing window for liq-per-impact regime estimate
LIQ_IMPACT_DELTA_MS = 5_000       # short-horizon markout per liquidation (t+delta <= T0 kept causal)


def _mark_series(cur, sym, lo, hi):
    """(ts, mark) ascending over [lo, hi]; bisect-able for causal point lookups."""
    rows = cur.execute("SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms BETWEEN ? AND ? "
                       "ORDER BY ts_ms ASC", (sym, lo, hi)).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows if r[1]]


def _mark_bps(cur, sym, ts, lookback_ms):
    a = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts - lookback_ms)).fetchone()
    b = cur.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                    (sym, ts)).fetchone()
    if a and b and a[0] and float(a[0]) > 0:
        return (float(b[0]) - float(a[0])) / float(a[0]) * 1e4
    return None


def _at_or_before(series_ts, series_val, t):
    """Causal point lookup: last sample with ts <= t."""
    i = bisect.bisect_right(series_ts, t) - 1
    return series_val[i] if i >= 0 else None


def proxies_at(cur, ts):
    """Four impact/liquidity proxies computed causally at anchor T0=ts. Returns dict + coverage."""
    win = MICRO_WIN_MIN * 60_000
    lo = ts - win
    tr = cur.execute("SELECT price, notional, is_buyer_maker FROM agg_trades "
                     "WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms", (SYMBOL, lo, ts)).fetchall()
    bk = cur.execute("SELECT mid_price FROM book_ticker WHERE symbol=? AND ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                     (SYMBOL, lo, ts)).fetchall()
    out = {"n_trades": len(tr), "n_book": len(bk),
           "kyle_lambda": None, "amihud": None, "rv_bps": None, "bv_bps": None,
           "jump_frac": None, "liq_impact_bps_per_M": None, "liq_impact_n": 0}

    # --- Kyle lambda + Amihud (mirror microstructure_indicators.py) ---
    if len(tr) >= 5:
        p0, p1 = tr[0][0], tr[-1][0]
        ret_bps = (p1 - p0) / p0 * 1e4 if p0 else None
        abuy = sum((n or 0.0) for _, n, ibm in tr if ibm == 0)
        asell = sum((n or 0.0) for _, n, ibm in tr if ibm == 1)
        signed_vol = abuy - asell
        tot_vol = abuy + asell
        if ret_bps is not None and abs(signed_vol) > 1e-9:
            out["kyle_lambda"] = ret_bps / (signed_vol / 1e6)      # bps per $M signed
        if ret_bps is not None and tot_vol > 1e-9:
            out["amihud"] = abs(ret_bps) / (tot_vol / 1e6)         # |bps| per $M total

    # --- Realized vol + bipower variation -> jump fraction (Barndorff-Nielsen & Shephard) ---
    mids = [r[0] for r in bk if r[0]]
    if len(mids) >= 3:
        rets = [(mids[i] - mids[i - 1]) / mids[i - 1] for i in range(1, len(mids)) if mids[i - 1]]
        if len(rets) >= 2:
            rv = sum(x * x for x in rets)
            # BV = (pi/2) * sum |r_i| |r_{i-1}|  (jump-robust integrated variance estimator)
            bv = (math.pi / 2.0) * sum(abs(rets[i]) * abs(rets[i - 1]) for i in range(1, len(rets)))
            out["rv_bps"] = math.sqrt(rv) * 1e4
            out["bv_bps"] = math.sqrt(max(bv, 0.0)) * 1e4
            out["jump_frac"] = max(rv - bv, 0.0) / rv if rv > 0 else 0.0   # discontinuous share of variance

    # --- Liq-per-impact: causal trailing bps move per $M liquidation notional (S34-specific) ---
    ms = _mark_series(cur, SYMBOL, ts - LIQ_IMPACT_WIN_MS - LIQ_IMPACT_DELTA_MS, ts)
    if ms:
        mts = [x[0] for x in ms]; mval = [x[1] for x in ms]
        liqs = cur.execute("SELECT ts_ms, notional FROM liquidations WHERE symbol=? AND "
                           "ts_ms>=? AND ts_ms<=? ORDER BY ts_ms", (SYMBOL, ts - LIQ_IMPACT_WIN_MS, ts - LIQ_IMPACT_DELTA_MS)).fetchall()
        impacts = []
        for lt, notional in liqs:
            if not notional or notional <= 0:
                continue
            m0 = _at_or_before(mts, mval, int(lt))
            m1 = _at_or_before(mts, mval, int(lt) + LIQ_IMPACT_DELTA_MS)  # t+delta <= T0 by window construction
            if m0 and m1 and m0 > 0:
                impacts.append(abs((m1 - m0) / m0 * 1e4) / (float(notional) / 1e6))
        if impacts:
            out["liq_impact_bps_per_M"] = st.median(impacts)   # median = robust regime level
            out["liq_impact_n"] = len(impacts)
    return out


def _pct(vals, q):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def _spearman(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 5:
        return None, len(pairs)
    xs, ys = zip(*pairs)
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v); i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(list(xs)), rank(list(ys))
    n = len(rx); mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return (num / (dx * dy) if dx > 0 and dy > 0 else None), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-ms", type=int, default=None, help="anchor window start (default: full history)")
    ap.add_argument("--end-ms", type=int, default=None)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_URI, uri=True)
    conn.execute("PRAGMA query_only=1")
    cur = conn.cursor()

    lo = args.start_ms if args.start_ms is not None else cur.execute(
        "SELECT MIN(ts_ms) FROM liquidations WHERE symbol=? AND side='SELL'", (SYMBOL,)).fetchone()[0]
    hi = args.end_ms if args.end_ms is not None else cur.execute(
        "SELECT MAX(ts_ms) FROM liquidations WHERE symbol=? AND side='SELL'", (SYMBOL,)).fetchone()[0]
    lo, hi = int(lo), int(hi)

    liqs = load_liquidations(conn, SYMBOL, "SELL", lo, hi)
    anchors = reconstruct_anchors(liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
                                  thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_SEC)
    anchors = sorted(anchors, key=lambda a: int(a.anchor_ts_ms))

    recs = []
    for a in anchors:
        ts = int(a.anchor_ts_ms)
        btc4h = _mark_bps(cur, "BTCUSDT", ts, 4 * 3600_000)
        btc7d = _mark_bps(cur, "BTCUSDT", ts, 7 * 24 * 3600_000)
        regime = ((btc4h or 0.0) < 0) or ((btc7d or 0.0) < 0)
        px = proxies_at(cur, ts)
        px.update({"anchor_ts_ms": ts, "regime_gate": regime,
                   "running_notional": float(a.running_notional)})
        recs.append(px)
    conn.close()

    keys = ["kyle_lambda", "amihud", "rv_bps", "bv_bps", "jump_frac", "liq_impact_bps_per_M"]
    def dist(vals):
        v = [x for x in vals if x is not None]
        return {"n": len(v), "cover_pct": round(100 * len(v) / max(len(recs), 1), 1),
                "p10": _pct(vals, .10), "p25": _pct(vals, .25), "p50": _pct(vals, .50),
                "p75": _pct(vals, .75), "p90": _pct(vals, .90),
                "mean": (sum(v) / len(v) if v else None)}
    distributions = {k: dist([r[k] for r in recs]) for k in keys}

    # mutual Spearman among proxies + regime gate (as 0/1) + rv (structure/redundancy only)
    corr = {}
    cols = keys + ["regime_int"]
    colvals = {k: [r.get(k) for r in recs] for k in keys}
    colvals["regime_int"] = [1 if r["regime_gate"] else 0 for r in recs]
    for i, ka in enumerate(cols):
        for kb in cols[i + 1:]:
            rho, n = _spearman(colvals[ka], colvals[kb])
            if rho is not None:
                corr[f"{ka}~{kb}"] = {"rho": round(rho, 3), "n": n}

    # descriptive cross-tab: proxy medians split by regime gate (T0-knowable, NOT outcome)
    crosstab = {}
    for k in keys:
        on = [r[k] for r in recs if r["regime_gate"] and r[k] is not None]
        off = [r[k] for r in recs if not r["regime_gate"] and r[k] is not None]
        crosstab[k] = {"regime_on_median": (st.median(on) if on else None), "regime_on_n": len(on),
                       "regime_off_median": (st.median(off) if off else None), "regime_off_n": len(off)}

    genstamp = dt.datetime.now(dt.timezone.utc).isoformat()  # metadata only, not a computed value
    result = {
        "tool": "research_s34_echo_liq_impact_proxies", "tag": args.tag, "generated_utc": genstamp,
        "frame": "DESCRIPTIVE characterization only. NO outcome/net_bps conditioning. Predictive claims "
                 "are FORWARD-ONLY (OD-029). Anchor population identical to echo_forward_ledger frozen params.",
        "anchor_params": {"symbol": SYMBOL, "thresh_usd": ETH_THRESH, "bucket_sec": BUCKET_SEC,
                          "min_gap_sec": MIN_GAP_SEC, "accel_sec": ACCEL_SEC},
        "proxy_windows": {"micro_win_min": MICRO_WIN_MIN, "liq_impact_win_ms": LIQ_IMPACT_WIN_MS,
                          "liq_impact_delta_ms": LIQ_IMPACT_DELTA_MS},
        "window_ms": [lo, hi], "n_anchors": len(recs),
        "regime_on_n": sum(1 for r in recs if r["regime_gate"]),
        "distributions": distributions, "spearman": corr, "regime_crosstab": crosstab,
    }

    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = OUTDIR / f"ECHO_LIQ_IMPACT_PROXIES_{args.tag}"
    (base.with_suffix(".json")).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    # also dump per-anchor records for auditability (no outcome fields present)
    (OUTDIR / f"ECHO_LIQ_IMPACT_PROXIES_{args.tag}_records.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in recs), encoding="utf-8")

    lines = [f"# Echo Liq/Impact Proxies — Descriptive Characterization ({args.tag})", "",
             f"_generated {genstamp} · READ-ONLY · OD-029 SAFE (no outcome mining)_", "",
             result["frame"], "",
             f"- Anchors: **{len(recs)}** (ETHUSDT SELL, frozen echo params) · regime-gate ON: "
             f"{result['regime_on_n']} · window {lo}–{hi}", "",
             "## Proxy distributions & coverage", "",
             "| proxy | cover% | p10 | p25 | p50 | p75 | p90 | mean |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        d = distributions[k]
        def f(x): return "—" if x is None else (f"{x:.4g}")
        lines.append(f"| {k} | {d['cover_pct']} | {f(d['p10'])} | {f(d['p25'])} | {f(d['p50'])} | "
                     f"{f(d['p75'])} | {f(d['p90'])} | {f(d['mean'])} |")
    lines += ["", "## Mutual rank-correlation (redundancy check)", "",
              "| pair | Spearman rho | n |", "|---|---|---|"]
    for k, v in sorted(corr.items(), key=lambda kv: -abs(kv[1]["rho"])):
        lines.append(f"| {k} | {v['rho']} | {v['n']} |")
    lines += ["", "## Regime-gate cross-tab (T0-knowable split, NOT outcome)", "",
              "| proxy | regime ON median (n) | regime OFF median (n) |", "|---|---|---|"]
    for k in keys:
        c = crosstab[k]
        def g(x): return "—" if x is None else f"{x:.4g}"
        lines.append(f"| {k} | {g(c['regime_on_median'])} ({c['regime_on_n']}) | "
                     f"{g(c['regime_off_median'])} ({c['regime_off_n']}) |")
    lines += ["", "## Boundary (do not cross without forward data)",
              "- This is structure/feasibility only. Whether any proxy improves the echo lead's NET is a",
              "  FORWARD question: wire the surviving proxies into research_s34_echo_forward_ledger.py's",
              "  indicator_snapshot() as new causal fields (dev-list #15/#19 `o candidate` -> `captured`),",
              "  then accumulate post-2026-07-20 anchors. No threshold is selected here.", ""]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"n_anchors": len(recs), "regime_on_n": result["regime_on_n"],
                      "coverage": {k: distributions[k]["cover_pct"] for k in keys},
                      "out_md": str(base.with_suffix(".md"))}, indent=2))


if __name__ == "__main__":
    main()
