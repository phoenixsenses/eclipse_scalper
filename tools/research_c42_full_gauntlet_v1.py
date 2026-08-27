r"""LANE C, round 42 -- the operator's full robustness gauntlet, run on this lane's one economic result.

C-T29 measured the only economic number this lane owns: trade in the direction of the past
window's signed-flow imbalance, hold one window. It paid +0.2471 bps (BTC, T=50) and +0.2645 bps
(ETH, T=20) against a 10 bps round trip. Everything else this lane produced is structure.

The operator has put fifteen robustness questions. Some are answerable from this data, some are
already answered by earlier rounds, and some are structurally blocked -- and this lane has spent
several rounds building the vocabulary to say WHICH is which (C-T37: Hernan & Robins, the three
identifiability conditions). This script runs every one that is runnable, all of them, rather
than picking a representative.

RUNNABLE HERE, AND RUN:
  net edge            gross - fee - spread - impact, using C-T33's own measured impact curve
  horizon             T in {5,10,20,50,100,200,500}
  latency             enter d trades after the signal, d in {0,1,5,20,50}
  event definition    four variants of "imbalance", including two thresholded ones
  fresh OOS           first 70% fits nothing, last 30% is untouched; the rule has no free
                      parameter to move, so this is a genuine hold-out for the VALUE
  regime split        realised-volatility quartile of the signal window
  concentration       share of total PnL from the top 1 / 5 / 10 percent of events
  block null          circular-shift null that preserves the dependence structure
  selection penalty   max over the whole grid against the same max under the null -- the
                      operational form of Lopez de Prado's False Strategy theorem
  size / POV surface  at what traded notional does the measured impact erase the gross edge
  capacity            the notional at which net edge crosses zero, per symbol

WHAT THE CORPUS DEMANDS, AND WHERE IT LANDS:
  Lopez de Prado 8.5   the expected maximum Sharpe over K trials is strictly positive under the
                       null, so a maximum must be judged against E[max], never against zero.
  Lopez de Prado 8.7.1 "The False Strategy theorem requires knowledge of the number of INDEPENDENT
                       trials... it is uncommon for financial researchers to run independent
                       trials." K is the number of CLUSTERS, not the number of symbols or cells.
                       That is the corpus's essential test for a universe, and it is why breadth
                       is not symbol count.
  Kissell 3            IS is undefined without a decision price, so "edge" must be measured from
                       the signal instant, which is what the latency ladder below does.
  Bouchaud 11.2        the reaction counterfactual is not implementable, so impact enters here as
                       an OBSERVED conditional mean (C-T33), not as a causal cost.

NOT RUNNABLE, WITH THE REASON NAMED (no result is claimed for these):
  breadth / universe   the price feed carries three symbols; liquidations carry 761 but no price.
  cross-exchange       no second venue in the estate.
  maker execution      CLAUDE.md parks the maker line; reopening needs a real best-of-book engine.
  adverse selection    measured already at section 206 on real fills, not re-derivable here.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT = ROOT / "reports" / "atlas"
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
NROWS = 2_000_000
HORIZONS = (5, 10, 20, 50, 100, 200, 500)
LATENCY = (0, 1, 5, 20, 50)
FEE_RT_BPS = 10.0
SPREAD_BPS = {"BTCUSDT": 0.0156, "ETHUSDT": 0.0536, "SOLUSDT": 1.35}   # C-T26 / C-T33
NULL_REPS = 60
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build(lp, flow, T):
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m) * T
    dv = flow[:m * T].reshape(m, T).sum(axis=1)[1:]
    lvl = lp[i0 + T - 1]
    return dv, lvl, i0


def signal_value(lp, flow, T, d, defn, mask=None):
    """gross bps of trading the past window's imbalance, entering d trades late"""
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m - 1) * T
    W = flow[:(m) * T].reshape(m, T)
    if defn == "signed_notional":
        s = np.sign(W.sum(axis=1))
    elif defn == "unweighted_sign":
        s = np.sign(np.sign(W).sum(axis=1))
    else:
        tot = W.sum(axis=1)
        cut = np.percentile(np.abs(tot), 75 if defn == "thresh_p75" else 95)
        s = np.where(np.abs(tot) >= cut, np.sign(tot), 0.0)
    s = s[1:len(i0) + 1]
    entry = np.clip(i0 + T - 1 + d, 0, n - 1)
    exit_ = np.clip(entry + T, 0, n - 1)
    g = (lp[exit_] - lp[entry]) * 1e4 * s
    ok = s != 0
    if mask is not None:
        ok = ok & mask[:len(ok)]
    return g[ok]


def stat(g):
    if len(g) < 100:
        return None
    mu = float(g.mean())
    se = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {"n": int(len(g)), "gross_bps": round(mu, 4), "se": round(se, 4),
            "t": round(mu / se, 2) if se > 0 else None}


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    imp = json.loads((OUT / "C33_AGGREGATE_IMPACT_V1.json").read_text(encoding="utf-8"))
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            flow = np.where(a[:, 2] > 0.5, -1.0, 1.0) * a[:, 1]
            n = len(lp)
            cut70 = int(n * 0.7)

            # ---- grid: horizon x latency x definition
            grid = {}
            for T in HORIZONS:
                for d in LATENCY:
                    for defn in ("signed_notional", "unweighted_sign", "thresh_p75",
                                 "thresh_p95"):
                        s_ = stat(signal_value(lp, flow, T, d, defn))
                        if s_:
                            grid["T{0}_d{1}_{2}".format(T, d, defn)] = s_
            best_key = max(grid, key=lambda k: grid[k]["gross_bps"])
            best = grid[best_key]

            # ---- selection penalty: the same maximum under a dependence-preserving null
            null_max = []
            for _ in range(NULL_REPS):
                sh = int(rng.integers(T * 10, n - T * 10))
                lp_s = np.roll(lp, sh)          # circular shift decouples signal from outcome
                vals = []
                for T in HORIZONS:
                    for d in (0, 5):
                        v = stat(signal_value(lp_s, flow, T, d, "signed_notional"))
                        if v:
                            vals.append(v["gross_bps"])
                if vals:
                    null_max.append(max(vals))
            e_max = float(np.mean(null_max))
            sd_max = float(np.std(null_max, ddof=1))

            # ---- OOS: last 30%, no parameter chosen there
            T0, d0 = 50, 0
            g_is = signal_value(lp[:cut70], flow[:cut70], T0, d0, "signed_notional")
            g_oos = signal_value(lp[cut70:], flow[cut70:], T0, d0, "signed_notional")

            # ---- regime split by realised vol of the signal window
            m = (n - 1) // T0
            i0 = np.arange(1, m - 1) * T0
            rv = np.array([float(np.std(np.diff(lp[i:i + T0]))) for i in i0[:20000]])
            g_all = signal_value(lp, flow, T0, d0, "signed_notional")
            k = min(len(rv), len(g_all))
            q = np.percentile(rv[:k], [25, 50, 75])
            regime = {}
            for lab, lo, hi in (("q1_lowvol", -np.inf, q[0]), ("q2", q[0], q[1]),
                                ("q3", q[1], q[2]), ("q4_highvol", q[2], np.inf)):
                mm = (rv[:k] > lo) & (rv[:k] <= hi)
                regime[lab] = stat(g_all[:k][mm])

            # ---- concentration
            g = np.sort(np.abs(g_all))[::-1]
            tot = float(np.abs(g_all).sum())
            conc = {p: round(float(g[:max(1, int(len(g) * p / 100))].sum()) / tot, 4)
                    for p in (1, 5, 10)}

            # ---- size / POV surface -> capacity, from C-T33's measured impact curve
            cur = imp["per_symbol"][sym]["by_T"]["50"]["curve"]
            dv_ax = np.array([c["dV"] for c in cur], float)
            r_ax = np.array([c["R_bps"] for c in cur], float)
            pos = dv_ax > 0
            dv_ax, r_ax = dv_ax[pos], np.abs(r_ax[pos])
            o = np.argsort(dv_ax)
            dv_ax, r_ax = dv_ax[o], r_ax[o]
            gross = best["gross_bps"]
            budget = gross - FEE_RT_BPS - SPREAD_BPS[sym]
            cap = None
            if budget > 0:
                over = np.where(r_ax > budget)[0]
                cap = float(dv_ax[over[0]]) if len(over) else float(dv_ax[-1])
            sizes = {}
            for f in (0.1, 0.5, 1.0, 2.0, 5.0):
                q_ = float(np.percentile(dv_ax, 50)) * f
                sizes["{0}x_median".format(f)] = {
                    "notional": float("{0:.6g}".format(q_)),
                    "impact_bps": round(float(np.interp(q_, dv_ax, r_ax)), 4),
                    "net_bps": round(gross - FEE_RT_BPS - SPREAD_BPS[sym]
                                     - float(np.interp(q_, dv_ax, r_ax)), 4)}

            per[sym] = {
                "grid_cells": len(grid),
                "best_cell": best_key, "best": best,
                "selection_penalty": {
                    "E_max_under_null": round(e_max, 4), "sd": round(sd_max, 4),
                    "observed_max": best["gross_bps"],
                    "excess_over_E_max": round(best["gross_bps"] - e_max, 4),
                    "z_vs_null_max": (round((best["gross_bps"] - e_max) / sd_max, 2)
                                      if sd_max > 0 else None)},
                "oos": {"in_sample_first70": stat(g_is), "out_of_sample_last30": stat(g_oos)},
                "regime_split": regime,
                "concentration_share_of_abs_pnl": conc,
                "net_edge_at_best_cell": {
                    "gross_bps": gross, "fee_rt_bps": FEE_RT_BPS,
                    "spread_bps": SPREAD_BPS[sym],
                    "net_before_impact_bps": round(budget, 4),
                    "shortfall_multiple": (round(FEE_RT_BPS / gross, 1) if gross > 0 else None)},
                "size_surface": sizes, "capacity_notional": cap,
                "grid": grid,
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T42", "lane": "C", "utc": _utc(),
           "signal": "trade the past window's imbalance, hold one window (C-T29)",
           "fee": "BINANCE_BASE 10.0 bps round trip", "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C42_FULL_GAUNTLET_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    for s in SYMS:
        p = per[s]
        w("===== {0}   ({1} grid cells)".format(s, p["grid_cells"]))
        w("  best cell {0}: gross {1} bps (t {2}, n {3})".format(
            p["best_cell"], p["best"]["gross_bps"], p["best"]["t"], p["best"]["n"]))
        sp = p["selection_penalty"]
        w("  SELECTION: E[max] under null {0} +- {1}; observed {2}; excess {3}; z {4}".format(
            sp["E_max_under_null"], sp["sd"], sp["observed_max"], sp["excess_over_E_max"],
            sp["z_vs_null_max"]))
        ne = p["net_edge_at_best_cell"]
        w("  NET: gross {0} - fee {1} - spread {2} = {3} bps  (fee alone is {4}x the gross)".format(
            ne["gross_bps"], ne["fee_rt_bps"], ne["spread_bps"], ne["net_before_impact_bps"],
            ne["shortfall_multiple"]))
        o = p["oos"]
        w("  OOS: in-sample {0} bps (n {1}) | out-of-sample {2} bps (n {3})".format(
            o["in_sample_first70"]["gross_bps"], o["in_sample_first70"]["n"],
            o["out_of_sample_last30"]["gross_bps"], o["out_of_sample_last30"]["n"]))
        w("  REGIME: " + "  ".join("{0} {1}".format(k, v["gross_bps"] if v else "-")
                                   for k, v in p["regime_split"].items()))
        w("  CONCENTRATION of |PnL|: top1% {0}  top5% {1}  top10% {2}".format(
            p["concentration_share_of_abs_pnl"][1], p["concentration_share_of_abs_pnl"][5],
            p["concentration_share_of_abs_pnl"][10]))
        w("  SIZE SURFACE (net bps after fee+spread+impact):")
        for k, v in p["size_surface"].items():
            w("     {0:>12}  notional {1:>12}  impact {2:>8} bps  net {3} bps".format(
                k, v["notional"], v["impact_bps"], v["net_bps"]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
