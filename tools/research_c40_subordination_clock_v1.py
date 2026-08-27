r"""LANE C, round 40 -- are C-T38's three regimes a CLOCK artefact? The corpus predicts, in advance,
which symbol it should work on.

C-T38 found that returns in TRADE time have three regimes -- sub-diffusive below T = 10, a
super-diffusive hump, then settling near diffusive above T = 300 -- and that no series carries a
global power law. The corpus has a standing explanation for exactly that shape, and this lane had
never used it.

    "Since the early works of Mandelbrot and Taylor, the concept of SUBORDINATION by a trading or
     transaction clock that maps the physical time to the number of trades (or the cumulated
     volume) has been widely used... according to the model, the physical time does not play any
     role in the way market prices vary from trade to trade. This implies notably that THE
     VARIANCE PER TRADE (OR PER UNIT OF VOLUME TRADED) IS CONSTANT."
                                                            Econophysics of Order-driven Markets

A subordinated price is a Brownian motion evaluated at a random time change. Measured in the
WRONG clock it shows scale-dependent apparent exponents; measured in the RIGHT one it is a plain
random walk. So the three regimes are a candidate clock artefact, and the test is to change clocks.

AND THE CORPUS MAKES A DIRECTIONAL, CROSS-SYMBOL PREDICTION BEFORE ANY MEASUREMENT:

    "the ability of the subordination hypothesis in explaining fat tails of returns and volatility
     clustering is strongly dependent on TICK SIZE. While for LARGE tick sizes the subordination
     hypothesis has significant explanatory power, for SMALL tick sizes we show that subordination
     is not the main driver."

This estate has exactly one large-tick symbol and two small-tick ones, and the ordering is already
measured rather than assumed: C-T26 put the spread at 11.62% of cost on SOL against 0.527% on ETH
and 0.154% on BTC, and C-T30 measured ticks-per-trade k at 0.24-0.68 (SOL, always binding) versus
0.72-4.73 (BTC and ETH, mostly not). So the prediction is:

    switching to a volume clock should FLATTEN the local-slope profile on SOL, and do much less on
    BTC and ETH.

That is a real prediction: it is directional, it is stated before the measurement, and it is
differentiated across symbols by a property measured in an earlier round for another purpose.

Two things are measured:
  1. the subordination hypothesis itself -- is the variance per unit volume constant?
  2. the local-slope drift of the price, in trade time and in volume time, against C-T38's
     calibrated tolerance of 0.0686 (what a TRUE power law drifts at this length).
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
TOL = 0.0686                      # C-T38's measured tolerance, not re-chosen here
HALF_DECADE = 10 ** 0.5
NT = 30
SEED = 20260827

# measured in earlier rounds, for another purpose
TICK_BINDING = {"SOLUSDT": 0.24, "ETHUSDT": 0.72, "BTCUSDT": 0.80}   # min k, C-T30
SPREAD_SHARE = {"SOLUSDT": 11.62, "ETHUSDT": 0.527, "BTCUSDT": 0.154}  # % of cost, C-T26


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sd_curve(x, tmin=4, tmax=20000):
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(tmin, min(tmax, n // 200), NT)).astype(int))
    T, S = [], []
    for t in Ts:
        m = n // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    return np.array(T), np.array(S)


def local_slopes(T, S):
    lt, ls = np.log(T), np.log(S)
    out = []
    for i in range(len(T)):
        m = (T >= T[i]) & (T <= T[i] * HALF_DECADE)
        if m.sum() < 4:
            continue
        A = np.column_stack([np.ones(m.sum()), lt[m]])
        b, *_ = np.linalg.lstsq(A, ls[m], rcond=None)
        out.append((float(T[i]), float(b[1])))
    return out


def drift(sl):
    v = [s for _, s in sl]
    return (max(v) - min(v)) if v else float("nan")


def volume_clock(lp, vol, n_target):
    """resample the log-price path at equal increments of cumulative traded notional"""
    cv = np.cumsum(vol)
    step = cv[-1] / n_target
    marks = np.arange(1, n_target + 1) * step
    idx = np.searchsorted(cv, marks)
    idx = np.clip(idx, 0, len(lp) - 1)
    p = lp[idx]
    d = np.empty_like(p)
    d[0] = 0.0
    d[1:] = np.diff(p)
    return d, step


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional from agg_trades where symbol=? order by ts_ms limit ?",
                (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            vol = a[:, 1]
            n = len(lp)

            # --- trade clock
            r_t = np.empty_like(lp)
            r_t[0] = 0.0
            r_t[1:] = np.diff(lp)
            sl_t = local_slopes(*sd_curve(r_t))

            # --- volume clock, same number of steps so the comparison is like for like
            r_v, step = volume_clock(lp, vol, n)
            sl_v = local_slopes(*sd_curve(r_v))

            # --- the subordination hypothesis itself: is variance per unit volume constant?
            #     bin trades by activity and compare return variance per unit volume
            B = 20
            m = n // B
            var_per_trade, var_per_vol, blk_vol = [], [], []
            for b in range(B):
                s_ = slice(b * m, (b + 1) * m)
                rr = r_t[s_]
                vv = vol[s_].sum()
                var_per_trade.append(float(np.var(rr, ddof=1)))
                var_per_vol.append(float(np.var(rr, ddof=1) * len(rr) / vv))
                blk_vol.append(float(vv))
            cv_trade = float(np.std(var_per_trade, ddof=1) / np.mean(var_per_trade))
            cv_vol = float(np.std(var_per_vol, ddof=1) / np.mean(var_per_vol))

            per[sym] = {
                "tick_binding_k_min_C_T30": TICK_BINDING[sym],
                "spread_share_of_cost_pct_C_T26": SPREAD_SHARE[sym],
                "trade_clock": {"drift": round(drift(sl_t), 4),
                                "drift_over_tolerance": round(drift(sl_t) / TOL, 2),
                                "slopes": [(int(t), round(s, 4)) for t, s in sl_t]},
                "volume_clock": {"drift": round(drift(sl_v), 4),
                                 "drift_over_tolerance": round(drift(sl_v) / TOL, 2),
                                 "notional_per_step": float("{0:.6g}".format(step)),
                                 "slopes": [(int(t), round(s, 4)) for t, s in sl_v]},
                "drift_reduction_factor": round(drift(sl_t) / drift(sl_v), 3),
                "subordination_hypothesis": {
                    "cv_of_variance_per_trade": round(cv_trade, 4),
                    "cv_of_variance_per_unit_volume": round(cv_vol, 4),
                    "per_volume_is_steadier": bool(cv_vol < cv_trade),
                    "note": ("the hypothesis says variance per trade OR per unit volume is "
                             "constant; a coefficient of variation near 0 supports it")},
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    order_pred = sorted(SYMS, key=lambda s: TICK_BINDING[s])          # most binding first
    order_obs = sorted(SYMS, key=lambda s: -per[s]["drift_reduction_factor"])
    art = {"study": "C-T40", "lane": "C", "utc": _utc(),
           "prediction_stated_before_measurement": (
               "the corpus says subordination has explanatory power for LARGE tick sizes and is "
               "not the main driver for small ones, so the volume clock should flatten the "
               "local-slope profile most on SOL and least on BTC/ETH"),
           "tolerance_from_C_T38": TOL,
           "predicted_order_most_helped_first": order_pred,
           "observed_order_most_helped_first": order_obs,
           "prediction_holds": bool(order_pred[0] == order_obs[0]),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C40_SUBORDINATION_CLOCK_V1.json").write_text(json.dumps(art, indent=2),
                                                         encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("%-9s %8s %10s %12s %12s %10s" % ("sym", "k_min", "spread%", "drift trade", "drift volume",
                                        "reduction"))
    for s in SYMS:
        p = per[s]
        w("%-9s %8.2f %10.3f %12.4f %12.4f %10.3f" % (
            s, p["tick_binding_k_min_C_T30"], p["spread_share_of_cost_pct_C_T26"],
            p["trade_clock"]["drift"], p["volume_clock"]["drift"],
            p["drift_reduction_factor"]))
    w("")
    w("predicted order (most helped first): {0}".format(order_pred))
    w("observed  order (most helped first): {0}".format(order_obs))
    w("prediction holds: {0}".format(art["prediction_holds"]))
    w("")
    w("SUBORDINATION HYPOTHESIS -- coefficient of variation across 20 activity blocks")
    for s in SYMS:
        h = per[s]["subordination_hypothesis"]
        w("   %-9s per-trade %.4f   per-unit-volume %.4f   volume steadier: %s" % (
            s, h["cv_of_variance_per_trade"], h["cv_of_variance_per_unit_volume"],
            h["per_volume_is_steadier"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
