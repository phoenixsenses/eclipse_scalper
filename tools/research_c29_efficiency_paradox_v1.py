r"""LANE C, round 29 -- how much of the flow's predictability the price refuses to inherit.

C-T28 established, across four weightings and 12 of 12 cells, that order-sign imbalance on this
estate grows as T^chi with chi well above 1/2 (0.70-0.89 on the unweighted sign series, z = 18
to 91 against a measured null of 0.4991-0.5012), and that the entire excess is SIGN memory:
shuffling signs collapses chi to 0.5, shuffling sizes does not.

Bouchaud calls the consequence the efficiency paradox (Section 10.4): order flow is strongly
predictable, yet prices are not. He states the resolution outright --

    "the removal of the order-flow correlations is the result of the counter-balancing role of
     liquidity providers"

-- and gives the object that measures it, Equation (8.7):

    R(l) := < (m_{t+l} - m_t) . eps_t >          the lag-l response function

This round turns that into a counterfactual that this estate's own data can answer, with no new
hypothesis and no alpha claim.

THE COUNTERFACTUAL. Let G := R(1), the immediate impact of one trade in bps. IF impact were
permanent, each trade would displace the price by G in the direction of its sign and the
displacement would stay. Then over T trades the price would move G * sum(eps), so

    predicted price dispersion under permanent impact  =  G * sd( sum_T eps )  ~  G * T^chi

against the actual sd( sum_T d ) ~ T^H, where H is the price's own diffusion exponent in trade
time. The ratio of actual to predicted is the fraction of the naive permanent-impact move that
SURVIVES; one minus it is what the book's liquidity providers cancel. Because the ratio goes as
T^(H - chi) and chi > H, the cancellation is not a constant -- it grows with horizon, and its
exponent is a measured quantity rather than a fitted one.

THREE OUTCOMES ARE POSSIBLE AND THEY ARE DISTINGUISHABLE.
    H = chi   -> impact is permanent, price inherits the flow's memory, and the sign
                 predictability is directly tradeable. Nothing in this estate suggests it.
    H = 0.5   -> the flow's memory is cancelled EXACTLY, to the last decimal. Efficient.
    H < 0.5   -> over-cancellation: prices mean-revert in trade time.

H is measured with the same partial-sum machinery as chi -- sd of sums over T -- which C-T28
showed is well conditioned (null sd ~0.004) and robust to the infinite variance of notional,
unlike the E|r| statistic C-T23 used for alpha, whose null sat at 0.726 rather than 0.5.

WHAT THIS IS NOT. It is not a test of whether anything is tradeable, and it consumes no
hypothesis budget: G, H and chi are all descriptive scalings of the same 2,000,000 trades, and
the counterfactual is arithmetic on them. It is the MECHANISM rung of the measurement ladder,
not the ECONOMICS rung.
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
WINDOW_T = (20, 50, 100, 200, 500, 1000)
LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def slope(x, y):
    x = np.log(np.asarray(x, float))
    y = np.log(np.asarray(y, float))
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    tot = float(((y - y.mean()) ** 2).sum())
    return float(b[1]), float(1 - float(r @ r) / tot) if tot > 0 else float("nan")


def partial_sum_sd(x):
    """sd of the sum of x over non-overlapping windows of T, for each T in WINDOW_T"""
    n = len(x)
    T, S = [], []
    for t in WINDOW_T:
        m = n // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    return T, S


def response(lp, eps):
    """R(l) = < (m_{t+l} - m_t) eps_t > in bps, with m_t taken as the price BEFORE trade t.

    aggTrades carry no mid, so the pre-trade price is the previous trade's price. With
    pre[t] = lp[t-1], R(1) = < (lp[t] - lp[t-1]) eps_t > is the immediate impact.
    """
    out = {}
    n = len(lp)
    pre = lp[:-1]          # pre[i] = lp[i]   plays the role of m_t for trade t = i+1
    e = eps[1:]            # aligned: e[i] is the sign of trade i+1
    for l in LAGS:
        if n - 1 - l < 1000:
            continue
        post = lp[l:l + len(pre)]
        k = min(len(pre), len(post), len(e))
        out[l] = float(np.mean((post[:k] - pre[:k]) * e[:k]) * 1e4)
    return out


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            eps = np.where(a[:, 1] > 0.5, -1.0, 1.0)
            d = np.empty_like(lp)
            d[0] = 0.0
            d[1:] = np.diff(lp)
            n = len(d)

            # --- the two exponents, same machinery
            Tp, Sp = partial_sum_sd(d)
            H, r2H = slope(Tp, Sp)
            Tf, Sf = partial_sum_sd(eps)
            chi, r2chi = slope(Tf, Sf)

            # --- nulls: destroy temporal dependence, keep the marginals
            nH = [slope(*partial_sum_sd(d[rng.permutation(n)]))[0] for _ in range(REPS)]
            nchi = [slope(*partial_sum_sd(eps[rng.permutation(n)]))[0] for _ in range(REPS)]

            # --- the response function and the immediate impact
            R = response(lp, eps)
            G = R[1]

            # --- the counterfactual, evaluated at every T on the grid
            rows = []
            for t, sd_price, sd_flow in zip(Tp, Sp, Sf):
                actual = sd_price * 1e4                    # bps
                predicted = abs(G) * sd_flow               # bps, permanent impact
                rows.append({"T": int(t),
                             "actual_bps": round(actual, 2),
                             "permanent_impact_bps": round(predicted, 2),
                             "surviving_fraction": round(actual / predicted, 4),
                             "cancelled_fraction": round(1.0 - actual / predicted, 4)})
            cx, _ = slope([r["T"] for r in rows], [r["surviving_fraction"] for r in rows])

            per[sym] = {
                "n_trades": int(n),
                "H_price": round(H, 4), "r2_H": round(r2H, 4),
                "H_null_mean": round(float(np.mean(nH)), 4),
                "H_null_sd": round(float(np.std(nH, ddof=1)), 4),
                "H_z": round((H - float(np.mean(nH))) / float(np.std(nH, ddof=1)), 1),
                "chi_flow": round(chi, 4), "r2_chi": round(r2chi, 4),
                "chi_null_mean": round(float(np.mean(nchi)), 4),
                "chi_z": round((chi - float(np.mean(nchi))) / float(np.std(nchi, ddof=1)), 1),
                "chi_minus_H": round(chi - H, 4),
                "G_lag1_impact_bps": round(G, 5),
                "response_function_bps": {str(k): round(v, 4) for k, v in R.items()},
                "response_ratio_2048_over_1": (round(R[2048] / G, 3)
                                               if 2048 in R and G != 0 else None),
                "counterfactual": rows,
                "surviving_fraction_exponent": round(cx, 4),
                "predicted_exponent_H_minus_chi": round(H - chi, 4),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T29", "lane": "C", "utc": _utc(), "reps": REPS, "seed": SEED,
           "source": ("Bouchaud, Trades Quotes and Prices: Eq. (8.7) for R(l); Sec. 10.4 for the "
                      "efficiency paradox and its stated resolution"),
           "counterfactual": ("if impact were permanent at G bps per trade, price dispersion over "
                              "T trades would be G * sd(sum_T eps); the ratio of actual to that "
                              "is the surviving fraction and goes as T^(H - chi)"),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C29_EFFICIENCY_PARADOX_V1.json").write_text(json.dumps(art, indent=2),
                                                        encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    brief = {s: {k: per[s][k] for k in
                 ("H_price", "H_null_mean", "H_z", "chi_flow", "chi_z", "chi_minus_H",
                  "G_lag1_impact_bps", "response_ratio_2048_over_1",
                  "surviving_fraction_exponent", "predicted_exponent_H_minus_chi")}
             for s in SYMS}
    sys.stdout.write(json.dumps(brief, indent=2).encode(enc, "replace").decode(enc, "replace")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
