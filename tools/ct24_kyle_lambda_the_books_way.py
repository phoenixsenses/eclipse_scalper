# -*- coding: utf-8 -*-
"""C-T24 -- kappa - chi THE BOOK'S WAY, TO LOCATE A CROSS-LANE DISAGREEMENT.

The joint table in Sec 486 left one number unexplained, and it is the largest one there:

    kappa - chi     BTC   A 0.255  vs  C 0.300      gap 0.045
                    ETH   A 0.361  vs  C 0.250      gap 0.111

The two lanes disagree with EACH OTHER by more than either disagrees with the fine-balance
prediction, which is exactly why that test could not separate gamma = 0.37 from the book's
gamma = 0.5.

The corpus defines the quantity from the INNER region, not from a global fit.  Sec 11.4:

    "The slope of the linear region of R(dV,T) is usually called Kyle's lambda ...
     R(dV,T) ~ Lambda(T) dV  as |dV| -> 0;   Lambda(T) ~= Lambda(1) T^-(kappa-chi)"

A-S30's T-cutoff ladder (T>=1, 2, 5, 10, 20) is consistent with that route.  C-T21 instead
took kappa and chi from a GLOBAL collapse grid search over the whole binned curve -- which
mixes F's linear part with its concave part, the same defect ERR-HU-012 already recorded for
zeta.  So the suspect estimator here is C's, not A's, and my previous message had the
direction backwards.

THIS IS A 2x2, NOT A REPEAT.  A ran the inner-region estimator on DL-002.  This runs the SAME
estimator on C's pipeline (raw_trades_v1 zips + book_ticker).  Same estimator, different data:

    result lands near A's 0.255 / 0.361  -> the lane gap is ESTIMATOR-driven; C-T21's
                                            collapse value is the odd one out
    result lands near C's 0.300 / 0.250  -> the gap is DATA-driven and both estimators agree
                                            within a lane

DECLARED IN ADVANCE, so it is not tuned afterwards:
  - Lambda(T) is the OLS slope of R on dV restricted to |dV| BELOW ITS WITHIN-T MEDIAN.  The
    inner quartile is reported as a robustness check.  Neither is chosen after seeing the
    exponent.
  - The T cutoff is a researcher degree of freedom and A said so out loud: the T>=20 cut moved
    BTC from outside the book's band to inside.  Both T>=1 and T>=20 are printed.
  - Clock: market-order time (Sec 11.4's own choice), non-overlapping windows.
  - A's stated reason for a cutoff is reproduced too: T_min = (tick/sigma_tilde)^2, which
    binds only SOL.

ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct24_kyle_lambda_the_books_way --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D
from tools import hb4_is_a_liquidation_special as B4

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
T_LIST = (1, 2, 5, 10, 20, 50, 100, 200)
INNER = {"median": 50.0, "quartile": 25.0}
A_S30 = {"BTCUSDT": 0.255, "ETHUSDT": 0.361, "SOLUSDT": 0.193}
C_T21 = {"BTCUSDT": 0.300, "ETHUSDT": 0.250, "SOLUSDT": 0.100}
BOOK_BAND = (0.25, 0.30)


def lam(dv, dp, pct):
    """Kyle's lambda: OLS slope of dp on dv inside the |dv| percentile cut, no intercept
    forced -- the intercept is kept and reported implicitly by the fit."""
    ok = np.isfinite(dv) & np.isfinite(dp)
    dv, dp = dv[ok], dp[ok]
    if len(dv) < 500:
        return None
    cut = np.percentile(np.abs(dv), pct)
    m = np.abs(dv) <= cut
    if m.sum() < 300:
        return None
    X = np.column_stack([np.ones(int(m.sum())), dv[m]])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ dp[m])
    return float(b[1])


def fit_exponent(Ts, L):
    xs = [np.log(t) for t, v in zip(Ts, L) if v is not None and v > 0]
    ys = [np.log(v) for v in L if v is not None and v > 0]
    if len(xs) < 3:
        return None, None
    A = np.column_stack([np.ones(len(xs)), np.array(xs)])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.array(ys))
    pred = A @ np.array(c)
    ss = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
    return float(-c[1]), (float(1 - np.sum((np.array(ys) - pred) ** 2) / ss)
                          if ss > 0 else None)


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    res = {"days": list(DAYS), "T_list": list(T_LIST),
           "clock": "market-order time, non-overlapping windows",
           "estimand": "Lambda(T) = slope of R on dV in the inner region; "
                       "Lambda(T) ~ T^-(kappa-chi)  (Sec 11.4)",
           "design": "same estimator as A-S30, different data pipeline (2x2)",
           "A_S30": A_S30, "C_T21_collapse": C_T21, "book_band": list(BOOK_BAND),
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    for sym in H2.SYMBOLS:
        DV = {T: [] for T in T_LIST}
        DP = {T: [] for T in T_LIST}
        tick_bps, sig1 = None, None
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo - 2000, hi + 60000)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            bid = np.array([r[1] for r in rows], float)
            ask = np.array([r[2] for r in rows], float)
            del rows
            mid = 0.5 * (bid + ask)
            if tick_bps is None:
                dd = np.abs(np.diff(np.unique(np.round(bid, 8))))
                tick_bps = float(np.min(dd[dd > 0]) / np.median(mid) * 1e4)

            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            ots, oeps = ts[idx], eps[idx]
            ovol = np.add.reduceat(qty, idx)
            del ts, px, eps, qty
            ib = np.searchsorted(bts, ots, side="left") - 1
            ok = ib >= 0
            oeps, ovol, ib = oeps[ok], ovol[ok], ib[ok]
            m = mid[ib]
            if sig1 is None:
                sig1 = float(np.std((m[1:] / m[:-1] - 1.0) * 1e4))
            sv = oeps * ovol
            for T in T_LIST:
                k = len(m) // T
                if k < 30:
                    continue
                DV[T].append(sv[:k * T].reshape(k, T).sum(axis=1))
                st = m[:k * T:T]
                en = np.concatenate([m[T:k * T:T], [m[k * T - 1]]])
                DP[T].append((en / st - 1.0) * 1e4)
            del bts, bid, ask, mid

        if not DV[T_LIST[0]]:
            continue
        t_min = (tick_bps / sig1) ** 2 if sig1 else None
        out = {"tick_bps": tick_bps, "sigma_tilde_1_bps": sig1, "T_min": t_min,
               "lambda": {}, "kappa_minus_chi": {}}
        print("=== %s   tick %.4f bps   sigma_tilde(1) %.4f   T_min %.2f"
              % (sym, tick_bps, sig1, t_min), flush=True)
        for name, pct in INNER.items():
            L = []
            for T in T_LIST:
                v = (lam(np.concatenate(DV[T]), np.concatenate(DP[T]), pct)
                     if DV[T] else None)
                L.append(v)
            out["lambda"][name] = {str(T): v for T, v in zip(T_LIST, L)}
            print("    Lambda(T) [%s]  " % name +
                  "  ".join("T%d %s" % (T, "%.3e" % v if v else "n/a")
                            for T, v in zip(T_LIST, L)), flush=True)
            for cut in (1, 20):
                Ts = [T for T in T_LIST if T >= cut]
                Ls = [L[T_LIST.index(T)] for T in Ts]
                e, r2 = fit_exponent(Ts, Ls)
                out["kappa_minus_chi"]["%s_T>=%d" % (name, cut)] = {
                    "value": e, "r2": r2}
                print("        kappa-chi (T>=%-2d) %s   r2 %s"
                      % (cut, "%+.4f" % e if e is not None else "n/a",
                         "%.3f" % r2 if r2 is not None else "n/a"), flush=True)
        best = out["kappa_minus_chi"].get("median_T>=20", {}).get("value")
        if best is not None:
            da, dc = abs(best - A_S30[sym]), abs(best - C_T21[sym])
            out["closer_to"] = "A-S30" if da < dc else "C-T21"
            out["gap_to_A"], out["gap_to_C"] = da, dc
            print("    -> this run %+.4f | A-S30 %.3f (gap %.3f) | C-T21 %.3f (gap %.3f)"
                  "  => closer to %s"
                  % (best, A_S30[sym], da, C_T21[sym], dc, out["closer_to"]), flush=True)
        res["per_symbol"][sym] = out

    votes = [v.get("closer_to") for v in res["per_symbol"].values() if v.get("closer_to")]
    res["verdict"] = {
        "votes": votes,
        "conclusion": ("ESTIMATOR_DRIVEN_C_T21_IS_THE_OUTLIER"
                       if votes.count("A-S30") > votes.count("C-T21")
                       else "DATA_DRIVEN_OR_UNRESOLVED")}
    print("VERDICT  votes %s  -> %s" % (votes, res["verdict"]["conclusion"]), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT24_KYLE_LAMBDA_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
