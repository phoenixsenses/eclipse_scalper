# -*- coding: utf-8 -*-
"""C-T27 -- THE kappa-chi NULL, AT THE REAL SIGNAL-TO-NOISE.  Closing ERR-HU-021.

C-T26 showed the kappa-chi estimator can return anything in [-1.52, +0.92] when the true
exponent is zero -- but at a hand-picked synthetic noise level, which is why that run
explicitly refused to conclude kappa-chi is zero.  ERR-HU-021 recorded the real requirement:
the null has to be built ON THE REAL DATA, at the real scatter, or it decides nothing.

That is what this does.  Two calibrations, both surrogate constructions on the measured
(dV, dP) pairs, so the heavy tails, the per-T residual scale, the counts and the binning are
the real ones and only the thing under test is imposed:

  NULL A -- CONSTANT LAMBDA.  Fit one lambda_0 pooled across all T.  Rebuild
            dP_null = lambda_0 * dV + resampled residuals of that T.
            True kappa-chi is exactly 0 BY CONSTRUCTION.  Run the identical pipeline.
            Question: does the real value fall outside this distribution?

  NULL B -- RECOVERY.  Impose Lambda(T) = lambda_0 * T^(-0.25), i.e. a true kappa-chi of
            0.25, sitting in the book's own band.  Same residual resampling.
            Question: does the pipeline give 0.25 back, and with what spread?

A null alone cannot support a number -- it can only fail to reject.  The recovery test is
what says whether the estimator could have SEEN a real effect of the size claimed.  Running
only the first is the mistake C-T26 stopped short of.

PREREGISTERED VERDICT RULE, fixed before any number is read:
    SUPPORTED   real kappa-chi lies OUTSIDE null A's 95 percent interval
                AND null B recovers 0.25 with |bias| < 0.05
    NOT_SUPPORTED otherwise, with the failing leg named.

Corpus motivation: Sec 11.4 claims "the empirical determination of kappa - chi ~= 0.25-0.3 is
MORE ROBUST AND STABLE across assets than chi and kappa independently".  That is a claim about
an estimator's behaviour, and a null plus a recovery test is how such a claim is checked.

Clock declared (Sec 474): market-order time, non-overlapping windows, inner-median cut,
T >= 20 -- identical to C-T24 so the real value is reproduced inside this run rather than
imported.

ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct27_kappa_minus_chi_null_at_the_real_noise --i-have-approval
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
T_FIT_MIN = 20
INNER_PCT = 50.0
N_SIM = 120
TRUE_KX_B = 0.25
RNG_SEED = 20260827
REAL_CT24 = {"BTCUSDT": 0.2245, "ETHUSDT": 0.3786, "SOLUSDT": 0.2032}


def slope(dv, dp, pct=INNER_PCT):
    cut = np.percentile(np.abs(dv), pct)
    m = np.abs(dv) <= cut
    if m.sum() < 300:
        return None
    X = np.column_stack([np.ones(int(m.sum())), dv[m]])
    return float((np.linalg.pinv(X.T @ X) @ (X.T @ dp[m]))[1])


def exponent(Ts, Ls):
    ok = [(t, l) for t, l in zip(Ts, Ls) if l is not None and l > 0]
    if len(ok) < 3:
        return None
    A = np.column_stack([np.ones(len(ok)), np.log([t for t, _ in ok])])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log([l for _, l in ok]))
    return float(-c[1])


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "T_list": list(T_LIST), "T_fit_min": T_FIT_MIN,
           "n_sim": N_SIM, "true_kx_for_recovery": TRUE_KX_B,
           "closes": "ERR-HU-021",
           "verdict_rule": "SUPPORTED iff real outside null A's 95% interval AND null B "
                           "recovers 0.25 with |bias| < 0.05",
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    for sym in H2.SYMBOLS:
        DV = {T: [] for T in T_LIST}
        DP = {T: [] for T in T_LIST}
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
            mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
            del rows
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            oeps, ovol = eps[idx], np.add.reduceat(qty, idx)
            ots = ts[idx]
            del ts, px, eps, qty
            ib = np.searchsorted(bts, ots, side="left") - 1
            ok = ib >= 0
            oeps, ovol, ib = oeps[ok], ovol[ok], ib[ok]
            m = mid[ib]
            sv = oeps * ovol
            for T in T_LIST:
                k = len(m) // T
                if k < 30:
                    continue
                DV[T].append(sv[:k * T].reshape(k, T).sum(axis=1))
                st = m[:k * T:T]
                en = np.concatenate([m[T:k * T:T], [m[k * T - 1]]])
                DP[T].append((en / st - 1.0) * 1e4)
            del bts, mid

        if not DV[T_LIST[0]]:
            continue
        dv = {T: np.concatenate(DV[T]) for T in T_LIST if DV[T]}
        dp = {T: np.concatenate(DP[T]) for T in T_LIST if DP[T]}
        Ts_fit = [T for T in sorted(dv) if T >= T_FIT_MIN]

        # real
        L_real = [slope(dv[T], dp[T]) for T in Ts_fit]
        kx_real = exponent(Ts_fit, L_real)

        # per-T residuals from the real fit, and a pooled lambda_0
        resid, lam_T = {}, {}
        for T in sorted(dv):
            s = slope(dv[T], dp[T])
            lam_T[T] = s
            resid[T] = dp[T] - (s if s else 0.0) * dv[T]
        lam0 = float(np.mean([lam_T[T] for T in Ts_fit if lam_T[T] is not None]))

        def run(kx_true):
            Ls = []
            for T in Ts_fit:
                lam = lam0 * (T ** (-kx_true)) / (Ts_fit[0] ** (-kx_true))
                r = resid[T][rng.integers(0, len(resid[T]), len(resid[T]))]
                Ls.append(slope(dv[T], lam * dv[T] + r))
            return exponent(Ts_fit, Ls)

        nullA = [v for v in (run(0.0) for _ in range(N_SIM)) if v is not None]
        nullB = [v for v in (run(TRUE_KX_B) for _ in range(N_SIM)) if v is not None]
        a_lo, a_hi = float(np.percentile(nullA, 2.5)), float(np.percentile(nullA, 97.5))
        b_mean, b_sd = float(np.mean(nullB)), float(np.std(nullB))
        bias = b_mean - TRUE_KX_B
        outside = bool(kx_real is not None and (kx_real < a_lo or kx_real > a_hi))
        recovers = bool(abs(bias) < 0.05)
        out = {"kappa_minus_chi_real": kx_real, "ct24_published": REAL_CT24.get(sym),
               "lambda_0": lam0,
               "null_A_constant_lambda": {"mean": float(np.mean(nullA)),
                                          "sd": float(np.std(nullA)),
                                          "p2.5": a_lo, "p97.5": a_hi, "n": len(nullA)},
               "null_B_recovery_of_0.25": {"mean": b_mean, "sd": b_sd, "bias": bias,
                                           "n": len(nullB)},
               "real_outside_null_A": outside, "recovers_within_0.05": recovers,
               "verdict": ("SUPPORTED" if (outside and recovers) else "NOT_SUPPORTED")}
        if not outside:
            out["failing_leg"] = "real value is INSIDE the constant-lambda null"
        elif not recovers:
            out["failing_leg"] = "estimator does not recover a known 0.25 (bias %.3f)" % bias
        res["per_symbol"][sym] = out
        print("=== %s   real kappa-chi %+.4f   (C-T24 published %.4f)  lambda_0 %.4e"
              % (sym, kx_real, REAL_CT24.get(sym, float("nan")), lam0), flush=True)
        print("    NULL A (constant lambda, true 0):  mean %+.4f  sd %.4f  "
              "95%% [%+.4f, %+.4f]"
              % (out["null_A_constant_lambda"]["mean"], out["null_A_constant_lambda"]["sd"],
                 a_lo, a_hi), flush=True)
        print("    NULL B (recover a true 0.25):      mean %+.4f  sd %.4f  bias %+.4f"
              % (b_mean, b_sd, bias), flush=True)
        print("    real outside null A? %s   recovers 0.25? %s   => %s%s"
              % (outside, recovers, out["verdict"],
                 "  (" + out.get("failing_leg", "") + ")" if "failing_leg" in out else ""),
              flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT27_KAPPA_CHI_NULL_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
