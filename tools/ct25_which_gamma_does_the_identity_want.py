# -*- coding: utf-8 -*-
"""C-T25 -- THE IDENTITY WANTS A DIFFERENT gamma: SIGN, OR SIGNED VOLUME?

Sec 489 reopened the fine balance with a sharp tension: gamma agrees between BTC and ETH to
0.004, so (1-gamma)/2 predicts essentially the same beta for both, while the measured
kappa - chi differ by 0.13 (0.2398 vs 0.3698, now agreed between lanes).

Re-reading Sec 13.4.3's own premise shows a candidate cause.  The book writes:

    "we assume that the SIGNED VOLUME v_n = eps_n upsilon_n of each transaction n is a
     Gaussian variable with zero mean and UNIT VARIANCE, and long-range autocorrelations
     given by C(l) ~ l^-gamma"

The gamma that enters kappa - chi = beta is therefore the decay of the SIGNED-VOLUME
autocorrelation.  C-T19 measured the decay of the SIGN autocorrelation.  Those are the same
object only if order size is independent of the sign structure and has finite variance --
and on this estate H-U8 measured Hill(v) = 1.30-1.79, i.e. BELOW 2, so v has INFINITE
VARIANCE and the book's "unit variance Gaussian" premise is violated outright.

PREREGISTERED PREDICTION, derived before measuring.  If the tension is caused by using the
wrong gamma, then inverting kappa - chi = (1-gamma)/2 on the agreed values gives what the
identity WANTS:

    gamma_wanted(BTC) = 1 - 2(0.2398) = 0.5204
    gamma_wanted(ETH) = 1 - 2(0.3698) = 0.2604       a gap of 0.260

against gamma_sign's gap of 0.004.  So the hypothesis predicts a LARGE and SIGN-SPECIFIC
split: gamma_v(BTC) > gamma_v(ETH), by roughly 0.26.  Any other outcome -- a small gap, or
the wrong direction -- refutes it.

ESTIMATORS.  Because Var[v] is infinite, the raw correlation is not consistent: it is
dominated by a handful of observations.  Three are computed and compared rather than one
being chosen:

    sign      C(l) on eps                      -- C-T19's estimator, reproduced
    raw       C(l) on eps * upsilon            -- the book's literal variable; unstable by
                                                  construction here, reported to show it
    rank      C(l) on eps * (rank(upsilon)/N)  -- bounded, finite variance, keeps the ordering
    winsor    C(l) on eps * min(upsilon, q99)  -- bounded, keeps the scale

A jackknife dropping the top 0.1 percent of |v| is run on the raw estimator; if gamma_raw
moves materially, that estimator is struck rather than reported.

No DB is opened -- this needs only the trade stream.
ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct25_which_gamma_does_the_identity_want --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import hb4_is_a_liquidation_special as B4

OUT = "reports/atlas"
DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
KX_AGREED = {"BTCUSDT": 0.2398, "ETHUSDT": 0.3698, "SOLUSDT": None}
GAMMA_SIGN_CT19 = {"BTCUSDT": 0.373, "ETHUSDT": 0.369, "SOLUSDT": None}
JACK_TOP_FRAC = 0.001


def acf_sums(x, lags):
    """returns dict lag -> (numerator, denominator) so days can be pooled"""
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    out = {}
    for L in lags:
        if len(xc) <= L + 10:
            continue
        out[L] = (float(np.sum(xc[L:] * xc[:-L])), den)
    return out


def fit_gamma(cs):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and cs[L] > 0]
    if len(ls) < 4:
        return None, None
    A = np.column_stack([np.ones(len(ls)), np.log(ls)])
    y = np.log([cs[L] for L in ls])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    pred = A @ c
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(-c[1]), (float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else None)


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    wanted = {s: (1 - 2 * v) if v is not None else None for s, v in KX_AGREED.items()}
    res = {"days": list(DAYS), "lags": list(LAGS), "fit_range": [FIT_LO, FIT_HI],
           "book_premise": "Sec 13.4.3 assumes SIGNED VOLUME v = eps*upsilon, unit variance",
           "kappa_minus_chi_agreed": KX_AGREED,
           "gamma_wanted_by_identity": wanted,
           "gamma_sign_CT19": GAMMA_SIGN_CT19,
           "prediction": "gamma_v(BTC) > gamma_v(ETH) by about 0.26 if the tension is a "
                         "wrong-gamma artefact",
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}
    print("IDENTITY WANTS:  BTC %.4f   ETH %.4f   (gap %.3f)   vs gamma_sign gap 0.004"
          % (wanted["BTCUSDT"], wanted["ETHUSDT"],
             wanted["BTCUSDT"] - wanted["ETHUSDT"]), flush=True)

    for sym in H2.SYMBOLS:
        acc = {k: {} for k in ("sign", "raw", "rank", "winsor", "raw_jack")}
        hill = None
        for day in DAYS:
            try:
                ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            except Exception as exc:
                print("    %s %s SKIP (%s)" % (sym, day, exc), flush=True)
                continue
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            oeps = eps[idx].astype(float)
            ovol = np.add.reduceat(qty, idx)
            del ts, px, eps, qty
            if hill is None:
                a = np.sort(ovol[ovol > 0])
                k = max(200, int(0.005 * len(a)))
                hill = float(1.0 / np.mean(np.log(a[-k:] / a[-k])))
            r = (np.argsort(np.argsort(ovol)) + 1.0) / len(ovol)
            q99 = float(np.percentile(ovol, 99))
            series = {"sign": oeps,
                      "raw": oeps * ovol,
                      "rank": oeps * r,
                      "winsor": oeps * np.minimum(ovol, q99)}
            v = series["raw"]
            cut = float(np.percentile(np.abs(v), 100 * (1 - JACK_TOP_FRAC)))
            series["raw_jack"] = np.where(np.abs(v) <= cut, v, 0.0)
            for k, x in series.items():
                s = acf_sums(x, LAGS)
                for L, (nu, de) in s.items():
                    a0, b0 = acc[k].get(L, (0.0, 0.0))
                    acc[k][L] = (a0 + nu, b0 + de)

        if not acc["sign"]:
            continue
        out = {"hill_v": hill, "var_finite": bool(hill and hill > 2.0), "gamma": {},
               "C_lags": {}}
        print("=== %s   Hill(v) %.3f   Var[v] finite? %s"
              % (sym, hill, out["var_finite"]), flush=True)
        for k in ("sign", "raw", "raw_jack", "rank", "winsor"):
            cs = {L: nu / de for L, (nu, de) in acc[k].items() if de > 0}
            g, r2 = fit_gamma(cs)
            out["gamma"][k] = {"gamma": g, "r2": r2}
            out["C_lags"][k] = {str(L): cs[L] for L in sorted(cs)}
            print("    %-9s gamma %s  r2 %s   C(1) %+.4f  C(100) %+.5f"
                  % (k, "%.4f" % g if g is not None else "n/a",
                     "%.3f" % r2 if r2 is not None else "n/a",
                     cs.get(1, float("nan")), cs.get(100, float("nan"))), flush=True)
        gr, gj = out["gamma"]["raw"]["gamma"], out["gamma"]["raw_jack"]["gamma"]
        out["raw_estimator_stable"] = bool(gr is not None and gj is not None
                                           and abs(gr - gj) < 0.05)
        print("    raw vs raw_jack (top %.1f%% zeroed): %s vs %s -> raw estimator %s"
              % (100 * JACK_TOP_FRAC, "%.4f" % gr if gr else "n/a",
                 "%.4f" % gj if gj else "n/a",
                 "STABLE" if out["raw_estimator_stable"] else "STRUCK"), flush=True)
        res["per_symbol"][sym] = out

    P = res["per_symbol"]
    if "BTCUSDT" in P and "ETHUSDT" in P:
        print("=== DOES ANY gamma VARIANT PRODUCE THE WANTED GAP (~0.26, BTC > ETH)? ===",
              flush=True)
        verdict = {}
        for k in ("sign", "raw", "raw_jack", "rank", "winsor"):
            b = P["BTCUSDT"]["gamma"][k]["gamma"]
            e = P["ETHUSDT"]["gamma"][k]["gamma"]
            if b is None or e is None:
                continue
            gap = b - e
            verdict[k] = {"BTC": b, "ETH": e, "gap": gap,
                          "right_direction": bool(gap > 0),
                          "size_ok": bool(abs(gap - 0.260) < 0.10)}
            print("    %-9s BTC %.4f  ETH %.4f  gap %+.4f   direction %s   size %s"
                  % (k, b, e, gap, "OK" if gap > 0 else "WRONG",
                     "OK" if abs(gap - 0.260) < 0.10 else "no"), flush=True)
        res["verdict"] = verdict
        any_ok = any(v["right_direction"] and v["size_ok"] for v in verdict.values())
        res["hypothesis"] = ("SUPPORTED_BY_AT_LEAST_ONE_VARIANT" if any_ok
                             else "REFUTED_NO_VARIANT_PRODUCES_THE_GAP")
        print("    => %s" % res["hypothesis"], flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT25_WHICH_GAMMA_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
