# -*- coding: utf-8 -*-
"""C-T28 -- CAN THE gamma ESTIMATOR RECOVER A KNOWN gamma?  Closing ERR-HU-022 / ERR-HU-024.

C-T27 gave kappa-chi a real-data null plus a recovery test and it passed 3/3.  gamma is the
remaining input to the fine balance and has had neither.  C-T26's gamma null was run where
C(l) is PURE NOISE, and ERR-HU-024 suspended its conclusion precisely because C-T27 had just
shown, 27-fold, that a null built where the signal is absent says nothing about precision
where the signal is strong.

WHAT DOES AND DOES NOT NEED TESTING HERE.  The EXISTENCE of long memory is not in question:
the real C(100) = 0.083 sits about 145 standard errors above the noise floor, and an AR(1)
with the measured C(1) = 0.26 would put C(10) at 1.4e-6, i.e. indistinguishable from zero
across the entire fit range.  So a "no long memory" null is uninformative by construction.
What is in question is the PRECISION of the fitted exponent, and precision is measured by
recovery, not by a null.

CONSTRUCTION.  A sign series with a PRESCRIBED long-memory exponent, built the standard way:
ARFIMA(0,d,0) filter coefficients psi_k ~ k^(d-1) applied to white noise by FFT, then the
SIGN is taken.  For fractional noise C(l) ~ l^(2d-1), so gamma = 1 - 2d; and taking signs of
a Gaussian process preserves the exponent, because the sign correlation is
(2/pi) arcsin(rho) ~= (2/pi) rho for the small rho that matters at large lags.

Everything else mirrors the real pipeline exactly: the same n per symbol (taken at the 200 ms
merge, which is where the real gamma was fitted), the same lag grid, the same fit range
[10, 1000], the same positive-lags-only selection, and the same pooling of numerator and
denominator sums across seven day-blocks.

PREREGISTERED QUESTIONS, fixed before reading:
  Q1  bias and sd of the fitted gamma at each true gamma in {0.2, 0.3, 0.4, 0.5, 0.6}
  Q2  given that sd, is the measured 0.373 distinguishable from the book's 0.5?
  Q3  is BTC's 0.373 distinguishable from ETH's 0.369?  (expected NO -- and if so, that
      confirms ERR-HU-022's PRECISION claim honestly, by recovery rather than by a
      miscalibrated null)

No DB, no market data.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct28_gamma_recovery_at_the_real_n --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "reports/atlas"
LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
N_DAYS = 7
N_PER_DAY = {"BTCUSDT": 235_811, "ETHUSDT": 228_063, "SOLUSDT": 151_877}  # 200 ms merge
TRUE_GAMMAS = (0.2, 0.3, 0.4, 0.5, 0.6)
N_SIM = 40
RNG_SEED = 20260827
REAL = {"BTCUSDT": 0.407, "ETHUSDT": 0.379, "SOLUSDT": 0.411}   # C-T19, 200 ms merge
BOOK_GAMMA = 0.5


def arfima_signs(n, d, rng, ktrunc=1 << 16):
    """signs of an ARFIMA(0,d,0) series: psi_k ~ k^(d-1), applied by FFT."""
    k = np.arange(1, ktrunc)
    psi = np.concatenate([[1.0], np.exp((d - 1.0) * np.log(k))])
    m = 1 << int(np.ceil(np.log2(n + len(psi))))
    e = rng.standard_normal(m)
    x = np.fft.irfft(np.fft.rfft(e) * np.fft.rfft(psi, m), m)[len(psi):len(psi) + n]
    return np.sign(x)


def acf_sums(x, lags):
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    return {L: (float(np.sum(xc[L:] * xc[:-L])), den) for L in lags if len(xc) > L + 10}


def fit_gamma(cs):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and cs[L] > 0]
    if len(ls) < 4:
        return None
    A = np.column_stack([np.ones(len(ls)), np.log(ls)])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log([cs[L] for L in ls]))
    return float(-c[1])


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"true_gammas": list(TRUE_GAMMAS), "n_sim": N_SIM, "n_days": N_DAYS,
           "n_per_day": N_PER_DAY, "fit_range": [FIT_LO, FIT_HI],
           "real_gamma_CT19_200ms": REAL, "book_gamma": BOOK_GAMMA,
           "closes": ["ERR-HU-022", "ERR-HU-024"],
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    for sym, npd in N_PER_DAY.items():
        rows = {}
        print("=== %s   n/day %d x %d days" % (sym, npd, N_DAYS), flush=True)
        for g in TRUE_GAMMAS:
            d = (1.0 - g) / 2.0
            got = []
            for _ in range(N_SIM):
                acc = {}
                for _day in range(N_DAYS):
                    s = arfima_signs(npd, d, rng)
                    for L, (nu, de) in acf_sums(s, LAGS).items():
                        a0, b0 = acc.get(L, (0.0, 0.0))
                        acc[L] = (a0 + nu, b0 + de)
                cs = {L: nu / de for L, (nu, de) in acc.items() if de > 0}
                v = fit_gamma(cs)
                if v is not None:
                    got.append(v)
            rows[str(g)] = {"n_ok": len(got), "mean": float(np.mean(got)),
                            "sd": float(np.std(got)), "bias": float(np.mean(got) - g)}
            print("    true %.2f  ->  fitted %.4f +- %.4f   bias %+.4f   (%d/%d fits)"
                  % (g, rows[str(g)]["mean"], rows[str(g)]["sd"], rows[str(g)]["bias"],
                     len(got), N_SIM), flush=True)
        sds = [rows[str(g)]["sd"] for g in TRUE_GAMMAS]
        biases = [rows[str(g)]["bias"] for g in TRUE_GAMMAS]
        sd_typ = float(np.mean(sds))
        real = REAL[sym]
        # Q2: distinguishable from the book's 0.5?
        near = min(TRUE_GAMMAS, key=lambda g: abs(g - BOOK_GAMMA))
        z_book = abs(real - (BOOK_GAMMA + rows[str(near)]["bias"])) / sd_typ
        out = {"recovery": rows, "typical_sd": sd_typ,
               "mean_abs_bias": float(np.mean(np.abs(biases))),
               "real_gamma": real,
               "z_vs_book_0.5": float(z_book),
               "distinguishable_from_book": bool(z_book > 2.0)}
        res["per_symbol"][sym] = out
        print("    typical sd %.4f   mean |bias| %.4f   real %.3f -> z vs book 0.5 = %.2f  "
              "=> %s" % (sd_typ, out["mean_abs_bias"], real, z_book,
                         "DISTINGUISHABLE" if out["distinguishable_from_book"]
                         else "NOT distinguishable"), flush=True)

    P = res["per_symbol"]
    if "BTCUSDT" in P and "ETHUSDT" in P:
        sd = float(np.mean([P["BTCUSDT"]["typical_sd"], P["ETHUSDT"]["typical_sd"]]))
        diff = abs(P["BTCUSDT"]["real_gamma"] - P["ETHUSDT"]["real_gamma"])
        z = diff / (sd * np.sqrt(2))
        res["BTC_vs_ETH"] = {"difference": diff, "sd_of_difference": float(sd * np.sqrt(2)),
                             "z": float(z), "distinguishable": bool(z > 2.0)}
        print("=== Q3  BTC %.3f vs ETH %.3f: diff %.4f, sd of diff %.4f, z %.2f => %s"
              % (P["BTCUSDT"]["real_gamma"], P["ETHUSDT"]["real_gamma"], diff,
                 sd * np.sqrt(2), z,
                 "DISTINGUISHABLE" if z > 2.0 else "NOT distinguishable"), flush=True)
        # what the fine balance prediction's uncertainty becomes
        res["fine_balance_prediction_sd"] = float(sd / 2.0)
        print("    => (1-gamma)/2 therefore carries sd %.4f, against kappa-chi's 0.03"
              % (sd / 2.0), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT28_GAMMA_RECOVERY_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
