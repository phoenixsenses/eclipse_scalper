# -*- coding: utf-8 -*-
"""C-T29 -- THE FINE BALANCE, TESTED FOR THE FIRST TIME WITH BOTH INPUTS CALIBRATED.

Sec 490 closed the fine balance as NOT EVALUABLE because neither input was identified.  Two
runs have since changed that:

    C-T27  kappa-chi got a real-data null and a recovery test: 3/3 supported, 7-13 sigma
           outside a constant-lambda null, recovery bias +0.003, sd 0.029-0.032
    C-T28  gamma got a recovery test at the real n: sd 0.019-0.033, and a systematic
           SHRINKAGE toward about 0.45 that must be inverted before the number is used

C-T28's bias curve is the piece that makes this possible.  The estimator does not return the
true gamma; it pulls every value toward the middle of the fit range:

    true   0.20  0.30  0.40  0.50  0.60
    BTC    .282  .342  .422  .487  .559        bias crosses zero near true gamma ~ 0.45

So the published gamma must be DE-BIASED by inverting that curve before (1-gamma)/2 is formed.
Doing so, and propagating both standard errors, gives the first honest test of

    kappa - chi  =  beta  =  (1 - gamma)/2

WHAT A REJECTION WOULD AND WOULD NOT MEAN, stated before the numbers.  This is a COMPOSITE
hypothesis with two legs:
    (i)  kappa - chi = beta         -- the propagator's difference prediction, Sec 13.4.3
    (ii) beta = (1 - gamma)/2       -- the fine balance itself, Eq 13.17
and the book is explicit that the propagator gets chi and kappa INDIVIDUALLY wrong while
claiming only the difference is right.  A rejection therefore does not say which leg failed.
It is reported as a rejection of the composite, not of the fine balance.

Inputs are read from the two published artifacts; nothing is re-measured.
ESTIMATION.  Ceiling: RECONCILIATION.

  python -m tools.ct29_fine_balance_now_that_both_inputs_are_calibrated --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "reports/atlas"
GAMMA_JSON = "reports/atlas/CT28_GAMMA_RECOVERY_V1.json"
KX_JSON = "reports/atlas/CT27_KAPPA_CHI_NULL_V1.json"


def debias(fitted, rows):
    """invert the recovery curve: find the true gamma whose fitted mean equals `fitted`"""
    ts = sorted(float(k) for k in rows)
    ms = [rows[str(t)]["mean"] for t in ts]
    if fitted <= ms[0]:
        return ts[0] + (fitted - ms[0]) * (ts[1] - ts[0]) / (ms[1] - ms[0])
    if fitted >= ms[-1]:
        return ts[-1] + (fitted - ms[-1]) * (ts[-1] - ts[-2]) / (ms[-1] - ms[-2])
    return float(np.interp(fitted, ms, ts))


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    G = json.load(open(GAMMA_JSON, encoding="utf-8"))
    K = json.load(open(KX_JSON, encoding="utf-8"))
    res = {"inputs": {"gamma": GAMMA_JSON, "kappa_minus_chi": KX_JSON},
           "identity": "kappa - chi = beta = (1 - gamma)/2",
           "composite_legs": ["kappa-chi = beta (Sec 13.4.3, propagator difference)",
                              "beta = (1-gamma)/2 (Eq 13.17, the fine balance)"],
           "caveat": "a rejection does not say which leg failed",
           "per_symbol": {}, "ceiling": "RECONCILIATION"}
    print("=== the fine balance, both inputs calibrated ===", flush=True)
    print("    sym       gamma_fit  gamma_true  beta_pred  sd    kx_meas   sd     z", flush=True)
    for sym in G["per_symbol"]:
        g = G["per_symbol"][sym]
        k = K["per_symbol"].get(sym)
        if k is None:
            continue
        gf = g["real_gamma"]
        gt = debias(gf, g["recovery"])
        sd_g = g["typical_sd"]
        beta = (1.0 - gt) / 2.0
        sd_beta = sd_g / 2.0
        kx = k["kappa_minus_chi_real"]
        sd_kx = k["null_A_constant_lambda"]["sd"]
        d = kx - beta
        sd_d = float(np.sqrt(sd_beta ** 2 + sd_kx ** 2))
        z = d / sd_d
        row = {"gamma_fitted": gf, "gamma_debiased": gt, "gamma_sd": sd_g,
               "beta_predicted": beta, "beta_sd": sd_beta,
               "kappa_minus_chi": kx, "kappa_minus_chi_sd": sd_kx,
               "difference": d, "sd_of_difference": sd_d, "z": z,
               "rejects_composite_at_2sigma": bool(abs(z) > 2.0)}
        res["per_symbol"][sym] = row
        print("    %-9s %8.4f  %9.4f  %9.4f %.4f %8.4f %.4f %+6.2f  %s"
              % (sym, gf, gt, beta, sd_beta, kx, sd_kx, z,
                 "REJECT" if row["rejects_composite_at_2sigma"] else "no reject"), flush=True)

    rej = [s for s, v in res["per_symbol"].items() if v["rejects_composite_at_2sigma"]]
    res["summary"] = {"rejected": rej, "n_rejected": len(rej),
                      "n_total": len(res["per_symbol"])}
    res["tokens"] = [
        "FINE_BALANCE_IS_NOW_TESTABLE_BOTH_INPUTS_CALIBRATED",
        "GAMMA_ESTIMATOR_SHRINKS_TOWARD_0_45_AND_MUST_BE_DEBIASED",
        "GAMMA_PRECISION_IS_FINE_SD_0_02_TO_0_03",
        "COMPOSITE_REJECTED_ON_%d_OF_%d" % (len(rej), len(res["per_symbol"])),
        "REJECTION_DOES_NOT_IDENTIFY_WHICH_LEG_FAILED",
    ]
    print("    => composite rejected on %d of %d: %s"
          % (len(rej), len(res["per_symbol"]), rej or "none"), flush=True)
    for t in res["tokens"]:
        print("    " + t, flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT29_FINE_BALANCE_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
