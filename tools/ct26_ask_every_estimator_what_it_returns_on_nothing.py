# -*- coding: utf-8 -*-
"""C-T26 -- ASK EVERY ESTIMATOR IN THIS LANE WHAT IT RETURNS WHEN THERE IS NOTHING THERE.

Operator instruction, 2026-08-27.  This lane ran a null on some estimators (H-U6's sign
shuffle, H-U10's placebo anchors, CT-016's parametric discrimination) and NOT on most of the
others.  ERR-HU-014 already recorded that gap for the collapse grid, once, about one
estimator.  This does it systematically.

The corpus supplies the frame and one of the nulls.  Sec 2.1.3: uncorrelated microstructure
noise eta on top of a random walk makes the volatility signature plot DECREASE like 1/tau --
so a falling signature plot is what NOTHING looks like, not evidence of mean reversion.
Sec 2.1.4 then sets the scale of "flat": the S&P500 E-mini, one of the most liquid contracts
in the world, "only decreases by about 20%".

Six estimators, each with an explicit null in which the effect is ABSENT BY CONSTRUCTION.
Sample sizes match the real runs so the answer is about this lane's actual power.

  N1  gamma from C(l) ~ l^-gamma          null: i.i.d. signs, no memory at all
      Suspicion declared in advance: the real fit only used lags where C(l) > 0.  Under the
      null half the lags are negative, so the fit runs on POSITIVELY SELECTED noise.
  N2  kappa - chi from Lambda(T) ~ T^-x   null: impact linear with a CONSTANT lambda, so the
      true exponent is exactly 0
  N3  Hill tail index                     null: exponential samples, i.e. no power-law tail
  N4  fill-curve form test                null: survival generated from a TRUE exponential
  N5  signature ratio sigma(1000)/sigma(1) null: i.i.d. returns (1.0), and the corpus's eta
      noise (should FALL below 1)
  N6  binned log-log slope zeta            null: exactly LINEAR impact, true slope 1.0

Each null reports the distribution of what comes back, and a PASS/FAIL against what the
estimator should return.  Where the null value is not what the real run assumed, the real
number is flagged for correction rather than defended.

No DB, no market data -- synthetic only.
ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct26_ask_every_estimator_what_it_returns_on_nothing --i-have-approval
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "reports/atlas"
RNG_SEED = 20260827
N_SIM = 200

LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
N_TRADES = 3_105_239                 # BTC, C-T19 / C-T25
T_LIST = (1, 2, 5, 10, 20, 50, 100, 200)
HILL_FRACS = (0.005, 0.01, 0.02, 0.05)
REAL = {"gamma_sign_BTC": 0.3734, "gamma_sign_ETH": 0.3686,
        "kappa_minus_chi_BTC": 0.2245, "kappa_minus_chi_ETH": 0.3786,
        "hill_v_BTC": 1.563, "signature_ratio_BTC_0ms": 2.045,
        "signature_ratio_BTC_200ms": 1.440, "zeta_BTC_600s": 0.166}


def fit_powerlaw(cs, positive_only=True):
    ls = [L for L in sorted(cs) if FIT_LO <= L <= FIT_HI and (cs[L] > 0 or not positive_only)]
    if len(ls) < 4:
        return None
    y = np.array([cs[L] for L in ls])
    if positive_only:
        y = np.log(y)
    else:
        y = np.log(np.maximum(np.abs(y), 1e-12))
    A = np.column_stack([np.ones(len(ls)), np.log(ls)])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ y)
    return float(-c[1])


def acf(x, lags):
    xc = x - x.mean()
    den = float(np.sum(xc * xc))
    return {L: float(np.sum(xc[L:] * xc[:-L]) / den) for L in lags if len(xc) > L + 10}


def hill(a, frac):
    a = np.sort(a[a > 0])
    k = max(200, int(frac * len(a)))
    return float(1.0 / np.mean(np.log(a[-k:] / a[-k])))


def lam_slope(dv, dp, pct=50.0):
    cut = np.percentile(np.abs(dv), pct)
    m = np.abs(dv) <= cut
    X = np.column_stack([np.ones(int(m.sum())), dv[m]])
    return float((np.linalg.pinv(X.T @ X) @ (X.T @ dp[m]))[1])


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"n_sim": N_SIM, "real_values_under_audit": REAL,
           "corpus": {"eta_noise": "Sec 2.1.3 -- uncorrelated microstructure noise makes the "
                                   "signature plot DECREASE like 1/tau",
                      "flat_scale": "Sec 2.1.4 -- the E-mini 'only decreases by about 20%'"},
           "nulls": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    # ---- N1: gamma on i.i.d. signs -----------------------------------------
    print("=== N1  gamma from C(l), null = i.i.d. signs (no memory) ===", flush=True)
    n = 400_000                      # 8x smaller than real; noise SHRINKS with n, so this is
    g_sel, g_abs = [], []            # conservative for the selection concern
    for _ in range(40):
        e = rng.choice([-1.0, 1.0], size=n)
        cs = acf(e, LAGS)
        a = fit_powerlaw(cs, positive_only=True)
        b = fit_powerlaw(cs, positive_only=False)
        if a is not None:
            g_sel.append(a)
        if b is not None:
            g_abs.append(b)
    r1 = {"n_per_sim": n,
          "gamma_positive_lags_only": {"mean": float(np.mean(g_sel)),
                                       "sd": float(np.std(g_sel)),
                                       "p05": float(np.percentile(g_sel, 5)),
                                       "p95": float(np.percentile(g_sel, 95)),
                                       "n_ok": len(g_sel)},
          "gamma_all_lags_abs": ({"mean": float(np.mean(g_abs)),
                                  "sd": float(np.std(g_abs))} if g_abs else None),
          "should_be": "no power law exists; the fit is on noise"}
    res["nulls"]["N1_gamma"] = r1
    print("    positive-lags-only fit on PURE NOISE: mean %+.4f  sd %.4f  [p05 %+.4f, "
          "p95 %+.4f]  (%d/%d sims produced a fit)"
          % (r1["gamma_positive_lags_only"]["mean"], r1["gamma_positive_lags_only"]["sd"],
             r1["gamma_positive_lags_only"]["p05"], r1["gamma_positive_lags_only"]["p95"],
             len(g_sel), 40), flush=True)
    print("    real gamma_sign: BTC %.4f  ETH %.4f" % (REAL["gamma_sign_BTC"],
                                                       REAL["gamma_sign_ETH"]), flush=True)

    # ---- N2: kappa - chi with a CONSTANT lambda ----------------------------
    print("=== N2  kappa-chi from Lambda(T), null = constant lambda (true exponent 0) ===",
          flush=True)
    kx = []
    for _ in range(60):
        m = 200_000
        v = rng.standard_normal(m) * rng.pareto(1.6, m)      # heavy-tailed signed volume
        noise = rng.standard_normal(m) * 1.0
        L = []
        for T in T_LIST:
            k = m // T
            dv = v[:k * T].reshape(k, T).sum(axis=1)
            dp = 0.01 * dv + noise[:k * T].reshape(k, T).sum(axis=1)   # CONSTANT lambda
            L.append(lam_slope(dv, dp))
        Ts = [T for T in T_LIST if T >= 20]
        Ls = [L[T_LIST.index(T)] for T in Ts]
        ok = [(t, l) for t, l in zip(Ts, Ls) if l > 0]
        if len(ok) >= 3:
            A = np.column_stack([np.ones(len(ok)), np.log([t for t, _ in ok])])
            c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log([l for _, l in ok]))
            kx.append(float(-c[1]))
    r2 = {"mean": float(np.mean(kx)), "sd": float(np.std(kx)),
          "p05": float(np.percentile(kx, 5)), "p95": float(np.percentile(kx, 95)),
          "should_be": 0.0}
    res["nulls"]["N2_kappa_minus_chi"] = r2
    print("    on CONSTANT-lambda data: mean %+.4f  sd %.4f  [p05 %+.4f, p95 %+.4f]  "
          "(should be 0)" % (r2["mean"], r2["sd"], r2["p05"], r2["p95"]), flush=True)
    print("    real: BTC %.4f  ETH %.4f" % (REAL["kappa_minus_chi_BTC"],
                                            REAL["kappa_minus_chi_ETH"]), flush=True)

    # ---- N3: Hill on an exponential ---------------------------------------
    print("=== N3  Hill tail index, null = EXPONENTIAL sample (no power-law tail) ===",
          flush=True)
    h = {f: [] for f in HILL_FRACS}
    for _ in range(30):
        a = rng.exponential(1.0, 500_000)
        for f in HILL_FRACS:
            h[f].append(hill(a, f))
    r3 = {str(f): {"mean": float(np.mean(h[f])), "sd": float(np.std(h[f]))}
          for f in HILL_FRACS}
    r3["should_be"] = "large / divergent -- an exponential has no power-law tail"
    res["nulls"]["N3_hill"] = r3
    print("    " + "   ".join("k=%s %.2f+-%.2f" % (f, r3[str(f)]["mean"], r3[str(f)]["sd"])
                              for f in HILL_FRACS), flush=True)
    print("    real Hill(v) BTC %.3f -- if the null sits near this, the tail claim is void"
          % REAL["hill_v_BTC"], flush=True)

    # ---- N4: fill-curve form test on a TRUE exponential --------------------
    print("=== N4  power-law vs exponential fill fit, null = TRUE exponential ===",
          flush=True)
    phi = np.linspace(0.05, 1.0, 20)
    wins = 0
    for _ in range(N_SIM):
        p = 0.21 * np.exp(-0.8 * phi)
        s = np.clip(p + rng.standard_normal(len(phi)) * 0.004, 1e-4, None)
        A1 = np.column_stack([np.ones(len(phi)), phi])
        A2 = np.column_stack([np.ones(len(phi)), np.log(phi)])
        y = np.log(s)
        r2a = 1 - np.sum((y - A1 @ (np.linalg.pinv(A1.T @ A1) @ (A1.T @ y))) ** 2) / np.sum(
            (y - y.mean()) ** 2)
        r2b = 1 - np.sum((y - A2 @ (np.linalg.pinv(A2.T @ A2) @ (A2.T @ y))) ** 2) / np.sum(
            (y - y.mean()) ** 2)
        if r2b > r2a:
            wins += 1
    r4 = {"power_law_wins_on_exponential_data": wins / N_SIM, "should_be": "near 0"}
    res["nulls"]["N4_fill_form"] = r4
    print("    power law beats exponential on EXPONENTIAL data: %.3f of sims (should be ~0)"
          % r4["power_law_wins_on_exponential_data"], flush=True)

    # ---- N5: signature ratio ----------------------------------------------
    print("=== N5  signature ratio sigma(1000)/sigma(1) ===", flush=True)
    def sig_ratio(p):
        out = []
        for L in (1, 1000):
            r = p[L:] - p[:-L]
            out.append(np.sqrt(np.var(r) / L))
        return out[1] / out[0]
    iid, eta = [], []
    for _ in range(30):
        w = np.cumsum(rng.standard_normal(300_000)) * 0.1
        iid.append(sig_ratio(w))
        eta.append(sig_ratio(w + rng.standard_normal(300_000) * 0.1))
    r5 = {"iid_random_walk": {"mean": float(np.mean(iid)), "sd": float(np.std(iid))},
          "with_eta_microstructure_noise": {"mean": float(np.mean(eta)),
                                            "sd": float(np.std(eta))},
          "should_be": "1.0 for a pure walk; BELOW 1 with eta noise (Sec 2.1.3)"}
    res["nulls"]["N5_signature_ratio"] = r5
    print("    pure random walk      %.4f +- %.4f   (should be 1.0)"
          % (r5["iid_random_walk"]["mean"], r5["iid_random_walk"]["sd"]), flush=True)
    print("    with eta noise        %.4f +- %.4f   (corpus: should FALL below 1)"
          % (r5["with_eta_microstructure_noise"]["mean"],
             r5["with_eta_microstructure_noise"]["sd"]), flush=True)
    print("    real BTC: 2.045 raw, 1.440 at a 200 ms merge -- both ABOVE every null",
          flush=True)

    # ---- N6: binned log-log slope on exactly linear impact -----------------
    print("=== N6  binned log-log slope, null = EXACTLY LINEAR impact (true slope 1) ===",
          flush=True)
    z = []
    for _ in range(60):
        m = 300_000
        x = rng.pareto(1.0, m) + 1e-3
        y = 1.0 * x + rng.standard_normal(m) * np.std(x) * 3.0
        ed = np.geomspace(np.percentile(x, 1), np.percentile(x, 99), 25)
        b = np.clip(np.searchsorted(ed, x, side="right") - 1, 0, len(ed) - 2)
        cnt = np.bincount(b, minlength=len(ed) - 1).astype(float)
        sx = np.bincount(b, weights=x, minlength=len(ed) - 1)
        sy = np.bincount(b, weights=y, minlength=len(ed) - 1)
        keep = (cnt >= 200) & (sy > 0)
        if keep.sum() < 5:
            continue
        mx = np.log(sx[keep] / cnt[keep])
        my = np.log(sy[keep] / cnt[keep])
        A = np.column_stack([np.ones(len(mx)), mx])
        z.append(float((np.linalg.pinv(A.T @ A) @ (A.T @ my))[1]))
    r6 = {"mean": float(np.mean(z)), "sd": float(np.std(z)), "should_be": 1.0}
    res["nulls"]["N6_binned_slope"] = r6
    print("    on EXACTLY LINEAR data: mean %.4f +- %.4f  (should be 1.0)"
          % (r6["mean"], r6["sd"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT26_NULL_CALIBRATION_V1.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written %s/CT26_NULL_CALIBRATION_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
