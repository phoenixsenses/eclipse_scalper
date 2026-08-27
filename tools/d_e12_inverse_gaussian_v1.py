# -*- coding: utf-8 -*-
"""D-E12 -- the specification the preregistration named as the one to beat.

Prereg V2 section 7: *"The specification to beat: inverse-Gaussian first passage, ABG 10.3.1
eq (10.2), two free parameters c/sigma and mu/sigma."*  It had not been run.  This runs it.

THE MODEL (ABG 10.3.1).  A Wiener process starts at distance `c` above an absorbing barrier and
diffuses with drift `mu` toward it:

    f(t) = c / (sigma sqrt(2 pi t^3)) * exp( -(c - mu t)^2 / (2 sigma^2 t) )
    S(t) = Phi((c - mu t)/(sigma sqrt t)) - exp(2 c mu / sigma^2) * Phi((-c - mu t)/(sigma sqrt t))

ABG: *"the distribution only depends on these through the functions c/sigma and mu/sigma.  Hence,
from a statistical point of view, there are only TWO free parameters"* -- so sigma = 1 is without
loss of generality and is fixed here.

WHY THE SIGN OF `mu` IS THE WHOLE POINT.  If `mu > 0` the process drifts INTO the barrier and
P(T < inf) = 1.  If `mu < 0` it drifts AWAY and P(T < inf) = exp(2 c mu) < 1 -- ABG's CURE MODEL,
a DEFECTIVE distribution.  `D-E2` predicted that case from H2's PEAK_NOT_OBSERVED and `D-E10`
measured its symptom, P00(tau) = 0.100.  A fitted `mu < 0` would be the mechanism behind the symptom.

WHAT IS FITTED, AND WHAT THAT LICENSES.  The fit is on the CAUSE-SPECIFIC hazard of EDGE_GONE:
event rows contribute f(t), INTERRUPTED and ADMINISTRATIVE rows contribute S(t) as censoring.
Cause-specific hazards ARE identified that way (ABG 3.4.1).  The fitted `S` is NOT a marginal
survival and may not be read as one -- that is the `1 - KM` error the prereg forbids by name.
NEVER_ALIVE rows are EXCLUDED: they never crossed the barrier upward, so they are not a first
passage at all, and their share is reported separately.

AND WHAT A GOOD FIT DOES NOT MEAN.  ABG 10.3.4: convergence to a quasi-stationary distribution makes
"an approximately constant hazard rate ... a common phenomenon for MANY models", so hazard shape
does not identify the mechanism.  A good fit here is consistency, never proof.  The goodness number
is therefore reported against its OWN null: refit on data simulated from the fitted parameters.

Usage:  python tools/d_e12_inverse_gaussian_v1.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.d_e11_p2_p3_v1 import collect  # noqa: E402
from tools.d_e8_evaluator_v1 import (  # noqa: E402
    FLOOR_PRIMARY, K_BPS, TAU_MIN, aalen_johansen, assert_spec_unchanged, mu_tau)

OUT = os.path.join(ROOT, "reports", "atlas", "D_E12_INVERSE_GAUSSIAN_V1.json")
SEED = 20260827
NULL_REPS = 200


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def ig_pdf(t, c, mu):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = t > 0
    tt = t[m]
    out[m] = (c / np.sqrt(2.0 * np.pi * tt ** 3)) * np.exp(-((c - mu * tt) ** 2) / (2.0 * tt))
    return out


def ig_surv(t, c, mu):
    t = np.asarray(t, float)
    out = np.ones_like(t)
    m = t > 0
    tt = t[m]
    a = _phi((c - mu * tt) / np.sqrt(tt))
    b = np.exp(np.clip(2.0 * c * mu, -700, 700)) * _phi((-c - mu * tt) / np.sqrt(tt))
    out[m] = np.clip(a - b, 1e-300, 1.0)
    return out


def p_ever(c, mu):
    """P(T < inf).  1 if the drift is into the barrier, exp(2 c mu) if away -- ABG's cure model."""
    return 1.0 if mu >= 0 else float(math.exp(min(0.0, 2.0 * c * mu)))


def negloglik(c, mu, t_ev, t_cens):
    if c <= 0:
        return 1e18
    ll = np.log(np.clip(ig_pdf(t_ev, c, mu), 1e-300, None)).sum()
    if len(t_cens):
        ll += np.log(ig_surv(t_cens, c, mu)).sum()
    return -ll if np.isfinite(ll) else 1e18


def fit(t_ev, t_cens, c_grid=None, mu_grid=None):
    """Two parameters, so a grid then a local refine is robust and needs no optimiser."""
    c_grid = np.linspace(0.05, 12.0, 240) if c_grid is None else c_grid
    mu_grid = np.linspace(-1.5, 1.5, 301) if mu_grid is None else mu_grid
    best = (1e18, None, None)
    for c in c_grid:
        for mu in mu_grid:
            v = negloglik(c, mu, t_ev, t_cens)
            if v < best[0]:
                best = (v, c, mu)
    _, c0, m0 = best
    for _ in range(4):                      # refine
        cs = np.linspace(max(1e-3, c0 * 0.7), c0 * 1.3, 41)
        ms = np.linspace(m0 - 0.15, m0 + 0.15, 61)
        for c in cs:
            for mu in ms:
                v = negloglik(c, mu, t_ev, t_cens)
                if v < best[0]:
                    best = (v, c, mu)
        _, c0, m0 = best
    return {"c": float(c0), "mu": float(m0), "negloglik": float(best[0]),
            "p_ever_absorbed": round(p_ever(c0, m0), 4)}


def cif_from_fit(g_min, c, mu, share_alive):
    """CIF of EDGE_GONE implied by the fit, on the same grid, scaled by the alive share."""
    return share_alive * (1.0 - ig_surv(g_min, c, mu))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    h = assert_spec_unchanged()
    print("D-E12  IG first passage   prereg sha256 %s VERIFIED" % h[:16])

    rows = collect(FLOOR_PRIMARY, K_BPS)
    t = np.array([r["t_ms"] / 60000.0 for r in rows], float)      # minutes
    c = np.array([r["cause"] for r in rows])
    n = len(rows)
    never = c == "NEVER_ALIVE"
    alive = ~never
    t_ev = t[alive & (c == "EDGE_GONE")]
    t_cs = t[alive & (c != "EDGE_GONE")]
    share_alive = float(alive.mean())
    print("  n=%d  never_alive=%d (%.1f%%)  events=%d  censored=%d"
          % (n, never.sum(), 100 * never.mean(), len(t_ev), len(t_cs)))

    f = fit(t_ev, t_cs)
    print("\nFIT (sigma = 1 WLOG, so these ARE c/sigma and mu/sigma)")
    print("  c_hat  = %+.4f      distance to the barrier, in sigma units" % f["c"])
    print("  mu_hat = %+.4f      %s" % (f["mu"], "drift INTO the barrier"
                                        if f["mu"] > 0 else "drift AWAY -- ABG's CURE MODEL"))
    print("  P(T < inf) = %.4f   %s" % (f["p_ever_absorbed"],
                                        "" if f["mu"] >= 0 else "<- DEFECTIVE by construction"))

    # observed vs fitted CIF_1
    tau = TAU_MIN
    g, p00, cif1, cif2 = aalen_johansen(t * 60000.0, c, tau * 60000.0)
    g_min = g / 60000.0
    fitted = cif_from_fit(g_min, f["c"], f["mu"], share_alive)
    # NEVER_ALIVE enters the observed CIF at t = 0; add the same point mass to the fitted curve
    fitted = fitted + (1.0 - share_alive)
    dev = float(np.max(np.abs(cif1 - fitted)))
    print("\nGOODNESS   max |observed CIF_1 - fitted CIF_1| = %.4f" % dev)

    # ...against its OWN null: refit on data simulated FROM the fit
    rng = np.random.default_rng(SEED)
    devs = []
    for _ in range(NULL_REPS):
        k_ev = len(t_ev)
        u = rng.random(k_ev)
        grid = np.linspace(1e-4, tau, 4000)
        S = ig_surv(grid, f["c"], f["mu"])
        cdf = (1.0 - S) / max(1e-12, (1.0 - S[-1]))
        sim_ev = np.interp(u, cdf, grid)
        sim_t = np.concatenate([sim_ev, t_cs, np.zeros(int(never.sum()))])
        sim_c = np.array(["EDGE_GONE"] * len(sim_ev) + list(c[alive & (c != "EDGE_GONE")])
                         + ["NEVER_ALIVE"] * int(never.sum()))
        gg, pp, cc1, _ = aalen_johansen(sim_t * 60000.0, sim_c, tau * 60000.0)
        ff = cif_from_fit(gg / 60000.0, f["c"], f["mu"], share_alive) + (1.0 - share_alive)
        devs.append(float(np.max(np.abs(cc1 - ff))))
    devs = np.array(devs)
    z = (dev - devs.mean()) / devs.std(ddof=1)
    print("  null (refit on data simulated FROM the fit): %.4f +/- %.4f   p95 %.4f   z %+.2f"
          % (devs.mean(), devs.std(ddof=1), np.percentile(devs, 95), z))

    mu_obs = mu_tau(g, p00)
    print("\n  mu_tau observed %.2f min" % mu_obs)

    res = {"prereg_sha256": h, "n": n, "never_alive": int(never.sum()),
           "share_alive": round(share_alive, 4),
           "n_events": int(len(t_ev)), "n_censored": int(len(t_cs)),
           "fit": f,
           "interpretation": ("mu < 0 means the process drifts AWAY from the barrier, so the "
                              "first-passage distribution is DEFECTIVE -- ABG's cure model, the "
                              "case D-E2 predicted and D-E10 measured as P00(tau) > 0"),
           "goodness_max_abs_cif_deviation": round(dev, 4),
           "goodness_null_mean": round(float(devs.mean()), 4),
           "goodness_null_sd": round(float(devs.std(ddof=1)), 4),
           "goodness_null_p95": round(float(np.percentile(devs, 95)), 4),
           "goodness_z": round(float(z), 2),
           "verdict": ("IG_ADEQUATE" if dev <= np.percentile(devs, 95) else "IG_REJECTED"),
           "mu_tau_observed_min": round(mu_obs, 4),
           "caveat": ("ABG 10.3.4: quasi-stationarity makes many processes converge to the same "
                      "limiting hazard, so an adequate fit is CONSISTENCY, never identification")}
    print("\n  VERDICT: %s" % res["verdict"])
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
