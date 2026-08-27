# -*- coding: utf-8 -*-
"""S47 -- is the 3.27 effective bets of §475 a measurement or an artifact?

§475 froze a feasibility verdict on one number: trace(C)/lambda_max = 3.27 across 186
symbols.  That estimator was MY choice.  Lopez de Prado MLAM §2 says two things about a
correlation matrix used this way, and §475 did neither:

  §2.2  the Marcenko-Pastur theorem holds for 1 < T/N.  Eigenvalues below
        lambda_+ = (1 + sqrt(N/T))^2 are NOISE and carry no information.
  §2.6  "Detoning is the principal components analogue to computing beta-adjusted
        returns."  A cross-sectional test's signs are cross-sectionally balanced, so
        the market mode is hedged away and does not consume a bet.
  §3638 "in a nonexperimental setting, the researcher should denoise and detone."

This measures the same matrix the evaluator builds -- imported, not re-derived, so the
object under test is byte-identical to the one §475 published -- under four treatments.

Reads no outcome: a correlation of returns is a design quantity, and the prereg's §6
requires N_eff to be measured.
"""

import io
import json
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.lane_a_evaluator_v1 import (          # the SAME code path as §475
    universe, daily_sigmas, h_star_days, corr_matrix, effective_bets,
    COV_WINDOW, MIN_SIGMA_BPS, LAWFUL_CUTOFF_MS, K_MEAN_ABS, F_DESIGN, T_BAR, PANEL,
)

try:
    import numpy as np
except ImportError:
    np = None


def jacobi(A, tol=1e-9, sweeps=100):
    """Symmetric eigenvalues by cyclic Jacobi -- no numpy dependency.

    Returns (eigenvalues desc, eigenvectors as columns).  186x186 converges in a few
    sweeps; this exists so the result does not depend on whether numpy is installed,
    which is the kind of thing that silently changes a published number."""
    n = len(A)
    a = [row[:] for row in A]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-14:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    ev = [(a[i][i], [v[k][i] for k in range(n)]) for i in range(n)]
    ev.sort(key=lambda x: -x[0])
    return [e[0] for e in ev], [e[1] for e in ev]


def eig(C):
    if np is not None:
        w, V = np.linalg.eigh(np.array(C))
        idx = w.argsort()[::-1]
        return list(w[idx]), [list(V[:, i]) for i in idx]
    return jacobi(C)


def rebuild(vals, vecs, n):
    """C = sum_i lambda_i v_i v_i'  -- reassemble from a modified spectrum."""
    C = [[0.0] * n for _ in range(n)]
    for lam, v in zip(vals, vecs):
        if lam == 0.0:
            continue
        for i in range(n):
            li = lam * v[i]
            row = C[i]
            for j in range(i, n):
                row[j] += li * v[j]
    for i in range(n):
        for j in range(i):
            C[i][j] = C[j][i]
    return C


def renorm(C, n):
    """Rescale to unit diagonal -- MLAM §2.6's final step.  A detoned matrix whose
    diagonal is not renormalised is not a correlation matrix, and trace(C)/lambda_max
    silently means something else."""
    d = [math.sqrt(C[i][i]) if C[i][i] > 1e-12 else 1e-6 for i in range(n)]
    return [[C[i][j] / (d[i] * d[j]) for j in range(n)] for i in range(n)]


def enb(vals, n):
    """Two estimators, reported side by side.

      trace/lambda_max   -- what §475 used; measures distance from rank-1
      exp(entropy)       -- the spectral-entropy count; uses the WHOLE spectrum,
                            not just its top, and is the one that answers
                            'how many directions actually carry variance'
    """
    tr = sum(vals)
    pos = [v for v in vals if v > 1e-12]
    p = [v / tr for v in pos]
    H = -sum(x * math.log(x) for x in p if x > 0)
    return {"trace_over_lmax": tr / max(vals) if max(vals) > 0 else 0.0,
            "exp_entropy": math.exp(H), "rank": len(pos), "trace": tr,
            "lmax": max(vals), "n": n}


def main():
    con = sqlite3.connect("file:%s?mode=ro" % PANEL.replace("\\", "/"), uri=True)
    end = LAWFUL_CUTOFF_MS
    lo = end - COV_WINDOW * 86400000
    keep, _ = universe(con, end)
    sig = daily_sigmas(con, lo, end)
    hs = sorted([(s, sig[s], h_star_days(sig[s], 10.0)) for s in keep
                 if sig.get(s) and sig[s] >= MIN_SIGMA_BPS], key=lambda x: x[2])
    syms = [x[0] for x in hs]
    C, days = corr_matrix(con, syms, lo, end)
    con.close()

    n = len(syms)
    T = days - 1                     # returns, not days
    print("THE MATRIX §475 PUBLISHED A VERDICT ON")
    print("  symbols N %d   return observations T %d   T/N %.3f" % (n, T, float(T) / n))
    lam_plus = (1.0 + math.sqrt(float(n) / T)) ** 2
    lam_minus = (1.0 - math.sqrt(float(n) / T)) ** 2
    print("  Marcenko-Pastur holds for 1 < T/N.  Here T/N = %.3f." % (float(T) / n))
    print("  MP noise band  [%.3f, %.3f]  (lambda_+ = (1+sqrt(N/T))^2)" % (lam_minus, lam_plus))
    if T <= n:
        print("  *** T <= N: the sample correlation matrix is SINGULAR by construction.")
        print("      rank <= T = %d of %d.  %d eigenvalues are exactly zero and carry" % (T, n, n - T))
        print("      no information whatever.  MLAM's theorem does not cover this regime.")

    vals, vecs = eig(C)
    n_signal = sum(1 for v in vals if v > lam_plus)
    print()
    print("SPECTRUM")
    print("  top 8 eigenvalues " + " ".join("%.2f" % v for v in vals[:8]))
    print("  above lambda_+ %d of %d   <- the number of NON-NOISE factors" % (n_signal, n))
    print("  variance in the top mode %.1f%%" % (100.0 * vals[0] / sum(vals)))

    res = {}
    res["raw"] = enb(vals, n)

    # denoise: MLAM §2.5 residual-eigenvalue method -- every noise eigenvalue is
    # replaced by their common average, which preserves the trace exactly.
    noise = [v for v in vals[n_signal:]]
    avg = sum(noise) / len(noise) if noise else 0.0
    dn_vals = vals[:n_signal] + [avg] * len(noise)
    res["denoised"] = enb(dn_vals, n)

    # detone: MLAM §2.6 -- drop the market eigenvector, renormalise to unit diagonal.
    dt_vals = [0.0] + dn_vals[1:]
    Cd = renorm(rebuild(dt_vals, vecs, n), n)
    dvals, _ = eig(Cd)
    res["denoised_detoned"] = enb(dvals, n)

    print()
    print("EFFECTIVE BETS UNDER FOUR TREATMENTS")
    print("  %-20s %14s %14s %8s" % ("", "trace/lambda_max", "exp(entropy)", "rank"))
    for k in ("raw", "denoised", "denoised_detoned"):
        r = res[k]
        print("  %-20s %14.2f %14.2f %8d" % (k, r["trace_over_lmax"], r["exp_entropy"], r["rank"]))
    print("  %-20s %14.2f %14s %8s" % ("MP signal count", n_signal, "-", "-"))

    print()
    print("WHAT EACH DOES TO §475's FEASIBILITY")
    s_trade = K_MEAN_ABS * F_DESIGN / 2.0
    hmed = hs[len(hs) // 2][2]
    print("  per-trade Sharpe k*f/2 %.6f   median h* %.2f d   (both unchanged)" % (s_trade, hmed))
    print("  %-24s %10s %12s %14s" % ("effective bets from", "value", "S_pooled", "years to t=2"))
    rows = [("§475 (raw, trace/lmax)", res["raw"]["trace_over_lmax"]),
            ("raw, exp(entropy)", res["raw"]["exp_entropy"]),
            ("denoised+detoned t/l", res["denoised_detoned"]["trace_over_lmax"]),
            ("denoised+detoned entropy", res["denoised_detoned"]["exp_entropy"]),
            ("MP signal count", float(n_signal))]
    out = []
    for lab, eb in rows:
        sp = math.sqrt(365.0 / hmed) * s_trade * math.sqrt(eb)
        yrs = (T_BAR / sp) ** 2
        need = F_DESIGN * (yrs / 1.0) ** 0.25
        print("  %-24s %10.2f %12.4f %14s" % (lab, eb, sp, "%s y" % format(int(yrs), ",")))
        out.append({"basis": lab, "eff_bets": eb, "S_pooled": sp, "years": yrs,
                    "f_for_one_year": need})

    print()
    print("REQUIRED CAPTURE FOR A ONE-YEAR VERDICT (the f_design-free inversion)")
    for r in out:
        print("  %-24s f >= %6.2f%%" % (r["basis"], r["f_for_one_year"] * 100))
    print("  every single-leg capture ever measured here: 1-2%% (A-S14); best cell 2.09%% (A-S43)")

    doc = {"n_symbols": n, "T": T, "T_over_N": float(T) / n,
           "lambda_plus": lam_plus, "n_signal": n_signal,
           "top_mode_share": vals[0] / sum(vals), "treatments": res,
           "feasibility": out, "median_h_star_days": hmed,
           "s_per_trade": s_trade, "numpy": np is not None}
    p = "reports/research/h2_response_shape_v1/S47_EFFECTIVE_BETS_V1.json"
    io.open(p, "w", encoding="utf-8").write(json.dumps(doc, indent=1))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()


def null_test(n, T, trials=3, seed=20260827):
    """THE CHECK THAT DECIDES WHICH ESTIMATOR IS A MEASUREMENT.

    Five estimators disagreed by a factor of 55 on the real matrix.  A statistic that
    returns the same value on pure noise is not measuring the market; it is measuring
    its own construction.  Same N, same T, iid normal returns, identical pipeline."""
    import random
    rnd = random.Random(seed)
    lam_plus = (1.0 + math.sqrt(float(n) / T)) ** 2
    acc = {}
    for _ in range(trials):
        R = [[rnd.gauss(0, 1) for _ in range(T)] for _ in range(n)]
        mu = [sum(r) / T for r in R]
        sd = [math.sqrt(sum((x - mu[i]) ** 2 for x in R[i]) / (T - 1)) for i in range(n)]
        C = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                cov = sum((R[i][t] - mu[i]) * (R[j][t] - mu[j]) for t in range(T)) / (T - 1)
                C[i][j] = C[j][i] = cov / (sd[i] * sd[j])
        vals, vecs = eig(C)
        ns = sum(1 for v in vals if v > lam_plus)
        noise = vals[ns:]
        avg = sum(noise) / len(noise) if noise else 0.0
        dn = vals[:ns] + [avg] * len(noise)
        dt = [0.0] + dn[1:]
        dvals, _ = eig(renorm(rebuild(dt, vecs, n), n))
        for k, v in (("raw t/l", enb(vals, n)["trace_over_lmax"]),
                     ("raw entropy", enb(vals, n)["exp_entropy"]),
                     ("dn+dt t/l", enb(dvals, n)["trace_over_lmax"]),
                     ("dn+dt entropy", enb(dvals, n)["exp_entropy"]),
                     ("MP signal count", float(ns))):
            acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}
