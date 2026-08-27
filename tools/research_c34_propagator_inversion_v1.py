r"""LANE C, round 34 -- invert the propagator and test the efficiency condition it must satisfy.

This lane has now measured, separately, all three quantities the book's central model relates:

    R(l)  the response function          C-T29
    C(l)  the order-sign autocorrelation C-T24 / C-T28 (and gamma from it)
    R(dV,T) the aggregate impact         C-T33

Bouchaud Chapter 13 links them. Eq. (13.7) writes the mid-price as a sum over past trades weighted
by a propagator G(.); Eq. (13.10) gives R in terms of G and C. The book then warns off the direct
route explicitly --

    "this direct method is very sensitive to finite-size effects, and therefore provides poor
     estimates of G(l)"

-- and supplies the alternative it recommends:

    S(l) := E[r_{t+l} . eps_t] = R(l+1) - R(l)                       (13.13)
    S(l)  = sum_{n>=0} C(|n - l|) K(n),   K(l) := G(l+1) - G(l)      (13.14)
    S     = C K,  with  C_{n,m} = C(|n - m|)                         (13.15)

so G is recovered by solving a Toeplitz system for K and cumulating. S(l) is measured directly
here rather than differenced out of R, since differencing amplifies exactly the noise the book
is warning about.

THE TEST. Chapter 13's opening argument is that permanent impact plus long-range sign memory would
produce strongly autocorrelated returns, which is not what markets show, so "a large fraction of a
market order's impact must relax over time". The efficiency condition that follows is

    G(l) ~ l^(-beta)   with   beta = (1 - gamma) / 2

and beta is then measurable here WITHOUT using gamma at all. C-T28 measured gamma two ways --
0.4447 / 0.5914 / 0.2164 (variance route) and 0.7746 / 0.7892 / 0.2092 (C-T24's direct C(l) fit) --
which predict beta in [0.11, 0.28] for BTC, [0.10, 0.20] for ETH and about 0.39 for SOL. An
independently inverted beta either lands in those bands or it does not.

INSTRUMENT FIRST. The other C-lane's section 493 established that an uncalibrated null can kill a
correct finding, and C-T31 established that my own scatter estimates were measured in the wrong
world. So the inversion is not read until it has recovered a KNOWN propagator from synthetic data
built on the REAL sign series -- real C, real length, known beta. Bias and spread of the recovered
beta are reported before any real number is quoted.
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
L = 128                      # max lag in the Toeplitz system
FIT_LAGS = (4, 96)           # range over which G is fitted to a power law
RECOVERY_BETAS = (0.10, 0.20, 0.30, 0.40)
BOOT = 24
BLOCK = 100_000
SEED = 20260827

GAMMA = {"BTCUSDT": (0.4447, 0.7746), "ETHUSDT": (0.5914, 0.7892),
         "SOLUSDT": (0.2164, 0.2092)}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sign_autocorr(eps, L):
    """C(l) = E[eps_t eps_{t+l}], l = 0..L"""
    e = eps - eps.mean()
    v = float(np.dot(e, e) / len(e))
    return np.array([1.0] + [float(np.dot(e[:-l], e[l:]) / (len(e) - l) / v)
                             for l in range(1, L + 1)])


def S_of(r, eps, L):
    """S(l) = E[r_{t+l} eps_t] measured directly, l = 0..L-1"""
    out = np.empty(L)
    for l in range(L):
        out[l] = float(np.dot(r[l:], eps[:len(eps) - l]) / (len(eps) - l))
    return out


def invert(S, C):
    """solve the Toeplitz system S = C K for K, then cumulate to G. Returns G, cond(C)."""
    L = len(S)
    M = np.empty((L, L))
    for i in range(L):
        M[i, :] = C[np.abs(np.arange(L) - i)]
    cond = float(np.linalg.cond(M))
    K, *_ = np.linalg.lstsq(M, S, rcond=None)
    G = np.concatenate([[0.0], np.cumsum(K)])       # G(0) = 0, G(l) = sum_{n<l} K(n)
    return G, cond, K


def beta_of(G, lo=FIT_LAGS[0], hi=FIT_LAGS[1]):
    """fit G(l) ~ l^(-beta) over the stated lag range; returns beta (positive = decaying)"""
    l = np.arange(lo, hi + 1)
    g = G[lo:hi + 1]
    ok = g > 0
    if ok.sum() < 8:
        return float("nan")
    x = np.log(l[ok])
    y = np.log(g[ok])
    A = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(-b[1])


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
            r = np.empty_like(lp)
            r[0] = 0.0
            r[1:] = np.diff(lp) * 1e4
            n = len(r)

            C = sign_autocorr(eps, L)

            # ---- INSTRUMENT CHECK: recover a KNOWN propagator using the REAL sign series
            recovery = {}
            for bt in RECOVERY_BETAS:
                Gt = np.concatenate([[0.0], (np.arange(1, L + 1.0)) ** (-bt)])
                Kt = np.diff(Gt)                                  # K(l) = G(l+1) - G(l)
                r_sim = np.convolve(eps, Kt)[:n] + rng.normal(0.0, float(np.std(r)), n)
                G_rec, _, _ = invert(S_of(r_sim, eps, L), C)
                recovery["beta_{0}".format(bt)] = {
                    "true": bt, "recovered": round(beta_of(G_rec), 4)}
            errs = [abs(v["recovered"] - v["true"]) for v in recovery.values()
                    if np.isfinite(v["recovered"])]
            worst = round(max(errs), 4) if errs else None

            # ---- the real inversion
            S = S_of(r, eps, L)
            G, cond, K = invert(S, C)
            beta = beta_of(G)

            bs = []
            nb = n // BLOCK
            for _ in range(BOOT):
                idx = np.concatenate([np.arange(i, i + BLOCK)
                                      for i in rng.integers(0, n - BLOCK, nb)])
                rr, ee = r[idx], eps[idx]
                Gb, _, _ = invert(S_of(rr, ee, L), sign_autocorr(ee, L))
                b_ = beta_of(Gb)
                if np.isfinite(b_):
                    bs.append(b_)

            g_var, g_dir = GAMMA[sym]
            band = sorted([(1 - g_var) / 2.0, (1 - g_dir) / 2.0])
            per[sym] = {
                "instrument_check": recovery,
                "worst_recovery_error": worst,
                "instrument_usable": bool(worst is not None and worst <= 0.10),
                "cond_number_of_C_matrix": round(cond, 1),
                "G1_bare_impact_bps": round(float(G[1]), 5),
                "R1_observed_impact_bps": round(float(S[0]), 5),
                "G1_over_R1": (round(float(G[1] / S[0]), 3) if S[0] != 0 else None),
                "G_at_lags": {str(k): round(float(G[k]), 5)
                              for k in (1, 2, 4, 8, 16, 32, 64, 128)},
                "beta_inverted": round(beta, 4),
                "beta_boot_sd": round(float(np.std(bs, ddof=1)), 4) if len(bs) > 5 else None,
                "beta_predicted_band_from_gamma": [round(band[0], 4), round(band[1], 4)],
                "inside_band": bool(band[0] <= beta <= band[1]),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T34", "lane": "C", "utc": _utc(), "L": L, "fit_lags": list(FIT_LAGS),
           "recipe": "Bouchaud Eqs. (13.13)-(13.15): S measured directly, Toeplitz solve, cumulate",
           "efficiency_condition": "G(l) ~ l^-beta with beta = (1 - gamma)/2",
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C34_PROPAGATOR_INVERSION_V1.json").write_text(json.dumps(art, indent=2),
                                                          encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(json.dumps(per, indent=2).encode(enc, "replace").decode(enc, "replace")
                     + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
