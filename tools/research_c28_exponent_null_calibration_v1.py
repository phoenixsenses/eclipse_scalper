r"""LANE C, round 28 -- the null of my own exponent table.

C-T27 published a rule for other people's gates: measure what the gate returns when there is
nothing there, BEFORE freezing pass/fail. C-T23 -- the round that closed this lane's charter --
never had that applied to it. This round applies it, to C-T23.

WHAT PROMPTED IT. reports/atlas/CT26_NULL_CALIBRATION_V1.json, written by another lane, reports
a null for `kappa_minus_chi` with mean -0.093 and sd 0.872. Every kappa-chi C-T23 published
(+0.0009 / -0.1035 / -0.0693) sits far inside one sd of that. Their RAW values differ from mine
(0.2245 vs 0.0009 on BTC), so they measured a different estimator and their null is not
transferable -- but the question it raises transfers exactly, and I never asked it.

THE NULL. Not a shuffle that destroys impact: impact is not in doubt and a null without it is
answering a question nobody asked. The null here is a process in which impact is REAL and the
marginals are REAL, and only the exponents are trivial:

    take the per-trade pairs (d_i, sv_i) -- log return into trade i, signed notional of trade i --
    and permute the PAIR INDEX jointly.

This preserves the return distribution (heavy tails and all), the volume distribution, and the
contemporaneous return-to-flow relation exactly. It destroys temporal dependence only. Under it
the CLT fixes every exponent at 1/2:

    dV(T) = sum of T iid signed volumes      -> sd ~ T^0.5   -> chi   = 0.5
    r(T)  = sum of T iid returns             -> E|r| ~ T^0.5 -> alpha = 0.5
    R(T)  = E[sign(dV) r], both ~ T^0.5      ->              -> kappa = 0.5

so p = kappa - alpha = 0, kappa - chi = 0, and chi - alpha = 0. Those three zeros are the null
values of the three quantities C-T23 built its verdict on.

WHAT IS AND IS NOT BEING TESTED. The identity p - (kappa-chi) = chi - alpha_E|r| is ALGEBRA. It
holds in every replicate and needs no null. What needs a null is the substantive claim C-T23
attached to it: that the measured chi - alpha_E|r| is NONZERO, which is the whole ground on
which p and kappa-chi were declared different quantities.

METHOD NOTE ON A SELECTION EFFECT. loglog() drops non-positive y. Under the null R(T) stays
positive because sign(dV) and r keep their real correlation, but any replicate that loses a
point is recorded rather than silently dropped, and the finite-fit rate is reported.

Verification first: the same estimator is re-run on the unshuffled data and must reproduce
C-T23's published table before any null is read.
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
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
SEED = 20260827

# what C-T23 published, for the reproduction check
CT23 = {
    "BTCUSDT": {"kappa": 0.6507, "alpha_E_abs_r": 0.6765, "chi": 0.6498,
                "p_direct": -0.0258, "kappa_minus_chi": 0.0009, "chi_minus_alpha": -0.0267},
    "ETHUSDT": {"kappa": 0.5782, "alpha_E_abs_r": 0.5924, "chi": 0.6817,
                "p_direct": -0.0141, "kappa_minus_chi": -0.1035, "chi_minus_alpha": 0.0893},
    "SOLUSDT": {"kappa": 0.5209, "alpha_E_abs_r": 0.5206, "chi": 0.5902,
                "p_direct": 0.0003, "kappa_minus_chi": -0.0693, "chi_minus_alpha": 0.0696},
}
KEYS = ("kappa", "alpha_E_abs_r", "chi", "p_direct", "kappa_minus_chi", "chi_minus_alpha")
NULL_VALUE = {"kappa": 0.5, "alpha_E_abs_r": 0.5, "chi": 0.5,
              "p_direct": 0.0, "kappa_minus_chi": 0.0, "chi_minus_alpha": 0.0}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def loglog(x, y):
    """identical to C-T23's, including the y > 0 filter"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), 0
    A = np.column_stack([np.ones(len(x)), np.log(x)])
    b, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    return float(b[1]), int(len(x))


def exponents(d, sv):
    """C-T23's contemporaneous() re-expressed on increments instead of levels.

    C-T23 used r = log(px[iT + T - 1] / px[iT]) * 1e4 and dv = sum(sv[iT : iT + T]).
    With lp = cumsum(d) that first-to-last log ratio is lp[iT + T - 1] - lp[iT], so the two
    forms are the same statistic and the shuffle acts on exactly what it should.
    """
    lp = np.cumsum(d)
    n = len(d)
    Tv, Rv, Ev, Cv, Fv = [], [], [], [], []
    for T in WINDOW_T:
        m = n // T
        if m < 200:
            continue
        dv = sv[:m * T].reshape(m, T).sum(axis=1)
        w = lp[:m * T].reshape(m, T)
        r = (w[:, -1] - w[:, 0]) * 1e4
        ok = np.isfinite(r) & np.isfinite(dv)
        dv, r = dv[ok], r[ok]
        if len(r) < 200:
            continue
        R = float(np.mean(np.sign(dv) * r))
        Er = float(np.mean(np.abs(r)))
        Tv.append(T)
        Rv.append(R)
        Ev.append(Er)
        Cv.append(float(np.std(dv, ddof=1)))
        Fv.append(abs(R / Er) if Er > 0 else float("nan"))
    kap, nk = loglog(Tv, Rv)
    aer, _ = loglog(Tv, Ev)
    chi, _ = loglog(Tv, Cv)
    p, _ = loglog(Tv, Fv)
    return {"kappa": kap, "alpha_E_abs_r": aer, "chi": chi, "p_direct": p,
            "kappa_minus_chi": kap - chi, "chi_minus_alpha": chi - aer,
            "n_points_kappa": nk}


def load(con, sym):
    rows = con.execute("select price,notional,is_buyer_maker from agg_trades "
                       "where symbol=? order by ts_ms limit ?", (sym, NROWS)).fetchall()
    a = np.array(rows, dtype=np.float64)
    px, nt, bm = a[:, 0], a[:, 1], a[:, 2]
    lp = np.log(px)
    d = np.empty_like(lp)
    d[0] = 0.0
    d[1:] = np.diff(lp)
    sv = np.where(bm > 0.5, -1.0, 1.0) * nt
    return d, sv


def summarise(vals, obs, null_val):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 10:
        return {"n_finite": int(len(v)), "insufficient": True}
    sd = float(v.std(ddof=1))
    return {"n_finite": int(len(v)),
            "null_mean": round(float(v.mean()), 4),
            "null_sd": round(sd, 4),
            "null_p05": round(float(np.percentile(v, 5)), 4),
            "null_p95": round(float(np.percentile(v, 95)), 4),
            "theory_null": null_val,
            "observed": round(float(obs), 4),
            "z_vs_null_mean": (round((float(obs) - float(v.mean())) / sd, 2)
                               if sd > 0 else None),
            "two_sided_p": round(float((np.abs(v - v.mean())
                                        >= abs(float(obs) - v.mean())).mean()), 4)}


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            d, sv = load(con, sym)
            obs = exponents(d, sv)
            repro = {k: {"c_t23": CT23[sym][k], "here": round(obs[k], 4),
                         "abs_diff": round(abs(obs[k] - CT23[sym][k]), 4)} for k in KEYS}
            worst = max(repro[k]["abs_diff"] for k in KEYS)

            draws = {k: [] for k in KEYS}
            gdraws = {k: [] for k in KEYS}
            n = len(d)
            # correlation of the real pairs, to be reproduced by the Gaussian control
            rho = float(np.corrcoef(d, sv)[0, 1])
            for _ in range(REPS):
                idx = rng.permutation(n)
                e = exponents(d[idx], sv[idx])
                for k in KEYS:
                    draws[k].append(e[k])
                # SAME estimator, SAME rho, but LIGHT tails: isolates what the tails do
                z1 = rng.standard_normal(n)
                z2 = rng.standard_normal(n)
                gd = z1 * float(np.std(d))
                gs = (rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2) * float(np.std(sv))
                ge = exponents(gd, gs)
                for k in KEYS:
                    gdraws[k].append(ge[k])
            per[sym] = {
                "n_trades": int(n),
                "reproduction_of_c_t23": repro,
                "worst_abs_reproduction_diff": round(worst, 4),
                "reproduced": bool(worst <= 0.005),
                "pair_correlation": round(rho, 5),
                "null": {k: summarise(draws[k], obs[k], NULL_VALUE[k]) for k in KEYS},
                "gaussian_control": {k: summarise(gdraws[k], obs[k], NULL_VALUE[k])
                                     for k in KEYS},
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T28", "lane": "C", "utc": _utc(), "reps": REPS, "seed": SEED,
           "gaussian_control": ("the same estimator on iid GAUSSIAN pairs carrying the same "
                               "correlation: if it returns 0.5/0/0 while the real-marginal "
                               "shuffle does not, the gap is caused by the tails, not by the "
                               "machinery"),
           "null": ("joint permutation of the per-trade pairs (d_i, sv_i): impact and both "
                    "marginals preserved, temporal dependence destroyed, so every exponent is "
                    "1/2 by the CLT and all three differences are 0"),
           "prompted_by": ("reports/atlas/CT26_NULL_CALIBRATION_V1.json (another lane) reports a "
                           "kappa_minus_chi null of sd 0.872; its raw values differ from mine, so "
                           "its null is not transferable, but the question is"),
           "per_symbol": per}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C28_EXPONENT_NULL_CALIBRATION_V1.json").write_text(
        json.dumps(art, indent=2), encoding="utf-8")
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(json.dumps(
        {s: {"reproduced": per[s]["reproduced"],
             "worst_diff": per[s]["worst_abs_reproduction_diff"],
             "null": {k: per[s]["null"][k] for k in
                      ("chi_minus_alpha", "kappa_minus_chi", "p_direct", "kappa")},
             "gauss": {k: {kk: per[s]["gaussian_control"][k].get(kk)
                           for kk in ("null_mean", "null_sd")}
                       for k in ("chi_minus_alpha", "kappa_minus_chi", "p_direct", "kappa")}}
         for s in SYMS}, indent=2).encode(enc, "replace").decode(enc, "replace") + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
