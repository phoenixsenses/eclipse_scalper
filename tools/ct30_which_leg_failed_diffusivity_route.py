# -*- coding: utf-8 -*-
"""C-T30 -- WHICH LEG FAILED?  THE DIFFUSIVITY ROUTE TO beta - beta_c, AND A TEST OF Sigma^2.

C-T29 rejected the composite {kappa-chi = beta} AND {beta = (1-gamma)/2} on 2 of 3 symbols
without saying which leg failed.  Sec 13.2.2 supplies a route that touches neither
kappa - chi nor gamma:

    D_Psi(l) := Psi(l)/l  ~=  Gamma_inf^2 c_inf I(gamma,beta) l^(1 - 2 beta - gamma)

so if the lag-dependent diffusion coefficient grows as l^theta, then theta = 1 - 2 beta -
gamma, and since beta_c = (1 - gamma)/2,

    beta - beta_c  =  -theta/2

theta > 0 (super-diffusive) => beta < beta_c, the fine balance fails toward trends
theta = 0                   => the fine balance holds
theta < 0 (sub-diffusive)   => beta > beta_c

That tests LEG (ii) alone.  Sec 481 had recorded this as blocked because a propagator
inversion needs C(l)'s LEVEL, which is convention-dependent; this route needs only the
signature plot's slope.

BUT THE MODEL HAS A SECOND TERM, AND ON THIS ESTATE IT IS THE WHOLE THING.  The book writes
D(l) = D_Psi(l) + Sigma^2, with Sigma^2 the variance of an i.i.d. public-news term xi_t --
CONSTANT in l.  Figure 13.2's caption reports that for equities "the trade-only contribution
accounts for 0.65-0.8 of the long-term squared volatility".  H-U11 measured the opposite here:
the trade/spread channel is AT MOST 1.25 percent of the per-trade variance on BTC and ETH.

A constant term cannot produce growth.  So if D(l) grows anyway, the growth cannot be
attributed to D_Psi without checking, and the alternative is that Sigma^2 is NOT constant --
that the non-trade price variance is itself autocorrelated.  H-U4 already measured that
67-77 percent of the post-trade price move arrives in intervals containing NO TRADE, and
C-T8 that book repricings carry 63 percent of variance.

SO THE RUN DOES BOTH, and the second decides whether the first is readable:

  (1) theta from D(l) in TRADE time, de-fragmented at 200 ms, with a null (i.i.d. returns,
      theta = 0 by construction) and a recovery (impose a known theta and read it back) --
      the discipline C-T27/C-T28 established.
  (2) the same D(l) split by CARRIER using H-U4's accounting: every mid move is assigned to
      the interval it happened in, and intervals are trade-carrying or not.  Variance does not
      decompose as cleanly as a mean, so all three terms are reported --
      D_trade(l), D_book(l), and 2*Cov(l)/l -- and each gets its own exponent.

PREREGISTERED READING:
  if the growth sits in D_book  -> Sigma^2 is autocorrelated, the xi_t assumption is
                                   MISSPECIFIED here, and theta cannot be read as 1-2beta-gamma
  if the growth sits in D_trade -> theta is readable and beta - beta_c = -theta/2 stands

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct30_which_leg_failed_diffusivity_route --i-have-approval
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
LAGS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
FIT_LO, FIT_HI = 10, 1000
MERGE_MS = 200
N_SIM = 60
RECOVERY_THETAS = (0.0, 0.10, 0.20)
RNG_SEED = 20260827


def fit_theta(dd):
    ls = [L for L in sorted(dd) if FIT_LO <= L <= FIT_HI and dd[L] > 0]
    if len(ls) < 4:
        return None
    A = np.column_stack([np.ones(len(ls)), np.log(ls)])
    c = np.linalg.pinv(A.T @ A) @ (A.T @ np.log([dd[L] for L in ls]))
    return float(c[1])


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "lags": list(LAGS), "merge_ms": MERGE_MS,
           "identity": "D_Psi(l) ~ l^(1-2beta-gamma)  =>  beta - beta_c = -theta/2",
           "second_term": "D(l) = D_Psi(l) + Sigma^2, Sigma^2 CONSTANT in l (i.i.d. news)",
           "per_symbol": {}, "ceiling": "MECHANISM_CHARACTERISATION"}

    for sym in H2.SYMBOLS:
        TOT, TR, BK = [], [], []
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo - 2000, hi + 2000)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
            del rows
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.concatenate([[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])])
            idx = np.flatnonzero(new)
            ots0, oeps0 = ts[idx], eps[idx]
            del ts, px, eps, qty
            keep = np.concatenate([[True], (np.diff(ots0) >= MERGE_MS)
                                   | (oeps0[1:] != oeps0[:-1])])
            ots = ots0[np.flatnonzero(keep)]

            # per book update: the mid move, and whether a trade fell in its interval
            dm = np.zeros(len(bts))
            dm[1:] = (mid[1:] / mid[:-1] - 1.0) * 1e4
            ntr = np.zeros(len(bts))
            ntr[1:] = (np.searchsorted(ots, bts[1:], side="right")
                       - np.searchsorted(ots, bts[:-1], side="right"))
            tr = ntr > 0
            c_tot = np.concatenate([[0.0], np.cumsum(dm)])
            c_tr = np.concatenate([[0.0], np.cumsum(np.where(tr, dm, 0.0))])
            c_bk = np.concatenate([[0.0], np.cumsum(np.where(~tr, dm, 0.0))])

            # trade-time sampling points: the book index just before each merged order
            ib = np.searchsorted(bts, ots, side="left") - 1
            ib = ib[ib >= 0]
            TOT.append(c_tot[ib])
            TR.append(c_tr[ib])
            BK.append(c_bk[ib])
            del bts, mid

        if not TOT:
            continue

        def dcoef(paths):
            acc = {}
            for p in paths:
                for L in LAGS:
                    if len(p) <= L + 10:
                        continue
                    r = p[L:] - p[:-L]
                    a0, n0 = acc.get(L, (0.0, 0.0))
                    acc[L] = (a0 + float(np.sum(r * r)), n0 + len(r))
            return {L: (a / n) / L for L, (a, n) in acc.items() if n > 0}

        d_tot, d_tr, d_bk = dcoef(TOT), dcoef(TR), dcoef(BK)
        d_cov = {L: d_tot[L] - d_tr[L] - d_bk[L] for L in d_tot if L in d_tr and L in d_bk}
        th_tot, th_tr, th_bk = fit_theta(d_tot), fit_theta(d_tr), fit_theta(d_bk)

        # null + recovery on the total-path estimator
        n_pts = int(np.mean([len(p) for p in TOT]))
        nulls, recov = [], {str(t): [] for t in RECOVERY_THETAS}
        for _ in range(N_SIM):
            w = np.cumsum(rng.standard_normal(n_pts))
            v = fit_theta(dcoef([w]))
            if v is not None:
                nulls.append(v)
        for t in RECOVERY_THETAS:
            for _ in range(N_SIM // 2):
                # fGn-like: filter white noise so that Var of l-step grows as l^(1+t)
                d_h = t / 2.0
                k = np.arange(1, 1 << 14)
                psi = np.concatenate([[1.0], np.exp((d_h - 1.0) * np.log(k))])
                m = 1 << int(np.ceil(np.log2(n_pts + len(psi))))
                e = rng.standard_normal(m)
                x = np.fft.irfft(np.fft.rfft(e) * np.fft.rfft(psi, m), m)
                w = np.cumsum(x[len(psi):len(psi) + n_pts])
                v = fit_theta(dcoef([w]))
                if v is not None:
                    recov[str(t)].append(v)

        nl = {"mean": float(np.mean(nulls)), "sd": float(np.std(nulls)),
              "p2.5": float(np.percentile(nulls, 2.5)),
              "p97.5": float(np.percentile(nulls, 97.5))}
        rc = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                  "bias": float(np.mean(v) - float(k))} for k, v in recov.items() if v}
        share_bk = (d_bk.get(1000, 0.0) / d_tot.get(1000, 1.0)) if d_tot else None
        out = {"theta_total": th_tot, "theta_trade_carried": th_tr,
               "theta_book_carried": th_bk,
               "D_total": {str(L): d_tot[L] for L in sorted(d_tot)},
               "D_trade": {str(L): d_tr[L] for L in sorted(d_tr)},
               "D_book": {str(L): d_bk[L] for L in sorted(d_bk)},
               "D_cov": {str(L): d_cov[L] for L in sorted(d_cov)},
               "book_share_of_D_at_1000": share_bk,
               "null_theta_iid": nl, "recovery": rc,
               "theta_outside_null": bool(th_tot is not None
                                          and (th_tot < nl["p2.5"] or th_tot > nl["p97.5"])),
               "beta_minus_beta_c": (-th_tot / 2.0) if th_tot is not None else None}
        res["per_symbol"][sym] = out
        print("=== %s   n_pts/day %d" % (sym, n_pts), flush=True)
        print("    D(l) total : " + "  ".join("%d %.4f" % (L, d_tot[L])
                                              for L in sorted(d_tot)), flush=True)
        print("    theta  total %+.4f | trade-carried %+.4f | book-carried %+.4f"
              % (th_tot, th_tr, th_bk), flush=True)
        print("    book share of D at l=1000: %.3f" % share_bk, flush=True)
        print("    NULL (i.i.d. walk, true theta 0): mean %+.4f sd %.4f 95%% [%+.4f,%+.4f]"
              % (nl["mean"], nl["sd"], nl["p2.5"], nl["p97.5"]), flush=True)
        for k in sorted(rc):
            print("    RECOVERY true theta %s -> %+.4f +- %.4f  bias %+.4f"
                  % (k, rc[k]["mean"], rc[k]["sd"], rc[k]["bias"]), flush=True)
        print("    theta outside null? %s   =>  beta - beta_c = %+.4f"
              % (out["theta_outside_null"], out["beta_minus_beta_c"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT30_DIFFUSIVITY_ROUTE_V1.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
