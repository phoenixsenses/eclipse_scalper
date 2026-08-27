# -*- coding: utf-8 -*-
"""C-T31 -- IS THIS ESTATE IN PART VI (PROPAGATOR) OR PART VIII (LATENT LIQUIDITY)?

C-T30 found that the trade channel carries only 17-24 percent of the lag-dependent diffusion
coefficient here, against the 65-80 percent the book reports for equities (Figure 13.2), and
that the remaining 76-83 percent -- the model's constant "news" term Sigma^2 -- is itself
AUTOCORRELATED on all three symbols.  Sec 18.1 and Sec 18.2 turn out to name that situation
exactly:

    "an LOB only reflects a tiny fraction of the true total supply and demand - which, for the
     most part, actually remain latent"
    "The dynamics of the liquidity results from two distinct mechanisms: (i) order matching
     and market clearing when transactions take place, and (ii) evolution of the intentions of
     traders BETWEEN transactions."

So the propagator identities may not have failed because the corpus is wrong.  They may have
failed because Part VI's machinery was applied where the book itself says Part VIII applies.
That reframing is testable, because the two parts predict DIFFERENT aggregate impact
exponents:

    PART VI   Eq (16.16): delta = gamma.  De-biased gamma here (C-T28) is 0.381 / 0.347 /
              0.374, so the propagator route predicts an exponent near 0.37.
    PART VIII Sec 18.5 "From Linear to Square-Root Impact": latent liquidity gives the
              SQUARE-ROOT law, exponent 0.5.

A-S30 measured the OUTER-region exponent at 0.416 / 0.439 / 0.495 and reported it flat over a
50x range in T -- but with no standard error and no null.  ERR-HU-020 explicitly declined to
extend C-T26's N6 verdict to A's estimator, because N6 tested a WHOLE-RANGE fit and A's is an
outer-region fit.  This run gives A's estimator the treatment N6 gave mine, and then uses it
to discriminate.

THREE LEGS, in the order C-T27 and C-T28 established:
  NULL      exactly linear impact, true exponent 1.0.  Does the outer fit return 1.0, or does
            it attenuate the way the whole-range fit did (0.256)?
  RECOVERY  impose dP = k |dV|^psi sign(dV) + noise for psi in {0.3, 0.5, 0.7} at the real
            counts and the real residual scale.  Bias and sd.
  REAL      zeta_outer per symbol, de-biased by inverting the recovery curve, then tested
            against 0.5 and against C-T28's de-biased gamma.

"OUTER" is declared here and not tuned afterwards: bins whose |dV| is above the 75th
percentile within that T.  Clock: market-order time, non-overlapping, T in {20, 50, 100, 200}
-- A-S30's own range.

ESTIMATION.  Ceiling: MECHANISM_CHARACTERISATION.

  python -m tools.ct31_part_six_or_part_eight --i-have-approval
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
T_LIST = (20, 50, 100, 200)
OUTER_PCT = 75.0
N_BINS = 14
N_SIM = 60
RECOVERY_PSI = (0.3, 0.5, 0.7)
RNG_SEED = 20260827
GAMMA_DEBIASED = {"BTCUSDT": 0.3810, "ETHUSDT": 0.3466, "SOLUSDT": 0.3736}
ZETA_A_S30 = {"BTCUSDT": 0.416, "ETHUSDT": 0.439, "SOLUSDT": 0.495}
SQRT_LAW = 0.5


def outer_slope(dv, dp, pct=OUTER_PCT, nb=N_BINS, min_n=200):
    a = np.abs(dv)
    y = np.sign(dv) * dp
    ok = np.isfinite(a) & np.isfinite(y) & (a > 0)
    a, y = a[ok], y[ok]
    if len(a) < 5 * min_n:
        return None
    cut = np.percentile(a, pct)
    m = a >= cut
    if m.sum() < 5 * min_n:
        return None
    a, y = a[m], y[m]
    ed = np.unique(np.percentile(a, np.linspace(0, 100, nb + 1)))
    if len(ed) < 6:
        return None
    b = np.clip(np.searchsorted(ed, a, side="right") - 1, 0, len(ed) - 2)
    cnt = np.bincount(b, minlength=len(ed) - 1).astype(float)
    sa = np.bincount(b, weights=a, minlength=len(ed) - 1)
    sy = np.bincount(b, weights=y, minlength=len(ed) - 1)
    keep = (cnt >= min_n) & (sy > 0)
    if keep.sum() < 5:
        return None
    mx = np.log(sa[keep] / cnt[keep])
    my = np.log(sy[keep] / cnt[keep])
    A = np.column_stack([np.ones(len(mx)), mx])
    return float((np.linalg.pinv(A.T @ A) @ (A.T @ my))[1])


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "T_list": list(T_LIST), "outer_pct": OUTER_PCT,
           "part_vi_prediction": GAMMA_DEBIASED, "part_viii_prediction": SQRT_LAW,
           "zeta_A_S30": ZETA_A_S30, "per_symbol": {},
           "ceiling": "MECHANISM_CHARACTERISATION"}

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
            oeps, ovol, ots = eps[idx], np.add.reduceat(qty, idx), ts[idx]
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

        real = {T: outer_slope(dv[T], dp[T]) for T in dv}
        zr = [v for v in real.values() if v is not None]
        zeta_real = float(np.median(zr)) if zr else None

        # NULL: exactly linear, on the real |dV| with real residual scale
        Tref = T_LIST[0]
        base_scale = float(np.std(dp[Tref]))
        nulls, recov = [], {str(p): [] for p in RECOVERY_PSI}
        for _ in range(N_SIM):
            vals = []
            for T in dv:
                x = dv[T]
                k = base_scale / max(1e-12, np.std(x))
                y = k * x + rng.standard_normal(len(x)) * np.std(dp[T])
                v = outer_slope(x, y)
                if v is not None:
                    vals.append(v)
            if vals:
                nulls.append(float(np.median(vals)))
        for psi in RECOVERY_PSI:
            for _ in range(N_SIM // 2):
                vals = []
                for T in dv:
                    x = dv[T]
                    sgn = np.sign(x)
                    amp = np.abs(x) ** psi
                    k = base_scale / max(1e-12, np.std(amp))
                    y = k * sgn * amp + rng.standard_normal(len(x)) * np.std(dp[T])
                    v = outer_slope(x, y)
                    if v is not None:
                        vals.append(v)
                if vals:
                    recov[str(psi)].append(float(np.median(vals)))

        nl = {"mean": float(np.mean(nulls)), "sd": float(np.std(nulls)),
              "should_be": 1.0, "bias": float(np.mean(nulls) - 1.0)}
        rc = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                  "bias": float(np.mean(v) - float(k))} for k, v in recov.items() if v}

        # de-bias the real value by inverting the recovery curve
        ps = sorted(float(k) for k in rc)
        ms = [rc[str(p)]["mean"] for p in ps]
        zeta_db = float(np.interp(zeta_real, ms, ps)) if len(ps) >= 2 else None
        sd_typ = float(np.mean([rc[str(p)]["sd"] for p in ps])) if ps else None
        g = GAMMA_DEBIASED[sym]
        z_sqrt = abs(zeta_db - SQRT_LAW) / sd_typ if (zeta_db and sd_typ) else None
        z_gam = abs(zeta_db - g) / sd_typ if (zeta_db and sd_typ) else None
        out = {"zeta_outer_by_T": {str(T): real[T] for T in sorted(real)},
               "zeta_outer_median": zeta_real, "zeta_debiased": zeta_db,
               "typical_sd": sd_typ, "null_linear": nl, "recovery": rc,
               "z_vs_sqrt_law_0.5": z_sqrt, "z_vs_gamma_part_vi": z_gam,
               "closer_to": ("PART_VIII_sqrt" if (z_sqrt is not None and z_gam is not None
                                                 and z_sqrt < z_gam) else "PART_VI_gamma"),
               "zeta_A_S30": ZETA_A_S30[sym], "gamma_debiased": g}
        res["per_symbol"][sym] = out
        print("=== %s" % sym, flush=True)
        print("    zeta_outer by T: " + "  ".join("T%d %s" % (T, "%.4f" % real[T]
                                                              if real[T] else "n/a")
                                                  for T in sorted(real))
              + "   median %.4f   (A-S30 %.3f)" % (zeta_real, ZETA_A_S30[sym]), flush=True)
        print("    NULL exactly-linear (should be 1.0): %.4f +- %.4f   bias %+.4f"
              % (nl["mean"], nl["sd"], nl["bias"]), flush=True)
        for p in ps:
            print("    RECOVERY true %.1f -> %.4f +- %.4f  bias %+.4f"
                  % (p, rc[str(p)]["mean"], rc[str(p)]["sd"], rc[str(p)]["bias"]), flush=True)
        print("    de-biased zeta %.4f   vs sqrt-law 0.5 z=%.2f   vs gamma %.3f z=%.2f  => %s"
              % (zeta_db, z_sqrt, g, z_gam, out["closer_to"]), flush=True)

    votes = [v["closer_to"] for v in res["per_symbol"].values()]
    res["verdict"] = {"votes": votes,
                      "part_viii": votes.count("PART_VIII_sqrt"),
                      "part_vi": votes.count("PART_VI_gamma")}
    print("VERDICT  Part VIII %d / Part VI %d   %s"
          % (res["verdict"]["part_viii"], res["verdict"]["part_vi"], votes), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT31_PART_VI_OR_VIII_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
