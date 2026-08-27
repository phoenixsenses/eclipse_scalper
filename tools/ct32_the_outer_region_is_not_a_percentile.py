# -*- coding: utf-8 -*-
"""C-T32 -- THE "OUTER REGION" IS NOT A PERCENTILE, AND THE CUT IS A FREE PARAMETER.

C-T31 measured zeta_outer = 0.369 / 0.427 / 0.394 with a null of 1.004 and a recovery sd of
0.004-0.010, and A-S30 published 0.416 / 0.439 / 0.495 for the same object.  Gaps of 0.047,
0.012 and 0.102 against a standard error of 0.007 are enormous, and the only structural
difference between the two runs is where "outer" was drawn: C declared a rule (|dV| above its
75th percentile), A did not publish one.

Sec 11.4 says the boundary is not a percentile at all.  F(u) is "linear for small arguments
and concave for large arguments" with u = dV / (V_D T^kappa), so the outer region is where
u >> 1 -- a property of the data, not a quantile of it.  A percentile cut can land anywhere
relative to that crossover, and if it lands BELOW it the fit is measuring the LINEAR region
while being called outer.

TWO THINGS, and the second removes the free parameter entirely.

  (1) SWEEP THE CUT.  zeta as a function of the outer percentile, 50 to 99.  If it swings
      across the gap between the lanes, then neither published value is a measurement -- it
      is a choice, and the lane disagreement is fully explained.

  (2) DROP THE CUT.  Report the LOCAL log-log slope of R against |dV| across the whole range,
      bin by bin.  A curve that is linear near the origin and concave far out shows a local
      slope that starts near 1 and decays toward an asymptote; that asymptote IS the exponent
      and needs no cut.  The crossover can then be read off empirically, with no kappa and no
      quantile.  The asymptotic slope gets the same null-and-recovery treatment C-T31 used,
      since an estimator with no free parameter is still an estimator.

PREREGISTERED READING:
  if zeta(cut) is flat over 50-99      -> the cut is not the explanation, and the lane gap is
                                          data or pipeline, not definition
  if zeta(cut) spans the lane gap      -> the cut IS the explanation; both published values
                                          are cut-dependent and must be reported as such
  the local-slope profile is reported either way, as the parameter-free alternative

ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct32_the_outer_region_is_not_a_percentile --i-have-approval
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
CUTS = (50.0, 60.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 99.0)
N_BINS_FULL = 26
N_BINS_OUTER = 14
N_SIM = 40
RECOVERY_PSI = (0.3, 0.5, 0.7)
RNG_SEED = 20260827
C_T31 = {"BTCUSDT": 0.3691, "ETHUSDT": 0.4270, "SOLUSDT": 0.3935}
A_S30 = {"BTCUSDT": 0.416, "ETHUSDT": 0.439, "SOLUSDT": 0.495}


def binned(dv, dp, nb, lo_pct=0.0, min_n=200):
    a, y = np.abs(dv), np.sign(dv) * dp
    ok = np.isfinite(a) & np.isfinite(y) & (a > 0)
    a, y = a[ok], y[ok]
    if lo_pct > 0:
        m = a >= np.percentile(a, lo_pct)
        a, y = a[m], y[m]
    if len(a) < 5 * min_n:
        return None
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
    return np.log(sa[keep] / cnt[keep]), np.log(sy[keep] / cnt[keep])


def slope(dv, dp, nb, lo_pct=0.0):
    r = binned(dv, dp, nb, lo_pct)
    if r is None:
        return None
    mx, my = r
    A = np.column_stack([np.ones(len(mx)), mx])
    return float((np.linalg.pinv(A.T @ A) @ (A.T @ my))[1])


def asymptotic_slope(dv, dp, nb=N_BINS_FULL, tail_bins=6):
    """no cut: local slope over the LAST `tail_bins` bins of the full curve"""
    r = binned(dv, dp, nb)
    if r is None:
        return None, None
    mx, my = r
    if len(mx) < tail_bins + 3:
        return None, None
    loc = np.gradient(my, mx)
    A = np.column_stack([np.ones(tail_bins), mx[-tail_bins:]])
    asym = float((np.linalg.pinv(A.T @ A) @ (A.T @ my[-tail_bins:]))[1])
    return asym, {"log_dV": [float(v) for v in mx], "local_slope": [float(v) for v in loc]}


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    rng = np.random.default_rng(RNG_SEED)
    res = {"days": list(DAYS), "T_list": list(T_LIST), "cuts": list(CUTS),
           "C_T31_at_75pct": C_T31, "A_S30": A_S30,
           "book": "Sec 11.4: F linear for small u, concave for large u; u = dV/(V_D T^kappa)",
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

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

        # (1) the sweep
        sweep = {}
        for c in CUTS:
            vals = [slope(dv[T], dp[T], N_BINS_OUTER, c) for T in dv]
            vals = [v for v in vals if v is not None]
            sweep[str(c)] = float(np.median(vals)) if vals else None
        sv_ = [v for v in sweep.values() if v is not None]
        span = (max(sv_) - min(sv_)) if sv_ else None

        # (2) the parameter-free profile
        asy, prof = [], None
        for T in dv:
            a, pr = asymptotic_slope(dv[T], dp[T])
            if a is not None:
                asy.append(a)
            if pr is not None and prof is None:
                prof = pr
        asym = float(np.median(asy)) if asy else None

        # null + recovery for the parameter-free estimator
        base = float(np.std(dp[T_LIST[0]]))
        nulls, recov = [], {str(p): [] for p in RECOVERY_PSI}
        for _ in range(N_SIM):
            vals = []
            for T in dv:
                x = dv[T]
                k = base / max(1e-12, np.std(x))
                a, _ = asymptotic_slope(x, k * x + rng.standard_normal(len(x))
                                        * np.std(dp[T]))
                if a is not None:
                    vals.append(a)
            if vals:
                nulls.append(float(np.median(vals)))
        for psi in RECOVERY_PSI:
            for _ in range(N_SIM // 2):
                vals = []
                for T in dv:
                    x = dv[T]
                    amp = np.abs(x) ** psi
                    k = base / max(1e-12, np.std(amp))
                    a, _ = asymptotic_slope(x, k * np.sign(x) * amp
                                            + rng.standard_normal(len(x)) * np.std(dp[T]))
                    if a is not None:
                        vals.append(a)
                if vals:
                    recov[str(psi)].append(float(np.median(vals)))
        nl = {"mean": float(np.mean(nulls)), "sd": float(np.std(nulls)),
              "bias_vs_1": float(np.mean(nulls) - 1.0)} if nulls else None
        rc = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                  "bias": float(np.mean(v) - float(k))} for k, v in recov.items() if v}
        ps = sorted(float(k) for k in rc)
        asym_db = (float(np.interp(asym, [rc[str(p)]["mean"] for p in ps], ps))
                   if (asym is not None and len(ps) >= 2) else None)

        a_val = A_S30[sym]
        reaches_A = bool(sv_ and min(sv_) - 0.02 <= a_val <= max(sv_) + 0.02)
        out = {"sweep": sweep, "sweep_span": span, "reaches_A_S30": reaches_A,
               "asymptotic_slope_no_cut": asym, "asymptotic_debiased": asym_db,
               "null_linear": nl, "recovery": rc, "local_slope_profile_T20": prof}
        res["per_symbol"][sym] = out
        print("=== %s" % sym, flush=True)
        print("    zeta by outer cut: " + "  ".join(
            "%.0f%% %s" % (c, "%.3f" % sweep[str(c)] if sweep[str(c)] else "n/a")
            for c in CUTS), flush=True)
        print("    span across cuts %.3f   (C-T31@75%% %.3f, A-S30 %.3f)  reaches A? %s"
              % (span, C_T31[sym], a_val, reaches_A), flush=True)
        if prof:
            print("    local slope vs log|dV|: " + "  ".join(
                "%.1f:%.2f" % (x, s) for x, s in zip(prof["log_dV"][::4],
                                                     prof["local_slope"][::4])), flush=True)
        print("    NO-CUT asymptotic slope %.4f  -> de-biased %s   null(linear) %.4f +- %.4f"
              % (asym, "%.4f" % asym_db if asym_db else "n/a",
                 nl["mean"] if nl else float("nan"), nl["sd"] if nl else float("nan")),
              flush=True)
        for p in ps:
            print("        RECOVERY true %.1f -> %.4f +- %.4f  bias %+.4f"
                  % (p, rc[str(p)]["mean"], rc[str(p)]["sd"], rc[str(p)]["bias"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "CT32_OUTER_REGION_V1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print("written", flush=True)


if __name__ == "__main__":
    main()
