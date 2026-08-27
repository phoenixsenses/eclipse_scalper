r"""LANE C, round 41 -- every clock, both metrics, and then the question no clock can answer.

C-T40 tried ONE alternative clock and reported a split verdict because I had pre-registered one
of two natural metrics. Two families of study were left on the table; both are run here.

FAMILY A -- ALL THE CLOCKS, BOTH METRICS.
The corpus names the subordinator as "the number of trades (or the cumulated volume)", and the
literature it cites (Mandelbrot & Taylor, Clark 1973, Ane & Geman) uses several. There is no
reason to test one. Five are constructible from aggTrades:

    calendar      equal intervals of physical time
    trade         equal counts of trades                        (C-T38's clock)
    volume        equal increments of cumulated notional        (C-T40's clock, Clark's)
    sqrt_volume   equal increments of cumulated sqrt(notional)  (the impact-motivated one)
    tick_event    equal counts of NON-ZERO price changes

and both metrics from C-T40 are reported for every one of them, rather than one being chosen:

    (i)  drift of the local slope against C-T38's measured floor 0.0686
    (ii) the hypothesis's own premise -- is the variance per unit of THAT clock constant?

FAMILY B -- WHAT NO CLOCK CAN FIX.
Even in volume time C-T40 left 1.5-1.9x the power-law floor. A time change cannot remove
MULTIFRACTALITY, because that is a statement about the whole family of moments rather than about
one exponent:

    E|sum_T x|^q  ~  T^zeta(q)

A monofractal has zeta(q) = q H exactly, so zeta(q)/q is CONSTANT in q. A multifractal has
zeta(q) concave, so zeta(q)/q FALLS with q. If this estate's prices are multifractal, then "the
exponent" is not a single number for a reason no clock choice can repair -- which would be the
structural end of the four window findings rather than another instance of them.

Calibrated, as always, before it is read: fractional Gaussian noise is monofractal by
construction, so whatever curvature it shows at this length is the floor.
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
TOL = 0.0686
HALF_DECADE = 10 ** 0.5
NT = 30
QS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sd_curve(x, tmin=4, tmax=20000):
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(tmin, min(tmax, n // 200), NT)).astype(int))
    T, S = [], []
    for t in Ts:
        m = n // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    return np.array(T), np.array(S)


def local_slope_drift(x):
    T, S = sd_curve(x)
    lt, ls = np.log(T), np.log(S)
    v = []
    for i in range(len(T)):
        m = (T >= T[i]) & (T <= T[i] * HALF_DECADE)
        if m.sum() < 4:
            continue
        A = np.column_stack([np.ones(m.sum()), lt[m]])
        b, *_ = np.linalg.lstsq(A, ls[m], rcond=None)
        v.append(float(b[1]))
    return (max(v) - min(v)) if v else float("nan"), v


def zeta_of_q(x, qs=QS):
    """scaling exponent of the q-th absolute moment of partial sums"""
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(8, min(4000, n // 300), 14)).astype(int))
    out = {}
    for q in qs:
        T, M = [], []
        for t in Ts:
            m = n // t
            if m < 300:
                continue
            s = np.abs(x[:m * t].reshape(m, t).sum(axis=1))
            mom = float(np.mean(s ** q))
            if mom <= 0:
                continue
            T.append(float(t))
            M.append(mom)
        if len(T) < 6:
            continue
        A = np.column_stack([np.ones(len(T)), np.log(T)])
        b, *_ = np.linalg.lstsq(A, np.log(M), rcond=None)
        out[q] = float(b[1])
    return out


def curvature(zq):
    """zeta(q)/q falls with q for a multifractal; return the total fall and a linear-fit slope"""
    qs = sorted(zq)
    h = [zq[q] / q for q in qs]
    A = np.column_stack([np.ones(len(qs)), np.asarray(qs, float)])
    b, *_ = np.linalg.lstsq(A, np.asarray(h), rcond=None)
    return {"h_of_q": {str(q): round(zq[q] / q, 4) for q in qs},
            "total_fall": round(h[0] - h[-1], 4),
            "slope_of_h_on_q": round(float(b[1]), 4)}


def fgn(n, H, rng):
    m = 1 << int(np.ceil(np.log2(2 * n)))
    k = np.arange(m // 2 + 1)
    f = np.maximum(k, 1) / m
    psd = f ** (1.0 - 2.0 * H)
    psd[0] = 0.0
    spec = np.sqrt(psd) * np.exp(1j * rng.uniform(0, 2 * np.pi, len(k)))
    spec[0] = 0.0
    spec[-1] = np.abs(spec[-1])
    x = np.fft.irfft(spec, m)[:n]
    return x / np.std(x)


def resample(lp, key, n_target):
    """equal increments of the cumulative key; returns the return series in that clock"""
    cv = np.cumsum(key)
    marks = np.arange(1, n_target + 1) * (cv[-1] / n_target)
    idx = np.clip(np.searchsorted(cv, marks), 0, len(lp) - 1)
    p = lp[idx]
    d = np.empty_like(p)
    d[0] = 0.0
    d[1:] = np.diff(p)
    return d


def main() -> int:
    rng = np.random.default_rng(SEED)

    # ---- calibration for family B: a monofractal at this length
    cal = {}
    for H in (0.5, 0.6, 0.75):
        z = zeta_of_q(fgn(NROWS // 4, H, rng))
        cal["fgn_H_{0}".format(H)] = curvature(z)
    fall_floor = round(float(max(abs(c["total_fall"]) for c in cal.values())), 4)

    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,notional from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, vol = a[:, 0], a[:, 1], a[:, 2]
            lp = np.log(px)
            n = len(lp)
            r_trade = np.empty_like(lp)
            r_trade[0] = 0.0
            r_trade[1:] = np.diff(lp)

            dpx = np.abs(np.diff(lp))
            moved = np.concatenate([[0.0], (dpx > 0).astype(float)])

            clocks = {
                "calendar": resample(lp, np.concatenate([[0.0], np.diff(ts)]), n),
                "trade": r_trade,
                "volume": resample(lp, vol, n),
                "sqrt_volume": resample(lp, np.sqrt(vol), n),
                "tick_event": resample(lp, moved, int(moved.sum())),
            }

            d = {}
            B = 20
            for nm, x in clocks.items():
                drift, _ = local_slope_drift(x)
                m = len(x) // B
                vpu = [float(np.var(x[b * m:(b + 1) * m], ddof=1)) for b in range(B)]
                cv = float(np.std(vpu, ddof=1) / np.mean(vpu)) if np.mean(vpu) > 0 else np.nan
                d[nm] = {"n_steps": int(len(x)),
                         "drift": round(drift, 4),
                         "drift_over_floor": round(drift / TOL, 2),
                         "cv_variance_per_unit_of_this_clock": round(cv, 4)}
            best_drift = min(d, key=lambda k: d[k]["drift"])
            best_cv = min(d, key=lambda k: d[k]["cv_variance_per_unit_of_this_clock"])

            # family B in the two clocks that matter
            mf = {}
            for nm in ("trade", "volume"):
                mf[nm] = curvature(zeta_of_q(clocks[nm]))
                mf[nm]["fall_over_monofractal_floor"] = round(
                    abs(mf[nm]["total_fall"]) / fall_floor, 2)
            per[sym] = {"clocks": d, "best_by_drift": best_drift, "best_by_premise_cv": best_cv,
                        "two_metrics_agree": bool(best_drift == best_cv),
                        "multifractal": mf}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T41", "lane": "C", "utc": _utc(),
           "family_A": "all five clocks, BOTH metrics, no metric chosen over the other",
           "family_B": "zeta(q)/q constant means monofractal; falling means multifractal",
           "drift_floor_from_C_T38": TOL, "monofractal_fall_floor": fall_floor,
           "calibration": cal, "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C41_ALL_CLOCKS_AND_MULTIFRACTAL_V1.json").write_text(
        json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("FAMILY A -- every clock, both metrics (drift floor 0.0686)")
    w("%-9s %-13s %9s %12s %14s" % ("sym", "clock", "drift", "x floor", "CV var/unit"))
    for s in SYMS:
        for nm, v in per[s]["clocks"].items():
            w("%-9s %-13s %9.4f %12.2f %14.4f" % (s, nm, v["drift"], v["drift_over_floor"],
                                                  v["cv_variance_per_unit_of_this_clock"]))
        w("%-9s  best by drift: %-12s best by premise: %-12s agree: %s" % (
            s, per[s]["best_by_drift"], per[s]["best_by_premise_cv"],
            per[s]["two_metrics_agree"]))
        w("")
    w("FAMILY B -- is it multifractal? (monofractal floor for the fall: {0})".format(fall_floor))
    w("%-9s %-9s %s %12s %10s" % ("sym", "clock", "h(q) = zeta(q)/q", "total fall", "x floor"))
    for s in SYMS:
        for nm, v in per[s]["multifractal"].items():
            hq = " ".join("%s:%.3f" % (q, h) for q, h in v["h_of_q"].items())
            w("%-9s %-9s %s  %8.4f %10.2f" % (s, nm, hq, v["total_fall"],
                                              v["fall_over_monofractal_floor"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
