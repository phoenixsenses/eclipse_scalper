r"""LANE C, round 38 -- do any of this estate's exponents have a scaling regime at all?

Four rounds have now ended with the same shape of finding:

    C-T33  chi depends on which dispersion measure   0.57 - 1.30
    C-T35  beta depends on which lag range           0.004 - 0.151
    C-T36  delta depends on the direction of the cut 0.04 - 1.41
    C-T31  a permutation null is not a standard error

I recorded each as a caution. Taken together they suggest something stronger and testable: that
this data may have NO CLEAN SCALING REGIME, in which case "the exponent" was never a well-defined
quantity for any of them and the four instances have one cause rather than four.

That matters beyond bookkeeping. Bouchaud's framework -- the collapse in 11.4, the propagator
power law in 13, the tail exponents in 2 -- is built on power laws calibrated largely on equities
and futures. If crypto perpetuals lack the scaling regimes those forms presume, then fitting them
returns a number whatever the data does, and the number is an artefact of the fit range.

THE TEST. A power law has a FLAT local slope. So instead of fitting one exponent over a chosen
range, compute the local log-log slope as a function of scale and look for a plateau:

    S(T) = sd( sum of x over windows of T )        on a dense log grid
    m(T) = local slope of log S against log T      over a sliding half-decade

A genuine scaling regime shows m(T) flat over at least a decade. Continuous drift means there is
no regime and any single fitted value belongs to its window.

CALIBRATION FIRST, as this lane now does by default. The tolerance is not chosen -- it is measured
on synthetic series of the SAME LENGTH with a KNOWN power law (fractional Gaussian noise at
several Hurst exponents, and an iid series whose true slope is exactly 1/2). Whatever drift those
show is the floor; only drift above it counts.

Applied to the three series this lane has built results on: order signs (chi), returns (H), and
signed notional (the C-T23 chi).
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
T_MIN, T_MAX, NT = 4, 20000, 34
HALF_DECADE = 10 ** 0.5
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sd_curve(x):
    """sd of the sum of x over non-overlapping windows, on a log grid of window sizes"""
    n = len(x)
    Ts = np.unique(np.round(np.geomspace(T_MIN, min(T_MAX, n // 200), NT)).astype(int))
    T, S = [], []
    for t in Ts:
        m = n // t
        if m < 200:
            continue
        T.append(float(t))
        S.append(float(np.std(x[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    return np.array(T), np.array(S)


def local_slopes(T, S):
    """slope of log S on log T inside a sliding half-decade window"""
    lt, ls = np.log(T), np.log(S)
    out = []
    for i in range(len(T)):
        m = (T >= T[i]) & (T <= T[i] * HALF_DECADE)
        if m.sum() < 4:
            continue
        A = np.column_stack([np.ones(m.sum()), lt[m]])
        b, *_ = np.linalg.lstsq(A, ls[m], rcond=None)
        out.append((float(T[i]), float(b[1])))
    return out


def plateau_width(slopes, tol):
    """widest span of T, in decades, over which the local slope stays inside +-tol of its median"""
    if len(slopes) < 3:
        return 0.0, None
    best, bounds = 0.0, None
    for i in range(len(slopes)):
        ref = slopes[i][1]
        j = i
        while j + 1 < len(slopes) and abs(slopes[j + 1][1] - ref) <= tol:
            j += 1
        if j > i:
            width = np.log10(slopes[j][0] / slopes[i][0])
            if width > best:
                best, bounds = width, (slopes[i][0], slopes[j][0], ref)
    return float(best), bounds


def fgn(n, H, rng):
    """fractional Gaussian noise by spectral synthesis -- a series with an EXACT power law"""
    m = 1 << int(np.ceil(np.log2(2 * n)))
    k = np.arange(m // 2 + 1)
    f = np.maximum(k, 1) / m
    psd = f ** (1.0 - 2.0 * H)
    psd[0] = 0.0
    ph = rng.uniform(0, 2 * np.pi, len(k))
    spec = np.sqrt(psd) * np.exp(1j * ph)
    spec[0] = 0.0
    spec[-1] = np.abs(spec[-1])
    x = np.fft.irfft(spec, m)[:n]
    return x / np.std(x)


def main() -> int:
    rng = np.random.default_rng(SEED)

    # ---- calibration: what drift does a TRUE power law show at this length?
    cal = {}
    n = NROWS
    for name, x in (("iid_true_slope_0.5", rng.standard_normal(n)),
                    ("fgn_H_0.60", fgn(n, 0.60, rng)),
                    ("fgn_H_0.75", fgn(n, 0.75, rng)),
                    ("fgn_H_0.90", fgn(n, 0.90, rng))):
        T, S = sd_curve(x)
        sl = local_slopes(T, S)
        v = [s for _, s in sl]
        cal[name] = {"slope_min": round(min(v), 4), "slope_max": round(max(v), 4),
                     "drift": round(max(v) - min(v), 4),
                     "slopes": [(int(t), round(s, 4)) for t, s in sl]}
    tol = round(float(max(c["drift"] for c in cal.values())), 4)

    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            eps = np.where(a[:, 2] > 0.5, -1.0, 1.0)
            ret = np.empty_like(lp)
            ret[0] = 0.0
            ret[1:] = np.diff(lp)
            series = {"order_signs_chi": eps,
                      "returns_H": ret,
                      "signed_notional_chi_of_C_T23": eps * a[:, 1]}
            d = {}
            for nm, x in series.items():
                T, S = sd_curve(x)
                sl = local_slopes(T, S)
                v = [s for _, s in sl]
                width, bounds = plateau_width(sl, tol)
                d[nm] = {"slope_at_smallest_T": round(v[0], 4),
                         "slope_at_largest_T": round(v[-1], 4),
                         "slope_min": round(min(v), 4), "slope_max": round(max(v), 4),
                         "drift": round(max(v) - min(v), 4),
                         "drift_over_calibrated_tolerance": round((max(v) - min(v)) / tol, 2),
                         "widest_plateau_decades": round(width, 3),
                         "plateau_bounds": ([int(bounds[0]), int(bounds[1]),
                                             round(bounds[2], 4)] if bounds else None),
                         "has_a_scaling_regime": bool(width >= 1.0),
                         "slopes": [(int(t), round(s, 4)) for t, s in sl]}
            per[sym] = d
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T38", "lane": "C", "utc": _utc(),
           "test": "a power law has a FLAT local slope; a scaling regime = a plateau >= 1 decade",
           "tolerance_source": ("measured, not chosen: the largest local-slope drift shown by "
                                "synthetic series of the same length with EXACT power laws"),
           "calibrated_tolerance": tol, "calibration": cal, "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C38_SCALING_REGIME_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("CALIBRATION -- drift of the local slope for series with EXACT power laws, n = {0}".format(n))
    for k, c in cal.items():
        w("   %-22s slope %.4f .. %.4f   drift %.4f" % (k, c["slope_min"], c["slope_max"],
                                                        c["drift"]))
    w("   => calibrated tolerance {0}".format(tol))
    w("")
    w("%-9s %-30s %8s %8s %8s %8s %10s %8s" % ("sym", "series", "small T", "large T", "min",
                                               "max", "drift/tol", "plateau"))
    for s in SYMS:
        for nm, d in per[s].items():
            w("%-9s %-30s %8.4f %8.4f %8.4f %8.4f %10.2f %8.2f" % (
                s, nm, d["slope_at_smallest_T"], d["slope_at_largest_T"], d["slope_min"],
                d["slope_max"], d["drift_over_calibrated_tolerance"],
                d["widest_plateau_decades"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
