r"""LANE C, round 45 -- CT-017: the two books disagree, and the data agrees with neither.

CT-017 (A-S50, 2026-08-27) is open and it is this lane's territory:

    Kissell   MI = b1 . I* . POV^a4 + (1-b1) . I*      -> trading SLOWER REDUCES impact; an
                                                          interior optimum exists.
    Bouchaud  TQP 12.3.2, verbatim: "the time horizon T does not appear explicitly". With
              sigma_T = sigma_d sqrt(T), V_T = ADV.T and delta = 1/2, T CANCELS:
              I = Y sigma_d sqrt(Q/ADV).                -> impact depends on SIZE, not SPEED.

A-S50 named the discriminator as `a4` and recorded that it has never been measured on crypto here,
and that measuring it needs execution data the estate is forbidden to touch.

`--who temporary permanent POV` returns C-T36 and C-T37, this lane's own, and neither was ever
connected to CT-017. Two things follow, one a status change and one a measurement.

STATUS. C-T36 and C-T37 already establish something stronger than "a4 is unmeasured here": the
temporary/permanent split is NOT IDENTIFIABLE from anonymised aggregate data, because it requires
Q and V to be varied independently and the estate's (Q, V) support does not contain that
variation -- a positivity violation in Hernan & Robins' sense, with the available POV lever
measured at about 2x against the ~100x the contrast needs. So CT-017 is not waiting for a
measurement; it is structurally blocked.

MEASUREMENT. But both readings make a claim that IS testable without a4, because they agree on a
weaker statement and disagree with the data:

    Kissell   R(Q, T) at fixed Q is DECREASING in T
    Bouchaud  R(Q, T) at fixed Q is FLAT in T
    both      R(Q, T) at fixed Q is NON-INCREASING in T

That is a single directional prediction and it can be checked. C-T37 already saw the answer inside
its positivity region and did not connect it either.

CALIBRATION BEFORE THE TEST, as the estate now requires. "R rises with T at fixed Q" is only
meaningful against what a purely SIZE-DEPENDENT impact world would show under the identical
estimator and the identical binning. That world is built here from the real trades: each trade
moves the price permanently by a fitted function of its own size, plus iid noise, and nothing
depends on T by construction. If T cancels there and does not cancel in the real series, the
difference is the finding.

THRESHOLD SWEEP, as the estate now requires. C-T37's positivity region used MIN_CELL = 200. That
threshold is swept here at 100 / 200 / 500 and the result is reported at all three.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db, table agg_trades, first 2,000,000 rows by
ts_ms per symbol -- the same population as C-T33/C-T36/C-T37.
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
T_GRID = (5, 10, 20, 50, 100, 200, 500)
NQ = 16
MIN_CELLS = (100, 200, 500)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def windows(lp, flow, T):
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m) * T
    dv = flow[:m * T].reshape(m, T).sum(axis=1)[1:]
    dp = (lp[i0 + T - 1] - lp[i0 - 1]) * 1e4
    return dv, dp


def profile(lp, flow, edges, min_cell):
    """mean R per (Q-bin, T) cell; returns the T-profile averaged over bins where EVERY T is
    populated -- the positivity region"""
    grid, R = {}, {}
    for T in T_GRID:
        dv, dp = windows(lp, flow, T)
        cnt, rr = [], []
        for b in range(len(edges) - 1):
            m = (dv >= edges[b]) & (dv < edges[b + 1])
            cnt.append(int(m.sum()))
            rr.append(float(dp[m].mean()) if m.sum() >= min_cell else None)
        grid[T], R[T] = cnt, rr
    full = [b for b in range(len(edges) - 1)
            if all(grid[T][b] >= min_cell for T in T_GRID)]
    if not full:
        return None, [], grid
    prof = {}
    for T in T_GRID:
        vals = [R[T][b] for b in full if R[T][b] is not None]
        prof[T] = float(np.mean(vals)) if vals else None
    return prof, full, grid


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            vol = a[:, 1]
            eps = np.where(a[:, 2] > 0.5, -1.0, 1.0)
            flow = eps * vol
            n = len(lp)

            allpos = np.concatenate([windows(lp, flow, T)[0][windows(lp, flow, T)[0] > 0]
                                     for T in T_GRID])
            lo, hi = np.percentile(allpos, [5, 99.5])
            edges = np.exp(np.linspace(np.log(lo), np.log(hi), NQ + 1))

            # ---- CALIBRATION: a world where impact depends on SIZE ONLY, never on T
            r_real = np.empty_like(lp)
            r_real[0] = 0.0
            r_real[1:] = np.diff(lp)
            # fitted permanent per-trade kick g(v) = c * v^p, estimated on sign-conditioned moves
            lv = np.log(np.maximum(vol, 1e-9))
            y = eps * r_real
            A = np.column_stack([np.ones(n), lv])
            bb, *_ = np.linalg.lstsq(A, np.sign(y) * np.log(np.abs(y) + 1e-12), rcond=None)
            kick = eps * np.exp(bb[0]) * vol ** bb[1]
            kick *= float(np.dot(eps * r_real, eps * kick) / max(np.dot(kick, kick), 1e-30))
            noise = rng.normal(0.0, float(np.std(r_real - kick)), n)
            lp_null = np.cumsum(kick + noise)

            res = {}
            for mc in MIN_CELLS:
                p_real, bins_real, _ = profile(lp, flow, edges, mc)
                p_null, bins_null, _ = profile(lp_null, flow, edges, mc)
                def ratio(p):
                    if not p or p.get(5) in (None, 0):
                        return None
                    return round(p[500] / p[5], 3) if p.get(500) else None
                res["min_cell_{0}".format(mc)] = {
                    "positivity_bins": bins_real,
                    "real_profile_bps": ({str(k): round(v, 4) for k, v in p_real.items()}
                                         if p_real else None),
                    "size_only_null_profile_bps": ({str(k): round(v, 4)
                                                    for k, v in p_null.items()}
                                                   if p_null else None),
                    "real_T500_over_T5": ratio(p_real),
                    "null_T500_over_T5": ratio(p_null),
                }
            per[sym] = res
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T45", "lane": "C", "utc": _utc(),
           "contradiction": "CT-017",
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "prediction_kissell": "R(Q,T) at fixed Q DECREASES in T",
           "prediction_bouchaud": "R(Q,T) at fixed Q is FLAT in T (T cancels, TQP 12.3.2)",
           "shared_weaker_prediction": "R(Q,T) at fixed Q is NON-INCREASING in T",
           "calibration": ("a size-only impact world built from the real trades: each trade kicks "
                           "the price permanently by a fitted c*v^p, plus iid noise; nothing "
                           "depends on T by construction"),
           "threshold_swept": list(MIN_CELLS),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C45_CT017_T_DEPENDENCE_V1.json").write_text(json.dumps(art, indent=2),
                                                        encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("CT-017 -- both books predict R(Q,T) is NON-INCREASING in T at fixed Q.")
    w("%-9s %10s %6s %12s %12s %12s %12s" % ("sym", "min_cell", "bins", "real T=5",
                                             "real T=500", "real ratio", "null ratio"))
    for s in SYMS:
        for mc in MIN_CELLS:
            r = per[s]["min_cell_{0}".format(mc)]
            pr = r["real_profile_bps"]
            w("%-9s %10d %6d %12s %12s %12s %12s" % (
                s, mc, len(r["positivity_bins"]),
                pr["5"] if pr else "-", pr["500"] if pr else "-",
                r["real_T500_over_T5"], r["null_T500_over_T5"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
