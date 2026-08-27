r"""LANE C, round 36 -- does trading slower reduce impact here? Measured, not modelled.

Kissell Eq. (4.7) splits market impact into two pieces with different arguments:

    MI(bps) = b1 . I* . POV^a4          temporary, depends on the RATE
            + (1 - b1) . I*             permanent, does NOT depend on the rate

    I* = a1 (Q/ADV)^a2 sigma^a3         instantaneous impact, depends on the SIZE

The structural claim is what matters, not the parameterisation: there is a floor you cannot
execute your way out of. Slowing down shrinks the first term and leaves the second untouched. If
the floor is most of the cost, capacity is bounded by size alone and execution skill buys little.

C-T33 measured R(dV, T) directly -- the mean price move conditioned on the signed flow dV over T
trades -- so the decomposition is available WITHOUT fitting Kissell's functional form:

    hold dV FIXED and vary T.

Same imbalance, more trades to absorb it, lower participation rate. If R falls with T, a
temporary component exists and slowing helps by that much. If R is flat in T, the impact of a
given imbalance is permanent and there is nothing execution can do.

THIS ALSO CLOSES AN OLD CELL. C-T20 reported a permanent fraction of 0.595, derived from Eq.
(16.17) under a fair-pricing assumption and flagged in its own verdict as
PERMANENT_FRACTION_0_595_DERIVED_NOT_MEASURED. Nobody measured it. The quantity measured here --
the ratio of the large-T plateau of R to its small-T value at fixed dV -- is the empirical
counterpart.

METHOD DISCIPLINE, carried forward and now non-negotiable in this lane.
  - C-T35: a fitted exponent belongs to its window. Nothing is fitted here that is not also
    swept: the plateau is read across the full T range and the raw R(T) profile is published.
  - C-T31: block bootstrap for every uncertainty, with dependence intact.
  - C-T32: no counterfactual. R(dV, T) is a conditional mean of observed quantities.
  - C-T33: the window spans T trades from the price BEFORE the first to the price after the last.
  - C-T30: bin composition is reported.

READ-ONLY. This measures the market's impact function; it changes no sizing, no notional and no
configuration.
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
BOOT = 40
BLOCK = 100_000
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def windows(lp, flow, T):
    """dV over T trades, price change spanning exactly those T trades, and gross volume"""
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m) * T
    dv = flow[:m * T].reshape(m, T).sum(axis=1)[1:]
    gross = np.abs(flow[:m * T]).reshape(m, T).sum(axis=1)[1:]
    dp = (lp[i0 + T - 1] - lp[i0 - 1]) * 1e4
    return dv, dp, gross


def R_at(dv, dp, targets, rel=0.25):
    """mean price move for windows whose dV is within +-rel of each target (signed, buy side)"""
    out = []
    for q in targets:
        m = (dv >= q * (1 - rel)) & (dv <= q * (1 + rel))
        out.append({"target_dV": float("{0:.6g}".format(q)),
                    "n": int(m.sum()),
                    "R_bps": round(float(dp[m].mean()), 4) if m.sum() >= 50 else None})
    return out


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
            flow = np.where(a[:, 2] > 0.5, -1.0, 1.0) * a[:, 1]
            n = len(lp)

            # targets chosen from the SMALLEST T so they are attainable at every T
            dv5, dp5, g5 = windows(lp, flow, T_GRID[0])
            hi = float(np.percentile(dv5[dv5 > 0], 95))
            targets = [hi * f for f in (0.25, 0.5, 1.0)]

            by_T = {}
            for T in T_GRID:
                dv, dp, gross = windows(lp, flow, T)
                rows = R_at(dv, dp, targets)
                by_T[T] = {"rows": rows,
                           "mean_gross_volume": float("{0:.6g}".format(float(gross.mean()))),
                           "pov_at_targets": [round(float(q / gross.mean()), 5)
                                              for q in targets]}

            # permanent fraction: plateau of R(T) over its value at the smallest T, per target
            perm = []
            for k, q in enumerate(targets):
                prof = [(T, by_T[T]["rows"][k]["R_bps"]) for T in T_GRID
                        if by_T[T]["rows"][k]["R_bps"] is not None]
                if len(prof) < 4:
                    continue
                first = prof[0][1]
                plateau = float(np.mean([v for _, v in prof[-2:]]))
                perm.append({"target_dV": float("{0:.6g}".format(q)),
                             "R_at_smallest_T": first,
                             "R_plateau_large_T": round(plateau, 4),
                             "ratio_plateau_over_small": (round(plateau / first, 3)
                                                          if first else None),
                             "profile": [(T, v) for T, v in prof]})

            # bootstrap the ratio at the middle target, dependence intact
            bs = []
            nb = n // BLOCK
            mid = 1
            for _ in range(BOOT):
                idx = np.concatenate([np.arange(i, i + BLOCK)
                                      for i in rng.integers(0, n - BLOCK, nb)])
                lp2, fl2 = lp[idx], flow[idx]
                vals = []
                for T in T_GRID:
                    d2, p2, _ = windows(lp2, fl2, T)
                    r2 = R_at(d2, p2, [targets[mid]])[0]["R_bps"]
                    vals.append(r2)
                if all(v is not None for v in vals) and vals[0]:
                    bs.append(float(np.mean(vals[-2:])) / vals[0])
            per[sym] = {"targets_notional": [float("{0:.6g}".format(q)) for q in targets],
                        "by_T": by_T, "permanent_fraction": perm,
                        "ratio_boot_sd_mid_target": (round(float(np.std(bs, ddof=1)), 3)
                                                     if len(bs) > 5 else None),
                        "ratio_boot_n": len(bs)}
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T36", "lane": "C", "utc": _utc(),
           "structure_tested": ("Kissell Eq. (4.7): temporary impact scales with the trading RATE, "
                                "permanent impact does not. Held dV fixed and varied T."),
           "old_cell_closed": ("C-T20's PERMANENT_FRACTION_0_595_DERIVED_NOT_MEASURED"),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C36_PERMANENT_FLOOR_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    for s in SYMS:
        w("== {0}   boot sd of ratio {1} (n={2})".format(
            s, per[s]["ratio_boot_sd_mid_target"], per[s]["ratio_boot_n"]))
        w("%14s %9s %9s %9s %9s %9s %9s %9s %9s" % (("dV",) + tuple("T=%d" % t for t in T_GRID)
                                                    + ("plateau/T5",)))
        for p in per[s]["permanent_fraction"]:
            prof = dict(p["profile"])
            cells = "".join("%9s" % (("%.4f" % prof[t]) if t in prof else "-") for t in T_GRID)
            w("%14.6g%s %9s" % (p["target_dV"], cells, p["ratio_plateau_over_small"]))
        w("   POV at targets, T=5:   {0}".format(per[s]["by_T"][5]["pov_at_targets"]))
        w("   POV at targets, T=500: {0}".format(per[s]["by_T"][500]["pov_at_targets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
