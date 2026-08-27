r"""LANE C, round 37 -- classify this lane's three non-identifiabilities, and map the one that is
a positivity violation.

Three rounds have now ended in "this estate cannot measure that". Hernan and Robins is the corpus
source about exactly this question, and it turns out the three failures land in three DIFFERENT
places, one of which is empirically checkable and one of which corrects a verdict I published.

    C-T24  metaorder impact       the TREATMENT ITSELF is unobserved (which child orders belong
                                  to which parent). That is upstream of the identifiability
                                  conditions, not a violation of them: H&R assume A is measured.

    C-T32  reaction impact        I wrote that the counterfactual "cannot be implemented" because
                                  the two worlds are mutually exclusive and history cannot be
                                  replayed. That is Bouchaud's statement about the INDIVIDUAL
                                  effect, and H&R Chapters 1-3 exist to show that AVERAGE causal
                                  effects are identifiable from observational data without ever
                                  observing a counterfactual. So the obstacle is not that
                                  counterfactuals are unobservable -- it is EXCHANGEABILITY:
                                  Bouchaud's own F_t (the information that triggered the order)
                                  confounds, and it is unmeasured. Different obstacle, different
                                  remedy: not "replay history" but "measure or block F_t".

    C-T36  temporary/permanent    needs Q and V varied independently. In H&R's language that is
                                  POSITIVITY -- Pr[A = a | L = l] > 0 for all l in the population
                                  of interest -- and positivity, unlike exchangeability, is
                                  EMPIRICALLY CHECKABLE.

    "Thus we say that there is positivity if Pr[A = a | L = l] > 0 for all a involved in the
     causal contrast... Positivity is only needed for the values l that are present in the
     population of interest."                                        Hernan & Robins, Sec. 3.3

THE MEASUREMENT. For the temporary/permanent contrast the covariate L is the order size Q and the
treatment A is the horizon T over which it is absorbed. Positivity asks whether every (Q, T) cell
involved in the contrast carries mass. C-T36 hit this informally -- 9 of 40 bootstrap replicates
usable -- and reported it as an estimator failure. It is a structural one, and it has a shape:
this maps the joint support and asks whether a RESTRICTED region exists where positivity holds.

If it does, the contrast is identifiable THERE, and C-T36's blanket verdict narrows from
"not identifiable" to "not identifiable outside this region" -- which is a materially different
and more useful statement.

READ-ONLY. Nothing is executed, sized or configured.
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
MIN_CELL = 200          # a cell with fewer windows than this is treated as empty
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


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            flow = np.where(a[:, 2] > 0.5, -1.0, 1.0) * a[:, 1]

            W = {T: windows(lp, flow, T) for T in T_GRID}

            # Q bins on the BUY side, log-spaced over the pooled support of all horizons
            allpos = np.concatenate([W[T][0][W[T][0] > 0] for T in T_GRID])
            lo, hi = np.percentile(allpos, [5, 99.5])
            edges = np.exp(np.linspace(np.log(lo), np.log(hi), NQ + 1))

            grid, R = {}, {}
            for T in T_GRID:
                dv, dp = W[T]
                cnt, rr = [], []
                for b in range(NQ):
                    m = (dv >= edges[b]) & (dv < edges[b + 1])
                    cnt.append(int(m.sum()))
                    rr.append(round(float(dp[m].mean()), 4) if m.sum() >= MIN_CELL else None)
                grid[T] = cnt
                R[T] = rr

            # the positivity region: Q bins where EVERY horizon in the contrast is populated
            full = [b for b in range(NQ) if all(grid[T][b] >= MIN_CELL for T in T_GRID)]
            # and the largest set of horizons that share at least one populated Q bin
            best = None
            for i in range(len(T_GRID)):
                for j in range(i + 1, len(T_GRID) + 1):
                    sub = T_GRID[i:j]
                    bins = [b for b in range(NQ)
                            if all(grid[T][b] >= MIN_CELL for T in sub)]
                    if bins and (best is None or len(sub) > len(best[0])):
                        best = (sub, bins)

            # if a region exists, run the contrast inside it
            contrast = None
            if best and len(best[0]) >= 3:
                sub, bins = best
                prof = {}
                for T in sub:
                    vals = [R[T][b] for b in bins if R[T][b] is not None]
                    prof[T] = round(float(np.mean(vals)), 4) if vals else None
                ok = [T for T in sub if prof[T] is not None]
                contrast = {"horizons": list(sub), "q_bins_used": bins,
                            "q_range": [float("{0:.6g}".format(edges[min(bins)])),
                                        float("{0:.6g}".format(edges[max(bins) + 1]))],
                            "mean_R_by_T": prof,
                            "ratio_last_over_first": (
                                round(prof[ok[-1]] / prof[ok[0]], 3)
                                if len(ok) >= 2 and prof[ok[0]] else None)}

            per[sym] = {
                "q_edges": [float("{0:.6g}".format(x)) for x in edges],
                "counts_by_T": {str(T): grid[T] for T in T_GRID},
                "mean_R_by_T": {str(T): R[T] for T in T_GRID},
                "q_bins_with_full_positivity": full,
                "positivity_holds_anywhere_for_all_seven_horizons": bool(len(full) > 0),
                "largest_horizon_set_with_common_support": (list(best[0]) if best else None),
                "common_support_bins": (best[1] if best else None),
                "contrast_inside_the_region": contrast,
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T37", "lane": "C", "utc": _utc(), "min_cell": MIN_CELL,
           "source": "Hernan & Robins Sec. 3.1-3.3 (identifiability conditions; positivity)",
           "classification": {
               "C-T24_metaorder": "treatment unobserved -- upstream of the conditions",
               "C-T32_reaction": ("EXCHANGEABILITY, not counterfactual unobservability. "
                                  "C-T32's stated reason is corrected."),
               "C-T36_temp_perm": "POSITIVITY -- and positivity is empirically checkable"},
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C37_POSITIVITY_MAP_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    for s in SYMS:
        p = per[s]
        w("== {0}".format(s))
        w("%10s %s" % ("Q bin", "".join("%8s" % ("T=%d" % T) for T in T_GRID)))
        for b in range(NQ):
            w("%10.4g %s" % (p["q_edges"][b],
                             "".join("%8d" % p["counts_by_T"][str(T)][b] for T in T_GRID)))
        w("   full positivity (all 7 horizons): bins {0}".format(
            p["q_bins_with_full_positivity"] or "NONE"))
        w("   largest horizon set with common support: {0} over bins {1}".format(
            p["largest_horizon_set_with_common_support"], p["common_support_bins"]))
        c = p["contrast_inside_the_region"]
        if c:
            w("   contrast inside the region: R by T = {0}   ratio {1}".format(
                c["mean_R_by_T"], c["ratio_last_over_first"]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
