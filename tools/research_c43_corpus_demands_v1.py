r"""LANE C, round 43 -- what the corpus asks US, and the one demand of it that is runnable here.

The operator inverted the question: not what we ask the corpus, but what it asks us. A mechanical
sweep of all thirteen sources for demand constructions ("must be", "requires that", "cannot be
... unless", "the key question", "care must be taken") returned 437 methodological passages. Most
are generic. Two from sources this lane had never opened are not, and one of them completes a
structure this lane has been building for six rounds.

DEMAND 1 -- CHAN, AND IT HAS A NAME IN HERNAN & ROBINS.

    "One has to question whether there is a 'Heisenberg uncertainty principle' at work: THE ACT OF
     PLACING OR EXECUTING AN ORDER MIGHT ALTER THE BEHAVIOR OF THE OTHER MARKET PARTICIPANTS. So be
     very skeptical of a so-called backtest of a high-frequency strategy."      Chan, ch. 1

    "Our definition of a counterfactual outcome implicitly assumes that an individual's
     counterfactual outcome under treatment value a does not depend on other individuals'
     treatment values... IN THE PRESENCE OF INTERFERENCE, THE COUNTERFACTUAL Y_a FOR AN INDIVIDUAL
     IS NOT WELL DEFINED."                             Hernan & Robins, Fine Point 1.1

Chan's principle is H&R's INTERFERENCE, and naming it makes the consequence much sharper than "be
skeptical": the estimand is not well defined, not merely hard to estimate. That is the fourth
identifiability condition, and this lane's family of blocked measurements did not have it:

    C-T24  metaorder impact          the treatment is unobserved      (upstream)
    C-T32  reaction impact           exchangeability
    C-T36  temporary/permanent       positivity
    C-T43  any order-placing backtest INTERFERENCE                    (new)

The fourth is the widest of the four: it applies to every backtest of a strategy that places
orders, including C-T29's, and it cannot be repaired by more data from the same source.

DEMAND 2 -- CHAN AGAIN, AND THIS ONE IS RUNNABLE.

    "the predictive power of any backtest rests on the central assumption that THE STATISTICAL
     PROPERTIES OF THE PRICE SERIES ARE UNCHANGING, so that the trading rules that were profitable
     in the past will be profitable in the future. This assumption is, of course, invalidated
     often in varying degrees."                                        Chan, ch. 1

C-T42 answered the weak form of this with a 70/30 hold-out. The strong form is different: not
"does the value hold in one later window" but "are the PROPERTIES THE RULE DEPENDS ON stationary
at all". Those properties are named rather than guessed, because earlier rounds measured them:

    the edge itself          C-T29
    sign memory chi          C-T28, the mechanism the edge rests on
    immediate impact R(1)    C-T33/C-T34
    realised volatility      the scale everything is denominated in
    mean trade notional      the size axis of the impact surface

All five are measured block by block, and the question asked of each is whether its variation
across blocks exceeds what its own sampling error allows.

Per the operator's standing instruction, every property is tested rather than a representative one.
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
NBLOCK = 10
T0 = 50
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def chi_of(eps, lo=20, hi=1000):
    n = len(eps)
    Ts = np.unique(np.round(np.geomspace(lo, hi, 8)).astype(int))
    T, S = [], []
    for t in Ts:
        m = n // t
        if m < 100:
            continue
        T.append(float(t))
        S.append(float(np.std(eps[:m * t].reshape(m, t).sum(axis=1), ddof=1)))
    if len(T) < 4:
        return float("nan")
    lt, ls = np.log(T), np.log(S)
    A = np.column_stack([np.ones(len(lt)), lt])
    b, *_ = np.linalg.lstsq(A, ls, rcond=None)
    return float(b[1])


def edge_of(lp, flow, T=T0):
    n = len(lp)
    m = (n - 1) // T
    if m < 30:
        return None
    i0 = np.arange(1, m - 1) * T
    s = np.sign(flow[:m * T].reshape(m, T).sum(axis=1))[1:len(i0) + 1]
    entry = np.clip(i0 + T - 1, 0, n - 1)
    exit_ = np.clip(entry + T, 0, n - 1)
    g = (lp[exit_] - lp[entry]) * 1e4 * s
    g = g[s != 0]
    if len(g) < 50:
        return None
    return float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g))), int(len(g))


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            lp = np.log(a[:, 0])
            eps = np.where(a[:, 2] > 0.5, -1.0, 1.0)
            flow = eps * a[:, 1]
            n = len(lp)
            m = n // NBLOCK

            props = {k: [] for k in ("edge_bps", "edge_se", "chi_sign", "R1_bps",
                                     "realised_vol_bps", "mean_notional")}
            for b in range(NBLOCK):
                sl = slice(b * m, (b + 1) * m)
                lpb, epsb, flb = lp[sl], eps[sl], flow[sl]
                r = np.diff(lpb) * 1e4
                e_ = edge_of(lpb, flb)
                props["edge_bps"].append(round(e_[0], 4) if e_ else None)
                props["edge_se"].append(round(e_[1], 4) if e_ else None)
                props["chi_sign"].append(round(chi_of(epsb), 4))
                props["R1_bps"].append(round(float(np.mean(epsb[1:] * r)), 5))
                props["realised_vol_bps"].append(round(float(np.std(r, ddof=1)), 4))
                props["mean_notional"].append(float("{0:.6g}".format(float(a[sl, 1].mean()))))

            def spread(v):
                x = np.array([q for q in v if q is not None], float)
                return {"min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
                        "ratio_max_over_min": (round(float(x.max() / x.min()), 2)
                                               if x.min() > 0 else None),
                        "cv": round(float(x.std(ddof=1) / abs(x.mean())), 4)}

            # is the edge's block-to-block variation larger than its own sampling error?
            e = np.array([q for q in props["edge_bps"] if q is not None], float)
            se = np.array([q for q in props["edge_se"] if q is not None], float)
            chi2 = float(np.sum((e - e.mean()) ** 2 / se ** 2))
            dof = len(e) - 1
            per[sym] = {
                "blocks": NBLOCK,
                "properties": props,
                "spread": {k: spread(v) for k, v in props.items()
                           if k not in ("edge_se",)},
                "edge_homogeneity_chi2": round(chi2, 1),
                "dof": dof,
                "chi2_over_dof": round(chi2 / dof, 2),
                "edge_is_stationary_at_its_own_error": bool(chi2 / dof < 2.0),
                "blocks_with_positive_edge": int(sum(1 for q in props["edge_bps"]
                                                     if q is not None and q > 0)),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T43", "lane": "C", "utc": _utc(),
           "demand_1": ("Chan's Heisenberg principle IS Hernan & Robins' INTERFERENCE "
                        "(Fine Point 1.1): under interference the counterfactual is NOT WELL "
                        "DEFINED. Fourth member of this lane's family of blocked measurements, "
                        "and the widest -- it applies to every backtest that places orders."),
           "demand_2": ("Chan: a backtest's predictive power rests on the statistical properties "
                        "of the series being unchanging. Tested here property by property."),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C43_CORPUS_DEMANDS_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("CHAN'S STATIONARITY DEMAND, property by property, across {0} blocks".format(NBLOCK))
    for s in SYMS:
        p = per[s]
        w("== {0}   edge chi2/dof {1} (stationary at its own error: {2})   positive blocks {3}/{4}"
          .format(s, p["chi2_over_dof"], p["edge_is_stationary_at_its_own_error"],
                  p["blocks_with_positive_edge"], NBLOCK))
        for k, v in p["spread"].items():
            w("     %-18s min %-12s max %-12s max/min %-8s cv %s" % (
                k, v["min"], v["max"], v["ratio_max_over_min"], v["cv"]))
        w("     edge by block: " + " ".join(
            str(x) for x in p["properties"]["edge_bps"]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
