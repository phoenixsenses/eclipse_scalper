r"""LANE C, round 51 -- the large-tick model, applied to the symbol that turns out to need it.

C-T50 established, by Bouchaud's own numerical criterion (sec. 4.1 iv, trade-through rate), that
SOL is a LARGE-TICK instrument at 0.297% while BTC and ETH are small-tick at 10.752% and 13.814%.
Its own `next:` recorded a debt: "applying it retrospectively to this lane's SOL results is the
obvious next step and I have not done it." This round pays that debt with the machinery the corpus
assigns to that regime and that this lane has never used.

`--who queue race refill reversion` returns nothing in English and nothing in Turkish. Both
languages, discriminating terms. Nobody here has measured it.

THE CORPUS'S LARGE-TICK MECHANISM, sec. 7.5 "What Happens After a Race Ends?", verbatim:

    "Imagine the bid-queue has emptied first. Two things can happen at the now vacant price level:
     - With probability rho_0, a buy limit order immediately refills the old bid position. In this
       case, the mid-price REVERTS to its previous position (after having briefly moved down by
       half a tick).
     - With probability 1 - rho_0, a sell limit order immediately refills the old bid position. In
       this case the mid-price has moved down by one tick."

rho_0 itself needs book data this estate does not carry. Its OBSERVABLE CONSEQUENCE does not: if a
large fraction of one-tick moves are immediately undone, then consecutive non-zero price changes
should REVERSE more often than they continue, and more so in the large-tick regime.

THE PREDICTION, stated before measurement and directional across symbols:
    P(next non-zero move reverses) > 1/2, and LARGER on SOL than on BTC and ETH.

This is also the mechanism behind two results this lane already published without a mechanism:
C-T38's sub-diffusive short scale (H = 0.33-0.38 below T = 10) and the sharp dip at lag 2 in
C-T35's propagator, which I labelled "bounce" and left there.

CALIBRATION BEFORE THE TEST. The same estimator is run on the same series with the non-zero moves
SHUFFLED, which destroys ordering and leaves the marginal distribution of move sizes untouched. It
must return 0.5. Whatever it returns instead is the floor.

DECLARED THRESHOLDS, BOTH REPORTED.
    any non-zero move       every consecutive pair of non-zero price changes
    exactly one tick        both moves are exactly one tick, which is the case sec. 7.5 describes

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol -- the same population as every round of this lane since C-T33.
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
NULL_REPS = 40
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def measure_tick(px):
    d = np.abs(np.diff(np.unique(px)))
    d = d[d > 0]
    if not len(d):
        return None
    q = float(np.percentile(d, 1))
    return float(np.min(d[d >= q * 0.5])) if np.any(d >= q * 0.5) else float(d.min())


def reversal_rates(steps_ticks):
    """P(next non-zero move reverses), overall and restricted to one-tick pairs"""
    s = np.sign(steps_ticks)
    a, b = s[:-1], s[1:]
    flip_any = float((a * b < 0).mean())
    m = (np.abs(steps_ticks[:-1]) == 1) & (np.abs(steps_ticks[1:]) == 1)
    flip_one = float((a[m] * b[m] < 0).mean()) if m.sum() > 200 else None
    return flip_any, flip_one, int(m.sum())


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            px = np.array(con.execute(
                "select price from agg_trades where symbol=? order by ts_ms limit ?",
                (sym, NROWS)).fetchall(), dtype=np.float64).ravel()
            tick = measure_tick(px)
            d = np.diff(px)
            nz = d[np.abs(d) > tick * 0.5]
            steps = np.round(nz / tick).astype(np.int64)
            steps = steps[steps != 0]

            flip_any, flip_one, n_one = reversal_rates(steps)

            null_any, null_one = [], []
            for _ in range(NULL_REPS):
                sh = steps[rng.permutation(len(steps))]
                fa, fo, _ = reversal_rates(sh)
                null_any.append(fa)
                if fo is not None:
                    null_one.append(fo)
            na, nas = float(np.mean(null_any)), float(np.std(null_any, ddof=1))
            no = float(np.mean(null_one)) if null_one else None
            nos = float(np.std(null_one, ddof=1)) if len(null_one) > 5 else None

            per[sym] = {
                "tick": tick,
                "n_nonzero_moves": int(len(steps)),
                "zero_move_share_of_trades": round(float(1.0 - len(steps) / max(len(d), 1)), 4),
                "share_of_moves_that_are_one_tick": round(
                    float((np.abs(steps) == 1).mean()), 4),
                "P_reverse_any": round(flip_any, 4),
                "P_reverse_one_tick": (round(flip_one, 4) if flip_one is not None else None),
                "n_one_tick_pairs": n_one,
                "null_any_mean": round(na, 4), "null_any_sd": round(nas, 5),
                "null_one_mean": (round(no, 4) if no is not None else None),
                "z_any": round((flip_any - na) / nas, 1) if nas > 0 else None,
                "z_one": (round((flip_one - no) / nos, 1)
                          if (flip_one is not None and nos) else None),
                "excess_over_half_any": round(flip_any - 0.5, 4),
                "excess_over_half_one_tick": (round(flip_one - 0.5, 4)
                                              if flip_one is not None else None),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    order_pred = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]           # large-tick first
    order_obs = sorted(SYMS, key=lambda s: -per[s]["P_reverse_any"])
    art = {"study": "C-T51", "lane": "C", "utc": _utc(),
           "corpus_mechanism": ("Bouchaud sec. 7.5: after a queue race, with probability rho_0 the "
                                "vacated level is refilled and the MID-PRICE REVERTS"),
           "prediction_stated_before_measurement": (
               "P(next non-zero move reverses) > 1/2, and LARGER on the large-tick symbol"),
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "thresholds_declared": "any non-zero move; and both moves exactly one tick",
           "null": "non-zero moves shuffled: ordering destroyed, marginal sizes kept, must give 0.5",
           "predicted_order_most_reverting_first": order_pred,
           "observed_order_most_reverting_first": order_obs,
           "prediction_holds": bool(order_obs[0] == "SOLUSDT"),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C51_TICK_REVERSION_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("BOUCHAUD 7.5 -- a refilled level makes the mid REVERT.  Prediction: P(reverse) > 0.5, "
      "largest on the large-tick symbol.")
    w("%-9s %8s %12s %11s %13s %14s %10s %9s" % (
        "sym", "zero %", "n non-zero", "1-tick %", "P(rev) any", "P(rev) 1-tick",
        "null", "z"))
    for s in SYMS:
        p = per[s]
        w("%-9s %8.4f %12d %11.4f %13.4f %14s %10.4f %9s" % (
            s, p["zero_move_share_of_trades"], p["n_nonzero_moves"],
            p["share_of_moves_that_are_one_tick"], p["P_reverse_any"],
            p["P_reverse_one_tick"], p["null_any_mean"], p["z_any"]))
    w("")
    w("predicted order (most reverting first): {0}".format(order_pred))
    w("observed  order (most reverting first): {0}".format(order_obs))
    w("prediction holds (large-tick most reverting): {0}".format(art["prediction_holds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
