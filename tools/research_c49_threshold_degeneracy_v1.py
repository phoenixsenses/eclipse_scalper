r"""LANE C, round 49 -- D-E10's threshold-degeneracy challenge, applied to this lane's own edge.

D-E10, to C: "if any exponent of yours is estimated off a threshold crossing, its low-threshold
limit may be measuring the same degeneracy rather than the mechanism", with the measured instance
that lowering their threshold SHORTENED the duration (18.10 -> 15.32 -> 9.17 minutes as k went
10 -> 4 -> 0) because at low k the crossing happens immediately.

`--who threshold degeneracy selection` returns nothing in English and nothing in Turkish. Both
languages, discriminating terms. Nobody has tested this lane's quantities for it.

WHICH OF MINE IS EXPOSED. Most are not: chi, H, beta and h(q) are partial-sum scalings with no
threshold anywhere. One is: C-T42's headline cell was `thresh_p95` -- the imbalance rule applied
only to windows whose |imbalance| clears the 95th percentile -- and it paid 0.42 bps against 0.26
with no threshold at all. That number is in every economic statement this lane has made since.

THE DISCRIMINATING TEST, because the naive one decides nothing. Edge rising with the threshold is
expected under BOTH a real mechanism and a selection artefact, so the aggregate curve cannot
separate them. What separates them is WHAT the threshold buys, and the edge factorises:

    edge_bps  ~  (2 * hit_rate - 1)  x  E|forward move|
                 \_______________/      \____________/
                    DIRECTION              MAGNITUDE

If the hit rate is flat while E|move| rises, the threshold is buying magnitude only: it selects
windows that move more in both directions and a sign-conditioned mean inherits that. If the hit
rate rises too, the threshold is buying direction, which is a mechanism.

That distinction is economically load-bearing here and not a technicality. Against a FIXED fee in
bps, buying magnitude is worth something -- so the test is not "is the edge real" but "is what the
threshold buys the thing C-T43 already showed does not scale".

CALIBRATION BEFORE THE TEST. The same sweep is run on a null that destroys direction and keeps
magnitude exactly (the imbalance sign is randomised, the forward moves are untouched). Its hit
rate must be 0.5 and its edge 0 at every threshold; whatever it does instead is the floor.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol -- the same population as C-T29/C-T42.
THRESHOLD DECLARED AND SWEPT: |imbalance| percentile at 0 / 50 / 75 / 90 / 95 / 99 / 99.5.
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
T0 = 50
PCTS = (0.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.5)
NULL_REPS = 40
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def build(lp, flow, T=T0):
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m - 1) * T
    tot = flow[:m * T].reshape(m, T).sum(axis=1)[1:len(i0) + 1]
    entry = np.clip(i0 + T - 1, 0, n - 1)
    exit_ = np.clip(entry + T, 0, n - 1)
    fwd = (lp[exit_] - lp[entry]) * 1e4
    return tot, fwd


def cells(tot, fwd, sign_override=None):
    s = np.sign(tot) if sign_override is None else sign_override
    a = np.abs(tot)
    out = {}
    for p in PCTS:
        cut = 0.0 if p <= 0 else float(np.percentile(a, p))
        m = (a >= cut) & (s != 0)
        if m.sum() < 200:
            continue
        g = fwd[m] * s[m]
        mu = float(g.mean())
        se = float(g.std(ddof=1) / np.sqrt(m.sum()))
        hit = float((g > 0).mean())
        mag = float(np.abs(fwd[m]).mean())
        out[str(p)] = {"n": int(m.sum()), "cut_notional": float("{0:.6g}".format(cut)),
                       "edge_bps": round(mu, 4), "se": round(se, 4),
                       "t": round(mu / se, 2) if se > 0 else None,
                       "hit_rate": round(hit, 4),
                       "mean_abs_forward_bps": round(mag, 4),
                       "direction_term_2h_minus_1": round(2 * hit - 1, 4),
                       "implied_edge_from_factorisation": round((2 * hit - 1) * mag, 4)}
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
            tot, fwd = build(lp, flow)

            real = cells(tot, fwd)

            # NULL: direction destroyed, magnitude and the selection kept exactly
            acc = {}
            for _ in range(NULL_REPS):
                s_rand = rng.choice([-1.0, 1.0], size=len(tot))
                c = cells(tot, fwd, sign_override=s_rand)
                for k, v in c.items():
                    acc.setdefault(k, {"edge": [], "hit": []})
                    acc[k]["edge"].append(v["edge_bps"])
                    acc[k]["hit"].append(v["hit_rate"])
            null = {k: {"edge_mean": round(float(np.mean(v["edge"])), 4),
                        "edge_sd": round(float(np.std(v["edge"], ddof=1)), 4),
                        "hit_mean": round(float(np.mean(v["hit"])), 4)}
                    for k, v in acc.items()}

            ks = [k for k in real]
            d0, d1 = real[ks[0]], real[ks[-1]]
            per[sym] = {
                "real": real, "null_direction_destroyed": null,
                "hit_rate_lift": round(d1["hit_rate"] - d0["hit_rate"], 4),
                "magnitude_lift_x": round(d1["mean_abs_forward_bps"]
                                          / d0["mean_abs_forward_bps"], 3),
                "edge_lift_x": (round(d1["edge_bps"] / d0["edge_bps"], 3)
                                if d0["edge_bps"] else None),
                "direction_term_lift_x": (round(d1["direction_term_2h_minus_1"]
                                                / d0["direction_term_2h_minus_1"], 3)
                                          if d0["direction_term_2h_minus_1"] else None),
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T49", "lane": "C", "utc": _utc(),
           "challenge_from": "D-E10 (threshold degeneracy)",
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "threshold_swept": list(PCTS),
           "factorisation": "edge_bps ~ (2*hit_rate - 1) x E|forward move|",
           "null": ("imbalance sign randomised; magnitude and the selection kept exactly. "
                    "hit rate must be 0.5 and edge 0 at every threshold"),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C49_THRESHOLD_DEGENERACY_V1.json").write_text(json.dumps(art, indent=2),
                                                          encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    for s in SYMS:
        p = per[s]
        w("== {0}".format(s))
        w("%8s %9s %10s %8s %10s %12s %14s %11s" % (
            "pct", "n", "edge bps", "t", "hit rate", "2h-1", "E|fwd| bps", "null edge"))
        for k, v in p["real"].items():
            nz = p["null_direction_destroyed"].get(k, {})
            w("%8s %9d %10s %8s %10s %12s %14s %11s" % (
                k, v["n"], v["edge_bps"], v["t"], v["hit_rate"],
                v["direction_term_2h_minus_1"], v["mean_abs_forward_bps"],
                "{0}+-{1}".format(nz.get("edge_mean"), nz.get("edge_sd"))))
        w("   from lowest to highest threshold:  edge x{0}   magnitude x{1}   "
          "direction term x{2}   hit-rate lift {3}".format(
              p["edge_lift_x"], p["magnitude_lift_x"], p["direction_term_lift_x"],
              p["hit_rate_lift"]))
        w("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
