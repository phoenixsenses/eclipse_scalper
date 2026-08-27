r"""LANE C, round 54 -- the two load-bearing levels this lane never re-measured in the book's clock.

C-T53's `next:` claimed: "what remains exposed is any LEVEL in bps per trade, and C-T52 already
corrected the only one that was load-bearing." That was too quick, and this round says so with a
measurement rather than a retraction of style. TWO more load-bearing quantities are in the
aggTrade clock:

  (1) C-T42's IMPACT LEVELS at the median window notional -- 0.439 / 0.772 / 1.330 bps -- which
      carry the headline "the binding cost is the FEE, not depth" via the ratios 22.8x / 13.0x /
      7.5x. They are levels in bps, measured over T aggTrades.

  (2) C-T29's ECONOMICS itself -- +0.2471 (BTC, T=50) and +0.2645 (ETH, T=20) bps -- the only
      economic number this lane owns. The rule trades "the past window's imbalance and holds one
      window", and the window is 50 AGGTRADES. In the book's clock that is about 25 market orders
      on BTC, so the rule was never evaluated at the horizon it appeared to be at.

Neither is a level-per-trade in the narrow sense C-T53 had in mind, which is exactly why the claim
slipped through: (1) is a level per WINDOW and (2) is a rate whose HORIZON is counted in the wrong
unit. `--who impact level capacity clock` returns nothing in English and one unrelated hit in
Turkish.

WHAT IS TESTED, AND WHAT WOULD OVERTURN WHAT.
  - if the impact level rises enough in the event clock, "the fee binds, not depth" weakens;
  - if the economics survives at the same NUMBER of events as it did at that number of aggTrades,
    the horizon relabelling is cosmetic;
  - if it survives only at a different horizon, then C-T42's grid was searched in the wrong unit
    and its best cell is not the best cell.

To separate those, the economics is run on a horizon grid in BOTH clocks rather than at one point.

CALIBRATION. The rule's null is the circular-shift null of C-T42 -- shift the price path relative
to the flow so the signal is decoupled while every marginal is preserved -- run in each clock
separately, because the two clocks have different lengths.

SAMPLE BY ARTIFACT PATH: data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms
per symbol -- the same population as C-T29 and C-T42.
THRESHOLD: event = consecutive aggTrades with identical ts_ms AND side (C-T53 measured this is an
upper bound on merging: 64.5% / 72.9% / 4.8% of multi-row events are genuine book walks).
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
HORIZONS = (5, 10, 20, 50, 100, 200)
FEE_RT = 10.0
NULL_REPS = 30
SEED = 20260827

CT42_IMPACT = {"BTCUSDT": 0.4390, "ETHUSDT": 0.7717, "SOLUSDT": 1.3298}
CT42_NOTIONAL = {"BTCUSDT": 194102.0, "ETHUSDT": 134504.0, "SOLUSDT": 64676.1}
CT42_EDGE = {"BTCUSDT": 0.2636, "ETHUSDT": 0.2903, "SOLUSDT": 0.0924}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def collapse(ts, px, vol, bm):
    ch = np.empty(len(ts), dtype=bool)
    ch[0] = True
    ch[1:] = (ts[1:] != ts[:-1]) | (bm[1:] != bm[:-1])
    st = np.flatnonzero(ch)
    en = np.append(st[1:] - 1, len(ts) - 1)
    csum = np.concatenate([[0.0], np.cumsum(vol)])
    v = csum[en + 1] - csum[st]
    return np.log(px[en]), np.where(bm[st] > 0.5, -1.0, 1.0) * v, v


def edge(lp, flow, T, shift=0):
    n = len(lp)
    m = (n - 1) // T
    if m < 30:
        return None
    i0 = np.arange(1, m - 1) * T
    s = np.sign(flow[:m * T].reshape(m, T).sum(axis=1))[1:len(i0) + 1]
    lpx = np.roll(lp, shift) if shift else lp
    en = np.clip(i0 + T - 1, 0, n - 1)
    ex = np.clip(en + T, 0, n - 1)
    g = (lpx[ex] - lpx[en]) * 1e4 * s
    g = g[s != 0]
    if len(g) < 100:
        return None
    mu = float(g.mean())
    se = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {"n": int(len(g)), "bps": round(mu, 4), "se": round(se, 4),
            "t": round(mu / se, 2) if se > 0 else None}


def impact_at(lp, flow, T, target):
    """mean |price move| for windows whose |signed flow| is within +-25% of `target`"""
    n = len(lp)
    m = (n - 1) // T
    i0 = np.arange(1, m) * T
    dv = flow[:m * T].reshape(m, T).sum(axis=1)[1:]
    dp = (lp[i0 + T - 1] - lp[i0 - 1]) * 1e4
    mk = (np.abs(dv) >= target * 0.75) & (np.abs(dv) <= target * 1.25)
    if mk.sum() < 100:
        return None, int(mk.sum())
    return round(float(np.mean(np.abs(dp[mk]))), 4), int(mk.sum())


def main() -> int:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,notional,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, vol, bm = a[:, 0], a[:, 1], a[:, 2], a[:, 3]

            lp_a = np.log(px)
            fl_a = np.where(bm > 0.5, -1.0, 1.0) * vol
            lp_e, fl_e, _ = collapse(ts, px, vol, bm)

            grids = {}
            for label, lp, fl in (("aggTrade", lp_a, fl_a), ("market_order", lp_e, fl_e)):
                g = {}
                for T in HORIZONS:
                    r = edge(lp, fl, T)
                    if r:
                        nul = []
                        for _ in range(NULL_REPS):
                            sh = int(rng.integers(T * 10, len(lp) - T * 10))
                            rr = edge(lp, fl, T, shift=sh)
                            if rr:
                                nul.append(rr["bps"])
                        r["null_mean"] = round(float(np.mean(nul)), 4) if nul else None
                        r["null_sd"] = round(float(np.std(nul, ddof=1)), 4) if len(nul) > 5 \
                            else None
                        g[T] = r
                best = max(g, key=lambda k: g[k]["bps"]) if g else None
                grids[label] = {"grid": g, "best_T": best,
                                "best_bps": g[best]["bps"] if best else None}

            imp = {}
            for label, lp, fl in (("aggTrade", lp_a, fl_a), ("market_order", lp_e, fl_e)):
                v, nn = impact_at(lp, fl, 50, CT42_NOTIONAL[sym])
                imp[label] = {"impact_bps_at_C_T42_notional": v, "n_windows": nn,
                              "fee_over_impact": (round(FEE_RT / v, 1) if v else None)}

            per[sym] = {
                "n_aggtrades": int(len(px)), "n_events": int(len(lp_e)),
                "economics": grids,
                "impact_level": imp,
                "C_T42_published": {"impact_bps": CT42_IMPACT[sym],
                                    "edge_bps_T50": CT42_EDGE[sym],
                                    "notional": CT42_NOTIONAL[sym]},
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T54", "lane": "C", "utc": _utc(),
           "corrects": ("C-T53's `next:` claim that C-T52 had corrected the only load-bearing "
                        "level; two more were in the aggTrade clock"),
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "fee_rt_bps": FEE_RT, "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C54_ECONOMICS_EVENT_CLOCK_V1.json").write_text(json.dumps(art, indent=2),
                                                           encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("ECONOMICS -- the imbalance rule, gross bps, on a horizon grid in BOTH clocks")
    w("%-9s %-14s %s" % ("sym", "clock", "".join("%10s" % ("T=%d" % t) for t in HORIZONS)))
    for s in SYMS:
        for k in ("aggTrade", "market_order"):
            g = per[s]["economics"][k]["grid"]
            w("%-9s %-14s %s   best T=%s (%s)" % (
                s, k, "".join("%10s" % (g[t]["bps"] if t in g else "-") for t in HORIZONS),
                per[s]["economics"][k]["best_T"], per[s]["economics"][k]["best_bps"]))
    w("")
    w("IMPACT LEVEL at C-T42's median notional, and the ratio that carried the headline")
    w("%-9s %14s %16s %16s %14s" % ("sym", "notional", "impact aggTrade", "impact EVENT",
                                    "fee/impact ev"))
    for s in SYMS:
        p = per[s]
        w("%-9s %14s %16s %16s %14s" % (
            s, p["C_T42_published"]["notional"],
            p["impact_level"]["aggTrade"]["impact_bps_at_C_T42_notional"],
            p["impact_level"]["market_order"]["impact_bps_at_C_T42_notional"],
            p["impact_level"]["market_order"]["fee_over_impact"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
