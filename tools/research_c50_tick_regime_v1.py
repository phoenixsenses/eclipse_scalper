r"""LANE C, round 50 -- which tick regime is this estate in, by the corpus's own numerical criterion?

Read with tools/corpus_text_v1.py. `--who large tick regime classification` returns nothing in
English and one irrelevant hit in Turkish; both languages, discriminating terms. Nobody here has
classified these instruments.

WHY IT MATTERS MORE THAN IT SOUNDS. Bouchaud does not treat large-tick and small-tick instruments
as the same market with a different parameter. He gives large-tick stocks their OWN CHAPTERS --
ch. 6 "Single-Queue Dynamics for Large-Tick Stocks" and ch. 7 "Joint-Queue Dynamics for Large-Tick
Stocks" -- while the propagator and impact machinery this lane has spent fifteen rounds fitting
(ch. 11-13) is developed for the small-tick case. If these symbols are large-tick by the book's own
criterion, the lane has been applying the wrong model family, and that is a larger structural
finding than anything it has published.

THE CRITERION IS NUMERICAL AND IT IS MEASURABLE HERE. Bouchaud sec. 4.1, point (iv), verbatim:

    "The number of trade-through market orders (i.e. orders that match at several different prices
     and therefore walk up the order book) is on the order of A FEW PERCENT for small-tick stocks,
     and A FEW PER THOUSAND for large-tick stocks."

aggTrades aggregate the same-price fills of one taker order, so an order that walks the book
appears as SEVERAL consecutive aggTrades sharing a timestamp and a side at different prices. That
reconstruction is the measurement.

DECLARED THRESHOLDS, AND BOTH REPORTED RATHER THAN ONE CHOSEN.
    run definition   consecutive aggTrades with identical ts_ms AND identical side
    strict walk      the run's prices must be MONOTONE in the aggressor's direction, which is what
                     "walks up the book" means; a run that is merely multi-priced can also be two
                     unrelated orders landing in the same millisecond
Both rates are reported. The gap between them is the ambiguity the millisecond stamp imposes, and
it is published rather than resolved by preference.

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
SEED = 20260827

# the book's own bands, sec. 4.1 (iv)
SMALL_TICK_BAND = (0.01, 0.10)      # "a few percent"
LARGE_TICK_BAND = (0.001, 0.01)     # "a few per thousand"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def runs(ts, side):
    """start index of each maximal run of identical (ts_ms, side)"""
    change = np.empty(len(ts), dtype=bool)
    change[0] = True
    change[1:] = (ts[1:] != ts[:-1]) | (side[1:] != side[:-1])
    return np.flatnonzero(change)


def main() -> int:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    per = {}
    try:
        for sym in SYMS:
            a = np.array(con.execute(
                "select ts_ms,price,is_buyer_maker from agg_trades where symbol=? "
                "order by ts_ms limit ?", (sym, NROWS)).fetchall(), dtype=np.float64)
            ts, px, bm = a[:, 0], a[:, 1], a[:, 2]
            buy = bm < 0.5                     # aggressor is a buyer
            starts = runs(ts, bm)
            ends = np.append(starts[1:], len(ts))
            n_events = len(starts)

            multi = 0
            walk = 0
            lens = []
            depth_ticks = []
            tick = None
            d = np.abs(np.diff(np.unique(px)))
            d = d[d > 0]
            if len(d):
                q = float(np.percentile(d, 1))
                tick = float(np.min(d[d >= q * 0.5])) if np.any(d >= q * 0.5) else float(d.min())

            for s, e in zip(starts, ends):
                L = e - s
                lens.append(L)
                if L < 2:
                    continue
                p = px[s:e]
                if np.unique(p).size < 2:
                    continue
                multi += 1
                up = bool(buy[s])
                mono = np.all(np.diff(p) >= 0) if up else np.all(np.diff(p) <= 0)
                if mono:
                    walk += 1
                    if tick:
                        depth_ticks.append(float((p.max() - p.min()) / tick))

            lens = np.array(lens)
            rate_multi = multi / n_events
            rate_walk = walk / n_events

            def band(r):
                if LARGE_TICK_BAND[0] <= r < LARGE_TICK_BAND[1]:
                    return "LARGE_TICK (a few per thousand)"
                if SMALL_TICK_BAND[0] <= r <= SMALL_TICK_BAND[1]:
                    return "SMALL_TICK (a few percent)"
                if r < LARGE_TICK_BAND[0]:
                    return "BELOW the large-tick band"
                return "ABOVE the small-tick band"

            per[sym] = {
                "tick_measured": tick,
                "n_aggtrades": int(len(ts)),
                "n_market_order_events": int(n_events),
                "mean_aggtrades_per_event": round(float(lens.mean()), 4),
                "share_of_events_with_more_than_one_aggtrade": round(
                    float((lens > 1).mean()), 5),
                "trade_through_rate_multi_price": round(rate_multi, 6),
                "trade_through_rate_strict_walk": round(rate_walk, 6),
                "strict_walk_share_of_multi_price": (round(walk / multi, 4) if multi else None),
                "median_walk_depth_ticks": (round(float(np.median(depth_ticks)), 2)
                                            if depth_ticks else None),
                "classification_multi_price": band(rate_multi),
                "classification_strict_walk": band(rate_walk),
                "book_bands": {"small_tick": "1% to 10%", "large_tick": "0.1% to 1%"},
            }
            sys.stderr.write("{0} done\n".format(sym))
    finally:
        con.close()

    art = {"study": "C-T50", "lane": "C", "utc": _utc(),
           "criterion": ("Bouchaud sec. 4.1 (iv): trade-through market orders are 'a few percent' "
                         "for small-tick stocks and 'a few per thousand' for large-tick"),
           "why": ("large-tick instruments get their own chapters (6, 7) while the propagator and "
                   "impact machinery this lane has fitted since C-T33 is the small-tick family"),
           "sample": ("data/microstructure_02.db :: agg_trades, first 2,000,000 rows by ts_ms "
                      "per symbol"),
           "thresholds_declared": ("run = consecutive aggTrades with identical ts_ms and side; "
                                   "strict walk additionally requires monotone prices in the "
                                   "aggressor's direction. Both rates reported."),
           "per_symbol": per}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "C50_TICK_REGIME_V1.json").write_text(json.dumps(art, indent=2), encoding="utf-8")

    enc = sys.stdout.encoding or "utf-8"

    def w(s):
        sys.stdout.write(s.encode(enc, "replace").decode(enc, "replace") + "\n")

    w("BOUCHAUD 4.1 (iv): small-tick ~ a few PERCENT; large-tick ~ a few PER THOUSAND")
    w("%-9s %8s %12s %14s %16s %12s %12s" % ("sym", "tick", "events", "multi-price %",
                                             "strict walk %", "walk/multi", "walk depth"))
    for s in SYMS:
        p = per[s]
        w("%-9s %8s %12d %13.3f%% %15.3f%% %12s %12s" % (
            s, p["tick_measured"], p["n_market_order_events"],
            100 * p["trade_through_rate_multi_price"],
            100 * p["trade_through_rate_strict_walk"],
            p["strict_walk_share_of_multi_price"], p["median_walk_depth_ticks"]))
    w("")
    for s in SYMS:
        p = per[s]
        w("  %-9s multi-price -> %-34s | strict walk -> %s" % (
            s, p["classification_multi_price"], p["classification_strict_walk"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
