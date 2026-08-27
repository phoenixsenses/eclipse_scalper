# -*- coding: utf-8 -*-
"""S52 -- Harris's question, asked of this estate for the first time: how big is the pot?

THE QUESTION THE CORPUS INSISTS ON AND NOBODY HERE HAS ASKED
------------------------------------------------------------
Harris, Trading and Exchanges, ch.5:

    "Trading is a zero-sum game in an important accounting sense.  In a zero-sum game,
     the total gains of the winners are exactly equal to the total losses of the losers.
     ...  To trade profitably, traders must trade with people who will lose.
     Profit-motivated traders therefore must understand WHY LOSERS TRADE to know when
     they should trade."

Futures are exactly zero-sum before fees and strictly negative-sum after.  So every basis
point this estate could ever earn has to come from a counterparty trading for a reason
other than profit.  **And this estate holds the one dataset that identifies such a
counterparty by construction: the liquidation feed.**  A forced liquidation is, by
definition, someone who did not choose to trade.  That is Harris's loser, with a
timestamp and a notional attached.

The size of that flow is a CEILING on the whole forced-flow line, computed from the
market's own accounting and independent of anyone's skill.  It has never been computed.

WHAT THIS IS NOT
----------------
Not a power calculation.  A previous lane established that 167,393 liquidations is a ROW
COUNT and not an N, and refused a download on power grounds.  That verdict stands and is
about ESTIMATION.  This is about SUPPLY: a sum, not a test.  The two are unrelated and
the distinction is the whole reason this can be asked at all.

THE ACCOUNTING
--------------
    pot/day  =  liquidated notional/day  x  (edge_bps - cost_bps) / 1e4

with the edge from §311/§315 (123.7 bps BUY, 136.9 SELL, beta-neutral continuation) and
cost from BINANCE_BASE.  Reported across a range of the edge, because the edge is an
input taken from a burned sample and not re-derived here.

One distinction Harris's accounting forces and which the naive version misses: §311/§315
measured a CONTINUATION.  The price keeps going after the liquidation.  So the immediate
counterparty -- whoever filled the forced order -- is not the winner; they are holding
something that keeps moving against them.  The 123.7 bps accrues to whoever trades WITH
the direction after the fact.  The pot is real but it is not the liquidated party's loss
handed to their counterparty; it is a flow available to whoever is positioned for the
continuation.
"""

import io
import json
import math
import sqlite3

DB = "file:data/microstructure_02.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S52_THE_POT_V1.json"

COST_BPS = 10.0
EDGES = (60.0, 123.7, 136.9)
# S50's duration-bound, pooled at POV=20% over the 212-symbol priced panel
S50_DURATION_BOUND = {1: 1025336, 5: 5126680, 15: 15380039, 60: 61520154, 240: 246080617}


def main():
    c = sqlite3.connect(DB, uri=True)
    row = c.execute(
        "SELECT COUNT(*), SUM(notional), MIN(ts_ms), MAX(ts_ms), COUNT(DISTINCT symbol) "
        "FROM liquidations WHERE ts_ms < ?", (CUT,)).fetchone()
    n, tot, lo, hi, nsym = row
    days = (hi - lo) / 86400000.0
    per_day = tot / days

    print("THE POT  --  forced liquidation flow, the identifiable non-profit-motivated")
    print("side of Harris's zero-sum game")
    print("  rows %s   symbols %d   span %.1f days (pre-cutoff)" % (format(n, ","), nsym, days))
    print("  total liquidated notional      $%s" % fmt(tot))
    print("  per day                        $%s" % fmt(per_day))
    print("  mean per liquidation           $%s" % fmt(tot / n))

    # concentration -- a pot held by three symbols is not a 761-symbol opportunity
    rows = c.execute(
        "SELECT symbol, SUM(notional) s FROM liquidations WHERE ts_ms < ? "
        "GROUP BY symbol ORDER BY s DESC", (CUT,)).fetchall()
    c.close()
    cum, top = 0.0, []
    for i, (s, v) in enumerate(rows):
        cum += v
        if i < 5:
            top.append((s, v, 100.0 * v / tot))
        if len(top) < 6 and cum / tot >= 0.5:
            pass
    n50 = 0
    run = 0.0
    for s, v in rows:
        run += v
        n50 += 1
        if run / tot >= 0.50:
            break
    n90 = 0
    run = 0.0
    for s, v in rows:
        run += v
        n90 += 1
        if run / tot >= 0.90:
            break
    print()
    print("  CONCENTRATION -- half the pot sits in %d symbols, ninety per cent in %d of %d"
          % (n50, n90, len(rows)))
    for s, v, p in top:
        print("    %-14s $%-16s %5.1f%%" % (s, fmt(v), p))

    print()
    print("THE POT IN DOLLARS PER DAY, AT EACH READING OF THE EDGE")
    print("  pot = liquidated notional x (edge - cost) / 1e4")
    print("  %-16s %14s %16s" % ("edge bps", "surplus bps", "pot per day"))
    pots = {}
    for e in EDGES:
        p = per_day * (e - COST_BPS) / 1e4
        pots[e] = p
        tag = "  <- §315 BUY" if e == 123.7 else ("  <- §311 SELL" if e == 136.9 else "")
        print("  %-16.1f %14.1f %16s%s" % (e, e - COST_BPS, "$" + fmt(p), tag))

    print()
    print("AND WHAT THIS ESTATE COULD ACTUALLY TAKE FROM IT  (A-S50's duration bound)")
    print("  %-14s %18s %18s %12s" % ("alpha window", "deployable", "take at 113.7bps", "of the pot"))
    base = pots[123.7]
    for w, x in sorted(S50_DURATION_BOUND.items()):
        take = x * (123.7 - COST_BPS) / 1e4
        print("  %-14s %18s %18s %11.1f%%"
              % ("%d min" % w, "$" + fmt(x), "$" + fmt(take), 100.0 * take / base))

    print()
    print("WHAT IT MEANS")
    print("  The pot is not the constraint.  Even a single 1-minute window's deployable")
    print("  size takes a visible share of a whole DAY's pot, and the 60-minute window")
    print("  exceeds it -- which is the arithmetic saying the same thing S50 said: what")
    print("  limits this line is getting on inside the alpha, not the supply of losers.")
    print()
    print("  Two things the accounting forces that the naive version misses:")
    print("  1  the pot is CONCENTRATED -- half of it in %d symbols.  the priced panel" % n50)
    print("     that any strategy could actually trade is 15 symbols with prices in the")
    print("     liquidation window, carrying 11.7%% of the rows.  a 761-symbol feed is not")
    print("     a 761-symbol opportunity.")
    print("  2  §311/§315 measured a CONTINUATION, so the immediate counterparty of the")
    print("     forced order is NOT the winner -- they are holding something that keeps")
    print("     moving against them.  the pot accrues to whoever is positioned for the")
    print("     continuation, which is a different trade from providing the liquidity.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S52_THE_POT", "rows": n, "symbols": nsym, "span_days": days,
         "total_notional": tot, "per_day": per_day, "cost_bps": COST_BPS,
         "pot_per_day": {str(k): v for k, v in pots.items()},
         "symbols_for_half_the_pot": n50, "symbols_for_ninety": n90,
         "top": [{"symbol": s, "notional": v, "share": p} for s, v, p in top],
         "s50_duration_bound": S50_DURATION_BOUND}, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
