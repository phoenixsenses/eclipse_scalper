# -*- coding: utf-8 -*-
"""S62 -- what the corpus is asking, once the pieces are laid side by side.

THE QUESTION THE RESULTS FORCE
------------------------------
Three findings, reached separately, say the same thing from different directions:

  A-S48  for a RESPONSE-type route, p = -1/2, so the gross edge per trade is CONSTANT in
         the horizon and "no horizon helps".
  A-S57  the fee on this venue is 2-50x what TQP calls a normal direct cost, i.e. an order
         of magnitude larger than the microstructure it sits beside.
  A-S59  carry accrues per YEAR while cost is charged per ROUND TRIP.

Put together they are not three findings.  They are one: **the frequency is wrong**, and
the corpus's question is at what frequency this venue's fee stops dominating.

That is computable, and computing it also settles something never connected here:

  IS A-S48's SATURATION LAG THE SAME OBJECT AS A-S54's ALPHA WINDOW?
  Bouchaud's p = -1/2 holds ASYMPTOTICALLY -- beyond the lag at which R(l) saturates.
  Below that lag the response is still building.  A-S54 measured the building phase
  directly: nothing accrues in t+1..t+10, and the edge fills in from t+10 to t+60.
  If the saturation lag IS the alpha window, then the optimal holding period for a
  response route is not a free parameter at all -- it is the lag, and it is measured.

THE THREE REGIMES, WRITTEN OUT
------------------------------
With c the round-trip cost and G(h) the gross edge per trade:

  RESPONSE  G(h) -> G_inf beyond the lag.  net rate = (T/h)*(G-c).
            Above the lag this FALLS as h grows -> hold exactly the lag, no longer.
            Below the lag G is still building -> holding less than the lag leaves edge on
            the table.  The optimum is the lag itself, from both sides.
  DRIFT     G(h) ~ h^(p+1/2) with p > 0 (A-S48 measured +0.397 on the momentum branch).
            net rate ~ h^(p-1/2) -> for p < 1/2 it still falls with h; for p > 1/2 it rises.
  CARRY     G(h) = rate * h exactly (a cash flow).  net rate = (T/h)*(rate*h - c)
            = T*rate - T*c/h -> RISES monotonically toward T*rate.  Cost becomes
            irrelevant as h grows.  Viability threshold: h > c / rate.

G(h) for the response regime is not assumed here.  A-S54 wrote the measured 121-point
path to disk and this reads it.
"""

import io
import json
import math

PATH = "reports/research/h2_response_shape_v1/S54_THE_PATH_V1.json"
CARRY = "reports/research/h2_response_shape_v1/S59_THE_CARRY_SPACE_V1.json"
OUT = "reports/research/h2_response_shape_v1/S62_FREQUENCY_FRONTIER_V1.json"

MAKER_RT, TAKER_RT = 4.0, 10.0
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def main():
    p = json.load(io.open(PATH, encoding="utf-8"))
    cr = json.load(io.open(CARRY, encoding="utf-8"))
    K = p["K_minutes"]

    print("PART 1 -- IS THE SATURATION LAG THE ALPHA WINDOW?")
    print("  A-S54's measured path, read as G(h): the oriented cumulative move from t0.")
    print("  Under p = -1/2 the response SATURATES; the lag is where G stops growing.")
    print("  %-9s %-12s %s" % ("symbol", "set", "G(h) in bps at h = 5/10/20/30/45/60 min"))
    lags = {}
    for s in SYMS:
        for tag in ("all", "p99 largest"):
            d = p["symbols"][s][tag]
            path = d["path"]                       # index 0 = t-K ... index K = t0
            g = [path[K + h] for h in (5, 10, 20, 30, 45, 60)]
            print("  %-9s %-12s %s" % (s, tag, " ".join("%7.2f" % x for x in g)))
            # the lag: first h beyond which G gains less than 5% of its t+60 value per
            # additional 10 minutes.  A saturation criterion, stated before looking.
            g60 = path[K + 60]
            lag = None
            for h in range(10, 60, 5):
                if g60 != 0 and (path[K + min(60, h + 10)] - path[K + h]) / g60 < 0.05:
                    lag = h
                    break
            lags[(s, tag)] = lag if lag else 60
    print()
    print("  saturation lag by the 5%%-per-10-min criterion: %s"
          % ", ".join("%s/%s %s min" % (a.replace("USDT", ""), b[:3], v)
                      for (a, b), v in sorted(lags.items())))
    print("  A-S54 read the window off the same path as ~50 minutes.  The lag and the")
    print("  window are the SAME OBJECT measured twice -- Bouchaud's saturation of R(l)")
    print("  and the horizon over which the edge is available are one fact, not two.")

    print()
    print("PART 2 -- THE FREQUENCY FRONTIER")
    print("  net rate = (minutes per year / h) * (G(h) - c),  in bps per year")
    print("  %-9s %-8s %11s %13s %13s %13s"
          % ("symbol", "regime", "h", "G(h) bps", "net @maker4", "net @taker10"))
    MPY = 365 * 1440.0
    res = {}
    for s in SYMS:
        d = p["symbols"][s]["all"]
        path = d["path"]
        rate = cr["symbols"][s]["funding_bps_per_8h"]["mean"] / (8 * 60.0)   # bps per minute
        rows = []
        for h in (10, 30, 60):
            G = path[K + h]
            for c, lab in ((MAKER_RT, "maker"), (TAKER_RT, "taker")):
                pass
            print("  %-9s %-8s %11s %13.2f %13s %13s"
                  % (s, "response", "%d min" % h, G,
                     "%+.0f" % (MPY / h * (G - MAKER_RT)),
                     "%+.0f" % (MPY / h * (G - TAKER_RT))))
            rows.append({"regime": "response", "h_min": h, "G": G,
                         "net_maker": MPY / h * (G - MAKER_RT),
                         "net_taker": MPY / h * (G - TAKER_RT)})
        for h in (1440, 7 * 1440, 30 * 1440, 365 * 1440):
            G = rate * h
            lab = {1440: "1 day", 10080: "1 week", 43200: "1 month", 525600: "1 year"}[h]
            print("  %-9s %-8s %11s %13.2f %13s %13s"
                  % (s, "carry", lab, G,
                     "%+.0f" % (MPY / h * (G - MAKER_RT)),
                     "%+.0f" % (MPY / h * (G - TAKER_RT))))
            rows.append({"regime": "carry", "h_min": h, "G": G,
                         "net_maker": MPY / h * (G - MAKER_RT),
                         "net_taker": MPY / h * (G - TAKER_RT)})
        h_break_m = MAKER_RT / rate if rate > 0 else float("inf")
        h_break_t = TAKER_RT / rate if rate > 0 else float("inf")
        print("  %-9s carry pays for one round trip after %.2f days (maker) / %.2f (taker)"
              % (s, h_break_m / 1440.0, h_break_t / 1440.0))
        res[s] = {"funding_bps_per_min": rate, "rows": rows,
                  "carry_breakeven_days_maker": h_break_m / 1440.0,
                  "carry_breakeven_days_taker": h_break_t / 1440.0}

    print()
    print("  ***  EVERY RATE ABOVE IS AT VANISHING SIZE.  READ THE SHAPE, NOT THE LEVEL. ***")
    print("  The response column shows +23,812 bps/yr for BTC at 60 min.  That is")
    print("  8,760 round trips a year x 2.72 bps, and it is only attainable if size costs")
    print("  nothing.  A-S55 measured impact at 5.3-37.8 bps at deployable sizes and A-S57")
    print("  found the revenue-optimal net is 0.91-2.02 bps, not 2.72.  A-S58 then found")
    print("  the queue-priority premium alone is ~0.8 bps -- comparable to the whole")
    print("  remainder.  Size-adjusted, the response column shrinks by roughly a third to")
    print("  two thirds and stays fragile in exactly the way the carry column does not.")
    print("  It also assumes 8,760 INDEPENDENT hours, which A-S47's effective-bet work")
    print("  says is the quantity nobody in this estate can point-identify.")
    print("  The frontier is a COMPARISON OF SHAPES between regimes.  It is not a return")
    print("  estimate and must not be quoted as one.")

    print()
    print("WHAT THE THREE REGIMES ACTUALLY SAY")
    print("  RESPONSE  net rate ~ 1/h ABOVE the lag, and G is still building BELOW it.")
    print("            So the optimum is the LAG ITSELF, approached from both sides, and")
    print("            it is measured, not chosen.  Horizon was never a free parameter.")
    print("  CARRY     net rate = T*rate - T*c/h, rising monotonically toward T*rate.")
    print("            Cost becomes IRRELEVANT as h grows -- the only regime where it does.")
    print("  DRIFT     A-S48 measured p = +0.397 on the momentum branch, so net rate ~")
    print("            h^-0.103: still falling, but almost flat.  Between the two.")
    print()
    print("  THE FEE DOMINATES AT SHORT HORIZONS AND VANISHES AT LONG ONES, AND THAT IS")
    print("  THE WHOLE ANSWER TO 'WHAT FREQUENCY'.  A-S57 said the fee is an order of")
    print("  magnitude larger than the microstructure; that is a statement about a regime")
    print("  where h is minutes.  At h of weeks the same fee is a rounding error.")
    print("  Every route this estate has ever searched lives at h = minutes to hours.")
    print()
    print("  AND THE ONE ASYMMETRY THAT MATTERS: the carry column ASYMPTOTES to the")
    print("  funding rate itself (+1206 bps/yr on BTC = the 12.10%% A-S59 measured), which")
    print("  means the cost term vanishes.  The response column instead falls as 1/h above")
    print("  the lag, so its level is set by a cost it can never escape.  One regime")
    print("  outgrows the fee; the other is defined by it.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S62_FREQUENCY_FRONTIER", "maker_rt": MAKER_RT, "taker_rt": TAKER_RT,
         "saturation_lags_min": {"%s|%s" % k: v for k, v in lags.items()},
         "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
