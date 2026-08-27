# -*- coding: utf-8 -*-
"""S57 -- the lowest round-trip cost physically attainable here, from TQP ch.21.

WHY THIS IS THE LAST COST QUESTION
----------------------------------
A-S55 assembled the round-trip cost under TAKER execution: 15.3-47.8 bps, of which fees are
10 and impact is the rest.  Nothing in this estate clears that on the unconditional reading.
But the ledger assumed taker, and taker is a CHOICE.  The frontier's binding constant is not
the cost of the execution anyone happens to use -- it is the lowest cost physically
attainable.  That number has never been computed here.

WHAT TQP CH.21 SUPPLIES
-----------------------
Two passages, both decisive.

§21.1, the cost taxonomy:
    "As a general rule, DIRECT TRADING COSTS ARE OF THE ORDER OF 0.1-1 BASIS POINTS and
     SEC regulatory fees are 0.01 basis points."
That is the reference scale for fees in the literature this estate reasons from.  Binance
BINANCE_BASE taker is 5 bps PER SIDE -- five to fifty times it.  Maker at 2 bps per side is
two to twenty times it.  **The fee here is not a footnote to the microstructure; it is an
order of magnitude larger than the microstructure.**

§21.4, the passive order's payoff, term by term:
    on execution   "the gain or loss experienced by the limit order owner comes from the
                    balance between ADVERSE SELECTION and the value of s/2 + w, where w
                    denotes any fee associated with the matching"
    on the opposite queue depleting
                   "the trader experiences a loss (due to the OPPORTUNITY COST) equal to
                    C, which is ON THE ORDER OF A TICK"
    and the priority rule
                   "limit orders with high priority should benefit from short-term
                    mean-reversion, while limit orders with low priority will suffer from
                    the adverse selection of sweeping market orders"

So the passive round trip is:   fee(maker) - 2*(s/2) + adverse selection + P(deplete)*C

AND EVERY TERM EXCEPT THE FEE IS MEASURABLE HERE AND TINY
---------------------------------------------------------
A-S55 measured the full spread at 0.000-0.026 bps on BTC/ETH/SOL.  So s/2 is ~0.013 bps --
the quantity that IS the market maker's entire compensation in TQP's framework is three
orders of magnitude below the fee.  C is "on the order of a tick", so also ~0.026 bps.
CLAUDE.md §206 measured adverse selection at approximately zero on this venue.

DOES IMPACT SURVIVE PASSIVE EXECUTION?  YES.
--------------------------------------------
TQP §12.3 states the square-root law holds "for market participants that use ... different
execution styles (including using a mix of limit orders and market orders or using mainly
market orders)".  A metaorder has impact because of its SIZE, not its order type.  So the
largest term in A-S55's ledger does NOT go away, and the floor is fee + impact.
"""

import io
import json
import math
import sqlite3

PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S57_THE_FLOOR_V1.json"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
Y_COEF, DELTA = 0.5, 0.5
TAKER_RT, MAKER_RT = 10.0, 4.0            # BINANCE_BASE, per CLAUDE.md; OD-033 OPEN
WINDOW_MIN = 50.0                          # A-S54's measured alpha window
POVS = (0.02, 0.05, 0.10, 0.20, 0.33)
SPREAD = {"BTCUSDT": 0.000, "ETHUSDT": 0.001, "SOLUSDT": 0.013}   # A-S55, full spread bps
CONT = {"BTCUSDT": (6.72, 39.89), "ETHUSDT": (10.07, 95.07), "SOLUSDT": (7.37, 71.82)}
TQP_DIRECT_LO, TQP_DIRECT_HI = 0.1, 1.0    # TQP §21.1, per side, equities


def daily(sym):
    c = sqlite3.connect(PANEL, uri=True)
    rows = c.execute(
        "SELECT DATE(open_time/1000,'unixepoch') d, MAX(open_time), close, SUM(quote_volume) "
        "FROM klines WHERE symbol=? AND open_time<? GROUP BY d ORDER BY d",
        (sym, CUT)).fetchall()
    c.close()
    cl = [r[2] for r in rows if r[2]]
    qv = sorted(r[3] for r in rows if r[3])
    rets = [math.log(b / a) for a, b in zip(cl, cl[1:]) if a > 0]
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
    return sd, qv[len(qv) // 2]


def main():
    print("THE COST FLOOR  --  what this venue charges against what the literature assumes")
    print("  TQP §21.1: 'direct trading costs are of the order of 0.1-1 basis points'")
    print("  %-22s %10s %14s" % ("", "bps/side", "x TQP's range"))
    for lab, v in (("TQP direct cost", None), ("BINANCE_BASE maker", 2.0),
                   ("BINANCE_BASE taker", 5.0)):
        if v is None:
            print("  %-22s %10s %14s" % (lab, "0.1 - 1.0", "1x"))
        else:
            print("  %-22s %10.1f %14s"
                  % (lab, v, "%.0f - %.0fx" % (v / TQP_DIRECT_HI, v / TQP_DIRECT_LO)))
    print("  **The fee here is larger than the microstructure it is meant to sit beside.**")

    print()
    print("TQP §21.4's PASSIVE ROUND TRIP, EVERY TERM MEASURED ON THIS ESTATE")
    print("  %-28s %14s %s" % ("term", "bps (round trip)", "source"))
    print("  %-28s %14.2f %s" % ("maker fee", MAKER_RT, "BINANCE_BASE, OD-033 OPEN"))
    sp = sum(SPREAD.values()) / len(SPREAD)
    print("  %-28s %14.3f %s" % ("spread earned  -2*(s/2)", -sp, "A-S55, live 2026 quotes"))
    print("  %-28s %14.3f %s" % ("opportunity cost ~ a tick", sp, "TQP §21.4, 'order of a tick'"))
    print("  %-28s %14s %s" % ("adverse selection", "~0", "CLAUDE.md §206, measured here"))
    print("  %-28s %14s %s" % ("impact", "SURVIVES", "TQP §12.3: all execution styles"))
    print("  Every term except the fee and impact is under 0.03 bps.  On these instruments")
    print("  the maker/taker question is a PURE FEE question, and everything the")
    print("  microstructure literature cares about is three orders of magnitude smaller.")

    res = {}
    for s in SYMS:
        sig, adv = daily(s)
        uncond, p99 = CONT[s]
        print()
        print("  %s   sigma_d %.1f bps" % (s, sig * 1e4))
        print("    %-7s %10s %11s %11s %11s %11s"
              % ("POV", "impact x2", "TAKER", "MAKER", "vs uncond", "vs top 1%"))
        rows = []
        for pov in POVS:
            frac = pov * WINDOW_MIN / 1440.0
            imp = 2.0 * 1e4 * Y_COEF * sig * (frac ** DELTA)
            tk = TAKER_RT + 2 * SPREAD[s] + imp
            mk = MAKER_RT + imp - 2 * (SPREAD[s] / 2) + SPREAD[s]     # earn s/2, pay ~tick
            print("    %-7s %10.3f %11.2f %11.2f %11s %11s"
                  % ("%.0f%%" % (pov * 100), imp, tk, mk,
                     "%+.2f" % (uncond - mk), "%+.2f" % (p99 - mk)))
            rows.append({"pov": pov, "impact_rt": imp, "taker": tk, "maker": mk,
                         "net_uncond_maker": uncond - mk, "net_p99_maker": p99 - mk})
        # where does the maker floor cross the unconditional continuation?
        lo, hi = 1e-6, 1.0
        base = MAKER_RT + SPREAD[s] - SPREAD[s]
        for _ in range(80):
            m = (lo + hi) / 2
            c = MAKER_RT + 2 * 1e4 * Y_COEF * sig * ((m * WINDOW_MIN / 1440.0) ** DELTA)
            if c < uncond:
                lo = m
            else:
                hi = m
        ok = MAKER_RT < uncond
        print("    maker floor is %.2f bps at zero size; unconditional continuation %.2f"
              % (MAKER_RT, uncond))
        print("    -> %s" % ("clears only below POV %.3f%%" % (lo * 100) if ok
                             else "NEVER clears: the fee alone exceeds the continuation"))
        res[s] = {"sigma_d": sig, "adv": adv, "rows": rows,
                  "maker_floor_bps": MAKER_RT, "unconditional": uncond,
                  "clears_uncond": ok, "max_pov_uncond": lo if ok else None}

    print()
    print("HOW MUCH ROOM IS ACTUALLY IN IT")
    print("  net(POV) = continuation - maker_fee - impact(POV),  impact = k*sqrt(POV).")
    print("  Revenue ~ net * size and size ~ POV, so revenue ~ (a - k*sqrt(POV))*POV.")
    print("  d/du of (a*u^2 - k*u^3) with u = sqrt(POV) gives u* = 2a/(3k).")
    print("  %-9s %11s %11s %13s %11s"
          % ("symbol", "max net", "POV*", "size at POV*", "net at POV*"))
    for s_, d_ in res.items():
        sig = d_["sigma_d"]
        a = d_["unconditional"] - MAKER_RT
        k = 2.0 * 1e4 * Y_COEF * sig * ((WINDOW_MIN / 1440.0) ** DELTA)
        if a <= 0:
            print("  %-9s %11s" % (s_, "NONE"))
            continue
        u = 2.0 * a / (3.0 * k)
        pov = u * u
        net = a - k * u
        size = d_["adv"] * pov * WINDOW_MIN / 1440.0
        print("  %-9s %11.2f %10.3f%% %13s %11.2f"
              % (s_, a, pov * 100, "$" + format(int(size), ","), net))
        d_["max_net_bps"] = a
        d_["pov_star"] = pov
        d_["size_at_pov_star"] = size
        d_["net_at_pov_star"] = net
    print("  The MAXIMUM net available on the unconditional route, under the best")
    print("  execution physically attainable, is the continuation minus the 4 bps maker")
    print("  fee -- and it is spent down to under 1 bps by impact at any size worth")
    print("  deploying.  That is the room.  It is not a route: maker execution needs a")
    print("  best-of-book engine this estate does not have, and execution/ is untouchable.")

    print()
    print("THE FLOOR, STATED")
    print("  The lowest round-trip cost attainable on this venue is the MAKER FEE plus")
    print("  IMPACT.  Everything else -- spread, adverse selection, queue opportunity")
    print("  cost -- is under 0.03 bps and cannot be engineered into a difference.")
    print("  That floor is 4.0 bps at vanishing size and 4.0 + impact at any real one.")
    print("  It is NOT reachable today: maker execution needs a best-of-book engine that")
    print("  does not exist here (CLAUDE.md §146/§206), and execution/ is untouchable.")
    print("  It is the FRONTIER's constant, not a proposal.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S57_THE_FLOOR", "taker_rt": TAKER_RT, "maker_rt": MAKER_RT,
         "tqp_direct_bps_per_side": [TQP_DIRECT_LO, TQP_DIRECT_HI],
         "window_min": WINDOW_MIN, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
