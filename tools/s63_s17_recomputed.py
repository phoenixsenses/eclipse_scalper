# -*- coding: utf-8 -*-
"""S63 -- S17's Sharpe frontier, recomputed with the two things it could not have had.

WHAT S17 SAID AND WHAT IT LEFT OPEN
-----------------------------------
S17 built the Sharpe frontier from LdP MLAM §8 and concluded:

    "At the capture this estate has demonstrated, the best annualised Sharpe available
     anywhere is 0.134 -- positive, and untradeable."
    SHARPE_OPTIMAL_HORIZON_IS_ABOUT_3_DAYS
    INVESTABLE_BAR_IS_10_6_PERCENT_AT_1_DAY  ->  3.2x the demonstrated capture
    GAP_NARROWED_FROM_10X_TO_3_2X
    FEE_TIER_WORTH_3_3X_ON_SHARPE

and closed with a caveat it could not then resolve:

    "Costs are fee only.  S7's adverse selection reduces every Sharpe above."

TWO THINGS HAVE CHANGED SINCE, BOTH MEASURED
--------------------------------------------
1  THE CAPTURE.  S17's "demonstrated 3.35%" was traced in A-S56/§475 to
   `SYSTEM_STATE` L39494: it is the MEDIAN |rho_1| of HEDGED HOURLY PAIR SPREADS -- two
   legs, 20 bps of cost -- and A-S14's own verdict on it was 8.8x short.  The single-leg
   figure in the same table is 1-2%, and the best dark-family cell ever measured here is
   2.09% (A-S43).  S17's headline column is computed in the wrong regime.

2  THE COST.  "Fee only" is no longer necessary.  A-S55 assembled the full round trip
   (fees + spread + impact) at 15.3-47.8 bps taker; A-S57 put the maker floor at 4.0 +
   impact and showed spread is under 0.03 bps here; A-S58 measured adverse selection by
   queue position, -0.09 at the front of the queue to +0.79 when swept.

So the frontier can be recomputed on the corrected capture AND with the caveat closed.
That is not a new hypothesis -- it is the same algebra with measured inputs replacing
assumed ones.

THE ALGEBRA, UNCHANGED FROM S17
-------------------------------
    Sharpe per trade  = f*k - c/(sigma*sqrt(h))
    annualised Sharpe = that * sqrt(365/h)
    per-trade ceiling = f*k              (no horizon, no cost: capture alone sets it)
"""

import io
import json
import math
import sqlite3

PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S63_S17_RECOMPUTED_V1.json"

K_MEAN_ABS = 0.6966                      # A-S39/§467, measured
HORIZONS_D = (60 / 1440.0, 240 / 1440.0, 1.0, 3.0, 10.0, 30.0)
HLAB = ("60m", "240m", "1d", "3d", "10d", "30d")

# capture, in the regimes the estate has actually measured
CAPTURES = ((0.0113, "1.13% single leg low"),
            (0.0209, "2.09% best cell A-S43"),
            (0.0335, "3.35% S17 headline (WRONG REGIME, A-S56)"))

# cost scenarios, all round trip, in bps
COSTS = ((10.0, "S17: fee only, taker"),
         (4.0, "maker fee alone (A-S57 floor at zero size)"),
         (4.0 + 5.31 + 0.79, "maker + impact@POV2% + swept AS (A-S55/57/58)"),
         (10.0 + 5.31 + 0.79, "taker + impact@POV2% + swept AS"))


def median_sigma():
    """Daily sigma of the median symbol on the same panel S17 used, cutoff-bounded."""
    c = sqlite3.connect(PANEL, uri=True)
    rows = c.execute(
        "SELECT symbol, DATE(open_time/1000,'unixepoch') d, MAX(open_time), close "
        "FROM klines WHERE open_time<? GROUP BY symbol, d ORDER BY symbol, d",
        (CUT,)).fetchall()
    c.close()
    ser = {}
    for s, _d, _t, cl in rows:
        if cl and cl > 0:
            ser.setdefault(s, []).append(cl)
    sig = []
    for s, cl in ser.items():
        if len(cl) < 100:
            continue
        r = [math.log(b / a) for a, b in zip(cl, cl[1:]) if a > 0]
        m = sum(r) / len(r)
        sd = math.sqrt(sum((x - m) ** 2 for x in r) / (len(r) - 1))
        if sd * 1e4 >= 50.0:
            sig.append(sd)
    sig.sort()
    return sig[len(sig) // 2], len(sig)


def ann_sharpe(f, c_bps, sigma, h_days):
    per = f * K_MEAN_ABS - (c_bps / 1e4) / (sigma * math.sqrt(h_days))
    return per * math.sqrt(365.0 / h_days), per


def main():
    sigma, n = median_sigma()
    print("S17's FRONTIER, RECOMPUTED  --  median symbol sigma_d %.1f bps, %d symbols"
          % (sigma * 1e4, n))
    print("  Sharpe per trade = f*k - c/(sigma*sqrt(h))   ceiling = f*k, no horizon no cost")
    print("  k = %.4f (measured, A-S39)" % K_MEAN_ABS)
    print()
    print("  PER-TRADE CEILINGS")
    for f, lab in CAPTURES:
        print("    %-42s f*k = %.4f" % (lab, f * K_MEAN_ABS))

    res = {"sigma_d": sigma, "n_symbols": n, "k": K_MEAN_ABS, "grid": {}}
    for c_bps, clab in COSTS:
        print()
        print("  COST: %s  (%.2f bps round trip)" % (clab, c_bps))
        print("    %-30s %s" % ("capture", "".join("%9s" % h for h in HLAB)))
        for f, flab in CAPTURES:
            row = [ann_sharpe(f, c_bps, sigma, h)[0] for h in HORIZONS_D]
            best = max(row)
            bi = row.index(best)
            print("    %-30s %s   best %+.3f @ %s"
                  % (flab.split(" ")[0], "".join("%9.3f" % x for x in row), best, HLAB[bi]))
            res["grid"]["%s|%s" % (clab, flab)] = {
                "cost_bps": c_bps, "capture": f,
                "annualised": dict(zip(HLAB, row)), "best": best, "best_h": HLAB[bi]}

    print()
    print("WHAT CHANGED")
    def g(c, f):
        return next(v for k, v in res["grid"].items()
                    if k.startswith(c) and k.endswith(f))
    rows = [("S17 as published", "S17: fee only", "3.35% S17 headline (WRONG REGIME, A-S56)"),
            ("corrected capture only", "S17: fee only", "2.09% best cell A-S43"),
            ("+ the caveat closed, maker", "maker + impact", "2.09% best cell A-S43"),
            ("+ the caveat closed, taker", "taker + impact", "2.09% best cell A-S43")]
    print("  %-28s %10s %10s" % ("", "best SR", "at h"))
    for lab, c, f in rows:
        v = g(c, f)
        print("  %-28s %10.3f %10s" % (lab, v["best"], v["best_h"]))
    print()
    print("  1  THE CAPTURE WAS THE LOAD-BEARING ERROR, NOT THE COST.  Correcting f from")
    print("     3.35%% to 2.09%% takes the best Sharpe from 0.137 to 0.052 -- a 62%% cut.")
    print("     Closing S17's own cost caveat on top of that costs 0.001 more.")
    print("  2  AND THE REASON IS WORTH SEEING: the full cost under BEST execution")
    print("     (maker 4.0 + impact 5.31 + swept adverse selection 0.79 = 10.10 bps) is")
    print("     almost exactly S17's fee-only TAKER assumption of 10.00.  S17 guessed the")
    print("     right total for the wrong reason -- it charged one big term instead of")
    print("     three medium ones.")
    print()
    print("  AND ONE THING S17 CLAIMED THAT DOES NOT SURVIVE")
    print("  S17: SHARPE_OPTIMAL_HORIZON_IS_ABOUT_3_DAYS.  It is not stable.  It moves to")
    print("  10 days at the corrected capture and to 30 days at the full taker cost.")
    print("  (I wrote into this driver, before running it, that the optimum would survive")
    print("   every correction.  It does not.  Third time in this lane that a conclusion")
    print("   written before the measurement failed on contact with it.)")
    print("  The movement is not noise -- §460's closed form gives it exactly:")
    print("     h* = [2c / (k*f*sigma)]^2   =>   h* scales as (c/f)^2")
    for lab, c, f in rows[1:]:
        v = g(c, f)
        a0 = g("S17: fee only", "3.35% S17 headline (WRONG REGIME, A-S56)")
        r = (v["cost_bps"] / v["capture"]) ** 2 / ((a0["cost_bps"] / a0["capture"]) ** 2)
        print("     %-26s (c/f)^2 ratio %5.2fx  ->  3d becomes %4.1fd  (measured %s)"
              % (lab, r, 3 * r, v["best_h"]))
    print("  So the LOCATION of the optimum is not a property of the algebra, as I had")
    print("  written -- it is a function of the two inputs that were both wrong.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
