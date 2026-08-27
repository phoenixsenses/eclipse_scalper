# -*- coding: utf-8 -*-
"""S55 -- the complete round-trip cost, assembled from Kissell's own decomposition.

WHAT HAS NEVER EXISTED HERE
---------------------------
Every study in this estate uses COST_BPS = 10.0 -- the BINANCE_BASE taker round trip.  That
is ONE term.  Kissell SATPM ch.3 gives the decomposition:

    IS = (S*Pn - S*Pd) - (sum s_j*Pn - sum s_j*p_j - fees)

and ch.4's cost list has nine components, of which the ones that bite here are: fees,
spread, market impact, timing risk, and opportunity cost.  This estate has now measured
every one of them separately -- A-S49 (impact), A-S50 (timing risk, opportunity cost's
structure), §452 (spread = one tick), OD-033 (fees, open) -- and has never put them in one
place.  Without that, "does the edge clear its cost" has never actually been asked.

THE DISTINCTION THAT MUST NOT BE FUDGED
---------------------------------------
Timing risk is a STANDARD DEVIATION, not an expected cost.  Kissell is explicit that impact
and timing risk are "two conflicting terms" traded off under a risk appetite, not summed.
Adding a standard deviation to a mean would overstate the cost and, worse, would make the
answer depend on a risk aversion nobody has stated.  So:

    EXPECTED COST   = fees + spread + impact          (a mean, comparable to the edge)
    UNCERTAINTY     = timing risk                     (a sd, reported as a band)
    OPPORTUNITY     = zero BY CONSTRUCTION here, because the size is defined as what fits
                      inside the window at the stated participation rate.  That is not a
                      free lunch: it is exactly why the size is duration-bounded (A-S50).

THE SIZE
--------
A-S54 measured the alpha window: the continuation does not accrue in t+1..t+10 (that is the
liquidation's own transient impact reverting) and does accrue from t+10 to t+60.  So the
window is ~50 minutes and the size that fits is

    X = ADV * POV * (50/1440)

swept over POV, because the participation rate is a choice and not a measurement.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S55_THE_LEDGER_V1.json"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
Y_COEF, DELTA = 0.5, 0.5
FEE_RT_BPS = 10.0                 # BINANCE_BASE taker round trip; OD-033 OPEN
WINDOW_MIN = 50.0                 # A-S54's measured alpha window
POVS = (0.02, 0.05, 0.10, 0.20, 0.33)
# A-S54's measured continuations, oriented, over t0..t+60
CONT = {"BTCUSDT": (6.72, 39.89), "ETHUSDT": (10.07, 95.07), "SOLUSDT": (7.37, 71.82)}


def spread_bps(sym, lo, hi, cap=200000):
    """Median half-spread in bps from the precomputed spread_pct, sampled in the window.

    The index idx_bt_symbol_ts serves this; a bare GROUP BY symbol on this table does not
    and once cost 3,399 s."""
    c = sqlite3.connect(LIQ, uri=True)
    v = [r[0] for r in c.execute(
        "SELECT spread_pct FROM book_ticker WHERE symbol=? AND ts_ms>=? AND ts_ms<? "
        "AND spread_pct>0 LIMIT ?", (sym, lo, hi, cap))]
    c.close()
    if len(v) < 1000:
        return None, len(v)
    v.sort()
    # spread_pct is a percentage of mid; full spread in bps = pct * 100
    return v[len(v) // 2] * 100.0, len(v)


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
    c = sqlite3.connect(LIQ, uri=True)
    lo, hi = c.execute("SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE ts_ms<?",
                       (CUT,)).fetchone()
    c.close()

    print("THE COMPLETE ROUND-TRIP COST  (Kissell SATPM ch.3-4 decomposition)")
    print("  EXPECTED = fees + spread + impact      (a mean; comparable to the edge)")
    print("  UNCERTAINTY = timing risk              (a sd; a band, NOT added to the mean)")
    print("  OPPORTUNITY = 0 by construction        (size = what fits in the window)")
    print("  size X = ADV * POV * (%.0f/1440)   <- A-S54's measured alpha window" % WINDOW_MIN)
    print("  fees %.1f bps round trip (BINANCE_BASE, OD-033 OPEN)" % FEE_RT_BPS)

    res = {}
    for s in SYMS:
        sp, n = spread_bps(s, lo, hi)
        sig, adv = daily(s)
        if sp is None:
            print("\n  %s: spread unavailable (%d quotes)" % (s, n))
            continue
        uncond, p99 = CONT[s]
        print()
        print("  %s   sigma_d %.1f bps   ADV $%s   full spread %.3f bps (%s quotes)"
              % (s, sig * 1e4, fmt(adv), sp, format(n, ",")))
        print("    %-7s %13s %10s %9s %9s %11s %11s %11s"
              % ("POV", "size $", "impact x2", "spread", "fees", "EXPECTED", "timing sd", "vs uncond"))
        rows = []
        for pov in POVS:
            frac = pov * WINDOW_MIN / 1440.0          # = X/ADV
            X = frac * adv
            imp = 1e4 * Y_COEF * sig * (frac ** DELTA)
            imp_rt = 2.0 * imp                        # in and out
            spread_rt = 2.0 * sp                      # cross on both legs
            exp = FEE_RT_BPS + spread_rt + imp_rt
            tr = 1e4 * sig * math.sqrt((1.0 / 3.0) * frac * (1 - pov) / pov)
            print("    %-7s %13s %10.3f %9.3f %9.1f %11.2f %11.2f %11s"
                  % ("%.0f%%" % (pov * 100), "$" + fmt(X), imp_rt, spread_rt, FEE_RT_BPS,
                     exp, tr, "%+.2f" % (uncond - exp)))
            rows.append({"pov": pov, "X": X, "impact_rt_bps": imp_rt,
                         "spread_rt_bps": spread_rt, "fees_bps": FEE_RT_BPS,
                         "expected_bps": exp, "timing_sd_bps": tr,
                         "net_vs_unconditional": uncond - exp,
                         "net_vs_p99": p99 - exp})
        print("    measured continuation t0->t+60:  unconditional %.2f bps   top 1%% %.2f bps"
              % (uncond, p99))
        res[s] = {"sigma_d": sig, "adv": adv, "spread_bps": sp, "quotes": n,
                  "continuation": {"unconditional": uncond, "p99": p99}, "rows": rows}

    print()
    print("WHAT THE LEDGER SAYS")
    print("  (These four replace the four I wrote into this file BEFORE running it.  Two")
    print("   of those were wrong, and they were wrong in the flattering direction.)")
    print("  1  SPREAD IS GENUINELY NEGLIGIBLE -- 0.000 to 0.026 bps, and it enters twice.")
    print("     §452 said one tick; this confirms it on live 2026 quotes, 200k per symbol.")
    print("  2  IMPACT IS NOT NEGLIGIBLE.  I wrote that it was.  It is 5.3 bps at POV 2%")
    print("     and 37.8 at POV 33% -- comparable to the fee at the smallest rate and")
    print("     nearly four times it at the largest.  A-S49 said capacity never BINDS,")
    print("     meaning the size can be got on.  It still COSTS.  Those are not the same")
    print("     statement and I collapsed them.")
    print("  3  So the expected cost is NOT 'fees to within a rounding error'.  It is")
    print("     15.3 to 47.8 bps, of which fees are 10 and impact is the rest.")
    print("  4  AND NOTHING CLEARS.  Against the unconditional continuation of 6.7-10.1")
    print("     bps every cell is negative, from -8.6 to -37.8.  There is no participation")
    print("     rate at which the unconditional forced-flow route pays for itself.")
    print("  5  TIMING RISK dwarfs the mean at every rate below 57%, exactly as A-S50's")
    print("     invariant requires.  It is not added to the mean, but it is why even a")
    print("     positive expectation here would be hard to hold.")
    print()
    print("  THE ONE CELL THAT WOULD CLEAR, AND WHY IT IS NOT A RESULT")
    for s_, d_ in res.items():
        r0 = d_["rows"][0]
        print("    %-9s top 1%% continuation %6.2f vs expected %5.2f at POV 2%%  ->  %+6.2f"
              % (s_, d_["continuation"]["p99"], r0["expected_bps"], r0["net_vs_p99"]))
    print("    Size IS observable at t=0, so conditioning on it is implementable.  But the")
    print("    p99 cut was MINE, chosen after seeing the path, on a burned sample.  That is")
    print("    the error CLAUDE.md §200 closes, and no amount of it being a large number")
    print("    changes that.  It is a CANDIDATE for a rule-level preregistration on fresh")
    print("    data -- which is what LANE_A_PREREG_V1 exists to be the template for -- and")
    print("    it is nothing else until then.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S55_THE_LEDGER", "fee_rt_bps": FEE_RT_BPS, "window_min": WINDOW_MIN,
         "Y": Y_COEF, "delta": DELTA, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
