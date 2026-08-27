# -*- coding: utf-8 -*-
"""S50 -- three constraints on the forced-flow line, and which one actually binds.

S49 established that capacity was never limited by impact.  It ended by pointing at
S96-S97's answer -- TIMING RISK -- without measuring it.  The corpus has that half too,
and putting the two halves together turns up a disagreement between two of its books.

THE TWO BOOKS DISAGREE ABOUT WHETHER SLOWING DOWN HELPS
-------------------------------------------------------
Kissell SATPM Eq. 4.4 / 7.20:      MI_bps = b1*I* *POV^a4 + (1-b1)*I*
    impact has a component that scales with the participation rate, so trading slower
    reduces it.  This is what makes his "trader's dilemma" an optimisation with an
    interior solution: pay impact or pay timing risk, pick a point.

Bouchaud TQP §12.3.2, third surprising feature, stated in as many words:
    "the time horizon T does not appear explicitly"
    With sigma_T = sigma_d*sqrt(T) and V_T = ADV*T and delta = 1/2, the algebra is
    exact:  I = Y*sigma_d*sqrt(T)*sqrt(Q/(ADV*T)) = Y*sigma_d*sqrt(Q/ADV).  T cancels.
    Impact depends on SIZE, not on SPEED.

If Bouchaud is right, slowing down buys nothing on impact and only adds timing risk, so
the dilemma evaporates and the answer is always "as fast as the book allows".  Logged as
CT-017; this driver reports the consequence under BOTH readings rather than picking one.

THE THREE CONSTRAINTS
---------------------
  1  IMPACT-BOUND      Bouchaud, measured in S49.  I = Y*sigma_d*sqrt(X/ADV).
  2  VARIANCE-BOUND    Kissell Eq. 7.21:
                       TR_bps = sigma_ann*sqrt((1/250)*(1/3)*(X/ADV)*(1-POV)/POV)*1e4
  3  DURATION-BOUND    what actually fits inside the alpha's own window:
                       X_max = ADV * POV * t_window
                       A forced-flow event is a cascade.  If working the order takes
                       longer than the event, the edge is gone before the order is on.
                       Kissell calls the unfilled remainder's loss OPPORTUNITY COST and
                       Hasbrouck (ch.10) notes it is NOT zero-sum -- it is a real loss
                       to the trader that nobody else books as a gain.

The alpha window is NOT known from this estate's records, so it is swept and the answer
is reported across it.  Inventing one would make the whole table decorative.
"""

import io
import json
import math
import sqlite3

DB = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S50_WHICH_CONSTRAINT_V1.json"

Y_COEF = 0.5
DELTA = 0.5
COST_BPS = 10.0
EDGE_BPS = 123.7                    # §315 BUY side; the conservative of the two
MIN_SIGMA_D_BPS = 50.0
WINDOWS_MIN = (1, 5, 15, 60, 240)   # candidate alpha windows, swept not chosen
POVS = (0.05, 0.10, 0.20, 0.33)


def panel():
    """Daily sigma and daily notional per symbol, one pass."""
    c = sqlite3.connect(DB, uri=True)
    rows = c.execute(
        "SELECT symbol, DATE(open_time/1000,'unixepoch') AS d, MAX(open_time), close, "
        "SUM(quote_volume) FROM klines WHERE open_time < %d GROUP BY symbol, d "
        "ORDER BY symbol, d" % CUT).fetchall()
    c.close()
    ser = {}
    for s, d, _t, cl, qv in rows:
        if cl and cl > 0 and qv:
            ser.setdefault(s, []).append((cl, qv))
    out = {}
    for s, v in ser.items():
        if len(v) < 100:
            continue
        rets = [math.log(b[0] / a[0]) for a, b in zip(v, v[1:]) if a[0] > 0]
        m = sum(rets) / len(rets)
        sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
        if sd * 1e4 < MIN_SIGMA_D_BPS:
            continue
        vols = sorted(x[1] for x in v)
        out[s] = {"sigma_d": sd, "adv": vols[len(vols) // 2], "days": len(v)}
    return out


def impact_bps(X, adv, sigma_d):
    """Bouchaud.  Speed does not appear."""
    return 1e4 * Y_COEF * sigma_d * (X / adv) ** DELTA


def timing_risk_bps(X, adv, sigma_d, pov):
    """Kissell Eq. 7.21.  sigma there is ANNUAL and the 1/250 converts it back to daily,
    so it is written here directly in daily terms to avoid a round trip through 250."""
    return 1e4 * sigma_d * math.sqrt((1.0 / 3.0) * (X / adv) * (1 - pov) / pov)


def x_at_impact_budget(adv, sigma_d, budget_bps):
    """Invert impact for the X whose impact equals a stated share of the edge."""
    return adv * (budget_bps / (1e4 * Y_COEF * sigma_d)) ** (1.0 / DELTA)


def main():
    d = panel()
    surplus = EDGE_BPS - COST_BPS
    print("THREE CONSTRAINTS ON THE FORCED-FLOW LINE  (%d symbols, sigma floor %.0f bps)"
          % (len(d), MIN_SIGMA_D_BPS))
    print("  edge %.1f bps  cost %.1f  surplus %.1f" % (EDGE_BPS, COST_BPS, surplus))

    # 1 + 2, at a fixed impact budget of half the surplus
    budget = surplus / 2.0
    print()
    print("1+2  SIZE AT WHICH IMPACT EATS HALF THE SURPLUS (%.1f bps), AND THE TIMING"
          % budget)
    print("     RISK OF WORKING THAT SIZE AT EACH PARTICIPATION RATE")
    print("     %-12s %14s %10s %s"
          % ("symbol", "X impact-bound", "dur@20%", "  TR bps at POV 5/10/20/33%"))
    tot = 0.0
    rows = []
    for s in sorted(d, key=lambda k: -d[k]["adv"])[:6]:
        v = d[s]
        X = x_at_impact_budget(v["adv"], v["sigma_d"], budget)
        trs = [timing_risk_bps(X, v["adv"], v["sigma_d"], p) for p in POVS]
        dur_d = (X / v["adv"]) / 0.20
        print("     %-12s %14s %9.1fm %s"
              % (s, "$" + fmt(X), dur_d * 1440,
                 " ".join("%6.1f" % t for t in trs)))
        rows.append({"symbol": s, "X_impact_bound": X, "dur_min_at_pov20": dur_d * 1440,
                     "TR_bps": dict(zip([str(p) for p in POVS], trs))})
    for s, v in d.items():
        tot += x_at_impact_budget(v["adv"], v["sigma_d"], budget)
    print("     POOLED across all %d symbols: $%s" % (len(d), fmt(tot)))

    # 3
    print()
    print("3    DURATION-BOUND -- what fits INSIDE the alpha's own window")
    print("     X = ADV * POV * t_window.  The window is swept, not chosen: this estate")
    print("     has no record of the forced-flow alpha's half-life.")
    print("     %-12s %10s %s" % ("", "", "  X at POV=20% for a window of"))
    print("     %-12s %10s %s"
          % ("symbol", "ADV $", "".join("%12s" % ("%dm" % w) for w in WINDOWS_MIN)))
    dur = {}
    for s in sorted(d, key=lambda k: -d[k]["adv"])[:6]:
        v = d[s]
        xs = [v["adv"] * 0.20 * (w / 1440.0) for w in WINDOWS_MIN]
        dur[s] = xs
        print("     %-12s %10s %s"
              % (s, fmt(v["adv"]), "".join("%12s" % fmt(x) for x in xs)))
    pooled = [sum(v["adv"] * 0.20 * (w / 1440.0) for v in d.values()) for w in WINDOWS_MIN]
    print("     %-12s %10s %s" % ("POOLED", "", "".join("%12s" % fmt(x) for x in pooled)))

    print()
    print("WHICH BINDS")
    print("     %-10s %16s %16s %10s" % ("window", "duration-bound", "impact-bound", "binds"))
    for w, pl in zip(WINDOWS_MIN, pooled):
        b = "DURATION" if pl < tot else "IMPACT"
        print("     %-10s %16s %16s %10s" % ("%d min" % w, "$" + fmt(pl), "$" + fmt(tot), b))

    print()
    print("AND WHAT THE DISAGREEMENT COSTS  (CT-017)")
    print("  Under BOUCHAUD, impact is speed-independent, so the only reason to slow down")
    print("  is to fit the book -- and slowing down strictly ADDS timing risk and")
    print("  opportunity cost.  Optimal execution is AS FAST AS THE BOOK ALLOWS.")
    print("  Under KISSELL, impact falls with POV^a4, so there is an interior optimum and")
    print("  a slower schedule can be correct.")
    print("  The two prescriptions are OPPOSITE at the same parameters.  Nothing in this")
    print("  estate has ever measured a4 on crypto, so the question is open here, not")
    print("  merely open in the literature.")

    # ---- the invariant.  The TR column above is IDENTICAL for every symbol and that
    # is not a bug: fixing the impact budget fixes sigma*sqrt(X/ADV), which is the only
    # place the symbol enters either formula.  Everything cancels.
    #
    #   I  = 1e4 * Y * sigma * sqrt(X/ADV)
    #   TR = 1e4 *     sigma * sqrt(X/ADV) * sqrt((1/3)(1-POV)/POV)
    #   TR / I = (1/Y) * sqrt( (1-POV) / (3*POV) )
    #
    # Symbol-free, size-free, volatility-free.  It depends on the participation rate
    # and NOTHING else.
    print()
    print("THE INVARIANT UNDERNEATH  --  TR/I = (1/Y)*sqrt((1-POV)/(3*POV))")
    print("  symbol-free, size-free, volatility-free.  only the participation rate.")
    print("  %-10s %12s %14s" % ("POV", "TR / impact", "who is bigger"))
    for pov in (0.02, 0.05, 0.10, 0.20, 0.33, 0.571, 0.75):
        r = (1.0 / Y_COEF) * math.sqrt((1 - pov) / (3.0 * pov))
        print("  %-10s %12.3f %14s" % ("%.1f%%" % (pov * 100), r,
                                       "timing" if r > 1 else "impact"))
    cross = 4.0 / (4.0 + 3.0 * Y_COEF ** 2 / (Y_COEF ** 2))   # placeholder, solved below
    # solve (1/Y)^2 (1-p)/(3p) = 1  ->  p = 1 / (1 + 3*Y^2)
    cross = 1.0 / (1.0 + 3.0 * Y_COEF ** 2)
    print("  CROSSOVER  POV = 1/(1+3Y^2) = %.3f" % cross)
    print()
    print("  So at any participation rate BELOW %.0f%%, the price noise you sit through"
          % (cross * 100))
    print("  costs more than the impact you were slowing down to avoid.  Combined with")
    print("  Bouchaud's T-independence -- slowing down does not reduce impact at all --")
    print("  a slow schedule is wrong twice over.  Under Kissell's reading it is wrong")
    print("  once and possibly not at all.  That is the whole weight of CT-017.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S50_WHICH_CONSTRAINT", "edge_bps": EDGE_BPS, "cost_bps": COST_BPS,
         "impact_budget_bps": budget, "n_symbols": len(d),
         "pooled_impact_bound_usd": tot, "windows_min": list(WINDOWS_MIN),
         "pooled_duration_bound_usd": pooled, "top": rows,
         "tr_over_impact_crossover_pov": 1.0/(1.0+3.0*Y_COEF**2),
         "contradiction": "CT-017 Bouchaud T-independent vs Kissell POV^a4"}, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
