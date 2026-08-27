# -*- coding: utf-8 -*-
"""S49 -- capacity from the square-root law, not from a top-of-book snapshot.

WHAT IS BEING REPLACED
----------------------
The forced-flow line is the one mechanism in this estate that clears its cost (§311/§315:
beta-neutral continuation, SELL -136.9 t=-10.65, BUY +123.7 t=+6.62, direction predicted
first, 84-85% symbol agreement).  Its capacity ceiling was then set at "~$9k deployable,
~$75/day" -- and S96-S97 recorded the defect in that number: **it was a TOP-OF-BOOK
SNAPSHOT**.  A snapshot of the best quote says what one order sees at one instant; it says
nothing about what a metaorder worked over minutes can absorb, because the book refills.

WHAT THE CORPUS PUTS IN ITS PLACE
---------------------------------
Bouchaud TQP §12.3, the square-root law of metaorder impact:

    I_peak(Q,T)  ~=  Y * sigma_T * (Q / V_T)^delta          (Q << V_T)

    Y     numerical coefficient of order 1  (Y ~= 0.5 for US stocks)
    delta 0.4-0.7; "delta ~= 0.5 for Bitcoin" is stated explicitly
    sigma_T, V_T   contemporaneous volatility and volume over the SAME horizon T

TQP calls it "well established empirically" across equities, futures, FX, options AND
Bitcoin; pre- and post-HFT; small- and large-tick; every strategy style.  Figure 12.2
covers Q/V_T from 1e-5 to a few per cent -- that range is the law's validity domain and
this driver refuses to extrapolate outside it.

THE CALCULATION
---------------
A metaorder is worth doing while its edge exceeds what it costs to put on:

    edge_bps  >=  1e4 * Y * sigma_T * (Q/V_T)^delta  +  cost_bps

    Q_max = V_T * [ (edge_bps - cost_bps) / (1e4 * Y * sigma_T) ] ^ (1/delta)

At delta = 1/2 the exponent is 2: **capacity scales as the SQUARE of the surplus edge.**
Halving the surplus quarters the capacity.  That is the sharpest thing the law says and
it is why a 10 bps fee uncertainty matters more than it looks.

WHAT THIS DOES NOT DO
---------------------
It does not re-measure the forced-flow edge.  That number is §311/§315's, it came off a
burned sample, and re-deriving it here would consume nothing and prove nothing.  It is
taken as an input and the answer is reported ACROSS a range of it, so the reader can see
how much of the conclusion rests on it.
"""

import io
import json
import math
import sqlite3

DB = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000                      # lawful cutoff 2026-08-21
OUT = "reports/research/h2_response_shape_v1/S49_CAPACITY_LAW_V1.json"

Y_COEF = 0.5                             # TQP §12.3: "Y ~= 0.5 for US stocks"
DELTA = 0.5                              # TQP §12.3: "delta ~= 0.5 for Bitcoin"
COST_BPS = 10.0                          # BINANCE_BASE round trip; OD-033 open
VALID_LO, VALID_HI = 1e-5, 0.03
MIN_SIGMA_BPS = 50.0                     # daily; excludes pegged instruments (evaluator U4)          # Figure 12.2's plotted domain

# §311/§315's measured beta-neutral forced-flow continuation, both sides.
EDGES = ((123.7, "BUY  §315"), (136.9, "SELL §311"))


def per_symbol(hours):
    """Contemporaneous sigma_T and V_T at horizon T, per symbol.

    Both must be measured on the SAME horizon -- the law is dimensionally a ratio of a
    move to a volume over one window, and mixing a daily sigma with an hourly volume
    silently rescales the answer by sqrt(24).
    """
    ms = hours * 3600000
    c = sqlite3.connect(DB, uri=True)
    rows = c.execute(
        "SELECT symbol, open_time/%d AS b, MAX(open_time), close, MIN(open_time), "
        "SUM(quote_volume) FROM klines WHERE open_time < %d GROUP BY symbol, b"
        % (ms, CUT)).fetchall()
    c.close()
    by = {}
    for s, b, _mx, cl, _mn, qv in rows:
        if cl and cl > 0 and qv:
            by.setdefault(s, []).append((int(b), float(cl), float(qv)))
    out = {}
    for s, v in by.items():
        v.sort()
        if len(v) < 100:
            continue
        rets, vols = [], []
        for (b0, c0, _q0), (b1, c1, q1) in zip(v, v[1:]):
            if b1 == b0 + 1:
                rets.append(math.log(c1 / c0))
                vols.append(q1)
        if len(rets) < 100:
            continue
        m = sum(rets) / len(rets)
        sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
        vols.sort()
        out[s] = {"sigma": sd, "V": vols[len(vols) // 2], "n": len(rets)}
    return out


def q_max(edge_bps, sigma, V):
    """Invert the law.  Returns (Q, Q/V, in_validity_domain)."""
    surplus = edge_bps - COST_BPS
    if surplus <= 0 or sigma <= 0:
        return 0.0, 0.0, False
    ratio = (surplus / (1e4 * Y_COEF * sigma)) ** (1.0 / DELTA)
    return ratio * V, ratio, VALID_LO <= ratio <= VALID_HI


def main():
    res = {"study": "S49_CAPACITY_LAW", "Y": Y_COEF, "delta": DELTA,
           "cost_bps": COST_BPS, "validity_domain": [VALID_LO, VALID_HI],
           "replaces": "the ~$9k top-of-book capacity snapshot (S96-S97 flagged it)",
           "horizons": {}}

    print("CAPACITY FROM THE SQUARE-ROOT LAW  (TQP 12.3)")
    print("  I_peak = Y*sigma_T*(Q/V_T)^delta   Y=%.2f  delta=%.2f  cost=%.1f bps"
          % (Y_COEF, DELTA, COST_BPS))
    print()
    print("  FIRST, THE QUESTION I ASKED WRONG.  Solving for the Q at which impact eats")
    print("  the edge puts Q/V at 32-48x the window volume -- three orders of magnitude")
    print("  outside Figure 12.2's validated domain (1e-5 to a few per cent).  The law")
    print("  cannot be evaluated there and extrapolating it would be inventing a number.")
    print("  What it CAN answer is the bounded question: at the TOP of its own validated")
    print("  domain, how big is the impact, and how much notional is that?")

    for hours, hl in ((1, "1h"), (24, "1d")):
        d = per_symbol(hours)
        d = {k: v for k, v in d.items()
             if v["sigma"] * 1e4 >= MIN_SIGMA_BPS * math.sqrt(hours / 24.0)}
        if not d:
            continue
        print()
        print("HORIZON %s -- %d symbols (pegged instruments excluded, sigma floor)" % (hl, len(d)))
        print("  impact at the domain boundary Q/V = %.0f%%:  I = Y*sigma*sqrt(%.2f)"
              % (VALID_HI * 100, VALID_HI))
        tot = 0.0
        per = []
        for s_, v in d.items():
            imp = 1e4 * Y_COEF * v["sigma"] * (VALID_HI ** DELTA)
            per.append((s_, VALID_HI * v["V"], imp))
            tot += VALID_HI * v["V"]
        per.sort(key=lambda x: -x[1])
        imps = sorted(x[2] for x in per)
        print("  DEPLOYABLE per %s window, pooled over %d symbols   $%s"
              % (hl, len(per), fmt(tot)))
        print("  impact there: median %.2f bps, p90 %.2f bps, max %.2f bps"
              % (imps[len(imps) // 2], imps[int(0.9 * len(imps))], imps[-1]))
        print("  %-12s %18s %12s" % ("largest", "deployable $", "impact bps"))
        for s_, Q, imp in per[:5]:
            print("  %-12s %18s %12.2f" % (s_, fmt(Q), imp))
        surplus = EDGES[0][0] - COST_BPS
        print("  against a %.1f bps surplus edge, median impact is %.1f%% of it"
              % (surplus, 100.0 * imps[len(imps) // 2] / surplus))
        res["horizons"][hl] = {"n_symbols": len(per), "deployable_usd": tot,
                               "impact_median_bps": imps[len(imps) // 2],
                               "impact_p90_bps": imps[int(0.9 * len(imps))],
                               "boundary_q_over_v": VALID_HI,
                               "top": [{"symbol": a, "Q": b, "impact_bps": c}
                                       for a, b, c in per[:10]]}

    print()
    print("WHAT THIS REPLACES")
    dd = res["horizons"].get("1h")
    if dd:
        print("  the retired figure: ~$9,000 deployable, ~$75/day, from a TOP-OF-BOOK")
        print("  SNAPSHOT of displayed depth at the best quote (S96-S97 flagged it).")
        print("  inside the square-root law's own validated domain, on the SAME estate:")
        print("    $%s per 1h window across %d symbols, median impact %.2f bps."
              % (fmt(dd["deployable_usd"]), dd["n_symbols"], dd["impact_median_bps"]))
        print("  ratio to the retired figure: %.0fx" % (dd["deployable_usd"] / 9000.0))
        print()
        print("  These measure DIFFERENT THINGS and that is the finding.  Displayed depth")
        print("  at an instant is not absorbable volume over a window: the book refills,")
        print("  and TQP 12.3 says the price a metaorder pays scales with sqrt(Q/V), not")
        print("  with what happens to be showing.  A snapshot cannot bound a metaorder.")
        print()
        print("  IT DOES NOT SAY THE LINE IS TRADEABLE.  It says capacity was never the")
        print("  binding constraint, so the constraint is somewhere else -- and S96-S97")
        print("  already named it: TIMING RISK.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(res, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
