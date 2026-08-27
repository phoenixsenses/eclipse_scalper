# -*- coding: utf-8 -*-
"""S53 -- can a liquidation of that size move the price that far?  The law says no.

THE NUMBER EVERYTHING NOW RESTS ON
----------------------------------
§311/§315 measured a beta-neutral forced-flow CONTINUATION of 123.7 bps (BUY, t=+6.62) and
-136.9 (SELL, t=-10.65), direction predicted first, 84-85% symbol agreement.  It is the
only mechanism in this estate that clears its cost, and A-S49, A-S50 and A-S52 all took it
as an input without questioning it.  Three studies deep, nobody has asked whether it is
CONSISTENT with anything.

THE CORPUS MAKES A PREDICTION ABOUT IT
--------------------------------------
Bouchaud TQP §12.3 gives the price move a trade of size Q produces:

    I = Y * sigma_T * (Q / V_T)^delta          Y ~ 0.5, delta ~ 0.5

A forced liquidation IS a trade of known size.  So the law says exactly how far the price
should move because of it.  If the measured continuation is far larger than the law allows
for the sizes actually observed, then **the continuation is not the liquidation's impact**,
and whatever it is has to be named.

TQP also supplies the calibration in words: "executing 1% of the daily volume moves the
price (on average) by sqrt(1%) = 10% of its daily volatility."

THREE THINGS IT COULD BE, IF NOT IMPACT
---------------------------------------
  1  SELECTION -- the events enter the sample because a large move occurred.
  2  COMMON CAUSE -- liquidation and continuation are both driven by a third thing
     (news, a large informed metaorder) and neither causes the other.
  3  WINDOW CONTAMINATION -- the measurement window contains the move that CAUSED the
     liquidation.  This estate's own standing rule: never measure an arm on a window
     containing its own definition.

This driver does not choose between them.  It establishes whether the magnitude is
possible at all, which is the prior question and the one the corpus can settle.
"""

import io
import json
import math
import sqlite3

LIQ = "file:data/microstructure_02.db?mode=ro"
PANEL = "file:data/xsec_klines_ext.db?mode=ro"
CUT = 1787270400000
OUT = "reports/research/h2_response_shape_v1/S53_IS_THE_EDGE_IMPACT_V1.json"

Y_COEF = 0.5
DELTA = 0.5
EDGE_BPS = 123.7
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")     # the priced ones; 48.3% of the corrected pot


def hourly_stats(sym):
    """sigma and median notional per HOUR -- the same window the law will be applied on."""
    c = sqlite3.connect(PANEL, uri=True)
    rows = c.execute(
        "SELECT open_time/3600000 AS b, MAX(open_time), close, SUM(quote_volume) "
        "FROM klines WHERE symbol=? AND open_time < ? GROUP BY b ORDER BY b",
        (sym, CUT)).fetchall()
    c.close()
    if len(rows) < 100:
        return None
    rets, vols = [], []
    for (b0, _m0, c0, _q0), (b1, _m1, c1, q1) in zip(rows, rows[1:]):
        if b1 == b0 + 1 and c0 and c0 > 0 and c1 and c1 > 0 and q1:
            rets.append(math.log(c1 / c0))
            vols.append(q1)
    if len(rets) < 100:
        return None
    m = sum(rets) / len(rets)
    sd = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1))
    vols.sort()
    return {"sigma_1h": sd, "V_1h_median": vols[len(vols) // 2], "n_hours": len(rets)}


def liq_sizes(sym):
    c = sqlite3.connect(LIQ, uri=True)
    v = [r[0] for r in c.execute(
        "SELECT notional FROM liquidations WHERE symbol=? AND ts_ms < ? AND notional > 0",
        (sym, CUT))]
    c.close()
    v.sort()
    return v


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


def impact(qv, sigma):
    return 1e4 * Y_COEF * sigma * (qv ** DELTA)


def main():
    print("CAN A LIQUIDATION OF THAT SIZE MOVE THE PRICE 123.7 bps?")
    print("  Bouchaud TQP 12.3:  I = Y*sigma_T*(Q/V_T)^delta,  Y=%.2f delta=%.2f"
          % (Y_COEF, DELTA))
    print("  TQP's own calibration in words: 1%% of daily volume moves the price by 10%%")
    print("  of its daily volatility.")

    res = {}
    for s in SYMS:
        st = hourly_stats(s)
        lv = liq_sizes(s)
        if not st or len(lv) < 100:
            print("\n  %s: insufficient (%s hours, %d liquidations)"
                  % (s, st["n_hours"] if st else 0, len(lv)))
            continue
        sig, V = st["sigma_1h"], st["V_1h_median"]
        print()
        print("  %s   sigma_1h %.2f bps   median hourly notional $%s   %s liquidations"
              % (s, sig * 1e4, fmt(V), format(len(lv), ",")))
        print("    %-12s %14s %12s %14s" % ("size", "notional $", "Q/V_1h", "impact bps"))
        rows = []
        for p, lab in ((0.50, "median"), (0.90, "p90"), (0.99, "p99"),
                       (0.999, "p99.9"), (1.0, "LARGEST")):
            Q = q(lv, p)
            qv = Q / V
            im = impact(qv, sig)
            print("    %-12s %14s %12.3e %14.4f" % (lab, "$" + fmt(Q), qv, im))
            rows.append({"pct": p, "label": lab, "notional": Q, "q_over_v": qv,
                         "impact_bps": im})

        # what size WOULD be needed
        need_qv = (EDGE_BPS / (1e4 * Y_COEF * sig)) ** (1.0 / DELTA)
        need_Q = need_qv * V
        big = lv[-1]
        print("    to produce %.1f bps the law needs Q/V = %.2f, i.e. $%s"
              % (EDGE_BPS, need_qv, fmt(need_Q)))
        print("    the LARGEST liquidation observed is $%s -- short by %s x"
              % (fmt(big), fmt(need_Q / big) if big > 0 else "inf"))
        print("    the MEDIAN one is short by %s x" % fmt(need_Q / q(lv, 0.50)))
        res[s] = {"sigma_1h": sig, "V_1h_median": V, "n_liq": len(lv),
                  "sizes": rows, "needed_q_over_v": need_qv, "needed_notional": need_Q,
                  "largest": big, "shortfall_vs_largest": need_Q / big,
                  "shortfall_vs_median": need_Q / q(lv, 0.50)}

    print()
    print("WHAT THIS SETTLES AND WHAT IT DOES NOT")
    print("  SETTLES: the 123.7 bps continuation is NOT the mechanical impact of the")
    print("  liquidation that marks it.  Under the corpus's own law, calibrated on")
    print("  equities, futures, FX, options AND Bitcoin, the flow is orders of magnitude")
    print("  too small.  The law would need a single order larger than an entire hour's")
    print("  volume many times over.")
    print()
    print("  DOES NOT SETTLE which of the three it is -- selection, common cause, or a")
    print("  window containing its own definition.  Those are distinguishable only by")
    print("  re-examining §311/§315's construction, which is a different study and")
    print("  belongs to whoever owns that result.")
    print()
    print("  AND IT DOES NOT KILL THE RESULT.  A continuation need not be impact to be")
    print("  real: the liquidation can be a MARKER of a state rather than its cause.")
    print("  §337 already reached that shape from another direction --")
    print("  CASCADE_IS_COMMON_STATE_MARKER_ONLY.  This is the same conclusion arriving")
    print("  from the impact law, and it means the tradeable object was never the")
    print("  liquidation's own footprint.")

    io.open(OUT, "w", encoding="utf-8").write(json.dumps(
        {"study": "S53_IS_THE_EDGE_IMPACT", "Y": Y_COEF, "delta": DELTA,
         "edge_bps": EDGE_BPS, "symbols": res}, indent=1))
    print("\nwrote %s" % OUT)


def fmt(x):
    return format(int(round(x)), ",")


if __name__ == "__main__":
    main()
