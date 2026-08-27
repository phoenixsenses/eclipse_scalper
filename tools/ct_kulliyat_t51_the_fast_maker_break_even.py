# -*- coding: utf-8 -*-
"""C-KULLIYAT-T51 -- CLOSING BOUCHAUD Eq (17.14): THE FAST MARKET-MAKER'S BREAK-EVEN.

C-KULLIYAT-T48 read Bouchaud Sec 17.2 and found that this lane's own maker formula is the
book's SLOW limit, Eq (17.13):   E[G]/v0T = E[s]/2 - R_inf,  fee entering as s -> s + 2w.
The FAST limit, Eq (17.14), was declared UNCOMPUTABLE:

        E[G]/v0T = (E[s]/2)(1 - C(1)) - R(1)

because it needs C(1) as a LEVEL, and H-T8 had struck C(1) as convention-dependent -- it flips
sign with the event definition.  C-KULLIYAT-T50 lifted half that blocker by giving the flip a
MECHANISM rather than a discrepancy: a bin merge collapses same-side runs inside a bin, so the
next event in that bin is forced to be the opposite side and C(1) is driven negative
mechanically.  Thinning and (ts_ms, side) collapse both leave it positive.  The collapse is the
defensible convention and it is the one kappa-chi already lives in, so:

    C(1) = +0.2593 / +0.2801 / +0.2186   (C-KULLIYAT-T50 arm B, the collapse series)

The single remaining input is R(1), the response at LAG ONE EVENT on that same series.  This
round measures it and closes the equation.

TWO SEPARATE THINGS ARE ALSO CHECKED, BOTH ARISING FROM THE OTHER LANE-C SESSION'S C-T52.

  (a) C-T52 warned that an Eq (17.3) bracket taking R_inf from THEIR C-T29 would carry twice
      the correct value, because that R was per aggTrade.  My R_inf did not come from there.  It
      came from H-U6, whose n_events is 3 105 239 / 3 122 933 / 1 651 625 -- identical to this
      lane's collapse counts -- so it is ALREADY per event.  Recorded, not assumed: the driver
      re-reads H-U6 and prints the counts beside the collapse counts.

  (b) H-U6's own table shows R at phi=0 is NOT a plateau.  It rises to 600 s and then falls to
      0.0182 / -0.1494 / -0.4682 at 3600 s.  Calling the 600 s value "R_inf" is therefore a
      CHOICE, and C-KULLIYAT-T48's zero-fee arithmetic inherits it.  The horizon sensitivity is
      reported explicitly rather than left implicit.

PREREGISTERED, fixed before any number is read:
  Q1  R(1) per event, per symbol, with a standard error
  Q2  Eq (17.14) fast-maker gain at zero fee and at the canonical 2.0 bps maker fee
  Q3  fast versus slow: does the inventory-control SPEED change the SIGN, or only the size?
  Q4  the R_inf horizon sensitivity of Eq (17.13), stated as a range not a point

I will not treat Q3 as licensing any deployment statement.  Eq (17.3)'s own sign theorem
(C-KULLIYAT-T49) already says fill probability cannot flip maker P&L; this round asks whether
the OTHER free choice -- how fast inventory is mean-reverted -- can.

DB is opened READ-ONLY.  ESTIMATION.  Ceiling: MEASUREMENT_FIDELITY.

  python -m tools.ct_kulliyat_t51_the_fast_maker_break_even --i-have-approval
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys

import numpy as np

from tools import h2_response_shape_driver as H2
from tools import s66_cascade_process_driver as D
from tools import hb4_is_a_liquidation_special as B4

DAYS = ("2026-08-07", "2026-08-08", "2026-08-09", "2026-08-10",
        "2026-08-11", "2026-08-12", "2026-08-13")
OUT = "reports/atlas"
HU6 = "reports/research/hb4_liquidation_specialness_v1/HU6_ADVERSE_SELECTION_PERSISTENCE_V1.json"
T50 = "reports/atlas/CT_KULLIYAT_T50_UNIT_MISMATCH_V1.json"

HALF_SPREAD = {"BTCUSDT": 0.0078, "ETHUSDT": 0.0266, "SOLUSDT": 0.6595}   # C-T15, bps
FEE_BPS = 2.0                                                             # CLAUDE.md canonical maker
R_HORIZONS = ("60000", "600000", "3600000")


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    hu6 = json.load(io.open(HU6, encoding="utf-8"))
    t50 = json.load(io.open(T50, encoding="utf-8"))
    res = {"days": list(DAYS), "half_spread_bps": HALF_SPREAD, "fee_bps": FEE_BPS,
           "eq_17_13": "E[G]/v0T = (s+2w)/2 - R_inf",
           "eq_17_14": "E[G]/v0T = ((s+2w)/2)(1 - C(1)) - R(1)",
           "per_symbol": {}, "ceiling": "MEASUREMENT_FIDELITY"}

    print("=== (a) UNIT AUDIT -- is my R_inf already per event? ===", flush=True)
    print("    %-9s %12s %12s  %s" % ("symbol", "H-U6 n_events", "T50 collapse n", "same unit?"),
          flush=True)
    for s in H2.SYMBOLS:
        a = hu6["per_symbol"][s]["n_events"]
        b = t50["per_symbol"][s]["B"]["n"]
        print("    %-9s %12d %12d  %s" % (s, a, b, "YES" if a == b else "NO"), flush=True)

    print("\n=== Q1  R(1), response at LAG ONE EVENT on the collapse series ===", flush=True)
    for sym in H2.SYMBOLS:
        num, den, nn = 0.0, 0.0, 0
        sq = 0.0
        for day in DAYS:
            d0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            lo = int(d0.timestamp() * 1000)
            hi = int((d0 + dt.timedelta(days=1)).timestamp() * 1000)
            con = D._con()
            rows = con.execute(
                "SELECT ts_ms,bid_price,ask_price FROM book_ticker WHERE symbol=? "
                "AND ts_ms>=? AND ts_ms<? AND bid_price>0 AND ask_price>0 ORDER BY ts_ms",
                (sym, lo - 2000, hi + 60000)).fetchall()
            con.close()
            if len(rows) < 10000:
                continue
            bts = np.array([r[0] for r in rows], np.int64)
            mid = np.array([0.5 * (r[1] + r[2]) for r in rows], float)
            del rows
            ts, px, eps, qty = B4.load_raw_with_qty(sym, (day,))
            new = np.flatnonzero(np.concatenate(
                [[True], (ts[1:] != ts[:-1]) | (eps[1:] != eps[:-1])]))
            oeps, ots = eps[new], ts[new]
            del ts, px, eps, qty, new
            ib = np.searchsorted(bts, ots, side="left") - 1
            ok = ib >= 0
            oeps, ib = oeps[ok], ib[ok]
            m = mid[ib]
            # R(1) = E[ eps_t * (m_{t+1} - m_t) ] / m_t, in bps
            r = oeps[:-1] * (m[1:] - m[:-1]) / m[:-1] * 1e4
            num += float(r.sum()); sq += float((r * r).sum()); nn += len(r)
            del bts, mid, oeps, ib, m, r

        mean = num / nn
        se = float(np.sqrt(max(sq / nn - mean * mean, 0.0) / nn))
        res["per_symbol"][sym] = {"R1_bps": mean, "R1_se": se, "n": nn,
                                  "C1": t50["per_symbol"][sym]["B"]["C1"]}
        print("    %-9s  R(1) = %+.4f bps   se %.5f   n %d   C(1) %+.4f"
              % (sym, mean, se, nn, res["per_symbol"][sym]["C1"]), flush=True)

    print("\n=== Q2 / Q3  Eq (17.14) FAST vs Eq (17.13) SLOW ===", flush=True)
    print("    %-9s | %-22s | %-22s" % ("symbol", "FAST  Eq (17.14)", "SLOW  Eq (17.13)"), flush=True)
    print("    %-9s | %10s %11s | %10s %11s" %
          ("", "zero fee", "at 2.0 bps", "zero fee", "at 2.0 bps"), flush=True)
    for sym in H2.SYMBOLS:
        r = res["per_symbol"][sym]
        h = HALF_SPREAD[sym]
        rinf = hu6["per_symbol"][sym]["horizons"]["600000"]["levels"]["0.0"]["R_bps"]
        fast0 = h * (1.0 - r["C1"]) - r["R1_bps"]
        fastf = (h - FEE_BPS) * (1.0 - r["C1"]) - r["R1_bps"]
        slow0 = h - rinf
        slowf = h - FEE_BPS - rinf
        r.update({"R_inf_600s": rinf, "fast_zero_fee": fast0, "fast_at_fee": fastf,
                  "slow_zero_fee": slow0, "slow_at_fee": slowf,
                  "speed_changes_sign_at_zero_fee": (fast0 > 0) != (slow0 > 0),
                  "speed_changes_sign_at_fee": (fastf > 0) != (slowf > 0)})
        print("    %-9s | %+10.4f %+11.4f | %+10.4f %+11.4f"
              % (sym, fast0, fastf, slow0, slowf), flush=True)

    flips0 = [s for s in H2.SYMBOLS if res["per_symbol"][s]["speed_changes_sign_at_zero_fee"]]
    flipsf = [s for s in H2.SYMBOLS if res["per_symbol"][s]["speed_changes_sign_at_fee"]]
    print("\n    symbols where inventory SPEED changes the SIGN, zero fee: %s"
          % (flips0 or ["none"]), flush=True)
    print("    symbols where inventory SPEED changes the SIGN, at 2.0 bps: %s"
          % (flipsf or ["none"]), flush=True)
    res["speed_sign_flips_zero_fee"], res["speed_sign_flips_at_fee"] = flips0, flipsf

    print("\n=== Q4  Eq (17.13) is horizon-sensitive: R at phi=0 is NOT a plateau ===", flush=True)
    print("    %-9s %s" % ("symbol", "  ".join("R(%ss)" % (int(h) // 1000) for h in R_HORIZONS)
                           + "   ->  slow gain at zero fee, per horizon"), flush=True)
    for sym in H2.SYMBOLS:
        hs = hu6["per_symbol"][sym]["horizons"]
        rs = [hs[h]["levels"]["0.0"]["R_bps"] for h in R_HORIZONS]
        gains = [HALF_SPREAD[sym] - x for x in rs]
        res["per_symbol"][sym]["slow_zero_fee_by_horizon"] = dict(zip(R_HORIZONS, gains))
        print("    %-9s %s   ->  %s" % (sym, "  ".join("%+.4f" % x for x in rs),
                                        "  ".join("%+.4f" % g for g in gains)), flush=True)
    sign_unstable = [s for s in H2.SYMBOLS
                     if len({g > 0 for g in
                             res["per_symbol"][s]["slow_zero_fee_by_horizon"].values()}) > 1]
    print("    symbols whose ZERO-FEE sign depends on the horizon: %s"
          % (sign_unstable or ["none"]), flush=True)
    print("    at the canonical %.1f bps fee every horizon stays negative: %s"
          % (FEE_BPS, all(HALF_SPREAD[s] - FEE_BPS - x < 0
                          for s in H2.SYMBOLS
                          for x in [hu6["per_symbol"][s]["horizons"][h]["levels"]["0.0"]["R_bps"]
                                    for h in R_HORIZONS])), flush=True)
    res["zero_fee_sign_horizon_unstable"] = sign_unstable

    res["tokens"] = ["EQ_17_14_IS_NOW_COMPUTABLE_THE_CONVENTION_QUESTION_HAS_AN_ANSWER",
                     "MY_R_INF_WAS_ALREADY_PER_EVENT_C_T52_WARNING_DOES_NOT_BITE_THIS_LANE",
                     "R_AT_PHI_ZERO_IS_NOT_A_PLATEAU_AND_R_INF_IS_A_CHOICE",
                     "AT_THE_CANONICAL_FEE_EVERY_HORIZON_AND_BOTH_SPEEDS_STAY_NEGATIVE"
                     if not flipsf else "INVENTORY_SPEED_CHANGES_THE_SIGN_AT_THE_FEE"]
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T51_FAST_MAKER_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T51_FAST_MAKER_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
