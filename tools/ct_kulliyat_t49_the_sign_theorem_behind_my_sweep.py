# -*- coding: utf-8 -*-
"""C-KULLIYAT-T49 -- SEC 17.1 IS A SIGN THEOREM, AND IT PREDICTS THE SYMBOL PATTERN I MEASURED.

C-KULLIYAT-T48 opened Sec 17.2 (small-tick inventory control) and found it was this lane's own
slow-maker formula.  Sec 17.1 was the last unread section of that chapter and it carries two
things this lane never had.

(1) A SIGN THEOREM.  Eq (17.3) reads

        E[G_T]  ~  T * v0 * E[theta] * ( E[s]/2 + w - R_inf )

    where theta is the EXECUTION INDICATOR -- 1 if the limit order was matched, 0 otherwise.
    E[theta] enters MULTIPLICATIVELY and lies in [0,1].  Therefore FILL PROBABILITY CANNOT FLIP
    THE SIGN OF MAKER P&L; it can only scale it.  This lane spent rounds sweeping queue position
    phi looking for a critical h_c.  Under Eq (17.3)'s assumptions there IS no h_c, because phi
    acts only through theta.

(2) THE REASON h_c EXISTS AT ALL, AND ONLY WHERE IT DOES.  Eq (17.3) assumes theta is
    uncorrelated with spread and price history.  Footnote 3 says exactly when that fails:

      "reasonable for small-tick assets, for which theta ~ 1, but is not justified for
       large-tick assets, for which the execution probability increases either when a large
       market order arrives or when the queue is very short.  This leads to increased adverse
       selection, which we neglect here - but see Section 17.3."

    So h_c is a Sec 17.3 object and exists ONLY where the theta-independence assumption breaks,
    i.e. LARGE TICK.  This lane measured h_c existing only on SOL, the large-tick symbol.  The
    corpus predicts the symbol pattern a priori.

(3) A BRIDGE TO C-T29 THAT WAS NEVER DRAWN.  Sec 17.1 ties the SIGN of R_inf -- hence of maker
    P&L -- to which side of the fine balance the market sits on:

      beta > (1-gamma)/2  ->  sub-diffusive, R -> -inf  ->  "a boon for market-makers"
      beta < (1-gamma)/2  ->  trending,      R -> +inf  ->  "market-making extremely difficult"
      "market-making is easy when prices mean-revert but difficult when prices trend"

    C-T29 measured precisely difference = (kappa-chi) - (1-gamma)/2 = beta - (1-gamma)/2.  This
    driver reads that published artifact and reports which side each symbol falls on, WITH the
    limitation this lane already established: gamma is fit-range dependent, so the direction is
    suggestive and NOT established, and a composite rejection does not say which leg failed.

No DB, no market data, no new estimate.  Published artifacts + corpus.  Ceiling: RECONCILIATION.

  python -m tools.ct_kulliyat_t49_the_sign_theorem_behind_my_sweep --i-have-approval
"""
from __future__ import annotations

import io
import json
import os
import sys

CT29 = "reports/atlas/CT29_FINE_BALANCE_V1.json"
OUT = "reports/atlas"
TICK_REGIME = {"BTCUSDT": "small", "ETHUSDT": "small", "SOLUSDT": "large"}
H_C_MEASURED = {"BTCUSDT": False, "ETHUSDT": False, "SOLUSDT": True}   # C-T16, zero fee only


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    d = json.load(io.open(CT29, encoding="utf-8"))

    print("=== (1) THE SIGN THEOREM, Eq (17.3) ===", flush=True)
    print("    E[G] ~ T * v0 * E[theta] * ( E[s]/2 + w - R_inf )", flush=True)
    print("    E[theta] in [0,1], MULTIPLICATIVE  =>  fill probability SCALES P&L, never flips it.",
          flush=True)
    print("    => no queue position can rescue a negative bracket.  Structural, not empirical.",
          flush=True)

    print("\n=== (2) WHERE THE ASSUMPTION BREAKS, footnote 3 ===", flush=True)
    print("    small-tick: theta ~ 1, queue position nearly irrelevant, adverse selection NEGLECTED",
          flush=True)
    print("    large-tick: theta ~ 0 except at the front, adverse selection -> see Sec 17.3",
          flush=True)
    print("    %-9s %-7s %-20s %s" % ("symbol", "tick", "book predicts h_c?", "lane measured h_c"),
          flush=True)
    agree = 0
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        pred = TICK_REGIME[s] == "large"
        got = H_C_MEASURED[s]
        agree += int(pred == got)
        print("    %-9s %-7s %-20s %-18s %s"
              % (s, TICK_REGIME[s], "yes" if pred else "no",
                 "yes" if got else "no", "MATCH" if pred == got else "MISMATCH"), flush=True)
    print("    agreement %d of 3" % agree, flush=True)

    print("\n=== (3) THE BRIDGE TO C-T29, difference = beta - (1-gamma)/2 ===", flush=True)
    print("    beta > (1-gamma)/2  ->  mean-reverting  ->  BOON for market-makers", flush=True)
    print("    beta < (1-gamma)/2  ->  trending        ->  market-making EXTREMELY DIFFICULT",
          flush=True)
    sides = {}
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        r = d["per_symbol"][s]
        diff, z = r["difference"], r["z"]
        side = "TRENDING_HARD" if diff < 0 else "MEANREVERT_BOON"
        sides[s] = {"difference": diff, "z": z, "side": side,
                    "rejects": r["rejects_composite_at_2sigma"]}
        print("    %-9s diff %+.4f  z %+.2f  -> %-16s  composite %s"
              % (s, diff, z, side,
                 "REJECTED" if r["rejects_composite_at_2sigma"] else "not rejected"), flush=True)
    hard = sorted(s for s in sides if sides[s]["side"] == "TRENDING_HARD")
    rej = sorted(s for s in sides if sides[s]["rejects"])
    print("    both rejecting symbols fall on the SAME side: %s" % (rej == hard), flush=True)

    print("\n=== LIMITATION, declared before the reading ===", flush=True)
    print("    gamma is FIT-RANGE DEPENDENT (this lane's own earlier finding), so the SIGN of",
          flush=True)
    print("    `difference` is SUGGESTIVE, NOT ESTABLISHED.  And C-T29 already records that a",
          flush=True)
    print("    composite rejection does not identify WHICH leg failed.  The bridge is therefore",
          flush=True)
    print("    a STRUCTURAL LINK, not a confirmation of direction.", flush=True)

    res = {"source": "BOUCHAUD_TQP Sec 17.1, via corpus_text_v1",
           "sign_theorem": "E[G] ~ T v0 E[theta] ( E[s]/2 + w - R_inf ); E[theta] multiplicative",
           "theta_regimes": {"small_tick": "theta ~ 1, adverse selection neglected",
                             "large_tick": "theta ~ 0 except at front, adverse selection -> Sec 17.3"},
           "h_c_prediction_vs_measurement":
               {s: {"tick": TICK_REGIME[s], "book_predicts_h_c": TICK_REGIME[s] == "large",
                    "lane_measured_h_c": H_C_MEASURED[s]} for s in TICK_REGIME},
           "h_c_agreement": "%d/3" % agree,
           "fine_balance_sides": sides,
           "rejecting_symbols_share_a_side": rej == hard,
           "limitation": "gamma fit-range dependent => direction suggestive not established; "
                         "composite rejection does not identify the failing leg",
           "tokens": ["EQ_17_3_IS_A_SIGN_THEOREM_FILL_PROBABILITY_CANNOT_FLIP_MAKER_PNL",
                      "H_C_EXISTS_ONLY_WHERE_THETA_INDEPENDENCE_BREAKS_I_E_LARGE_TICK",
                      "THE_CORPUS_PREDICTS_THE_SYMBOL_PATTERN_THIS_LANE_MEASURED",
                      "SEC_17_1_TIES_MAKER_PNL_SIGN_TO_THE_FINE_BALANCE_SIDE",
                      "THE_DIRECTION_REMAINS_UNESTABLISHED_BECAUSE_GAMMA_IS_FIT_RANGE_DEPENDENT"],
           "ceiling": "RECONCILIATION"}
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT_KULLIYAT_T49_SIGN_THEOREM_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT_KULLIYAT_T49_SIGN_THEOREM_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
