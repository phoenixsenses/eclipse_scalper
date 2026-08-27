# -*- coding: utf-8 -*-
"""C-T41 -- SEC 12.3.5 IS A LIST OF THE ERRORS THIS LANE MADE; AND WHERE THE HAIRCUT BINDS.

Asked what the corpus connects these results to, the answer turned out to be a section this
lane never opened.  Sec 12.3.5 is titled COMMON MISCONCEPTIONS, and it enumerates, in advance,
several of the mistakes this lane spent the day discovering by measurement.

    M1  "The impact of a metaorder of volume Q is not equal to the aggregate impact of order
         imbalance dV, which is LINEAR, and not square-root-like, for small Q.  Therefore, one
         cannot measure the impact of a metaorder without being able to ascribe its
         constituent trades to a given investor."

    M2  "The square-root impact law applies to slow metaorders composed of several individual
         trades, but not to the individual trades themselves."

    M3  "at the single-trade level ... the impact of a single market order, R(v,1), does not
         behave like a square-root, although it behaves as a concave function of v.  This
         concavity HAS NO IMMEDIATE RELATION with the concavity of the square-root impact for
         metaorders."

    M4  footnote 9: "the assumption that the aggregate order-flow imbalance dV in a given time
         window is mostly due to the presence of a single trader executing a metaorder of size
         Q, such that dV = Q + noise, is NOT WARRANTED."

    M5  Eq (12.9): E[(m_{t+T}-m_t)^2 | sum v = V] ~ V is "a trivial consequence of the
         diffusive nature of prices ... it tells us NOTHING about the average directional
         price change."  A square root found on UNSIGNED price changes is diffusion, not
         impact.

    M6  "why it [the square-root law] is insensitive to the tick size" -- a claim that stands
         against everything else this lane found to be tick-sensitive.

PART 1 maps this lane's errata onto that list.  PART 2 runs the calculation the corpus's
Sec 22.3 warning implies, using this lane's own measured impact curve rather than the book's
generic square root, and asks where it binds for this repo.

    Sec 22.3: "the impact-corrected value of a 1% holding of the market capitalisation of a
     stock with 2% daily volatility should be ~1.5% below the market price!"

This lane measured dP = k Q^0.68 (r2 0.99, $12k to $19M) and Lane A measured that impact
equals the round-trip fee at $3.4M on BTC, with Eclipse 379x below that.  Those two numbers
plus the exponent give the haircut at Eclipse's size without any new data.

No DB.  Arithmetic on published values plus a keyword crosswalk.
ESTIMATION.  Ceiling: RECONCILIATION.

  python -m tools.ct41_misconceptions_crosswalk_and_the_haircut --i-have-approval
"""
from __future__ import annotations

import glob
import io
import json
import os
import sys

OUT = "reports/atlas"
LEDGER_GLOBS = ("reports/atlas/IMMUTABLE_ERRATA_LEDGER_*.json",
                "reports/research/hb4_liquidation_specialness_v1/"
                "IMMUTABLE_ERRATA_LEDGER_*.json")

MISCONCEPTIONS = {
    "M1": {"text": "metaorder impact is not aggregate dV impact; dV impact is LINEAR for "
                   "small Q; metaorder impact needs trade-to-investor attribution",
           "keys": ("aggregate", "imbalance", "metaorder", "attribut", "ascribe",
                    "NOT_MEASURABLE", "ZETA_IS_NOT_DELTA", "mixed estimand",
                    "MIXED_ESTIMANDS", "target-semantics", "TARGET_SEMANTICS")},
    "M2": {"text": "the square-root law applies to slow metaorders, not to individual trades",
           "keys": ("single order", "single market order", "individual trade",
                    "SINGLE_ORDER", "zeta_single", "ladder")},
    "M3": {"text": "R(v,1) concavity has NO relation to metaorder square-root concavity",
           "keys": ("concav", "ladder", "LADDER", "aggregation", "AGGREGATION")},
    "M4": {"text": "dV = Q + noise is not warranted; dV superposes overlapping metaorders",
           "keys": ("episode", "EPISODE", "cascade", "CASCADE", "aggregate of many",
                    "many agents")},
    "M5": {"text": "unsigned price change ~ sqrt(V) is diffusion, not impact",
           "keys": ("unsigned", "UNSIGNED", "diffusiv", "DIFFUS")},
    "M6": {"text": "the square-root law is insensitive to tick size",
           "keys": ("tick", "TICK")},
}

# Part 2 inputs, all published
DELTA_MEASURED = 0.68          # C-T6 / H-K1, dP = k Q^0.68, r2 0.99, $12k-$19M
IMPACT_EQUALS_FEE_AT = 3.4e6   # A-S30, BTCUSDT
ECLIPSE_BELOW_FACTOR = 379.0   # A-S30
FEE_ROUND_TRIP_BPS = 10.0      # CLAUDE.md canonical BINANCE_BASE taker, 5.0 per side
BOOK_DELTA = 0.5


def main():
    if "--i-have-approval" not in set(sys.argv[1:]):
        print("REFUSED")
        return
    entries = []
    for g in LEDGER_GLOBS:
        for p in sorted(glob.glob(g)):
            try:
                d = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            for e in d.get("entries", []):
                entries.append({
                    "id": e.get("errata_id"),
                    "text": " ".join(str(e.get(k, "")) for k in
                                     ("old_statement", "corrected_statement",
                                      "source_section_or_line")).lower()})

    print("=== PART 1  Sec 12.3.5 'Common Misconceptions' vs this lane's errata ===",
          flush=True)
    cross = {}
    for mid, m in MISCONCEPTIONS.items():
        hits = [e["id"] for e in entries
                if any(k.lower() in e["text"] for k in m["keys"])]
        cross[mid] = {"text": m["text"], "errata": hits, "n": len(hits)}
        print("  %-3s  %-68s  %d errata" % (mid, m["text"][:68], len(hits)), flush=True)
        if hits:
            print("       %s" % ", ".join(hits), flush=True)
    total_hit = len({x for v in cross.values() for x in v["errata"]})
    print("  distinct errata touching a named misconception: %d of %d"
          % (total_hit, len(entries)), flush=True)

    print("\n=== PART 2  where the haircut binds ===", flush=True)
    eclipse = IMPACT_EQUALS_FEE_AT / ECLIPSE_BELOW_FACTOR
    rows = []
    for Q in (eclipse, 1e5, 1e6, IMPACT_EQUALS_FEE_AT, 1e7, 1e8):
        rel = Q / IMPACT_EQUALS_FEE_AT
        imp_meas = FEE_ROUND_TRIP_BPS * (rel ** DELTA_MEASURED)
        imp_book = FEE_ROUND_TRIP_BPS * (rel ** BOOK_DELTA)
        rows.append({"notional_usd": Q, "impact_bps_delta_0_68": imp_meas,
                     "impact_bps_delta_0_50": imp_book,
                     "impact_over_fee": imp_meas / FEE_ROUND_TRIP_BPS})
        print("    $%12s   impact %8.3f bps (delta 0.68) | %8.3f bps (delta 0.50) | "
              "%.4f x the fee"
              % ("{:,.0f}".format(Q), imp_meas, imp_book, imp_meas / FEE_ROUND_TRIP_BPS),
              flush=True)
    ec = rows[0]
    print("\n    At Eclipse's own size ($%s, from A-S30's 379x): impact is %.3f bps, "
          "%.2f%% of the %.0f bps round-trip fee."
          % ("{:,.0f}".format(eclipse), ec["impact_bps_delta_0_68"],
             100 * ec["impact_over_fee"], FEE_ROUND_TRIP_BPS), flush=True)
    print("    The corpus's haircut warning is about marking LARGE portfolios; at this size "
          "it does not bind.", flush=True)
    print("    What binds is the FEE -- which is what every cost measurement in this lane "
          "converged on.", flush=True)

    res = {"part1_crosswalk": cross,
           "n_errata_total": len(entries), "n_errata_touching_a_misconception": total_hit,
           "part2_haircut": {"delta_measured": DELTA_MEASURED, "book_delta": BOOK_DELTA,
                             "impact_equals_fee_at_usd": IMPACT_EQUALS_FEE_AT,
                             "eclipse_notional_usd": eclipse,
                             "fee_round_trip_bps": FEE_ROUND_TRIP_BPS, "rows": rows},
           "tokens": ["SEC_12_3_5_ENUMERATES_ERRORS_THIS_LANE_MADE",
                      "THE_CORPUS_WAS_READ_FOR_CLAIMS_NEVER_FOR_ERRORS",
                      "AT_ECLIPSE_SIZE_IMPACT_IS_UNDER_TWO_PERCENT_OF_THE_FEE",
                      "THE_HAIRCUT_WARNING_DOES_NOT_BIND_HERE",
                      "THE_FEE_IS_THE_BINDING_CONSTRAINT"],
           "ceiling": "RECONCILIATION"}
    os.makedirs(OUT, exist_ok=True)
    with io.open(os.path.join(OUT, "CT41_MISCONCEPTIONS_AND_HAIRCUT_V1.json"), "w",
                 encoding="utf-8") as f:
        f.write(json.dumps(res, indent=2, ensure_ascii=False))
    print("\nwritten %s/CT41_MISCONCEPTIONS_AND_HAIRCUT_V1.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
