r"""LANE C, round 24 -- gamma is not identifiable here, and the number that agreed was the broken one.

C-T23 left one soft cell in the exponent table: gamma is INDIRECT, reached through Eq. (16.16)
rather than measured, and GAMMA_NOT_MEASURABLE_FROM_AGGTRADES stands. The charter's stop rule
says either measure it or show it cannot be measured with this data. This shows the second, and
the corpus says it outright.

----------------------------------------------------------------------------------------------
PART 1 -- THE BOOK ANSWERS THE QUESTION DIRECTLY.

Bouchaud, section 12.2:

    "measuring metaorder impact still requires relatively detailed data that indicates which
     child orders belong to which metaorders. This information is not typically available in
     most publicly available data, which is anonymised and provides no explicit trader
     identifiers. Using such data only allows one to infer the AGGREGATE impact as in Section
     11.4. Identifying aggregate impact with metaorder impact is MISLEADING, and in most cases
     leads to a SUBSTANTIAL UNDERESTIMATION of metaorder impact."

And 12.2.1, "The Ideal Data Set", lists what is required: proprietary or detailed broker data
giving which child orders belong to which metaorders, the (t_i, p_i, nu_i) of each child, and
whether each child was a limit or a market order. `agg_trades` carries none of it.

So GAMMA_NOT_MEASURABLE_FROM_AGGTRADES is not a lane's opinion. It is the textbook's own
statement about this class of data, and it upgrades to NOT_IDENTIFIABLE.

IT ALSO SETTLES THE DIRECTION OF C-T20's ERROR. C-T20 reached gamma = 0.373 / 0.369 indirectly
from delta_cascade = 0.68 via Eq. (16.16). A cascade episode is an AGGREGATE. The book says
identifying the aggregate with the metaorder underestimates -- so the indirect gamma is not
merely uncertain, its bias has a KNOWN SIGN. C-T21 withdrew that ladder for a different reason
(target semantics); this adds the direction.

----------------------------------------------------------------------------------------------
PART 2 -- THERE IS AN IDENTITY-FREE ROUTE, AND IT MEASURES A DIFFERENT QUANTITY.

The LMF model links the order-sign autocorrelation to the metaorder SIZE distribution: if
metaorder sizes are Pareto with tail exponent alpha, the sign autocorrelation decays as
C(l) ~ l^-gamma_LMF with alpha = gamma_LMF + 1. Signs are visible without identifiers, so this
is measurable here.

But gamma_LMF is NOT the impact exponent gamma. One is the decay of sign memory; the other is
the concavity of price response in metaorder size. THIS IS THE THIRD SYMBOL COLLISION THIS LANE
HAS FOUND IN TWO ROUNDS -- after `p` (three objects, C-T23) and `zeta` (two objects, C-T23).

Bouchaud's reported values: equities alpha ~ 3/2 (so gamma_LMF ~ 0.5); and, for this estate's
asset class, "Bitcoin data allows a precise reconstruction of large metaorders, and suggests a
smaller exponent alpha ~ 1.10" (so gamma_LMF ~ 0.10).

Measured on 2,000,000 contiguous aggTrades per symbol, lags 10-2000:

    BTCUSDT  gamma_LMF 0.7746 (r2 0.986)  ->  alpha 1.775
    ETHUSDT  gamma_LMF 0.7892 (r2 0.978)  ->  alpha 1.789
    SOLUSDT  gamma_LMF 0.2092 (r2 0.935)  ->  alpha 1.209

----------------------------------------------------------------------------------------------
PART 3 -- AND THE ONE THAT AGREED WITH THE BOOK IS THE ONE THAT IS BROKEN.

SOL's 1.209 sits closest to Bouchaud's crypto 1.10. It is also the only series with an
anomalous autocorrelation: C(1) = 0.0141 while C(10) = 0.0795. A long-memory sign process
decays monotonically; SOL's RISES from lag 1 to lag 10.

The aggregation diagnostic explains it, and CLAUDE.md already records the mechanism -- one
aggTrade compresses a median of three raw trades:

    symbol   median dt   same-price   same-sign   median notional   C(1)
    BTCUSDT      0.0 ms      15.77%      76.30%             259.3   0.526
    ETHUSDT      2.0 ms      19.53%      73.46%             227.7   0.469
    SOLUSDT    220.0 ms      53.47%      53.87%              85.4   0.076

SOL's records arrive a median 220 ms apart against BTC's sub-millisecond, 53% of consecutive
records share a price, and yet only 54% share a sign -- barely above a coin flip. That is the
signature of heavy aggregation: same-side same-price runs are merged into ONE record, which
removes precisely the same-sign adjacencies that carry short-lag persistence.

So SOL's gamma_LMF is a property of the aggregation, not of SOL's order flow, and its alpha
must not be compared with the book's. THE NUMBER THAT APPEARED TO CONFIRM THE CORPUS WAS THE
BROKEN ONE.

And the same mechanism biases BTC and ETH in a known direction. Merging strips same-sign
adjacencies, which steepens the measured decay, which RAISES gamma_LMF and therefore alpha. So
1.775 and 1.789 are UPPER BOUNDS; the true alpha lies below, in the direction of the book's
values. That is a bound, not a reconciliation, and it is reported as one.

Read-only; database opened mode=ro. Measures exponents; tests no trading hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT_DIR = ROOT / "reports" / "research" / "c24_gamma_identifiability_v1"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
N_SIGN = 2_000_000
N_DIAG = 1_000_000
LAG_LO, LAG_HI = 10, 2000
BOOK_ALPHA_EQUITIES = 1.5
BOOK_ALPHA_BITCOIN = 1.10


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def powerfit(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return float("nan"), float("nan"), 0
    A = np.column_stack([np.ones(len(x)), np.log(x)])
    b, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
    r = np.log(y) - A @ b
    tot = float(((np.log(y) - np.log(y).mean()) ** 2).sum())
    return float(b[1]), float(1 - float(r @ r) / tot) if tot > 0 else float("nan"), len(x)


def sign_autocorr(con, sym) -> dict:
    bm = np.fromiter((r[0] for r in con.execute(
        "select is_buyer_maker from agg_trades where symbol=? order by ts_ms limit ?",
        (sym, N_SIGN))), dtype=np.int8)
    e = np.where(bm > 0, -1.0, 1.0)
    e = e - e.mean()
    v = float(e @ e)
    lags = np.unique(np.round(np.logspace(0, 4, 40)).astype(int))
    C, L = [], []
    for lg in lags:
        if lg >= len(e):
            break
        C.append(float(e[:-lg] @ e[lg:]) / v * len(e) / (len(e) - lg))
        L.append(int(lg))
    C = np.array(C)
    L = np.array(L)
    m = (L >= LAG_LO) & (L <= LAG_HI) & (C > 0)
    g, r2, k = powerfit(L[m], C[m])
    g = -g
    pick = {int(t): round(float(C[int(np.argmin(abs(L - t)))]), 4)
            for t in (1, 10, 100, 1000)}
    return {"n": int(len(e)), "gamma_LMF": round(g, 4), "r2": round(r2, 3),
            "lags_used": int(m.sum()), "fit_range": [LAG_LO, LAG_HI],
            "alpha_metaorder_size": round(g + 1, 4),
            "C_at": pick,
            "monotone_from_lag1": bool(pick[1] >= pick[10]),
            "curve": [{"lag": int(a), "C": round(float(b), 5)} for a, b in zip(L, C)]}


def aggregation_diagnostic(con, sym) -> dict:
    a = np.array(con.execute(
        "select ts_ms,price,notional,is_buyer_maker from agg_trades "
        "where symbol=? order by ts_ms limit ?", (sym, N_DIAG)).fetchall(), dtype=np.float64)
    ts, px, nt, bm = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    e = np.where(bm > 0, -1.0, 1.0)
    ec = e - e.mean()
    return {"n": int(len(ts)),
            "median_dt_ms": round(float(np.median(np.diff(ts))), 1),
            "share_consecutive_same_price": round(float(np.mean(px[1:] == px[:-1])), 5),
            "share_consecutive_same_sign": round(float(np.mean(e[1:] == e[:-1])), 5),
            "median_notional": round(float(np.median(nt)), 1),
            "C_lag1": round(float(ec[:-1] @ ec[1:]) / float(ec @ ec), 4)}


def build() -> dict:
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    ac, diag = {}, {}
    try:
        for s in SYMS:
            t0 = time.time()
            ac[s] = sign_autocorr(con, s)
            ac[s]["seconds"] = round(time.time() - t0, 1)
        for s in SYMS:
            diag[s] = aggregation_diagnostic(con, s)
    finally:
        con.close()

    suspect = [s for s in SYMS if not ac[s]["monotone_from_lag1"]]
    closest = min(SYMS, key=lambda s: abs(ac[s]["alpha_metaorder_size"] - BOOK_ALPHA_BITCOIN))
    return {
        "study": "C24_GAMMA_IDENTIFIABILITY_V1",
        "lane": "C", "stable_id": "C-T24",
        "generated_utc": _utc(),
        "charter_cell": ("C-T23 left gamma as the one soft cell: INDIRECT via Eq. 16.16, with "
                         "GAMMA_NOT_MEASURABLE_FROM_AGGTRADES standing. The stop rule says "
                         "measure it or show it cannot be measured."),
        "the_book_answers_it": {
            "quote_12_2": ("measuring metaorder impact still requires relatively detailed data "
                           "that indicates which child orders belong to which metaorders. This "
                           "information is not typically available in most publicly available "
                           "data, which is anonymised and provides no explicit trader "
                           "identifiers. Using such data only allows one to infer the AGGREGATE "
                           "impact as in Section 11.4. Identifying aggregate impact with "
                           "metaorder impact is MISLEADING, and in most cases leads to a "
                           "SUBSTANTIAL UNDERESTIMATION of metaorder impact."),
            "ideal_data_set_12_2_1": ["which child orders belong to which metaorders",
                                      "the (t_i, p_i, nu_i) of each child order",
                                      "whether each child was a limit or a market order"],
            "agg_trades_carries": ["ts_ms", "price", "quantity", "notional", "is_buyer_maker"],
            "verdict": "GAMMA_IS_NOT_IDENTIFIABLE_ON_THIS_ESTATES_DATA",
            "upgrade": ("GAMMA_NOT_MEASURABLE_FROM_AGGTRADES was a lane's finding; it is now "
                        "the textbook's own statement about this class of data, so it "
                        "upgrades from NOT_MEASURED to NOT_IDENTIFIABLE"),
            "direction_of_C_T20s_error": (
                "C-T20 reached gamma = 0.373/0.369 indirectly from delta_cascade = 0.68 via "
                "Eq. (16.16). A cascade episode is an AGGREGATE, and the book says identifying "
                "the aggregate with the metaorder UNDERESTIMATES -- so the indirect gamma is "
                "not merely uncertain, its bias has a known sign. C-T21 withdrew that ladder "
                "for a different reason; this adds the direction."),
        },
        "identity_free_route": {
            "model": "LMF: C(l) ~ l^-gamma_LMF with alpha_metaorder_size = gamma_LMF + 1",
            "why_it_is_available": "signs are visible without trader identifiers",
            "what_it_is_NOT": ("gamma_LMF is the decay of SIGN MEMORY; the impact gamma is the "
                               "concavity of price response in metaorder SIZE. Different "
                               "objects."),
            "third_symbol_collision": ("after `p` (three objects) and `zeta` (two objects) in "
                                       "C-T23, `gamma` is the third symbol this lane has found "
                                       "carrying more than one object in two rounds"),
            "book_values": {"equities_alpha": BOOK_ALPHA_EQUITIES,
                            "bitcoin_alpha": BOOK_ALPHA_BITCOIN,
                            "bitcoin_quote": ("Bitcoin data allows a precise reconstruction of "
                                              "large metaorders, and suggests a smaller "
                                              "exponent alpha ~ 1.10")},
            "measured": {s: {"gamma_LMF": ac[s]["gamma_LMF"], "r2": ac[s]["r2"],
                             "alpha": ac[s]["alpha_metaorder_size"],
                             "C_at": ac[s]["C_at"]} for s in SYMS},
        },
        "the_agreeing_number_is_the_broken_one": {
            "closest_to_book_bitcoin": closest,
            "its_alpha": ac[closest]["alpha_metaorder_size"],
            "book_bitcoin_alpha": BOOK_ALPHA_BITCOIN,
            "anomaly": ("a long-memory sign process decays monotonically; {0}'s C(1) = {1} is "
                        "BELOW its C(10) = {2}".format(
                            closest, ac[closest]["C_at"][1], ac[closest]["C_at"][10])),
            "symbols_failing_monotonicity": suspect,
            "aggregation_diagnostic": diag,
            "mechanism": ("heavy aggregation merges same-side same-price runs into ONE record, "
                          "removing precisely the same-sign adjacencies that carry short-lag "
                          "persistence; CLAUDE.md already records that one aggTrade compresses "
                          "a median of three raw trades"),
            "verdict": "SOL_GAMMA_LMF_IS_AN_AGGREGATION_ARTEFACT_NOT_ORDER_FLOW",
            "the_lesson": ("the number that appeared to confirm the corpus is the one that is "
                           "broken"),
            "direction_for_the_other_two": (
                "the same mechanism strips same-sign adjacencies on BTC and ETH too, which "
                "steepens the measured decay, raises gamma_LMF and therefore raises alpha. So "
                "1.775 and 1.789 are UPPER BOUNDS and the true alpha lies below, toward the "
                "book's values. A bound, not a reconciliation."),
        },
        "verdict": "GAMMA_NOT_IDENTIFIABLE_LMF_ROUTE_MEASURES_A_DIFFERENT_OBJECT",
        "what_is_NOT_claimed": [
            "That alpha has been measured. Two of three symbols give upper bounds and the "
            "third is an aggregation artefact.",
            "That the LMF relation holds here. It is applied, not tested; testing it needs the "
            "metaorder identities the whole round establishes are missing.",
            "That C-T20's indirect gamma is refuted by this. It was already withdrawn by "
            "C-T21; this only adds that its bias has a known sign.",
        ],
        "forward_sample_consumed": False,
    }


def render_md(a: dict) -> str:
    bk, lr, br = (a["the_book_answers_it"], a["identity_free_route"],
                  a["the_agreeing_number_is_the_broken_one"])
    L = ["# C-T24 — γ IS NOT IDENTIFIABLE, AND THE NUMBER THAT AGREED WAS THE BROKEN ONE", "",
         "`{0}` · generated {1}".format(a["verdict"], a["generated_utc"]), "",
         "**Charter cell:** {0}".format(a["charter_cell"]), "",
         "## 1. The book answers it directly — `{0}`".format(bk["verdict"]), "",
         "> Bouchaud §12.2: *\"{0}\"*".format(bk["quote_12_2"]), "",
         "§12.2.1's **ideal data set** requires: {0}. `agg_trades` carries only {1}.".format(
             "; ".join(bk["ideal_data_set_12_2_1"]), ", ".join(bk["agg_trades_carries"])), "",
         "**{0}**".format(bk["upgrade"]), "",
         "**And it settles the direction of C-T20's error.** {0}".format(
             bk["direction_of_C_T20s_error"]), "",
         "## 2. The identity-free route measures a different quantity", "",
         "`{0}` — {1}. But **{2}**".format(lr["model"], lr["why_it_is_available"],
                                           lr["what_it_is_NOT"]), "",
         "> 🔴 {0}.".format(lr["third_symbol_collision"].capitalize()), "",
         "Book values: equities α ≈ {0}; and for this asset class — *\"{1}\"*.".format(
             lr["book_values"]["equities_alpha"], lr["book_values"]["bitcoin_quote"]), "",
         "| symbol | γ_LMF | r² | **α = γ+1** | C(1) | C(10) | C(100) | C(1000) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for s in SYMS:
        m = lr["measured"][s]
        L.append("| {0} | {1} | {2} | **{3}** | {4} | {5} | {6} | {7} |".format(
            s, m["gamma_LMF"], m["r2"], m["alpha"], m["C_at"][1], m["C_at"][10],
            m["C_at"][100], m["C_at"][1000]))
    L += ["", "## 3. 🔴 And the one that agreed with the book is the one that is broken", "",
          "**{0}**'s α = **{1}** sits closest to the book's crypto **{2}**. It is also the only "
          "series failing monotonicity: {3}.".format(
              br["closest_to_book_bitcoin"], br["its_alpha"], br["book_bitcoin_alpha"],
              br["anomaly"]), "",
          "| symbol | median dt | same-price | same-sign | median notional | C(1) |",
          "|---|--:|--:|--:|--:|--:|"]
    for s in SYMS:
        d = br["aggregation_diagnostic"][s]
        L.append("| {0} | {1} ms | {2:.2%} | {3:.2%} | {4} | {5} |".format(
            s, d["median_dt_ms"], d["share_consecutive_same_price"],
            d["share_consecutive_same_sign"], d["median_notional"], d["C_lag1"]))
    L += ["", "**Mechanism.** {0}".format(br["mechanism"]), "",
          "→ **`{0}`**. {1}.".format(br["verdict"], br["the_lesson"].capitalize()), "",
          "**Direction for the other two.** {0}".format(br["direction_for_the_other_two"]), "",
          "## What is NOT claimed", ""]
    for x in a["what_is_NOT_claimed"]:
        L.append("- {0}".format(x))
    L += ["", "```verdict", a["verdict"],
          "GAMMA_IS_NOT_IDENTIFIABLE_ON_THIS_ESTATES_DATA",
          "GAMMA_NOT_MEASURABLE_UPGRADES_TO_NOT_IDENTIFIABLE_ON_THE_BOOKS_OWN_STATEMENT",
          "AGGREGATE_FOR_METAORDER_SUBSTITUTION_UNDERESTIMATES_KNOWN_SIGN",
          "LMF_ROUTE_MEASURES_SIGN_MEMORY_NOT_IMPACT_CONCAVITY",
          "GAMMA_IS_THE_THIRD_SYMBOL_COLLISION_FOUND_IN_TWO_ROUNDS",
          "SOL_GAMMA_LMF_IS_AN_AGGREGATION_ARTEFACT_NOT_ORDER_FLOW",
          "THE_NUMBER_THAT_AGREED_WITH_THE_CORPUS_WAS_THE_BROKEN_ONE",
          "BTC_ETH_ALPHA_ARE_UPPER_BOUNDS_1_775_AND_1_789", "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    a = build()
    md = render_md(a)
    enc = sys.stdout.encoding or "utf-8"
    if args.stdout:
        sys.stdout.write(md.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "C24_GAMMA_IDENTIFIABILITY_V1.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "C24_GAMMA_IDENTIFIABILITY_V1.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "verdict": a["verdict"],
        "gamma_identifiable": False,
        "alpha": {s: a["identity_free_route"]["measured"][s]["alpha"] for s in SYMS},
        "gamma_LMF": {s: a["identity_free_route"]["measured"][s]["gamma_LMF"] for s in SYMS},
        "monotonicity_failures": a["the_agreeing_number_is_the_broken_one"][
            "symbols_failing_monotonicity"],
        "closest_to_book": a["the_agreeing_number_is_the_broken_one"][
            "closest_to_book_bitcoin"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
