r"""LANE C, round 26 -- the design family joins the registry, and A's k turns out to be C's tail index.

C-T25 closed with "other families will have their own collisions." This checks the DESIGN family
-- the letters lane A's frozen preregistration actually runs on -- and finds three things, one of
which is a bridge neither lane knew it had.

----------------------------------------------------------------------------------------------
1. A NON-FINDING, RECORDED AS ONE.

`h* = [2c/(k f sigma_d)]^2` makes the design horizon go as the SQUARE of cost, and A's own abort
trigger S5 says so: "A 2x cost change moves h* by 4x." So whether `c` uses the per-side or the
round-trip fee is a 4x question. The parameter table (line 83) says only "spread + fee".

It is not ambiguous. Line 56 states it: "section 460's frontier is single-leg at c = 10 bps",
and line 45 contrasts the pairs case, "two legs, cost 20 bps". The file resolves its own
convention. NO DEFECT -- recorded because the check was run and came back clean, which is worth
as much as a finding and is usually not written down.

----------------------------------------------------------------------------------------------
2. THE BRIDGE. A'S `k` IS NOT A FREE CONSTANT -- IT IS A FUNCTION OF THE TAIL INDEX.

A's `k = E|r|/sigma`, frozen at 0.6966. For a Gaussian that ratio is sqrt(2/pi) = 0.7979, so A's
measurement sits 12.69% BELOW Gaussian. That deficit is a fat-tail signature: the ratio falls
monotonically as the tail thickens.

Inverting it on a standardised Student-t gives the tail index A's constant implies:

    k = 0.6966   ->   nu = 3.765

C measured the tail directly, by an entirely different estimator, in section 478: Hill on
60-minute moves, zeta = 2.33 to 3.83 across k. And Bouchaud reports zeta ~ 3 as near-universal
across markets.

    A's bulk-shape statistic   ->  nu = 3.765
    C's tail-order estimator   ->  zeta = 2.33 - 3.83
    the book                   ->  zeta ~ 3

THEY AGREE. Two statistics that share no machinery -- one a mean-absolute-to-sigma ratio computed
over a whole distribution, the other an order-statistic estimator using only the largest
observations -- land in the same tail-index range, on different objects, in different lanes,
with no contact. That is the second byte-level coincidence this atlas has surfaced, after the
identical REACTION_VS_PREDICTION token.

AND IT GIVES A'S FREEZE A FREE CHECK. The freeze block says "k = 0.6966 (section 467; re-measure
at freeze)". Since k is a function of the tail index, and the tail index is the quantity
Bouchaud reports as universal, the re-measurement has a PREDICTED RANGE:

    nu in [3, 5]   <->   k in [0.6366, 0.7351]

A freeze-date k inside that band confirms the tail regime is unchanged and nothing needs
revisiting. A k outside it means either the tail regime moved or the estimator broke -- and
because N_required goes as k^-2, the cost of not noticing is a mis-sized sample. This is a
zero-cost check A can run at freeze and did not know was available.

----------------------------------------------------------------------------------------------
3. THE COST BASIS, MEASURED WHERE A ASSUMED.

Line 83: "Spread = one tick (section 452, 12 of 15 symbols)". C-T23 measured the same thing on
the three majors and found it holds on 3 of 3, at 97.7% / 98.8% / 99.9% of quotes -- so the
assumption is stronger than A recorded, not weaker.

What A does not record is what the spread is WORTH inside c, and it differs by two orders of
magnitude across the very symbols the design pools:

    symbol     spread bps    c bps    spread/c    h* factor (c/fee)^2
    BTCUSDT        0.0154   10.0154      0.154%          1.0031
    ETHUSDT        0.0530   10.0530      0.527%          1.0106
    SOLUSDT        1.3148   11.3148     11.620%          1.2802

So `c` is FEE-DOMINATED on the small-tick majors: the spread moves the design horizon by 0.3%
and 1.1%. On SOL it is 11.6% of cost and lifts h* by 28%. Cross-symbol variation in h* therefore
comes from sigma_d, not from the spread -- except on the large-tick symbol, where it is a
quarter of the horizon.

----------------------------------------------------------------------------------------------
AND THE DESIGN FAMILY HAS ITS OWN COLLISIONS. `k` carries two objects, `h` is used with THREE
different units across lanes (days in A's frontier, trades in C-T23's lag grid, minutes in the
episode work) and that ambiguity already cost this lane an inference in C-T23, where A-S40's h
grid could not be placed. `f` is the exception worth recording: A's capture scalar and A-S40's
f(h) are the SAME object at one horizon, verified consistent, not a collision.

Read-only. Rebuilds the registry with both families; measures the cost shares and the implied
tail index from constants already published.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "atlas"
V1_JSON = OUT_DIR / "EXPONENT_SYMBOL_REGISTRY_V1.json"
OUT_MD = OUT_DIR / "SYMBOL_REGISTRY_V2.md"
OUT_JSON = OUT_DIR / "SYMBOL_REGISTRY_V2.json"

A_K = 0.6966
FEE_BPS = 10.0
SPREAD_BPS = {"BTCUSDT": 0.0154, "ETHUSDT": 0.0530, "SOLUSDT": 1.3148}
HILL_RANGE = [2.33, 3.83]
BOOK_ZETA = 3.0

DESIGN_FAMILY = [
    {"object_id": "K_MEAN_ABS_OVER_SIGMA", "letter": "k", "family": "design",
     "definition": "k = E|r|/sigma, the mean-absolute-to-sigma ratio of returns",
     "conditions_on": "the return distribution's shape",
     "measured": "0.6966 (A-S45 freeze block, from section 467)", "owner": "A-S45",
     "note": "Gaussian value sqrt(2/pi) = 0.7979; A sits 12.69% below it, a fat-tail signature"},
    {"object_id": "K_IMPACT_PREFACTOR", "letter": "k", "family": "design",
     "definition": "prefactor in dP = k Q^delta",
     "conditions_on": "traded quantity",
     "measured": "not reported separately from delta", "owner": "C-T6",
     "note": "Kissell/Bouchaud usage; dimensional, unlike A's dimensionless ratio"},
    {"object_id": "F_CAPTURE", "letter": "f", "family": "design",
     "definition": "the capture fraction, f = R/E|r| evaluated at the design horizon",
     "conditions_on": "a predictor and a horizon",
     "measured": "the estimand -- NOT preregistered; f_design = 0.010 enters only through power",
     "owner": "A prereg section 3",
     "note": "VERIFIED CONSISTENT with A-S40's f(h): same object, the scalar being the function "
             "at one h. Not a collision."},
    {"object_id": "C_SINGLE_LEG_COST", "letter": "c", "family": "design",
     "definition": "c = spread + fee, single-leg round trip",
     "conditions_on": "a symbol and a fee tier",
     "measured": "10 bps (prereg line 56); two-leg pairs case is 20 bps (line 45)",
     "owner": "A prereg", "note": "convention IS stated -- the check for ambiguity came back "
                                  "clean"},
    {"object_id": "H_HORIZON_DAYS", "letter": "h", "family": "design",
     "definition": "design horizon h* = [2c/(k f sigma_d)]^2",
     "conditions_on": "cost, capture and daily volatility", "measured": "derived per symbol",
     "owner": "A prereg", "note": "units: DAYS"},
    {"object_id": "H_LAG_IN_TRADES", "letter": "h", "family": "design",
     "definition": "lag in trades in the lagged capture curve f(h)",
     "conditions_on": "a trade count", "measured": "grid 1..4096", "owner": "C-T23",
     "note": "units: TRADES. A-S40's own h grid could not be placed on either scale from its "
             "text, and that ambiguity cost C-T23 an inference."},
    {"object_id": "H_HORIZON_MINUTES", "letter": "h", "family": "design",
     "definition": "holding horizon of the episode outcome imp_H",
     "conditions_on": "wall-clock time", "measured": "grid 1..360", "owner": "episode work",
     "note": "units: MINUTES"},
    {"object_id": "SIGMA_DAILY", "letter": "sigma", "family": "design",
     "definition": "daily volatility entering h*", "conditions_on": "a day",
     "measured": "per symbol at freeze", "owner": "A prereg", "note": "units: bps/day"},
    {"object_id": "SIGMA_REALISED_30MIN", "letter": "sigma", "family": "design",
     "definition": "realised volatility over the prior 30 minutes",
     "conditions_on": "a 30-minute window", "measured": "4.64 vs 3.36 bps, tail vs rest",
     "owner": "C episode work", "note": "same KIND of object at a different scale -- an "
                                        "ambiguity, not a collision"},
    {"object_id": "N_EFF_EFFECTIVE_BETS", "letter": "N_eff", "family": "design",
     "definition": "effective independent observation count",
     "conditions_on": "the dependence structure",
     "measured": "3.27, not the 8 first assumed", "owner": "A prereg section 8b",
     "note": "three sections got this wrong before it was measured (A-S35, A-S41, A-S43)"},
]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def k_of_nu(nu: float) -> float:
    """E|X|/sigma for a standardised Student-t with nu degrees of freedom."""
    lg = math.lgamma
    e_abs_t = (2 * math.sqrt(nu) * math.exp(lg((nu + 1) / 2) - lg(nu / 2))
               / ((nu - 1) * math.sqrt(math.pi)))
    return math.sqrt((nu - 2) / nu) * e_abs_t


def nu_of_k(k: float) -> float:
    lo, hi = 2.05, 1e6
    for _ in range(300):
        mid = (lo + hi) / 2
        if k_of_nu(mid) < k:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def build() -> dict:
    gauss = math.sqrt(2 / math.pi)
    nu = nu_of_k(A_K)
    band = {"nu_3": round(k_of_nu(3.0), 4), "nu_5": round(k_of_nu(5.0), 4)}
    cost = []
    for s, sp in SPREAD_BPS.items():
        c = sp + FEE_BPS
        cost.append({"symbol": s, "spread_bps": sp, "c_bps": round(c, 4),
                     "spread_share_of_c": round(sp / c, 5),
                     "h_star_factor": round((c / FEE_BPS) ** 2, 4)})
    v1 = json.loads(V1_JSON.read_text(encoding="utf-8")) if V1_JSON.exists() else {}
    exponent_family = v1.get("registry", [])
    for e in exponent_family:
        e.setdefault("family", "exponent")
    allreg = exponent_family + DESIGN_FAMILY
    letters = {}
    for e in allreg:
        letters.setdefault(e["letter"], []).append(e["object_id"])
    collisions = {k: v for k, v in letters.items() if len(v) > 1}
    for e in allreg:
        e["shares_its_letter_with"] = [o for o in letters[e["letter"]] if o != e["object_id"]]

    return {
        "study": "C26_DESIGN_FAMILY_REGISTRY_V1", "lane": "C", "stable_id": "C-T26",
        "generated_utc": _utc(),
        "supersedes": {"file": "EXPONENT_SYMBOL_REGISTRY_V1", "how": "contained verbatim; V1 is "
                                                                    "not withdrawn"},
        "non_finding_recorded": {
            "checked": ("whether the frozen prereg leaves the fee convention ambiguous, since "
                        "h* goes as c^2 and A's own S5 trigger says a 2x cost change moves h* "
                        "by 4x"),
            "result": ("line 56 states it -- 'section 460's frontier is single-leg at c = 10 "
                       "bps' -- and line 45 contrasts the two-leg pairs case at 20 bps"),
            "verdict": "NO_DEFECT_THE_FILE_RESOLVES_ITS_OWN_CONVENTION",
            "why_recorded": ("the check was run and came back clean, which is worth as much as "
                             "a finding and is usually not written down"),
        },
        "the_bridge": {
            "label": "A_K_IS_C_TAIL_INDEX",
            "A_k": A_K, "gaussian_k": round(gauss, 4),
            "below_gaussian_pct": round(100 * (A_K / gauss - 1), 2),
            "implied_nu": round(nu, 3),
            "C_hill_range": HILL_RANGE, "book_zeta": BOOK_ZETA,
            "agree": bool(HILL_RANGE[0] <= nu <= HILL_RANGE[1]),
            "why_it_matters": ("k is not a free constant. It is a function of the tail index, "
                               "and the tail index is the quantity Bouchaud reports as "
                               "near-universal."),
            "independence": ("the two statistics share no machinery: a mean-absolute-to-sigma "
                             "ratio computed over a whole distribution, and an order-statistic "
                             "estimator using only the largest observations"),
            "free_check_for_As_freeze": {
                "predicted_band": "nu in [3, 5]  <->  k in [{0}, {1}]".format(
                    band["nu_3"], band["nu_5"]),
                "inside": "tail regime unchanged, nothing to revisit",
                "outside": ("either the tail regime moved or the estimator broke; since "
                            "N_required goes as k^-2, the cost of not noticing is a mis-sized "
                            "sample"),
                "cost": "zero -- A already plans to re-measure k at freeze",
            },
        },
        "cost_basis": {
            "A_assumption": "Spread = one tick (section 452, 12 of 15 symbols)",
            "C_measurement": ("holds on 3 of 3 majors at 97.7% / 98.8% / 99.9% of quotes -- "
                              "the assumption is stronger than A recorded, not weaker"),
            "fee_bps_single_leg": FEE_BPS,
            "rows": cost,
            "reading": ("c is FEE-DOMINATED on the small-tick majors: the spread moves the "
                        "design horizon by 0.3% and 1.1%. On SOL it is 11.6% of cost and lifts "
                        "h* by 28%. Cross-symbol variation in h* comes from sigma_d, not the "
                        "spread -- except on the large-tick symbol."),
        },
        "registry": allreg, "letters": letters, "collisions": collisions,
        "counts": {"letters": len(letters), "objects": len(allreg),
                   "overloaded_letters": len(collisions),
                   "exponent_family": len(exponent_family),
                   "design_family": len(DESIGN_FAMILY)},
        "design_family_collisions": {
            "k": "two objects -- a dimensionless distribution ratio and a dimensional impact "
                 "prefactor",
            "h": "THREE units across lanes: days (A's frontier), trades (C-T23's lag grid), "
                 "minutes (the episode work). This already cost C-T23 an inference, where "
                 "A-S40's h grid could not be placed on either scale from its text.",
            "f": "NOT a collision -- A's capture scalar and A-S40's f(h) are the same object "
                 "at one horizon, verified consistent",
        },
        "verdict": "DESIGN_FAMILY_ADDED_AND_A_K_IS_C_TAIL_INDEX",
        "what_is_NOT_claimed": [
            "That A's k is wrong. It is measured; this only shows what it implies and gives a "
            "band to check it against at freeze.",
            "That the Student-t inversion identifies the true distribution. It is one "
            "parametric family; the agreement with an independent order-statistic estimator is "
            "the evidence, not the model.",
            "That the spread finding changes the design. It does not: A pools symbols and the "
            "spread is negligible on two of three. It is recorded so the third is not read as "
            "the same case.",
        ],
    }


def render_md(a: dict) -> str:
    br, cb, nf = a["the_bridge"], a["cost_basis"], a["non_finding_recorded"]
    L = ["# SYMBOL REGISTRY V2 — exponent family plus design family", "",
         "`{0}` · built {1} · lane C, `{2}`".format(a["verdict"], a["generated_utc"],
                                                    a["stable_id"]), "",
         "Supersedes `{0}` by containing it verbatim; V1 is **not withdrawn**.".format(
             a["supersedes"]["file"]), "",
         "**{0} letters carry {1} objects** ({2} exponent, {3} design); **{4} letters are "
         "overloaded**.".format(a["counts"]["letters"], a["counts"]["objects"],
                                a["counts"]["exponent_family"], a["counts"]["design_family"],
                                a["counts"]["overloaded_letters"]), "",
         "---", "", "## 1. A non-finding, recorded as one — `{0}`".format(nf["verdict"]), "",
         "**Checked:** {0}.".format(nf["checked"]), "",
         "**Result:** {0}.".format(nf["result"]), "",
         "*{0}*".format(nf["why_recorded"].capitalize()), "",
         "## 2. 🔴 The bridge — `{0}`".format(br["label"]), "",
         "A's `k = E|r|/σ` is frozen at **{0}**. For a Gaussian that ratio is **{1}**, so A sits "
         "**{2}% below** it — a fat-tail signature, since the ratio falls monotonically as the "
         "tail thickens. Inverting on a standardised Student-t:".format(
             br["A_k"], br["gaussian_k"], br["below_gaussian_pct"]), "",
         "| statistic | source | tail index |", "|---|---|--:|",
         "| `k = E\\|r\\|/σ` → ν | A-S45 (bulk shape) | **{0}** |".format(br["implied_nu"]),
         "| Hill on 60-min moves | §478 (order statistic) | **{0}–{1}** |".format(
             br["C_hill_range"][0], br["C_hill_range"][1]),
         "| universality | Bouchaud | ~{0} |".format(br["book_zeta"]), "",
         "**They agree: {0}.** {1}.".format(br["agree"], br["independence"].capitalize()), "",
         "> {0}".format(br["why_it_matters"]), "",
         "### A free check for A's freeze", "",
         "The freeze block says *\"k = 0.6966 (§467; **re-measure at freeze**)\"*. Since `k` is a "
         "function of the tail index, the re-measurement has a **predicted band**:", "",
         "> **{0}**".format(br["free_check_for_As_freeze"]["predicted_band"]), "",
         "- **inside** → {0}".format(br["free_check_for_As_freeze"]["inside"]),
         "- **outside** → {0}".format(br["free_check_for_As_freeze"]["outside"]),
         "- **cost:** {0}".format(br["free_check_for_As_freeze"]["cost"]), "",
         "## 3. The cost basis, measured where A assumed", "",
         "A: *\"{0}\"*. C: {1}.".format(cb["A_assumption"], cb["C_measurement"]), "",
         "| symbol | spread bps | c bps | spread / c | h\\* factor |", "|---|--:|--:|--:|--:|"]
    for r in cb["rows"]:
        L.append("| {0} | {1} | {2} | **{3:.3%}** | **{4}×** |".format(
            r["symbol"], r["spread_bps"], r["c_bps"], r["spread_share_of_c"],
            r["h_star_factor"]))
    L += ["", "> {0}".format(cb["reading"]), "",
          "## 4. Design-family collisions", ""]
    for k, v in a["design_family_collisions"].items():
        L.append("- **`{0}`** — {1}".format(k, v))
    L += ["", "---", "", "## The full registry", "",
          "| family | letter | object | definition | measured | owner | shares letter with |",
          "|---|---|---|---|---|---|---|"]
    for e in a["registry"]:
        L.append("| {0} | `{1}` | **{2}** | {3} | {4} | {5} | {6} |".format(
            e.get("family", "?"), e["letter"], e["object_id"], e["definition"],
            e["measured"], e["owner"], ", ".join(e["shares_its_letter_with"]) or "—"))
    L += ["", "## What is NOT claimed", ""]
    for x in a["what_is_NOT_claimed"]:
        L.append("- {0}".format(x))
    L += ["", "```verdict", a["verdict"],
          "NO_DEFECT_THE_FILE_RESOLVES_ITS_OWN_CONVENTION",
          "A_K_IMPLIES_NU_3_765_AGREEING_WITH_C_HILL_2_33_TO_3_83",
          "K_IS_NOT_A_FREE_CONSTANT_IT_IS_A_FUNCTION_OF_THE_TAIL_INDEX",
          "FREEZE_DATE_K_HAS_A_PREDICTED_BAND_0_6366_TO_0_7351",
          "SPREAD_IS_ONE_TICK_ON_THREE_OF_THREE_MAJORS",
          "COST_IS_FEE_DOMINATED_EXCEPT_ON_THE_LARGE_TICK_SYMBOL",
          "H_IS_USED_WITH_THREE_DIFFERENT_UNITS_ACROSS_LANES",
          "F_IS_NOT_A_COLLISION_VERIFIED_CONSISTENT", "```", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    a = build()
    md = render_md(a)
    enc = sys.stdout.encoding or "utf-8"
    if args.stdout:
        sys.stdout.write(md.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(a, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"verdict": a["verdict"], "counts": a["counts"],
                      "collisions": {k: len(v) for k, v in a["collisions"].items()},
                      "implied_nu": a["the_bridge"]["implied_nu"],
                      "agree": a["the_bridge"]["agree"],
                      "k_band": a["the_bridge"]["free_check_for_As_freeze"]["predicted_band"],
                      "spread_share": {r["symbol"]: r["spread_share_of_c"]
                                       for r in a["cost_basis"]["rows"]}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
