r"""LANE C, round 25 -- the symbol registry, and a read-only audit of lane A's frozen prereg.

TWO THINGS THIS ROUND DOES.

(1) AUDITS LANE A'S PREREGISTRATION for exponent usage, because C-T23's `to A` line asked that
any impact exponent it quotes must name WHICH object, and four of the five symbols in that
family are overloaded. The prereg is already FROZEN (sha256 d997e8fb..., 2026-08-27T03:21:20Z),
so this is read-only: a finding is recorded, nothing is edited. Charter rule 5.

(2) BUILDS THE ARTEFACT THAT PREVENTS THE NEXT COLLISION. Three rounds have now found the same
failure: `p` carries three objects (C-T23), `zeta` two (C-T23), `gamma` two (C-T24). The atlas
index cannot see any of it, because both objects emit verdict tokens containing the same letter.
A token index keys on strings; a collision is a fact about MEANINGS. So the registry below is
keyed on the object, not the letter.

----------------------------------------------------------------------------------------------
THE AUDIT, AND IT CLEARS LANE A.

§10 of the prereg, "What would falsify this design itself":

    "If capture varies with horizon (A-S40 measured p ~ -0.5 for order flow), then a single f at
     a single h* is a point on a curve, not a constant. This design is valid at its own horizon
     and says nothing about others. p is unmeasured for every family except order flow."

That is correct practice and the audit says so plainly: the exponent appears in the FALSIFICATION
section, not the machinery; it is attributed (A-S40); it is scoped by family ("for order flow");
and the design's claim is explicitly restricted to its own horizon. NO MISUSE. C-T23's warning
was already satisfied before it was written.

TWO THINGS THE FROZEN TEXT CANNOT KNOW, recorded here rather than in it.

  (a) `p` IS NOT A CONSTANT WITHIN ORDER FLOW EITHER. C-T23 measured it on independent windows:
      +0.215 / +0.009 at h <= 16, and -0.721 / -0.785 at h >= 256. A-S40's own text says the
      same ("a single fitted p ~ -0.5 is the AVERAGE OF A TRANSITION, not a law"), but the
      prereg's clause quotes the single number. The clause's DIRECTION is right; its MAGNITUDE
      is unbounded, because near the transition the local exponent moves fast. A design sitting
      at one h* is unaffected; a reader inferring a sensitivity from "p ~ -0.5" is not.

  (b) "p IS UNMEASURED FOR EVERY FAMILY EXCEPT ORDER FLOW" IS SUPERSEDED, in A's favour and
      against it at once. C-T23 measured p on two further constructions -- the contemporaneous
      flow-response ratio (~0) and the lagged one (the transition above). So more is measured
      than the frozen text knew; and what is measured shows the single-number reading is the
      one thing it should not be.

Neither is a defect in the design. Both are facts a future reader of a frozen file would
otherwise take from it and carry away wrong.

----------------------------------------------------------------------------------------------
THE REGISTRY. Keyed on OBJECT. Every entry names its definition, its measurement, its owning
sections, and every other object sharing its letter.

The count that matters: SEVEN letters carry FIFTEEN distinct objects in this estate, and
FIVE of the seven carry more than one. And the lane
writing this is implicated -- §478 used `zeta` for the Hill tail exponent of returns, which is
Bouchaud's own usage for that quantity and a THIRD object under a letter C had already found
carrying two.

`delta` deserves separate mention because a contradiction turned on it. CT-016 pitted A's
exponential fill curve against C's power law; C-T22 closed it by showing the axes differ -- A's
`delta` is DEPTH in bps (Cartea Eq. 8.1), C's `x` is QUEUE POSITION. That is a symbol collision
that cost two lanes a day and a register entry. It is the case for this file existing.

Read-only. Builds a registry from measurements already published; measures nothing new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "reports" / "atlas" / "LANE_A_PREREG_V1.md"
OUT_DIR = ROOT / "reports" / "atlas"
OUT_MD = OUT_DIR / "EXPONENT_SYMBOL_REGISTRY_V1.md"
OUT_JSON = OUT_DIR / "EXPONENT_SYMBOL_REGISTRY_V1.json"

REGISTRY = [
    {"object_id": "ZETA_WINDOW_IMBALANCE", "letter": "zeta",
     "definition": "outer-region exponent of R against |dV| over windows of T trades",
     "conditions_on": "net imbalance of ALL participants in a window",
     "measured": "0.416 / 0.439 / 0.495", "owner": "A-S30",
     "note": "A itself holds ZETA_IS_NOT_DELTA"},
    {"object_id": "ZETA_SINGLE_ORDER_SIZE", "letter": "zeta",
     "definition": "R(v,1) = A (v/V_best)^zeta <s>, Bouchaud Eq. 11.7",
     "conditions_on": "ONE market order's size",
     "measured": "0.166 / 0.230 / 0.262 at 600 s; 0.63-0.72 at lag-1 and mechanical",
     "owner": "C-T20", "note": "book range 0-0.3"},
    {"object_id": "ZETA_RETURN_TAIL", "letter": "zeta",
     "definition": "tail exponent of the unconditional return distribution, P(|r|>x) ~ x^-zeta",
     "conditions_on": "nothing -- an unconditional distributional exponent",
     "measured": "Hill 2.33-3.83 across k on 60-minute moves; Bouchaud reports ~3 universally",
     "owner": "section 478", "note": "Bouchaud's own usage; the lane writing this registry "
                                     "used the letter for a third object"},
    {"object_id": "GAMMA_METAORDER_IMPACT", "letter": "gamma",
     "definition": "concavity of price response in metaorder size, I ~ Q^gamma",
     "conditions_on": "a metaorder, requiring child-to-parent identity",
     "measured": "NOT IDENTIFIABLE on anonymised data (Bouchaud 12.2)",
     "owner": "C-T24", "note": "C-T20's 0.373/0.369 was indirect via Eq. 16.16 and withdrawn "
                               "by C-T21; its bias sign is known (aggregate substitution "
                               "underestimates)"},
    {"object_id": "GAMMA_LMF_SIGN_MEMORY", "letter": "gamma",
     "definition": "decay of the order-sign autocorrelation, C(l) ~ l^-gamma_LMF",
     "conditions_on": "trade signs only -- no identity needed",
     "measured": "0.7746 / 0.7892 / 0.2092 (SOL an aggregation artefact)",
     "owner": "C-T24", "note": "alpha_metaorder_size = gamma_LMF + 1 under the LMF model"},
    {"object_id": "DELTA_CASCADE_IMPACT", "letter": "delta",
     "definition": "dP = k Q^delta on cascade episodes",
     "conditions_on": "a whole cascade episode, simultaneous aggregate",
     "measured": "0.684 / 0.666 / 0.696", "owner": "C-T20",
     "note": "A holds DELTA_IS_ASSUMED_NOT_MEASURED and "
             "DELTA_IS_NOT_MEASURABLE_ON_PUBLIC_DATA for its own delta"},
    {"object_id": "DELTA_QUOTE_DEPTH", "letter": "delta",
     "definition": "depth in bps from the mid, the abscissa of Cartea Eq. (8.1) exp(-kappa*delta)",
     "conditions_on": "a price level, not a quantity",
     "measured": "A-S45 fitted kappa ~ 0.0097/bps over an hour",
     "owner": "A-S45", "note": "THIS COLLISION COST A DAY: CT-016 pitted A's exponential "
                               "against C's power law until C-T22 showed A's abscissa is DEPTH "
                               "and C's is QUEUE POSITION"},
    {"object_id": "KAPPA_RESPONSE_T_EXPONENT", "letter": "kappa",
     "definition": "R(dV,T) = R(1) T^kappa F(dV/(V_D T^chi)) -- prefactor exponent in the "
                   "collapsed scaling form",
     "conditions_on": "a scaled imbalance argument held fixed",
     "measured": "not reported alone; enters as kappa-chi = 0.25-0.30",
     "owner": "A-S30, C-T21", "note": "confirmed three times as the DIFFERENCE kappa-chi"},
    {"object_id": "KAPPA_UNCONDITIONAL_RESPONSE", "letter": "kappa",
     "definition": "d log R / d log T with no scaling collapse",
     "conditions_on": "nothing held fixed",
     "measured": "0.6507 / 0.5782 / 0.5209", "owner": "C-T23",
     "note": "NOT the collapsed-scaling kappa; this is why C-T23's kappa-chi differs from "
             "A-S30's and C-T21's"},
    {"object_id": "KAPPA_FILL_DECAY_RATE", "letter": "kappa",
     "definition": "rate constant in Cartea Eq. (8.1), P(fill) = exp(-kappa*delta)",
     "conditions_on": "depth in bps", "measured": "0.0097/bps (A-S45); 0.00956 re-derived "
                                                  "(C-T22)",
     "owner": "A-S45", "note": "a RATE with units 1/bps, not a dimensionless exponent"},
    {"object_id": "CHI_VOLUME_SCALE", "letter": "chi",
     "definition": "exponent of the volume normaliser inside F(dV/(V_D T^chi))",
     "conditions_on": "aggregation window length",
     "measured": "0.6498 / 0.6817 / 0.5902 as sd(dV) ~ T^chi", "owner": "C-T23",
     "note": "p - (kappa-chi) = chi - alpha_E|r| exactly"},
    {"object_id": "P_PREDICTOR_CAPTURE_DECAY", "letter": "p",
     "definition": "f(h) = R(h)/E|r|(h) ~ h^p for a real predictor",
     "conditions_on": "a forecast formed before the horizon",
     "measured": "-0.409 / -0.495 / -0.508", "owner": "A-S40",
     "note": "A-S40 itself: a single fitted p ~ -0.5 is the AVERAGE OF A TRANSITION"},
    {"object_id": "P_CONTEMPORANEOUS_FLOW_RATIO", "letter": "p",
     "definition": "same formula, flow and return measured over the SAME window",
     "conditions_on": "nothing lagged", "measured": "-0.026 / -0.014 / +0.000",
     "owner": "C-T23", "note": "shares A's formula and is not A's object"},
    {"object_id": "P_LAGGED_FLOW_RATIO", "letter": "p",
     "definition": "same formula, signal from a prior window, response measured forward",
     "conditions_on": "a lag",
     "measured": "+0.215/+0.009 at h<=16; -0.721/-0.785 at h>=256", "owner": "C-T23",
     "note": "reproduces A-S40's transition on independent windows"},
    {"object_id": "ALPHA_METAORDER_SIZE_TAIL", "letter": "alpha",
     "definition": "Pareto tail exponent of the metaorder size distribution",
     "conditions_on": "metaorder sizes",
     "measured": "1.775 / 1.789 as UPPER BOUNDS via LMF; book: equities ~1.5, Bitcoin ~1.10",
     "owner": "C-T24", "note": "SOL's 1.209 is an aggregation artefact and must not be "
                               "compared with the book"},
]

AUDIT = {
    "artifact": "reports/atlas/LANE_A_PREREG_V1.md",
    "status_at_audit": "FROZEN",
    "frozen_at": "2026-08-27T03:21:20Z",
    "declared_sha256": "d997e8fb2bd75376edac546e0148ce7d8dcd758c759350fdee08dd9c64a4801b",
    "read_only": True,
    "why_read_only": "charter rule 5 -- a lane may contradict another, never silently overwrite",
    "verdict": "NO_MISUSE_OF_AN_EXPONENT_IN_LANE_A_PREREG",
    "grounds": [
        "the exponent appears in section 10, the FALSIFICATION section, not in the machinery",
        "it is attributed to A-S40",
        "it is scoped by family -- 'for order flow'",
        "the design's claim is explicitly restricted to its own horizon",
    ],
    "two_things_the_frozen_text_cannot_know": [
        {"item": "p is not a constant within order flow either",
         "measured": ("C-T23 on independent windows: +0.215/+0.009 at h<=16, "
                      "-0.721/-0.785 at h>=256"),
         "effect": ("the clause's DIRECTION is right; its MAGNITUDE is unbounded, because near "
                    "the transition the local exponent moves fast. A design at one h* is "
                    "unaffected; a reader inferring a sensitivity from 'p ~ -0.5' is not.")},
        {"item": "'p is unmeasured for every family except order flow' is superseded",
         "measured": ("C-T23 measured p on two further constructions: the contemporaneous "
                      "flow-response ratio (~0) and the lagged one"),
         "effect": ("more is measured than the frozen text knew, and what is measured shows "
                    "the single-number reading is the one thing it should not be")},
    ],
    "not_a_defect": ("neither item is a defect in the design; both are facts a future reader of "
                     "a frozen file would otherwise take from it and carry away wrong"),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def verify_prereg() -> dict:
    if not PREREG.exists():
        return {"present": False}
    raw = PREREG.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    txt = raw.decode("utf-8", "replace")
    m = re.search(r"sha256 of this file\s+([0-9a-f]{64})", txt)
    declared = m.group(1) if m else None
    return {"present": True, "bytes": len(raw),
            "declared_sha256": declared,
            "recomputed_sha256_of_current_bytes": digest,
            "match": declared == digest if declared else None,
            "note": ("a self-referential hash cannot match its own file once written -- the "
                     "declared digest covers the file as it stood before the line was filled "
                     "in. Recorded, not treated as a defect.")}


def build() -> dict:
    letters = {}
    for e in REGISTRY:
        letters.setdefault(e["letter"], []).append(e["object_id"])
    collisions = {k: v for k, v in letters.items() if len(v) > 1}
    for e in REGISTRY:
        e["shares_its_letter_with"] = [o for o in letters[e["letter"]]
                                       if o != e["object_id"]]
    return {
        "study": "C25_SYMBOL_REGISTRY_V1", "lane": "C", "stable_id": "C-T25",
        "generated_utc": _utc(),
        "why": ("three rounds found the same failure -- p carries three objects, zeta two, "
                "gamma two -- and the atlas index cannot see any of it, because both objects "
                "emit tokens containing the same letter. A token index keys on STRINGS; a "
                "collision is a fact about MEANINGS."),
        "keyed_on": "object, never the letter",
        "audit_of_lane_A_prereg": {**AUDIT, "hash_check": verify_prereg()},
        "registry": REGISTRY,
        "letters": letters,
        "collisions": collisions,
        "counts": {"letters": len(letters), "objects": len(REGISTRY),
                   "letters_with_more_than_one_object": len(collisions)},
        "the_case_for_this_file": (
            "CT-016 pitted A's exponential fill curve against C's power law and stayed open a "
            "day. C-T22 closed it by showing the abscissae differ: A's delta is DEPTH in bps "
            "(Cartea Eq. 8.1), C's x is QUEUE POSITION. That was a symbol collision, and it "
            "cost two lanes a day and a register entry."),
        "self_implication": (
            "section 478, written by this lane, used `zeta` for the Hill tail exponent of "
            "returns -- Bouchaud's own usage for that quantity, and a THIRD object under a "
            "letter this lane had already found carrying two."),
        "verdict": "SEVEN_LETTERS_CARRY_FIFTEEN_OBJECTS_FIVE_ARE_OVERLOADED",
        "what_is_NOT_claimed": [
            "That the registry is complete. It covers the exponent family this lane has "
            "measured or read; other families will have their own collisions.",
            "That any measurement changes. Nothing is re-measured here; entries cite the "
            "sections that own them.",
            "That lane A's prereg needs editing. It is frozen and it passes the audit.",
        ],
    }


def render_md(a: dict) -> str:
    au = a["audit_of_lane_A_prereg"]
    L = ["# EXPONENT SYMBOL REGISTRY V1 — keyed on the object, never the letter", "",
         "`{0}` · built {1} · lane C, `{2}`".format(a["verdict"], a["generated_utc"],
                                                    a["stable_id"]), "",
         "> {0}".format(a["why"]), "",
         "**{0} letters carry {1} distinct objects; {2} letters carry more than one.**".format(
             a["counts"]["letters"], a["counts"]["objects"],
             a["counts"]["letters_with_more_than_one_object"]), "",
         "**The case for this file.** {0}".format(a["the_case_for_this_file"]), "",
         "**And this lane is implicated.** {0}".format(a["self_implication"]), "",
         "---", "", "## The registry", "",
         "| letter | object | definition | conditions on | measured | owner | shares its letter with |",
         "|---|---|---|---|---|---|---|"]
    for e in a["registry"]:
        L.append("| `{0}` | **{1}** | {2} | {3} | {4} | {5} | {6} |".format(
            e["letter"], e["object_id"], e["definition"], e["conditions_on"],
            e["measured"], e["owner"],
            ", ".join(e["shares_its_letter_with"]) or "—"))
    L += ["", "### Notes that matter", ""]
    for e in a["registry"]:
        if e.get("note"):
            L.append("- **{0}** — {1}".format(e["object_id"], e["note"]))
    L += ["", "---", "", "## Read-only audit of lane A's frozen preregistration", "",
          "Artifact `{0}`, status **{1}**, frozen {2}. **Read-only:** {3}.".format(
              au["artifact"], au["status_at_audit"], au["frozen_at"], au["why_read_only"]), "",
          "### Verdict: `{0}`".format(au["verdict"]), ""]
    for g in au["grounds"]:
        L.append("- {0}".format(g))
    L += ["", "**C-T23's warning was already satisfied before it was written.**", "",
          "### Two things the frozen text cannot know", ""]
    for t in au["two_things_the_frozen_text_cannot_know"]:
        L += ["**{0}**".format(t["item"]), "", "- measured: {0}".format(t["measured"]),
              "- effect: {0}".format(t["effect"]), ""]
    L += ["> {0}".format(au["not_a_defect"]), "",
          "*Hash check:* {0}".format(au["hash_check"].get("note", "")), "",
          "## What is NOT claimed", ""]
    for x in a["what_is_NOT_claimed"]:
        L.append("- {0}".format(x))
    L += ["", "```verdict", a["verdict"],
          "NO_MISUSE_OF_AN_EXPONENT_IN_LANE_A_PREREG",
          "P_IS_NOT_A_CONSTANT_WITHIN_ORDER_FLOW_EITHER",
          "P_UNMEASURED_EXCEPT_ORDER_FLOW_IS_SUPERSEDED",
          "SYMBOL_COLLISION_IS_INVISIBLE_TO_A_TOKEN_INDEX",
          "ZETA_KAPPA_AND_P_EACH_CARRY_THREE_OBJECTS",
          "CT_016_WAS_A_SYMBOL_COLLISION_AND_IT_COST_A_DAY",
          "THIS_LANE_IS_IMPLICATED_ZETA_HAS_THREE_OBJECTS", "```", ""]
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
                      "audit": a["audit_of_lane_A_prereg"]["verdict"],
                      "hash_check": a["audit_of_lane_A_prereg"]["hash_check"].get("match"),
                      "written": [str(OUT_MD.relative_to(ROOT)).replace("\\", "/"),
                                  str(OUT_JSON.relative_to(ROOT)).replace("\\", "/")]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
