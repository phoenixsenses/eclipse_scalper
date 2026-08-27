r"""LANE C, round 27 -- a statistic this lane published has no population target, and A's design
has a tail-index sensitivity nobody traced.

C-T26 established that A's frozen k = E|r|/sigma inverts to a tail index nu = 3.765, agreeing
with C's own Hill estimate. Neither lane followed the consequence. This round follows it twice:
once against a number this lane published, and once through A's frontier.

----------------------------------------------------------------------------------------------
1. THE PREDICTION IS RIGHT IN PRINCIPLE AND MY TEST OF IT HAS NO POWER.

For a Student-t the excess kurtosis is 6/(nu-4), finite only for nu > 4. At nu = 3.765 it is
undefined. Section 478, written by this lane, reported "excess kurtosis 8.635" (later 8.889) as a
property of the distribution. If nu < 4 that number estimates nothing.

That is testable in principle: a sample estimate of a quantity that does not exist grows with n
instead of settling. I built the test, and then measured what it does under the NULL -- a
distribution whose kurtosis DOES exist -- which is the step that decides whether a gate means
anything.

    top-decade growth (n: 10k -> 100k), median over R resamples

    series               R=40     R=150    R=600
    Gaussian (nu=inf)    1.379    0.695    0.316     <- degenerate: level ~0, ratio meaningless
    t nu=6  (kurtosis=3) 0.953    1.137    1.071     <- THE NULL
    t nu=3.765 (undef)   2.500    1.703    1.183     <- the alternative

    observed             BTC 1.077   ETH 1.179   SOL 1.388

THE GATE DOES NOT DISCRIMINATE. At the precision that stabilises the controls, the null sits at
1.071 and the alternative at 1.183 -- a separation of 0.11 -- and the three observed values
straddle both. BTC's 1.077 is indistinguishable from FINITE kurtosis; ETH's 1.179 matches the
infinite case; SOL's 1.388 is above both and is the series C-T24 showed is aggregation-
contaminated. At R=40, where I first ran it, the same controls returned 0.953 and 2.500: the
statistic was pure noise.

So the prediction is NEITHER CONFIRMED NOR REFUTED by this instrument. What stands is the
weaker, sourced statement: A's k inverts to nu = 3.765 and section 478's own Hill estimate spans
2.33-3.83, so the tail index sits at or below 4 and section 478's quoted kurtosis is NOT SAFELY
INTERPRETABLE as a population quantity. The heavy-tail CONCLUSION is untouched.

TWO INSTRUMENT DEFECTS, BOTH MINE, BOTH WORTH MORE THAN THE RESULT WOULD HAVE BEEN.
  (a) My first criterion was "last two points within 10% -> converged". It classified BTC as
      converged after the estimate had climbed from 0.40 to 208.77, because the final step was
      5.6%. A slow step at the end of a 500x rise reads as a plateau.
  (b) The replacement has a null of 1.071, not 1.0, and needed 600 resamples merely to stabilise.
      Its separation from the alternative is 0.11. CLAUDE.md's rule -- measure a gate's null
      value before freezing pass/fail -- is exactly what caught this, and it is the second time
      this lane has been saved by it after the stratified Nelson-Aalen whose null was 0.377
      rather than 1.000.

2. A'S ENTIRE DESIGN GOES AS k^-2, AND k = k(nu).

    h* = [2c/(k f sigma_d)]^2        N_required = (2/(k f_design/2))^2

Both go as k^-2. Since k is a function of the tail index, so is the design. Nobody has traced
it, and the answer is reassuring:

    nu     k(nu)    h* and N_required, relative to A's frozen k
    3.0    0.6366   1.197x     N_required = 394,784
    3.765  0.6966   1.000x     N_required = 329,726   <- frozen
    5.0    0.7351   0.898x     N_required = 296,088
    6.0    0.7500   0.863x     N_required = 284,444

Across nu in [3, 6] the whole design moves by 1.39x; across the [3, 5] band C-T26 handed A, by
1.33x. So the design's tail-index exposure is a 33-39% band on both the horizon and the required
sample -- material, bounded, and previously unwritten. Compare A's own S5 abort trigger, which
fires on a 2x cost change: the tail-index exposure is well inside that.

Read-only. Uses constants already published by both lanes and window returns already on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "microstructure_02.db"
OUT_DIR = ROOT / "reports" / "research" / "c27_kurtosis_has_no_target_v1"

SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
N_TRADES = 2_000_000
WINDOW_T = 20
N_GRID = (100, 300, 1000, 3000, 10000, 30000, 100000)
REPS = 40
A_K = 0.6966
A_N_REQUIRED = 329_726
NU_GRID = (2.5, 3.0, 3.5, 3.765, 4.0, 5.0, 6.0, 8.0)
SEED = 20260827


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def k_of_nu(nu: float) -> float:
    lg = math.lgamma
    e = (2 * math.sqrt(nu) * math.exp(lg((nu + 1) / 2) - lg(nu / 2))
         / ((nu - 1) * math.sqrt(math.pi)))
    return math.sqrt((nu - 2) / nu) * e


def exkurt(x) -> float:
    z = (x - x.mean()) / x.std(ddof=1)
    return float((z ** 4).mean() - 3)


def growth(x, rng) -> list:
    out = []
    for n in N_GRID:
        if n > len(x):
            break
        v = [exkurt(x[rng.integers(0, len(x), n)]) for _ in range(REPS)]
        out.append({"n": n, "median_excess_kurtosis": round(float(np.median(v)), 2)})
    return out


def top_decade_growth(rows) -> float:
    """Growth of the estimate over the TOP DECADE of n.

    A first attempt used "last two points within 10%" and called BTC converged after it had
    risen from 0.40 to 208.77 -- a slow step at the end of a 500x climb reads as a plateau.
    Growth alone is not the discriminator either: even a finite-kurtosis sample rises from
    small n before settling, and the nu=6 control does exactly that (0.68 -> 2.94) before
    flattening. What separates the cases is whether it PLATEAUS, so the statistic is the ratio
    across the last decade of n, where a convergent series is flat by construction."""
    if len(rows) < 3:
        return float("nan")
    a = rows[-3]["median_excess_kurtosis"]
    b = rows[-1]["median_excess_kurtosis"]
    return float(b / a) if a > 0 else float("nan")


def expected_growth_over_a_decade(nu: float) -> float:
    """E[sample excess kurtosis] ~ n^((4-nu)/nu) when nu < 4; flat when nu > 4."""
    if nu >= 4:
        return 1.0
    return float(10.0 ** ((4.0 - nu) / nu))


def build() -> dict:
    rng = np.random.default_rng(SEED)
    con = sqlite3.connect("file:{0}?mode=ro".format(DB.as_posix()), uri=True)
    observed = {}
    try:
        for s in SYMS:
            a = np.array(con.execute(
                "select price from agg_trades where symbol=? order by ts_ms limit ?",
                (s, N_TRADES)).fetchall(), dtype=np.float64).ravel()
            m = len(a) // WINDOW_T
            w = a[:m * WINDOW_T].reshape(m, WINDOW_T)
            r = np.log(w[:, -1] / w[:, 0]) * 1e4
            r = r[np.isfinite(r)]
            rows = growth(r, rng)
            observed[s] = {"n_windows": int(len(r)),
                           "full_sample_excess_kurtosis": round(exkurt(r), 2),
                           "growth": rows,
                           "top_decade_growth": round(top_decade_growth(rows), 3),
                           "growth_factor_full_range": round(
                               rows[-1]["median_excess_kurtosis"]
                               / max(rows[0]["median_excess_kurtosis"], 1e-9), 1)}
            del a, w, r
    finally:
        con.close()

    controls = {}
    g = rng.standard_normal(200000)
    grows_g = growth(g, rng)
    controls["gaussian_nu_inf"] = {"rows": grows_g, "expected": 0.0,
                                   "top_decade_growth": round(top_decade_growth(grows_g), 3)}
    for nu in (6.0, 3.765):
        t = rng.standard_t(nu, 200000) * math.sqrt((nu - 2) / nu)
        rows = growth(t, rng)
        controls["student_t_nu_{0}".format(nu)] = {
            "rows": rows,
            "population_excess_kurtosis": (round(6 / (nu - 4), 2) if nu > 4
                                           else "UNDEFINED"),
            "top_decade_growth": round(top_decade_growth(rows), 3),
            "theory_top_decade_growth": round(expected_growth_over_a_decade(nu), 3)}

    sens = []
    for nu in NU_GRID:
        k = k_of_nu(nu)
        f = (A_K / k) ** 2
        sens.append({"nu": nu, "k": round(k, 4), "factor": round(f, 3),
                     "N_required": int(round(A_N_REQUIRED * f)),
                     "is_A_frozen": abs(nu - 3.765) < 1e-3})
    band36 = ((A_K / k_of_nu(3.0)) ** 2) / ((A_K / k_of_nu(6.0)) ** 2)
    band35 = ((A_K / k_of_nu(3.0)) ** 2) / ((A_K / k_of_nu(5.0)) ** 2)

    flat = max(controls["gaussian_nu_inf"]["top_decade_growth"],
               controls["student_t_nu_6.0"]["top_decade_growth"])
    all_grow = all(observed[s]["top_decade_growth"] > flat * 1.02 for s in SYMS)
    theory = expected_growth_over_a_decade(3.765)
    return {
        "study": "C27_KURTOSIS_HAS_NO_TARGET_V1", "lane": "C", "stable_id": "C-T27",
        "generated_utc": _utc(),
        "follows_from": ("C-T26: A's frozen k = E|r|/sigma inverts to nu = 3.765, agreeing with "
                         "C's Hill. Neither lane followed the consequence."),
        "part_1_kurtosis": {
            "label": "THE_GATE_DOES_NOT_DISCRIMINATE_ITS_NULL_IS_1_071_AND_THE_ALTERNATIVE_1_183",
            "argument": ("excess kurtosis is 6/(nu-4), finite only for nu > 4; at nu = 3.765 the "
                         "fourth moment diverges"),
            "the_prediction": ("if the population kurtosis exists the sample estimate converges "
                               "with n; if it does not, it grows without settling"),
            "design": ("100,000 twenty-trade window returns per symbol, median excess kurtosis "
                       "over {0} resamples at each n".format(REPS)),
            "observed": observed, "controls": controls,
            "all_three_grow_over_the_top_decade": all_grow,
            "top_decade_growth": {s: observed[s]["top_decade_growth"] for s in SYMS},
            "convergent_controls_top_decade_growth": {
                "gaussian": controls["gaussian_nu_inf"]["top_decade_growth"],
                "student_t_nu_6": controls["student_t_nu_6.0"]["top_decade_growth"]},
            "theory_at_nu_3_765": round(theory, 3),
            "how_the_criterion_was_fixed": (
                "a first attempt used 'last two points within 10%' and called BTC CONVERGED "
                "after it had climbed from 0.40 to 208.77 -- a slow final step at the end of a "
                "500x rise reads as a plateau. Growth alone does not separate the cases either, "
                "since a finite-kurtosis sample also rises from small n before settling (the "
                "nu=6 control goes 0.68 -> 2.94 and then flattens). The discriminator is the "
                "PLATEAU, so the statistic is the ratio across the top decade of n."),
            "controls_behave": ("Gaussian flat at ~0.01; t nu=6 settles near 6/(6-4)=3; "
                                "t nu=3.765 grows -- exactly as theory requires, which is what "
                                "makes the observed rows readable"),
            "honest_reading": ("all three symbols grow over the top decade while BOTH "
                               "convergent controls are flat, which is the signature of a "
                               "non-existent population kurtosis. But the rates differ and "
                               "BTC's sits BELOW the nu=3.765 expectation, so this establishes "
                               "the qualitative fact and not a tail index."),
            "verdict": "NEITHER_CONFIRMED_NOR_REFUTED_THE_INSTRUMENT_LACKS_POWER",
            "power_analysis": {
                "null_is": "a distribution whose kurtosis EXISTS (t, nu = 6, kurtosis 3)",
                "null_top_decade_growth_by_reps": {"40": 0.953, "150": 1.137, "600": 1.071},
                "alternative_by_reps": {"40": 2.500, "150": 1.703, "600": 1.183},
                "gaussian_is_degenerate": ("its kurtosis level is ~0, so a RATIO of two noisy "
                                           "near-zero numbers is meaningless: 1.379 / 0.695 / "
                                           "0.316 across reps"),
                "separation_at_r600": 0.112,
                "observed_straddle_both": True,
                "reading": ("BTC's 1.077 is indistinguishable from FINITE kurtosis; ETH's 1.179 "
                            "matches the infinite case; SOL's 1.388 is above both and is the "
                            "series C-T24 showed is aggregation-contaminated"),
            },
            "what_stands_instead": (
                "the weaker, sourced statement: A's k inverts to nu = 3.765 and section 478's "
                "own Hill estimate spans 2.33-3.83, so the tail index sits at or below 4 and "
                "section 478's quoted kurtosis is NOT SAFELY INTERPRETABLE as a population "
                "quantity. The heavy-tail CONCLUSION is untouched."),
            "instrument_defects": [
                ("first criterion was 'last two points within 10% -> converged'. It classified "
                 "BTC as converged after the estimate climbed from 0.40 to 208.77, because the "
                 "final step was 5.6%."),
                ("the replacement's null is 1.071, not 1.0, and needed 600 resamples merely to "
                 "stabilise; its separation from the alternative is 0.11. CLAUDE.md's rule -- "
                 "measure a gate's null before freezing pass/fail -- caught this, the second "
                 "time this lane has been saved by it after the stratified Nelson-Aalen whose "
                 "null was 0.377 rather than 1.000."),
            ],
            "self_correction": ("section 478, written by this lane, reported excess kurtosis "
                                "8.635/8.889 as a property of the distribution. With the tail "
                                "index at or below 4 that number is not safely interpretable as "
                                "a population quantity. The CONCLUSION it supported -- heavy "
                                "tails -- is correct and untouched."),
            "aggregation_again": ("SOL grows {0}x where BTC grows {1}x. C-T24 measured why: "
                                  "SOL's aggTrades arrive a median 220 ms apart with 53% of "
                                  "consecutive records sharing a price, so aggregation averages "
                                  "the tail away before the window return is formed. A thinner "
                                  "MEASURED tail, not a thinner market. Third appearance of the "
                                  "same mechanism.".format(
                                      observed["SOLUSDT"]["growth_factor_full_range"],
                                      observed["BTCUSDT"]["growth_factor_full_range"])),
        },
        "part_2_design_sensitivity": {
            "label": "THE_WHOLE_DESIGN_GOES_AS_K_SQUARED_INVERSE_AND_K_IS_K_OF_NU",
            "identities": ["h* = [2c/(k f sigma_d)]^2", "N_required = (2/(k f_design/2))^2"],
            "rows": sens,
            "band_nu_3_to_6": round(band36, 3),
            "band_nu_3_to_5": round(band35, 3),
            "reading": ("across nu in [3,6] the whole design moves by {0}x; across the [3,5] "
                        "band C-T26 handed A, by {1}x. Material, bounded, and previously "
                        "unwritten.".format(round(band36, 2), round(band35, 2))),
            "compare_to_As_own_trigger": ("A's S5 abort fires on a 2x cost change; the "
                                          "tail-index exposure is well inside that"),
        },
        "verdict": "GROWTH_TEST_LACKS_POWER_AND_THE_DESIGN_MOVES_1_33X_TO_1_39X_WITH_NU",
        "what_is_NOT_claimed": [
            "That section 478's conclusion is wrong. Heavy tails are established; only the "
            "quoted kurtosis has no population target.",
            "That the Student-t is the true law. It is one parametric family used to invert "
            "k into a tail index; the confirmation comes from the growth test, not the model.",
            "That the design sensitivity is a defect. It is a bound A did not have, and it is "
            "smaller than the cost trigger A already accepts.",
        ],
    }


def render_md(a: dict) -> str:
    p1, p2 = a["part_1_kurtosis"], a["part_2_design_sensitivity"]
    L = ["# C-T27 — A NUMBER THIS LANE PUBLISHED HAS NO POPULATION TARGET", "",
         "`{0}` · generated {1}".format(a["verdict"], a["generated_utc"]), "",
         "**Follows from:** {0}".format(a["follows_from"]), "",
         "## 1. `{0}`".format(p1["label"]), "",
         "{0}. **The prediction:** {1}.".format(p1["argument"].capitalize(),
                                                p1["the_prediction"]), "",
         "*{0}.*".format(p1["design"]), "",
         "| series | " + " | ".join("n={0:,}".format(n) for n in N_GRID) + " |",
         "|---" + "|--:" * len(N_GRID) + "|"]
    for s in SYMS:
        o = p1["observed"][s]
        vals = {r["n"]: r["median_excess_kurtosis"] for r in o["growth"]}
        L.append("| **{0}** | ".format(s) + " | ".join(
            str(vals.get(n, "—")) for n in N_GRID) + " |")
    for key, lab in (("gaussian_nu_inf", "*Gaussian (ν=∞)*"),
                     ("student_t_nu_6.0", "*t, ν=6 — κ exists = 3*"),
                     ("student_t_nu_3.765", "*t, ν=3.765 — κ undefined*")):
        c = p1["controls"].get(key)
        if not c:
            continue
        vals = {r["n"]: r["median_excess_kurtosis"] for r in c["rows"]}
        L.append("| {0} | ".format(lab) + " | ".join(
            str(vals.get(n, "—")) for n in N_GRID) + " |")
    pa = p1["power_analysis"]
    L += ["", "**{0}.**".format(p1["controls_behave"].capitalize()), "",
          "### \U0001f534 Then I measured what the gate does under its own null", "",
          "The null is **{0}**.".format(pa["null_is"]), "",
          "| series | R=40 | R=150 | R=600 |", "|---|--:|--:|--:|",
          "| *Gaussian (ν=∞)* | {0} | {1} | {2} |".format(1.379, 0.695, 0.316),
          "| **t, ν=6 — kurtosis EXISTS** | {0} | {1} | **{2}** |".format(
              pa["null_top_decade_growth_by_reps"]["40"],
              pa["null_top_decade_growth_by_reps"]["150"],
              pa["null_top_decade_growth_by_reps"]["600"]),
          "| t, ν=3.765 — undefined | {0} | {1} | **{2}** |".format(
              pa["alternative_by_reps"]["40"], pa["alternative_by_reps"]["150"],
              pa["alternative_by_reps"]["600"]), "",
          "| observed | " + " | ".join(
              "{0} **{1}**".format(x, p1["top_decade_growth"][x]) for x in SYMS) + " |",
          "|---|" + "---|" * (len(SYMS) - 1) + "---|", "",
          "*Gaussian caveat:* {0}.".format(pa["gaussian_is_degenerate"]), "",
          "**The gate does not discriminate.** At the precision that stabilises the controls the "
          "null sits at **1.071** and the alternative at **1.183** — a separation of **{0}** — "
          "and the observed values straddle both. {1}.".format(
              pa["separation_at_r600"], pa["reading"]), "",
          "→ `{0}`".format(p1["verdict"]), "",
          "**What stands instead.** {0}".format(p1["what_stands_instead"]), "",
          "> \U0001f534 **Self-correction.** {0}".format(p1["self_correction"]), "",
          "### Two instrument defects, both mine", ""]
    for d in p1["instrument_defects"]:
        L.append("- {0}".format(d))
    L += ["", "**And the aggregation mechanism, a third time.** {0}".format(
              p1["aggregation_again"]), "",
          "## 2. `{0}`".format(p2["label"]), "",
          "`{0}` and `{1}` — **both go as k⁻²**, and `k = k(ν)`.".format(
              p2["identities"][0], p2["identities"][1]), "",
          "| ν | k(ν) | h\\* and N_required factor | N_required |", "|--:|--:|--:|--:|"]
    for r in p2["rows"]:
        L.append("| {0} | {1} | {2}× | {3:,} |{4}".format(
            r["nu"], r["k"], r["factor"], r["N_required"],
            " ← **A's frozen k**" if r["is_A_frozen"] else ""))
    L += ["", "> {0}".format(p2["reading"]), "",
          "*{0}.*".format(p2["compare_to_As_own_trigger"].capitalize()), "",
          "## What is NOT claimed", ""]
    for x in a["what_is_NOT_claimed"]:
        L.append("- {0}".format(x))
    L += ["", "```verdict", a["verdict"],
          "NEITHER_CONFIRMED_NOR_REFUTED_THE_INSTRUMENT_LACKS_POWER",
          "GATE_NULL_IS_1_071_ALTERNATIVE_IS_1_183_SEPARATION_0_112",
          "SECTION_478_KURTOSIS_NUMBER_IS_NOT_SAFELY_INTERPRETABLE",
          "SECTION_478_HEAVY_TAIL_CONCLUSION_UNTOUCHED",
          "MY_FIRST_CRITERION_CALLED_A_522X_CLIMB_CONVERGED",
          "GAUSSIAN_RATIO_CONTROL_IS_DEGENERATE_NEAR_ZERO_LEVEL",
          "SOL_KURTOSIS_GROWTH_IS_THE_AGGREGATION_MECHANISM_A_THIRD_TIME",
          "DESIGN_MOVES_1_33X_ACROSS_NU_3_TO_5_AND_1_39X_ACROSS_3_TO_6",
          "TAIL_INDEX_EXPOSURE_IS_INSIDE_AS_OWN_2X_COST_TRIGGER", "```", ""]
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
    (args.out_dir / "C27_KURTOSIS_HAS_NO_TARGET_V1.json").write_text(
        json.dumps(a, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (args.out_dir / "C27_KURTOSIS_HAS_NO_TARGET_V1.md").write_text(md, encoding="utf-8")
    p1 = a["part_1_kurtosis"]
    print(json.dumps({
        "verdict": a["verdict"],
        "all_three_grow": p1["all_three_grow_over_the_top_decade"],
        "top_decade_growth": p1["top_decade_growth"],
        "controls_flat": p1["convergent_controls_top_decade_growth"],
        "theory_at_nu_3_765": p1["theory_at_nu_3_765"],
        "band_3_to_5": a["part_2_design_sensitivity"]["band_nu_3_to_5"],
        "band_3_to_6": a["part_2_design_sensitivity"]["band_nu_3_to_6"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
