# D-E2 — A SPELL IS A DETECTOR SETTING, AND THE DURATION IS A FIRST PASSAGE TIME

**Lane D · study `D-E2` · 2026-08-27 · read-only, outcome-blind · corpus round**

Continues `D-E1`. Two texts read in full for the first time on this line — **Honoré (1993)**, all
10 pages, and **ABG chapter 10, "First passage time models: Understanding the shape of the hazard
rate"** — plus the measurement they forced.

Artifacts: `reports/atlas/D_E2_SPELL_PROCESS_V1.json` · `tools/d_e2_spell_process_audit_v1.py`
(reads `sym`, `t0`, `q` only; the `imp_*` and `pre_bps` columns sit in the same pickles and are
deliberately not read). **No threshold is selected. No duration is estimated.**

---

## 1. Honoré, read in full: the identification menu is three assumption sets, not one

`S101` (`§437`) discharged two MPH conditions **by theorem** on Honoré's authority. Reading the
paper end to end shows the discharge is real and **narrower than it was published**.

| | model | what it needs | what it costs |
|---|---|---|---|
| **Thm 1** | 2 spells, **shared `θ`**, hazards `Z'_i(t)·θ` | differentiable, non-constant `Z_i`. **No covariates.** | **nothing at all about `G`** — no moment condition, no tail condition |
| **Thm 2** | separate `(θ₁,θ₂)`, joint `G` | ER1+ER2+ER3 on **each** margin | `E[θᵢ]=1`; `G` then follows from uniqueness of the multi-dim Laplace transform |
| **Thm 3** | **lagged duration dependence** — `Z₂(t₂;t₁)`, `φ₂(x;t₁)` | 1), 2) **and one of** 3a / 3b / 3c | **every branch puts assumptions back on `G`** |

Theorem 3's three branches, in full, because the prereg has to pick one:
- **3a** `θ₂ = R(θ₁)` for a **known** `R`, `E[R(θ₁)θ₂] < ∞`, `E[θ₁] = 1`.
- **3b** `θ₂ ⟂ θ₁`, `E[θ₁] = E[θ₂] = 1`. *(Note this contradicts shared frailty.)*
- **3c** `Z₂(t₂;t₁)` does not depend on `t₁` **and** `φ₂(x;t₁) = φ₂(x)·h(t₁)` — lagged duration
  enters only as a **multiplicative scale**, never as a shape change — plus `E[θ₁]=1`,
  `E[θ₁θ₂] < ∞`, `P(θ₁ > 0) = 1`. Honoré calls this assumption "strong" himself.

Two further details that matter here and are easy to miss:
- **ER3 requires the covariate support to be an OPEN SET** in `ℜᵏ` with `φ` non-constant on it. So
  Theorems 2 and 3 need continuous covariate variation; **Theorem 1 needs none.**
- **§3, time-varying covariates.** A covariate that makes a **discrete jump at a fixed time `t*`**
  identifies the scale `α` that Ridder (1990) proved is otherwise unidentified — but Honoré warns
  "some kinds of time–varying covariates (such as **time trends**) can **ruin** identification."

> **So the discharge is Theorem 1's, and Theorem 1 is conditional on there being no lagged duration
> dependence.** Under Theorem 3 the moment condition returns in all three branches. `S101`'s
> `DISCHARGED_BY_THEOREM` is therefore exactly as strong as the lagged-dependence measurement behind
> it — and that measurement turns out to have been made on a different population than the published
> result it is used to license.

---

## 2. Two samples, one name — and it is one population with one threshold

Both are called *"the forced-flow episode sample"*. Both span the same 24 UTC days and the same three
symbols. **Proved by set comparison, not inferred:**

```
_h1_deep.pkl  ==  _s97_extended.pkl  filtered at  q >= $50,000
629 of 1,271 · subset check exact · q byte-identical on all 629 · 0 rows either way
```

| | `S101` / `§437` (`_s97_extended`) | H2 `§426` + `D-E1` (`_h1_deep`) |
|---|--:|--:|
| episodes | **1 271** | **629** |
| `q` min / median | $2 009 / $47 558 | $50 384 / **$302 223** (**6.35×**) |
| spells per unit (BTC/ETH/SOL) | 457 / 416 / 395 | 256 / 232 / 138 |
| **median inter-episode gap** | **60.5 min** | **109.5 min** |

**Replication first, so the comparison is of populations and not of estimators.** My per-symbol
lagged-dependence regression on the 1 271 sample reproduces `S101` to four decimals — BTC `+0.06505`
(z 1.390), ETH `−0.00679` (z −0.138), SOL `+0.00189` (z 0.037); `S101`'s published `MDE 0.141` is
SOL's. Same estimator, confirmed.

**On the gated population the same estimator gives different numbers:**

| | β | z | MDE₈₀ |
|---|--:|--:|--:|
| BTC | **+0.1262** | **+2.02** | 0.175 |
| ETH | +0.0628 | +0.95 | 0.185 |
| SOL | +0.0136 | +0.16 | 0.241 |
| pooled within-symbol | +0.0734 | +1.84 | 0.112 |
| *(same, on `S101`'s population)* | *+0.0204* | *+0.73* | *0.079* |

**The fair reading, stated before the use.** `z = +2.02` is one of three symbols with no multiplicity
control; Holm over three gives `p ≈ 0.129`. **This does not establish lagged duration dependence.**
What it establishes is that **the evidence for its absence is population-dependent**, and the
population was fixed by a threshold nobody has justified.

---

## 3. The whole family: the spell process is a detector setting

Sweep of the notional floor on the **superset**, so every row is the same feed, the same 24 days and
the same three symbols. **No threshold is selected; this is a sensitivity analysis of a measurement.**

```
floor $      n   median gap   p25    CIF@60m  CIF@240m    LDD beta      z    MDE80
      0   1271     60.5 min   38.6     0.495    0.963      +0.0204   0.73   0.0787
  5,000   1093     69.2       45.1     0.416    0.945      +0.0485   1.60   0.0848
 10,000    952     78.4       49.0     0.357    0.919      +0.0111   0.34   0.0910
 25,000    759     92.4       56.7     0.274    0.868      -0.0017  -0.05   0.1020
 50,000    629    109.5       63.5     0.223    0.811      +0.0734   1.84   0.1120   <- H2 / D-E1
100,000    506    131.1       73.2     0.168    0.755      +0.0332   0.74   0.1254
250,000    353    165.1       92.3     0.091    0.632      +0.0104   0.19   0.1516
500,000    225    243.8      127.5     0.044    0.484      +0.0317   0.46   0.1934
```

**The median inter-episode gap runs 60.5 → 243.8 minutes — 4.0×, monotone in one threshold.**
`D-E1`'s headline 109.5 minutes is one point on that curve.
`A_SPELL_IS_A_DETECTOR_SETTING_NOT_A_MARKET_OBJECT`.

Two more readings of the same table:
- The competing-event CIF at 240 m runs **96.3% → 48.4%**. So `D-E1`'s "81% contaminated" is the
  $50k reading — but **the direction of A-S50's problem survives at every floor**: contamination at
  240 minutes never falls below **48%**, even when only the largest 225 episodes are kept.
- `MDE₈₀` grows **monotonically** 0.079 → 0.193 while β wanders **non-monotonically** with no sign
  structure (`−0.002` to `+0.073`). **No floor gives a family-wise significant lagged dependence, and
  no floor gives a null tight enough to assert its absence at the small end.** That is the honest
  state of Honoré's Theorem-1 premise on this estate.

---

## 4. And the corpus names what this duration actually IS — ABG chapter 10, never opened here

**"First passage time models: Understanding the shape of the hazard rate."** A forced-flow edge ends
when a price path reaches a level. That is a **first hitting time**, and ABG gives it in closed form:
a Wiener process starting at `c > 0` with drift `−μ` and diffusion `σ`, absorbed at `0`, has hitting
time `T` distributed **inverse Gaussian**,

```
f(t) = (c/σ)·√(2π)⁻¹·t^(−3/2)·exp[ −(c−μt)² / (2σ²t) ]                              (ABG 10.2)
S(t) = Φ((c−μt)/(σ√t)) − exp(2cμ/σ²)·Φ((−c−μt)/(σ√t))
```

and — the point — *"the distribution only depends on these through the functions `c/σ` and `μ/σ`.
Hence, from a statistical point of view, there are only **two free parameters**."*

Four consequences, each landing on something already published on this estate:

**(a) Three hazard shapes out of one mechanism, so a shape is not a finding.**
*"If `c` is close to zero relative to the quasi-stationary distribution, one essentially gets a
decreasing hazard rate; a value of `c` far from zero gives essentially an increasing hazard rate; an
intermediate value of `c` yields a hazard that first increases and then decreases."* Strictly, the IG
hazard **always rises to a maximum and then declines**. So *"the edge builds and then decays"* is the
**default output of a one-parameter distance-to-barrier**, not evidence about alpha.

**(b) The distribution can be defective — so a half-life may fail to exist mechanically, not merely
statistically.** With drift **away** from the barrier, *"It will always be the case that
`0 < P(T < ∞) < 1`, so there is a positive probability that the individual will never experience an
event"* — ABG's **cure model**. H2 published `PEAK_NOT_OBSERVED_WITHIN_SUPPORTED_WINDOW` with the
mean still rising at 240 m, which is that signature. **You cannot halve a survival function that
never reaches zero.** This is a **third, mechanical** reason `A-S50`'s scalar has no referent,
independent of `D-E1`'s two statistical ones.

**(c) The last open MPH condition is predicted to fail for this class of duration.** ABG, comparing
IG hazards across groups: *"The hazard comparisons given here thus clearly suggest that **proportional
hazards models would not provide an appropriate description of the covariate effects, particularly
not at the earliest part of the time scale**."* `§437` records `p5_ph_compatible` as the one
`PROTECTED` condition still genuinely blocked. The corpus does not unblock it — it says **do not
expect it to pass** if the duration is a first passage time.

And the trap underneath it: a **declining hazard ratio** between a big-episode and a small-episode
group is **automatic** when the groups differ in `c`, because *"convergence toward a quasi-stationary
distribution implies that the relative hazards decline toward a ratio of one."* ABG names it as the
same phenomenon as frailty's declining relative risk (§6.5) — which is the same family as the false
protectivity (§6.6) `D-E1` already pre-registered against. **Three published warnings, one mechanism.**

**(d) Quasi-stationarity kills mechanism claims from hazard shape.**
*"An approximately constant hazard rate will be a common phenomenon for many models due to
convergence to quasi-stationarity. Hence the development of risk at the level of an individual … is
very hard to deduce, and it therefore appears difficult to draw conclusions about the underlying
process with any degree of certainty."* This is the duration analogue of `D-E1` §1:
**the hazard shape does not identify the mechanism.**

**The positive half, and it is the largest thing this round found.** This is a *model*, with **two**
parameters, and the estate already measures both: `c` is the distance from the anchor to the exit
level, and `μ/σ` is drift over volatility — §311/§315 measured the drift and §426 measured
`pre_realised_vol_60m` on the same 629 episodes. **An inverse-Gaussian first-passage specification is
writable today**, and it would be the first duration object on this line with a mechanism behind it
instead of a curve through it. ABG ch.10 was in the 1,353 pages the charter said had zero uses; this
is the chapter that was worth opening.

---

## 5. What this does to D-E1 (nothing withdrawn; two things become conditional)

- `INTER_EPISODE_MEDIAN_109_5_MINUTES_SAME_SYMBOL` → **at the $50 000 floor.** Family: 60.5–243.8 min.
- `A_S50_240M_DURATION_BOUND_RESTS_ON_81PCT_CONTAMINATED_WINDOWS` → **at the $50 000 floor.** 96.3% at
  no floor, 48.4% at $500k. The verdict's **direction is floor-independent**; its number is not.
- `FRAILTY_IDENTIFICATION_BARRIER_DOES_NOT_BIND_SUPPLY_DOES` is **duplicated work.** `S101` / `§437`
  established it first, on the ungated sample, with the same estimator. I did not find it before
  publishing, and the shared log exists precisely to prevent that. What `D-E2` adds is not the
  finding but its **boundary**: the discharge is Theorem 1's, and Theorem 1's premise is weaker on
  the population that carries the published result.

## 6. Fixed for D-E3

1. **Any duration statement must name its notional floor**, and publish the sweep, or it is not
   interpretable (§3).
2. **The event model is a first passage time**, and the specification to beat is inverse Gaussian
   with `(c/σ, μ/σ)` (§4). A non-parametric hazard shape may **not** be read as mechanism.
3. **Proportional hazards is a rival to be tested, not a default** — the corpus predicts against it
   here (§4c), and a declining hazard ratio is the null, not the signal.
4. **Honoré's branch must be declared:** Thm 1 (needs LDD ≈ 0, buys freedom from `G`) or Thm 3 with
   3a / 3b / 3c named. The LDD evidence is `z ≤ 2.02` on one symbol, family-wise null, MDE 0.079–0.193
   — so the branch is a **declaration**, not a measurement, and must be labelled one.
5. **Check for a defective distribution before quoting any quantile** (§4b).

## 7. Open, and owed to the operator

- **The $50 000 floor has no recorded justification.** It defines the population of the estate's only
  cost-clearing result. D does not propose changing it — changing it would be a new family with its
  own multiplicity budget — but the sweep is now on record and the choice should be **declared**
  rather than inherited.
- **The one identification instrument the corpus points at**, Honoré §3's discrete-jump time-varying
  covariate at a fixed `t*`: this venue has one that is exogenous and calendar-fixed — **funding
  settlement at 00:00 / 08:00 / 16:00 UTC**. Naming it as an identification resource is not a
  hypothesis; testing it would be. **Not opened.**

```verdict
D_E2_HONORE_READ_IN_FULL_AND_ABG_CHAPTER_10_OPENED_FOR_THE_FIRST_TIME
HONORE_DISCHARGE_IS_THEOREM_1_AND_THEOREM_1_REQUIRES_NO_LAGGED_DURATION_DEPENDENCE
THEOREM_3_PUTS_THE_MOMENT_CONDITION_BACK_IN_ALL_THREE_BRANCHES_3A_3B_3C
BRANCH_3B_INDEPENDENT_FRAILTIES_CONTRADICTS_SHARED_FRAILTY
ER3_NEEDS_AN_OPEN_COVARIATE_SUPPORT_THEOREM_1_NEEDS_NO_COVARIATES_AT_ALL
TWO_SAMPLES_ONE_NAME_PROVED_BY_SET_COMPARISON_NOT_INFERRED
H1_DEEP_IS_S97_EXTENDED_FILTERED_AT_FIFTY_THOUSAND_DOLLARS_629_OF_1271
MEDIAN_EPISODE_SIZE_DIFFERS_6_35X_BETWEEN_THE_TWO_POPULATIONS
S101_PER_SYMBOL_LDD_REPRODUCED_TO_FOUR_DECIMALS_SAME_ESTIMATOR
ON_THE_GATED_POPULATION_BTC_LDD_IS_PLUS_0_1262_z_PLUS_2_02
HOLM_OVER_THREE_SYMBOLS_GIVES_P_0_129_SO_LDD_IS_NOT_ESTABLISHED
WHAT_IS_ESTABLISHED_IS_THAT_THE_EVIDENCE_FOR_ITS_ABSENCE_IS_POPULATION_DEPENDENT
A_SPELL_IS_A_DETECTOR_SETTING_NOT_A_MARKET_OBJECT
MEDIAN_INTER_EPISODE_GAP_RUNS_60_5_TO_243_8_MINUTES_ACROSS_THE_FLOOR_SWEEP
CIF_AT_240M_RUNS_96_3_TO_48_4_PERCENT_BUT_NEVER_BELOW_48_PERCENT
A_S50_CONTAMINATION_DIRECTION_IS_FLOOR_INDEPENDENT_ITS_NUMBER_IS_NOT
MDE_GROWS_MONOTONICALLY_WHILE_BETA_WANDERS_WITH_NO_SIGN_STRUCTURE
NO_FLOOR_ESTABLISHES_LDD_AND_NO_FLOOR_ESTABLISHES_ITS_ABSENCE_AT_THE_SMALL_END
THE_DURATION_IS_A_FIRST_PASSAGE_TIME_AND_ABG_CH10_GIVES_IT_IN_CLOSED_FORM
INVERSE_GAUSSIAN_HAS_ONLY_TWO_FREE_PARAMETERS_C_OVER_SIGMA_AND_MU_OVER_SIGMA
RISING_THEN_FALLING_HAZARD_IS_THE_DEFAULT_OUTPUT_NOT_A_FINDING
DRIFT_AWAY_FROM_THE_BARRIER_GIVES_A_DEFECTIVE_DISTRIBUTION_A_CURE_MODEL
H2_PEAK_NOT_OBSERVED_IS_THAT_SIGNATURE_SO_A_HALF_LIFE_MAY_NOT_EXIST_MECHANICALLY
THIRD_INDEPENDENT_REASON_A_S50_SCALAR_HAS_NO_REFERENT_AND_THIS_ONE_IS_MECHANICAL
P5_PH_COMPATIBLE_IS_PREDICTED_AGAINST_BY_THE_CORPUS_FOR_FIRST_PASSAGE_DURATIONS
A_DECLINING_HAZARD_RATIO_IS_THE_NULL_WHEN_GROUPS_DIFFER_IN_DISTANCE_TO_BARRIER
QUASI_STATIONARITY_MEANS_HAZARD_SHAPE_DOES_NOT_IDENTIFY_MECHANISM
INVERSE_GAUSSIAN_SPECIFICATION_IS_WRITABLE_TODAY_BOTH_PARAMETERS_ALREADY_MEASURED
FRAILTY_BARRIER_FINDING_WAS_DUPLICATED_WORK_S101_GOT_THERE_FIRST_I_MISSED_IT
FIFTY_THOUSAND_DOLLAR_FLOOR_HAS_NO_RECORDED_JUSTIFICATION_DECLARED_NOT_CHANGED
FUNDING_SETTLEMENT_IS_THE_ONE_DISCRETE_JUMP_INSTRUMENT_HONORE_SECTION_3_ASKS_FOR
NOT_OPENED_NO_THRESHOLD_SELECTED_NO_DURATION_ESTIMATED
READ_ONLY_OUTCOME_BLIND_NO_ORDERS
IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW
```
