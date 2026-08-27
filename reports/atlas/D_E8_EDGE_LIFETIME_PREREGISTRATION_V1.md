# D-E8 — PREREGISTRATION: THE EDGE LIFETIME OF A FORCED-FLOW EPISODE

**Lane D · study `D-E8` · 2026-08-27 · FROZEN BEFORE ANY OUTCOME IS READ**

This document is the freeze. **No price, return or outcome column has been opened by lane D in any
of D-E1…D-E7**, and none is opened here. Everything below is fixed now so that Chan's disclosed
look-ahead — *"in-sample data to find the half-life and therefore the lookback"* — cannot enter
through the back door when this is executed.

An independent reader must be able to run this without asking lane D anything. If a choice below is
ambiguous, that is a defect in this document, not a licence to choose.

---

## 0. What is inherited, not re-derived

| inherited | from | what it fixes here |
|---|---|---|
| `Y_i(t)` and ABG **delayed entry** | `S125` / §471 | the at-risk indicator; 38.0% of the start-to-start risk set cannot fail and the error is size-confounded (r = +0.5212) |
| **end-to-start** clock for the *recurrence* hazard | `S125` / `S126` / §471–472, confirmed by `D-E7` | which clock the recurrence process is measured on |
| **symbol-day** as the frailty stratum | `S127` / §473 | 72 strata, median 18 spells; within-stratum lagged dependence −0.0012 |
| Honoré **Theorem 1** branch | `S127` + `D-E2` | multi-spell, shared θ within symbol-day ⟹ **no assumption on the mixing distribution** |
| the competing-event CIF | `D-E1` / `D-E2` / `D-E4` | 0% ≤15 m by construction, 22.26% @60 m, 55.17% @120 m at the `$50k` floor |
| the independence unit is **not the symbol** | `D-E4` | symbols co-fire at 6.2× chance within ±1 minute |

**`D-E4`'s dead-time-Poisson verdict is withdrawn (`D-E7`) and is NOT inherited.**

---

## 1. Population — named by artifact path, with its own defect stated

```
episode set   data/pve_01_v1/_s97_extended.pkl        1,271 episodes, 24 UTC days, BTC/ETH/SOL
declared floor  q >= $50,000  ->  629 episodes         == data/pve_01_v1/_h1_deep.pkl exactly
secondary       q >= $0       ->  1,271 episodes        reported alongside, never instead
```

**Known defect, carried openly (`D-E7`):** the 1,271-episode set **cannot be rebuilt from its own
documented rule**. Applying `liquidations`, gap `> 900 s`, inside the published window yields
**1,808** episodes, of which the published set is a **strict subset** — every published episode is
present and 537 small ones are absent. At the `$50k` floor the rebuild matches to a single episode.
So the **primary analysis runs at `$50k`**, where the population is reproducible, and the `$0` arm
is reported as a sensitivity carrying this caveat.

---

## 2. Unit, time zero, time scale — declared, because three books asked

- **Unit.** One episode = one observation. `Y_i(t) = 1` from `t = 0` (the anchor) until the first
  terminating cause below. There is no delayed entry on this clock — see the clock note.
- **Time zero.** `t0` = the **first liquidation timestamp of the episode**. This is Hernán & Robins'
  option **(a) the first eligible time**, chosen because it is the first instant a decision could
  be taken. Options (b) random eligible time and (c) every eligible time are **rejected here** and
  the rejection is recorded rather than left implicit.
- **Time scale.** Wall-clock milliseconds from `t0`. Not volume time, not trade time. (`C-T41`
  found volume time minimises drift for *price* scaling; that is a different estimand and does not
  transfer.)
- **Time unit / eligibility granularity.** The `900 000 ms` episode gap rule. H&R warn a coarse
  time unit introduces bias; 15 minutes is coarse and it is the detector's, not a choice made here.

**The clock note, and it is the point `D-E7` cost a withdrawal to learn.**
`S125`'s end-to-start correction applies to the **recurrence hazard** — how long until the *next
episode*. It does **not** apply to the **edge lifetime**, whose window genuinely begins at `t0`
because that is when a trader could act. **Two processes, two clocks, both named here.** Any
statement produced by this prereg carries its clock in its label.

---

## 3. The event — "the edge ends"

The edge is **ALIVE** at time `t` when the direction-signed return from the reference price is at
or above the declared cost floor:

```
r_signed(t) = d * ( P(t) / P_ref - 1 ) * 1e4      bps,   d in {+1,-1} from the episode's SELL share
P_ref       = the last mark strictly before t0        (H2's own reference, reused unchanged)
ALIVE(t)   <=>  r_signed(t) >= k
k           = 10.0 bps                                round-trip taker, CLAUDE.md canonical
                                                      BINANCE_BASE (5.0 bps/side)
```

**`T_1` = the end of the FIRST alive-spell**: the first `t` at which `ALIVE` is true, followed by
the first later `t` at which it is false. That second time is the event `EDGE_GONE`.

- If the episode is **never alive** within `τ`, it is recorded as `NEVER_ALIVE` with lifetime `0`.
  This is a **category, not a censoring**, and its share is reported first, before any curve.
- The barrier `k` is **economic, not fitted**. It is not chosen by looking at any outcome.
- **Sensitivity, declared now:** `k = 4.0` bps (maker round-trip) and `k = 0.0` bps. Reported
  alongside, never instead. **No other `k` may be introduced after execution.**

---

## 4. Competing risks — enumerated and estimated cause-specific

| cause | definition | status |
|---|---|---|
| **1 `EDGE_GONE`** | as §3 | the event of interest |
| **2 `INTERRUPTED`** | the next same-symbol episode's `t0` arrives | competing, measured; **start-to-start** by construction, because the window begins at `t0` |
| **3 `ADMINISTRATIVE`** | `τ` reached, or the price series ends, or the lawful cutoff | censoring; **type I, independent** (`D-E1`, ABG §2.2.8) |
| **4 `SLIP_DROPPED`** | a horizon read whose mark timestamp exceeds its target by **> 60 s** | the observation is **dropped and counted**, never silently walked forward (`D-E1` §3c: 19 of 5,032 cells exceeded 60 s, worst 5.18 h) |

**Estimator: Aalen-Johansen** for `CIF_1` and `CIF_2` (ABG §3.4.1, eqs 3.67–3.68).
**Forbidden, explicitly:** `1 − KM` with cause 2 treated as censoring. ABG calls that reading
*"quite speculative"*, and `D-E1` recorded why: the latent marginal is **not point-identified at any
`N`** (STK4080 Slides 9 p.6/28), and its bounds (Peterson 1975) are **not on this shelf**.

---

## 5. The estimand — one scalar, and it is identified

```
mu_tau  =  integral_0^tau  P00(u) du          P00 = still ALIVE and not yet INTERRUPTED
tau     =  60 minutes
```

`P00` is the Aalen-Johansen "still in state 0" probability, so `mu_tau` is the **expected number of
minutes the edge is both alive and un-interrupted inside `[0, τ]`**, in the world as it is. That is
identified under the type-I censoring `D-E1` certified, and it is exactly what a capacity bound
consumes:

```
X  =  ADV * POV * mu_tau
```

**`τ = 60 minutes` is fixed by a rule, not by a look**: the largest horizon on the published grid at
which the measured competing-event CIF at the declared floor is below **25%** — 22.26% at 60 m,
55.17% at 120 m (`D-E1`, outcome-blind). The rule and the value are both frozen here.

**Not the estimand, and forbidden as an output:** a half-life · a median lifetime · any marginal
`E[min(T_1, τ)]` in a world where episodes do not recur · a "time to peak" from the H2 response
curve.

---

## 6. Inference

- **Clustering: symbol-day** (`S127`'s stratum, 72 strata, median 18 spells). **Not symbol** —
  `D-E4` measured the three symbols co-firing at 6.2× chance within ±1 minute, so three symbols are
  not three independent panels. Day-level clustering reported alongside.
- **Frailty**: Honoré Theorem 1 branch is **declared**, not measured — multi-spell with shared θ
  within symbol-day, no lagged duration dependence (within-stratum −0.0012, `S127`). Under Theorem 1
  **no assumption about the mixing distribution is required**. If an executor finds within-stratum
  lagged dependence materially non-zero, it must switch to Theorem 3 and **name the branch
  (3a/3b/3c)** before estimating.
- **Every null is calibrated before its test is read.** The null world is: constant individual
  hazard, the 900 s dead time as built, no frailty, spans resampled from the data (`D-E7`'s
  calibrator, reusable).
- **Multiplicity**: two primary tests (§7), Holm. Sensitivities spend no alpha.

---

## 7. Primary tests, and what each would show

| id | test | null | classification |
|---|---|---|---|
| **P1** | `mu_tau` at the declared floor and `k = 10` bps | — | estimation, not a test; reported with its cluster-robust CI |
| **P2** | `CIF_1(τ)` vs the calibrated null | edge ends at the null rate | `EDGE_DECAY_FASTER_THAN_NULL` / `NOT_DISTINGUISHABLE` |
| **P3** | proportional hazards on `log(Q/ADV)` for cause 1 | PH holds | **PH is a RIVAL, not a default.** ABG §10.3.2 predicts against it for first-passage durations *"particularly not at the earliest part of the time scale"* |

**Pre-registered rivals for any negative covariate sign on a hazard** — all three must be named in
the result, not discovered afterwards:
1. **False protectivity**, ABG §6.6 eq (6.28) — correlated frailties across competing risks.
2. **Distance-to-barrier**, ABG §10.3.2 — a declining hazard *ratio* is automatic when groups differ
   in `c`; quasi-stationarity drives the ratio toward 1.
3. **Crossover by frailty selection**, ABG §6.5.2 eq (6.23) — after an effect stops, the treated
   group's population hazard rises **above** control. ABG ties it to Simpson's paradox.

**The specification to beat**: inverse-Gaussian first passage, ABG §10.3.1 eq (10.2), two free
parameters `c/σ` and `μ/σ`. A non-parametric hazard shape **may not be read as mechanism** —
quasi-stationarity means many processes converge to the same limiting hazard.

**Before quoting any quantile**, report `P00(τ)`. If the distribution is defective — `Ŝ` constant and
positive at large `t`, ABG's cure-model case — **no median and no half-life may be quoted at all**.

---

## 8. Success, failure, stop rule

**Success.** `mu_tau` published with: the cause-specific CIFs, the `NEVER_ALIVE` share, `P00(τ)`,
cluster-robust CIs at symbol-day, the calibrated nulls, and the three rivals addressed — **or** a
demonstration that it is not identifiable on this estate, which is equally a result.

**Failure.** A half-life. A median quoted over a defective distribution. A `τ` or a `k` chosen after
seeing an outcome. A hazard statement whose clock is not in its label.

**Stop.** When `mu_tau` is published with its causes, or non-identifiability is shown. **Do not
widen `τ`, lower `k`, or extend the window to obtain a cleaner curve** — publish the shortfall.
`A-S50`'s duration bound is proportional to the window, so a longer window always looks better and
means less.

---

## 9. What execution may NOT change

`τ` · `k` and its two declared sensitivities · the floor · the event definition · the cause list ·
the clock of each process · the stratum · the estimator · the primary tests. Any change is a **new
preregistration with a new hash**, and the old one stays on the record.

```verdict
D_E8_EDGE_LIFETIME_PREREGISTRATION_FROZEN
NO_OUTCOME_COLUMN_HAS_BEEN_OPENED_BY_LANE_D_IN_ANY_ROUND
UNIT_AND_TIME_ZERO_DECLARED_HR_OPTION_A_FIRST_ELIGIBLE_TIME
TWO_PROCESSES_TWO_CLOCKS_BOTH_NAMED_IN_EVERY_LABEL
S125_END_TO_START_APPLIES_TO_RECURRENCE_NOT_TO_THE_EDGE_WINDOW
EVENT_IS_THE_END_OF_THE_FIRST_ALIVE_SPELL_BARRIER_IS_ECONOMIC_NOT_FITTED
K_EQUALS_TEN_BPS_CANONICAL_BINANCE_BASE_WITH_FOUR_AND_ZERO_AS_DECLARED_SENSITIVITIES
NEVER_ALIVE_IS_A_CATEGORY_NOT_A_CENSORING_AND_ITS_SHARE_IS_REPORTED_FIRST
FOUR_CAUSES_ENUMERATED_INCLUDING_SLIP_DROPPED_WHICH_IS_COUNTED_NOT_WALKED_FORWARD
AALEN_JOHANSEN_CIF_ONE_MINUS_KM_EXPLICITLY_FORBIDDEN
ESTIMAND_IS_MU_TAU_INTEGRAL_OF_P00_IDENTIFIED_UNDER_TYPE_I_CENSORING
TAU_SIXTY_MINUTES_FIXED_BY_A_RULE_NOT_BY_A_LOOK_COMPETING_CIF_UNDER_25_PERCENT
HALF_LIFE_AND_MEDIAN_AND_LATENT_MARGINAL_ALL_FORBIDDEN_AS_OUTPUTS
CLUSTERING_IS_SYMBOL_DAY_NOT_SYMBOL_BECAUSE_SYMBOLS_CO_FIRE_AT_SIX_POINT_TWO_TIMES
HONORE_THEOREM_1_BRANCH_DECLARED_NOT_MEASURED_AND_SWITCHING_REQUIRES_NAMING_3A_3B_3C
THREE_TEXTBOOK_RIVALS_PREREGISTERED_FALSE_PROTECTIVITY_BARRIER_DISTANCE_CROSSOVER
INVERSE_GAUSSIAN_FIRST_PASSAGE_IS_THE_SPECIFICATION_TO_BEAT
PH_IS_A_RIVAL_NOT_A_DEFAULT_AND_THE_CORPUS_PREDICTS_AGAINST_IT
P00_AT_TAU_REPORTED_BEFORE_ANY_QUANTILE_DEFECTIVE_MEANS_NO_QUANTILE_AT_ALL
PRIMARY_POPULATION_IS_THE_FIFTY_K_FLOOR_BECAUSE_IT_IS_REPRODUCIBLE
PUBLISHED_1271_SET_IS_NOT_REBUILDABLE_FROM_ITS_RULE_AND_THE_DEFECT_IS_CARRIED_OPENLY
D_E4_DEAD_TIME_POISSON_NOT_INHERITED_IT_WAS_WITHDRAWN_BY_D_E7
EXECUTION_MAY_CHANGE_NOTHING_IN_SECTION_9_WITHOUT_A_NEW_HASH
READ_ONLY_NO_ORDERS
FROZEN_AWAITING_OPERATOR_SIGN_OFF_AND_INDEPENDENT_REVIEW
```
