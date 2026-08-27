# D-E1 — WHAT ENDED THE OBSERVATIONS

**Lane D · study `D-E1` · 2026-08-27 · read-only, outcome-blind · corpus round + integrity audit**

Charter (`LANE_CHARTERS_V1.md`, LANE D): *"D's first question is not 'what is the half-life'. It is
'what ended the observations'."* **Failure is defined as a half-life** — a single number quoted
without an account of what terminated the observations that produced it. So no duration is estimated
here. This round establishes what may be estimated at all, and what actually ended the observations
in the sample the forced-flow line was measured on.

Artifacts: `reports/atlas/D_E1_OBSERVATION_SCHEME_V1.json` ·
`tools/d_e1_observation_scheme_audit_v1.py` (reads `liquidations.ts_ms`, `mark_prices.ts_ms` and the
`agg_trades` day index only — **it never forms a return**).

---

## 1. The corpus answers before the data does: a half-life is not identifiable, at any `N`

`A-S50` needs a scalar `t_window` for its duration bound `X = ADV · POV · t_window`. That scalar is
the marginal distribution of one cause acting alone. The corpus is unambiguous about it.

**STK4080, Slides 9 (Lindqvist), p. 6/28**, on the latent failure time approach — potential times
`T_1 … T_k`, observing only `T = min_h T_h` and `H = argmin_h T_h`:

> *"The marginal distributions of the `T_j` are often of primary interest, but are **non-identiﬁable
> in general** by observation of `(T, H)` only (**even if we have an inﬁnite number of observations**
> of `(T,H)`)"* · *"Additional, but **non-testable**, assumptions may lead to identiﬁability (for
> example, independence of the `T_j`)."* · *"In biostatistics, one usually **avoids** the latent
> failure time approach and restricts attention to the pair `(T, H)`."*

The same slide set's exercise **E2.1(c)** states the asymmetry that follows: *cause-specific hazards
are not influenced by the other risk, while the cumulative incidence function is* — *"a general
property of competing risks which one should be aware of."*

**ABG §3.4.1**, eqs. (3.67)–(3.68), gives what replaces it:
`P00(s,t) = exp(−∫ Σ_h α_0h(u) du)` and `P0h(s,t) = ∫ P00(s,u) α_0h(u) du`. And ABG warns against
exactly the substitute a trading desk reaches for first — one minus Kaplan-Meier with the other
causes treated as censorings, *"sometimes interpreted as estimating the probability of death due to
cancer assuming this to be the only possible cause… Such an interpretation may be quite speculative."*

**Consequence.** `ALPHA_HALF_LIFE_IS_A_LATENT_MARGINAL_NOT_IDENTIFIABLE`. This is a **theorem, not a
sample-size complaint**: infinite data does not repair it, so no window extension, no extra symbol
and no extra month is a remedy. That independently re-derives the charter's stop rule from a second
direction.

**And the identifiable object is the one the desk actually wants.** An execution schedule faces the
competing risks; it never faces the counterfactual world in which only decay can end the trade. The
CIF is not a consolation prize for the half-life — it is the correct estimand and the half-life was
the wrong one.

**But "non-identifiable" is not "nothing", and the estate has made that mistake before.** Memory's
`§394` records `DECISION_VALUE_PARTIALLY_IDENTIFIED` as *"the missing middle category"*; the same
middle exists here. Hernán & Robins **Technical Point 16.2, "Bounds: Partial identification of
causal effects"** is the general frame: a non-identified quantity is normally still **bounded**, the
data narrows the bounds, and *"all these partial identification methods … are often relatively
uninformative because the bounds are wide"*, with narrowing available only through parametric
assumptions. The competing-risks instance is **Peterson (1975)**, which ABG cites in its reference
list and **nowhere states**.

Machine-checked with the corrected reader (§6): `peterson bound(s)` · `bounds on the marginal` ·
`sharp bounds for competing` · `crude probability` — **0 hits across all 13 sources.**
So the correct standing verdict is `ALPHA_HALF_LIFE_PARTIALLY_IDENTIFIED_BOUNDS_NOT_ON_THIS_SHELF`:
the point value is unavailable by theorem, bounds exist in the literature, and deriving them here
would be new methodological work, not a lookup. **`NOT_IDENTIFIABLE` alone would have overstated
the closure** — the same error class the estate has already named once.

---

## 2. What actually ended the observations — measured

The H2 complete case was reproduced exactly (759 raw → **629**; cluster ladder identical to
`COMPLETE_CASE_REFERENCE` at all eight horizons), then its exclusions were classified.

| cause | n | share of raw | type |
|---|--:|--:|---|
| **left truncation** — delayed entry, ≥5 prior UTC days of daily notional | **124** | 16.34% | type I, calendar-deterministic |
| **right censoring** — required window outside the price span or crossing the lawful cutoff | **6** | 0.79% | type I |
| horizon unresolved | **0** | 0 | — |

The 124 are **exactly the leading days of each symbol**: BTC 52 and ETH 45 on `2026-07-23…27`,
SOL 27 on `2026-07-24…27`. The 6 are 2 at the series start and **4 crossing the `2026-08-21 00:00Z`
lawful cutoff**.

**ABG §2.2.8** settles their status. Censoring at fixed times, censoring at a specified count, and
censoring when the event has not occurred in a certain interval are **stopping times relative to the
complete history `F^c`**, so *"no additional randomness is introduced by the censoring… the
independent censoring assumption (2.54) is **automatically fulﬁlled**."* Both exclusion rules here
are deterministic functions of the calendar and of the data's own endpoints.

> **The forced-flow sample does not inherit `CENSORING_DEPENDENT_BY_STRUCTURE_ADJUSTMENT_NOT_
> CONSTRUCTIBLE`.** That verdict (S95) is about the **forward ledger** — 33 days, whole-day
> quarantine from feed outages plus the 2026-08-21 mass kill, positivity violated on 87.9% of days.
> Different sample, different observation scheme, and the charter's expectation that D would inherit
> it is **not what the sample says**. (`C-T31`'s lesson, applied: do not inherit a verdict, measure
> it on your own object.)

---

## 3. Three things do bite. None of them is censoring.

### 3a. The competing event is suppressed by definition below 15 minutes, then explodes

An episode is defined by liquidations separated by `> 900 000 ms`. A *new* episode inside the
response window is therefore **impossible below 15 minutes by construction**, and then:

| window | share of the 629 with a later same-symbol episode inside it |
|---|--:|
| 1m / 5m / 15m | **0.00%** *(by construction, not by measurement)* |
| 30m | 3.97% |
| 60m | **22.26%** |
| 120m | **55.17%** |
| 240m | **81.08%** |
| 360m | 90.46% |

Inter-episode gap, same symbol (`n = 626`), minutes:
`p5 31.3 · p10 41.0 · p25 63.5 ·` **`p50 109.5`** `· p75 197.6 · p90 336.5 · p95 457.9`.

**This is a cumulative incidence function** — for the one competing risk that is fully observable
without defining an outcome at all — and it is **the first duration this line has ever recorded.**

**What it does to `A-S50`.** The duration bound was swept at 1 / 5 / 15 / 60 / 240 minutes because no
record existed. At 240 minutes, four of five windows contain another episode of the same symbol; at
60 minutes, better than one in five. The 240-minute row (`$246 080 617`) is therefore not a longer
look at the same object — it is largely a **burst**, not an episode. H2's own
`SLOW_DISCOVERY_VS_LIQUIDITY_IMPACT_VS_CASCADE_NOT_SEPARATED` anticipated this in words; the size is
measured here for the first time.

### 3b. The published ladder is a clustering, not a risk set

`COMPLETE_CASE_REFERENCE["components"] = {1m 573, 5m 513, 15m 429, 30m 346, 60m 205, 120m 66,
240m 15, 360m 7}`. Read as attrition, that is a 98.9% decay curve — and it is the single most likely
way a spurious survival function gets built on this estate.

Measured: the **risk set is 629 at every horizon**, 1 m through 360 m, with `horizon_unresolved = 0`
in all eight cells. The falling numbers are `components()` merging anchors within `window_ms` for
cluster-robust standard errors.

`THE_PUBLISHED_LADDER_IS_A_CLUSTERING_NOT_A_RISK_SET` — recorded before anyone fits a curve to it.

The cluster count is still a real statement, in its own place: a 240-minute claim rests on **15**
independent blocks and a 360-minute claim on **7** — and `§371` already measured components as
support-disjoint but **not** independent (lag-1 `ρ = +0.24`), so even those are upper bounds.

### 3c. The clock slips, because the coverage rule checks a span and not coverage

`event_is_measurable` compares the required window against the symbol's price-series **endpoints**
and the lawful cutoff. The horizon reader takes `searchsorted(a, t0 + h·60 000, "left")` with **no
tolerance**. An internal feed hole therefore does not censor an observation — **it moves the clock**
and the observation is kept under its original label.

`mark_prices` gap structure, per symbol: median `1 000 ms`; **104–111 gaps over 60 s**; 4 gaps over
600 s; **maximum gap 325.6 minutes on all three symbols.**

Slip = (timestamp of the mark actually used) − (`t0 + h`):

| horizon | p50 | p99 | max | cells > 60 s |
|---|--:|--:|--:|--:|
| 1m | 604 ms | 935 ms | 24.7 s | 0 |
| 30m | 605 ms | 6.7 s | 178 s | 2 |
| **60m** | 604 ms | 11.7 s | **5.18 h** | 1 |
| 120m | 606 ms | 1.9 s | 4.18 h | 1 |
| **240m** | 615 ms | **250 s** | 4.80 h | **9** |
| 360m | 605 ms | 39.8 s | 4.03 h | 5 |

The median is the 1-second mark grid and is harmless. The tail is not: at least one *"60-minute"*
return was read from a mark **5.18 hours** after its anchor and passed the coverage gate.

**This is measurement error on the time axis, not censoring, and it is bounded**: 19 of 5 032
episode-horizon cells (**0.38%**), worst concentration 9/629 (**1.43%**) at 240 m. Negligible where
time is only a label — which is all H2 asked of it. **Not** negligible for a duration analysis, where
the time axis *is* the outcome.

And the recursion deserves its own line: `tools/coverage_invariant.py` exists to enforce
`NO_GLOBAL_SPAN_COVERAGE_INFERENCE` — *"a global span is not coverage"* — and it then infers coverage
from a **per-symbol span**, which carries the identical defect one level down. **A span is not
coverage when the series has holes.**

---

## 4. Frailty: the barrier `CLAUDE.md` records does not bind on this sample

ABG §6.4.4 and §7.2: *"for univariate data one will typically have an identiﬁability problem when
attempting to estimate frailty models, there will generally be **no such problem in the multivariate
case**"*; §6.9: *"this identiﬁability problem can only be avoided in the multivariate case where
several events are observed for an individual or a group of related individuals."*

`CLAUDE.md`'s "frailty is not separable" note and the MPH gate inherited at `§437` were written
against a **single-spell** argument. The forced-flow sample is not single-spell: **629 episodes
recurring within 3 symbols over 24 UTC days** (BTC 257 / ETH 233 / SOL 139) is recurrent-event,
clustered data.

`FRAILTY_IDENTIFICATION_BARRIER_DOES_NOT_BIND_SUPPLY_DOES` — the constraint here is 3 symbols,
24 days and 15 independent blocks at 240 minutes, which is a **supply** limit, not an identification
limit. The two must not be reported as the same refusal.

---

## 5. A named trap, pre-registered against, for whoever fits the hazard

ABG §6.6 (Di Serio 1997), eq. (6.28): with frailties correlated across two competing risks,
`μ_B(t)` depends on `A_C(t)`, so *"any factor that inﬂuences the risk of C will, on the population
level, also be seen to inﬂuence the risk of B **even when `α_B(t)` is independent of this risk
factor**. This creates a false association."* — **false protectivity.**

H2 publishes a negative covariate coefficient and reads it causally:
`MOMENTUM_CONTINUATION_OF_PRE_MOVE_FALSIFIED` (`pre_return_30m = −0.358`, `z = −3.99`). §6.6 does
**not** touch that reading — H2's outcome is a return, not a hazard, and there is no at-risk
selection in it (risk set 629 at every horizon, §3b). But the instant the same covariate is placed on
a hazard for *"the edge ends"*, false protectivity is the leading alternative explanation for a
negative sign, and it must be named in the prereg rather than discovered afterwards. **Flag, not a
claim.**

---

## 6. The corpus reader — a rediscovery, my own numbers corrected, and the missing guard

**This is not a new finding and the first version of this section was wrong.** It is written out in
full because how it went wrong is the useful part.

**What happened.** The round's first search — for the identification literature it needed — was
`grep -ci "identifiab"` and it returned **0** on Aalen-Borgan-Gjessing. I treated that as a
ligature problem, measured it *with grep*, and published `8 of 13 files · 1 073 distinct forms` and a
per-term miss table. Every one of those numbers was wrong, and the corrective was already on record.

**Correction 1 — this was already published.** Memory `§462`, *"NEVER `grep` this corpus (measured)"*,
records both defects and prescribes the method: **NUL bytes** in 3 files (`ABERGEL_LOB` 1 018,
`HERNAN_ROBINS_WHATIF` 240, `SURVIVAL_STK4080` 79) make `grep` treat them as **binary and skip
them**, and ligature glyphs (13 146 of them) defeat plain terms.

**Correction 2 — my census was corrupted by the very defect it was measuring.** NUL-safe recount:
**10 of 13 files carry ligatures, not 8** (grep had silently dropped `ABERGEL_LOB` and
`SURVIVAL_STK4080`), and the total is **13 146 glyphs** — exactly the already-published figure.

**Correction 3 — my per-term table used a broken substitution.** I mapped `fi→ﬁ, fl→ﬂ` only, so
`coefficient` became `cofﬁcient` and `efficient` became `efﬁcient`; the true forms use the **`ﬃ`**
ligature. Both terms read as `0` ligature hits when they in fact have hundreds.

**The measurement, done correctly** — what a naive grep sees, against the true normalised count:

| term | grep sees | truth | **missed** |
|---|--:|--:|--:|
| `identifiability` | **0** | **78** | **100.0%** |
| `positivity` | 1 | 160 | 99.4% |
| `confidence` | 24 | 350 | 93.1% |
| `coefficient` | 45 | 218 | 79.4% |
| `specific` | 176 | 658 | 73.3% |
| `first` | 573 | 1 730 | 66.9% |
| `effect` | 1 175 | 3 365 | 65.1% |
| `flow` | 241 | 674 | 64.2% |
| `efficient` | 233 | 592 | 60.6% |
| `significant` | 100 | 241 | 58.5% |
| `profit` | 357 | 579 | 38.3% |
| `competing risks` | 46 | 64 | 28.1% |

And the concentration matters for this lane specifically: of the 78 `identifiability` hits,
**62 are in Hernán & Robins** — the single richest source on the exact question D opened with — and
it is one of the three files grep refuses to read at all.

**What is actually new, and it is a process finding, not a corpus finding.** The corrective existed
in the repo **twice**, and neither copy could stop this:

- `tools/research_s100_corpus_absence_claim_audit_v1.py` — the estate's **published absence-claim
  auditor** — reads NUL-safely (`read_bytes().decode(...)`) but does **not** normalise ligatures. Its
  verdicts survive only because the source it leaned on happens to be zero-ligature. Luck, not
  design, and memory already says so.
- `tools/research_s120_cross_lane_claim_audit_v1.py` has the correct `LIGATURES` map **and hyphen
  folding** — privately, inside its own study file, importable by nobody.

So the knowledge was published, the code existed, and a session holding both still ran the bad read
first. **The gap is that nothing refuses it.** `tools/coverage_invariant.py` exists precisely so a
span-based coverage check cannot be written by accident; the corpus had no equivalent.

**Closed here:** `tools/corpus_text_v1.py` — S120's map lifted verbatim into one importable module
(`load` / `bodies` / `count` / `absence`), with the measured recall table in its docstring so the
next reader sees the cost before choosing a reader. **Neither existing script was modified**;
S100's ligature blindness is *reported*, not fixed — the producing lane owns corrections.

The new reader was then used immediately, on this round's own §1: `peterson bound(s)` ·
`bounds on the marginal` · `sharp bounds for competing` · `crude probability` → **0 hits, 13 of 13
sources**, which is what licenses the "bounds are not on this shelf" claim above.

Handed to lane B: **every absence claim ever made over this corpus with a `ff`/`fi`/`fl` term, or
touching the three NUL files, is unverified** — and the affected set is bounded and enumerable.

---

## 7. What is now fixed for D-E2, and what D will refuse

**Fixed, before any outcome is defined:**

1. The estimand is a **cumulative incidence function**, never a marginal half-life (§1).
2. The competing-risk set has at least three members and the third one — *the next episode arrives* —
   is already measured (§3a) and is **zero below 15 minutes by construction**, so no CIF may be read
   in that band as if the risk were absent.
3. The risk set is 629 at every horizon; the 573…7 ladder may never be used as one (§3b).
4. Any duration read must first pass a **slip gate**, because the coverage rule does not provide one
   (§3c).
5. Truncation and censoring are type I and independent (§2). That does **not** license independent
   *competing risks*, which §1 says is non-testable.

**D will refuse:** a half-life · a "time to peak" taken from the H2 response curve (its 240 m and
360 m rows rest on 15 and 7 independent blocks and are 81% / 90% contaminated by a later episode) ·
and any extension of the window to obtain a cleaner curve, which the charter forbids and §1 shows
cannot work in principle.

**Open and owed to the operator:** defining *"the edge ends"* has real freedom in it, and the
definition selects the answer. D-E2 is a **preregistration of the event definition and the cause
list**, frozen before estimation — not an estimate.

```verdict
D_E1_WHAT_ENDED_THE_OBSERVATIONS_ANSWERED_BEFORE_ANY_DURATION
ALPHA_HALF_LIFE_IS_A_LATENT_MARGINAL_NOT_POINT_IDENTIFIED_AT_ANY_N
NON_IDENTIFIABILITY_IS_A_THEOREM_NOT_A_SAMPLE_SIZE_COMPLAINT
ALPHA_HALF_LIFE_PARTIALLY_IDENTIFIED_BOUNDS_NOT_ON_THIS_SHELF
PARTIAL_IDENTIFICATION_IS_THE_MISSING_MIDDLE_AGAIN_HR_TECHNICAL_POINT_16_2
PETERSON_1975_CITED_BY_ABG_BUT_NEVER_STATED_ZERO_HITS_ON_FIVE_TERMS
IDENTIFIABLE_OBJECT_IS_THE_CUMULATIVE_INCIDENCE_FUNCTION
CIF_IS_THE_CORRECT_ESTIMAND_NOT_A_CONSOLATION_FOR_THE_HALF_LIFE
H2_COMPLETE_CASE_REPRODUCED_EXACTLY_759_TO_629_LADDER_IDENTICAL
LEFT_TRUNCATION_124_OF_759_IS_THE_LEADING_DAYS_OF_EACH_SYMBOL
RIGHT_CENSORING_6_OF_759_TWO_AT_SERIES_START_FOUR_AT_LAWFUL_CUTOFF
BOTH_ARE_TYPE_I_STOPPING_TIMES_SO_INDEPENDENT_BY_ABG_2_2_8
FORCED_FLOW_SAMPLE_DOES_NOT_INHERIT_S95_DEPENDENT_CENSORING_VERDICT
CHARTERS_EXPECTED_NOT_IDENTIFIABLE_FROM_CENSORING_AND_THE_SAMPLE_SAYS_OTHERWISE
COMPETING_EVENT_CIF_MEASURED_0PCT_TO_15M_22PCT_AT_60M_81PCT_AT_240M
ZERO_BELOW_15M_IS_BY_CONSTRUCTION_MIN_GAP_900000_NOT_BY_MEASUREMENT
INTER_EPISODE_MEDIAN_109_5_MINUTES_SAME_SYMBOL
A_S50_240M_DURATION_BOUND_RESTS_ON_81PCT_CONTAMINATED_WINDOWS
THE_PUBLISHED_LADDER_IS_A_CLUSTERING_NOT_A_RISK_SET
RISK_SET_IS_629_AT_EVERY_HORIZON_HORIZON_UNRESOLVED_ZERO
CLUSTER_COUNT_15_AT_240M_AND_7_AT_360M_IS_A_SUPPLY_STATEMENT_ONLY
HORIZON_SLIP_IS_TIME_AXIS_MEASUREMENT_ERROR_NOT_CENSORING
SLIP_BOUNDED_19_OF_5032_CELLS_WORST_CASE_60M_READ_5_18_HOURS_LATE
COVERAGE_INVARIANT_STILL_INFERS_COVERAGE_FROM_A_PER_SYMBOL_SPAN
A_SPAN_IS_NOT_COVERAGE_WHEN_THE_SERIES_HAS_HOLES
FRAILTY_IDENTIFICATION_BARRIER_DOES_NOT_BIND_SUPPLY_DOES
CLAUDE_MD_FRAILTY_CAVEAT_WAS_A_SINGLE_SPELL_ARGUMENT_SAMPLE_IS_MULTI_SPELL
FALSE_PROTECTIVITY_ABG_6_6_PREREGISTERED_AS_THE_RIVAL_FOR_ANY_NEGATIVE_HAZARD_SIGN
CORPUS_GREP_DEFECT_REDISCOVERED_NOT_DISCOVERED_MEMORY_462_ALREADY_HAD_IT
MY_FIRST_CENSUS_WAS_MEASURED_WITH_THE_TOOL_IT_WAS_INDICTING_AND_WAS_WRONG
LIGATURE_FILES_ARE_10_OF_13_NOT_8_GREP_DROPPED_TWO_NUL_FILES
MY_FI_FL_SUBSTITUTION_MANGLED_THE_FFI_FORMS_COEFFICIENT_AND_EFFICIENT_READ_ZERO
CORRECTED_GREP_SEES_0_OF_78_IDENTIFIABILITY_HITS_100_PERCENT_MISSED
SIXTY_TWO_OF_THEM_ARE_IN_A_FILE_GREP_REFUSES_TO_OPEN
S100_ABSENCE_AUDITOR_IS_NUL_SAFE_BUT_LIGATURE_BLIND_REPORTED_NOT_FIXED
THE_FIX_EXISTED_TWICE_AND_NEITHER_COPY_COULD_REFUSE_A_BAD_READ
CORPUS_TEXT_V1_ADDED_AS_THE_MISSING_GUARD_NO_EXISTING_SCRIPT_MODIFIED
NO_DURATION_ESTIMATED_THIS_ROUND_BY_DESIGN
READ_ONLY_OUTCOME_BLIND_NO_ORDERS
IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW
```
