# D-E3 — WHAT THE CORPUS ASKS US

**Lane D · study `D-E3` · 2026-08-27 · read-only, no market data touched · corpus round**

The question was inverted: not *what does the corpus answer*, but **what does it demand of us**.

**Method, stated so the selection can be audited.** Every interrogative sentence in all 13 sources
was extracted mechanically — **914 unique**, of which a coarse keyword filter flags **43** as demands
on the analyst rather than exposition. The full extraction is saved at
`reports/atlas/D_E3_CORPUS_QUESTIONS_ALL_V1.json` (`tools/d_e3_corpus_question_extract_v1.py`).
**Extraction is mechanical; the selection below is a lane-D judgement and is not.** Anyone can
re-filter the artifact and disagree.

Per source, unique / design-relevant: `STK4080 100/23 · Bouchaud 159/5 · ABG 34/5 · Chan 114/4 ·
Kissell 133/4 · Hernán-Robins 117/1 · Hasbrouck 67/1 · López de Prado 78/0 · Econophys 44/0 ·
Cartea 33/0 · Harris 22/0 · Abergel 12/0 · Honoré 1/0`.

---

## A. Three books ask the same question in three vocabularies, and it has never been answered here

| source | the question, verbatim |
|---|---|
| **STK4080, Ex. 1.1** | *"What is the **'at risk' indicator `Y_i(t)`** for the i-th process?"* |
| **STK4080, Slides 1** | *"Definition of **starting time and failure time** may be difficult ▶ Definition of **time scale** (in reliability: operation time, calendar time or number of cycles?)"* |
| **Hernán & Robins, target-trial ch.** | *"If a woman meets these eligibility criteria continuously between age 51 and 65, **when should her follow-up start**? At age 51, 52, 53…?"* |

Hernán & Robins do not leave it rhetorical. They give the menu — **(a)** the first eligible time,
**(b)** a randomly chosen eligible time, **(c)** *every* eligible time, i.e. sequential target-trial
emulation, which "can be more efficient… However, because individuals may be included in multiple
target trials, appropriate adjustment of the variance" is required — and they flag the cost of
getting it wrong in the margin: *"discrepancies between observational studies and a randomized trial
was partly due to **mishandling of time zero**."* Then the granularity rule: *"choosing a week or a
month as the time unit will introduce bias. This bias could be eliminated by choosing day as the
time unit."*

**Eclipse's answers, such as they are.** `Y_i(t)`: never defined — `D-E1` measured the risk set at
**629 at every horizon from 1 m to 360 m**, which is what "no event was ever defined" looks like in
a table. Time zero: `t0` = the first liquidation of an episode, where an episode is a run separated
by a **15-minute** gap — that is strategy **(a)**, with a time unit of 15 minutes, and `D-E2`
measured that the *eligibility* rule underneath it (the `$50 000` floor) moves the central duration
by **4.0×**. Time scale: wall-clock, never compared against a volume or event clock, though memory
already records `clock is part of the estimand`.

> `THE_CORPUS_ASKS_ONE_QUESTION_IN_THREE_VOCABULARIES_WHAT_IS_THE_UNIT_AND_WHEN_DOES_ITS_CLOCK_START`
> — status **`NEVER_DECLARED`**. Not wrong; undeclared, which under H&R's own margin note is the
> documented cause of observational-vs-trial discrepancies.

---

## B. And one question answers `A-S50` outright — the corpus hands over the estimator

**STK4080, Slides 8**, verbatim:

> *"Can we estimate mean survival time, `E(T)`, from KM? Recall that `E(T) = ∫₀^∞ S(u)du`. So, can we
> estimate `E(T)` by `Ê(T) = ∫₀^∞ Ŝ(u)du`? **This is, however, problematic due to censoring, and the
> fact that the right tail is poorly estimated (and `Ŝ(t)` may even be constant and positive for all
> large `t`.)** But we can instead estimate the **restricted mean**, i.e., the expected survival in
> `[0,t]`, `μ_t = ∫₀^t S(u)du`."*

and, as an exam question: *"**Why can't the median survival time be estimated** for the DES
patients? **What is meant by the restricted mean survival time?**"*

**`Ŝ(t)` constant and positive for all large `t` is exactly `D-E2`'s defective distribution** — the
cure-model case ABG ch.10 derives when the drift is away from the barrier, whose signature H2 already
published as `PEAK_NOT_OBSERVED_WITHIN_SUPPORTED_WINDOW`. Under it, the mean does not exist, the
median may not exist, and **a half-life certainly does not.**

**The restricted mean survival time needs none of that.** `μ_τ = ∫₀^τ S(u)du` is defined whether or
not `S` reaches zero; it is identifiable under the **type-I, independent** censoring `D-E1`
established for this sample; and it is dimensionally and semantically **exactly** what `A-S50`'s
frontier wants — the expected amount of time the edge is alive *inside the window you will actually
trade*:

```
X  =  ADV · POV · μ_τ            μ_τ = restricted mean edge lifetime over a DECLARED horizon τ
```

> `RESTRICTED_MEAN_SURVIVAL_TIME_IS_THE_QUANTITY_A_S50_ACTUALLY_NEEDS`
> — status **`ANSWERED_BY_THE_CORPUS_NOT_YET_BY_US`**.

**Its price, and it is the same discipline `D-E2` imposed on the floor:** every RMST is
`τ`-conditional. `τ` is declared in advance or the number means nothing — which is the correct
version of the charter's ban on extending the window, not a loophole in it.

---

## C. One question would falsify H2's specification, and the diagnostic is a single plot

**ABG §8.4**, on recurrent-event data: *"We will also address another question: **to what extent is
the effect of treatment working directly and to what extent is it working indirectly through
`N(t−)`**?"*

ABG then reports the empirical lesson on the bladder-tumour data: the **marginal** model's
standardised martingale residual SD *"is increasing with time to above 2… clearly revealing that
there are patterns in the data that the marginal model fails to catch"*, while the **dynamic** model
including `N(t−)` has SD *"almost constant in time and close to 1"*. And the warning that makes it
matter here: *"from the point of view of solely judging the treatment effect, the marginal model may
appear the most appropriate, but when considering whether a correct description of the data set is
given, the dynamic model appears most correct."*

**Eclipse:** H2 is a marginal model. Its state vector is a **closed list** of five variables and
`N(t−)` — the count of prior episodes as a *dynamic* covariate — is not among them; `liq_intensity_60m`
is a count in a fixed prior window, which is not the same object. And `D-E1` measured that
**22.3% / 55.2% / 81.1%** of outcome windows at 60/120/240 minutes contain a later episode, i.e.
`N(t−)` incrementing *inside the outcome*.

> `DIRECT_VS_MEDIATED_THROUGH_N_T_MINUS_IS_UNASKED` — status **`OPEN_ANSWERABLE`**, and ABG's own
> diagnostic is one residual-SD-over-time plot. **D reports; the producing lane corrects.**

---

## D. One question invalidates a *reading*, and the estate makes that reading often

**ABG §6.5.2**: *"**How wise is this approach? Is it possible to conclude anything about the present
treatment effect at time `t` by comparing `μ₁(t)` and `μ₂(t)`?**"*

The derivation (eq. 6.23) answers no, and worse than no. When a treatment effect stops, the former
treatment group's **population hazard rises above the control's** — the ratio goes *below 1* —
purely from frailty selection: *"the original treatment group will suddenly have a higher population
hazard than the nontreatment group when treatment is discontinued. Hence, **the changing effect of
treatment cannot be directly discerned from the observed hazard rates.**"* ABG ties the crossover
explicitly to **Simpson's paradox**: *"the selection that occurs over time creates a skew
distribution with respect to the frailty variable, and this skewness is responsible for the
paradoxical crossing effect."*

**Stated as a conditional, because that is what it is.** The estate's arm-versus-control results are
measured on **returns**, not hazards, so §6.5.2 does not touch them as published. But this line has a
standing habit of reading convergence and crossing as substantive — *the arm fell to the control,
therefore the edge is over; the arm went below the control, therefore the gates subtract*. **The
moment any such contrast is restated as a hazard or a rate — which is precisely what a duration
analysis does — a crossing arm becomes the null, not the finding.**

> `A_CROSSING_ARM_IS_THE_FRAILTY_NULL_WHEN_THE_CONTRAST_IS_A_HAZARD` — status **`PRE-REGISTERED
> AGAINST`**, third member of the family with `D-E1`'s false protectivity (ABG §6.6) and `D-E2`'s
> declining-hazard-ratio-by-distance-to-barrier (ABG §10.3.2). **Three textbook generators, one
> curve.**

---

## E. And one question is aimed at what D is about to hand lane A

**Chan, Example 2.5 (continued)**: *"Since the goal for traders is ultimately to determine whether
the expected return or Sharpe ratio of a mean-reverting trading strategy is good enough, **why do we
bother to go through the stationarity tests (ADF or Variance Ratio) and the calculation of half-life
at all?** Can't we just run a backtest on the trading strategy directly and be done with it?"*

His answer **defends** the duration, on power: *"their statistical significance is usually higher than
a direct backtest… These preliminary tests make use of every day's price data for the test, while a
backtest usually generates a significantly smaller number of round trip trades."* That is a real
argument for lane D existing at all, and it is the same argument as `CLAUDE.md`'s N-non-consuming
work class.

But in the **same example** he discloses the trap: *"there is a **look-ahead bias** involved in this
particular example due to the use of **in-sample data to find the half-life and therefore the
lookback**."*

**That is the exact pipeline `D → A` would create.** If D estimates a duration on the burned sample
and A substitutes it into the frontier as `t_window`, the frontier inherits Chan's disclosed
look-ahead by name.

> `A_DURATION_ESTIMATED_IN_SAMPLE_AND_USED_TO_SET_A_HOLDING_PERIOD_IS_CHANS_DISCLOSED_LOOKAHEAD`
> — status **`BINDING_CONSTRAINT_ON_LANE_D_OUTPUT`**, and it goes into `D-E4`'s prereg, not into a
> footnote. Either `τ` and the RMST are frozen before the outcome is read, or the number may be
> reported as a **description** and never as a capacity input.

---

## F. Bouchaud opens optimal execution with five questions; the estate has answered two

> *"**What is the optimal horizon `T`? What is the corresponding trading schedule? Should one trade at
> a uniform rate during the time interval `[0,T]`, or should one front- or back-load the execution?
> Should one use market orders or limit orders? Is it wise to join a long bid- or ask-queue?**"*

| # | question | Eclipse | owner |
|---|---|---|---|
| 1 | optimal horizon `T` | **OPEN** — `A-S50`; and `CT-017` is a disagreement *inside* it (Bouchaud: `T` cancels; Kissell: `POV^a4`, never measured on crypto) | A, with D supplying the horizon input |
| 2 | trading schedule | **UNOPENED** | A |
| 3 | uniform / front- / back-load | **UNOPENED** | A |
| 4 | market vs limit orders | **ANSWERED** — §206: maker is a ~3 bps fee saving on both legs, no spread capture, no adverse-selection penalty | A |
| 5 | join a long queue? | **PARTLY** — §198 reachability stands at 98.8%; §201 killed the top-of-book queue model | C |

Lane D claims none of these except the input to (1). Recorded so the map is complete.

---

## G. Questions the corpus asks that this estate cannot answer, and why

- **ABG:** *"What is the observable intensity process of an individual counting process **if the `Z_i`
  are unknown** (which one would usually assume)?"* — the frailty is unobservable by construction.
  ABG §8.3's own consequence is already on record here: what you then have is a **rate** model, and
  *"the variance estimates from martingale theory… will typically **underestimate** the true
  variance"* → sandwich estimators. **`ANSWERED_ON_RECORD`.**
- **ABG:** *"Wei, Lin and Weissfeld's marginal analysis of multivariate failure time data: **should it
  be applied to a recurrent events outcome?**"* — a live methodological controversy in the field, not
  an Eclipse question. **`NOT_OURS`.**
- **STK4080:** *"**Can valve life in these systems be modeled as a renewal process?**"* — §406 already
  recorded that *the renewal question and the frailty question are the same question in different
  coordinates*. **`OPEN`**, and `D-E2` narrowed it: under Honoré, which branch you are in *is* the
  renewal answer.
- **STK4080:** *"**Why is this not an intensity process but merely a rate function?**"* — asked and
  answered on record (memory §469), but never applied to the forced-flow episode process, which is
  recurrent and whose covariates carry nothing about the unit's own history. **`OPEN_ANSWERABLE`.**

---

## Tally

```
NEVER_DECLARED                     1   (the unit and its time zero -- asked by 3 sources)
ANSWERED_BY_THE_CORPUS_NOT_BY_US   1   (restricted mean survival time)
OPEN_ANSWERABLE                    3   (N(t-) mediation · rate-vs-intensity on episodes · renewal)
PRE_REGISTERED_AGAINST             1   (crossing arm as frailty null)
BINDING_CONSTRAINT_ON_D            1   (Chan's in-sample-duration look-ahead)
ANSWERED_ON_RECORD                 2   (market-vs-limit  ·  rate-model sandwich)
UNOPENED_OTHER_LANE                3   (schedule · loading · queue joining)
NOT_OURS                           1   (WLW controversy)
```

## What `D-E4` must carry, from this round alone

1. **Declare the unit and time zero** before anything else — `Y_i(t)`, the eligibility rule, the
   choice among H&R's (a)/(b)/(c), and the time unit. Undeclared is the documented failure mode (§A).
2. **The estimand is `μ_τ`, the restricted mean**, not a half-life, not a median — and `τ` is frozen
   in advance (§B). This supersedes `D-E1`'s "the estimand is a CIF" only in the sense of adding the
   scalar the CIF cannot supply; both are needed and neither is a marginal.
3. **`τ` and the RMST are frozen before any outcome is read, or the number is descriptive only** (§E).
4. **Any hazard contrast between arms carries the frailty null explicitly** (§D), alongside the two
   already registered in `D-E1` and `D-E2`.
5. **`N(t−)` enters as a dynamic covariate, or its absence is declared** with ABG's residual-SD
   diagnostic reported either way (§C).

```verdict
D_E3_CORPUS_QUESTIONS_EXTRACTED_MECHANICALLY_914_UNIQUE_43_DESIGN_RELEVANT
EXTRACTION_IS_MECHANICAL_SELECTION_IS_A_LANE_D_JUDGEMENT_AND_SAYS_SO
THE_CORPUS_ASKS_ONE_QUESTION_IN_THREE_VOCABULARIES_UNIT_AND_TIME_ZERO
STK4080_AT_RISK_INDICATOR_AND_TIME_SCALE_AND_HR_MULTIPLE_ELIGIBILITY_TIMES
ECLIPSE_USES_HR_STRATEGY_A_FIRST_ELIGIBLE_TIME_WITH_A_FIFTEEN_MINUTE_UNIT_UNDECLARED
MISHANDLING_OF_TIME_ZERO_IS_HR_DOCUMENTED_CAUSE_OF_OBSERVATIONAL_TRIAL_DISCREPANCY
RESTRICTED_MEAN_SURVIVAL_TIME_IS_THE_QUANTITY_A_S50_ACTUALLY_NEEDS
S_HAT_CONSTANT_AND_POSITIVE_FOR_LARGE_T_IS_EXACTLY_D_E2_DEFECTIVE_DISTRIBUTION
RMST_IS_DEFINED_WITHOUT_S_REACHING_ZERO_AND_IDENTIFIABLE_UNDER_D_E1_TYPE_I_CENSORING
X_EQUALS_ADV_TIMES_POV_TIMES_MU_TAU_WITH_TAU_DECLARED_IN_ADVANCE
EVERY_RMST_IS_TAU_CONDITIONAL_WHICH_IS_THE_CHARTERS_WINDOW_BAN_DONE_CORRECTLY
DIRECT_VS_MEDIATED_THROUGH_N_T_MINUS_IS_UNASKED_AND_THE_DIAGNOSTIC_IS_ONE_PLOT
H2_IS_A_MARGINAL_MODEL_WITH_A_CLOSED_STATE_LIST_AND_NO_DYNAMIC_COVARIATE
ABG_MARGINAL_RESIDUAL_SD_ABOVE_2_VS_DYNAMIC_NEAR_1_IS_THE_PUBLISHED_TELL
A_CROSSING_ARM_IS_THE_FRAILTY_NULL_WHEN_THE_CONTRAST_IS_A_HAZARD
ABG_6_5_2_EQ_6_23_TREATMENT_GROUP_HAZARD_RISES_ABOVE_CONTROL_AFTER_THE_EFFECT_STOPS
ABG_TIES_THE_CROSSOVER_TO_SIMPSONS_PARADOX_VIA_FRAILTY_SKEWNESS
THREE_TEXTBOOK_GENERATORS_ONE_CURVE_FALSE_PROTECTIVITY_BARRIER_DISTANCE_AND_CROSSOVER
STATED_AS_A_CONDITIONAL_THE_ESTATES_ARMS_ARE_MEASURED_ON_RETURNS_NOT_HAZARDS
A_DURATION_ESTIMATED_IN_SAMPLE_AND_USED_TO_SET_A_HOLDING_PERIOD_IS_CHANS_DISCLOSED_LOOKAHEAD
CHAN_DEFENDS_THE_DURATION_ON_POWER_AND_DISCLOSES_THE_LOOKAHEAD_IN_THE_SAME_EXAMPLE
THIS_IS_A_BINDING_CONSTRAINT_ON_LANE_D_OUTPUT_NOT_A_FOOTNOTE
BOUCHAUD_OPENS_OPTIMAL_EXECUTION_WITH_FIVE_QUESTIONS_TWO_ANSWERED_HERE
OPTIMAL_HORIZON_T_IS_A_S50_AND_CT_017_SITS_INSIDE_IT
RENEWAL_AND_FRAILTY_ARE_THE_SAME_QUESTION_AND_HONORES_BRANCH_IS_THE_ANSWER
RATE_VS_INTENSITY_ANSWERED_ON_RECORD_BUT_NEVER_APPLIED_TO_THE_EPISODE_PROCESS
NO_MARKET_DATA_TOUCHED_THIS_ROUND
NO_DURATION_ESTIMATED_NO_THRESHOLD_SELECTED
READ_ONLY_NO_ORDERS
IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW
```
