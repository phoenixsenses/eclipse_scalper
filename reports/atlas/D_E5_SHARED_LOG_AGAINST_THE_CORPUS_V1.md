# D-E5 — THE SHARED LOG AGAINST THE CORPUS

**Lane D · study `D-E5` · 2026-08-27 · read-only · no market data touched · corpus round**

Three questions were put: **what does the corpus generate**, **what does it demand**, and **audit the
shared log with the corpus**. This answers all three from one pass.

**Method.** `_SHARED_LOG.md` parsed mechanically: **47 blocks** (excluding the format template),
**329 distinct verdict tokens**, lanes **A 18 · C 23 · D 6 · B 1**. The parse and the traffic counts
are mechanical; the *selection* of claims confronted in §4 is a lane-D judgement and is labelled one.
Corpus read through `corpus_text_v1` (NUL-safe, ligature-normalised).

**Snapshot, and it moved while this was being written.** The counts below are a snapshot of
`_SHARED_LOG.md` at parse time; by the time this block was appended the file held **51** blocks —
the other lanes wrote four more mid-round. Every count here is therefore a **lower bound**, and the
one in §1 is a lower bound on the traffic addressed to lane B.

---

## 1. The first finding needs no corpus at all: 47 messages to lane B, one answer

The `to X` line is the point of the shared log — the charter says *"write it even when it is `-`"*.
Counting non-empty ones:

```
from -> to        A     B     C     D
A                18    18    18    13
C                23    23    21     -
D                 -     6     6     6
B                 -     1     -     -
                       --
messages ADDRESSED TO LANE B: 47          blocks WRITTEN BY LANE B: 1 (2026-08-26)
```

**Forty-seven requests to the review lane; one block from it, on the first day.** A and C have run
41 rounds between them since, and D six, and every one of them wrote to B.

This is the exact shape of the defect the shared log was created to fix. Its own header records it:
*"one asked for an independent reviewer eight times without anyone hearing it. All of that surfaced
only afterwards, by luck. **This file is the fix.**"* The file is working — the asking is now
visible and counted. **What is not working is the answering.** Eight became forty-seven.

And `CLAUDE.md`'s gated chain — `implementation → independent review → correction → independent
re-review → acceptance` — is written as **non-compressible**. It is not being compressed; it is
**stalled at phase two**, in public, across all three producing lanes. This is a governance
observation about the process, not about anyone's content, and it is the first thing an audit of
this file finds.

---

## 2. What the corpus GENERATES — objects derivable today that this estate does not have

Not questions. **Objects**: things that can be written down now, with the passage that supplies them.

| object | passage | who needs it | state |
|---|---|---|---|
| **`μ_τ`, restricted mean survival time** | STK4080 Sl.8 | `A-S50`/`A-S52`'s missing scalar | derivable; `τ` must be declared first |
| **Closed-form competing-risk CIF** `1 − exp(−λ(w−900 s))` | ABG 3.4.1 + `D-E4`'s dead-time Poisson | anyone quoting window contamination | **derived, `D-E4`** |
| **Inverse-Gaussian first passage**, two free parameters `c/σ`, `μ/σ` | ABG 10.3.1 eq (10.2) | the duration itself | writable today; both parameters already measured here |
| **Local dependence / Granger-Schweder**, and it is **directional** | ABG 9.4.1 (Schweder 1970) | `D-E4`'s cross-symbol result | the next tool — see below |
| **Aalen additive hazard** | ABG ch.4 | the named remedy when PH fails, which ABG 10.3.2 predicts here | available, unused on this line |
| **Dynamic path analysis**, direct vs mediated through `N(t−)` | ABG 8.4 | H2's marginal specification | one residual-SD plot |
| **Peterson (1975) competing-risks bounds** | ABG *cites* it, never states it | the partial-identification middle | **NOT on this shelf** — 0 hits on 5 discriminating terms; deriving them is new work |

**The one worth acting on next is local dependence, because `D-E4` left a symmetric answer to a
directional question.** `D-E4` measured coincidence, which is symmetric by construction: BTC|ETH
co-fire at 6.2× chance within ±1 minute. ABG §9.4.1 gives the asymmetric version, in Schweder's own
words: *"If event A occurs first, then the intensity of event B is changed, hence A influences B. On
the other hand, if event B occurs first, then the intensity for A is unchanged… **B is locally
dependent on A, while A is locally independent on B**."* That turns *"the three symbols are one
clock"* into *"which one is the clock"* — and it is still outcome-blind, still arrival times only.
**Any trading reading of a lead would be a different study with its own multiplicity budget; D does
not open one.**

---

## 3. What the corpus DEMANDS — and each demand names a lane

| demand | passage | lane | status |
|---|---|---|---|
| Declare the **unit and time zero** before anything else | STK4080 Ex 1.1 · H&R multiple-eligibility | **D** | owed, `D-E6` |
| **`τ` frozen before any outcome is read**, or the duration is descriptive only | Chan Ex 2.5's own disclosed look-ahead | **D → A** | binding, accepted |
| **Name the sample by artifact path, never in prose** | — (`D-E2`'s finding, no passage needed) | all | one population, two names, proved |
| **Calibrate every null before reading its test** | `CLAUDE.md` 380-C · `C-T31` | all | 2 of 6 needed it in `D-E4`; both changed answer |
| **Check `σ̃/s` before quoting a trade-channel formula** | memory §479 rule | **A** | applies to `A-S57`, see §4 |
| **A declining relative effect has three textbook generators** before it is a finding | ABG 6.5.2 · 6.6 · 10.3.2 | **C, D** | registered |
| **Interference makes the estimand ill-defined**, not merely noisy | H&R Fine Point 1.1 | **C** | `C-T43` found it independently |
| **Reaction vs prediction is permanently unidentified** on public data | Bouchaud 11.2 | **A** | `A-S54` is its empirical picture |

---

## 4. The confrontation — shared-log claims against the corpus

Selection is lane-D judgement over the 47 blocks: claims that are *general* (method or mechanism, not
a single number) and fall in a domain the corpus covers.

| claim | corpus | status |
|---|---|---|
| `A-S54` — the price has already moved **35–310 bps** in the flow's direction *before* the liquidation prints, against 7–95 bps after; TQP Fig 12.1's cause shape is absent | Bouchaud §11.2, `observed = reaction + prediction` | **`TEXTBOOK_PREDICTED`** — and the same passage says the decomposition is **permanently unidentifiable** on public data. The corpus predicted the picture *and* forbade undoing it. |
| `C-T33` — the scaling collapse holds on three symbols at 3–7% error and the book's TSLA `χ = 0.95` returns as **0.92 / 1.06 / 0.96** on crypto perps | Bouchaud, the collapse form | **`TEXTBOOK_PREDICTED`**, and the strongest positive transfer in the log: a published constant replicating on an asset class that had nothing to do with its discovery |
| `C-T36` — **there is no single impact exponent**; `R` is a surface, 0.04–0.13 at `T=1` rising to 0.58–0.74 at `T=100`, 0.97–1.41 at fixed POV | Econophys ODM: *"the universality of these exponents has been challenged, even the power-law form… depend on the type of stock and the market"* | **`TEXTBOOK_PREDICTED_BY_THE_SOURCE_NOBODY_CITES`** — the shelf disagrees with itself, and `C-T36` sides with the source the estate had not been quoting |
| `C-T34` — impact is near-permanent on BTC/ETH, *"exactly the configuration Chapter 13 opens by arguing markets cannot sustain"* | Bouchaud ch.13 | **`TEXTBOOK_CONTRADICTED_THEN_RESOLVED_IN_LANE`** — `C-T35` found `G` is not a power law and the efficiency condition is **not** violated at `L ≥ 1024`. Resolved in one round, by the same lane, without an outside prompt. Worth naming as the log working. |
| `C-T43` — Chan's "Heisenberg uncertainty principle" **is** H&R's interference, the fourth identifiability condition | H&R Fine Point 1.1 | **`TEXTBOOK_PREDICTED`** — and note the upgrade C got from naming it: from *"be skeptical"* to *"the estimand is not well defined"* |
| `A-S57` — the maker floor is the fee; `s/2 = 0.013 bps` is **three orders of magnitude below it**, and spread capture, adverse selection and queue opportunity cost are all under 0.03 bps | Bouchaud §21.4 names those as the terms | **`CORPUS_REGIME_MISMATCH`** — the terms exist, the regime does not. Memory §479's own rule applies and should be cited in `A-S57`: *check `σ̃/s` before quoting a trade-channel formula; above ~0.5 they do not apply.* |
| `A-S58` — adverse selection rises monotonically with queue consumption on four symbols across a 78× tick range, **and the sign flip TQP predicts is there**: the front of the queue gains | TQP queue priority | **`TEXTBOOK_PREDICTED_INCLUDING_THE_SIGN`** — a predicted sign found where it was predicted is the strongest evidence class in this log |
| `A-S62` — **only carry outgrows the fee**; its net asymptotes to the funding rate and the cost term vanishes | — | **`BEYOND_THE_SHELF`** — see §6. `funding rate`: **0 hits in 13 sources.** |
| `D-E4` — the three symbols co-fire at 6.2× chance within ±1 minute | ABG §1.5.4 asked the question; §9.4.1 supplies the directional refinement | **`ANSWERED_SYMMETRICALLY_WHERE_THE_CORPUS_OFFERS_DIRECTION`** |

---

## 5. Two cross-lane services, computed here

### 5a. `A-S53`'s robustness argument tested the wrong lever — and the right one is *more* favourable

`A-S53` concludes the square-root law needs a **$12.16 B** single order to move BTCUSDT 123.7 bps —
**57.7×** median hourly volume, and **558×** the largest liquidation ever recorded in the window
($21.8 M). Its robustness note is about the **amplitude**: *"a tenfold `Y` still leaves the largest
56× short."*

But `C-T36`, in the same log, says the **exponent** is not 0.5 — it ranges 0.04 to 1.41 depending on
the cut. The exponent enters as `1/δ` and is by far the bigger lever. Holding the implied
`Y·σ = 16.28 bps` fixed and varying `δ`:

```
delta      required Q     x median hr vol   x largest liquidation ($21.8M)
 0.13         $1.25e15         5.9e6              5.7e7
 0.30         $181.6B            862               8,329
 0.50 (sqrt)   $12.16B           57.7                558      <- A-S53's cell
 0.58           $6.95B           33.0                319
 0.74           $3.26B           15.5              149.7      <- C-T36's LARGEST measured exponent
 1.00           $1.60B            7.6               73.4
```

**Every exponent below 0.5 widens the gap, by orders of magnitude. The narrowest case in the measured
range still leaves the largest liquidation ever recorded 150× short.** `A-S53`'s conclusion is
robust to the lever it did not test, and more robust than to the one it did.
*(Caveat, stated: holding `Y·σ` fixed while varying `δ` is a direction check, not a refit — a proper
re-estimation would move the amplitude too. The order of magnitude is the claim, not the digits.)*

### 5b. `A-S52` builds on a scalar `D-E1` declared not point-identified

`A-S52` concludes *"below an hour DURATION binds and above it the POT binds"*, using `A-S50`'s
duration bound. `D-E1` showed that bound's scalar — the alpha lifetime — is a **latent marginal, not
point-identified at any `N`** (a theorem), and `D-E2` showed the empirical stand-in is
**floor-conditional by 4.0×**.

**The ordering may well survive**: `A-S52`'s statement is a *comparison* between two bounds, and a
comparison can be robust to a common scale factor its two sides do not share. **The crossover hour
cannot survive unexamined**, because it is a level, and the level moves with a threshold nobody has
justified. Not a withdrawal — a condition, and it belongs in `A-S52` rather than in D's log.

---

## 6. And the structural result: the corpus has closed everything it covers

Read the day's arc as one object. Every branch the corpus is **rich** on was measured and closed:

```
impact / square-root law      A-S53   558x short          corpus-covered   CLOSED
execution cost, taker         A-S55   every cell -8.6 to -37.8 bps         CLOSED
execution cost, maker         A-S57   floor is the fee, room 0.91-2.02 bps CLOSED
queue priority                A-S58   swept priority eats 34-89% of it     CLOSED
the 12.3x headline            A-S56   fee-only denominator                 CORRECTED
tradeable Sharpe              A-S63   0.051, half of "not fundable"        CLOSED
```

And the one branch that is **not** closed:

```
carry / funding               A-S62   only carry outgrows the fee; net asymptotes to the rate
```

Machine-checked with the correct reader and discriminating terms across all 13 sources:

```
"funding rate"   0 hits        "carry trade"    0 hits        "cost of carry"  1 (Hasbrouck, incidental)
"funding"        3 hits        "roll yield"     1 (Chan)      "contango"      21 (Chan, futures roll)
```

> **The only surviving branch is the one this corpus cannot advise on.** Its 4,299 pages are about
> impact, execution, queues, spreads and the statistics of durations — and it has now been pointed at
> each of those and closed them. On carry it has one book with 21 mentions of contango, and that book
> is the backtest-overfitting text, not a carry text.
>
> `THE_CORPUS_HAS_BEEN_SPENT_ON_THE_BRANCHES_IT_COVERS`. That is not a complaint about the shelf —
> it is a statement about where the estate now stands, and it means the next literature acquisition
> is **not** another microstructure book.

---

## 7. What D does next

`D-E6` is the preregistration, and it now carries: the unit and time zero declared (§3), `μ_τ` with
`τ` frozen before any outcome (§3), the closed-form CIF (§2), the non-symbol independence unit
(`D-E4`), the frailty nulls (`D-E1`/`D-E2`), and a null calibration before every test (`D-E4`).

The one thing D *adds* to its own queue from this round: **ABG §9.4.1 local dependence**, to turn
`D-E4`'s symmetric co-firing into a directional statement — outcome-blind, arrival times only, and
explicitly **not** a lead-lag trading claim.

```verdict
D_E5_SHARED_LOG_AUDITED_AGAINST_THE_CORPUS_47_BLOCKS_329_TOKENS
FORTY_SEVEN_MESSAGES_ADDRESSED_TO_LANE_B_AND_ONE_BLOCK_WRITTEN_BY_IT
THE_SHARED_LOG_MADE_THE_ASKING_VISIBLE_THE_ANSWERING_IS_WHAT_IS_MISSING
EIGHT_BECAME_FORTY_SEVEN
GATED_CHAIN_IS_NOT_COMPRESSED_IT_IS_STALLED_AT_PHASE_TWO_IN_PUBLIC
CORPUS_GENERATES_SEVEN_NAMED_OBJECTS_ONE_OF_THEM_NOT_ON_THIS_SHELF
PETERSON_BOUNDS_ARE_CITED_BY_ABG_AND_STATED_NOWHERE_SO_DERIVING_THEM_IS_NEW_WORK
LOCAL_DEPENDENCE_ABG_9_4_1_IS_DIRECTIONAL_WHERE_D_E4_WAS_SYMMETRIC
A_S54_IS_BOUCHAUD_11_2_MADE_VISIBLE_PREDICTED_AND_PERMANENTLY_UNIDENTIFIABLE
C_T33_IS_THE_STRONGEST_POSITIVE_TRANSFER_A_TSLA_CONSTANT_REPLICATING_ON_PERPS
C_T36_IS_PREDICTED_BY_THE_ONE_CORPUS_SOURCE_THE_ESTATE_WAS_NOT_CITING
THE_SHELF_DISAGREES_WITH_ITSELF_ON_EXPONENT_UNIVERSALITY
C_T34_CONTRADICTED_CHAPTER_13_AND_C_T35_RESOLVED_IT_IN_ONE_ROUND_UNPROMPTED
A_S58_FOUND_A_PREDICTED_SIGN_WHERE_IT_WAS_PREDICTED_STRONGEST_EVIDENCE_CLASS_IN_THE_LOG
A_S57_IS_A_CORPUS_REGIME_MISMATCH_AND_THE_SIGMA_TILDE_OVER_S_RULE_APPLIES
A_S53_ROBUSTNESS_TESTED_AMPLITUDE_BUT_THE_EXPONENT_IS_THE_BIGGER_LEVER
EVERY_EXPONENT_BELOW_ONE_HALF_WIDENS_THE_GAP_BY_ORDERS_OF_MAGNITUDE
AT_C_T36_LARGEST_MEASURED_EXPONENT_0_74_THE_GAP_IS_STILL_150X
A_S53_IS_MORE_ROBUST_TO_THE_LEVER_IT_DID_NOT_TEST_THAN_TO_THE_ONE_IT_DID
A_S52_CROSSOVER_HOUR_IS_A_LEVEL_AND_RESTS_ON_A_NOT_POINT_IDENTIFIED_SCALAR
THE_ORDERING_MAY_SURVIVE_THE_LEVEL_NEEDS_ITS_FLOOR_DECLARED
EVERY_BRANCH_THE_CORPUS_COVERS_WAS_MEASURED_AND_CLOSED_TODAY
THE_ONLY_SURVIVING_BRANCH_IS_CARRY_AND_FUNDING_RATE_HAS_ZERO_HITS_IN_THIRTEEN_SOURCES
THE_CORPUS_HAS_BEEN_SPENT_ON_THE_BRANCHES_IT_COVERS
NEXT_LITERATURE_ACQUISITION_IS_NOT_ANOTHER_MICROSTRUCTURE_BOOK
NO_MARKET_DATA_TOUCHED_THIS_ROUND
NO_OTHER_LANES_ARTEFACT_MODIFIED_REPORTED_ONLY
READ_ONLY_NO_ORDERS
IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW
```

**Caveats.** The block parse and traffic counts are mechanical; the **selection** in §4 is judgement
and is labelled one · §1 counts *messages*, not obligations — a `to B` line is a request, and no lane
is entitled to a reply, so this is a description of traffic, not an accusation · §5a holds `Y·σ` fixed
while varying `δ`, which is a **direction check, not a refit** · §5b withdraws nothing and is a
condition for `A-S52`'s producing lane to accept or reject · §6's absence claim uses the NUL-safe,
ligature-normalised reader and discriminating terms, and it is bounded to **this** shelf — it says
nothing about literature not on disk · no other lane's artifact was modified · four lanes share one
`§` space; identity is `D-E5`, **no renumbering**.
