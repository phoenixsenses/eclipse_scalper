# D-E4 — THE FORCED-FLOW ARRIVAL PROCESS: POISSON PER SYMBOL, ONE CLOCK ACROSS THEM

**Lane D · study `D-E4` · 2026-08-27 · read-only, OUTCOME-BLIND · corpus-led research**

`D-E3` extracted the corpus's own questions. **Two of them are aimed at this lane and had never been
answered here**, and neither needs a single outcome value:

> **ABG §1.5.4** — *"Independent or dependent data?"*
> **STK4080 Slides 1** — *"Can valve life in these systems be modeled as a **renewal process**?"*

Both are questions about the **episode arrival process** — which is also the competing risk `D-E1`
named and `D-E2` measured ("the next episode arrives"). This round answers them.

Artifacts: `reports/atlas/D_E4_ARRIVAL_PROCESS_V1.json` · `tools/d_e4_arrival_process_v1.py`.
Reads `sym`, `t0`, `q` only; `imp_*` and `pre_bps` sit in the same pickle and are never touched.
Declared family **T1–T6, Holm over 6**, run at **both** floors (`D-E2`'s rule: a duration statement
without its floor is not interpretable). **No threshold selected.**

---

## 0. Two nulls were calibrated before any result was read

`CLAUDE.md` §380-C: never freeze a gate without knowing its null value. `C-T31`: an uncalibrated
null is as dangerous as no null. Both applied here **before** the tests were interpreted — and both
changed the answer.

**T2b — the dead-time null.** The episode definition separates episodes by `> 900 s` **by
construction**. A dead time makes counts *more regular* than Poisson, so `index of dispersion < 1`
may be the detector rather than the market. Simulating the fitted dead-time model
(`gap = 900 s + Exp(mean − 900 s)`, each symbol's own rate and span, 400 reps) gives

```
null index of dispersion:  BTC 0.677 +/- 0.036   ETH 0.698 +/- 0.037   SOL 0.719 +/- 0.036
```

**not 1.** Scoring T2 against `1` instead of against `0.68–0.72` was worth **3–5× in `z` and 28
orders of magnitude in `p`** (below).

**T4b — the seasonality null.** The hourly histogram shows a peak/trough of `3.38 / 3.14 / 2.09`,
which looks like an intraday story. The uniform null at this `N` produces p50 `2.46 / 2.60 / 2.67`
and **p95 `3.57 / 4.00 / 4.01`**. Every observed value is inside the null's p95.
**A 3.4× intraday swing is what noise produces at `N ≈ 450`.** (At the `$50k` floor the null p50 is
already `3.6–6.0`, so seasonality is simply not detectable in that population.)

---

## 1. STK4080's renewal question — **yes, per symbol: Poisson seen through a 900-second dead time**

| test | floor `$0` | floor `$50k` |
|---|---|---|
| **T1** dead-time-corrected gaps exponential (CV = 1) | CV **1.036**, z 1.27 — **not rejected** | CV 1.107, z 2.67, Holm 0.038 — rejected |
| **T3** constant rate (Laplace) | mean U −0.23 — **not rejected**, but see below | mean U −0.43 — not rejected |
| **T4** intraday seasonality | **not detectable** (see §0) | **not detectable** |
| **T5** lagged duration dependence | β **+0.020** (z 0.73, MDE 0.079) — **not rejected** | β +0.073 (z 1.84, Holm 0.199) — not rejected |
| **T2** dispersion *vs the calibrated null* | z −1.97 / −2.74 / −3.89, Holm **0.021** — rejected | Holm **0.088** — **not rejected** |

Per-symbol dead-time-corrected CV: `BTC 1.040 · ETH 1.057 · SOL 1.011` at `$0`, and
`1.158 / 0.976 / 1.219` at `$50k`. Mean corrected gap `58.4 / 65.2 / 71.5` minutes.

> **Answer:** at the unfiltered floor, the arrival process of each symbol is **indistinguishable from
> a Poisson process observed through a 900-second dead time** — exponential gaps, constant rate, no
> lag-1 dependence, no detectable seasonality. The renewal question is answered **YES, and trivially
> so: a Poisson process is the memoryless special case.**
>
> The one survivor is a **slight extra regularity** beyond even the dead-time null (`z −2` to `−4`,
> Holm 0.021) at the unfiltered floor, which **does not survive at `$50k`**. That is a small,
> floor-dependent deviation and it is reported as such, not as clustering.

**T3's mean hides opposing signs, and that is the same trap a third time.** Per symbol at the `$0`
floor the Laplace statistics are `BTC +0.861 · ETH +0.646 · SOL −2.192`. The mean, `−0.228`, reads as
a flat rate because a rising pair and a falling SOL cancel. **SOL alone is at `|U| = 2.19`** (nominal
`p = 0.028`; not family-wise significant against 3 symbols × 6 tests, and not claimed). The
family-level non-rejection stands; the *reason* it stands is not "all three are flat". Recorded
because averaging away a disagreeing unit is exactly what §2 and §4 are about.

**And that closes §406's open item in D's direction.** §406 recorded *"the renewal question and the
frailty question are the same question in different coordinates."* In these coordinates the renewal
answer is **memoryless**, so there is no within-symbol duration dependence for a frailty to have to
explain away — consistent with `D-E2`'s Honoré Theorem-1 branch, and now measured rather than assumed.

---

## 2. ABG's question — **DEPENDENT, and the dependence lives at the minute scale**

The null is a **whole-day circular rotation** of the other symbol's timestamps: it preserves that
symbol's own clustering *and* its intraday seasonality, and destroys only cross-symbol alignment. So
a rejection cannot be "they share a clock-of-day".

**±5 minutes, floor `$0`:**

| pair | observed | share of A | rotation null | **excess** | z |
|---|--:|--:|--:|--:|--:|
| BTC \| ETH | 146 | **31.9%** | 58.0 ± 6.0 | **2.52×** | **+14.8** |
| BTC \| SOL | 98 | 21.4% | 51.1 ± 5.7 | 1.92× | +8.28 |
| ETH \| SOL | 88 | 21.1% | 46.9 ± 5.4 | 1.88× | +7.63 |

Holm-corrected `p = 8.1e-24` at `$0` and `3.7e-16` at `$50k`. **T6 is the only test in the family
that rejects at both floors.**

**And the tolerance family says what kind of dependence it is** (reported as a family, not a point —
`D-E2`'s rule; not in the Holm family, since these are the same test at other tolerances):

```
tolerance    BTC|ETH excess      BTC|SOL      ETH|SOL      mean z      floor $0
   +/- 1m       6.21x              5.23x        4.48x        12.6
  +/- 15m       1.53x              1.17x        1.15x         5.0
  +/- 30m       1.21x              1.06x        1.08x         3.4
  +/- 60m       1.12x              1.04x        1.03x         3.5

   +/- 1m       9.10x              6.84x        6.73x        11.0      floor $50k
  +/- 15m       2.37x              1.96x        1.60x         7.5
  +/- 30m       1.66x              1.44x        1.36x         6.2
  +/- 60m       1.32x              1.19x        1.19x         5.4
```

**The excess is 6.2× at ±1 minute and 1.1× at ±60 minutes.** That is not a shared slow regime — a
shared regime would show a flat excess across tolerances. It is **near-simultaneity**: a common
shock at the minute scale. And it is **stronger for larger episodes** (9.1× at ±1 min at the `$50k`
floor vs 6.2× unfiltered), which is what a cross-market cascade looks like.

> `THE_THREE_SYMBOLS_ARE_ONE_CLOCK_NOT_THREE_PANELS`.

**This corroborates `CASCADE_IS_COMMON_STATE_MARKER_ONLY` (§337, "96.9% of cascades have a peer
cascading") — measured on a different event family, with a null that preserves seasonality.** It is
corroboration, not inheritance: `D-E1`'s lesson was that a verdict must be re-measured on one's own
object, and this is that re-measurement.

---

## 3. What this does to Honoré, and to this lane's own supply claim

Honoré's multi-spell theorems treat the units as **independent panels**. `S101` / `§437` recorded
the premise as satisfied by a wide margin — `BTC 457 · ETH 416 · SOL 395` spells across three units.

The premise about **spells per unit** stands. What does not stand is reading `3` as three
independent panels: at ±1 minute, **16.8%** of BTC's episodes have an ETH episode alongside them
against a chance rate of `2.7%`; at ±5 minutes, `19.2` percentage points of BTC's episodes are
excess co-fires with ETH beyond the rotation null. The effective number of independent units is
**between 1 and 3 and closer to 1 than the count suggests**, and the exact value is
tolerance-dependent, so no single number is quoted.

`D-E2` said the binding constraint on this lane is **supply, not identification**. `D-E4` sharpens
that: the supply is smaller than the symbol count implies, and **it was never three.**

---

## 4. Two things I got wrong inside this round, and how they were caught

Both are the estate's own named failure families. Both were caught **before publication**, by the
calibrations in §0 — which is the entire argument for running them.

**(a) An uncalibrated null, exactly `C-T31`'s lesson.** I scored the index of dispersion against
`Poisson = 1` while the detector carries a 900-second dead time. Correcting the null:

```
                      z vs Poisson=1      z vs calibrated dead-time null
  BTC                   -10.05                      -1.97
  ETH                   -12.40                      -2.74
  SOL                   -12.86                      -3.89
  Holm p, floor $0      3.4e-31                      0.021
  Holm p, floor $50k    9.1e-14                      0.088   <- VERDICT FLIPS to non-reject
```

**28 orders of magnitude, and one flipped verdict, from a null nobody had computed.**

**(a-bis) And a third, of the same family: an average that cancelled opposing signs.** T3's mean
Laplace `U = −0.228` reads flat; the per-symbol values are `+0.861 / +0.646 / −2.192`. The
non-rejection is correct and stands, but it is not "three flat processes" — it is two mildly rising
and one at `|U| = 2.19`. Caught by printing the per-symbol row rather than the summary.

**(b) A pooled statistic that was a scale mixture, exactly `C-T30`'s lesson.** The raw pooled CV of
the corrected gaps at the `$50k` floor was `1.239` — **larger than every individual symbol**
(`1.158 / 0.976 / 1.219`), because the mean gaps differ by 2× (`116 / 129 / 232` minutes) and pooling
across scales inflates CV by construction. Standardising each symbol by its own mean before pooling:
`CV 1.107`, and `p` moves from `2.1e-9` to `0.0076`. The raw figure is kept in the artifact under
`pooled_raw_MIXTURE_DO_NOT_READ` so the trap stays visible.

> Two rounds ago I wrote that `C-T31`'s lesson generalises. It generalised onto me **three
> times**, in the first round where I ran real tests instead of reading — an uncalibrated null, a
> pooled scale mixture, and an average over disagreeing units. The defence was not care — it was **running
> the null calibration before reading the result**, mechanically, because §380-C says to.

---

## 5. What survives, and what `D-E5` inherits

**Survives Holm at both floors: T6 only.** Everything else is non-rejected, or rejects at one floor
and not the other. Stated plainly: **of six tests on this arrival process, exactly one carries
information that is robust to the detector's own settings, and it is the cross-symbol one.**

Fixed for `D-E5`:
1. The per-symbol arrival process may be modelled as **dead-time Poisson**; the CIF of the competing
   risk "the next episode arrives" therefore has a **closed form**, `1 − exp(−λ(w − 900 s))`, and no
   longer needs an empirical curve.
2. **Any standard error, cluster or panel argument that treats the three symbols as independent is
   wrong** by a factor that is large at short tolerances. The independence unit is **not the symbol**.
3. Seasonality is **not detectable at this N** — so it may not be used as an explanation *or* as a
   feature in this population, and a peak/trough ratio below ~4 is noise.
4. Every null gets calibrated before its test is read. Two of six needed it and both changed.

```verdict
D_E4_CORPUS_LED_RESEARCH_TWO_ABG_AND_STK4080_QUESTIONS_ANSWERED_OUTCOME_BLIND
RENEWAL_QUESTION_ANSWERED_YES_PER_SYMBOL_DEAD_TIME_POISSON
DEAD_TIME_CORRECTED_GAPS_ARE_EXPONENTIAL_CV_1_040_1_057_1_011
RATE_IS_STATIONARY_OVER_24_DAYS_LAPLACE_NOT_REJECTED
NO_LAG_1_DURATION_DEPENDENCE_BETA_PLUS_0_020_z_0_73_MDE_0_079
INTRADAY_SEASONALITY_NOT_DETECTABLE_AND_THE_NULL_SAYS_WHY
OBSERVED_PEAK_OVER_TROUGH_3_38_INSIDE_THE_UNIFORM_NULL_P95_OF_3_57
A_THREE_POINT_FOUR_TIMES_INTRADAY_SWING_IS_NOISE_AT_N_450
INDEPENDENT_OR_DEPENDENT_DATA_ANSWERED_DEPENDENT_AT_HOLM_8E_24
DAY_ROTATION_NULL_PRESERVES_SEASONALITY_SO_THIS_IS_NOT_A_SHARED_CLOCK_OF_DAY
BTC_ETH_COINCIDENCE_31_9_PERCENT_VS_NULL_2_52X_z_PLUS_14_8
EXCESS_IS_6_2X_AT_ONE_MINUTE_AND_1_1X_AT_SIXTY_MINUTES
THE_DEPENDENCE_IS_NEAR_SIMULTANEITY_NOT_A_SHARED_SLOW_REGIME
STRONGER_FOR_LARGER_EPISODES_9_1X_AT_ONE_MINUTE_AT_THE_FIFTY_K_FLOOR
THE_THREE_SYMBOLS_ARE_ONE_CLOCK_NOT_THREE_PANELS
HONORE_SPELLS_PER_UNIT_STANDS_BUT_THREE_INDEPENDENT_PANELS_DOES_NOT
EFFECTIVE_UNITS_ARE_BETWEEN_ONE_AND_THREE_AND_TOLERANCE_DEPENDENT_SO_NO_NUMBER_QUOTED
D_E2_SUPPLY_NOT_IDENTIFICATION_SHARPENED_THE_SUPPLY_WAS_NEVER_THREE
CORROBORATES_S337_CASCADE_IS_COMMON_STATE_MARKER_ONLY_MEASURED_NOT_INHERITED
MY_T2_NULL_WAS_UNCALIBRATED_POISSON_ONE_WHERE_A_DEAD_TIME_GIVES_ZERO_SEVEN
CORRECTING_IT_MOVED_z_FROM_MINUS_12_9_TO_MINUS_3_9_AND_28_ORDERS_OF_MAGNITUDE_IN_P
AND_FLIPPED_THE_FIFTY_K_VERDICT_FROM_REJECT_TO_NON_REJECT
T3_MEAN_U_MINUS_0_228_CANCELLED_PLUS_0_861_PLUS_0_646_AND_MINUS_2_192
MY_T1_RAW_POOL_WAS_A_SCALE_MIXTURE_CV_1_239_ABOVE_EVERY_SYMBOL
WITHIN_SYMBOL_STANDARDISATION_MOVED_P_FROM_2E_9_TO_0_0076
THREE_AGGREGATION_DEFECTS_IN_ONE_ROUND_ALL_THE_ESTATES_OWN_NAMED_FAMILIES
CAUGHT_BY_RUNNING_THE_CALIBRATION_BEFORE_READING_THE_RESULT_NOT_BY_CARE
ONLY_T6_SURVIVES_HOLM_AT_BOTH_FLOORS_OUT_OF_SIX
COMPETING_RISK_CIF_NOW_HAS_A_CLOSED_FORM_ONE_MINUS_EXP_MINUS_LAMBDA_W_MINUS_900S
THE_INDEPENDENCE_UNIT_IS_NOT_THE_SYMBOL
NO_OUTCOME_READ_NO_THRESHOLD_SELECTED_NO_DURATION_ESTIMATED
READ_ONLY_NO_ORDERS
IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW
```

**Caveats.** Sample is §311/§315's **burned** sample; this is the accounting/integrity class, not a
new hypothesis test on it · no outcome column was read · the day-rotation null preserves each
symbol's marginal structure but assumes the 24-day span is exchangeable at day granularity, which
T3's non-rejection supports but does not prove · T6's excess ratio is **tolerance-dependent by
construction** and the family is published for that reason; no single "effective N" is quoted ·
T2's residual regularity at the `$0` floor is small, floor-dependent, and is **not** claimed as a
market property · the dead-time null assumes the fitted exponential is the right corrected-gap law,
which T1 supports at the `$0` floor and only partially at `$50k` · four lanes share one `§` space;
identity is `D-E4`, **no renumbering**.
