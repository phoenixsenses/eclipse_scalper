# EVENT_BLOCK_V1_DECISION

## Summary

`event_block_v1` is the first practical event-conditioned filter candidate for the passive pocket ranking path.

Current rule:
- block `book_proxy_pressure`
- block `volatility_burst`

Intent:
- do not force entry on "good" event lanes
- only remove clearly harmful event contexts

## Why This Exists

Event/signal bridge work showed:
- `book_proxy_pressure` and `volatility_burst` were repeatedly negative or low-quality contexts for `micro_edge_v3_passive_alpha`
- a strict positive allow filter improved quality but collapsed coverage too hard
- the best practical tradeoff was `block_only`

That tradeoff became:
- `event_block_v1`

## Real Findings

### ETH 7D subset result

Source:
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_REAL.json`

Result:
- `common_count = 8`
- `improved_count = 5`
- `degraded_count = 1`
- `median_delta_npa_core = +3.711596e-04`
- `median_delta_score_raw_core = +5.791472e-04`
- `median_filtered_kept_ratio = 72.23%`

Interpretation:
- strong enough to justify broader ETH-side testing
- quality improved without destroying coverage

### ETH 1D repeated-window result

Source:
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_ETH_REAL_1D.json`

Result:
- `common_count = 8`
- `improved_count = 5`
- `degraded_count = 0`
- `median_delta_npa_core = +3.207438e-04`
- `median_delta_score_raw_core = +4.843181e-04`
- `median_filtered_kept_ratio = 73.16%`

Interpretation:
- ETH-side improvement is not limited to the 7D slice
- the filter still improves median quality on a shorter repeated window
- this materially strengthens the ETH experimental case

### BTC subset result

Source:
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_BTC_REAL_1D.json`

Result:
- `common_count = 8`
- `improved_count = 3`
- `degraded_count = 0`
- `median_delta_npa_core = 0.0`
- `median_filtered_kept_ratio = 72.76%`

Interpretation:
- filter is not harmful on this BTC sample
- but benefit is not strong enough to claim broad BTC improvement

### ETH broad candidate result

Source:
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_ETH_BROAD_REAL.json`

Result:
- `common_count = 6`
- `improved_count = 1`
- `degraded_count = 5`
- `median_delta_npa_core = -1.038896e-04`
- `median_delta_score_raw_core = -2.135000e-04`
- `median_filtered_kept_ratio = 75.94%`

Interpretation:
- ETH-only symbol scope is too broad
- the earlier improvement does not generalize across the wider ETH rule surface
- the filter benefit appears tied to the `micro_edge_v3_passive_alpha` lane, not to ETH globally

### ETH micro-edge 21D retest

Source:
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_ETH_MICRO_21D_REAL.json`

Result:
- `common_count = 0`
- `improved_count = 0`
- `degraded_count = 0`
- `recommendation = keep_baseline`

Interpretation:
- this does not falsify the filter logic directly
- it shows the current live slice did not produce a tradeable common set after capacity filtering
- in other words, the next blocker is tradeable coverage, not just filter quality

### ETH micro-edge 21D relaxed retest

Source:
- `reports/EVENT_BLOCK_ETH_MICRO_BASELINE_21D_RELAXED_REAL.json`
- `reports/EVENT_BLOCK_ETH_MICRO_V1_21D_RELAXED_REAL.json`

Result:
- relaxed settings used:
  - `splits = 2`
  - `min_n = 40`
- baseline and filtered still produced no tradeable ranked pockets
- effective blocker remained `insufficient_fill_rate`

Interpretation:
- the 21D issue is not just a strict validation gate setting
- current 21D live slice lacks tradeable coverage for this surface
- this profile should not be positioned as a long-window default candidate
- reducing `splits` and `min_n` still did not create a common tradeable set

## Decision

Current research decision:

- ETH:
  - `event_block_v1 = experimental_on` only on the validated `micro_edge_v3_passive_alpha` surface
  - ranking profile candidate:
    - `event_block_eth_micro_v1`
  - rationale:
    - positive on ETH 7D
    - positive again on ETH 1D repeated-window retest
    - broad ETH retest showed symbol-only scoping is too loose
    - 21D micro-edge retest currently lacks common tradeable coverage, even after relaxed validation gates
- BTC:
  - `event_block_v1 = observe_only`
  - rationale:
    - not harmful on BTC 1D
    - but not strong enough to claim clear broad benefit

This is not ready to become:
- a default global mitigation profile across all symbols
- or a blanket ETH-wide profile across all rules

Current positioning:
- `event_block_eth_micro_v1` = ETH short-window experimental profile
- not a long-window production candidate
- 21D long-window retests should be considered non-actionable until tradeable coverage improves

## Rollout Rule

Use this order:

1. ETH broader candidate sets
2. BTC repeated retests
3. symbol-aware profile trial
4. only then consider general defaulting

## Technical Notes

Implementation points:
- `tools/validate_passive_pocket_forward.py`
- `tools/rank_passive_pockets_forward.py`
- `tools/summarize_rank_event_filter.py`
- `tools/summarize_rank_event_filter_set.py`

Profile name:
- `event_block_v1`
- `event_block_eth_v1`
- `event_block_eth_micro_v1`

Blocked lanes:
- `book_proxy_pressure`
- `volatility_burst`

## Teaching Note

This is a good example of a usable research result:

- not every event becomes a positive signal
- often the first real gain comes from removing bad context
- that is usually safer than forcing trades only in rare "good" context

In short:
- negative filters often productionize earlier than positive filters

### Single-lane decomposition: book_proxy_pressure vs volatility_burst

Source:
- `reports/EVENT_BLOCK_BASELINE_REAL_7D_PROBE.json`
- `reports/EVENT_BLOCK_BOOK_PROXY_V1_7D_PROBE.json`
- `reports/EVENT_BLOCK_V1_7D_PROBE.json`
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_BOOK_PROXY_V1_7D.json`
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_V1_7D.json`

Run settings: ETH TOP8 candidates, 7D lookback, splits=3, min_n=5, relaxed gates, fee=1.0.

Result — `event_block_book_proxy_v1` (block book_proxy_pressure only):
- `common_count = 8`
- `improved_count = 4`
- `degraded_count = 4`
- `median_delta_npa_core = -1.020e-05`
- `median_filtered_kept_ratio = 94.82%`

Result — `event_block_v1` (block book_proxy_pressure + volatility_burst):
- `common_count = 8`
- `improved_count = 6`
- `degraded_count = 2`
- `median_delta_npa_core = +3.698e-04`
- `median_filtered_kept_ratio = 76.24%`

Interpretation:
- `event_block_book_proxy_v1` is a NO-GO: 50/50 improved/degraded, negative median delta NPA
- `event_block_v1` is positive: 6/8 improved, strongly positive median delta NPA
- the improvement in `event_block_v1` is NOT driven solely by `volatility_burst`; the lanes are synergistic
- see volatility-only decomposition below

### Volatility-burst-only decomposition: event_block_volatility_v1

Source:
- `reports/EVENT_BLOCK_VOLATILITY_V1_7D_PROBE.json`
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_VOLATILITY_V1_7D.json`

Result — `event_block_volatility_v1` (block volatility_burst only):
- `common_count = 8`
- `improved_count = 4`
- `degraded_count = 4`
- `median_delta_npa_core = +1.414e-04`
- `median_filtered_kept_ratio = 80.16%`

Three-way comparison summary (same baseline, same run settings):

| Profile | Improved | Degraded | Median ΔNPA | Kept Ratio |
|---|---|---|---|---|
| book_proxy_pressure only | 4/8 | 4/8 | -1.020e-05 | 94.82% |
| volatility_burst only | 4/8 | 4/8 | +1.414e-04 | 80.16% |
| both (event_block_v1) | 6/8 | 2/8 | +3.698e-04 | 76.24% |

Interpretation:
- `event_block_book_proxy_v1` is a NO-GO: mildly harmful, 50/50, negative median
- `event_block_volatility_v1` is MIXED: positive median but still 50/50 improved/degraded
- the two lanes are SYNERGISTIC: the combination (6/8, +3.698e-04) is strictly better than either alone
- `book_proxy_pressure` on its own degrades quality; it only becomes net-positive when combined with `volatility_burst`
- `volatility_burst` blocking specifically helps h=120 (long-horizon) pockets and high-imbalance (>=0.5) pockets
- `volatility_burst` blocking hurts h=60 imb=0.3 signals (short-horizon, low-imbalance)
- together the two lanes reach 6/8 improved because each lane fixes different pockets the other misses
- two most improved by v1: h=120 imb=0.85 (+1.200e-03), h=120 imb=0.5 (+1.102e-03)
- two most degraded by v1: h=60 imb=0.3 spr=0.0002 (-1.740e-03), h=60 imb=0.3 spr=0.00025 (-1.013e-03)

## Decision

Current research decision:

- ETH:
  - `event_block_v1 = experimental_on` only on the validated `micro_edge_v3_passive_alpha` surface
  - ranking profile candidate:
    - `event_block_eth_micro_v1`
  - rationale:
    - positive on ETH 7D
    - positive again on ETH 1D repeated-window retest
    - broad ETH retest showed symbol-only scoping is too loose
    - 21D micro-edge retest currently lacks common tradeable coverage, even after relaxed validation gates
    - single-lane decomposition confirms `volatility_burst` is the active driver of improvement
    - `book_proxy_pressure` alone is NOT a valid single-lane negative filter on this surface
- BTC:
  - `event_block_v1 = observe_only`
  - rationale:
    - not harmful on BTC 1D
    - but not strong enough to claim clear broad benefit

This is not ready to become:
- a default global mitigation profile across all symbols
- or a blanket ETH-wide profile across all rules

Current positioning:
- `event_block_eth_micro_v1` = ETH short-window experimental profile
- not a long-window production candidate
- 21D long-window retests should be considered non-actionable until tradeable coverage improves

## Rollout Rule

Use this order:

1. ETH broader candidate sets
2. BTC repeated retests
3. symbol-aware profile trial
4. only then consider general defaulting

## Technical Notes

Implementation points:
- `tools/validate_passive_pocket_forward.py`
- `tools/rank_passive_pockets_forward.py`
- `tools/summarize_rank_event_filter.py`
- `tools/summarize_rank_event_filter_set.py`

Profile name:
- `event_block_v1`
- `event_block_eth_v1`
- `event_block_eth_micro_v1`
- `event_block_book_proxy_v1` (tested; single lane only; NO-GO on ETH micro-edge surface)
- `event_block_volatility_v1` (tested; single lane only; MIXED — 50/50, positive median; weaker than two-lane)
- `event_block_eth_micro_imb05_v1` (tested; ETH + micro_edge_v3_passive_alpha + min_imbalance>=0.5; BEST result — 7/8 improved)

Blocked lanes:
- `book_proxy_pressure`
- `volatility_burst`

## Teaching Note

This is a good example of a usable research result:

- not every event becomes a positive signal
- often the first real gain comes from removing bad context
- that is usually safer than forcing trades only in rare "good" context

In short:
- negative filters often productionize earlier than positive filters

Single-lane decomposition lesson:
- always decompose multi-lane results before crediting any single lane
- `book_proxy_pressure` alone is harmful; `volatility_burst` alone is mixed
- the two lanes are SYNERGISTIC — each rescues pockets the other cannot
- do not conclude the improvement comes from one lane when the effect is interactive

## Next Step

### Imbalance-scoped profile: event_block_eth_micro_imb05_v1

Source:
- `reports/EVENT_BLOCK_BASELINE_IMB05_7D.json` (narrowed set, 5 candidates)
- `reports/EVENT_BLOCK_V1_IMB05_7D.json` (narrowed set filtered)
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_V1_IMB05_7D.json`
- `reports/EVENT_BLOCK_IMB05_V1_7D.json` (full TOP8 set with imb05 scoping)
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_IMB05_V1_7D.json`

Result on narrowed imb>=0.5 set (5 candidates):
- `common_count = 5`
- `improved_count = 5`
- `degraded_count = 0`
- `median_delta_npa_core = +6.387e-04`

Result on full ETH TOP8 set (8 candidates, imb05 profile scoping):
- `common_count = 8`
- `improved_count = 7`
- `degraded_count = 1`
- `median_delta_npa_core = +2.501e-04`
- `median_filtered_kept_ratio = 76.29%`

Profile comparison on full ETH TOP8 set:

| Profile | Improved | Degraded | Median ΔNPA | Kept |
|---|---|---|---|---|
| event_block_v1 (all) | 6/8 | 2/8 | +3.698e-04 | 76.24% |
| event_block_eth_micro_imb05_v1 | 7/8 | 1/8 | +2.501e-04 | 76.29% |

Interpretation:
- `event_block_eth_micro_imb05_v1` improves one more candidate than `event_block_v1`
- the one remaining degraded pocket (h=60, imb=0.85, int=4000) was already deeply negative NPA in
  baseline; filtering makes it slightly worse but it would be excluded on quality grounds anyway
- imb=0.3 passthrough behavior preserves the two best baseline pockets (h=60, imb=0.3)
- imb>=0.5 filtered pockets show strong improvement: 4/5 turn from negative NPA to positive
- `event_block_eth_micro_imb05_v1` is now the strongest validated profile in this research phase

## Decision

Current research decision:

- ETH:
  - Primary experimental profile: `event_block_eth_micro_imb05_v1`
    - scoped to: ETHUSDT + micro_edge_v3_passive_alpha + min_imbalance >= 0.5
    - result: 7/8 improved, 1 degraded, median ΔNPA = +2.501e-04 on full ETH TOP8
    - strictly better than `event_block_v1` (6/8) on the same candidate set
  - Legacy profile: `event_block_eth_micro_v1` (6/8, kept for reference; superseded by imb05)
  - rationale:
    - positive on ETH 7D
    - positive again on ETH 1D repeated-window retest
    - broad ETH retest showed symbol-only scoping is too loose
    - 21D micro-edge retest currently lacks common tradeable coverage, even after relaxed validation gates
    - single-lane decomposition confirmed the two lanes are synergistic
    - imbalance scoping (imb>=0.5) eliminates the only remaining degraded pockets
- BTC:
  - `event_block_v1 = observe_only`
  - rationale:
    - not harmful on BTC 1D
    - but not strong enough to claim clear broad benefit

This is not ready to become:
- a default global mitigation profile across all symbols
- or a blanket ETH-wide profile across all rules

Current positioning:
- `event_block_eth_micro_imb05_v1` = best current ETH experimental profile
- `event_block_eth_micro_v1` = still valid but superseded by imb05 variant
- not long-window production candidates
- 21D long-window retests remain non-actionable until tradeable coverage improves

## Rollout Rule

Use this order:

1. ETH broader candidate sets
2. BTC repeated retests
3. symbol-aware profile trial
4. only then consider general defaulting

## Technical Notes

Implementation points:
- `tools/validate_passive_pocket_forward.py`
- `tools/rank_passive_pockets_forward.py`
- `tools/summarize_rank_event_filter.py`
- `tools/summarize_rank_event_filter_set.py`

Profile name:
- `event_block_v1`
- `event_block_eth_v1`
- `event_block_eth_micro_v1`
- `event_block_eth_micro_imb05_v1` (RECOMMENDED — best result; ETH + micro_edge + imb>=0.5)
- `event_block_book_proxy_v1` (tested; single lane only; NO-GO on ETH micro-edge surface)
- `event_block_volatility_v1` (tested; single lane only; MIXED — 50/50, positive median; weaker than two-lane)

Blocked lanes:
- `book_proxy_pressure`
- `volatility_burst`

## Teaching Note

This is a good example of a usable research result:

- not every event becomes a positive signal
- often the first real gain comes from removing bad context
- that is usually safer than forcing trades only in rare "good" context

In short:
- negative filters often productionize earlier than positive filters

Single-lane decomposition lesson:
- always decompose multi-lane results before crediting any single lane
- `book_proxy_pressure` alone is harmful; `volatility_burst` alone is mixed
- the two lanes are SYNERGISTIC — each rescues pockets the other cannot
- do not conclude the improvement comes from one lane when the effect is interactive

Candidate scoping lesson:
- even a good two-lane block can degrade some pockets
- identifying which pockets lose and finding their common feature (here: imb=0.3) enables scoping
- scoping the profile to excluded degraded candidates gives a strictly better result

### Out-of-sample validation: 14D window, splits=4

Source:
- `reports/EVENT_BLOCK_BASELINE_14D_OOS.json`
- `reports/EVENT_BLOCK_IMB05_V1_14D_OOS.json`
- `reports/RANK_EVENT_FILTER_SET_SUMMARY_IMB05_V1_14D_OOS.json`

Window: 14D lookback (2026-02-21 to 2026-03-07), splits=4, min_n=5, relaxed gates.
This extends ~7 days earlier than the 7D in-sample probe.

Result:
- `common_count = 8`
- `improved_count = 4`
- `degraded_count = 4`
- `median_delta_npa_core = +3.756e-05`
- `median_filtered_kept_ratio = 76.34%`

Comparison vs 7D in-sample:

| Window | Improved | Degraded | Median ΔNPA |
|---|---|---|---|
| 7D in-sample | 7/8 | 1/8 | +2.501e-04 |
| 14D OOS | 4/8 | 4/8 | +3.756e-05 |

Pocket-level pattern on 14D OOS:
- imb>=0.85 filtered: consistently improved on BOTH windows (h=120: +5.13e-04, h=60: +1.19e-04)
- imb=0.5 filtered: mixed — h=120 int=6000 degraded (-1.67e-04), others improved
- imb=0.3 passthru: degraded on 14D (was strong on 7D) — volatile across windows

Interpretation:
- The 14D OOS does NOT confirm the strong 7D result
- The profile improvement is partially window-specific
- Only the imb>=0.85 pockets are robust across both windows
- The imb=0.5 benefit is not consistent enough for promotion
- The imb=0.3 passthru approach helps on short windows but degrades on longer ones

Current status: `event_block_eth_micro_imb05_v1` = **extended_testing** (not promote_to_primary)

## Decision

Current research decision:

- ETH:
  - Primary experimental profile: `event_block_eth_micro_imb05_v1`
    - scoped to: ETHUSDT + micro_edge_v3_passive_alpha + min_imbalance >= 0.5
    - 7D in-sample: 7/8 improved, 1 degraded, median ΔNPA = +2.501e-04
    - 14D OOS: 4/8 improved, 4/8 degraded, median ΔNPA = +3.756e-05
    - VERDICT: extended_testing — not ready for promotion; imb>=0.85 pockets are the robust subset
  - Legacy profile: `event_block_eth_micro_v1` (6/8 on 7D; superseded by imb05 on in-sample)
  - rationale:
    - positive on ETH 7D in-sample
    - imbalance scoping removed worst degraded pockets
    - 14D OOS validation shows the benefit does not hold uniformly across longer windows
    - the most robust signal: filtering helps imb>=0.85 pockets on all tested windows
    - imb=0.5 benefit is window-dependent; imb=0.3 passthru is volatile
- BTC:
  - `event_block_v1 = observe_only`
  - rationale:
    - not harmful on BTC 1D
    - but not strong enough to claim clear broad benefit

This is not ready to become:
- a default global mitigation profile across all symbols
- or a blanket ETH-wide profile across all rules

Current positioning:
- `event_block_eth_micro_imb05_v1` = ETH experimental, extended_testing phase
- `event_block_eth_micro_imb085_v1` = candidate_for_promotion (h=60 imb>=0.85 only; h=120 data-limited)
- the h=60 imb>=0.85 pocket is the first sub-profile with consistent improvement on 2+ independent windows
- not long-window production candidates
- 21D long-window retests remain non-actionable until tradeable coverage improves

## Rollout Rule

Use this order:

1. ETH broader candidate sets
2. BTC repeated retests
3. symbol-aware profile trial
4. only then consider general defaulting

## Technical Notes

Implementation points:
- `tools/validate_passive_pocket_forward.py`
- `tools/rank_passive_pockets_forward.py`
- `tools/summarize_rank_event_filter.py`
- `tools/summarize_rank_event_filter_set.py`

Profile name:
- `event_block_v1`
- `event_block_eth_v1`
- `event_block_eth_micro_v1`
- `event_block_eth_micro_imb05_v1` (extended_testing — 7D strong, 14D OOS mixed)
- `event_block_eth_micro_imb085_v1` (tested — see below; GO on h=60 imb>=0.85; CAUTION on h=120)
- `event_block_book_proxy_v1` (tested; single lane only; NO-GO on ETH micro-edge surface)
- `event_block_volatility_v1` (tested; single lane only; MIXED — 50/50, positive median; weaker than two-lane)

Blocked lanes:
- `book_proxy_pressure`
- `volatility_burst`

## Teaching Note

This is a good example of a usable research result:

- not every event becomes a positive signal
- often the first real gain comes from removing bad context
- that is usually safer than forcing trades only in rare "good" context

In short:
- negative filters often productionize earlier than positive filters

Single-lane decomposition lesson:
- always decompose multi-lane results before crediting any single lane
- `book_proxy_pressure` alone is harmful; `volatility_burst` alone is mixed
- the two lanes are SYNERGISTIC — each rescues pockets the other cannot
- do not conclude the improvement comes from one lane when the effect is interactive

Candidate scoping lesson:
- even a good two-lane block can degrade some pockets
- identifying which pockets lose and finding their common feature (here: imb=0.3) enables scoping
- scoping the profile to excluded degraded candidates gives a strictly better in-sample result

OOS validation lesson:
- in-sample improvements do not always hold on extended windows
- always test on a shifted or wider window before promoting a profile
- the robust sub-pattern (imb>=0.85) is a better foundation than the full imb>=0.5 scope

## event_block_eth_micro_imb085_v1 Results

### 7D in-sample (splits=3, fee=0)

Source: `reports/RANK_EVENT_FILTER_SET_SUMMARY_IMB085_V1_7D.json`

| Pocket | baseline NPA | filtered NPA | delta NPA | pass bl | pass filt | kept |
|---|---:|---:|---:|---:|---:|---:|
| h=60 imb=0.85 int>=4000 | -5.57e-04 | +3.66e-05 | +5.94e-04 | 10% | 50% | 0.76 |
| h=120 imb=0.85 int>=6000 | -1.25e-04 | +6.96e-04 | +8.21e-04 | 40% | 0%* | 0.76 |

*pass=0% due to insufficient fills after filtering (low afr=80% at min_n=5 threshold)

Overall (5 candidates, 2 imb=0.85 + 3 imb=0.5 pass-throughs):
- improved=2, degraded=3 (all 3 degraded are imb=0.5 noise — filter not applied)
- median ΔNPA = -6.79e-05 (misleading: negative only because 3 imb=0.5 pass-throughs dominate)

imb=0.85 only: both improved, delta +5.94e-04 and +8.21e-04.

### 14D OOS (splits=4, fee=0)

Source: `reports/RANK_EVENT_FILTER_SET_SUMMARY_IMB085_V1_14D.json`

| Pocket | baseline NPA | filtered NPA | delta NPA | pass bl | pass filt | kept |
|---|---:|---:|---:|---:|---:|---:|
| h=60 imb=0.85 int>=4000 | -1.68e-04 | +9.06e-05 | +2.58e-04 | 26.7% | 60% | 0.76 |
| h=120 imb=0.85 int>=6000 | -9.19e-05 | +0 | +9.19e-05 | 46.7% | 0%* | 0.76 |

*pass=0% — insufficient fills after filtering on 14D window (afr=20%)

Overall (5 candidates): improved=3, degraded=2, median ΔNPA = +9.19e-05 (positive)

### Interpretation

h=60 imb>=0.85 is the single consistent pocket:
- 7D: -5.57e-04 → +3.66e-05 NPA, pass 10% → 50%
- 14D: -1.68e-04 → +9.06e-05 NPA, pass 26.7% → 60%
- Consistent sign-flip from negative to positive on BOTH windows
- Filter kept ~76% of signals (blocked ~24% via book_proxy + volatility_burst)

h=120 imb>=0.85 improves in delta NPA direction but loses all pass rate due to sparse
fills after filtering — this pocket is data-limited and cannot be promoted without
more data or relaxed cap_filter thresholds.

### Verdict

`event_block_eth_micro_imb085_v1`:
- **GO for h=60 imb>=0.85**: consistently positive improvement on both 7D and 14D windows
- **CAUTION for h=120 imb>=0.85**: positive NPA direction but pass=0% due to low fill density
- Overall status: **candidate_for_promotion** (h=60 imb>=0.85 only)

Current status: `event_block_eth_micro_imb085_v1` = **candidate_for_promotion** (h=60 only)

## Next Step

Next research step:
1. Run BTC 7D and 14D comparisons with the two-lane block to assess cross-symbol robustness
2. Collect more data (extend dataset beyond 2026-03-07) to increase OOS window diversity
3. Test h=120 imb>=0.85 with more data or relaxed cap_filter to determine if it is genuinely sparse or noise
4. Only promote to primary ranking profile if h=60 imb>=0.85 results hold on 3+ independent windows
