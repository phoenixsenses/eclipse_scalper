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

## Next Step

Next research step:
1. Validate `event_block_eth_micro_imb05_v1` on a fresh out-of-sample window (different date range
   or different lookback start) to test whether the improvement holds
2. Investigate whether the h=60, imb=0.85 degradation is structural (short-horizon + extreme imbalance
   in volatile markets = good entry) or noise
3. Repeat BTC on more windows with the two-lane profile
4. Then decide whether to promote `event_block_eth_micro_imb05_v1` to the primary ranking profile
