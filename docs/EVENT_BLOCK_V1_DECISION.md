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
- BTC:
  - `event_block_v1 = observe_only`
  - rationale:
    - not harmful on BTC 1D
    - but not strong enough to claim clear broad benefit

This is not ready to become:
- a default global mitigation profile across all symbols
- or a blanket ETH-wide profile across all rules

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

## Next Step

Next research step:
- run broader `micro_edge_v3_passive_alpha` ETH candidate universes
- repeat BTC on more windows
- then decide whether `event_block_v1` should become:
  - a rule-aware ETH experimental profile in ranking
  - or a symbol/rule-aware profile family such as `event_block_eth_micro_v1`
