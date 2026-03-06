# RESEARCH LIQUIDATION REVERSAL SURFACE NOTES

## What Changed

- Added `tools/generate_liq_reversal_candidates.py` to create a rule-specific candidate surface for `high_liq_reversal_regime`.
- The generator emits:
  - markdown candidate sheet for `tools.rank_passive_pockets_forward`
  - JSON report with `run_summary`

## Why

`high_liq_reversal_regime` has measurable raw coverage, but it was previously evaluated inside a candidate surface tuned for `micro_edge_v3_passive_alpha`.

That mixes two different questions:

1. does the rule fire often enough?
2. does the current pocket formulation leave enough tradeable samples after filters?

The new generator isolates question 2.

## Real Findings

Using the new small research surface on live ETHUSDT data:

- candidate count: `8`
- raw parse: `8/8` candidates accepted
- rank result: all candidates were rejected by `insufficient_fill_rate`

Two diagnostic passes were tried:

1. `splits=3`, `min_n_frac=0.00005`
2. `splits=2`, `min_n=20`, `min_n_frac=0.00001`

Both still failed with:

- `insufficient_fill_rate = 1.0`
- `survive_fee>=1.0 with pass_rate>=0.5 = 0`

## Interpretation

This is different from low raw rule coverage.

Coverage audit already showed the rule fires often enough:

- 60 min: `20` fires
- 1 day: `584` fires
- 7 day: `5238` fires

So the current bottleneck is:

- fillability
- passive execution realism
- current horizon / spread / intensity shape

not raw event scarcity.

## Teaching Note

`coverage` and `tradeable coverage` are different.

A rule can be statistically present in the data, but still be unusable once:

- entry timing
- passive fill assumptions
- cost model
- split stability requirements

are applied.

That means the next iteration should not just loosen thresholds blindly. It should revisit the execution shape of the rule itself.

## Next Research Direction

Prefer one of these instead of widening the same surface again:

1. longer horizon liquidation reversal pockets
2. semi-passive or maker-then-scratch execution shape
3. alert-only regime tagging before trade simulation
4. separate fee/slippage assumptions specific to liquidation events
