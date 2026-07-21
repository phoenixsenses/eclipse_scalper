# S34 Risk Engineering: Same-Cluster Dedupe

Date: 2026-06-21

## Scope

This change is risk/exposure engineering only. It does not change S34 signal thresholds, TP/SL/BE values, fill pricing, or cost attribution.

## Problem

On 2026-06-20, multiple S34 rule variants opened the same ETH BUY liquidation cluster as separate long positions. The clearest failure was the 15:20 UTC cluster:

| Trade | Rule | Result |
| --- | --- | ---: |
| P346 | ETH 200K/TP60 | -51.64 bps |
| P347 | ETH 500K/daytrend | -51.64 bps |
| P348 | ETH 50K/TP120 | -51.64 bps |

These were not independent ideas. They were the same symbol, same direction, same liquidation-side, same 5-minute cluster. Counting and trading them separately tripled exposure to one event.

## Fix

The runner now enforces two forward-only guards:

1. Same-cluster priority dedupe:
   - Cluster key: `(symbol, direction, liq_side, bucket)`.
   - If multiple rules fire on the same cluster, only the highest-priority rule can open.
   - Lower-priority rules are journaled as `SAME_CLUSTER_LOWER_PRIORITY`.

2. Same symbol/direction open-position cap:
   - If a same-symbol, same-direction trade is already open, another rule cannot open a second position.
   - The skip reason is `MAX_SYMBOL_DIRECTION_OPEN_TRADES`.

## Current Priority Order

Lower number means higher priority.

| Rule | Priority |
| --- | ---: |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 10 |
| ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30 | 20 |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 30 |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 40 |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 50 |
| SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 10 |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 10 |

## Validation

Unit tests cover:

- Same-cluster lower-priority skip.
- Same-symbol/same-direction open-position block across rules.
- Existing same-rule `MAX_OPEN_TRADES` behavior.
- Existing fill, BE chronology, regime, BTC pre-filter, and geometry tests.

Command:

```powershell
python -m pytest tests\test_s34_shadow_paper_runner.py -q
```

Result: 16 passed.

## Interpretation

Future closed-trade counts will be lower during dense ETH regime days, but the sample will better represent independent event exposure. This is not edge tuning; it prevents duplicated risk from the same liquidation cluster.
