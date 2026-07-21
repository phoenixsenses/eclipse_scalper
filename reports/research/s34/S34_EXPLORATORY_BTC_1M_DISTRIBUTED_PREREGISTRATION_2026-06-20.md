# S34 Exploratory BTC 1M Distributed Pre-Registration

**Date:** 2026-06-20  
**Status:** exploratory live paper only  
**Rule:** `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30`  
**Scope:** isolated from the ETH 50K/TP120 pre-registered validation sample.

## Rule Definition

- Symbol: `BTCUSDT`
- Liquidation side: `BUY`
- Direction: `LONG`
- Cluster notional threshold: `>= 1,000,000 USDT`
- Cluster geometry gate: `cluster_max_single_liq_share < 50%`
- TP: `+60 bps`
- SL: `-30 bps`
- BE trigger: `+30 bps`
- Entry: taker, executable bookTicker fill
- Exit: taker TP/SL/BE/time under the existing runner fill model

## Why This Rule Exists

This rule is promoted only to exploratory forward paper because it survived a research scan better than the broad BTC 1M baseline:

| Candidate | Real Test N | Median Net bps | Mean Net bps | Cum Net bps | Top3 Removed | Notes |
|---|---:|---:|---:|---:|---:|---|
| BTC 1M TP60/SL30/BE30 baseline | 33 | +28.55 | +17.69 | +583.84 | +394.16 | broad candidate |
| BTC 1M distributed, max single share <50 | 21 | +47.27 | +30.94 | +649.84 | +469.04 | cleaner but smaller N |

The distributed filter is plausible because it selects clusters where forced flow is spread across several liquidation prints instead of dominated by one isolated order. That said, it was discovered through a research scan, so overfitting risk remains material.

## Forward Evaluation Criteria

No live tuning is allowed from this sample. The rule remains exploratory unless a future forward report shows:

- At least `N >= 30` valid closed trades.
- At least `8` distinct trading days.
- Median net bps > 0.
- Cumulative net bps remains positive after removing the top 3 winners.
- `NO_FILL_DATA` quarantine is low and not concentrated in the highest-intensity clusters.
- Results are not dominated by one day or one outlier cluster.

Failure on these criteria means the rule is not promoted. It may be retired or returned to research.

## Non-Contamination Clause

This rule does not count toward:

- ETH 50K/TP120 pre-registered N/40.
- ETH 200K/TP60 exploratory evidence.
- ETH 500K/daytrend exploratory evidence.
- SOL 200K/TP60 exploratory evidence.

It has its own rule id, own risk-gate slot, and own dashboard card.
