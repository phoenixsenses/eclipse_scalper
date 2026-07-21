# S34 Knowable-Anchor TP/SL/BE Route Recheck

Generated: `2026-06-28T17:14:11.942694+00:00`

RESEARCH_ONLY. Exact frozen route exits rerun on real-time-knowable running-notional anchors. No live/executor changes.

## Summary

| Family | Rule | Anchors | Filtered | Cal N | Cal Median | Cal Mean | Cal T3R | Hold N | Hold Median | Hold Mean | Hold T3R | No-fill Cal/Hold | Mark CF Cal/Hold | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ETH_BUY | `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` | 1354 | 1354 | 195 | -12.1 | -11.1 | -2546.9 | 292 | -9.8 | -0.8 | -641.9 | 0.794 / 0.281 | -7.9 / -8.5 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 547 | 547 | 112 | -9.8 | -10.0 | -1330.9 | 123 | -9.2 | 0.5 | -171.8 | 0.708 / 0.25 | -8.1 / -7.4 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60` | 547 | 479 | 100 | -9.2 | -10.0 | -1373.3 | 107 | -6.4 | 10.2 | 667.4 | 0.701 / 0.257 | -7.5 / -6.8 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | 215 | 150 | 46 | -27.4 | -13.3 | -803.9 | 36 | -12.1 | 5.0 | -23.6 | 0.562 / 0.2 | -7.5 / -9.1 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30` | 215 | 9 | 2 | -4.6 | -4.6 | -9.1 | 3 | 55.0 | 25.1 | 75.4 | 0.667 / 0.0 | -26.6 / 55.0 | BLOCKED_THIN_CALIBRATION |
| SOL_BUY | `SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30` | 115 | 115 | 58 | -6.7 | 2.2 | -95.0 | 34 | -15.6 | -6.5 | -419.0 | 0.284 / 0.0 | -6.3 / -7.7 | RESEARCH_ONLY |
| SOL_BUY | `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 78 | 78 | 39 | -13.0 | -6.9 | -483.7 | 23 | -6.1 | 1.2 | -172.1 | 0.291 / 0.0 | -9.0 / -7.0 | RESEARCH_ONLY |
| BTC_BUY_DISTRIBUTED | `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30` | 128 | 53 | 22 | -12.2 | -4.2 | -276.3 | 16 | -6.9 | -1.0 | -183.5 | 0.405 / 0.0 | -36.2 / -6.7 | RESEARCH_ONLY |
| ETH_SELL | `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40` | 225 | 225 | 57 | -20.8 | -10.2 | -783.1 | 68 | -7.7 | 3.5 | -7.8 | 0.637 / 0.0 | -26.2 / -6.5 | RESEARCH_ONLY |
| ETH_SELL | `ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40` | 117 | 117 | 44 | -12.8 | -12.0 | -764.4 | 35 | -7.8 | 13.1 | 109.8 | 0.463 / 0.0 | -8.3 / -6.7 | RESEARCH_ONLY |
| SOL_SELL | `SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40` | 111 | 111 | 57 | -36.6 | -11.2 | -870.3 | 33 | -36.8 | -3.6 | -321.0 | 0.269 / 0.0 | -36.3 / -36.2 | RESEARCH_ONLY |
| SOL_SELL | `SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30` | 61 | 61 | 31 | -15.6 | -11.2 | -506.7 | 18 | 51.9 | 25.3 | 201.8 | 0.279 / 0.0 | -13.6 / 54.3 | RESEARCH_ONLY |

## Read

- `Cal/Hold T3R` is top-3-winner-removed cumulative net bps.
- `Mark CF` is mark-price counterfactual median on all accepted anchors, useful for separating directional signal from executable-fill coverage.
- `PAPER_CANDIDATE` requires positive calibration and holdout median/mean/T3R with minimum filled N; otherwise the route remains `RESEARCH_ONLY` or `BLOCKED`.
