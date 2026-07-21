# S34 Knowable-Anchor TP/SL/BE Route Recheck

Generated: `2026-06-28T12:00:41.563051+00:00`

RESEARCH_ONLY. Exact frozen route exits rerun on real-time-knowable running-notional anchors. No live/executor changes.

## Summary

| Family | Rule | Anchors | Filtered | Cal N | Cal Median | Cal Mean | Cal T3R | Hold N | Hold Median | Hold Mean | Hold T3R | No-fill Cal/Hold | Mark CF Cal/Hold | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ETH_BUY | `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` | 201 | 201 | 140 | -10.7 | 0.5 | -321.6 | 59 | -10.0 | -6.9 | -770.8 | 0.007 / 0.017 | -7.6 / -7.2 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 95 | 95 | 67 | -14.0 | -1.8 | -344.9 | 28 | -7.7 | 3.5 | -83.8 | 0.0 / 0.0 | -7.9 / -7.2 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60` | 95 | 81 | 57 | -8.8 | -3.0 | -527.6 | 24 | -5.1 | 18.8 | 70.3 | 0.0 / 0.0 | -6.9 / -6.5 | RESEARCH_ONLY |
| ETH_BUY | `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | 49 | 23 | 16 | -37.8 | -5.0 | -249.3 | 7 | -17.7 | 1.7 | -151.3 | 0.0 / 0.0 | -44.4 / -9.6 | BLOCKED_THIN_CALIBRATION |
| SOL_BUY | `SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30` | 50 | 50 | 35 | -10.4 | -7.3 | -466.2 | 15 | 52.5 | 15.5 | 36.6 | 0.0 / 0.0 | -7.4 / 53.9 | RESEARCH_ONLY |
| SOL_BUY | `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 30 | 30 | 21 | -6.1 | -1.7 | -247.1 | 9 | -7.5 | -2.6 | -192.5 | 0.0 / 0.0 | -7.5 / -7.1 | RESEARCH_ONLY |
| BTC_BUY_DISTRIBUTED | `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30` | 28 | 12 | 8 | -22.0 | -6.6 | -158.6 | 4 | -13.3 | -6.0 | -42.0 | 0.0 / 0.0 | -21.3 / -16.7 | BLOCKED_THIN_CALIBRATION |
| ETH_SELL | `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40` | 57 | 57 | 40 | 10.8 | 10.5 | 174.8 | 17 | -8.1 | -1.1 | -212.3 | 0.0 / 0.0 | 11.0 / -6.2 | RESEARCH_ONLY |
| ETH_SELL | `ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40` | 38 | 38 | 27 | -6.1 | 17.0 | 131.1 | 11 | -9.2 | 5.4 | -185.9 | 0.0 / 0.0 | -6.4 / -6.7 | RESEARCH_ONLY |
| SOL_SELL | `SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40` | 59 | 59 | 41 | -37.9 | -11.8 | -741.9 | 18 | -14.9 | 3.9 | -95.3 | 0.0 / 0.0 | -36.4 / -14.7 | RESEARCH_ONLY |
| SOL_SELL | `SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30` | 35 | 35 | 25 | -10.2 | 4.6 | -139.7 | 10 | -17.9 | 0.5 | -159.6 | 0.0 / 0.0 | -6.8 / -0.1 | RESEARCH_ONLY |

## Read

- `Cal/Hold T3R` is top-3-winner-removed cumulative net bps.
- `Mark CF` is mark-price counterfactual median on all accepted anchors, useful for separating directional signal from executable-fill coverage.
- `PAPER_CANDIDATE` requires positive calibration and holdout median/mean/T3R with minimum filled N; otherwise the route remains `RESEARCH_ONLY` or `BLOCKED`.
