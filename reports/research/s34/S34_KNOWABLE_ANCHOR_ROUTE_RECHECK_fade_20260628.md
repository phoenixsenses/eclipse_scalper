# S34 Knowable-Anchor TP/SL/BE Route Recheck

Generated: `2026-06-28T17:33:31.463513+00:00`

RESEARCH_ONLY. Exact frozen route exits rerun on real-time-knowable running-notional anchors. No live/executor changes.

## Summary

| Family | Rule | Anchors | Filtered | Cal N | Cal Median | Cal Mean | Cal T3R | Hold N | Hold Median | Hold Mean | Hold T3R | No-fill Cal/Hold | Mark CF Cal/Hold | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ETH_BUY_FADE | `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30_FADE` | 1354 | 1354 | 199 | -8.2 | -7.7 | -1888.8 | 292 | -8.8 | -11.5 | -3728.9 | 0.79 / 0.281 | -7.9 / -7.4 | RESEARCH_ONLY |
| ETH_BUY_FADE | `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30_FADE` | 547 | 547 | 112 | -8.1 | -5.9 | -837.6 | 123 | -18.5 | -12.1 | -1686.6 | 0.708 / 0.25 | -8.0 / -15.9 | RESEARCH_ONLY |
| ETH_BUY_FADE | `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60_FADE` | 547 | 479 | 101 | -8.4 | -2.3 | -595.0 | 107 | -13.5 | -11.0 | -1533.8 | 0.699 / 0.257 | -7.8 / -8.4 | RESEARCH_ONLY |
| ETH_BUY_FADE | `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30_FADE` | 215 | 150 | 49 | -11.2 | -1.4 | -253.1 | 36 | -27.1 | -15.6 | -743.3 | 0.533 / 0.2 | -7.8 / -11.6 | RESEARCH_ONLY |
| ETH_BUY_FADE | `ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30_FADE` | 215 | 9 | 2 | -2.0 | -2.0 | -4.0 | 3 | -48.0 | -41.6 | -124.7 | 0.667 / 0.0 | -8.7 / -47.8 | BLOCKED_THIN_CALIBRATION |
| SOL_BUY_FADE | `SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30_FADE` | 115 | 115 | 58 | -33.0 | -13.8 | -975.3 | 34 | -11.7 | -6.4 | -420.7 | 0.284 / 0.0 | -12.7 / -6.7 | RESEARCH_ONLY |
| SOL_BUY_FADE | `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30_FADE` | 78 | 78 | 39 | -13.0 | -17.0 | -828.1 | 23 | -13.0 | -10.1 | -433.4 | 0.291 / 0.0 | -9.2 / -7.6 | RESEARCH_ONLY |
| BTC_BUY_DISTRIBUTED_FADE | `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30_FADE` | 128 | 53 | 22 | -34.4 | -9.7 | -391.1 | 16 | -36.3 | -16.3 | -396.2 | 0.405 / 0.0 | -6.9 / -36.9 | RESEARCH_ONLY |
| ETH_SELL_FADE | `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40_FADE` | 225 | 225 | 56 | -6.2 | -1.4 | -275.6 | 68 | -41.5 | -14.8 | -1198.7 | 0.643 / 0.0 | -6.5 / -46.3 | RESEARCH_ONLY |
| ETH_SELL_FADE | `ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40_FADE` | 117 | 117 | 44 | -8.5 | -5.1 | -473.3 | 35 | -44.8 | -16.2 | -801.2 | 0.463 / 0.0 | -11.8 / -46.4 | RESEARCH_ONLY |
| SOL_SELL_FADE | `SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40_FADE` | 111 | 111 | 57 | -7.5 | -0.1 | -196.8 | 33 | -18.8 | -5.2 | -346.4 | 0.269 / 0.0 | -7.2 / -18.9 | RESEARCH_ONLY |
| SOL_SELL_FADE | `SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30_FADE` | 61 | 61 | 30 | -35.3 | -2.5 | -253.7 | 18 | -36.0 | -20.4 | -538.6 | 0.302 / 0.0 | -36.1 / -37.0 | RESEARCH_ONLY |

## Read

- `Cal/Hold T3R` is top-3-winner-removed cumulative net bps.
- `Mark CF` is mark-price counterfactual median on all accepted anchors, useful for separating directional signal from executable-fill coverage.
- `PAPER_CANDIDATE` requires positive calibration and holdout median/mean/T3R with minimum filled N; otherwise the route remains `RESEARCH_ONLY` or `BLOCKED`.
