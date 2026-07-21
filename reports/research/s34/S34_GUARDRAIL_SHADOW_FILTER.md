# S34 Guardrail Shadow Filter

Generated at: `2026-06-23T08:34:58.069958+00:00`

Scope: closed S34 paper trades in `data/s34_intelligence.db`. This is a paper-only counterfactual. It does not change the runner, config, or live rules.

## Overall Scenarios

| Scenario | Kept N | Skipped N | Kept % | Cum Net | Delta vs Base | Median | WR % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_all_closed | 71 | 0 | 100.0 | 1097.71 | 0.0 | 31.09 | 52.11 |
| skip_warning | 44 | 27 | 61.97 | 1004.71 | -93.0 | 49.53 | 61.36 |
| skip_warning_caution | 41 | 30 | 57.75 | 1042.6 | -55.11 | 49.86 | 63.41 |
| only_ok | 34 | 37 | 47.89 | 899.1 | -198.61 | 51.49 | 64.71 |

## Guardrail Level Breakdown

| Level | N | Cum Net | Mean | Median | WR % | Loss % |
| --- | --- | --- | --- | --- | --- | --- |
| caution | 3 | -37.88 | -12.63 | -17.33 | 33.33 | 66.67 |
| ok | 34 | 899.1 | 26.44 | 51.49 | 64.71 | 35.29 |
| unknown | 7 | 143.49 | 20.5 | 34.6 | 57.14 | 42.86 |
| warning | 27 | 93.0 | 3.44 | -11.77 | 37.04 | 62.96 |

## Rule-Level Shadow Result

| Rule | Base N | Base Cum | Base Median | Skip Warning N | Skip Warning Cum | Delta | Only OK N | Only OK Cum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 2 | 42.15 | 21.08 | 1 | -12.87 | -55.02 | 0 | 0.0 |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 2 | -43.3 | -21.65 | 1 | -18.72 | 24.58 | 0 | 0.0 |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 24 | 453.63 | 41.15 | 21 | 409.64 | -43.99 | 18 | 361.28 |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 12 | 482.38 | 52.4 | 12 | 482.38 | 0.0 | 10 | 380.05 |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 24 | -41.93 | -27.65 | 2 | -60.49 | -18.56 | 0 | 0.0 |
| SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 7 | 204.77 | 48.6 | 7 | 204.77 | 0.0 | 6 | 157.77 |

## Warning Trades That Would Have Been Skipped

### Largest Skipped Winners

| Trade | Rule | Exit | Net | Guardrail |
| --- | --- | --- | --- | --- |
| P188 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 125.73 | MODEL WARNING: similar signals have negative expectancy. |
| P111 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 116.02 | MODEL WARNING: similar signals have negative expectancy. |
| P138 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 114.52 | MODEL WARNING: similar signals have negative expectancy. |
| P146 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 113.15 | MODEL WARNING: similar signals have negative expectancy. |
| P060 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TP | 99.67 | MODEL WARNING: similar signals have negative expectancy. |
| P412 | BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | TP | 55.02 | MODEL WARNING: similar signals have negative expectancy. |
| P133 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP | 53.49 | MODEL WARNING: similar signals have negative expectancy. |
| P131 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | TP | 47.69 | MODEL WARNING: similar signals have negative expectancy. |
| P351 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | 31.09 | MODEL WARNING: similar signals have negative expectancy. |
| P353 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | TIME | 10.4 | MODEL WARNING: similar signals have negative expectancy. |

### Largest Skipped Losers

| Trade | Rule | Exit | Net | Guardrail |
| --- | --- | --- | --- | --- |
| P114 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | SL | -57.2 | MODEL WARNING: similar signals have negative expectancy. |
| P065 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -56.76 | MODEL WARNING: similar signals have negative expectancy. |
| P150 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -55.59 | MODEL WARNING: similar signals have negative expectancy. |
| P056 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.45 | MODEL WARNING: similar signals have negative expectancy. |
| P418 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -53.37 | MODEL WARNING: similar signals have negative expectancy. |
| P169 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -51.87 | MODEL WARNING: similar signals have negative expectancy. |
| P419 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -49.36 | MODEL WARNING: similar signals have negative expectancy. |
| P416 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -48.03 | MODEL WARNING: similar signals have negative expectancy. |
| P063 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -47.76 | MODEL WARNING: similar signals have negative expectancy. |
| P149 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | SL | -46.3 | MODEL WARNING: similar signals have negative expectancy. |

## Read

If skipping warnings improves cumulative net while discarding many winners, the guardrail is useful but too blunt. If `only_ok` is strong but low-N, it is a candidate for a separate pre-registered validation gate, not an immediate production filter.
