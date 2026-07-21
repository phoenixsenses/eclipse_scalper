# S34 REGIME_FILTER Shadow Evaluation

- generated_at_utc: `2026-06-19T15:15:21.202798+00:00`
- window_hours: `72.0`
- cutoff_utc: `2026-06-16T15:15:20.897000+00:00`
- recent_trials: `108`
- recent_closed: `2`
- regime_filter_skips: `106`
- simulated: `106`
- complete_horizon: `105`
- incomplete_horizon: `1`

## Complete-Horizon Summary
| N | cum net bps | mean | median | win rate | exits |
| --- | --- | --- | --- | --- | --- |
| 105 | 713.58 | 6.80 | -8.79 | 40.00% | {"BE": 31, "SL": 25, "TIME": 27, "TP": 22} |

## By Rule
| rule | N | cum net bps | mean | median | win rate | exits |
| --- | --- | --- | --- | --- | --- | --- |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 11 | -19.83 | -1.80 | -11.79 | 18.18% | {"BE": 5, "SL": 2, "TIME": 3, "TP": 1} |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 21 | 155.28 | 7.39 | -8.06 | 47.62% | {"BE": 5, "SL": 5, "TIME": 2, "TP": 9} |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 5 | 106.68 | 21.34 | 49.82 | 60.00% | {"BE": 1, "SL": 1, "TP": 3} |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 68 | 471.45 | 6.93 | -9.26 | 39.71% | {"BE": 20, "SL": 17, "TIME": 22, "TP": 9} |

## Regime Fail Combos
| count | failed checks |
| --- | --- |
| 36 | trend_pct_gte |
| 28 | trend_pct_gte, buy_liq_notional_gte |
| 22 | trend_pct_gte, range_pct_gte, buy_liq_notional_gte |
| 15 | trend_pct_gte, range_pct_gte, buy_liq_notional_gte, agg_trade_count_gte |
| 5 | day_trend_bps_gte |

## Top Counterfactuals
| trade | rule | signal utc | net bps | exit | MFE | MAE | liq notional |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P199 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-16T17:34:08.586000+00:00 | 120.96 | TP | 125.08 | -2.87 | 61699 |
| P229 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T15:35:58.291000+00:00 | 116.03 | TP | 124.61 | -4.72 | 55586 |
| P230 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T15:48:24.789000+00:00 | 115.71 | TP | 122.35 | -3.71 | 62203 |
| P299 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-19T12:51:02.770000+00:00 | 113.74 | TP | 122.57 | -0.18 | 235531 |
| P282 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T19:33:10.617000+00:00 | 113.05 | TP | 119.93 | -3.55 | 76495 |
| P248 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T18:06:12.942000+00:00 | 112.86 | TP | 121.37 | 0.17 | 394725 |
| P283 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T19:38:08.137000+00:00 | 111.71 | TP | 120.08 | -0.06 | 72993 |
| P200 | ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 2026-06-16T17:46:04.065000+00:00 | 109.12 | TP | 120.42 | -0.25 | 1072950 |
| P203 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-16T17:46:04.065000+00:00 | 107.83 | TP | 121.11 | -9.30 | 1072950 |
| P213 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T01:24:09.770000+00:00 | 106.75 | TP | 116.64 | -28.11 | 218881 |

## Worst Counterfactuals
| trade | rule | signal utc | net bps | exit | MFE | MAE | liq notional |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P243 | ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 2026-06-17T17:55:02.637000+00:00 | -61.25 | SL | 18.73 | -49.07 | 447125 |
| P244 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T17:55:02.637000+00:00 | -61.25 | SL | 18.73 | -49.07 | 447125 |
| P271 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T10:52:45.564000+00:00 | -52.71 | SL | 27.94 | -41.22 | 66846 |
| P210 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-16T18:50:21.497000+00:00 | -52.64 | SL | 6.77 | -41.35 | 96478 |
| P292 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T20:20:15.248000+00:00 | -52.26 | SL | 9.62 | -40.45 | 158294 |
| P280 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T17:05:12.673000+00:00 | -51.35 | SL | 3.84 | -40.85 | 121770 |
| P295 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-19T01:05:05.104000+00:00 | -50.72 | SL | 1.57 | -40.04 | 128003 |
| P279 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T16:20:13.080000+00:00 | -50.01 | SL | 17.24 | -41.77 | 92743 |
| P277 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-18T14:23:57.616000+00:00 | -49.94 | SL | 24.20 | -41.89 | 127094 |
| P251 | ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 2026-06-17T18:21:40.598000+00:00 | -49.79 | SL | 11.21 | -39.98 | 65691 |

## Interpretation Guardrail

This is a shadow/counterfactual diagnostic only. It does not alter the live runner, does not count toward the pre-registered sample, and should not be used to retune gates without a separate OOS/real-fill validation.
