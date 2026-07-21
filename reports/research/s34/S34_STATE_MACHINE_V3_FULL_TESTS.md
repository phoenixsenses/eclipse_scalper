# S34 State Machine V3 Full Tests

- generated_at_utc: `2026-06-30T17:22:19.501780+00:00`
- research_only: `True`
- primary_config: `btc750_dow_score3`
- primary_hold: `{'n': 32, 'wr': 0.781, 'sum': 3359.0, 'mean': 105.0, 'median': 72.8, 't3r': 2299.4, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 70.5}`

## Executive Read

- provisional_best: `{'score': 't0_score_ge2', 'variant': 'confirmed_only_long_t30_or_short_btc750', 'hold': {'n': 55, 'wr': 0.6, 'sum': 3244.3, 'mean': 59.0, 'median': 19.4, 't3r': 1987.1, 'max_loss': -447.7, 'max_win': 559.1, 'max_dd_bps': 447.7}}`
- execution_book_hold: `{'n': 30, 'wr': 0.7, 'sum': 3344.1, 'mean': 111.5, 'median': 92.1, 't3r': 2276.0, 'max_loss': -53.7, 'max_win': 380.6, 'max_dd_bps': 73.0}`
- slippage_20bps_hold: `{'n': 32, 'wr': 0.594, 'sum': 2719.0, 'mean': 85.0, 'median': 52.8, 't3r': 1719.4, 'max_loss': -72.0, 'max_win': 350.0, 'max_dd_bps': 130.5}`
- shadow_id_parity: `1.0`

## Top Provisional / State-Resolution Variants

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t0_score_ge2:confirmed_only_long_t30_or_short_btc750 | 55 | 60.0% | 3244.3 | 59.0 | 19.4 | 1987.1 | -447.7 | 447.7 |
| t0_score_ge1:confirmed_only_long_t30_or_short_btc750 | 64 | 56.2% | 3197.6 | 50.0 | 9.5 | 1940.3 | -447.7 | 459.9 |
| t0_score_ge0:confirmed_only_long_t30_or_short_btc750 | 68 | 52.9% | 2951.7 | 43.4 | 6.1 | 1694.5 | -447.7 | 550.1 |
| t0_score_ge3:confirmed_only_long_t30_or_short_btc750 | 37 | 64.9% | 2621.3 | 70.8 | 35.8 | 1608.6 | -447.7 | 447.7 |
| t0_score_ge4:confirmed_only_long_t30_or_short_btc750 | 21 | 71.4% | 2518.0 | 119.9 | 137.9 | 1505.3 | -66.1 | 66.1 |
| t0_score_ge2:long_flip_short_on_eth_noisy | 107 | 52.3% | 2430.6 | 22.7 | 7.2 | 1105.6 | -452.3 | 575.0 |
| t0_score_ge1:long_flip_short_on_eth_noisy | 123 | 52.8% | 2317.8 | 18.8 | 7.2 | 992.8 | -452.3 | 597.1 |
| t0_score_ge4:long_flip_short_on_eth_noisy | 42 | 57.1% | 1877.5 | 44.7 | 26.6 | 842.6 | -197.2 | 428.7 |
| t0_score_ge4:long_flip_short_on_btc1000_else_exit | 42 | 35.7% | 1650.5 | 39.3 | -14.2 | 698.3 | -119.4 | 317.3 |
| t0_score_ge4:long_flip_short_on_btc750_else_exit | 42 | 35.7% | 1492.7 | 35.5 | -17.2 | 540.5 | -127.7 | 429.6 |
| t0_score_ge0:long_flip_short_on_eth_noisy | 130 | 50.8% | 1736.4 | 13.4 | 1.8 | 411.5 | -452.3 | 779.1 |
| t0_score_ge3:long_flip_short_on_eth_noisy | 72 | 48.6% | 1357.2 | 18.8 | -5.2 | 322.2 | -452.3 | 595.0 |

## Conflict Variants

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_independent | 33 | 78.8% | 3466.2 | 105.0 | 76.7 | 2406.5 | -52.0 | 52.0 |
| one_pos_ignore | 21 | 81.0% | 2730.4 | 130.0 | 106.5 | 1670.8 | -52.0 | 52.0 |
| short_replace | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| same_side_timer_reset | 21 | 71.4% | 1535.2 | 73.1 | 16.5 | 550.7 | -52.0 | 70.5 |

## Monthly Stability

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02 | 9 | 66.7% | 305.2 | 33.9 | 48.4 | -303.5 | -318.3 | 318.3 |
| 2026-03 | 20 | 65.0% | 1197.9 | 59.9 | 64.5 | 435.4 | -197.9 | 274.0 |
| 2026-04 | 13 | 30.8% | -173.7 | -13.4 | -26.1 | -516.9 | -121.2 | 516.9 |
| 2026-06 | 35 | 80.0% | 3592.3 | 102.6 | 68.9 | 2532.7 | -52.0 | 70.5 |

## DOW Stability

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mon | 1 | 100.0% | 46.2 | 46.2 | 46.2 | 46.2 | 46.2 | 0.0 |
| Tue | 18 | 66.7% | 407.3 | 22.6 | 23.6 | -98.9 | -197.9 | 357.1 |
| Wed | 9 | 55.6% | 889.0 | 98.8 | 115.2 | 150.6 | -52.0 | 79.3 |
| Thu | 16 | 68.8% | 1085.6 | 67.9 | 41.5 | 239.8 | -121.2 | 187.9 |
| Fri | 17 | 76.5% | 1704.0 | 100.2 | 95.8 | 802.7 | -123.5 | 123.5 |
| Sat | 8 | 37.5% | -84.6 | -10.6 | -17.9 | -454.2 | -318.3 | 318.3 |
| Sun | 8 | 75.0% | 874.2 | 109.3 | 132.2 | 139.4 | -107.7 | 141.6 |

## Exit Horizon Holdout

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h1 | 32 | 53.1% | 1230.0 | 38.4 | 13.5 | 373.0 | -105.5 | 192.6 |
| h1.5 | 32 | 68.8% | 1853.6 | 57.9 | 24.7 | 1037.2 | -72.3 | 133.6 |
| h2 | 32 | 71.9% | 2544.6 | 79.5 | 32.8 | 1603.4 | -52.0 | 75.3 |
| h2.5 | 32 | 75.0% | 2863.7 | 89.5 | 97.7 | 1923.0 | -122.1 | 122.1 |
| h3 | 32 | 68.8% | 3011.2 | 94.1 | 81.7 | 2081.7 | -281.0 | 281.0 |
| h4 | 32 | 78.1% | 3625.1 | 113.3 | 59.6 | 2151.8 | -271.1 | 271.2 |

## Stop Sweep Holdout

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sl40 | 32 | 53.1% | 2070.2 | 64.7 | 6.1 | 1055.8 | -45.0 | 213.0 |
| sl60 | 32 | 56.2% | 1884.1 | 58.9 | 9.0 | 869.7 | -65.0 | 260.0 |
| sl80 | 32 | 65.6% | 2245.6 | 70.2 | 25.3 | 1231.2 | -85.0 | 295.0 |
| sl100 | 32 | 75.0% | 3074.0 | 96.1 | 52.3 | 2014.4 | -105.0 | 175.5 |
| sl150 | 32 | 78.1% | 3244.2 | 101.4 | 72.8 | 2184.6 | -155.0 | 155.0 |
| sl200 | 32 | 78.1% | 3194.2 | 99.8 | 72.8 | 2134.6 | -205.0 | 205.0 |

## Latency Holdout

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| delay_0s | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| delay_5s | 32 | 71.9% | 3354.3 | 104.8 | 72.9 | 2303.2 | -61.9 | 63.0 |
| delay_15s | 32 | 71.9% | 3296.1 | 103.0 | 69.7 | 2265.2 | -65.1 | 65.1 |
| delay_30s | 32 | 81.2% | 3402.2 | 106.3 | 69.0 | 2359.5 | -59.1 | 59.1 |
| delay_60s | 32 | 78.1% | 3311.4 | 103.5 | 71.8 | 2242.2 | -62.4 | 62.4 |

## Slippage Holdout

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| slip_0bps | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| slip_5bps | 32 | 75.0% | 3199.0 | 100.0 | 67.8 | 2154.4 | -57.0 | 85.5 |
| slip_10bps | 32 | 68.8% | 3039.0 | 95.0 | 62.8 | 2009.4 | -62.0 | 100.5 |
| slip_20bps | 32 | 59.4% | 2719.0 | 85.0 | 52.8 | 1719.4 | -72.0 | 130.5 |
| slip_30bps | 32 | 59.4% | 2399.0 | 75.0 | 42.8 | 1429.4 | -82.0 | 160.5 |

## Book Realism Holdout

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| book_stale_1s | 30 | 70.0% | 3344.1 | 111.5 | 92.1 | 2276.0 | -53.7 | 73.0 |
| book_stale_5s | 30 | 70.0% | 3344.1 | 111.5 | 92.1 | 2276.0 | -53.7 | 73.0 |
| book_stale_10s | 30 | 70.0% | 3344.1 | 111.5 | 92.1 | 2276.0 | -53.7 | 73.0 |
| book_stale_30s | 31 | 71.0% | 3363.5 | 108.5 | 76.5 | 2295.4 | -53.7 | 73.0 |

## Markov / Transition Summary

| Transition | N | P from previous | WR | Mean | T3R |
| --- | ---: | ---: | ---: | ---: | ---: |
| SILENCE->SILENCE | 74 | 0.446 | 66.2% | 38.3 | 1751.7 |
| NOISY->SILENCE | 83 | 0.407 | 56.6% | 36.1 | 1742.1 |
| NOISY->NEITHER | 23 | 0.113 | 65.2% | 68.5 | 638.1 |
| NEITHER->NEITHER | 15 | 0.306 | 86.7% | 69.5 | 425.1 |
| SILENCE->NEITHER | 11 | 0.066 | 63.6% | 123.3 | 314.6 |
| NEITHER->SILENCE | 10 | 0.204 | 60.0% | 50.9 | 74.7 |
| SILENCE->NOISY | 81 | 0.488 | 51.9% | 9.1 | -108.3 |
| NEITHER->NOISY | 24 | 0.49 | 45.8% | 7.4 | -491.3 |
| NOISY->NOISY | 98 | 0.48 | 45.9% | 1.1 | -727.0 |

## Tail Cluster

- tail_count: `6`
- next_after_tail: `{'n': 6, 'wr': 0.667, 'sum': 498.4, 'mean': 83.1, 'median': 100.3, 't3r': -107.6, 'max_loss': -105.3, 'max_win': 260.8, 'max_dd_bps': 162.8}`
- next_after_win: `{'n': 50, 'wr': 0.7, 'sum': 3513.1, 'mean': 70.3, 'median': 39.0, 't3r': 2428.7, 'max_loss': -318.3, 'max_win': 370.0, 'max_dd_bps': 441.8}`

## Shadow Timestamp Parity

- {'exists': True, 'ledger_backfill_closes': 503, 'expected_ids': 503, 'matching_ids': 503, 'missing_expected_ids': 0, 'extra_ledger_ids': 0, 'parity_ratio': 1.0, 'note': 'ID-level parity only; P&L parity differs because backfill uses NAV labels while this suite recomputes mark/book outcomes.'}

## Interpretation

State-machine still looks strongest as provisional-entry plus state-resolution. The biggest live blocker remains not statistical edge, but executable parity and the fact that SILENCE is not knowable at T=0.
