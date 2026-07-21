# S34 State Machine V2 Gauntlet

- generated_at_utc: `2026-06-30T16:46:36.295368+00:00`
- events: `450` | cal: `315` | hold: `135`
- holdout_cutoff_utc: `2026-06-10T16:47:42.516000+00:00`
- research_only: `true`

## Verdict

- primary_config: `btc750_dow_score3`
- primary_hold: `{'n': 32, 'wr': 0.781, 'sum': 3359.0, 'mean': 105.0, 'median': 72.8, 't3r': 2299.4, 'max_loss': -52.0, 'max_win': 370.0, 'max_dd_bps': 70.5}`
- primary_walk_forward: positive_folds `5/5`, fold_t3r_sum `1269.2`
- corrected_permutation: `PASS_MC_5PCT` mc_p=`0.001`
- live_blocker: `SILENCE cannot be an entry filter at T=0; it is a management/resolution state.`

## Config Holdout Summary

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_500k_score3 | 42 | 69.0% | 3322.6 | 79.1 | 19.1 | 2262.9 | -166.4 | 178.4 |
| btc750_score3 | 39 | 74.4% | 3471.3 | 89.0 | 35.3 | 2411.6 | -78.4 | 140.5 |
| btc750_dow_score3 | 32 | 78.1% | 3359.0 | 105.0 | 72.8 | 2299.4 | -52.0 | 70.5 |
| btc750_dow_score4 | 17 | 82.4% | 2355.6 | 138.6 | 137.9 | 1311.3 | -40.2 | 40.2 |
| btc1000_dow_score3 | 30 | 83.3% | 3471.4 | 115.7 | 106.8 | 2411.8 | -52.0 | 52.0 |
| btc750_dow_score3_noisy | 61 | 60.7% | 2925.6 | 48.0 | 15.2 | 1850.9 | -162.2 | 191.4 |

## Corrected Permutation

| Config | Real hold T3R | Raw p | MC p | Null p95 | Max-null p95 | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_500k_score3 | 3054.8 | 0.001 | 0.001 | 683.0 | 834.8 | PASS_MC_5PCT |
| btc750_score3 | 3203.5 | 0.001 | 0.001 | 757.0 | 834.8 | PASS_MC_5PCT |
| btc750_dow_score3 | 2406.5 | 0.001 | 0.001 | 630.2 | 834.8 | PASS_MC_5PCT |
| btc750_dow_score4 | 1418.4 | 0.001 | 0.007 | 272.2 | 834.8 | PASS_MC_5PCT |
| btc1000_dow_score3 | 2423.8 | 0.001 | 0.001 | 644.8 | 834.8 | PASS_MC_5PCT |
| btc750_dow_score3_noisy | 2607.7 | 0.001 | 0.001 | -45.9 | 834.8 | PASS_MC_5PCT |

## Entry Timing

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| silence_long_t0 | 128 | 62.5% | 5313.0 | 41.5 | 34.4 | 4229.7 | -318.3 |  |
| silence_long_t15 | 128 | 52.3% | 1977.6 | 15.4 | 6.3 | 969.4 | -340.2 |  |
| silence_long_t30 | 128 | 46.1% | 578.3 | 4.5 | -3.4 | -254.3 | -336.5 |  |
| silence_eth_shift_15_bps | 128 | 82.0% | 3333.1 | 26.0 | 23.7 | 2888.8 | -93.0 |  |
| silence_eth_shift_30_bps | 128 | 85.2% | 4724.1 | 36.9 | 31.7 | 4205.5 | -57.5 |  |
| noisy_short_anchor_provisional | 253 | 52.6% | 4885.6 | 19.3 | 7.0 | 3749.5 | -243.8 |  |
| neither_short_btc500k_confirmed | 68 | 48.5% | 1819.0 | 26.8 | -7.7 | 890.2 | -213.7 |  |
| neither_eth_shift_to_btc500k_bps | 68 | 1.5% | -3865.1 | -56.8 | -44.9 | -3863.5 | -258.0 |  |
| neither_short_btc750k_confirmed | 49 | 53.1% | 1769.9 | 36.1 | 6.1 | 848.8 | -169.5 |  |
| neither_eth_shift_to_btc750k_bps | 49 | 2.0% | -2831.8 | -57.8 | -42.6 | -2826.9 | -293.4 |  |
| neither_short_btc1000k_confirmed | 39 | 61.5% | 2033.1 | 52.1 | 19.4 | 1112.0 | -169.5 |  |
| neither_eth_shift_to_btc1000k_bps | 39 | 2.6% | -2590.2 | -66.4 | -61.5 | -2578.8 | -293.4 |  |

## Conflict Policies

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all_independent | 89 | 70.8% | 5843.6 | 65.7 | 50.1 | 4759.3 | -318.3 | 510.0 |
| one_pos_ignore | 64 | 64.1% | 4227.3 | 66.1 | 47.3 | 3143.0 | -318.3 | 516.9 |
| short_replace | 77 | 66.2% | 4921.8 | 63.9 | 39.9 | 3837.4 | -318.3 | 516.9 |

## State Transitions

## next_within_2h

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NEITHER->NEITHER | 20 | 85.0% | 1493.2 | 74.7 | 85.5 | 875.8 | -122.6 |  |
| SILENCE->SILENCE | 27 | 74.1% | 1691.4 | 62.6 | 48.5 | 854.2 | -107.7 |  |
| NOISY->NEITHER | 15 | 86.7% | 1758.4 | 117.2 | 104.8 | 820.0 | -53.3 |  |
| NOISY->SILENCE | 30 | 60.0% | 1074.0 | 35.8 | 37.6 | 162.0 | -452.3 |  |
| NEITHER->SILENCE | 11 | 63.6% | 664.6 | 60.4 | 51.7 | 5.3 | -234.5 |  |
| SILENCE->NEITHER | 4 | 50.0% | 337.3 | 84.3 | 38.0 | -42.2 | -42.2 |  |
| NEITHER->NOISY | 21 | 42.9% | 79.3 | 3.8 | -21.9 | -587.4 | -173.3 |  |
| SILENCE->NOISY | 25 | 48.0% | -208.5 | -8.3 | -0.3 | -832.9 | -235.9 |  |

## next_within_4h

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SILENCE->SILENCE | 40 | 77.5% | 2248.2 | 56.2 | 43.1 | 1411.1 | -133.3 |  |
| NEITHER->NEITHER | 22 | 81.8% | 1474.9 | 67.0 | 73.3 | 857.5 | -122.6 |  |
| NOISY->NEITHER | 21 | 71.4% | 1568.1 | 74.7 | 68.8 | 629.8 | -185.3 |  |
| NOISY->SILENCE | 44 | 59.1% | 1150.6 | 26.2 | 27.8 | 225.6 | -452.3 |  |
| NEITHER->SILENCE | 14 | 64.3% | 822.0 | 58.7 | 51.5 | 162.7 | -234.5 |  |
| SILENCE->NEITHER | 9 | 66.7% | 992.3 | 110.3 | 80.7 | 83.2 | -74.7 |  |
| NEITHER->NOISY | 24 | 41.7% | 147.2 | 6.1 | -21.8 | -519.5 | -173.3 |  |
| SILENCE->NOISY | 51 | 52.9% | 99.2 | 1.9 | 5.4 | -656.4 | -235.9 |  |

## next_within_8h

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NOISY->SILENCE | 62 | 56.5% | 1800.4 | 29.0 | 22.6 | 875.3 | -452.3 |  |
| NOISY->NEITHER | 26 | 69.2% | 1811.2 | 69.7 | 50.5 | 872.9 | -185.3 |  |
| NEITHER->NEITHER | 23 | 78.3% | 1465.0 | 63.7 | 62.2 | 847.7 | -122.6 |  |
| SILENCE->SILENCE | 54 | 66.7% | 1403.1 | 26.0 | 34.3 | 566.0 | -328.5 |  |
| SILENCE->NEITHER | 11 | 63.6% | 1023.4 | 93.0 | 48.9 | 114.3 | -74.7 |  |
| NEITHER->SILENCE | 15 | 60.0% | 675.7 | 45.0 | 51.3 | 16.4 | -234.5 |  |
| NEITHER->NOISY | 25 | 40.0% | 112.0 | 4.5 | -21.9 | -554.7 | -173.3 |  |
| SILENCE->NOISY | 61 | 50.8% | 172.1 | 2.8 | 1.0 | -635.2 | -235.9 |  |

## next_within_24h

| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SILENCE->SILENCE | 74 | 66.2% | 2835.0 | 38.3 | 35.3 | 1751.7 | -328.5 |  |
| NOISY->SILENCE | 74 | 56.8% | 2787.7 | 37.7 | 27.1 | 1545.9 | -452.3 |  |
| NOISY->NEITHER | 27 | 66.7% | 1748.8 | 64.8 | 32.2 | 810.4 | -185.3 |  |
| NEITHER->NEITHER | 24 | 75.0% | 1340.0 | 55.8 | 55.1 | 722.7 | -125.0 |  |
| SILENCE->NEITHER | 14 | 71.4% | 1587.4 | 113.4 | 64.8 | 558.0 | -74.7 |  |
| NEITHER->SILENCE | 16 | 56.2% | 646.7 | 40.4 | 47.3 | -12.6 | -234.5 |  |
| NEITHER->NOISY | 27 | 44.4% | 174.2 | 6.5 | -21.7 | -492.5 | -173.3 |  |
| SILENCE->NOISY | 74 | 50.0% | 251.0 | 3.4 | 0.3 | -556.3 | -235.9 |  |

## Feature Availability

| Feature | Class | Knowable at entry? | Source / note |
| --- | --- | --- | --- |
| dow | POINT_IN_TIME | True | timestamp |
| session | POINT_IN_TIME | True | timestamp UTC |
| n2h | RUNNING_CLUSTER/HISTORY | True | liquidations before T |
| sync_k | RUNNING_CLUSTER/HISTORY | True | BTC+SOL liquidations before T |
| btc4h_bps | POINT_IN_TIME_HISTORY | True | mark_prices before T |
| vdepth_bps | POINT_IN_TIME_BOOK | conditional | book ticker at T; stale book must reject |
| sil_eth | FORWARD_STATE_RESOLUTION | False | requires 30m future; live must use provisional entry + resolve later |
| btc_cascade_in_30m | FORWARD_STATE_RESOLUTION | False | only knowable when BTC threshold actually crosses |
| net_2h/net_4h | FORWARD/OUTCOME | False | label only |

## Shadow Parity

- shadow_state_exists: `True`
- note: Backfill ledger uses NAV net labels for historical rows; gauntlet recomputes mark-price outcomes, so exact bps parity is not expected. For live promotion, a separate timestamp/decision parity test is required.

## Next Required Work

1. Do not live-promote from this report alone.
2. Build a timestamp-level realtime/backfill parity test before executor work.
3. If proceeding, model SILENCE as provisional-entry management, not as an entry-known filter.
