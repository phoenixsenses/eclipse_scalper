# S34 V Engine Failed Rebound

Generated: `2026-06-28T20:06:23.945798+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only. Tests whether post-fill failed-rebound states can improve exits or justify a SHORT pivot.

## Baseline

- ledger rows: `47`
- closed filled anatomy rows: `19`
- original v0.1 filled: N=19 sum=876.1 med=37.0 T3R=348.4

## Failure Conditions

| Rank | Condition | Trigger | Trigger N | Loser% | Triggered original | Not-triggered original | Kill/hold combined | SHORT to original exit | SHORT 60m |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| 1 | `failed_v_15m` | 15m | 4 | 75.0 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=686.4 med=32.3 T3R=165.1 | N=4 sum=-217.0 med=-46.9 T3R=-147.1 | N=4 sum=-235.2 med=-63.5 T3R=-78.4 |
| 2 | `btc_down_continues_15m` | 15m | 3 | 66.7 | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=19 sum=656.2 med=32.3 T3R=134.9 | N=3 sum=-240.9 med=-79.4 T3R=-240.9 | N=3 sum=-205.3 med=-71.2 T3R=-205.3 |
| 3 | `no_anchor_reclaim_30m` | 30m | 4 | 75.0 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-301.6 med=-72.1 T3R=-153.3 |
| 4 | `failed_v_30m` | 30m | 4 | 75.0 | N=4 sum=-117.0 med=-26.0 T3R=-146.0 | N=15 sum=993.1 med=41.7 T3R=471.8 | N=19 sum=635.4 med=32.3 T3R=114.1 | N=4 sum=-268.7 med=-26.6 T3R=-210.4 | N=4 sum=-301.6 med=-72.1 T3R=-153.3 |
| 5 | `no_rebound_mfe15` | 15m | 5 | 60.0 | N=5 sum=29.9 med=-16.2 T3R=-181.9 | N=14 sum=846.2 med=39.4 T3R=399.9 | N=19 sum=534.8 med=30.4 T3R=88.5 | N=5 sum=-374.8 med=-79.4 T3R=-304.9 | N=5 sum=-439.9 med=-71.2 T3R=-283.1 |
| 6 | `no_anchor_reclaim_15m` | 15m | 5 | 60.0 | N=5 sum=29.9 med=-16.2 T3R=-181.9 | N=14 sum=846.2 med=39.4 T3R=399.9 | N=19 sum=534.8 med=30.4 T3R=88.5 | N=5 sum=-374.8 med=-79.4 T3R=-304.9 | N=5 sum=-439.9 med=-71.2 T3R=-283.1 |
| 7 | `weak_first_30m` | 30m | 5 | 60.0 | N=5 sum=-57.9 med=-16.2 T3R=-181.9 | N=14 sum=934.0 med=39.4 T3R=412.7 | N=19 sum=564.0 med=30.4 T3R=42.7 | N=5 sum=-346.3 med=-41.2 T3R=-288.0 | N=5 sum=-354.1 med=-62.9 T3R=-234.6 |
| 8 | `weak_first_15m` | 15m | 6 | 50.0 | N=6 sum=-60.4 med=5.8 T3R=-198.1 | N=13 sum=936.5 med=44.6 T3R=415.2 | N=19 sum=542.5 med=32.3 T3R=21.2 | N=6 sum=-373.8 med=-78.4 T3R=-304.9 | N=6 sum=-463.0 med=-74.8 T3R=-306.2 |
| 9 | `low_rebreak_30m` | 30m | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 | N=6 sum=218.9 med=31.3 T3R=74.9 | N=19 sum=194.2 med=19.0 T3R=-80.1 | N=13 sum=-763.6 med=-31.7 T3R=-782.1 | N=13 sum=-887.4 med=-52.5 T3R=-877.2 |
| 10 | `low_rebreak_15m` | 15m | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 | N=6 sum=218.9 med=31.3 T3R=74.9 | N=19 sum=35.7 med=22.4 T3R=-165.2 | N=13 sum=-922.3 med=-55.5 T3R=-941.1 | N=13 sum=-1077.9 med=-71.2 T3R=-1037.4 |
| 11 | `trap_composite_15m` | 15m | 13 | 23.1 | N=13 sum=657.2 med=44.6 T3R=129.5 | N=6 sum=218.9 med=31.3 T3R=74.9 | N=19 sum=35.7 med=22.4 T3R=-165.2 | N=13 sum=-922.3 med=-55.5 T3R=-941.1 | N=13 sum=-1077.9 med=-71.2 T3R=-1037.4 |

## Read

- Best kill/hold condition by T3R: `failed_v_15m` -> N=19 sum=686.4 med=32.3 T3R=165.1; delta vs baseline T3R `-183.3` bps.
- A failed-rebound label is useful only if it both isolates losing originals and improves kill/hold or SHORT outcomes after fees.

## Triggered Worst Cards

### `failed_v_15m`

| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -217.6 | -79.4 | -71.2 | -211.9 | -9.7 | False | True | btc_down_continues | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-16T02:27:55.467000+00:00 | -35.9 | -5.7 | 23.9 | -29.9 | -0.7 | 8.1 | False | True | btc_supportive | `low_rebreak_15m,weak_first_15m` |
| 2026-06-25T16:32:03.169000+00:00 | -16.2 | -24.4 | -14.4 | -78.4 | -19.4 | 7.8 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,btc_down_continues` |
| 2026-04-16T13:52:14.594000+00:00 | 81.1 | -59.0 | -147.1 | -55.7 | -53.1 | -1.4 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |

### `btc_down_continues_15m`

| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -217.6 | -79.4 | -71.2 | -211.9 | -9.7 | False | True | btc_down_continues | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-25T16:32:03.169000+00:00 | -16.2 | -24.4 | -14.4 | -78.4 | -19.4 | 7.8 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,btc_down_continues` |
| 2026-04-16T13:52:14.594000+00:00 | 81.1 | -59.0 | -147.1 | -55.7 | -53.1 | -1.4 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |

### `no_anchor_reclaim_30m`

| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -180.3 | -41.2 | -4.1 | -211.9 | -9.7 | False | True | btc_down_continues | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-16T02:27:55.467000+00:00 | -35.9 | -34.7 | -5.1 | -62.9 | -0.7 | 8.1 | False | True | btc_supportive | `low_rebreak_15m,weak_first_15m` |
| 2026-06-25T16:32:03.169000+00:00 | -16.2 | -22.0 | -12.0 | -81.3 | -19.4 | 7.8 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,btc_down_continues` |
| 2026-04-16T13:52:14.594000+00:00 | 81.1 | -120.7 | -210.4 | -153.3 | -53.1 | -1.4 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |

### `failed_v_30m`

| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -180.3 | -41.2 | -4.1 | -211.9 | -9.7 | False | True | btc_down_continues | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-16T02:27:55.467000+00:00 | -35.9 | -34.7 | -5.1 | -62.9 | -0.7 | 8.1 | False | True | btc_supportive | `low_rebreak_15m,weak_first_15m` |
| 2026-06-25T16:32:03.169000+00:00 | -16.2 | -22.0 | -12.0 | -81.3 | -19.4 | 7.8 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,btc_down_continues` |
| 2026-04-16T13:52:14.594000+00:00 | 81.1 | -120.7 | -210.4 | -153.3 | -53.1 | -1.4 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |

### `no_rebound_mfe15`

| UTC | Orig | Kill | Short orig-exit | Short 60m | Ret15 | MFE15 | Reclaim15 | Rebreak15 | BTC | Tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-06-23T07:59:44.477000+00:00 | -146.0 | -217.6 | -79.4 | -71.2 | -211.9 | -9.7 | False | True | btc_down_continues | `low_rebreak_15m,late_fill_gt10m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-16T02:27:55.467000+00:00 | -35.9 | -5.7 | 23.9 | -29.9 | -0.7 | 8.1 | False | True | btc_supportive | `low_rebreak_15m,weak_first_15m` |
| 2026-06-25T16:32:03.169000+00:00 | -16.2 | -24.4 | -14.4 | -78.4 | -19.4 | 7.8 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,btc_down_continues` |
| 2026-04-16T13:52:14.594000+00:00 | 81.1 | -59.0 | -147.1 | -55.7 | -53.1 | -1.4 | False | True | btc_down_continues | `low_rebreak_15m,weak_first_15m,candle5_bear_followthrough,btc_down_continues` |
| 2026-06-26T02:48:30.475000+00:00 | 146.9 | -4.7 | -157.8 | -204.7 | 1.8 | 15.1 | False | True | btc_down_then_stable | `low_rebreak_15m` |
