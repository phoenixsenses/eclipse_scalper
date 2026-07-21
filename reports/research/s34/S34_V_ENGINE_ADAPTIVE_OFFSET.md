# S34 V Engine Adaptive Offset

Generated: `2026-06-28T19:43:58.436213+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only. Compares point-in-time adaptive maker offsets against fixed controls.

Events: `47`

## Ranked Policies

| Rank | Policy | Cross | Fill% | Offsets | Filled | No-fill CF | Missed CF |
| ---: | --- | ---: | ---: | --- | --- | --- | ---: |
| 1 | `eth_extreme_conservative` | 1.0 | 40.4 | `{'20.0': 37, '25.0': 10}` | N=19 sum=900.7 med=37.0 T3R=370.0 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 2 | `eth_extreme_conservative` | 2.0 | 40.4 | `{'20.0': 37, '25.0': 10}` | N=19 sum=891.3 med=37.0 T3R=363.6 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 3 | `fixed_o20` | 1.0 | 40.4 | `{'20.0': 47}` | N=19 sum=887.5 med=37.0 T3R=356.9 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 4 | `fixed_o20` | 2.0 | 40.4 | `{'20.0': 47}` | N=19 sum=876.1 med=37.0 T3R=348.4 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 5 | `vdepth_step_15_20_25` | 2.0 | 40.4 | `{'15.0': 18, '20.0': 18, '25.0': 11}` | N=19 sum=855.0 med=35.0 T3R=321.7 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 6 | `vdepth_step_15_20_25` | 1.0 | 40.4 | `{'15.0': 18, '20.0': 18, '25.0': 11}` | N=19 sum=854.5 med=35.0 T3R=321.1 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 7 | `vdepth_step_15_20_25` | 5.0 | 38.3 | `{'15.0': 18, '20.0': 18, '25.0': 11}` | N=18 sum=843.8 med=38.1 T3R=306.7 | N=9 sum=700.4 med=57.3 T3R=195.4 | 700.4 |
| 8 | `fixed_o15` | 5.0 | 40.4 | `{'15.0': 47}` | N=19 sum=793.3 med=32.0 T3R=277.2 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 9 | `accel_aggressive` | 2.0 | 40.4 | `{'10.0': 21, '20.0': 26}` | N=19 sum=795.6 med=35.8 T3R=275.5 | N=6 sum=536.1 med=66.2 T3R=82.9 | 536.1 |
| 10 | `fixed_o15` | 1.0 | 40.4 | `{'15.0': 47}` | N=19 sum=795.0 med=32.0 T3R=274.7 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 11 | `fixed_o15` | 2.0 | 40.4 | `{'15.0': 47}` | N=19 sum=787.5 med=32.0 T3R=274.2 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 12 | `btc_supportive_aggressive` | 2.0 | 40.4 | `{'10.0': 12, '20.0': 35}` | N=19 sum=795.9 med=30.5 T3R=268.2 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 13 | `risk_balanced` | 2.0 | 40.4 | `{'10.0': 26, '20.0': 11, '25.0': 10}` | N=19 sum=787.0 med=30.5 T3R=266.9 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 14 | `dominance_aggressive` | 1.0 | 40.4 | `{'10.0': 24, '20.0': 23}` | N=19 sum=794.2 med=32.3 T3R=263.5 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 15 | `accel_aggressive` | 1.0 | 42.6 | `{'10.0': 21, '20.0': 26}` | N=20 sum=779.0 med=32.3 T3R=258.9 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 16 | `btc_supportive_aggressive` | 1.0 | 42.6 | `{'10.0': 12, '20.0': 35}` | N=20 sum=785.8 med=28.7 T3R=255.1 | N=5 sum=543.1 med=75.2 T3R=89.9 | 543.1 |
| 17 | `dominance_aggressive` | 2.0 | 40.4 | `{'10.0': 24, '20.0': 23}` | N=19 sum=782.8 med=32.3 T3R=255.1 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 18 | `risk_balanced` | 1.0 | 42.6 | `{'10.0': 26, '20.0': 11, '25.0': 10}` | N=20 sum=772.9 med=29.6 T3R=252.8 | N=5 sum=543.1 med=75.2 T3R=89.9 | 543.1 |
| 19 | `vdepth_inverse_25_20_15` | 1.0 | 38.3 | `{'15.0': 11, '20.0': 18, '25.0': 18}` | N=18 sum=782.2 med=35.0 T3R=252.0 | N=8 sum=707.6 med=66.2 T3R=187.0 | 707.6 |
| 20 | `vdepth_inverse_25_20_15` | 2.0 | 38.3 | `{'15.0': 11, '20.0': 18, '25.0': 18}` | N=18 sum=768.1 med=34.6 T3R=250.8 | N=9 sum=734.2 med=57.3 T3R=213.6 | 734.2 |
| 21 | `missed_winner_rescue` | 2.0 | 40.4 | `{'10.0': 33, '20.0': 14}` | N=19 sum=756.3 med=30.5 T3R=236.3 | N=6 sum=536.1 med=66.2 T3R=82.9 | 536.1 |
| 22 | `vdepth_inverse_25_20_15` | 5.0 | 36.2 | `{'15.0': 11, '20.0': 18, '25.0': 18}` | N=17 sum=743.6 med=48.1 T3R=227.0 | N=11 sum=865.8 med=57.3 T3R=345.2 | 865.8 |
| 23 | `missed_winner_rescue` | 1.0 | 42.6 | `{'10.0': 33, '20.0': 14}` | N=20 sum=742.3 med=29.6 T3R=222.2 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 24 | `fixed_o10` | 5.0 | 40.4 | `{'10.0': 47}` | N=19 sum=728.4 med=27.0 T3R=222.0 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 25 | `btc_supportive_aggressive` | 5.0 | 38.3 | `{'10.0': 12, '20.0': 35}` | N=18 sum=744.9 med=28.5 T3R=215.3 | N=9 sum=718.6 med=57.3 T3R=213.6 | 718.6 |
| 26 | `eth_extreme_conservative` | 5.0 | 36.2 | `{'20.0': 37, '25.0': 10}` | N=17 sum=737.4 med=43.7 T3R=207.8 | N=11 sum=865.8 med=57.3 T3R=345.2 | 865.8 |
| 27 | `fixed_o20` | 5.0 | 36.2 | `{'20.0': 47}` | N=17 sum=726.8 med=43.7 T3R=197.3 | N=11 sum=865.8 med=57.3 T3R=345.2 | 865.8 |
| 28 | `fixed_o10` | 2.0 | 40.4 | `{'10.0': 47}` | N=19 sum=689.5 med=27.0 T3R=182.4 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 29 | `dominance_aggressive` | 5.0 | 38.3 | `{'10.0': 24, '20.0': 23}` | N=18 sum=708.8 med=29.4 T3R=179.2 | N=8 sum=591.6 med=44.9 T3R=138.4 | 591.6 |
| 30 | `risk_balanced` | 5.0 | 38.3 | `{'10.0': 26, '20.0': 11, '25.0': 10}` | N=18 sum=679.2 med=25.4 T3R=161.6 | N=8 sum=591.6 med=44.9 T3R=138.4 | 591.6 |
| 31 | `fixed_o10` | 1.0 | 42.6 | `{'10.0': 47}` | N=20 sum=646.4 med=24.6 T3R=136.6 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 32 | `missed_winner_rescue` | 5.0 | 38.3 | `{'10.0': 33, '20.0': 14}` | N=18 sum=649.8 med=25.4 T3R=132.2 | N=7 sum=559.0 med=57.3 T3R=105.8 | 559.0 |
| 33 | `accel_aggressive` | 5.0 | 38.3 | `{'10.0': 21, '20.0': 26}` | N=18 sum=633.2 med=25.4 T3R=115.6 | N=8 sum=686.0 med=66.2 T3R=181.0 | 686.0 |

## Read

- Best policy by T3R: `eth_extreme_conservative` C1.0 -> N=19 sum=900.7 med=37.0 T3R=370.0.
- Fixed O20 C1 control: N=19 sum=887.5 med=37.0 T3R=356.9.
- T3R delta vs O20 C1: `13.1` bps; sum delta `13.2` bps.
- Verdict: no new frozen variant. The best adaptive policy is only a small execution tweak over fixed O20, so keep it observation-only until forward N grows.
