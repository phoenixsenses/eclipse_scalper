# S34 V Engine Confirmation Cost - Current Execution

Generated: `2026-06-28T20:56:00.061959+00:00`

Config: `O20_W300_O5_C1`

Research-only. Tests whether waiting for confirmation still pays after price deterioration on the current cancel/replace execution model.

Baseline: N=22 sum=1120.7 med=39.4 T3R=441.8

| Condition | Pass | Filter original | Failed original | Kill@15 hold | Delayed@15 entry | Price deterioration |
| --- | ---: | --- | --- | --- | --- | --- |
| `anchor_reclaimed_15m` | 18/22 | N=18 sum=953.6 med=39.4 T3R=274.7 | N=4 sum=167.1 med=26.6 T3R=-35.9 | N=22 sum=834.3 med=28.7 T3R=155.3 | N=18 sum=794.1 med=30.5 T3R=221.4 | N=18 sum=140.7 med=17.3 T3R=-32.7 |
| `btc_not_down_continues` | 18/22 | N=18 sum=1207.6 med=43.2 T3R=528.7 | N=4 sum=-86.9 med=-11.8 T3R=-144.4 | N=22 sum=940.3 med=33.7 T3R=261.3 | N=18 sum=899.9 med=37.1 T3R=292.5 | N=18 sum=288.7 med=17.3 T3R=115.3 |
| `candle15_bull_reclaim` | 11/22 | N=11 sum=857.6 med=37.0 T3R=178.7 | N=11 sum=263.1 med=41.7 T3R=-27.5 | N=22 sum=482.5 med=5.7 T3R=-196.5 | N=11 sum=482.9 med=1.1 T3R=-89.8 | N=11 sum=361.0 med=26.9 T3R=187.6 |
| `anchor_and_btc` | 16/22 | N=16 sum=1093.6 med=43.2 T3R=414.7 | N=6 sum=27.1 med=-11.8 T3R=-208.3 | N=22 sum=805.3 med=28.7 T3R=126.3 | N=16 sum=776.8 med=37.1 T3R=204.1 | N=16 sum=299.7 med=21.8 T3R=126.3 |
| `all3` | 10/22 | N=10 sum=853.2 med=40.8 T3R=174.3 | N=12 sum=267.5 med=34.3 T3R=-23.0 | N=22 sum=485.4 med=7.1 T3R=-193.6 | N=10 sum=491.9 med=3.1 T3R=-80.8 | N=10 sum=348.7 med=30.1 T3R=175.3 |

## Read

- `btc_not_down_continues` delayed N=18 sum=899.9 med=37.1 T3R=292.5; kill@15 N=22 sum=940.3 med=33.7 T3R=261.3; deterioration N=18 sum=288.7 med=17.3 T3R=115.3.
- `anchor_reclaimed_15m` delayed N=18 sum=794.1 med=30.5 T3R=221.4; kill@15 N=22 sum=834.3 med=28.7 T3R=155.3; deterioration N=18 sum=140.7 med=17.3 T3R=-32.7.
- `anchor_and_btc` delayed N=16 sum=776.8 med=37.1 T3R=204.1; kill@15 N=22 sum=805.3 med=28.7 T3R=126.3; deterioration N=16 sum=299.7 med=21.8 T3R=126.3.
- `all3` delayed N=10 sum=491.9 med=3.1 T3R=-80.8; kill@15 N=22 sum=485.4 med=7.1 T3R=-193.6; deterioration N=10 sum=348.7 med=30.1 T3R=175.3.
- `candle15_bull_reclaim` delayed N=11 sum=482.9 med=1.1 T3R=-89.8; kill@15 N=22 sum=482.5 med=5.7 T3R=-196.5; deterioration N=11 sum=361.0 med=26.9 T3R=187.6.
