# S34 Navigation Gauntlet

Generated: `2026-06-29T12:36:23.505435+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Cell universe tested for max-stat correction: `338`
Permutations: `1000`; max-null p95 T3R: `2994.4`; p99: `4009.2`

## Candidate Gauntlet

| Candidate | Status | Full | Cal | Hold | Raw p | MC p |
| --- | --- | --- | --- | --- | ---: | ---: |
| k5_DANGER_reverse | PASS | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 | N=231 sum=1783.3 med=-9.6 T3R=670.1 tail150=18 maxLoss=-685.9 | N=83 sum=6863.8 med=42.3 T3R=5670.6 tail150=4 maxLoss=-259.4 | 0.001 | 0.001 |
| k5_CLEAN_normal | PASS | N=192 sum=8876.4 med=30.9 T3R=7876.5 tail150=4 maxLoss=-199.8 | N=126 sum=6393.4 med=35.0 T3R=5393.5 tail150=4 maxLoss=-199.8 | N=66 sum=2483.0 med=21.4 T3R=1675.7 tail150=0 maxLoss=-123.8 | 0.001 | 0.001 |
| k8_DANGER_reverse | FAIL | N=530 sum=7744.8 med=-7.4 T3R=6551.6 tail150=37 maxLoss=-685.9 | N=389 sum=806.1 med=-10.1 T3R=-307.1 tail150=28 maxLoss=-685.9 | N=141 sum=6938.7 med=4.8 T3R=5745.5 tail150=9 maxLoss=-276.9 | 0.001 | 0.001 |
| k10_DANGER_reverse | FAIL | N=655 sum=4993.0 med=-9.1 T3R=3799.8 tail150=50 maxLoss=-728.1 | N=477 sum=-1296.1 med=-14.6 T3R=-2409.3 tail150=35 maxLoss=-728.1 | N=178 sum=6289.1 med=-1.8 T3R=5095.9 tail150=15 maxLoss=-276.9 | 0.001 | 0.016 |
| k5_CLEAN_to_k20_DANGER_normal | FAIL | N=74 sum=3992.1 med=35.1 T3R=3143.4 tail150=1 maxLoss=-167.7 | N=54 sum=3402.7 med=41.0 T3R=2554.0 tail150=1 maxLoss=-167.7 | N=20 sum=589.4 med=12.7 T3R=88.1 tail150=0 maxLoss=-121.2 | 0.001 | 0.043 |
| k20_DANGER_reverse | FAIL | N=1094 sum=-366.3 med=-9.8 T3R=-1609.6 tail150=89 maxLoss=-728.1 | N=798 sum=-7399.6 med=-16.1 T3R=-8512.8 tail150=66 maxLoss=-728.1 | N=296 sum=7033.3 med=-4.1 T3R=5790.0 tail150=23 maxLoss=-276.9 | 0.005 | 1.0 |
| k20_DANGER_thr100k_reverse | FAIL | N=366 sum=2676.4 med=-7.3 T3R=1453.2 tail150=29 maxLoss=-412.2 | N=261 sum=-213.3 med=-8.5 T3R=-1284.1 tail150=21 maxLoss=-412.2 | N=105 sum=2889.7 med=-4.3 T3R=1666.5 tail150=8 maxLoss=-265.8 | 0.007 | 0.47 |
| k20_DANGER_thr200k_reverse | FAIL | N=248 sum=925.8 med=-9.8 T3R=-208.8 tail150=20 maxLoss=-464.2 | N=165 sum=-605.0 med=-20.5 T3R=-1567.6 tail150=13 maxLoss=-464.2 | N=83 sum=1530.8 med=-5.5 T3R=396.2 tail150=7 maxLoss=-276.9 | 0.06 | 1.0 |
| k20_DANGER_thr50k_reverse | FAIL | N=480 sum=-3968.5 med=-15.2 T3R=-5137.2 tail150=40 maxLoss=-728.1 | N=372 sum=-6581.3 med=-18.1 T3R=-7656.5 tail150=32 maxLoss=-728.1 | N=108 sum=2612.8 med=-0.3 T3R=1467.7 tail150=8 maxLoss=-260.8 | 0.664 | 1.0 |

## k20 DANGER interpretation

- Broad `k20 DANGER` is not a clean reversal: reverse 2h remains negative after costs and fails MC correction.
- The only attractive-looking broad subcell is `k20 DANGER + 100K + reverse`, but it fails the chronological holdout and MC-corrected threshold.
- Current interpretation: `k20 DANGER` is an avoid/risk label, not a standalone direction.

## Top In-Sample Cells Before Correction

| Cell | Family | Direction | Summary |
| --- | --- | --- | --- |
| k5_CLEAN_NORMAL | knn_pred | NORMAL | N=192 sum=8876.4 med=30.9 T3R=7876.5 tail150=4 maxLoss=-199.8 |
| k5_DANGER_REVERSE | knn_pred | REVERSE | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |
| k5_to_k20_DANGER_to_DANGER_REVERSE | scale_transition | REVERSE | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |
| danger_count_6_REVERSE | danger_count | REVERSE | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |
| k8_DANGER_REVERSE | knn_pred | REVERSE | N=530 sum=7744.8 med=-7.4 T3R=6551.6 tail150=37 maxLoss=-685.9 |
| k8_CLEAN_NORMAL | knn_pred | NORMAL | N=200 sum=7212.0 med=27.2 T3R=5539.9 tail150=7 maxLoss=-310.3 |
| k12_CLEAN_NORMAL | knn_pred | NORMAL | N=202 sum=6193.6 med=20.5 T3R=5238.1 tail150=5 maxLoss=-356.0 |
| k10_CLEAN_NORMAL | knn_pred | NORMAL | N=198 sum=5797.6 med=15.2 T3R=4842.1 tail150=6 maxLoss=-356.0 |
| k15_CLEAN_NORMAL | knn_pred | NORMAL | N=195 sum=5558.4 med=21.6 T3R=4662.6 tail150=8 maxLoss=-356.0 |
| k10_DANGER_REVERSE | knn_pred | REVERSE | N=655 sum=4993.0 med=-9.1 T3R=3799.8 tail150=50 maxLoss=-728.1 |
| k12_DANGER_REVERSE | knn_pred | REVERSE | N=754 sum=4578.8 med=-9.1 T3R=3335.8 tail150=56 maxLoss=-728.1 |
| k5_to_k20_CLEAN_to_DANGER_NORMAL | scale_transition | NORMAL | N=74 sum=3992.1 med=35.1 T3R=3143.4 tail150=1 maxLoss=-167.7 |
| k5_to_k20_CLEAN_to_MIXED_NORMAL | scale_transition | NORMAL | N=62 sum=3452.1 med=30.6 T3R=2452.2 tail150=1 maxLoss=-199.8 |
| tags_BULL_PULLBACK_NORMAL | tag_combo | NORMAL | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 |
| tags_BULL_PULLBACK+SIZE_34X_FRAGILE_NORMAL | tag_combo | NORMAL | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 |
| tags_BULL_PULLBACK+TAIL_HIGH_OR_UNKNOWN_NORMAL | tag_combo | NORMAL | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 |
| tags_BULL_PULLBACK+SIZE_34X_FRAGILE+TAIL_HIGH_OR_UNKNOWN_NORMAL | tag_combo | NORMAL | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 |
| danger_count_0_NORMAL | danger_count | NORMAL | N=912 sum=3486.7 med=5.3 T3R=2417.9 tail150=44 maxLoss=-356.0 |
| k15_DANGER_REVERSE | knn_pred | REVERSE | N=891 sum=3355.0 med=-9.6 T3R=2112.0 tail150=66 maxLoss=-728.1 |
| k20_CLEAN_NORMAL | knn_pred | NORMAL | N=176 sum=2731.9 med=16.8 T3R=1899.1 tail150=7 maxLoss=-356.0 |
| tags_BID_DEPTH_THIN+BULL_PULLBACK_NORMAL | tag_combo | NORMAL | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 |
| tags_BID_DEPTH_THIN+BULL_PULLBACK+SIZE_34X_FRAGILE_NORMAL | tag_combo | NORMAL | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 |
| tags_BID_DEPTH_THIN+BULL_PULLBACK+TAIL_HIGH_OR_UNKNOWN_NORMAL | tag_combo | NORMAL | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 |
| k20_DANGER_thr100000_REVERSE | k20_threshold | REVERSE | N=366 sum=2676.4 med=-7.3 T3R=1453.2 tail150=29 maxLoss=-412.2 |
| tags_VDEPTH_CORE_NORMAL | tag_combo | NORMAL | N=306 sum=2314.5 med=17.7 T3R=1314.6 tail150=24 maxLoss=-392.6 |
