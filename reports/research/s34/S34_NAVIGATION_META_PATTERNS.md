# S34 Navigation Meta-Patterns

Generated: `2026-06-29T09:30:20.156575+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

## k20 DANGER: second-order reversal?

Interpretation: if reverse fails at k20, the reverse-of-reverse is simply the normal direction. Both are checked below.

- `30m` normal: N=1094 sum=-9510.8 med=0.3 T3R=-10141.9 tail150=42 maxLoss=-512.2
- `30m` reverse: N=1094 sum=-1429.2 med=-10.3 T3R=-2848.7 tail150=15 maxLoss=-266.8
- `1h` normal: N=1094 sum=-7209.7 med=2.5 T3R=-8566.8 tail150=65 maxLoss=-411.4
- `1h` reverse: N=1094 sum=-3730.3 med=-12.5 T3R=-4903.8 tail150=42 maxLoss=-581.9
- `2h` normal: N=1094 sum=-10258.2 med=0.5 T3R=-12273.5 tail150=128 maxLoss=-456.1
- `2h` reverse: N=1094 sum=-681.8 med=-10.5 T3R=-1929.5 tail150=89 maxLoss=-728.2
- `4h` normal: N=1094 sum=-10994.3 med=4.7 T3R=-12706.3 tail150=170 maxLoss=-537.8
- `4h` reverse: N=1094 sum=54.3 med=-14.7 T3R=-1529.0 tail150=145 maxLoss=-597.7

### k20 2h by threshold

| Threshold | Normal | Reverse |
| --- | --- | --- |
| thr100000 | N=366 sum=-6165.8 med=-2.3 T3R=-7251.9 tail150=49 maxLoss=-456.1 | N=366 sum=2505.8 med=-7.7 T3R=1280.1 tail150=29 maxLoss=-411.8 |
| thr200000 | N=248 sum=-3346.3 med=0.4 T3R=-4472.7 tail150=35 maxLoss=-414.8 | N=248 sum=866.3 med=-10.4 T3R=-269.8 tail150=20 maxLoss=-464.3 |
| thr50000 | N=480 sum=-746.1 med=5.7 T3R=-2761.4 tail150=44 maxLoss=-406.7 | N=480 sum=-4053.9 med=-15.7 T3R=-5204.4 tail150=40 maxLoss=-728.2 |

## Scale Transition Groups

| Group | N | Normal 2h | Reverse 2h | Threshold mix |
| --- | ---: | --- | --- | --- |
| EMERGENT_BROAD_DANGER_k20 | 780 | N=780 sum=1213.4 med=5.9 T3R=-527.6 tail150=67 maxLoss=-455.2 | N=780 sum=-9013.4 med=-15.9 T3R=-10241.6 tail150=67 maxLoss=-728.1 | `{'thr100000': 264, 'thr200000': 173, 'thr50000': 343}` |
| NO_CLEAR_SCALE_PATTERN | 800 | N=800 sum=0.7 med=0.9 T3R=-1068.1 tail150=42 maxLoss=-356.0 | N=800 sum=-8000.7 med=-10.9 T3R=-9019.6 tail150=63 maxLoss=-412.2 | `{'thr100000': 273, 'thr200000': 179, 'thr50000': 348}` |
| PERSISTENT_CLEAN_k5_to_k20 | 56 | N=56 sum=1432.2 med=26.1 T3R=624.9 tail150=2 maxLoss=-197.1 | N=56 sum=-1992.2 med=-36.2 T3R=-2462.3 tail150=5 maxLoss=-296.1 | `{'thr100000': 18, 'thr200000': 13, 'thr50000': 25}` |
| PERSISTENT_DANGER_k5_to_k20 | 314 | N=314 sum=-11787.1 med=-4.5 T3R=-13306.5 tail150=61 maxLoss=-412.4 | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 | `{'thr100000': 102, 'thr200000': 75, 'thr50000': 137}` |
| SMALL_SCALE_CLEAN_DECAYS | 56 | N=56 sum=2053.8 med=13.2 T3R=1098.3 tail150=0 maxLoss=-110.4 | N=56 sum=-2613.8 med=-23.2 T3R=-2862.2 tail150=8 maxLoss=-343.3 | `{'thr100000': 17, 'thr200000': 10, 'thr50000': 29}` |

## Danger Count Bins

| Bin | N | Normal 2h | Reverse 2h |
| --- | ---: | --- | --- |
| danger_count_0 | 912 | N=912 sum=3486.7 med=5.3 T3R=2417.9 tail150=44 maxLoss=-356.0 | N=912 sum=-12606.7 med=-15.3 T3R=-13625.6 tail150=76 maxLoss=-412.2 |
| danger_count_1 | 203 | N=203 sum=1691.3 med=3.8 T3R=456.0 tail150=16 maxLoss=-405.7 | N=203 sum=-3721.3 med=-13.8 T3R=-4847.0 tail150=23 maxLoss=-630.7 |
| danger_count_2 | 137 | N=137 sum=-146.2 med=10.3 T3R=-1125.5 tail150=13 maxLoss=-326.3 | N=137 sum=-1223.8 med=-20.3 T3R=-2025.9 tail150=10 maxLoss=-412.2 |
| danger_count_3 | 99 | N=99 sum=-575.8 med=-0.7 T3R=-1119.9 tail150=8 maxLoss=-455.2 | N=99 sum=-414.2 med=-9.3 T3R=-1269.4 tail150=6 maxLoss=-204.5 |
| danger_count_4 | 125 | N=125 sum=1501.8 med=17.9 T3R=395.4 tail150=11 maxLoss=-310.3 | N=125 sum=-2751.8 med=-27.9 T3R=-3581.7 tail150=13 maxLoss=-728.1 |
| danger_count_5 | 216 | N=216 sum=-1257.7 med=-2.1 T3R=-2050.3 tail150=19 maxLoss=-364.0 | N=216 sum=-902.3 med=-7.9 T3R=-1899.5 tail150=15 maxLoss=-279.9 |
| danger_count_6 | 314 | N=314 sum=-11787.1 med=-4.5 T3R=-13306.5 tail150=61 maxLoss=-412.4 | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |

## Top Tag-Combo Candidates

Excluded non-knowable/outcome tags: `['EXIT_2H_ACTUAL_BETTER', 'EXIT_4H_ACTUAL_BETTER', 'EXIT_4H_FAVORED', 'TAIL_REALIZED']`

| Combo | Direction | N | Summary | Tail150 rate | Score |
| --- | --- | ---: | --- | ---: | ---: |
| BULL_PULLBACK | NORMAL | 164 | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 | 0.098 | 1629.1 |
| BULL_PULLBACK+SIZE_34X_FRAGILE | NORMAL | 164 | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 | 0.098 | 1629.1 |
| BULL_PULLBACK+TAIL_HIGH_OR_UNKNOWN | NORMAL | 164 | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 | 0.098 | 1629.1 |
| BULL_PULLBACK+SIZE_34X_FRAGILE+TAIL_HIGH_OR_UNKNOWN | NORMAL | 164 | N=164 sum=3503.0 med=15.3 T3R=2429.1 tail150=16 maxLoss=-387.6 | 0.098 | 1629.1 |
| BID_DEPTH_THIN+BULL_PULLBACK | NORMAL | 146 | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 | 0.096 | 1182.8 |
| BID_DEPTH_THIN+BULL_PULLBACK+SIZE_34X_FRAGILE | NORMAL | 146 | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 | 0.096 | 1182.8 |
| BID_DEPTH_THIN+BULL_PULLBACK+TAIL_HIGH_OR_UNKNOWN | NORMAL | 146 | N=146 sum=2956.7 med=13.0 T3R=1882.8 tail150=14 maxLoss=-387.6 | 0.096 | 1182.8 |
| VDEPTH_CORE | NORMAL | 306 | N=306 sum=2314.5 med=17.7 T3R=1314.6 tail150=24 maxLoss=-392.6 | 0.078 | 114.6 |
| BID_DEPTH_OK+RISK_OFF_REBOUND+VDEPTH_DANGER_LOW | REVERSE | 199 | N=199 sum=2162.4 med=-4.4 T3R=926.4 tail150=17 maxLoss=-270.6 | 0.085 | 76.4 |
| BID_DEPTH_CORE+NEUTRAL_CONTEXT | REVERSE | 48 | N=48 sum=913.4 med=11.6 T3R=174.1 tail150=2 maxLoss=-163.5 | 0.042 | 74.1 |
| BID_DEPTH_CORE+BID_DEPTH_OK+NEUTRAL_CONTEXT | REVERSE | 48 | N=48 sum=913.4 med=11.6 T3R=174.1 tail150=2 maxLoss=-163.5 | 0.042 | 74.1 |
| BID_DEPTH_CORE+NEUTRAL_CONTEXT+SIZE_34X_FRAGILE | REVERSE | 48 | N=48 sum=913.4 med=11.6 T3R=174.1 tail150=2 maxLoss=-163.5 | 0.042 | 74.1 |
| BID_DEPTH_CORE+NEUTRAL_CONTEXT+TAIL_HIGH_OR_UNKNOWN | REVERSE | 48 | N=48 sum=913.4 med=11.6 T3R=174.1 tail150=2 maxLoss=-163.5 | 0.042 | 74.1 |
| NEUTRAL_CONTEXT+VDEPTH_DANGER_HIGH | REVERSE | 64 | N=64 sum=1138.7 med=9.6 T3R=276.2 tail150=7 maxLoss=-341.7 | 0.109 | -73.8 |
| NEUTRAL_CONTEXT+SIZE_34X_FRAGILE+VDEPTH_DANGER_HIGH | REVERSE | 64 | N=64 sum=1138.7 med=9.6 T3R=276.2 tail150=7 maxLoss=-341.7 | 0.109 | -73.8 |

## Prediction Pattern Candidates

| Pattern | N | Normal 2h | Reverse 2h |
| --- | ---: | --- | --- |
| k5=CLEAN | 192 | N=192 sum=8876.4 med=30.9 T3R=7876.5 tail150=4 maxLoss=-199.8 | N=192 sum=-10796.4 med=-41.0 T3R=-11342.5 tail150=32 maxLoss=-343.3 |
| k5=DANGER | 314 | N=314 sum=-11787.1 med=-4.5 T3R=-13306.5 tail150=61 maxLoss=-412.4 | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |
| k5_to_k20=DANGER->DANGER | 314 | N=314 sum=-11787.1 med=-4.5 T3R=-13306.5 tail150=61 maxLoss=-412.4 | N=314 sum=8647.1 med=-5.5 T3R=7453.9 tail150=22 maxLoss=-685.9 |
| k8=DANGER | 530 | N=530 sum=-13044.8 med=-2.6 T3R=-14564.2 tail150=80 maxLoss=-412.4 | N=530 sum=7744.8 med=-7.4 T3R=6551.6 tail150=37 maxLoss=-685.9 |
| k8=CLEAN | 200 | N=200 sum=7212.0 med=27.2 T3R=5539.9 tail150=7 maxLoss=-310.3 | N=200 sum=-9212.0 med=-37.2 T3R=-9917.2 tail150=29 maxLoss=-728.1 |
| k10=CLEAN | 198 | N=198 sum=5797.6 med=15.2 T3R=4842.1 tail150=6 maxLoss=-356.0 | N=198 sum=-7777.6 med=-25.2 T3R=-8534.8 tail150=26 maxLoss=-343.3 |
| k10=DANGER | 655 | N=655 sum=-11543.0 med=-0.9 T3R=-13391.2 tail150=91 maxLoss=-412.4 | N=655 sum=4993.0 med=-9.1 T3R=3799.8 tail150=50 maxLoss=-728.1 |
| k5_to_k20=CLEAN->DANGER | 74 | N=74 sum=3992.1 med=35.1 T3R=3143.4 tail150=1 maxLoss=-167.7 | N=74 sum=-4732.1 med=-45.1 T3R=-5095.8 tail150=15 maxLoss=-292.9 |
| k5_to_k20=CLEAN->MIXED | 62 | N=62 sum=3452.1 med=30.6 T3R=2452.2 tail150=1 maxLoss=-199.8 | N=62 sum=-4072.1 med=-40.6 T3R=-4409.9 tail150=12 maxLoss=-343.3 |
| k20=CLEAN | 176 | N=176 sum=2731.9 med=16.8 T3R=1899.1 tail150=7 maxLoss=-356.0 | N=176 sum=-4491.9 med=-26.8 T3R=-5312.6 tail150=15 maxLoss=-296.1 |
| k5_to_k20=MIXED->CLEAN | 120 | N=120 sum=1299.7 med=11.2 T3R=631.7 tail150=5 maxLoss=-356.0 | N=120 sum=-2499.7 med=-21.1 T3R=-3320.4 tail150=10 maxLoss=-296.1 |
| k5_to_k20=CLEAN->CLEAN | 56 | N=56 sum=1432.2 med=26.1 T3R=624.9 tail150=2 maxLoss=-197.1 | N=56 sum=-1992.2 med=-36.2 T3R=-2462.3 tail150=5 maxLoss=-296.1 |
| k20=MIXED | 736 | N=736 sum=754.8 med=0.6 T3R=-314.0 tail150=37 maxLoss=-347.2 | N=736 sum=-8114.8 med=-10.6 T3R=-9095.7 tail150=61 maxLoss=-412.2 |
| k20=DANGER | 1094 | N=1094 sum=-10573.7 med=-0.2 T3R=-12588.4 tail150=128 maxLoss=-455.2 | N=1094 sum=-366.3 med=-9.8 T3R=-1609.6 tail150=89 maxLoss=-728.1 |
| k8=MIXED | 1276 | N=1276 sum=-1254.2 med=1.2 T3R=-2391.9 tail150=85 maxLoss=-455.2 | N=1276 sum=-11505.8 med=-11.2 T3R=-12734.0 tail150=99 maxLoss=-412.2 |
| k10=MIXED | 1153 | N=1153 sum=-1341.6 med=1.2 T3R=-2766.7 tail150=75 maxLoss=-455.2 | N=1153 sum=-10188.4 med=-11.2 T3R=-11416.6 tail150=89 maxLoss=-630.7 |
| k5_to_k20=MIXED->MIXED | 674 | N=674 sum=-2697.3 med=-3.5 T3R=-3726.2 tail150=36 maxLoss=-347.2 | N=674 sum=-4042.7 med=-6.5 T3R=-5023.6 tail150=49 maxLoss=-412.2 |
| k5_to_k20=MIXED->DANGER | 706 | N=706 sum=-2778.7 med=-0.8 T3R=-4519.7 tail150=66 maxLoss=-455.2 | N=706 sum=-4281.3 med=-9.2 T3R=-5509.5 tail150=52 maxLoss=-728.1 |
| k5=MIXED | 1500 | N=1500 sum=-4176.3 med=-0.5 T3R=-5917.3 tail150=107 maxLoss=-455.2 | N=1500 sum=-10823.7 med=-9.5 T3R=-12051.9 tail150=111 maxLoss=-728.1 |
