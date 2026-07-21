# S34 Navigation Regime-Reversal Tests

Generated: `2026-06-29T12:44:52.458953+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

## Cal -> Hold Correlation

- Cells with N>=40 in both: `130`
- Pearson T3R: `-0.395`; Spearman T3R: `-0.376`
- Permutation p(negative-or-lower): `0.001`; two-sided: `0.001`

## Top-Cal Meta Reversal

- Top cal cells: `20`
- Hold same positive: `0`
- Hold opposite positive: `10`
- Hold same median T3R: `-3364.2`
- Hold opposite median T3R: `663.8`

## Specific Reversal Checks

| Cell | Cal | Hold |
| --- | --- | --- |
| k5_DANGER_REVERSE | N=226 sum=3205.3 med=-6.8 T3R=2092.1 tail150=18 maxLoss=-685.9 | N=42 sum=-742.8 med=-29.4 T3R=-1481.4 tail150=2 maxLoss=-167.3 |
| k5_DANGER_NORMAL | N=226 sum=-5465.3 med=-3.2 T3R=-6984.7 tail150=38 maxLoss=-387.6 | N=42 sum=322.8 med=19.4 T3R=-119.9 tail150=3 maxLoss=-291.1 |
| k5_CLEAN_NORMAL | N=140 sum=7404.5 med=36.0 T3R=6404.6 tail150=4 maxLoss=-199.8 | N=84 sum=-3735.9 med=-23.9 T3R=-4563.9 tail150=9 maxLoss=-412.4 |
| k5_CLEAN_REVERSE | N=140 sum=-8804.5 med=-46.0 T3R=-9350.6 tail150=21 maxLoss=-343.3 | N=84 sum=2895.9 med=13.8 T3R=1702.7 tail150=5 maxLoss=-296.1 |
| k8_DANGER_REVERSE | N=392 sum=441.0 med=-9.1 T3R=-672.2 tail150=33 maxLoss=-728.1 | N=80 sum=856.0 med=-16.7 T3R=-325.8 tail150=3 maxLoss=-260.8 |
| k8_DANGER_NORMAL | N=392 sum=-4361.0 med=-0.9 T3R=-6209.2 tail150=47 maxLoss=-387.6 | N=80 sum=-1656.0 med=6.7 T3R=-2211.5 tail150=10 maxLoss=-455.2 |
| k10_DANGER_REVERSE | N=460 sum=-1196.4 med=-13.5 T3R=-2309.6 tail150=38 maxLoss=-728.1 | N=99 sum=738.0 med=-16.7 T3R=-443.8 tail150=5 maxLoss=-260.8 |
| k10_DANGER_NORMAL | N=460 sum=-3403.6 med=3.5 T3R=-5251.8 tail150=52 maxLoss=-387.6 | N=99 sum=-1728.0 med=6.7 T3R=-2386.6 tail150=11 maxLoss=-455.2 |
| k20_DANGER_REVERSE | N=800 sum=-6515.8 med=-16.1 T3R=-7629.0 tail150=71 maxLoss=-728.1 | N=195 sum=2282.0 med=-7.8 T3R=1100.2 tail150=16 maxLoss=-276.9 |
| k20_DANGER_NORMAL | N=800 sum=-1484.2 med=6.1 T3R=-3498.9 tail150=85 maxLoss=-387.6 | N=195 sum=-4232.0 med=-2.2 T3R=-5005.5 tail150=24 maxLoss=-455.2 |
| k20_DANGER_thr100000_REVERSE | N=270 sum=-17.1 med=-9.4 T3R=-1087.9 tail150=23 maxLoss=-412.2 | N=65 sum=1541.6 med=-4.4 T3R=373.2 tail150=5 maxLoss=-265.8 |
| k20_DANGER_thr100000_NORMAL | N=270 sum=-2682.9 med=-0.6 T3R=-3768.1 tail150=33 maxLoss=-387.6 | N=65 sum=-2191.6 med=-5.6 T3R=-2806.2 tail150=10 maxLoss=-455.2 |

## Sign-Flip Candidates

Count: `10`

| Cell | Opposite | Cal T3R | Hold Same T3R | Hold Opp T3R | Hold Opp Summary |
| --- | --- | ---: | ---: | ---: | --- |
| tags_NEUTRAL_CONTEXT_NORMAL | tags_NEUTRAL_CONTEXT_REVERSE | 3003.1 | -6707.0 | 2713.3 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+SIZE_34X_FRAGILE_NORMAL | tags_NEUTRAL_CONTEXT+SIZE_34X_FRAGILE_REVERSE | 3003.1 | -6707.0 | 2713.3 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+TAIL_HIGH_OR_UNKNOWN_NORMAL | tags_NEUTRAL_CONTEXT+TAIL_HIGH_OR_UNKNOWN_REVERSE | 3003.1 | -6707.0 | 2713.3 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| k5_CLEAN_NORMAL | k5_CLEAN_REVERSE | 6404.6 | -4563.9 | 1702.7 | N=84 sum=2895.9 med=13.8 T3R=1702.7 tail150=5 maxLoss=-296.1 |
| tags_BID_DEPTH_THIN+NEUTRAL_CONTEXT_NORMAL | tags_BID_DEPTH_THIN+NEUTRAL_CONTEXT_REVERSE | 2283.2 | -4096.4 | 1151.9 | N=145 sum=1902.7 med=9.3 T3R=1151.9 tail150=6 maxLoss=-296.1 |
| k15_CLEAN_NORMAL | k15_CLEAN_REVERSE | 3244.4 | -3281.0 | 753.7 | N=98 sum=1753.1 med=0.8 T3R=753.7 tail150=1 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW_NORMAL | tags_NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW_REVERSE | 2224.7 | -4200.2 | 691.7 | N=192 sum=1422.2 med=-1.0 T3R=691.7 tail150=7 maxLoss=-296.1 |
| k12_CLEAN_NORMAL | k12_CLEAN_REVERSE | 4286.0 | -3217.5 | 635.8 | N=90 sum=1728.6 med=-0.3 T3R=635.8 tail150=2 maxLoss=-296.1 |
| k10_CLEAN_NORMAL | k10_CLEAN_REVERSE | 3475.9 | -3447.4 | 596.6 | N=93 sum=1689.4 med=9.3 T3R=596.6 tail150=5 maxLoss=-296.1 |
| k8_CLEAN_NORMAL | k8_CLEAN_REVERSE | 4697.0 | -3193.9 | 272.7 | N=90 sum=1465.9 med=1.8 T3R=272.7 tail150=6 maxLoss=-296.1 |

## Top-Cal Rows

| Cell | Cal | Hold Same | Hold Opposite |
| --- | --- | --- | --- |
| k5_CLEAN_NORMAL | N=140 sum=7404.5 med=36.0 T3R=6404.6 tail150=4 maxLoss=-199.8 | N=84 sum=-3735.9 med=-23.9 T3R=-4563.9 tail150=9 maxLoss=-412.4 | N=84 sum=2895.9 med=13.8 T3R=1702.7 tail150=5 maxLoss=-296.1 |
| k8_CLEAN_NORMAL | N=142 sum=5652.5 med=31.4 T3R=4697.0 tail150=5 maxLoss=-310.3 | N=90 sum=-2365.9 med=-11.8 T3R=-3193.9 tail150=7 maxLoss=-412.4 | N=90 sum=1465.9 med=1.8 T3R=272.7 tail150=6 maxLoss=-296.1 |
| k12_CLEAN_NORMAL | N=132 sum=5241.5 med=30.9 T3R=4286.0 tail150=5 maxLoss=-273.4 | N=90 sum=-2628.6 med=-9.7 T3R=-3217.5 tail150=6 maxLoss=-405.4 | N=90 sum=1728.6 med=-0.3 T3R=635.8 tail150=2 maxLoss=-296.1 |
| k10_CLEAN_NORMAL | N=132 sum=4431.4 med=29.5 T3R=3475.9 tail150=5 maxLoss=-273.4 | N=93 sum=-2619.4 med=-19.3 T3R=-3447.4 tail150=7 maxLoss=-405.4 | N=93 sum=1689.4 med=9.3 T3R=596.6 tail150=5 maxLoss=-296.1 |
| k15_CLEAN_NORMAL | N=127 sum=4256.3 med=27.8 T3R=3244.4 tail150=5 maxLoss=-356.0 | N=98 sum=-2733.1 med=-10.8 T3R=-3281.0 tail150=6 maxLoss=-405.4 | N=98 sum=1753.1 med=0.8 T3R=753.7 tail150=1 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=559 sum=4261.7 med=9.3 T3R=3003.1 tail150=30 maxLoss=-368.0 | N=233 sum=-5849.0 med=-15.8 T3R=-6707.0 tail150=16 maxLoss=-284.1 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+SIZE_34X_FRAGILE_NORMAL | N=559 sum=4261.7 med=9.3 T3R=3003.1 tail150=30 maxLoss=-368.0 | N=233 sum=-5849.0 med=-15.8 T3R=-6707.0 tail150=16 maxLoss=-284.1 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+TAIL_HIGH_OR_UNKNOWN_NORMAL | N=559 sum=4261.7 med=9.3 T3R=3003.1 tail150=30 maxLoss=-368.0 | N=233 sum=-5849.0 med=-15.8 T3R=-6707.0 tail150=16 maxLoss=-284.1 | N=233 sum=3519.0 med=5.8 T3R=2713.3 tail150=8 maxLoss=-296.1 |
| k5_to_k20_CLEAN_to_DANGER_NORMAL | N=61 sum=3744.7 med=43.7 T3R=2896.0 tail150=2 maxLoss=-179.2 | - | - |
| tags_BID_DEPTH_THIN+NEUTRAL_CONTEXT_NORMAL | N=474 sum=3541.8 med=8.7 T3R=2283.2 tail150=30 maxLoss=-368.0 | N=145 sum=-3352.7 med=-19.3 T3R=-4096.4 tail150=10 maxLoss=-276.3 | N=145 sum=1902.7 med=9.3 T3R=1151.9 tail150=6 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT+VDEPTH_DANGER_LOW_NORMAL | N=411 sum=3483.3 med=7.0 T3R=2224.7 tail150=17 maxLoss=-356.0 | N=192 sum=-3342.2 med=-9.1 T3R=-4200.2 tail150=11 maxLoss=-275.3 | N=192 sum=1422.2 med=-1.0 T3R=691.7 tail150=7 maxLoss=-296.1 |
| k5_DANGER_REVERSE | N=226 sum=3205.3 med=-6.8 T3R=2092.1 tail150=18 maxLoss=-685.9 | N=42 sum=-742.8 med=-29.4 T3R=-1481.4 tail150=2 maxLoss=-167.3 | N=42 sum=322.8 med=19.4 T3R=-119.9 tail150=3 maxLoss=-291.1 |
| k5_to_k20_DANGER_to_DANGER_REVERSE | N=226 sum=3205.3 med=-6.8 T3R=2092.1 tail150=18 maxLoss=-685.9 | N=42 sum=-742.8 med=-29.4 T3R=-1481.4 tail150=2 maxLoss=-167.3 | N=42 sum=322.8 med=19.4 T3R=-119.9 tail150=3 maxLoss=-291.1 |
| k5_to_k20_CLEAN_to_MIXED_NORMAL | N=44 sum=2880.8 med=38.0 T3R=1880.9 tail150=2 maxLoss=-199.8 | - | - |
| k20_CLEAN_NORMAL | N=124 sum=2155.5 med=17.2 T3R=1612.0 tail150=2 maxLoss=-356.0 | N=96 sum=-1534.8 med=6.5 T3R=-2242.8 tail150=6 maxLoss=-405.4 | N=96 sum=574.8 med=-16.4 T3R=-387.9 tail150=2 maxLoss=-296.1 |
