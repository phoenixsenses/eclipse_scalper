# S34 Navigation Causal Gauntlet

Generated: `2026-06-29T12:40:25.363657+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Holdout rows are classified using calibration neighbors only. This is stricter than the descriptive full-sample navigation report.

Split: cal `1404`, hold `602`; holdout cell universe `282`
Permutations: `1000`; holdout max-null p95 T3R `3388.5`, p99 `4122.1`

| Candidate | Status | Cal | Hold | Hold raw p | Hold MC p |
| --- | --- | --- | --- | ---: | ---: |
| k5_DANGER_REVERSE | FAIL | N=226 sum=3205.3 med=-6.8 T3R=2092.1 tail150=18 maxLoss=-685.9 | N=42 sum=-742.8 med=-29.4 T3R=-1481.4 tail150=2 maxLoss=-167.3 | 0.786 | 1.0 |
| k5_CLEAN_NORMAL | FAIL | N=140 sum=7404.5 med=36.0 T3R=6404.6 tail150=4 maxLoss=-199.8 | N=84 sum=-3735.9 med=-23.9 T3R=-4563.9 tail150=9 maxLoss=-412.4 | 0.677 | 1.0 |
| k8_DANGER_REVERSE | FAIL | N=392 sum=441.0 med=-9.1 T3R=-672.2 tail150=33 maxLoss=-728.1 | N=80 sum=856.0 med=-16.7 T3R=-325.8 tail150=3 maxLoss=-260.8 | 0.395 | 1.0 |
| k10_DANGER_REVERSE | FAIL | N=460 sum=-1196.4 med=-13.5 T3R=-2309.6 tail150=38 maxLoss=-728.1 | N=99 sum=738.0 med=-16.7 T3R=-443.8 tail150=5 maxLoss=-260.8 | 0.454 | 1.0 |
| k5_to_k20_CLEAN_to_DANGER_NORMAL | FAIL | N=61 sum=3744.7 med=43.7 T3R=2896.0 tail150=2 maxLoss=-179.2 | N=27 sum=-1643.3 med=-23.9 T3R=-2128.8 tail150=6 maxLoss=-364.0 | None | None |
| k20_DANGER_REVERSE | FAIL | N=800 sum=-6515.8 med=-16.1 T3R=-7629.0 tail150=71 maxLoss=-728.1 | N=195 sum=2282.0 med=-7.8 T3R=1100.2 tail150=16 maxLoss=-276.9 | 0.181 | 0.925 |
| k20_DANGER_thr100000_REVERSE | FAIL | N=270 sum=-17.1 med=-9.4 T3R=-1087.9 tail150=23 maxLoss=-412.2 | N=65 sum=1541.6 med=-4.4 T3R=373.2 tail150=5 maxLoss=-265.8 | 0.189 | 1.0 |
| k20_DANGER_thr200000_REVERSE | FAIL | N=159 sum=-1062.5 med=-21.8 T3R=-2025.1 tail150=15 maxLoss=-464.2 | N=58 sum=912.7 med=-5.0 T3R=20.4 tail150=4 maxLoss=-276.9 | 0.216 | 1.0 |
| k20_DANGER_thr50000_REVERSE | FAIL | N=371 sum=-5436.2 med=-15.9 T3R=-6511.4 tail150=33 maxLoss=-728.1 | N=72 sum=-172.3 med=-16.7 T3R=-1144.7 tail150=7 maxLoss=-260.8 | 0.664 | 1.0 |
