# S34 Navigation Regime-Inversion Walk-Forward

Generated: `2026-06-29T13:12:55.845716+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

## Aggregate

- Fold correlation: `{'median_pearson_t3r': -0.431, 'negative_folds': 3, 'folds': 5}`
- Top10 inversion permutation: `{'observed_top10_inverted_minus_same_fold_t3r': 9322.8, 'permutations': 30, 'p_ge': 0.065}`

### Top-Cell Meta Rules

| Rule | Same positive folds | Inverted positive folds | Same fold T3R sum | Inverted fold T3R sum | Same median | Inverted median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top5 | 1 | 3 | -12576.1 | -181.6 | -2391.8 | 176.9 |
| top10 | 0 | 1 | -12551.1 | -3228.3 | -2188.0 | -335.6 |
| top20 | 1 | 2 | -13732.6 | -5936.5 | -2300.8 | -1615.3 |

### Fixed Branches

| Branch | Positive folds | Fold T3R sum | Median fold T3R | Fold sum bps |
| --- | ---: | ---: | ---: | ---: |
| k5_CLEAN_REVERSE | 1/5 | -2560.4 | -555.2 | 225.9 |
| k20_DANGER_NORMAL | 2/5 | -3511.4 | -450.3 | 287.3 |
| k5_CLEAN_NORMAL | 0/5 | -4828.3 | -534.8 | -1475.9 |
| k20_DANGER_thr100000_REVERSE | 1/5 | -4929.3 | -1105.8 | -1343.9 |
| tags_NEUTRAL_CONTEXT_REVERSE | 1/5 | -4997.5 | -1258.2 | -1843.3 |
| tags_NEUTRAL_CONTEXT_NORMAL | 1/5 | -6427.5 | -904.4 | -3136.7 |
| k20_DANGER_REVERSE | 1/5 | -10532.2 | -1799.3 | -6537.3 |

## Folds

### Fold 1

- Train N `802`, hold N `240`; hold `2026-03-19T14:09:11.346000+00:00` -> `2026-04-02T04:41:11.094000+00:00`
- Pearson T3R `-0.612`, Spearman T3R `-0.623`

| Top rule | Same event avg | Inverted event avg |
| --- | --- | --- |
| top5 | N=133 sum=-3635.8 med=-21.3 T3R=-4343.0 tail150=13 maxLoss=-685.9 | N=133 sum=2305.8 med=11.3 T3R=889.3 tail150=4 maxLoss=-248.9 |
| top10 | N=230 sum=-3678.3 med=-10.1 T3R=-4518.4 tail150=18 maxLoss=-685.9 | N=230 sum=1378.3 med=0.1 T3R=-38.2 tail150=13 maxLoss=-318.0 |
| top20 | N=240 sum=-4107.8 med=-6.1 T3R=-4830.4 tail150=16 maxLoss=-685.9 | N=240 sum=1707.8 med=-3.9 T3R=291.3 tail150=6 maxLoss=-259.2 |

| Fixed branch | Cal | Hold |
| --- | --- | --- |
| k20_DANGER_NORMAL | N=563 sum=-5611.9 med=-1.5 T3R=-7186.4 tail150=81 maxLoss=-387.6 | N=202 sum=2245.6 med=9.8 T3R=829.1 tail150=10 maxLoss=-292.9 |
| k20_DANGER_REVERSE | N=563 sum=-18.1 med=-8.5 T3R=-1131.3 tail150=60 maxLoss=-728.1 | N=202 sum=-4265.6 med=-19.8 T3R=-5021.9 tail150=17 maxLoss=-685.9 |
| k20_DANGER_thr100000_REVERSE | N=189 sum=1078.0 med=-2.9 T3R=7.2 tail150=21 maxLoss=-412.2 | N=69 sum=-1331.9 med=-18.7 T3R=-1895.3 tail150=6 maxLoss=-361.3 |
| k5_CLEAN_NORMAL | N=67 sum=3775.5 med=35.5 T3R=2775.6 tail150=2 maxLoss=-185.2 | N=19 sum=-91.0 med=10.2 T3R=-534.8 tail150=1 maxLoss=-310.3 |
| k5_CLEAN_REVERSE | N=67 sum=-4445.5 med=-45.5 T3R=-4911.0 tail150=10 maxLoss=-343.3 | N=19 sum=-99.0 med=-20.2 T3R=-555.2 tail150=1 maxLoss=-243.8 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=294 sum=1549.4 med=7.0 T3R=290.8 tail150=27 maxLoss=-368.0 | N=89 sum=496.5 med=-0.6 T3R=-220.3 tail150=3 maxLoss=-347.2 |
| tags_NEUTRAL_CONTEXT_REVERSE | N=294 sum=-4489.4 med=-17.0 T3R=-5529.1 tail150=39 maxLoss=-464.2 | N=89 sum=-1386.5 med=-9.4 T3R=-2206.3 tail150=5 maxLoss=-259.2 |

### Fold 2

- Train N `1042`, hold N `240`; hold `2026-04-02T05:08:46.300000+00:00` -> `2026-04-19T16:53:45.185000+00:00`
- Pearson T3R `0.23`, Spearman T3R `0.113`

| Top rule | Same event avg | Inverted event avg |
| --- | --- | --- |
| top5 | N=89 sum=-1025.2 med=-23.8 T3R=-1946.0 tail150=3 maxLoss=-256.7 | N=89 sum=135.2 med=13.8 T3R=-463.1 tail150=3 maxLoss=-321.1 |
| top10 | N=164 sum=-271.4 med=-10.7 T3R=-1221.4 tail150=3 maxLoss=-256.7 | N=164 sum=-1368.6 med=0.7 T3R=-1966.9 tail150=7 maxLoss=-337.8 |
| top20 | N=188 sum=-777.4 med=-8.2 T3R=-1727.4 tail150=8 maxLoss=-256.7 | N=188 sum=-1102.6 med=-1.8 T3R=-1761.1 tail150=7 maxLoss=-337.8 |

| Fixed branch | Cal | Hold |
| --- | --- | --- |
| k20_DANGER_NORMAL | N=733 sum=-2215.5 med=-0.2 T3R=-4230.2 tail150=84 maxLoss=-387.6 | N=175 sum=972.7 med=8.2 T3R=264.7 tail150=7 maxLoss=-256.7 |
| k20_DANGER_REVERSE | N=733 sum=-5114.5 med=-9.8 T3R=-6227.7 tail150=74 maxLoss=-728.1 | N=175 sum=-2722.7 med=-18.2 T3R=-3422.0 tail150=9 maxLoss=-337.8 |
| k20_DANGER_thr100000_REVERSE | N=246 sum=261.1 med=-7.0 T3R=-809.7 tail150=26 maxLoss=-412.2 | N=61 sum=-715.5 med=-13.4 T3R=-1359.1 tail150=3 maxLoss=-200.1 |
| k5_CLEAN_NORMAL | N=90 sum=5507.2 med=36.4 T3R=4507.3 tail150=2 maxLoss=-179.2 | N=18 sum=776.4 med=3.4 T3R=-144.4 tail150=0 maxLoss=-71.9 |
| k5_CLEAN_REVERSE | N=90 sum=-6407.2 med=-46.4 T3R=-6825.0 tail150=16 maxLoss=-343.3 | N=18 sum=-956.4 med=-13.4 T3R=-1120.5 tail150=3 maxLoss=-321.1 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=383 sum=2045.9 med=2.4 T3R=787.3 tail150=30 maxLoss=-368.0 | N=113 sum=1060.5 med=3.4 T3R=426.6 tail150=0 maxLoss=-116.7 |
| tags_NEUTRAL_CONTEXT_REVERSE | N=383 sum=-5875.9 med=-12.4 T3R=-6917.1 tail150=44 maxLoss=-464.2 | N=113 sum=-2190.5 med=-13.4 T3R=-2506.8 tail150=4 maxLoss=-337.8 |

### Fold 3

- Train N `1282`, hold N `240`; hold `2026-04-19T17:51:16.652000+00:00` -> `2026-06-10T15:47:52.118000+00:00`
- Pearson T3R `0.486`, Spearman T3R `0.405`

| Top rule | Same event avg | Inverted event avg |
| --- | --- | --- |
| top5 | N=41 sum=611.5 med=34.7 T3R=48.5 tail150=3 maxLoss=-214.6 | N=41 sum=-1021.5 med=-44.7 T3R=-1538.1 tail150=5 maxLoss=-199.3 |
| top10 | N=118 sum=-66.5 med=-1.2 T3R=-629.5 tail150=6 maxLoss=-214.6 | N=118 sum=-1113.5 med=-8.8 T3R=-1631.0 tail150=5 maxLoss=-199.3 |
| top20 | N=209 sum=1283.1 med=10.3 T3R=715.5 tail150=7 maxLoss=-214.6 | N=209 sum=-3373.1 med=-20.3 T3R=-3915.5 tail150=8 maxLoss=-199.3 |

| Fixed branch | Cal | Hold |
| --- | --- | --- |
| k20_DANGER_NORMAL | N=808 sum=-2675.3 med=2.8 T3R=-4690.0 tail150=90 maxLoss=-387.6 | N=119 sum=95.1 med=4.5 T3R=-450.3 tail150=6 maxLoss=-186.4 |
| k20_DANGER_REVERSE | N=808 sum=-5404.7 med=-12.8 T3R=-6517.9 tail150=73 maxLoss=-728.1 | N=119 sum=-1285.1 med=-14.5 T3R=-1799.3 tail150=8 maxLoss=-192.7 |
| k20_DANGER_thr100000_REVERSE | N=270 sum=217.5 med=-7.2 T3R=-853.3 tail150=25 maxLoss=-412.2 | N=40 sum=-641.9 med=-21.4 T3R=-1105.8 tail150=3 maxLoss=-192.7 |
| k5_CLEAN_NORMAL | N=112 sum=6479.1 med=35.5 T3R=5479.2 tail150=4 maxLoss=-199.8 | N=25 sum=469.8 med=34.7 T3R=-86.9 tail150=0 maxLoss=-131.2 |
| k5_CLEAN_REVERSE | N=112 sum=-7599.1 med=-45.5 T3R=-8145.2 tail150=19 maxLoss=-343.3 | N=25 sum=-719.8 med=-44.7 T3R=-1065.4 tail150=3 maxLoss=-199.3 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=496 sum=3106.4 med=2.7 T3R=1847.8 tail150=30 maxLoss=-368.0 | N=107 sum=-336.8 med=13.1 T3R=-904.4 tail150=6 maxLoss=-214.6 |
| tags_NEUTRAL_CONTEXT_REVERSE | N=496 sum=-8066.4 med=-12.7 T3R=-9107.6 tail150=48 maxLoss=-464.2 | N=107 sum=-733.2 med=-23.1 T3R=-1258.2 tail150=3 maxLoss=-199.3 |

### Fold 4

- Train N `1522`, hold N `240`; hold `2026-06-10T15:48:21.872000+00:00` -> `2026-06-21T08:34:12.389000+00:00`
- Pearson T3R `-0.507`, Spearman T3R `-0.522`

| Top rule | Same event avg | Inverted event avg |
| --- | --- | --- |
| top5 | N=63 sum=-1667.2 med=-19.9 T3R=-2391.8 tail150=4 maxLoss=-312.0 | N=63 sum=1037.2 med=9.9 T3R=176.9 tail150=4 maxLoss=-296.1 |
| top10 | N=81 sum=-1334.7 med=-19.3 T3R=-2188.0 tail150=4 maxLoss=-312.0 | N=81 sum=524.7 med=9.3 T3R=-335.6 tail150=7 maxLoss=-296.1 |
| top20 | N=218 sum=-1442.8 med=-5.9 T3R=-2300.8 tail150=10 maxLoss=-312.0 | N=218 sum=-737.2 med=-4.1 T3R=-1615.3 tail150=17 maxLoss=-296.1 |

| Fixed branch | Cal | Hold |
| --- | --- | --- |
| k20_DANGER_NORMAL | N=843 sum=-2085.6 med=3.8 T3R=-4100.3 tail150=87 maxLoss=-387.6 | N=57 sum=-2499.0 med=-1.9 T3R=-2854.3 tail150=8 maxLoss=-291.1 |
| k20_DANGER_REVERSE | N=843 sum=-6344.4 med=-13.8 T3R=-7457.6 tail150=72 maxLoss=-728.1 | N=57 sum=1929.0 med=-8.1 T3R=1085.7 tail150=1 maxLoss=-162.4 |
| k20_DANGER_thr100000_REVERSE | N=280 sum=146.9 med=-9.4 T3R=-923.9 tail150=22 maxLoss=-412.2 | N=18 sum=907.2 med=34.1 T3R=95.7 tail150=0 maxLoss=-72.0 |
| k5_CLEAN_NORMAL | N=151 sum=7876.2 med=35.5 T3R=6588.9 tail150=4 maxLoss=-199.8 | N=29 sum=-835.3 med=-51.7 T3R=-1559.9 tail150=2 maxLoss=-266.3 |
| k5_CLEAN_REVERSE | N=151 sum=-9386.2 med=-45.5 T3R=-9932.3 tail150=23 maxLoss=-630.7 | N=29 sum=545.3 med=41.7 T3R=-81.9 tail150=3 maxLoss=-296.1 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=603 sum=2769.6 med=7.0 T3R=1511.0 tail150=36 maxLoss=-368.0 | N=107 sum=-3684.6 med=-23.5 T3R=-4542.6 tail150=6 maxLoss=-284.1 |
| tags_NEUTRAL_CONTEXT_REVERSE | N=603 sum=-8799.6 med=-17.0 T3R=-9840.8 tail150=51 maxLoss=-464.2 | N=107 sum=2614.6 med=13.5 T3R=1852.0 tail150=3 maxLoss=-296.1 |

### Fold 5

- Train N `1762`, hold N `244`; hold `2026-06-21T08:34:12.389000+00:00` -> `2026-06-29T08:28:10.110000+00:00`
- Pearson T3R `-0.431`, Spearman T3R `-0.477`

| Top rule | Same event avg | Inverted event avg |
| --- | --- | --- |
| top5 | N=100 sum=-2946.6 med=-6.7 T3R=-3943.8 tail150=19 maxLoss=-412.4 | N=100 sum=1946.6 med=-3.4 T3R=753.4 tail150=11 maxLoss=-392.6 |
| top10 | N=106 sum=-2996.6 med=-6.7 T3R=-3993.8 tail150=20 maxLoss=-412.4 | N=106 sum=1936.6 med=-3.4 T3R=743.4 tail150=12 maxLoss=-392.6 |
| top20 | N=233 sum=-4587.6 med=-5.0 T3R=-5589.5 tail150=33 maxLoss=-412.4 | N=233 sum=2257.6 med=-5.0 T3R=1064.1 tail150=23 maxLoss=-392.6 |

| Fixed branch | Cal | Hold |
| --- | --- | --- |
| k20_DANGER_NORMAL | N=928 sum=-5698.3 med=1.5 T3R=-7713.0 tail150=101 maxLoss=-387.6 | N=72 sum=-527.1 med=-3.4 T3R=-1300.6 tail150=7 maxLoss=-455.2 |
| k20_DANGER_REVERSE | N=928 sum=-3581.7 med=-11.4 T3R=-4694.9 tail150=74 maxLoss=-728.1 | N=72 sum=-192.9 med=-6.7 T3R=-1374.7 tail150=8 maxLoss=-276.9 |
| k20_DANGER_thr100000_REVERSE | N=308 sum=1158.8 med=-7.6 T3R=88.0 tail150=24 maxLoss=-412.2 | N=24 sum=438.2 med=-6.1 T3R=-664.8 tail150=2 maxLoss=-265.8 |
| k5_CLEAN_NORMAL | N=166 sum=7171.9 med=31.4 T3R=6172.0 tail150=4 maxLoss=-199.8 | N=34 sum=-1795.8 med=-8.9 T3R=-2502.3 tail150=8 maxLoss=-412.4 |
| k5_CLEAN_REVERSE | N=166 sum=-8831.9 med=-41.5 T3R=-9378.0 tail150=22 maxLoss=-343.3 | N=34 sum=1455.8 med=-1.1 T3R=262.6 tail150=4 maxLoss=-265.8 |
| tags_NEUTRAL_CONTEXT_NORMAL | N=710 sum=-915.0 med=-0.6 T3R=-2173.6 tail150=42 maxLoss=-368.0 | N=82 sum=-672.3 med=-3.5 T3R=-1186.8 tail150=4 maxLoss=-275.3 |
| tags_NEUTRAL_CONTEXT_REVERSE | N=710 sum=-6185.0 med=-9.4 T3R=-7226.2 tail150=54 maxLoss=-464.2 | N=82 sum=-147.7 med=-6.5 T3R=-878.2 tail150=5 maxLoss=-181.5 |

