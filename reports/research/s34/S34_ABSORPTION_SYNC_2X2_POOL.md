# S34 Absorption x Synchronization 2x2 Pool

Generated: `2026-06-28T23:32:37.605566+00:00`

Research-only. Cross-asset pooled SELL fade rows with knowable prior-window sync.

- Rows: `541`
- Sync window: `10m`
- Sync threshold: `200.0K` other-asset SELL liq
- Overall: N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101

## Sync Gate

| Gate | All | Cal | Hold |
| --- | --- | --- | --- |
| `idio` | N=160 sum=-659.4 med=-11.4 T3R=-1650.2 max_loss=-429.5 tail<-100=23 | N=71 sum=-563.6 med=-34.1 T3R=-1216.4 max_loss=-285.3 tail<-100=6 | N=89 sum=-95.9 med=12.0 T3R=-1086.7 max_loss=-429.5 tail<-100=17 |
| `sync` | N=381 sum=-1955.2 med=10.7 T3R=-3837.8 max_loss=-507.2 tail<-100=78 | N=95 sum=2316.5 med=14.3 T3R=1172.3 max_loss=-271.1 tail<-100=5 | N=286 sum=-4271.7 med=10.5 T3R=-6154.3 max_loss=-507.2 tail<-100=73 |

## Absorption Gate

| Gate | All | Cal | Hold |
| --- | --- | --- | --- |
| `vacuum_like` | N=106 sum=1374.5 med=38.2 T3R=-508.1 max_loss=-507.2 tail<-100=27 | N=20 sum=280.4 med=14.9 T3R=-204.2 max_loss=-285.3 tail<-100=1 | N=86 sum=1094.1 med=41.2 T3R=-788.5 max_loss=-507.2 tail<-100=26 |
| `mixed` | N=204 sum=3326.1 med=12.6 T3R=1943.2 max_loss=-460.6 tail<-100=23 | N=63 sum=2017.2 med=13.2 T3R=873.1 max_loss=-113.7 tail<-100=1 | N=141 sum=1308.9 med=12.0 T3R=-74.0 max_loss=-460.6 tail<-100=22 |
| `absorbed` | N=231 sum=-7315.2 med=-15.5 T3R=-8670.0 max_loss=-494.0 tail<-100=51 | N=83 sum=-544.7 med=-15.5 T3R=-1072.6 max_loss=-271.1 tail<-100=9 | N=148 sum=-6770.5 med=-15.8 T3R=-8125.3 max_loss=-494.0 tail<-100=42 |

## Sync x Absorption

| Sync | Absorption | All | Cal | Hold |
| --- | --- | --- | --- | --- |
| `idio` | `vacuum_like` | N=18 sum=-619.9 med=-48.8 T3R=-1253.8 max_loss=-421.4 tail<-100=3 | N=9 sum=-494.5 med=-49.1 T3R=-540.7 max_loss=-285.3 tail<-100=1 | N=9 sum=-125.3 med=37.0 T3R=-704.1 max_loss=-421.4 tail<-100=2 |
| `idio` | `mixed` | N=53 sum=-80.9 med=12.0 T3R=-794.9 max_loss=-291.8 tail<-100=5 | N=17 sum=-4.0 med=-14.7 T3R=-422.8 max_loss=-70.8 tail<-100=0 | N=36 sum=-76.9 med=18.3 T3R=-629.2 max_loss=-291.8 tail<-100=5 |
| `idio` | `absorbed` | N=89 sum=41.4 med=-24.1 T3R=-949.4 max_loss=-429.5 tail<-100=15 | N=45 sum=-65.0 med=-28.4 T3R=-582.3 max_loss=-256.9 tail<-100=5 | N=44 sum=106.3 med=-11.7 T3R=-884.4 max_loss=-429.5 tail<-100=10 |
| `sync` | `vacuum_like` | N=88 sum=1994.4 med=45.9 T3R=111.8 max_loss=-507.2 tail<-100=24 | N=11 sum=775.0 med=53.6 T3R=330.9 max_loss=-10.0 tail<-100=0 | N=77 sum=1219.5 med=45.3 T3R=-663.2 max_loss=-507.2 tail<-100=24 |
| `sync` | `mixed` | N=151 sum=3407.0 med=14.8 T3R=2024.1 max_loss=-460.6 tail<-100=18 | N=46 sum=2021.2 med=20.0 T3R=877.1 max_loss=-113.7 tail<-100=1 | N=105 sum=1385.8 med=7.7 T3R=2.9 max_loss=-460.6 tail<-100=17 |
| `sync` | `absorbed` | N=142 sum=-7356.6 med=-15.2 T3R=-8711.3 max_loss=-494.0 tail<-100=36 | N=38 sum=-479.7 med=-11.5 T3R=-964.1 max_loss=-271.1 tail<-100=4 | N=104 sum=-6876.9 med=-15.8 T3R=-8231.6 max_loss=-494.0 tail<-100=32 |

## Sync x Bid Depth

| Sync | Bid Depth | All | Cal | Hold |
| --- | --- | --- | --- | --- |
| `idio` | `shallow_bid` | N=55 sum=-927.9 med=3.6 T3R=-1703.8 max_loss=-421.4 tail<-100=7 | N=16 sum=-731.0 med=-49.1 T3R=-937.9 max_loss=-285.3 tail<-100=1 | N=39 sum=-196.9 med=20.9 T3R=-972.8 max_loss=-421.4 tail<-100=6 |
| `idio` | `deep_bid` | N=105 sum=268.5 med=-14.7 T3R=-722.3 max_loss=-429.5 tail<-100=16 | N=55 sum=167.4 med=-14.7 T3R=-485.4 max_loss=-256.9 tail<-100=5 | N=50 sum=101.0 med=-3.2 T3R=-889.8 max_loss=-429.5 tail<-100=11 |
| `sync` | `shallow_bid` | N=213 sum=5978.4 med=23.6 T3R=4095.8 max_loss=-507.2 tail<-100=36 | N=48 sum=2662.8 med=38.6 T3R=1518.7 max_loss=-113.7 tail<-100=1 | N=165 sum=3315.6 med=23.6 T3R=1432.9 max_loss=-507.2 tail<-100=35 |
| `sync` | `deep_bid` | N=168 sum=-7933.6 med=-15.2 T3R=-9288.3 max_loss=-494.0 tail<-100=42 | N=47 sum=-346.3 med=-7.4 T3R=-830.7 max_loss=-271.1 tail<-100=4 | N=121 sum=-7587.2 med=-16.8 T3R=-8942.0 max_loss=-494.0 tail<-100=38 |

## By Symbol

| Symbol | All | Sync | Idio | Sync+Absorbed | Sync+NotAbsorbed |
| --- | --- | --- | --- | --- | --- |
| `BTCUSDT` | N=134 sum=-379.4 med=17.9 T3R=-1272.5 max_loss=-417.1 tail<-100=20 | N=105 sum=-902.1 med=13.8 T3R=-1744.3 max_loss=-417.1 tail<-100=17 | N=29 sum=522.6 med=27.1 T3R=-61.2 max_loss=-240.9 tail<-100=3 | N=42 sum=-209.4 med=19.2 T3R=-958.4 max_loss=-417.1 tail<-100=6 | N=63 sum=-692.7 med=7.7 T3R=-1344.9 max_loss=-397.1 tail<-100=11 |
| `ETHUSDT` | N=256 sum=-1868.2 med=-0.3 T3R=-3706.2 max_loss=-507.2 tail<-100=53 | N=146 sum=-977.4 med=17.6 T3R=-2815.5 max_loss=-507.2 tail<-100=33 | N=110 sum=-890.7 med=-13.1 T3R=-1881.5 max_loss=-429.5 tail<-100=20 | N=50 sum=-4297.3 med=-59.9 T3R=-5390.2 max_loss=-494.0 tail<-100=19 | N=96 sum=3319.8 med=23.6 T3R=1481.8 max_loss=-507.2 tail<-100=14 |
| `SOLUSDT` | N=151 sum=-367.0 med=-13.1 T3R=-2228.6 max_loss=-484.2 tail<-100=28 | N=130 sum=-75.7 med=0.4 T3R=-1937.3 max_loss=-484.2 tail<-100=28 | N=21 sum=-291.3 med=-25.5 T3R=-502.3 max_loss=-94.9 tail<-100=0 | N=50 sum=-2849.9 med=-30.1 T3R=-3772.9 max_loss=-484.2 tail<-100=11 | N=80 sum=2774.3 med=21.5 T3R=912.7 max_loss=-472.5 tail<-100=17 |

## Sync Threshold Sweep

| Threshold K | All | Cal | Hold |
| ---: | --- | --- | --- |
| 0.0 | N=541 sum=-2614.6 med=4.4 T3R=-4497.3 max_loss=-507.2 tail<-100=101 | N=166 sum=1752.9 med=-6.9 T3R=608.8 max_loss=-285.3 tail<-100=11 | N=375 sum=-4367.5 med=10.7 T3R=-6250.2 max_loss=-507.2 tail<-100=90 |
| 50.0 | N=460 sum=-3517.1 med=7.8 T3R=-5399.8 max_loss=-507.2 tail<-100=89 | N=125 sum=1782.0 med=0.9 T3R=637.9 max_loss=-271.1 tail<-100=8 | N=335 sum=-5299.2 med=10.7 T3R=-7181.8 max_loss=-507.2 tail<-100=81 |
| 100.0 | N=430 sum=-3592.9 med=7.8 T3R=-5475.5 max_loss=-507.2 tail<-100=86 | N=116 sum=1544.1 med=1.5 T3R=400.0 max_loss=-271.1 tail<-100=8 | N=314 sum=-5137.0 med=10.5 T3R=-7019.7 max_loss=-507.2 tail<-100=78 |
| 200.0 | N=381 sum=-1955.2 med=10.7 T3R=-3837.8 max_loss=-507.2 tail<-100=78 | N=95 sum=2316.5 med=14.3 T3R=1172.3 max_loss=-271.1 tail<-100=5 | N=286 sum=-4271.7 med=10.5 T3R=-6154.3 max_loss=-507.2 tail<-100=73 |
| 300.0 | N=315 sum=-4456.4 med=4.4 T3R=-6318.0 max_loss=-507.2 tail<-100=61 | N=70 sum=1599.2 med=-4.1 T3R=539.2 max_loss=-132.0 tail<-100=1 | N=245 sum=-6055.6 med=10.2 T3R=-7917.2 max_loss=-507.2 tail<-100=60 |
| 500.0 | N=239 sum=-2810.5 med=0.9 T3R=-4672.1 max_loss=-507.2 tail<-100=49 | N=49 sum=1029.8 med=2.1 T3R=433.9 max_loss=-92.7 tail<-100=0 | N=190 sum=-3840.3 med=0.4 T3R=-5701.9 max_loss=-507.2 tail<-100=49 |
| 1000.0 | N=134 sum=42.9 med=-1.5 T3R=-1818.7 max_loss=-484.2 tail<-100=25 | N=28 sum=-167.0 med=-8.5 T3R=-408.5 max_loss=-92.7 tail<-100=0 | N=106 sum=209.9 med=5.9 T3R=-1651.7 max_loss=-484.2 tail<-100=25 |

## Read

- If sync improves all-sample but fails holdout, treat it as route/time-period structure rather than a robust resonance gate.
- If absorbed only works inside one symbol, the next model must be hierarchical by route/symbol, not globally pooled.
