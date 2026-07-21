# S34 Sync x Absorption Real-Fill

Generated: `2026-06-28T23:09:37.644660+00:00`

Research-only. Real bid/ask fills on book_ticker subset; no live/paper state changed.

Route: `ETHUSDT SELL deep-V 28.0bps-40.0bps, 200K, 4h LONG fade`
Events: `36`; holdout months `['2026-06']`; sync threshold `200.0K`

## Overall

- All: N=36 sum=968.9 mean=26.9 med=20.6 win=58.3 T3R=137.2 max_loss=-291.8 tail<-100=3
- Cal: N=12 sum=375.8 mean=31.3 med=-7.5 win=41.7 T3R=-90.6 max_loss=-59.9 tail<-100=0
- Hold: N=24 sum=593.1 mean=24.7 med=22.2 win=66.7 T3R=-238.7 max_loss=-291.8 tail<-100=3

## Cuts

- `imbalance_med`: `0.0`
- `bid_depth_med`: `138038.3`
- `imbalance_p25`: `-0.6`
- `bid_depth_p25`: `69319.8`

## sync_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `sync` | N=21 sum=980.7 mean=46.7 med=39.4 win=66.7 T3R=244.8 max_loss=-213.5 tail<-100=1 | N=6 sum=192.4 mean=32.1 med=17.6 win=50.0 T3R=-74.8 max_loss=-59.9 tail<-100=0 | N=15 sum=788.3 mean=52.6 med=62.5 win=73.3 T3R=90.3 max_loss=-213.5 tail<-100=1 |
| `idio` | N=15 sum=-11.8 mean=-0.8 med=-11.5 win=46.7 T3R=-574.1 max_loss=-291.8 tail<-100=2 | N=6 sum=183.4 mean=30.6 med=-13.1 win=33.3 T3R=-97.4 max_loss=-45.1 tail<-100=0 | N=9 sum=-195.2 mean=-21.7 med=15.7 win=55.6 T3R=-538.7 max_loss=-291.8 tail<-100=2 |

## asset_count_200k

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `3` | N=2 sum=212.6 mean=106.3 med=106.3 win=100.0 T3R=212.6 max_loss=76.3 tail<-100=0 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=2 sum=212.6 mean=106.3 med=106.3 win=100.0 T3R=212.6 max_loss=76.3 tail<-100=0 |
| `2` | N=19 sum=768.2 mean=40.4 med=23.6 win=63.2 T3R=32.3 max_loss=-213.5 tail<-100=1 | N=6 sum=192.4 mean=32.1 med=17.6 win=50.0 T3R=-74.8 max_loss=-59.9 tail<-100=0 | N=13 sum=575.7 mean=44.3 med=23.6 win=69.2 T3R=-90.9 max_loss=-213.5 tail<-100=1 |
| `1` | N=15 sum=-11.8 mean=-0.8 med=-11.5 win=46.7 T3R=-574.1 max_loss=-291.8 tail<-100=2 | N=6 sum=183.4 mean=30.6 med=-13.1 win=33.3 T3R=-97.4 max_loss=-45.1 tail<-100=0 | N=9 sum=-195.2 mean=-21.7 med=15.7 win=55.6 T3R=-538.7 max_loss=-291.8 tail<-100=2 |

## imbalance_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `bid_support` | N=18 sum=1141.1 mean=63.4 med=30.0 win=55.6 T3R=316.7 max_loss=-74.6 tail<-100=0 | N=8 sum=342.6 mean=42.8 med=-7.5 win=37.5 T3R=-123.8 max_loss=-59.9 tail<-100=0 | N=10 sum=798.4 mean=79.8 med=56.4 win=70.0 T3R=44.1 max_loss=-74.6 tail<-100=0 |
| `ask_heavy` | N=18 sum=-172.1 mean=-9.6 med=19.4 win=61.1 T3R=-559.3 max_loss=-291.8 tail<-100=3 | N=4 sum=33.2 mean=8.3 med=12.4 win=50.0 T3R=-45.1 max_loss=-45.1 tail<-100=0 | N=14 sum=-205.4 mean=-14.7 med=19.4 win=64.3 T3R=-592.5 max_loss=-291.8 tail<-100=3 |

## bid_depth_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `deep_bid` | N=18 sum=869.8 mean=48.3 med=22.2 win=55.6 T3R=141.3 max_loss=-92.8 tail<-100=0 | N=8 sum=338.8 mean=42.3 med=-7.8 win=37.5 T3R=-127.6 max_loss=-59.9 tail<-100=0 | N=10 sum=531.1 mean=53.1 med=30.0 win=70.0 T3R=-56.1 max_loss=-92.8 tail<-100=0 |
| `shallow_bid` | N=18 sum=99.1 mean=5.5 med=19.4 win=61.1 T3R=-489.4 max_loss=-291.8 tail<-100=3 | N=4 sum=37.1 mean=9.3 med=14.3 win=50.0 T3R=-45.1 max_loss=-45.1 tail<-100=0 | N=14 sum=62.0 mean=4.4 med=19.4 win=64.3 T3R=-526.5 max_loss=-291.8 tail<-100=3 |

## absorption_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `absorbed` | N=15 sum=956.4 mean=63.8 med=36.4 win=60.0 T3R=227.9 max_loss=-67.1 tail<-100=0 | N=7 sum=353.4 mean=50.5 med=-4.1 win=42.9 T3R=-113.0 max_loss=-59.9 tail<-100=0 | N=8 sum=603.0 mean=75.4 med=56.4 win=75.0 T3R=15.8 max_loss=-67.1 tail<-100=0 |
| `vacuum_like` | N=7 sum=302.1 mean=43.2 med=39.4 win=85.7 T3R=49.7 max_loss=-45.1 tail<-100=0 | N=3 sum=47.9 mean=16.0 med=39.4 win=66.7 T3R=47.9 max_loss=-45.1 tail<-100=0 | N=4 sum=254.2 mean=63.6 med=49.8 win=100.0 T3R=18.4 max_loss=18.4 tail<-100=0 |
| `mixed` | N=14 sum=-289.6 mean=-20.7 med=-12.7 win=42.9 T3R=-810.6 max_loss=-291.8 tail<-100=3 | N=2 sum=-25.5 mean=-12.7 med=-12.7 win=0.0 T3R=-25.5 max_loss=-14.7 tail<-100=0 | N=12 sum=-264.1 mean=-22.0 med=-8.2 win=50.0 T3R=-785.1 max_loss=-291.8 tail<-100=3 |

## Sync x Imbalance Combos

| Combo | All | Cal | Hold |
| --- | --- | --- | --- |
| `sync+bid_support` | N=11 sum=644.8 mean=58.6 med=23.6 win=54.5 T3R=-13.8 max_loss=-74.6 tail<-100=0 | N=4 sum=99.4 mean=24.9 med=-7.5 win=25.0 T3R=-59.9 max_loss=-59.9 tail<-100=0 | N=7 sum=545.4 mean=77.9 med=76.3 win=71.4 T3R=-41.8 max_loss=-74.6 tail<-100=0 |
| `sync+ask_heavy` | N=10 sum=336.0 mean=33.6 med=46.5 win=80.0 T3R=-51.2 max_loss=-213.5 tail<-100=1 | N=2 sum=93.0 mean=46.5 med=46.5 win=100.0 T3R=93.0 max_loss=39.4 tail<-100=0 | N=8 sum=242.9 mean=30.4 med=41.5 win=75.0 T3R=-144.2 max_loss=-213.5 tail<-100=1 |
| `idio+bid_support` | N=7 sum=496.3 mean=70.9 med=36.4 win=57.1 T3R=-65.9 max_loss=-53.4 tail<-100=0 | N=4 sum=243.2 mean=60.8 med=53.0 win=50.0 T3R=-37.5 max_loss=-37.5 tail<-100=0 | N=3 sum=253.1 mean=84.4 med=36.4 win=66.7 T3R=253.1 max_loss=-53.4 tail<-100=0 |
| `idio+ask_heavy` | N=8 sum=-508.1 mean=-63.5 med=-29.9 win=37.5 T3R=-581.8 max_loss=-291.8 tail<-100=2 | N=2 sum=-59.8 mean=-29.9 med=-29.9 win=0.0 T3R=-59.8 max_loss=-45.1 tail<-100=0 | N=6 sum=-448.3 mean=-74.7 med=-38.5 win=50.0 T3R=-522.0 max_loss=-291.8 tail<-100=2 |

## Read

- Best combo by T3R: `sync+bid_support` -> N=11 sum=644.8 mean=58.6 med=23.6 win=54.5 T3R=-13.8 max_loss=-74.6 tail<-100=0.
- Within sync, bid_support vs ask_heavy delta T3R `37.4`, delta max_loss `138.9`.
- A valid overlay must improve tail/T3R in holdout, not only the all-sample mean.
