# S34 Sync x Absorption Real-Fill

Generated: `2026-06-28T23:09:37.650855+00:00`

Research-only. Real bid/ask fills on book_ticker subset; no live/paper state changed.

Route: `ETHUSDT SELL deep-V 28.0bps-infbps, 200K, 4h LONG fade`
Events: `51`; holdout months `['2026-06']`; sync threshold `200.0K`

## Overall

- All: N=51 sum=278.6 mean=5.5 med=15.7 win=51.0 T3R=-669.1 max_loss=-338.0 tail<-100=8
- Cal: N=16 sum=364.2 mean=22.8 med=17.6 win=50.0 T3R=-139.5 max_loss=-271.1 tail<-100=1
- Hold: N=35 sum=-85.6 mean=-2.4 med=15.7 win=51.4 T3R=-1033.3 max_loss=-338.0 tail<-100=7

## Cuts

- `imbalance_med`: `0.0`
- `bid_depth_med`: `137185.0`
- `imbalance_p25`: `-0.6`
- `bid_depth_p25`: `67620.0`

## sync_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `sync` | N=33 sum=338.4 mean=10.3 med=20.4 win=54.5 T3R=-521.4 max_loss=-338.0 tail<-100=6 | N=9 sum=121.2 mean=13.5 med=39.4 win=55.6 T3R=-261.4 max_loss=-271.1 tail<-100=1 | N=24 sum=217.2 mean=9.1 med=19.4 win=54.2 T3R=-642.6 max_loss=-338.0 tail<-100=5 |
| `idio` | N=18 sum=-59.8 mean=-3.3 med=-13.1 win=44.4 T3R=-622.1 max_loss=-291.8 tail<-100=2 | N=7 sum=243.1 mean=34.7 med=-11.5 win=42.9 T3R=-108.8 max_loss=-45.1 tail<-100=0 | N=11 sum=-302.9 mean=-27.5 med=-34.1 win=45.5 T3R=-646.4 max_loss=-291.8 tail<-100=2 |

## asset_count_200k

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `3` | N=3 sum=209.7 mean=69.9 med=76.3 win=66.7 T3R=209.7 max_loss=-2.9 tail<-100=0 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=3 sum=209.7 mean=69.9 med=76.3 win=66.7 T3R=209.7 max_loss=-2.9 tail<-100=0 |
| `1` | N=20 sum=91.3 mean=4.6 med=-13.1 win=45.0 T3R=-651.8 max_loss=-291.8 tail<-100=3 | N=7 sum=243.1 mean=34.7 med=-11.5 win=42.9 T3R=-108.8 max_loss=-45.1 tail<-100=0 | N=13 sum=-151.8 mean=-11.7 med=-34.1 win=46.2 T3R=-757.0 max_loss=-291.8 tail<-100=3 |
| `2` | N=28 sum=-22.3 mean=-0.8 med=19.4 win=53.6 T3R=-758.2 max_loss=-338.0 tail<-100=5 | N=9 sum=121.2 mean=13.5 med=39.4 win=55.6 T3R=-261.4 max_loss=-271.1 tail<-100=1 | N=19 sum=-143.5 mean=-7.6 med=18.4 win=52.6 T3R=-810.1 max_loss=-338.0 tail<-100=4 |

## imbalance_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `bid_support` | N=26 sum=1158.4 mean=44.6 med=30.0 win=53.8 T3R=210.7 max_loss=-271.1 tail<-100=2 | N=11 sum=285.9 mean=26.0 med=-4.1 win=45.5 T3R=-217.8 max_loss=-271.1 tail<-100=1 | N=15 sum=872.5 mean=58.2 med=36.4 win=60.0 T3R=-75.2 max_loss=-176.7 tail<-100=1 |
| `ask_heavy` | N=25 sum=-879.8 mean=-35.2 med=-2.9 win=48.0 T3R=-1266.9 max_loss=-338.0 tail<-100=6 | N=5 sum=78.4 mean=15.7 med=39.4 win=60.0 T3R=-59.8 max_loss=-45.1 tail<-100=0 | N=20 sum=-958.1 mean=-47.9 med=-17.5 win=45.0 T3R=-1345.3 max_loss=-338.0 tail<-100=6 |

## bid_depth_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `deep_bid` | N=26 sum=810.7 mean=31.2 med=8.4 win=50.0 T3R=-41.7 max_loss=-271.1 tail<-100=2 | N=11 sum=282.0 mean=25.6 med=-4.1 win=45.5 T3R=-221.7 max_loss=-271.1 tail<-100=1 | N=15 sum=528.7 mean=35.2 med=20.9 win=53.3 T3R=-253.8 max_loss=-147.1 tail<-100=1 |
| `shallow_bid` | N=25 sum=-532.1 mean=-21.3 med=15.7 win=52.0 T3R=-1120.6 max_loss=-338.0 tail<-100=6 | N=5 sum=82.2 mean=16.4 med=39.4 win=60.0 T3R=-56.0 max_loss=-45.1 tail<-100=0 | N=20 sum=-614.3 mean=-30.7 med=6.4 win=50.0 T3R=-1202.9 max_loss=-338.0 tail<-100=6 |

## absorption_gate

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `absorbed` | N=21 sum=1090.2 mean=51.9 med=36.4 win=57.1 T3R=237.7 max_loss=-271.1 tail<-100=1 | N=10 sum=296.7 mean=29.7 med=27.8 win=50.0 T3R=-207.0 max_loss=-271.1 tail<-100=1 | N=11 sum=793.5 mean=72.1 med=36.4 win=63.6 T3R=11.0 max_loss=-73.6 tail<-100=0 |
| `vacuum_like` | N=11 sum=-284.4 mean=-25.9 med=18.4 win=54.5 T3R=-536.8 max_loss=-338.0 tail<-100=2 | N=3 sum=47.9 mean=16.0 med=39.4 win=66.7 T3R=47.9 max_loss=-45.1 tail<-100=0 | N=8 sum=-332.3 mean=-41.5 med=7.7 win=50.0 T3R=-568.1 max_loss=-338.0 tail<-100=2 |
| `mixed` | N=19 sum=-527.2 mean=-27.7 med=-14.7 win=42.1 T3R=-1048.1 max_loss=-291.8 tail<-100=5 | N=3 sum=19.7 mean=6.6 med=-10.8 win=33.3 T3R=19.7 max_loss=-14.7 tail<-100=0 | N=16 sum=-546.8 mean=-34.2 med=-56.6 win=43.8 T3R=-1067.8 max_loss=-291.8 tail<-100=5 |

## Sync x Imbalance Combos

| Combo | All | Cal | Hold |
| --- | --- | --- | --- |
| `idio+bid_support` | N=10 sum=448.3 mean=44.8 med=12.5 win=50.0 T3R=-113.9 max_loss=-73.6 tail<-100=0 | N=5 sum=302.9 mean=60.6 med=59.7 win=60.0 T3R=-49.0 max_loss=-37.5 tail<-100=0 | N=5 sum=145.4 mean=29.1 med=-34.1 win=40.0 T3R=-127.0 max_loss=-73.6 tail<-100=0 |
| `sync+bid_support` | N=16 sum=710.1 mean=44.4 med=41.9 win=56.2 T3R=-141.7 max_loss=-271.1 tail<-100=2 | N=6 sum=-17.0 mean=-2.8 med=-7.5 win=33.3 T3R=-341.8 max_loss=-271.1 tail<-100=1 | N=10 sum=727.1 mean=72.7 med=68.3 win=70.0 T3R=-55.4 max_loss=-176.7 tail<-100=1 |
| `idio+ask_heavy` | N=8 sum=-508.1 mean=-63.5 med=-29.9 win=37.5 T3R=-581.8 max_loss=-291.8 tail<-100=2 | N=2 sum=-59.8 mean=-29.9 med=-29.9 win=0.0 T3R=-59.8 max_loss=-45.1 tail<-100=0 | N=6 sum=-448.3 mean=-74.7 med=-38.5 win=50.0 T3R=-522.0 max_loss=-291.8 tail<-100=2 |
| `sync+ask_heavy` | N=17 sum=-371.7 mean=-21.9 med=18.4 win=52.9 T3R=-758.8 max_loss=-338.0 tail<-100=4 | N=3 sum=138.2 mean=46.1 med=45.1 win=100.0 T3R=138.2 max_loss=39.4 tail<-100=0 | N=14 sum=-509.8 mean=-36.4 med=-17.5 win=42.9 T3R=-897.0 max_loss=-338.0 tail<-100=4 |

## Read

- Best combo by T3R: `idio+bid_support` -> N=10 sum=448.3 mean=44.8 med=12.5 win=50.0 T3R=-113.9 max_loss=-73.6 tail<-100=0.
- Within sync, bid_support vs ask_heavy delta T3R `617.1`, delta max_loss `66.9`.
- A valid overlay must improve tail/T3R in holdout, not only the all-sample mean.
