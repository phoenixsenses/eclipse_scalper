# S34 Wave Absorption

Generated: `2026-06-28T23:04:44.917327+00:00`

Research-only. Tests whether top-of-book liquidity at the knowable threshold cross separates revert vs runaway. No live/paper state changed.

Route: `ETHUSDT SELL deep-V >= 28.0bps, 200K, 4h LONG fade, cost 8.1bps`
Book coverage rows: `54`; split `chronological_month_tail`, holdout months `['2026-06']`

## Overall

- All: N=54 sum=230.2 mean=4.3 med=14.0 win=51.9 T3R=-699.3 max_loss=-342.3 tail<-100=8
- Calibration: N=18 sum=427.3 mean=23.7 med=16.0 win=50.0 T3R=-76.9 max_loss=-272.6 tail<-100=1
- Holdout: N=36 sum=-197.1 mean=-5.5 med=14.0 win=52.8 T3R=-1126.6 max_loss=-342.3 tail<-100=7

## Cuts

- `spread_med`: `0.1`
- `bid_depth_med`: `138228.7`
- `total_depth_med`: `286330.1`
- `imbalance_med`: `0.0`
- `spread_p75`: `0.1`
- `bid_depth_p25`: `71372.8`
- `imbalance_p25`: `-0.6`

## absorption_state

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `absorbed` | N=12 sum=746.5 mean=62.2 med=79.7 win=58.3 T3R=242.3 max_loss=-60.7 tail<-100=0 | N=11 sum=645.0 mean=58.6 med=57.9 win=54.5 T3R=140.8 max_loss=-60.7 tail<-100=0 | N=1 sum=101.4 mean=101.4 med=101.4 win=100.0 T3R=101.4 max_loss=101.4 tail<-100=0 |
| `vacuum` | N=3 sum=-385.6 mean=-128.5 med=-36.0 win=0.0 T3R=-385.6 max_loss=-342.3 tail<-100=1 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=3 sum=-385.6 mean=-128.5 med=-36.0 win=0.0 T3R=-385.6 max_loss=-342.3 tail<-100=1 |
| `mixed` | N=39 sum=-130.7 mean=-3.4 med=14.5 win=53.8 T3R=-1060.2 max_loss=-293.4 tail<-100=7 | N=7 sum=-217.8 mean=-31.1 med=-17.4 win=42.9 T3R=-355.8 max_loss=-272.6 tail<-100=1 | N=32 sum=87.1 mean=2.7 med=15.0 win=56.2 T3R=-842.4 max_loss=-293.4 tail<-100=6 |

## spread_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `wide_spread` | N=27 sum=698.3 mean=25.9 med=15.4 win=59.3 T3R=-231.2 max_loss=-342.3 tail<-100=2 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=27 sum=698.3 mean=25.9 med=15.4 win=59.3 T3R=-231.2 max_loss=-342.3 tail<-100=2 |
| `tight_spread` | N=27 sum=-468.1 mean=-17.3 med=-15.6 win=44.4 T3R=-972.3 max_loss=-293.4 tail<-100=6 | N=18 sum=427.3 mean=23.7 med=16.0 win=50.0 T3R=-76.9 max_loss=-272.6 tail<-100=1 | N=9 sum=-895.4 mean=-99.5 med=-149.0 win=33.3 T3R=-1075.7 max_loss=-293.4 tail<-100=5 |

## bid_depth_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `deep_bid` | N=27 sum=864.5 mean=32.0 med=18.4 win=51.9 T3R=26.9 max_loss=-272.6 tail<-100=2 | N=13 sum=354.2 mean=27.2 med=-7.2 win=46.2 T3R=-150.0 max_loss=-272.6 tail<-100=1 | N=14 sum=510.3 mean=36.5 med=18.7 win=57.1 T3R=-252.5 max_loss=-149.0 tail<-100=1 |
| `shallow_bid` | N=27 sum=-634.3 mean=-23.5 med=13.4 win=51.9 T3R=-1220.0 max_loss=-342.3 tail<-100=6 | N=5 sum=73.1 mean=14.6 med=39.3 win=60.0 T3R=-65.0 max_loss=-47.6 tail<-100=0 | N=22 sum=-707.4 mean=-32.2 med=3.1 win=50.0 T3R=-1293.1 max_loss=-342.3 tail<-100=6 |

## total_depth_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `shallow_top` | N=27 sum=1041.9 mean=38.6 med=19.0 win=66.7 T3R=303.1 max_loss=-212.7 tail<-100=3 | N=8 sum=815.3 mean=101.9 med=112.2 win=87.5 T3R=311.1 max_loss=-17.4 tail<-100=0 | N=19 sum=226.6 mean=11.9 med=14.5 win=57.9 T3R=-471.0 max_loss=-212.7 tail<-100=3 |
| `deep_top` | N=27 sum=-811.7 mean=-30.1 med=-18.2 win=37.0 T3R=-1432.0 max_loss=-342.3 tail<-100=5 | N=10 sum=-388.0 mean=-38.8 med=-18.1 win=20.0 T3R=-472.7 max_loss=-272.6 tail<-100=1 | N=17 sum=-423.7 mean=-24.9 med=-36.0 win=47.1 T3R=-1044.0 max_loss=-342.3 tail<-100=4 |

## imbalance_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `bid_support` | N=27 sum=1423.2 mean=52.7 med=35.1 win=55.6 T3R=493.6 max_loss=-183.1 tail<-100=1 | N=12 sum=627.7 mean=52.3 med=25.4 win=50.0 T3R=123.5 max_loss=-60.7 tail<-100=0 | N=15 sum=795.5 mean=53.0 med=35.1 win=60.0 T3R=-134.0 max_loss=-183.1 tail<-100=1 |
| `ask_heavy` | N=27 sum=-1193.0 mean=-44.2 med=-7.3 win=48.1 T3R=-1576.3 max_loss=-342.3 tail<-100=7 | N=6 sum=-200.4 mean=-33.4 med=10.5 win=50.0 T3R=-338.5 max_loss=-272.6 tail<-100=1 | N=21 sum=-992.6 mean=-47.3 med=-7.3 win=47.6 T3R=-1375.9 max_loss=-342.3 tail<-100=6 |

## Worst Cards

| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2026-06-25T13:31:45.283000+00:00 | -342.3 | `vacuum` | 0.1 | 4368.9 | -1.0 | 45.6 |
| 2026-06-16T12:37:04.398000+00:00 | -293.4 | `mixed` | 0.1 | 74663.2 | -0.6 | 39.7 |
| 2026-04-14T14:33:52.740000+00:00 | -272.6 | `mixed` | 0.0 | 216989.4 | 0.0 | 42.4 |
| 2026-06-17T18:00:30.110000+00:00 | -212.7 | `mixed` | 0.1 | 55708.5 | -0.6 | 37.6 |
| 2026-06-15T19:18:53.490000+00:00 | -183.1 | `mixed` | 0.1 | 135044.3 | 0.3 | 41.5 |
| 2026-06-16T13:08:35.106000+00:00 | -154.3 | `mixed` | 0.1 | 30980.2 | -0.5 | 45.8 |
| 2026-06-17T18:58:35.207000+00:00 | -149.0 | `mixed` | 0.1 | 159390.3 | -0.2 | 75.0 |
| 2026-06-23T07:59:44.477000+00:00 | -139.3 | `mixed` | 0.1 | 111445.1 | -0.4 | 30.6 |

## Best Cards

| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2026-06-26T13:18:54.877000+00:00 | 366.9 | `mixed` | 0.1 | 829882.1 | 0.8 | 36.3 |
| 2026-06-26T12:39:57.361000+00:00 | 294.5 | `mixed` | 0.1 | 152617.5 | 1.0 | 106.4 |
| 2026-06-26T02:48:30.475000+00:00 | 268.1 | `mixed` | 0.1 | 135918.4 | 1.0 | 39.6 |
| 2026-06-21T23:33:42.690000+00:00 | 182.6 | `mixed` | 0.1 | 136804.1 | -0.4 | 29.8 |
| 2026-04-20T14:41:11.191000+00:00 | 176.2 | `absorbed` | 0.0 | 192825.1 | 1.0 | 34.1 |
| 2026-04-15T15:30:53.166000+00:00 | 173.1 | `absorbed` | 0.0 | 171242.5 | 0.2 | 31.5 |
| 2026-04-26T22:34:24.614000+00:00 | 154.9 | `absorbed` | 0.0 | 272162.0 | 1.0 | 50.2 |
| 2026-06-18T15:57:31.634000+00:00 | 135.0 | `mixed` | 0.1 | 56439.3 | -0.6 | 29.2 |

## Read

- Absorbed vs vacuum delta max_loss: `281.6` bps; delta T3R: `627.9` bps.
- Treat this as a separator screen. A useful absorption feature must reduce tails and hold up in the later-month holdout, not just improve median.
