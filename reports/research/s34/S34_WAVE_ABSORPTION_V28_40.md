# S34 Wave Absorption

Generated: `2026-06-28T23:05:46.768336+00:00`

Research-only. Tests whether top-of-book liquidity at the knowable threshold cross separates revert vs runaway. No live/paper state changed.

Route: `ETHUSDT SELL deep-V 28.0bps-40.0bps, 200K, 4h LONG fade, cost 8.1bps`
Book coverage rows: `39`; split `chronological_month_tail`, holdout months `['2026-06']`

## Overall

- All: N=39 sum=969.3 mean=24.9 med=15.6 win=59.0 T3R=151.7 max_loss=-293.4 tail<-100=3
- Calibration: N=14 sum=441.0 mean=31.5 med=-11.4 win=42.9 T3R=-25.2 max_loss=-60.7 tail<-100=0
- Holdout: N=25 sum=528.3 mean=21.1 med=18.4 win=68.0 T3R=-289.3 max_loss=-293.4 tail<-100=3

## Cuts

- `spread_med`: `0.1`
- `bid_depth_med`: `139272.4`
- `total_depth_med`: `307715.8`
- `imbalance_med`: `0.1`
- `spread_p75`: `0.1`
- `bid_depth_p25`: `71726.1`
- `imbalance_p25`: `-0.6`

## absorption_state

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `absorbed` | N=10 sum=533.7 mean=53.4 med=47.1 win=50.0 T3R=67.5 max_loss=-60.7 tail<-100=0 | N=9 sum=432.3 mean=48.0 med=-7.2 win=44.4 T3R=-33.9 max_loss=-60.7 tail<-100=0 | N=1 sum=101.4 mean=101.4 med=101.4 win=100.0 T3R=101.4 max_loss=101.4 tail<-100=0 |
| `vacuum` | N=1 sum=-36.0 mean=-36.0 med=-36.0 win=0.0 T3R=-36.0 max_loss=-36.0 tail<-100=0 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=1 sum=-36.0 mean=-36.0 med=-36.0 win=0.0 T3R=-36.0 max_loss=-36.0 tail<-100=0 |
| `mixed` | N=28 sum=471.6 mean=16.8 med=17.0 win=64.3 T3R=-346.0 max_loss=-293.4 tail<-100=3 | N=5 sum=8.7 mean=1.7 med=-17.4 win=40.0 T3R=-65.8 max_loss=-47.6 tail<-100=0 | N=23 sum=462.9 mean=20.1 med=18.4 win=69.6 T3R=-354.7 max_loss=-293.4 tail<-100=3 |

## spread_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `wide_spread` | N=19 sum=937.3 mean=49.3 med=18.4 win=73.7 T3R=119.7 max_loss=-139.3 tail<-100=1 | N=0 sum=0.0 mean=None med=None win=None T3R=0.0 max_loss=None tail<-100=0 | N=19 sum=937.3 mean=49.3 med=18.4 win=73.7 T3R=119.7 max_loss=-139.3 tail<-100=1 |
| `tight_spread` | N=20 sum=32.0 mean=1.6 med=-11.4 win=45.0 T3R=-434.2 max_loss=-293.4 tail<-100=2 | N=14 sum=441.0 mean=31.5 med=-11.4 win=42.9 T3R=-25.2 max_loss=-60.7 tail<-100=0 | N=6 sum=-409.0 mean=-68.2 med=-32.1 win=50.0 T3R=-589.3 max_loss=-293.4 tail<-100=2 |

## bid_depth_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `deep_bid` | N=20 sum=900.7 mean=45.0 med=18.7 win=55.0 T3R=184.5 max_loss=-94.6 tail<-100=0 | N=10 sum=414.1 mean=41.4 med=-11.4 win=40.0 T3R=-52.1 max_loss=-60.7 tail<-100=0 | N=10 sum=486.7 mean=48.7 med=27.1 win=70.0 T3R=-82.2 max_loss=-94.6 tail<-100=0 |
| `shallow_bid` | N=19 sum=68.6 mean=3.6 med=15.4 win=63.2 T3R=-517.2 max_loss=-293.4 tail<-100=3 | N=4 sum=26.9 mean=6.7 med=11.0 win=50.0 T3R=-47.6 max_loss=-47.6 tail<-100=0 | N=15 sum=41.7 mean=2.8 med=15.4 win=66.7 T3R=-544.1 max_loss=-293.4 tail<-100=3 |

## total_depth_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `shallow_top` | N=19 sum=890.5 mean=46.9 med=19.0 win=73.7 T3R=273.1 max_loss=-212.7 tail<-100=1 | N=5 sum=556.3 mean=111.3 med=116.8 win=80.0 T3R=90.1 max_loss=-17.4 tail<-100=0 | N=14 sum=334.2 mean=23.9 med=17.0 win=71.4 T3R=-170.4 max_loss=-212.7 tail<-100=1 |
| `deep_top` | N=20 sum=78.8 mean=3.9 med=-11.4 win=45.0 T3R=-541.6 max_loss=-293.4 tail<-100=2 | N=9 sum=-115.3 mean=-12.8 med=-18.0 win=22.2 T3R=-200.0 max_loss=-60.7 tail<-100=0 | N=11 sum=194.1 mean=17.6 med=35.1 win=63.6 T3R=-426.2 max_loss=-293.4 tail<-100=2 |

## imbalance_bucket

| Group | All | Cal | Hold |
| --- | --- | --- | --- |
| `bid_support` | N=20 sum=1162.2 mean=58.1 med=26.7 win=55.0 T3R=350.9 max_loss=-83.1 tail<-100=0 | N=10 sum=414.9 mean=41.5 med=-11.4 win=40.0 T3R=-51.3 max_loss=-60.7 tail<-100=0 | N=10 sum=747.3 mean=74.7 med=53.0 win=70.0 T3R=10.8 max_loss=-83.1 tail<-100=0 |
| `ask_heavy` | N=19 sum=-192.9 mean=-10.2 med=15.4 win=63.2 T3R=-576.3 max_loss=-293.4 tail<-100=3 | N=4 sum=26.1 mean=6.5 med=10.5 win=50.0 T3R=-47.6 max_loss=-47.6 tail<-100=0 | N=15 sum=-219.0 mean=-14.6 med=15.4 win=66.7 T3R=-602.3 max_loss=-293.4 tail<-100=3 |

## Worst Cards

| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2026-06-16T12:37:04.398000+00:00 | -293.4 | `mixed` | 0.1 | 74663.2 | -0.6 | 39.7 |
| 2026-06-17T18:00:30.110000+00:00 | -212.7 | `mixed` | 0.1 | 55708.5 | -0.6 | 37.6 |
| 2026-06-23T07:59:44.477000+00:00 | -139.3 | `mixed` | 0.1 | 111445.1 | -0.4 | 30.6 |
| 2026-06-27T17:02:17.194000+00:00 | -94.6 | `mixed` | 0.1 | 605223.3 | -0.9 | 33.2 |
| 2026-06-16T02:27:55.467000+00:00 | -83.1 | `mixed` | 0.1 | 108279.9 | 1.0 | 34.0 |
| 2026-06-26T09:48:15.082000+00:00 | -75.1 | `mixed` | 0.1 | 278147.3 | 0.8 | 28.1 |
| 2026-04-17T15:03:56.583000+00:00 | -60.7 | `absorbed` | 0.0 | 490691.3 | 1.0 | 33.9 |
| 2026-06-12T15:34:01.192000+00:00 | -55.8 | `mixed` | 0.1 | 279368.1 | 1.0 | 33.0 |

## Best Cards

| UTC | Net | State | Spread | Bid depth | Imbalance | V-depth |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 2026-06-26T13:18:54.877000+00:00 | 366.9 | `mixed` | 0.1 | 829882.1 | 0.8 | 36.3 |
| 2026-06-26T02:48:30.475000+00:00 | 268.1 | `mixed` | 0.1 | 135918.4 | 1.0 | 39.6 |
| 2026-06-21T23:33:42.690000+00:00 | 182.6 | `mixed` | 0.1 | 136804.1 | -0.4 | 29.8 |
| 2026-04-20T14:41:11.191000+00:00 | 176.2 | `absorbed` | 0.0 | 192825.1 | 1.0 | 34.1 |
| 2026-04-15T15:30:53.166000+00:00 | 173.1 | `absorbed` | 0.0 | 171242.5 | 0.2 | 31.5 |
| 2026-06-18T15:57:31.634000+00:00 | 135.0 | `mixed` | 0.1 | 56439.3 | -0.6 | 29.2 |
| 2026-04-20T14:08:39.155000+00:00 | 116.8 | `absorbed` | 0.0 | 177537.4 | 0.6 | 31.4 |
| 2026-04-26T21:33:07.357000+00:00 | 107.5 | `absorbed` | 0.0 | 207974.4 | 0.9 | 29.9 |

## Read

- Absorbed vs vacuum delta max_loss: `-24.7` bps; delta T3R: `103.5` bps.
- Treat this as a separator screen. A useful absorption feature must reduce tails and hold up in the later-month holdout, not just improve median.
