# S34 Resonance Dose Response

Generated: `2026-06-28T23:10:35.553196+00:00`

Source: `reports\research\s34\S34_SYNC_ABSORPTION_REALFILL.json`

Route: `ETHUSDT SELL deep-V 28.0bps-infbps, 4h real-fill`
Rows: `51`

## Sync Threshold Sweep

| Rank | Threshold K | Summary |
| ---: | ---: | --- |
| 1 | 1000.0 | N=6 sum=337.7 mean=56.3 med=48.4 win=66.7 T3R=-57.1 max_loss=-74.6 tail<-100=0 |
| 2 | 500.0 | N=13 sum=611.3 mean=47.0 med=39.4 win=69.2 T3R=-105.1 max_loss=-338.0 tail<-100=1 |
| 3 | 300.0 | N=22 sum=570.0 mean=25.9 med=22.0 win=59.1 T3R=-146.4 max_loss=-338.0 tail<-100=2 |
| 4 | 100.0 | N=39 sum=420.9 mean=10.8 med=18.4 win=53.8 T3R=-439.0 max_loss=-338.0 tail<-100=6 |
| 5 | 200.0 | N=33 sum=338.4 mean=10.3 med=20.4 win=54.5 T3R=-521.4 max_loss=-338.0 tail<-100=6 |
| 6 | 50.0 | N=41 sum=322.3 mean=7.9 med=15.7 win=51.2 T3R=-537.5 max_loss=-338.0 tail<-100=6 |
| 7 | 0.0 | N=51 sum=278.6 mean=5.5 med=15.7 win=51.0 T3R=-669.1 max_loss=-338.0 tail<-100=8 |

## Sync Threshold + Bid Support

| Rank | Threshold K | Summary |
| ---: | ---: | --- |
| 1 | 0.0 | N=26 sum=1158.4 mean=44.6 med=30.0 win=53.8 T3R=210.7 max_loss=-271.1 tail<-100=2 |
| 2 | 300.0 | N=10 sum=647.8 mean=64.8 med=41.9 win=60.0 T3R=10.8 max_loss=-74.6 tail<-100=0 |
| 3 | 500.0 | N=6 sum=619.8 mean=103.3 med=68.3 win=83.3 T3R=9.3 max_loss=-74.6 tail<-100=0 |
| 4 | 1000.0 | N=2 sum=1.7 mean=0.9 med=0.9 win=50.0 T3R=1.7 max_loss=-74.6 tail<-100=0 |
| 5 | 100.0 | N=20 sum=739.7 mean=37.0 med=9.7 win=50.0 T3R=-112.7 max_loss=-271.1 tail<-100=2 |
| 6 | 200.0 | N=16 sum=710.1 mean=44.4 med=41.9 win=56.2 T3R=-141.7 max_loss=-271.1 tail<-100=2 |
| 7 | 50.0 | N=21 sum=686.3 mean=32.7 med=-4.1 win=47.6 T3R=-166.1 max_loss=-271.1 tail<-100=2 |

## Asset Count

| Asset Count | All | Bid Support Only |
| ---: | --- | --- |
| 1 | N=20 sum=91.3 mean=4.6 med=-13.1 win=45.0 T3R=-651.8 max_loss=-291.8 tail<-100=3 | N=11 sum=746.5 mean=67.9 med=36.4 win=54.5 T3R=3.4 max_loss=-73.6 tail<-100=0 |
| 2 | N=28 sum=-22.3 mean=-0.8 med=19.4 win=53.6 T3R=-758.2 max_loss=-338.0 tail<-100=5 | N=14 sum=335.6 mean=24.0 med=9.7 win=50.0 T3R=-372.8 max_loss=-271.1 tail<-100=2 |
| 3 | N=3 sum=209.7 mean=69.9 med=76.3 win=66.7 T3R=209.7 max_loss=-2.9 tail<-100=0 | N=1 sum=76.3 mean=76.3 med=76.3 win=100.0 T3R=76.3 max_loss=76.3 tail<-100=0 |

## Read

- Best threshold without absorption: `1000.0K` -> N=6 sum=337.7 mean=56.3 med=48.4 win=66.7 T3R=-57.1 max_loss=-74.6 tail<-100=0.
- Best threshold with bid_support: `0.0K` -> N=26 sum=1158.4 mean=44.6 med=30.0 win=53.8 T3R=210.7 max_loss=-271.1 tail<-100=2.
- This is a dose-response screen; a threshold is only believable if it improves T3R/tails without collapsing N.
