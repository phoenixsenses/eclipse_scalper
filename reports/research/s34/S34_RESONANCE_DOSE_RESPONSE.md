# S34 Resonance Dose Response

Generated: `2026-06-28T23:10:35.506359+00:00`

Source: `D:\eclipse_scalper\reports\research\s34\S34_SYNC_ABSORPTION_REALFILL_V28_40.json`

Route: `ETHUSDT SELL deep-V 28.0bps-40.0bps, 4h real-fill`
Rows: `36`

## Sync Threshold Sweep

| Rank | Threshold K | Summary |
| ---: | ---: | --- |
| 1 | 100.0 | N=25 sum=1170.8 mean=46.8 med=37.0 win=68.0 T3R=434.3 max_loss=-213.5 tail<-100=1 |
| 2 | 50.0 | N=27 sum=1072.3 mean=39.7 med=23.6 win=63.0 T3R=335.8 max_loss=-213.5 tail<-100=1 |
| 3 | 300.0 | N=15 sum=960.7 mean=64.0 med=39.4 win=73.3 T3R=262.7 max_loss=-74.6 tail<-100=0 |
| 4 | 200.0 | N=21 sum=980.7 mean=46.7 med=39.4 win=66.7 T3R=244.8 max_loss=-213.5 tail<-100=1 |
| 5 | 0.0 | N=36 sum=968.9 mean=26.9 med=20.6 win=58.3 T3R=137.2 max_loss=-291.8 tail<-100=3 |
| 6 | 500.0 | N=8 sum=783.1 mean=97.9 med=57.9 win=87.5 T3R=85.1 max_loss=-74.6 tail<-100=0 |
| 7 | 1000.0 | N=5 sum=340.6 mean=68.1 med=76.3 win=80.0 T3R=-54.2 max_loss=-74.6 tail<-100=0 |

## Sync Threshold + Bid Support

| Rank | Threshold K | Summary |
| ---: | ---: | --- |
| 1 | 0.0 | N=18 sum=1141.1 mean=63.4 med=30.0 win=55.6 T3R=316.7 max_loss=-74.6 tail<-100=0 |
| 2 | 100.0 | N=13 sum=782.1 mean=60.2 med=23.6 win=53.8 T3R=53.6 max_loss=-74.6 tail<-100=0 |
| 3 | 1000.0 | N=2 sum=1.7 mean=0.9 med=0.9 win=50.0 T3R=1.7 max_loss=-74.6 tail<-100=0 |
| 4 | 50.0 | N=14 sum=728.7 mean=52.0 med=9.7 win=50.0 T3R=0.2 max_loss=-74.6 tail<-100=0 |
| 5 | 200.0 | N=11 sum=644.8 mean=58.6 med=23.6 win=54.5 T3R=-13.8 max_loss=-74.6 tail<-100=0 |
| 6 | 500.0 | N=4 sum=404.8 mean=101.2 med=49.9 win=75.0 T3R=-74.6 max_loss=-74.6 tail<-100=0 |
| 7 | 300.0 | N=8 sum=432.8 mean=54.1 med=9.7 win=50.0 T3R=-125.8 max_loss=-74.6 tail<-100=0 |

## Asset Count

| Asset Count | All | Bid Support Only |
| ---: | --- | --- |
| 1 | N=15 sum=-11.8 mean=-0.8 med=-11.5 win=46.7 T3R=-574.1 max_loss=-291.8 tail<-100=2 | N=7 sum=496.3 mean=70.9 med=36.4 win=57.1 T3R=-65.9 max_loss=-53.4 tail<-100=0 |
| 2 | N=19 sum=768.2 mean=40.4 med=23.6 win=63.2 T3R=32.3 max_loss=-213.5 tail<-100=1 | N=10 sum=568.5 mean=56.8 med=9.7 win=50.0 T3R=-90.1 max_loss=-74.6 tail<-100=0 |
| 3 | N=2 sum=212.6 mean=106.3 med=106.3 win=100.0 T3R=212.6 max_loss=76.3 tail<-100=0 | N=1 sum=76.3 mean=76.3 med=76.3 win=100.0 T3R=76.3 max_loss=76.3 tail<-100=0 |

## Read

- Best threshold without absorption: `100.0K` -> N=25 sum=1170.8 mean=46.8 med=37.0 win=68.0 T3R=434.3 max_loss=-213.5 tail<-100=1.
- Best threshold with bid_support: `0.0K` -> N=18 sum=1141.1 mean=63.4 med=30.0 win=55.6 T3R=316.7 max_loss=-74.6 tail<-100=0.
- This is a dose-response screen; a threshold is only believable if it improves T3R/tails without collapsing N.
