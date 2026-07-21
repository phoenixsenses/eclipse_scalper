# S34 Butterfly Seed (does the cascade seed predict revert vs runaway?)

Generated: `2026-06-28T21:48:48.766836+00:00`  |  ETHUSDT SELL deep-V>= 28.0bps 200K 4h

Events: 51  winners(net>0): 26  losers(net<0): 25  runaways(net<-100): 8

## Seed feature: winners vs runaways (median)

| Feature | winners med | runaways med | separates? |
| --- | ---: | ---: | --- |
| accel | 6948.2 | 7102.3 | weak |
| elapsed_sec | 112.3 | 87.4 | weak |
| dominance_pct | 57.6 | 49.9 | weak |
| liq_count | 14.5 | 12.0 | weak |
| depth_bps | 33.2 | 42.0 | weak |
| intensity_k_per_sec | 2.6 | 2.7 | weak |
| btc_ret_bps | -49.4 | -29.4 | yes |

## Entry-filter tail test (cut runaways without whipsaw)

| Filter | cal N | cal sum | cal med | cal win% | cal max_loss | cal T3R | hold N | hold sum | hold med | hold max_loss | hold T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_all | 35 | -379.3 | -4.1 | 48.6 | -291.8 | -883.0 | 16 | 657.9 | 19.4 | -338.0 | -289.8 |
| decelerating(accel<0) | 2 | 59.7 | 29.9 | 50.0 | -45.1 | 59.7 | 2 | -116.5 | -58.2 | -137.4 | -116.5 |
| decel_bucket | 2 | 59.7 | 29.9 | 50.0 | -45.1 | 59.7 | 2 | -116.5 | -58.2 | -137.4 | -116.5 |
| high_dominance>=80 | 7 | -194.8 | -11.5 | 42.9 | -271.1 | -387.6 | 3 | 18.7 | 23.6 | -67.4 | 18.7 |
| slow_build(elapsed>=med) | 18 | -134.5 | 6.4 | 50.0 | -291.8 | -607.7 | 8 | 1035.2 | 72.5 | -45.8 | 87.5 |
| fast_build(elapsed<med) | 17 | 81.0 | 15.7 | 52.9 | -271.1 | -385.4 | 8 | -703.1 | -67.3 | -338.0 | -778.8 |
