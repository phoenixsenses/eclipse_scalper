# S34 Failure Geometry / Feedback (cut the tail by dynamics, not seed)

Generated: `2026-06-28T22:08:19.558715+00:00`  |  ETHUSDT SELL deep-V>= 28.0bps 200K 4h, cost 8.1bps RT, bridged span

Events: 169  |  winners: 94  runaways(net<-100): 32

## Reverter vs runaway separation (dynamic, knowable in real time)

| Feature | winners med | runaways med | separates? |
| --- | ---: | ---: | --- |
| reclaimed_5m | 1.0 | 0.0 | yes |
| mae_5m_bps | -16.1 | -23.6 | weak |
| liq_0_5_k | 101.8 | 140.6 | weak |
| liq_accel_5to10 | 0.0 | 0.2 | yes |

## Reclaim-stop variants (hold to 4h only if recovering at tau)

| Variant | cal N | cal sum | cal win | cal maxL | hold N | hold sum | hold win | hold maxL | pos months |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_4h | 118 | 1223.8 | 56.8 | -410.0 | 51 | 334.1 | 52.9 | -342.3 | 3/4 |
| reclaim_stop_5m | 118 | -302.8 | 37.3 | -410.0 | 51 | 207.0 | 33.3 | -149.0 | 3/4 |
| reclaim_stop_10m | 118 | -673.7 | 36.4 | -410.0 | 51 | -328.8 | 33.3 | -183.1 | 1/4 |
| reclaim_stop_15m | 118 | -64.4 | 39.0 | -324.0 | 51 | -393.3 | 35.3 | -255.2 | 1/4 |
| reclaim_stop_30m | 118 | 688.0 | 38.1 | -264.2 | 51 | -368.8 | 41.2 | -463.6 | 3/4 |
