# S34 Synchronization Gate (is the tail a market-wide synchronized cascade?)

Generated: `2026-06-28T22:53:02.849937+00:00`  |  ETH SELL deep-V>= 28.0bps 200K 4h fade, cost 8.1bps RT, sync window 10m, sync threshold 200.0K cross-asset sell-liq

Events: 169  (idiosyncratic 72 / synchronized 97)

## Winner vs runaway: concurrent cross-asset sell-liq (median)
- market_concurrent_k: winners=236.9 vs runaways=227.6  (weak)
- btc_ret_10m: winners=-48.9 vs runaways=-39.2

## Buckets

| Bucket | N | sum | mean | win | max_loss | T3R | cal N | cal sum | cal win | hold N | hold sum | hold win | cal&hold + | 2026-02 | 2026-03 | 2026-04 | 2026-06 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| idiosyncratic(<thr) | 72 | 6.9 | 0.1 | 50.0 | -331.4 | -905.6 | 50 | 376.0 | 54.0 | 22 | -369.1 | 40.9 |  | 394.5 | -365.1 | 623.4 | -645.9 |
| synchronized(>=thr) | 97 | 1551.1 | 16.0 | 59.8 | -410.0 | 559.6 | 67 | 1215.4 | 62.7 | 30 | 335.6 | 53.3 | YES | -617.8 | 1367.4 | 135.8 | 665.7 |
| all | 169 | 1557.9 | 9.2 | 55.6 | -410.0 | 474.3 | 118 | 1223.8 | 56.8 | 51 | 334.1 | 52.9 | YES | -223.3 | 1002.3 | 759.2 | 19.8 |
