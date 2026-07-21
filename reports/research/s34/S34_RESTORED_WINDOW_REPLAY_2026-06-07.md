# S34 Restored-Window Replay - 2026-06-07

This is a post-restore forensic replay over live liquidation data collected after the WebSocket route fix.

- window_start: `2026-06-06T17:43:26+00:00`
- window_end_for_analysis: `2026-06-10T10:30:00.699000+00:00`
- cost_model: `8.0 bps round trip`

## Fixed Horizon Results

| symbol | liq side | direction | threshold | n | 5m mean | 15m mean | 30m mean | 60m mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETHUSDT | BUY | SHORT | 25000 | 116 | -11.64 | -11.73 | -13.34 | -15.16 |
| ETHUSDT | BUY | LONG | 25000 | 116 | -4.36 | -4.27 | -2.66 | -0.84 |
| ETHUSDT | SELL | SHORT | 25000 | 104 | 2.06 | 2.23 | 0.41 | -0.14 |
| ETHUSDT | SELL | LONG | 25000 | 104 | -18.06 | -18.23 | -16.41 | -15.86 |
| ETHUSDT | BUY | SHORT | 50000 | 86 | -14.76 | -19.06 | -15.37 | -13.62 |
| ETHUSDT | BUY | LONG | 50000 | 86 | -1.24 | 3.06 | -0.63 | -2.38 |
| ETHUSDT | SELL | SHORT | 50000 | 78 | 3.30 | 2.83 | 0.96 | -2.13 |
| ETHUSDT | SELL | LONG | 50000 | 78 | -19.30 | -18.83 | -16.96 | -13.87 |
| ETHUSDT | BUY | SHORT | 100000 | 65 | -18.92 | -28.32 | -22.31 | -14.17 |
| ETHUSDT | BUY | LONG | 100000 | 65 | 2.92 | 12.32 | 6.31 | -1.83 |
| ETHUSDT | SELL | SHORT | 100000 | 54 | 7.38 | 6.39 | 5.83 | 3.69 |
| ETHUSDT | SELL | LONG | 100000 | 54 | -23.38 | -22.39 | -21.83 | -19.69 |
| ETHUSDT | BUY | SHORT | 200000 | 32 | -39.76 | -41.74 | -30.84 | -12.66 |
| ETHUSDT | BUY | LONG | 200000 | 32 | 23.76 | 25.74 | 14.84 | -3.34 |
| ETHUSDT | SELL | SHORT | 200000 | 32 | 12.03 | 13.55 | 15.60 | 18.06 |
| ETHUSDT | SELL | LONG | 200000 | 32 | -28.03 | -29.55 | -31.60 | -34.06 |

## Best Stop Route Per Threshold

| symbol | threshold | n | tp bps | sl bps | BE trigger | WR | mean net bps | exits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETHUSDT | 25000 | 116 | 80.00 | 40.00 | 30.00 | 35.34% | -6.98 | `{"tp":18,"sl":48,"be":16,"time":34}` |
| ETHUSDT | 25000 | 116 | 120.00 | 40.00 | 30.00 | 45.69% | 7.15 | `{"tp":16,"sl":36,"be":14,"time":50}` |
| ETHUSDT | 25000 | 104 | 120.00 | 40.00 | 30.00 | 38.46% | 4.88 | `{"tp":14,"sl":32,"be":19,"time":39}` |
| ETHUSDT | 25000 | 104 | 120.00 | 80.00 | 30.00 | 44.23% | -9.46 | `{"tp":8,"sl":23,"be":5,"time":68}` |
| ETHUSDT | 50000 | 86 | 80.00 | 40.00 | 30.00 | 34.88% | -5.01 | `{"tp":16,"sl":35,"be":13,"time":22}` |
| ETHUSDT | 50000 | 86 | 120.00 | 40.00 | 30.00 | 50.00% | 10.31 | `{"tp":12,"sl":21,"be":12,"time":41}` |
| ETHUSDT | 50000 | 78 | 120.00 | 40.00 | 30.00 | 35.90% | 6.07 | `{"tp":12,"sl":22,"be":17,"time":27}` |
| ETHUSDT | 50000 | 78 | 120.00 | 80.00 | 30.00 | 46.15% | -6.77 | `{"tp":8,"sl":18,"be":5,"time":47}` |
| ETHUSDT | 100000 | 65 | 80.00 | 40.00 | 30.00 | 35.38% | -8.03 | `{"tp":10,"sl":30,"be":9,"time":16}` |
| ETHUSDT | 100000 | 65 | 120.00 | 40.00 | 30.00 | 47.69% | 11.53 | `{"tp":10,"sl":14,"be":10,"time":31}` |
| ETHUSDT | 100000 | 54 | 120.00 | 40.00 | 30.00 | 37.04% | 8.29 | `{"tp":10,"sl":15,"be":11,"time":18}` |
| ETHUSDT | 100000 | 54 | 40.00 | 80.00 | n/a | 53.70% | -13.39 | `{"tp":25,"sl":16,"be":0,"time":13}` |
| ETHUSDT | 200000 | 32 | 120.00 | 40.00 | 30.00 | 28.12% | -15.83 | `{"tp":2,"sl":21,"be":2,"time":7}` |
| ETHUSDT | 200000 | 32 | 80.00 | 40.00 | 30.00 | 59.38% | 23.01 | `{"tp":13,"sl":4,"be":7,"time":8}` |
| ETHUSDT | 200000 | 32 | 120.00 | 40.00 | 30.00 | 50.00% | 25.25 | `{"tp":7,"sl":5,"be":8,"time":12}` |
| ETHUSDT | 200000 | 32 | 80.00 | 80.00 | 30.00 | 34.38% | -26.48 | `{"tp":5,"sl":14,"be":1,"time":12}` |

## Interpretation

- This is not a final alpha verdict; the window is still short.
- The goal is to identify whether restored liquidation data can explain stop/TP management better than the incomplete-sensor trade.
- Any result with small `n` should be treated as directional evidence only.
