# S34 Regime Gate (is the June-vs-April split knowable?)

Generated: `2026-06-28T22:25:37.499488+00:00`  |  ETHUSDT SELL deep-V>= 28.0bps 200K 4h fade, cost 8.1bps RT, bridged

Events: 169

## Regime feature by month (does June differ from April?)

| Month | N | net sum | net win | eth_rv24_bps | btc_abs24_bps | btc_ret24_bps | eth_day_trend_bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-02 | 35 | -223.3 | 54.3 | 80.0 | 250.8 | -212.7 | -166.8 |
| 2026-03 | 61 | 1002.3 | 59.0 | 79.5 | 197.9 | -43.0 | -63.3 |
| 2026-04 | 27 | 759.2 | 59.3 | 48.8 | 128.1 | 38.8 | 48.1 |
| 2026-06 | 46 | 19.8 | 50.0 | 68.6 | 154.1 | -70.4 | -78.5 |

## Winner vs loser (median regime feature)

| Feature | winners | losers | separates? |
| --- | ---: | ---: | --- |
| eth_rv24_bps | 69.5 | 76.3 | weak |
| btc_abs24_bps | 158.8 | 218.6 | weak |
| btc_ret24_bps | -79.8 | -46.3 | yes |
| eth_day_trend_bps | -77.5 | -63.8 | weak |

## Median-split gates (per-month P&L of the favorable half)

| Gate | half | N | sum | win | 2026-02 | 2026-03 | 2026-04 | 2026-06 | Apr&Jun both+ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| eth_rv24_bps | high | 85 | -192.4 | 50.6 | -519.8 | 271.3 | -297.3 | 353.4 |  |
| eth_rv24_bps | low | 84 | 1750.4 | 60.7 | 296.5 | 730.9 | 1056.5 | -333.5 |  |
| btc_abs24_bps | high | 82 | 377.5 | 47.6 | 72.0 | -739.6 | -63.7 | 1108.8 |  |
| btc_abs24_bps | low | 82 | 1432.9 | 64.6 | -42.9 | 1741.8 | 822.9 | -1088.9 |  |
| btc_ret24_bps | high | 82 | 517.5 | 54.9 | 192.9 | 698.7 | 282.0 | -656.0 |  |
| btc_ret24_bps | low | 82 | 1292.9 | 57.3 | -163.7 | 303.6 | 477.2 | 675.9 | YES |
| eth_day_trend_bps | high | 85 | -73.7 | 54.1 | -238.0 | 669.8 | 575.8 | -1081.2 |  |
| eth_day_trend_bps | low | 84 | 1631.6 | 57.1 | 14.7 | 332.5 | 183.4 | 1101.1 | YES |
