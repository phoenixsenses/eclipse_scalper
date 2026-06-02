# Narrow Event Lane Alpha Scan

- db: `data\microstructure.db`
- min_n: `8`
- events: `3077`

## Baselines

| family | n | WR | mean_bps | median_bps |
|---|---:|---:|---:|---:|
| BTCUSDT_BUY_liq_1000000_SHORT_900s | 47 | 57.45% | 6.73 | 2.50 |
| BTCUSDT_BUY_liq_100000_SHORT_900s | 250 | 52.40% | 4.66 | 0.69 |
| BTCUSDT_BUY_liq_250000_SHORT_900s | 250 | 51.60% | 5.27 | 0.69 |
| BTCUSDT_BUY_liq_500000_SHORT_900s | 143 | 60.14% | 8.47 | 5.01 |
| BTCUSDT_SELL_liq_1000000_LONG_900s | 31 | 61.29% | 6.70 | 9.44 |
| BTCUSDT_SELL_liq_100000_LONG_900s | 250 | 63.20% | 10.24 | 7.94 |
| BTCUSDT_SELL_liq_250000_LONG_900s | 250 | 62.00% | 4.04 | 8.26 |
| BTCUSDT_SELL_liq_500000_LONG_900s | 85 | 65.88% | 2.29 | 11.15 |
| ETHUSDT_BUY_liq_1000000_SHORT_900s | 50 | 64.00% | 4.88 | 13.04 |
| ETHUSDT_BUY_liq_100000_SHORT_900s | 250 | 57.20% | 9.01 | 6.89 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | 250 | 65.60% | 10.95 | 10.89 |
| ETHUSDT_BUY_liq_500000_SHORT_900s | 121 | 63.64% | 7.52 | 11.55 |
| ETHUSDT_S34_detector_SHORT_120s | 73 | 57.53% | 1.02 | 2.93 |
| ETHUSDT_S34_detector_SHORT_900s | 73 | 58.90% | 4.57 | 10.37 |
| ETHUSDT_SELL_liq_1000000_LONG_900s | 31 | 45.16% | -7.56 | -3.94 |
| ETHUSDT_SELL_liq_100000_LONG_900s | 250 | 54.40% | 5.73 | 3.54 |
| ETHUSDT_SELL_liq_250000_LONG_900s | 250 | 58.00% | 2.38 | 8.59 |
| ETHUSDT_SELL_liq_500000_LONG_900s | 101 | 52.48% | -1.64 | 5.16 |
| SOLUSDT_BUY_liq_100000_SHORT_900s | 24 | 70.83% | 14.41 | 11.34 |
| SOLUSDT_BUY_liq_25000_SHORT_900s | 76 | 68.42% | 13.22 | 12.93 |
| SOLUSDT_BUY_liq_50000_SHORT_900s | 46 | 73.91% | 15.78 | 15.33 |
| SOLUSDT_SELL_liq_100000_LONG_900s | 23 | 60.87% | 4.19 | 4.85 |
| SOLUSDT_SELL_liq_25000_LONG_900s | 114 | 58.77% | 3.13 | 4.60 |
| SOLUSDT_SELL_liq_50000_LONG_900s | 39 | 58.97% | 5.90 | 6.98 |

## Top Lane Uplifts

| family | lane | n | kept | WR | mean_bps | uplift_bps |
|---|---|---:|---:|---:|---:|---:|
| ETHUSDT_S34_detector_SHORT_900s | utc_hour_14 | 8 | 11.0% | 75.00% | 60.60 | 56.03 |
| ETHUSDT_BUY_liq_500000_SHORT_900s | utc_hour_14 | 15 | 12.4% | 80.00% | 60.81 | 53.29 |
| BTCUSDT_SELL_liq_100000_LONG_900s | utc_hour_00 | 10 | 4.0% | 100.00% | 59.62 | 49.38 |
| BTCUSDT_BUY_liq_100000_SHORT_900s | utc_hour_07 | 17 | 6.8% | 100.00% | 53.86 | 49.20 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | utc_hour_14 | 24 | 9.6% | 79.17% | 57.62 | 46.67 |
| ETHUSDT_BUY_liq_1000000_SHORT_900s | utc_hour_14 | 8 | 16.0% | 62.50% | 50.49 | 45.62 |
| BTCUSDT_BUY_liq_250000_SHORT_900s | utc_hour_09 | 13 | 5.2% | 84.62% | 47.14 | 41.87 |
| ETHUSDT_SELL_liq_100000_LONG_900s | utc_hour_00 | 17 | 6.8% | 70.59% | 47.50 | 41.77 |
| ETHUSDT_SELL_liq_100000_LONG_900s | utc_hour_15 | 8 | 3.2% | 87.50% | 46.23 | 40.50 |
| ETHUSDT_SELL_liq_250000_LONG_900s | utc_hour_00 | 11 | 4.4% | 63.64% | 41.10 | 38.72 |
| ETHUSDT_BUY_liq_500000_SHORT_900s | utc_hour_19 | 11 | 9.1% | 90.91% | 44.87 | 37.35 |
| ETHUSDT_BUY_liq_100000_SHORT_900s | utc_hour_07 | 16 | 6.4% | 75.00% | 40.21 | 31.20 |
| BTCUSDT_BUY_liq_500000_SHORT_900s | weekday_5 | 8 | 5.6% | 87.50% | 38.85 | 30.38 |
| ETHUSDT_BUY_liq_1000000_SHORT_900s | session_us | 20 | 40.0% | 70.00% | 34.74 | 29.86 |
| ETHUSDT_SELL_liq_1000000_LONG_900s | session_us | 11 | 35.5% | 63.64% | 21.35 | 28.91 |
| ETHUSDT_S34_detector_SHORT_900s | s34_session_us_peak | 22 | 30.1% | 72.73% | 32.97 | 28.40 |
| ETHUSDT_BUY_liq_500000_SHORT_900s | weekday_1 | 17 | 14.0% | 58.82% | 33.90 | 26.38 |
| ETHUSDT_SELL_liq_250000_LONG_900s | utc_hour_18 | 14 | 5.6% | 92.86% | 28.74 | 26.36 |
| ETHUSDT_S34_detector_SHORT_900s | session_us | 25 | 34.2% | 72.00% | 29.01 | 24.44 |
| ETHUSDT_S34_detector_SHORT_900s | weekday_2 | 13 | 17.8% | 76.92% | 28.99 | 24.42 |
| BTCUSDT_SELL_liq_500000_LONG_900s | utc_hour_13 | 9 | 10.6% | 66.67% | 26.34 | 24.05 |
| BTCUSDT_SELL_liq_250000_LONG_900s | utc_hour_13 | 21 | 8.4% | 71.43% | 27.49 | 23.45 |
| ETHUSDT_S34_detector_SHORT_900s | liq_comp_unknown | 8 | 11.0% | 62.50% | 27.44 | 22.87 |
| BTCUSDT_SELL_liq_100000_LONG_900s | utc_hour_15 | 15 | 6.0% | 93.33% | 33.08 | 22.84 |
| ETHUSDT_S34_detector_SHORT_900s | basis_positive | 31 | 42.5% | 80.65% | 27.38 | 22.81 |
| BTCUSDT_SELL_liq_100000_LONG_900s | utc_hour_13 | 28 | 11.2% | 78.57% | 32.78 | 22.54 |
| ETHUSDT_SELL_liq_500000_LONG_900s | solusdt_sell_100k_overlap_60s | 9 | 8.9% | 55.56% | 20.90 | 22.53 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | weekday_1 | 38 | 15.2% | 68.42% | 32.46 | 21.51 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | utc_hour_07 | 12 | 4.8% | 83.33% | 32.44 | 21.49 |
| SOLUSDT_BUY_liq_100000_SHORT_900s | funding_negative | 9 | 37.5% | 88.89% | 35.22 | 20.82 |
| SOLUSDT_SELL_liq_25000_LONG_900s | weekday_3 | 12 | 10.5% | 66.67% | 23.86 | 20.72 |
| ETHUSDT_BUY_liq_100000_SHORT_900s | weekday_1 | 20 | 8.0% | 75.00% | 29.70 | 20.69 |
| BTCUSDT_BUY_liq_100000_SHORT_900s | utc_hour_16 | 8 | 3.2% | 87.50% | 25.17 | 20.51 |
| BTCUSDT_SELL_liq_500000_LONG_900s | weekday_3 | 13 | 15.3% | 84.62% | 22.38 | 20.09 |
| ETHUSDT_S34_detector_SHORT_900s | confidence_medium | 22 | 30.1% | 72.73% | 24.63 | 20.05 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | utc_hour_01 | 8 | 3.2% | 100.00% | 30.66 | 19.72 |
| ETHUSDT_BUY_liq_250000_SHORT_900s | utc_hour_19 | 23 | 9.2% | 73.91% | 30.60 | 19.66 |
| BTCUSDT_BUY_liq_250000_SHORT_900s | utc_hour_07 | 10 | 4.0% | 70.00% | 24.61 | 19.35 |
| ETHUSDT_SELL_liq_1000000_LONG_900s | weekday_3 | 8 | 25.8% | 62.50% | 11.11 | 18.67 |
| ETHUSDT_BUY_liq_500000_SHORT_900s | session_us | 62 | 51.2% | 72.58% | 25.86 | 18.34 |
