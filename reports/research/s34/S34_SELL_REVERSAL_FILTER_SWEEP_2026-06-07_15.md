# S34 SELL Reversal LONG Regime Filter Sweep

Date: 2026-06-16

Base candidate: `SELL_REVERSAL_LONG 200000 TP40 DELAY300s`

Goal: find a no-lookahead discriminator that keeps the delayed SELL-liquidation bounce behavior while reducing the 2026-06-07 failure mode. Features use only data available at signal time or during the 300s wait before entry.

## Top Filters

| Rank | Filter | N | Days | Mean | Median | Cum | WR | TP/SL/BE/TIME | 06-07 Cum | 06-11 Cum | 06-14 Cum | 06-15 Cum |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | eth_wait5_bps <= 20 AND btc_wait5_bps >= 0 | 8 | 4 | +22.59 | +32.83 | +180.70 | 87.5% | 7/1/0/0 | +32.96 | +33.09 | -49.82 | +164.48 |
| 2 | eth_pre5_bps >= -40 AND day_sell_liq_m <= 5 | 8 | 4 | +22.85 | +32.81 | +182.79 | 87.5% | 7/1/0/0 | +32.30 | +100.38 | +17.53 | +32.58 |
| 3 | eth_pre5_bps >= -40 AND day_range_bps <= 600 | 8 | 3 | +12.44 | +32.80 | +99.54 | 75.0% | 6/2/0/0 | -18.37 | +100.38 | +17.53 | n/a |
| 4 | eth_pre5_bps >= -40 AND day_sell_liq_m >= 2 | 8 | 4 | +7.55 | +32.80 | +60.39 | 62.5% | 5/2/1/0 | -50.67 | +100.38 | +67.35 | -56.67 |
| 5 | btc_pre15_bps >= -40 AND day_sell_liq_m >= 2 | 11 | 4 | +14.74 | +32.74 | +162.13 | 63.6% | 7/1/1/2 | +65.70 | +66.14 | +67.35 | -37.06 |
| 6 | btc_wait5_bps >= 0 AND btc_wait5_bps <= 20 | 11 | 4 | +17.87 | +32.71 | +196.52 | 81.8% | 9/2/0/0 | -17.72 | +99.58 | -49.82 | +164.48 |
| 7 | day_sell_liq_m <= 5 AND day_agg_count >= 750000 | 12 | 3 | +12.80 | +32.66 | +153.65 | 66.7% | 8/2/0/2 | +1.08 | +100.38 | n/a | +52.19 |
| 8 | btc_pre15_bps >= -40 AND day_trend_bps <= 400 | 8 | 4 | +12.55 | +32.65 | +100.39 | 75.0% | 6/2/0/0 | +65.05 | +66.14 | +17.53 | -48.32 |
| 9 | btc_wait5_bps >= 0 AND day_trend_bps >= 300 | 11 | 3 | +25.36 | +32.63 | +278.98 | 90.9% | 10/1/0/0 | +32.96 | +33.04 | n/a | +212.98 |
| 10 | btc_pre15_bps >= -60 AND btc_wait5_bps >= 0 | 15 | 4 | +21.92 | +32.63 | +328.79 | 86.7% | 13/2/0/0 | +32.96 | +165.01 | -49.82 | +180.65 |
| 11 | btc_wait5_bps >= 0 AND day_agg_count >= 750000 | 15 | 3 | +16.34 | +32.63 | +245.09 | 80.0% | 12/3/0/0 | -68.27 | +100.38 | n/a | +212.98 |
| 12 | btc_pre15_bps >= -60 AND day_agg_count >= 750000 | 21 | 3 | +15.54 | +32.63 | +326.32 | 71.4% | 15/3/1/2 | +34.03 | +100.38 | n/a | +191.90 |
| 13 | eth_wait5_bps <= 20 AND day_sell_liq_m >= 2 | 19 | 4 | +13.65 | +32.63 | +259.28 | 68.4% | 13/3/1/2 | -48.36 | +33.09 | +131.39 | +143.16 |
| 14 | eth_wait5_bps <= 20 AND day_trend_bps >= 0 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 15 | eth_wait5_bps <= 20 AND day_trend_bps >= 100 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 16 | eth_wait5_bps <= 20 AND day_range_bps >= 200 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 17 | eth_wait5_bps <= 20 AND day_range_bps >= 300 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 18 | eth_wait5_bps <= 20 AND day_buy_liq_m >= 2 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 19 | eth_wait5_bps <= 20 AND day_buy_liq_m >= 5 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |
| 20 | eth_wait5_bps <= 20 AND day_agg_count >= 250000 | 17 | 3 | +11.34 | +32.63 | +192.77 | 64.7% | 11/3/1/2 | -16.06 | +33.09 | n/a | +175.73 |

## Verdict

This sweep is exploratory. The best-looking rows are small-N filters, often with only one trade on some days, so they are not strong enough to promote into live paper. The most interpretable candidate is still the base idea: `SELL liquidation >=200K`, wait 300 seconds, then LONG TP40/SL40/BE30. A filter may improve it, but the current data is too small to choose one without curve-fitting.

Current decision: keep SELL reversal LONG as research-only. Do not add it to the active runner yet. Revisit after more liquidation days, or test it as an offline replay family only.
