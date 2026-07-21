# S34 Feature Factory Phase 1 Query Results

Scope: query layer over `data/s34_feature_factory.db`.

Only `liq_event_features` no-lookahead columns are used as filters. Outcome columns are joined only after filtering to evaluate route labels.

Eligibility: N >= 30, days >= 5.

## Top Results

| Rank | Route | Filter | N | Days | Mean | Median | Cum | WR | TP/BE/SL/TIME | Top3 Removed Cum | Positive Days | Worst Day |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 0 AND day_range_bps >= 500 | 96 | 27 | +22.77 | +52.34 | +2186.12 | 58.3% | 52/31/9/4 | +1994.82 | 24/27 | -36.35 |
| 2 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_buy_liq_notional >= 5000000 | 63 | 23 | +26.26 | +52.21 | +1654.52 | 63.5% | 37/16/6/4 | +1465.14 | 20/23 | -49.14 |
| 3 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 50 AND day_trend_bps >= 100 | 49 | 29 | +29.30 | +52.66 | +1435.65 | 63.3% | 31/15/3/0 | +1245.52 | 25/29 | -48.79 |
| 4 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND day_trend_bps >= 0 | 53 | 28 | +29.16 | +52.66 | +1545.43 | 67.9% | 33/11/4/5 | +1362.90 | 24/28 | -11.63 |
| 5 | LONG_DELAY0_TP60 | cluster_count >= 3 AND day_range_bps >= 500 | 130 | 28 | +22.45 | +52.34 | +2918.27 | 58.5% | 71/40/14/5 | +2721.37 | 24/28 | -36.35 |
| 6 | LONG_DELAY0_TP60 | cluster_count >= 2 AND day_range_bps >= 500 | 131 | 28 | +21.90 | +52.33 | +2869.02 | 58.0% | 71/40/15/5 | +2672.12 | 24/28 | -36.35 |
| 7 | LONG_DELAY0_TP60 | day_range_bps >= 500 AND day_buy_liq_notional >= 5000000 | 80 | 19 | +20.17 | +52.26 | +1613.88 | 56.2% | 42/24/11/3 | +1417.75 | 16/19 | -106.20 |
| 8 | LONG_DELAY0_TP60 | day_trend_bps >= 0 AND day_range_bps >= 500 | 81 | 17 | +19.59 | +52.33 | +1587.10 | 55.6% | 43/24/12/2 | +1391.01 | 14/17 | -10.42 |
| 9 | LONG_DELAY0_TP60 | day_range_bps >= 500 | 132 | 28 | +21.37 | +52.26 | +2820.45 | 57.6% | 71/40/16/5 | +2623.55 | 23/28 | -36.35 |
| 10 | LONG_DELAY0_TP60 | cluster_notional >= 200000 AND day_range_bps >= 500 | 132 | 28 | +21.37 | +52.26 | +2820.45 | 57.6% | 71/40/16/5 | +2623.55 | 23/28 | -36.35 |
| 11 | LONG_DELAY0_TP60 | symbol_pre_5m_bps >= 0 AND day_range_bps >= 500 | 114 | 28 | +21.01 | +52.14 | +2394.73 | 56.1% | 59/38/12/5 | +2197.83 | 23/28 | -36.35 |
| 12 | LONG_DELAY0_TP60 | day_range_bps >= 500 AND day_agg_count >= 250000 | 130 | 27 | +21.34 | +52.26 | +2774.83 | 57.7% | 70/39/16/5 | +2577.92 | 22/27 | -36.35 |
| 13 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 0 AND day_range_bps >= 500 | 96 | 27 | +20.48 | +52.08 | +1965.62 | 55.2% | 49/33/10/4 | +1768.72 | 22/27 | -36.35 |
| 14 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_count >= 3 | 68 | 37 | +28.45 | +52.57 | +1934.33 | 66.2% | 42/16/5/5 | +1751.79 | 30/37 | -48.02 |
| 15 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 25 AND day_range_bps >= 500 | 43 | 21 | +22.11 | +52.34 | +950.53 | 58.1% | 24/13/5/1 | +780.95 | 17/21 | -49.14 |
| 16 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_range_bps >= 500 | 36 | 20 | +27.83 | +52.32 | +1001.73 | 66.7% | 21/10/2/3 | +814.19 | 16/20 | -49.14 |
| 17 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 0 AND day_trend_bps >= 100 | 153 | 35 | +17.66 | +52.11 | +2701.71 | 53.6% | 79/44/26/4 | +2504.47 | 28/35 | -68.30 |
| 18 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 0 AND day_trend_bps >= 100 | 156 | 35 | +16.36 | +28.17 | +2551.78 | 51.3% | 77/49/27/3 | +2348.90 | 28/35 | -68.30 |
| 19 | LONG_DELAY0_TP60 | day_trend_bps >= 100 AND day_buy_liq_notional >= 5000000 | 109 | 24 | +15.01 | +34.09 | +1635.67 | 51.4% | 54/29/23/3 | +1439.58 | 19/24 | -92.71 |
| 20 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 | 69 | 38 | +27.32 | +52.48 | +1885.30 | 65.2% | 42/16/6/5 | +1702.77 | 30/38 | -49.03 |
| 21 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_count >= 2 | 69 | 38 | +27.32 | +52.48 | +1885.30 | 65.2% | 42/16/6/5 | +1702.77 | 30/38 | -49.03 |
| 22 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 50 AND day_trend_bps >= 0 | 66 | 38 | +25.68 | +52.55 | +1695.17 | 60.6% | 39/20/6/1 | +1504.10 | 30/38 | -48.79 |
| 23 | LONG_DELAY0_TP60 | cluster_intensity_notional_per_sec >= 5000 AND day_range_bps >= 500 | 30 | 14 | +20.64 | +52.27 | +619.25 | 60.0% | 16/8/4/2 | +442.80 | 11/14 | -104.30 |
| 24 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND symbol_pre_5m_bps >= 0 | 67 | 37 | +27.99 | +52.48 | +1875.49 | 65.7% | 41/16/5/5 | +1692.96 | 29/37 | -49.03 |
| 25 | LONG_DELAY0_TP60 | cluster_count >= 3 AND day_trend_bps >= 100 | 190 | 37 | +18.18 | +52.17 | +3453.72 | 54.2% | 100/53/32/5 | +3250.84 | 29/37 | -68.30 |
| 26 | LONG_DELAY0_TP60 | cluster_count >= 2 AND day_trend_bps >= 100 | 193 | 37 | +17.35 | +52.11 | +3348.16 | 53.4% | 100/54/34/5 | +3145.28 | 29/37 | -68.30 |
| 27 | LONG_DELAY0_TP60 | symbol_pre_5m_bps >= 0 AND day_trend_bps >= 100 | 170 | 37 | +16.22 | +28.17 | +2757.03 | 51.2% | 84/52/29/5 | +2554.15 | 29/37 | -68.30 |
| 28 | LONG_DELAY0_TP60 | day_range_bps >= 500 AND day_agg_count >= 750000 | 125 | 27 | +21.80 | +52.33 | +2725.07 | 58.4% | 68/37/15/5 | +2528.16 | 21/27 | -50.22 |
| 29 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 25 AND day_trend_bps >= 100 | 81 | 30 | +19.48 | +52.38 | +1577.50 | 55.6% | 45/22/14/0 | +1395.81 | 23/30 | -59.67 |
| 30 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_intensity_notional_per_sec >= 5000 | 61 | 34 | +27.93 | +52.71 | +1703.43 | 65.6% | 38/14/5/4 | +1520.89 | 26/34 | -49.03 |

## Read

This is still research infrastructure. A filter is interesting only if it has positive median, survives top-3 removal, and is spread across days. It still needs live bid/ask forward validation before becoming a runner rule.
