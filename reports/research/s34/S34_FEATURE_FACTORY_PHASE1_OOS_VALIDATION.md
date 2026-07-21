# S34 Feature Factory Phase 1 OOS Validation

Purpose: control the Phase 1 query-layer multiple-testing risk by selecting filters on the first half of the event timeline and validating them on the second half.

Evaluated candidates: `783`
Surviving train-selected candidates with test support: `145`
Split timestamp ms: `1776396694616`

Selection rule on train: N >= 20, days >= 4, median > 0, top3-removed cum > 0.
Validation reporting on test: N >= 10, days >= 4.

## Top OOS Candidates

| Rank | Route | Filter | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test WR | Test Top3 Removed | Test Positive Days |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND day_trend_bps >= 0 | 20 | +41.11 | +431.86 | 33 | +53.51 | +33.74 | +1113.57 | 72.7% | +931.04 | 12/12 |
| 2 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_buy_liq_notional >= 5000000 | 21 | +52.16 | +462.66 | 42 | +52.63 | +28.38 | +1191.86 | 66.7% | +1009.83 | 11/11 |
| 3 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_trend_bps >= 0 | 45 | +29.91 | +955.72 | 52 | +52.63 | +26.43 | +1374.12 | 63.5% | +1191.59 | 12/12 |
| 4 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_trend_bps >= 100 | 34 | +37.20 | +663.17 | 35 | +52.77 | +24.62 | +861.56 | 60.0% | +679.54 | 8/8 |
| 5 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 0 AND day_range_bps >= 500 | 77 | +52.33 | +1774.85 | 19 | +52.81 | +21.65 | +411.27 | 57.9% | +236.73 | 5/5 |
| 6 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_count >= 3 | 29 | +52.31 | +658.93 | 39 | +52.86 | +32.70 | +1275.39 | 71.8% | +1092.86 | 13/14 |
| 7 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND symbol_pre_15m_bps >= 50 | 22 | +52.62 | +552.93 | 12 | +53.92 | +44.87 | +538.41 | 83.3% | +356.39 | 8/9 |
| 8 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 50 AND day_trend_bps >= 100 | 39 | +52.59 | +1110.54 | 10 | +53.14 | +32.51 | +325.11 | 70.0% | +143.09 | 7/8 |
| 9 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 0 AND day_trend_bps >= 100 | 101 | +52.11 | +1891.03 | 52 | +43.13 | +15.59 | +810.69 | 51.9% | +628.67 | 7/8 |
| 10 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_intensity_notional_per_sec >= 5000 | 23 | +52.44 | +481.21 | 38 | +53.19 | +32.16 | +1222.21 | 71.1% | +1039.68 | 13/15 |
| 11 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 | 29 | +52.31 | +658.93 | 40 | +52.82 | +30.66 | +1226.37 | 70.0% | +1043.83 | 13/15 |
| 12 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND cluster_count >= 2 | 29 | +52.31 | +658.93 | 40 | +52.82 | +30.66 | +1226.37 | 70.0% | +1043.83 | 13/15 |
| 13 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND symbol_pre_5m_bps >= 0 | 27 | +52.31 | +649.12 | 40 | +52.82 | +30.66 | +1226.37 | 70.0% | +1043.83 | 13/15 |
| 14 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND day_agg_count >= 250000 | 28 | +41.17 | +606.62 | 31 | +54.26 | +32.00 | +992.13 | 71.0% | +809.60 | 12/14 |
| 15 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 25 AND day_trend_bps >= 100 | 59 | +52.34 | +1103.39 | 22 | +52.79 | +21.55 | +474.12 | 59.1% | +296.06 | 6/7 |
| 16 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND symbol_pre_5m_bps >= 25 | 23 | +52.44 | +442.98 | 22 | +52.67 | +24.52 | +539.51 | 63.6% | +359.73 | 11/13 |
| 17 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND cluster_count >= 3 | 67 | +52.06 | +1453.20 | 70 | +52.27 | +26.29 | +1840.06 | 64.3% | +1657.53 | 16/19 |
| 18 | LONG_DELAY0_TP60 | cluster_notional >= 500000 | 69 | +29.91 | +1395.63 | 71 | +52.20 | +25.23 | +1791.03 | 63.4% | +1608.50 | 16/19 |
| 19 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND cluster_count >= 2 | 69 | +29.91 | +1395.63 | 71 | +52.20 | +25.23 | +1791.03 | 63.4% | +1608.50 | 16/19 |
| 20 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND day_range_bps >= 250 | 20 | +29.82 | +372.21 | 21 | +52.77 | +28.85 | +605.82 | 66.7% | +423.80 | 9/11 |
| 21 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 25 AND day_trend_bps >= 0 | 82 | +52.36 | +1540.92 | 35 | +4.38 | +15.72 | +550.25 | 51.4% | +372.20 | 9/11 |
| 22 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND btc_pre_15m_bps >= 0 | 56 | +21.34 | +1011.95 | 60 | +52.47 | +26.91 | +1614.41 | 66.7% | +1431.88 | 13/16 |
| 23 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 50 AND day_trend_bps >= 0 | 53 | +52.44 | +1269.77 | 13 | +53.51 | +32.72 | +425.40 | 69.2% | +243.38 | 8/10 |
| 24 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND symbol_pre_15m_bps >= 0 | 24 | +41.11 | +517.62 | 37 | +52.86 | +30.48 | +1127.67 | 70.3% | +945.14 | 12/15 |
| 25 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_agg_count >= 750000 | 48 | +52.13 | +1085.07 | 23 | +52.17 | +24.40 | +561.29 | 65.2% | +387.64 | 8/10 |
| 26 | LONG_DELAY0_TP60 | symbol_pre_15m_bps >= 0 AND day_range_bps >= 500 | 76 | +52.14 | +1664.88 | 20 | +0.93 | +15.04 | +300.74 | 50.0% | +126.21 | 4/5 |
| 27 | LONG_DELAY0_TP60 | cluster_count >= 3 AND day_range_bps >= 500 | 107 | +52.38 | +2592.92 | 23 | +9.89 | +14.15 | +325.35 | 52.2% | +150.82 | 4/5 |
| 28 | LONG_DELAY0_TP60 | symbol_pre_5m_bps >= 0 AND day_range_bps >= 500 | 91 | +52.33 | +2069.38 | 23 | +9.89 | +14.15 | +325.35 | 52.2% | +150.82 | 4/5 |
| 29 | LONG_DELAY0_TP60 | day_range_bps >= 500 | 108 | +52.36 | +2544.34 | 24 | +0.93 | +11.50 | +276.11 | 50.0% | +101.57 | 4/5 |
| 30 | LONG_DELAY0_TP60 | cluster_notional >= 200000 AND day_range_bps >= 500 | 108 | +52.36 | +2544.34 | 24 | +0.93 | +11.50 | +276.11 | 50.0% | +101.57 | 4/5 |

## Read

A candidate is not accepted just because it appears in this table. It still needs real bid/ask fill parity and forward paper validation. This table only checks whether a train-selected no-lookahead filter survives a simple temporal holdout.
