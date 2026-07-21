# S34 Day Context Scan

Generated: 2026-06-25T14:48:26.323171+00:00

Scope: ETH BUY feature-factory events, route `LONG_DELAY0_TP60`.

Day-context features are no-lookahead day-so-far values at event time.

- Rows: `450`
- Predicate count: `300`

## OOS Candidates

| Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | buy_liq_ge_5m AND day_range_bps_ge_p75_539 | 52 | +52.44 | +1241.20 | 20 | -8.37 | +10.02 | +200.31 | +29.15 | 3/6 |
| 2 | trend_ge_100 AND agg_ge_750k | 70 | +52.44 | +1788.54 | 65 | -8.03 | +11.62 | +755.39 | +579.09 | 10/15 |
| 3 | trend_ge_100 AND day_agg_count_ge_p50_942742 | 64 | +52.44 | +1639.30 | 51 | +52.11 | +12.42 | +633.59 | +459.21 | 9/13 |
| 4 | day_buy_liq_notional_ge_p50_3297789 AND day_agg_count_ge_p75_1371175 | 74 | +52.39 | +1862.19 | 24 | +14.20 | +18.72 | +449.25 | +278.64 | 7/10 |
| 5 | range_between_250_900 AND day_range_bps_ge_p75_539 | 75 | +52.39 | +1964.21 | 24 | +6.67 | +11.39 | +273.38 | +102.21 | 4/8 |
| 6 | day_range_bps_ge_p75_539 AND day_buy_liq_notional_ge_p50_3297789 | 69 | +52.39 | +1666.75 | 20 | -8.37 | +10.02 | +200.31 | +29.15 | 3/6 |
| 7 | range_ge_500 AND range_between_250_900 | 84 | +52.38 | +2115.29 | 33 | +18.50 | +13.78 | +454.61 | +278.50 | 7/10 |
| 8 | day_range_bps_ge_p50_357 AND day_buy_liq_notional_ge_p50_3297789 | 86 | +52.38 | +2111.11 | 69 | -8.37 | +6.18 | +426.52 | +244.27 | 10/16 |
| 9 | range_ge_500 AND day_buy_liq_notional_ge_p50_3297789 | 72 | +52.38 | +1707.13 | 27 | +9.89 | +14.12 | +381.14 | +205.03 | 5/7 |
| 10 | range_ge_500 AND agg_ge_750k | 95 | +52.38 | +2380.64 | 30 | +14.20 | +11.48 | +344.43 | +173.26 | 5/9 |
| 11 | day_range_bps_ge_p75_539 | 89 | +52.38 | +2225.90 | 25 | -8.03 | +10.49 | +262.24 | +91.07 | 4/8 |
| 12 | range_ge_250 AND day_range_bps_ge_p75_539 | 89 | +52.38 | +2225.90 | 25 | -8.03 | +10.49 | +262.24 | +91.07 | 4/8 |

## Real-Fill Parity

| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| buy_liq_ge_5m AND day_range_bps_ge_p75_539 | 72 | 13 | 59 (81.9%) | 13 | -9.06 | +4.35 | +56.50 | -124.83 | 1/3 |
| trend_ge_100 AND agg_ge_750k | 135 | 41 | 94 (69.6%) | 41 | -8.46 | +3.75 | +153.71 | -49.64 | 4/7 |
| trend_ge_100 AND day_agg_count_ge_p50_942742 | 115 | 30 | 85 (73.9%) | 30 | -8.24 | +3.88 | +116.40 | -83.68 | 4/7 |
| day_buy_liq_notional_ge_p50_3297789 AND day_agg_count_ge_p75_1371175 | 98 | 8 | 90 (91.8%) | 8 | -7.95 | +10.05 | +80.36 | -86.65 | 1/3 |
| range_between_250_900 AND day_range_bps_ge_p75_539 | 99 | 12 | 87 (87.9%) | 12 | -8.24 | +5.60 | +67.19 | -114.13 | 1/3 |
| day_range_bps_ge_p75_539 AND day_buy_liq_notional_ge_p50_3297789 | 89 | 13 | 76 (85.4%) | 13 | -9.06 | +4.35 | +56.50 | -124.83 | 1/3 |
| range_ge_500 AND range_between_250_900 | 117 | 13 | 104 (88.9%) | 13 | -7.42 | +9.73 | +126.45 | -55.77 | 2/3 |
| day_range_bps_ge_p50_357 AND day_buy_liq_notional_ge_p50_3297789 | 155 | 40 | 115 (74.2%) | 40 | -9.07 | -2.70 | -108.16 | -311.52 | 4/9 |
| range_ge_500 AND day_buy_liq_notional_ge_p50_3297789 | 99 | 14 | 85 (85.9%) | 14 | -8.24 | +8.27 | +115.76 | -66.47 | 2/3 |
| range_ge_500 AND agg_ge_750k | 125 | 13 | 112 (89.6%) | 13 | -9.06 | +4.35 | +56.50 | -124.83 | 1/3 |
| day_range_bps_ge_p75_539 | 114 | 13 | 101 (88.6%) | 13 | -9.06 | +4.35 | +56.50 | -124.83 | 1/3 |
| range_ge_250 AND day_range_bps_ge_p75_539 | 114 | 13 | 101 (88.6%) | 13 | -9.06 | +4.35 | +56.50 | -124.83 | 1/3 |

## Read

This is a day-context research scan. Positives remain hypothesis seeds unless separately pre-registered forward.
