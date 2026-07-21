# S34 Cluster Geometry Feature Scan

Generated: 2026-06-17T18:13:54.229780+00:00

Scope: add no-lookahead cluster geometry fields to `liq_event_features`, then scan relationship with `LONG_DELAY0_TP60` outcomes.

No live runner/config changes. `liq_event_outcome_labels` was not modified.

## 1. Schema / Fill Check

- Added columns: `{'cluster_liq_count': 'INTEGER', 'max_single_liq_share': 'REAL', 'intensity_per_sec': 'REAL', 'inter_cluster_gap_sec': 'REAL', 'shape_label': 'TEXT'}`
- Updated ETH BUY rows: `450`
- Predicate count: `12`

| event_id | utc | notional | duration | count | max_share | intensity/sec | gap_sec | shape |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ETHUSDT_BUY_5903985 | 2026-02-15T22:47:11.071000+00:00 | 216936 | 49.1 | 6 | 58.2% | 4419 | NA | distributed_mid_duration |
| ETHUSDT_BUY_5904001 | 2026-02-16T00:07:08.659000+00:00 | 375214 | 121.3 | 21 | 65.4% | 3094 | 4797.6 | stretched_120s |
| ETHUSDT_BUY_5904089 | 2026-02-16T07:25:17.535000+00:00 | 206266 | 169.0 | 10 | 60.4% | 1221 | 26288.9 | stretched_120s |
| ETHUSDT_BUY_5904155 | 2026-02-16T12:56:50.161000+00:00 | 474789 | 11.9 | 4 | 63.7% | 39768 | 19892.6 | distributed_mid_duration |
| ETHUSDT_BUY_5904162 | 2026-02-16T13:30:16.210000+00:00 | 221979 | 123.4 | 9 | 80.8% | 1799 | 2006.0 | single_dominant_80pct |
| ETHUSDT_BUY_5904196 | 2026-02-16T16:22:09.443000+00:00 | 220657 | 103.2 | 6 | 72.1% | 2137 | 10313.2 | distributed_mid_duration |
| ETHUSDT_BUY_5904477 | 2026-02-17T15:45:04.205000+00:00 | 301553 | 289.5 | 12 | 96.3% | 1042 | 84174.8 | single_dominant_80pct |
| ETHUSDT_BUY_5904481 | 2026-02-17T16:05:11.630000+00:00 | 544503 | 135.0 | 14 | 55.0% | 4033 | 1207.4 | stretched_120s |

## 2. OOS Geometry Candidates

| Scope | Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60_ALL_200K | 1 | liq_count_ge_p75_22 AND duration_ge_p75_221s | 38 | +52.70 | +1620.18 | 11 | +54.84 | +50.13 | +551.48 | +369.46 | 7/7 |
| LONG_DELAY0_TP60_ALL_200K | 2 | liq_count_ge_p75_22 | 94 | +53.04 | +3837.16 | 27 | +54.32 | +50.56 | +1365.04 | +1183.02 | 11/12 |
| LONG_DELAY0_TP60_ALL_200K | 3 | max_share_lt_50 AND liq_count_ge_p75_22 | 50 | +53.85 | +2410.22 | 21 | +54.32 | +49.10 | +1031.00 | +849.95 | 9/10 |
| LONG_DELAY0_TP60_ALL_200K | 4 | liq_count_ge_p75_22 AND shape_stretched_120s | 75 | +53.23 | +3287.70 | 23 | +54.26 | +49.83 | +1146.06 | +964.04 | 10/11 |
| LONG_DELAY0_TP60_ALL_200K | 5 | gap_lt_30m AND shape_stretched_120s | 20 | +52.65 | +783.97 | 16 | +53.16 | +27.84 | +445.47 | +267.06 | 6/9 |
| LONG_DELAY0_TP60_500K_DAYTREND | 1 | liq_count_ge_p75_22 | 27 | +52.59 | +851.52 | 19 | +54.59 | +52.18 | +991.34 | +809.32 | 9/9 |
| LONG_DELAY0_TP60_500K_DAYTREND | 2 | liq_count_ge_p75_22 AND shape_stretched_120s | 19 | +52.66 | +767.35 | 15 | +54.59 | +51.49 | +772.36 | +590.34 | 7/7 |
| LONG_DELAY0_TP60_500K_DAYTREND | 3 | duration_ge_p75_221s | 18 | +52.62 | +709.30 | 15 | +54.26 | +40.49 | +607.29 | +424.76 | 7/9 |
| LONG_DELAY0_TP60_500K_DAYTREND | 4 | max_share_lt_50 | 16 | +52.56 | +708.17 | 32 | +52.95 | +37.43 | +1197.72 | +1016.66 | 11/11 |
| LONG_DELAY0_TP60_500K_DAYTREND | 5 | shape_stretched_120s | 25 | +52.53 | +728.95 | 30 | +52.95 | +37.73 | +1131.93 | +949.40 | 10/10 |

## 3. Real-Fill Parity For Top Candidates

| Scope | Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60_ALL_200K | liq_count_ge_p75_22 AND duration_ge_p75_221s | 49 | 10 | 39 (79.6%) | 7 | +59.26 | +51.87 | +363.07 | +156.52 | 5/5 |
| LONG_DELAY0_TP60_ALL_200K | liq_count_ge_p75_22 | 121 | 27 | 94 (77.7%) | 17 | +55.22 | +53.59 | +910.98 | +697.66 | 8/8 |
| LONG_DELAY0_TP60_ALL_200K | max_share_lt_50 AND liq_count_ge_p75_22 | 71 | 20 | 51 (71.8%) | 13 | +56.27 | +52.93 | +688.03 | +487.40 | 7/7 |
| LONG_DELAY0_TP60_ALL_200K | liq_count_ge_p75_22 AND shape_stretched_120s | 98 | 22 | 76 (77.6%) | 14 | +55.70 | +53.65 | +751.10 | +537.78 | 7/7 |
| LONG_DELAY0_TP60_ALL_200K | gap_lt_30m AND shape_stretched_120s | 36 | 15 | 21 (58.3%) | 12 | +49.01 | +38.91 | +466.97 | +261.07 | 5/6 |
| LONG_DELAY0_TP60_500K_DAYTREND | liq_count_ge_p75_22 | 46 | 19 | 27 (58.7%) | 14 | +57.24 | +53.41 | +747.69 | +534.37 | 7/7 |
| LONG_DELAY0_TP60_500K_DAYTREND | liq_count_ge_p75_22 AND shape_stretched_120s | 34 | 15 | 19 (55.9%) | 11 | +59.26 | +53.44 | +587.81 | +374.50 | 5/5 |
| LONG_DELAY0_TP60_500K_DAYTREND | duration_ge_p75_221s | 33 | 17 | 16 (48.5%) | 13 | +52.16 | +43.49 | +565.37 | +311.53 | 7/8 |
| LONG_DELAY0_TP60_500K_DAYTREND | max_share_lt_50 | 48 | 29 | 19 (39.6%) | 25 | +49.65 | +36.75 | +918.68 | +718.05 | 9/9 |
| LONG_DELAY0_TP60_500K_DAYTREND | shape_stretched_120s | 55 | 30 | 25 (45.5%) | 25 | +50.17 | +36.95 | +923.80 | +669.51 | 7/8 |

## Read

These are geometry-only retrospective filters selected from the same feature surface. Treat them as hypothesis seeds unless they survive a separately pre-registered forward test.
