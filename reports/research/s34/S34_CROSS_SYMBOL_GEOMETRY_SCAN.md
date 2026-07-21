# S34 Cross-Symbol Geometry Scan

Generated: `2026-06-20T08:55:38.887558+00:00`

Scope: geometry filters for current/new BUY-liq continuation candidates. Research-only; live runner/config unchanged.

## Baselines

| Scope | Rows | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days | Exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SOL_200K_TP60_SL40_BE30 | 37 | 37 | 0 (0.0%) | 18 | +47.46 | +17.85 | +321.25 | +153.03 | 5/5 | {'BE': 5, 'SL': 3, 'TP': 10} |
| BTC_1M_TP60_SL40_BE30 | 68 | 67 | 1 (1.5%) | 33 | +28.55 | +16.10 | +531.32 | +341.64 | 8/12 | {'BE': 9, 'SL': 5, 'TIME': 5, 'TP': 14} |
| BTC_1M_TP60_SL30_BE30 | 68 | 67 | 1 (1.5%) | 33 | +28.55 | +17.69 | +583.84 | +394.16 | 8/12 | {'BE': 9, 'SL': 5, 'TIME': 5, 'TP': 14} |

## Top Geometry Candidates

| Scope | Rank | Candidate | Train N | Train Median | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| SOL_200K_TP60_SL40_BE30 | 1 | gap_ge_2h | 15 | +52.35 | 10 | +52.95 | +25.12 | +251.21 | +86.53 |
| SOL_200K_TP60_SL40_BE30 | 2 | max_share_lt_50 | 5 | +55.02 | 10 | +52.79 | +31.03 | +310.30 | +146.43 |
| SOL_200K_TP60_SL40_BE30 | 3 | shape_stretched_120s | 5 | +53.33 | 8 | +52.59 | +30.21 | +241.71 | +81.18 |
| SOL_200K_TP60_SL40_BE30 | 4 | liq_count_le_p25_11 | 5 | +24.58 | 5 | -8.00 | -0.02 | -0.12 | -99.45 |
| BTC_1M_TP60_SL40_BE30 | 1 | max_share_lt_50 AND intensity_le_p25_7133 | 5 | +53.45 | 5 | +52.97 | +28.85 | +144.24 | -16.00 |
| BTC_1M_TP60_SL40_BE30 | 2 | shape_distributed_mid_duration | 11 | +52.06 | 5 | +52.80 | +22.85 | +114.25 | -51.69 |
| BTC_1M_TP60_SL40_BE30 | 3 | max_share_lt_50 | 21 | +22.21 | 21 | +52.25 | +31.57 | +663.02 | +492.16 |
| BTC_1M_TP60_SL40_BE30 | 4 | max_share_lt_50 AND gap_ge_2h | 10 | +37.26 | 13 | +52.25 | +30.25 | +393.25 | +224.63 |
| BTC_1M_TP60_SL40_BE30 | 5 | liq_count_ge_p75_33 | 8 | +52.26 | 10 | +41.98 | +27.04 | +270.44 | +104.21 |
| BTC_1M_TP60_SL30_BE30 | 1 | max_share_lt_50 AND intensity_le_p25_7133 | 5 | +53.45 | 5 | +52.97 | +28.85 | +144.24 | -16.00 |
| BTC_1M_TP60_SL30_BE30 | 2 | shape_distributed_mid_duration | 11 | +52.06 | 5 | +52.80 | +24.90 | +124.51 | -41.42 |
| BTC_1M_TP60_SL30_BE30 | 3 | max_share_lt_50 | 21 | +22.21 | 21 | +52.25 | +31.57 | +663.02 | +492.16 |
| BTC_1M_TP60_SL30_BE30 | 4 | max_share_lt_50 AND gap_ge_2h | 10 | +37.26 | 13 | +52.25 | +30.25 | +393.25 | +224.63 |
| BTC_1M_TP60_SL30_BE30 | 5 | liq_count_ge_p75_33 | 8 | +52.26 | 10 | +41.98 | +27.04 | +270.44 | +104.21 |

## Real-Fill Parity For Top Geometry Candidates

| Scope | Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOL_200K_TP60_SL40_BE30 | gap_ge_2h | 25 | 25 | 0 (0.0%) | 10 | +47.46 | +23.59 | +235.88 | +68.07 | 5/5 |
| SOL_200K_TP60_SL40_BE30 | max_share_lt_50 | 15 | 15 | 0 (0.0%) | 10 | +51.72 | +30.28 | +302.82 | +135.01 | 5/5 |
| SOL_200K_TP60_SL40_BE30 | shape_stretched_120s | 13 | 13 | 0 (0.0%) | 8 | +50.88 | +28.99 | +231.94 | +68.08 | 3/3 |
| SOL_200K_TP60_SL40_BE30 | liq_count_le_p25_11 | 10 | 10 | 0 (0.0%) | 5 | -9.39 | -3.20 | -15.98 | -101.52 | 1/3 |
| BTC_1M_TP60_SL40_BE30 | max_share_lt_50 AND intensity_le_p25_7133 | 10 | 10 | 0 (0.0%) | 5 | +52.34 | +30.53 | +152.67 | -10.26 | 3/5 |
| BTC_1M_TP60_SL40_BE30 | shape_distributed_mid_duration | 16 | 15 | 1 (6.2%) | 5 | +54.55 | +24.53 | +122.65 | -57.89 | 3/5 |
| BTC_1M_TP60_SL40_BE30 | max_share_lt_50 | 42 | 42 | 0 (0.0%) | 21 | +47.27 | +30.94 | +649.84 | +469.04 | 8/12 |
| BTC_1M_TP60_SL40_BE30 | max_share_lt_50 AND gap_ge_2h | 23 | 23 | 0 (0.0%) | 13 | +47.27 | +29.69 | +385.99 | +209.19 | 8/12 |
| BTC_1M_TP60_SL40_BE30 | liq_count_ge_p75_33 | 18 | 18 | 0 (0.0%) | 10 | +37.98 | +26.33 | +263.33 | +94.91 | 4/6 |
| BTC_1M_TP60_SL30_BE30 | max_share_lt_50 AND intensity_le_p25_7133 | 10 | 10 | 0 (0.0%) | 5 | +52.34 | +30.53 | +152.67 | -10.26 | 3/5 |
| BTC_1M_TP60_SL30_BE30 | shape_distributed_mid_duration | 16 | 15 | 1 (6.2%) | 5 | +54.55 | +26.44 | +132.18 | -48.36 | 3/5 |
| BTC_1M_TP60_SL30_BE30 | max_share_lt_50 | 42 | 42 | 0 (0.0%) | 21 | +47.27 | +30.94 | +649.84 | +469.04 | 8/12 |
| BTC_1M_TP60_SL30_BE30 | max_share_lt_50 AND gap_ge_2h | 23 | 23 | 0 (0.0%) | 13 | +47.27 | +29.69 | +385.99 | +209.19 | 8/12 |
| BTC_1M_TP60_SL30_BE30 | liq_count_ge_p75_33 | 18 | 18 | 0 (0.0%) | 10 | +37.98 | +26.33 | +263.33 | +94.91 | 4/6 |

## Read

These filters are retrospective geometry screens. Use them to decide whether a narrower exploratory variant deserves pre-registration; do not mutate existing live rules directly from this report.
