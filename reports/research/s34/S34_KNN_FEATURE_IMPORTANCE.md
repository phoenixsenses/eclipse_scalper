# S34 KNN Feature Importance

- Events: 69
- K: 5
- Note: Temporal-safe diagnostic: each event only sees same-rule outcomes closed before its signal timestamp.

## Baseline

| N | MAE bps | Bias bps | Direction hit |
|---:|---:|---:|---:|
| 62 | 47.53 | 1.02 | 61.3% |

## Drop-One

| Removed feature | N | MAE bps | MAE delta | Direction hit | Direction delta |
|---|---:|---:|---:|---:|---:|
| log_cluster_notional | 62 | 44.79 | -2.74 | 64.5% | 3.2% |
| cluster_liq_count_ratio | 62 | 46.74 | -0.79 | 59.7% | -1.6% |
| shape_match | 62 | 47.46 | -0.07 | 61.3% | 0.0% |
| cluster_duration_sec | 62 | 47.64 | 0.11 | 61.3% | 0.0% |
| max_single_liq_share | 62 | 47.64 | 0.11 | 61.3% | 0.0% |
| btc_pre_return_bps | 62 | 47.53 | 0.00 | 61.3% | 0.0% |
