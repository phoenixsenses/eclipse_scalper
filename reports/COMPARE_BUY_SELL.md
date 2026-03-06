# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_BUY.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +4.538812e-05 | 77.78% | mixed | 0.00 | 2.42% | 56.05% | 0.000 | 0.256 | 0.791 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +4.528485e-05 | 83.33% | mixed | 0.00 | 2.88% | 57.05% | 0.000 | 0.256 | 0.865 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +4.685324e-05 | 61.11% | mixed | 0.00 | 4.22% | 62.70% | 0.000 | 0.349 | 0.406 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +3.655886e-05 | 77.78% | mixed | 0.00 | 2.23% | 60.76% | 0.000 | 0.271 | 0.645 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +2.474410e-05 | 66.67% | mixed | 0.00 | 2.60% | 61.40% | 0.000 | 0.271 | 0.520 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +5.797911e-07 | 50.00% | mixed | 0.00 | 3.55% | 55.57% | 0.000 | 0.260 | 0.277 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -9.906162e-07 | 44.44% | mixed | 0.00 | 3.34% | 57.79% | 0.000 | 0.254 | 0.273 |

Diagnosis
- dominant_failure_reason_top=mixed (100.00%)
- top10_mean_npa_core=+2.834548e-05
- top10_mean_pass_rate_core=65.87%

## PASSIVE_POCKET_RANKING_SELL.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +2.620285e-05 | 66.67% | mixed | 0.00 | 2.93% | 56.77% | 0.000 | 0.255 | 0.844 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +2.171111e-05 | 72.22% | mixed | 0.00 | 2.46% | 56.04% | 0.000 | 0.255 | 0.687 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +2.152102e-05 | 66.67% | mixed | 0.00 | 2.70% | 60.82% | 0.000 | 0.270 | 0.480 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +2.632851e-05 | 72.22% | mixed | 0.00 | 4.31% | 63.48% | 0.000 | 0.348 | 0.453 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.240773e-05 | 66.67% | mixed | 0.00 | 2.27% | 60.33% | 0.000 | 0.271 | 0.398 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +5.377276e-06 | 50.00% | mixed | 0.00 | 3.62% | 55.83% | 0.000 | 0.259 | 0.364 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +4.938594e-06 | 50.00% | mixed | 0.00 | 3.42% | 57.59% | 0.000 | 0.254 | 0.402 |

Diagnosis
- dominant_failure_reason_top=mixed (100.00%)
- top10_mean_npa_core=+1.835530e-05
- top10_mean_pass_rate_core=63.49%

## Cross-Run Diagnosis

- BUY/SELL delta (top-10 mean): delta_npa_core=+9.990178e-06, delta_pass_rate_core=+2.38%

