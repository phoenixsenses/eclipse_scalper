# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_BUY_fee1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -6.144945e-05 | 11.11% | fees_dominate | 1.00 | 3.49% | 60.38% | 2.000 | 0.342 | -1.336 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -7.702550e-05 | 11.11% | fees_dominate | 1.00 | 2.11% | 55.43% | 2.000 | 0.256 | -1.354 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -9.011828e-05 | 11.11% | fees_dominate | 1.00 | 2.56% | 57.01% | 2.000 | 0.256 | -1.411 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -8.275374e-05 | 5.56% | fees_dominate | 1.00 | 3.21% | 55.64% | 2.000 | 0.262 | -1.515 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -9.427038e-05 | 5.56% | fees_dominate | 1.00 | 2.40% | 61.46% | 2.000 | 0.271 | -1.600 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -8.136624e-05 | 0.00% | fees_dominate | 0.00 | 2.98% | 58.05% | 2.000 | 0.255 | -1.486 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.000202e-04 | 0.00% | fees_dominate | 0.00 | 1.92% | 60.13% | 2.000 | 0.273 | -1.484 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top10_mean_npa_core=-8.385769e-05
- top10_mean_pass_rate_core=6.35%

## PASSIVE_POCKET_RANKING_SELL_fee1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -6.612635e-05 | 5.56% | fees_dominate | 1.00 | 2.15% | 55.98% | 2.000 | 0.259 | -1.242 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -7.195610e-05 | 5.56% | fees_dominate | 1.00 | 1.99% | 60.78% | 2.000 | 0.277 | -1.275 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.024222e-04 | 5.56% | fees_dominate | 1.00 | 3.62% | 61.60% | 2.000 | 0.340 | -1.617 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -7.311500e-05 | 0.00% | fees_dominate | 0.00 | 2.63% | 57.02% | 2.000 | 0.257 | -1.425 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -8.051087e-05 | 0.00% | fees_dominate | 0.00 | 3.24% | 55.84% | 2.000 | 0.263 | -1.457 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -8.832004e-05 | 0.00% | fees_dominate | 0.00 | 3.00% | 58.21% | 2.000 | 0.257 | -1.477 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -9.533326e-05 | 0.00% | fees_dominate | 0.00 | 2.50% | 61.95% | 2.000 | 0.273 | -1.512 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top10_mean_npa_core=-8.254054e-05
- top10_mean_pass_rate_core=2.38%

## Cross-Run Diagnosis

- BUY/SELL delta (top-10 mean): delta_npa_core=-1.317142e-06, delta_pass_rate_core=+3.97%

