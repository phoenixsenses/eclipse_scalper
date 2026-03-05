# COMPARE_RANK_RUNS

## rank.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +4.682507e-05 | 61.11% | mixed | 0.00 | 2.83% | 56.97% | 0.000 | 0.253 | 0.540 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +4.122460e-05 | 61.11% | mixed | 0.00 | 2.52% | 56.23% | 0.000 | 0.252 | 0.500 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.773547e-05 | 55.56% | mixed | 0.00 | 2.27% | 61.29% | 0.000 | 0.269 | 0.248 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +2.515006e-05 | 55.56% | mixed | 0.00 | 4.33% | 62.84% | 0.000 | 0.345 | 0.052 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +1.340928e-05 | 50.00% | mixed | 0.00 | 2.49% | 61.62% | 0.000 | 0.268 | 0.258 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +1.041692e-05 | 50.00% | mixed | 0.00 | 3.39% | 58.15% | 0.000 | 0.252 | 0.160 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +9.492954e-06 | 50.00% | mixed | 0.00 | 3.46% | 55.87% | 0.000 | 0.258 | 0.176 |

Diagnosis
- dominant_failure_reason_top=mixed (100.00%)
- top10_mean_npa_core=+2.489348e-05
- top10_mean_pass_rate_core=54.76%

## rank.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +4.322357e-05 | 55.56% | mixed | 0.00 | 2.07% | 55.97% | 0.000 | 0.254 | 0.188 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +3.906215e-05 | 55.56% | mixed | 0.00 | 1.76% | 55.14% | 0.000 | 0.253 | 0.026 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.490058e-05 | 55.56% | adverse_dominates | 0.00 | 1.58% | 60.05% | 0.000 | 0.272 | -0.126 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +1.759471e-05 | 55.56% | adverse_dominates | 0.00 | 1.84% | 60.34% | 0.000 | 0.270 | -0.091 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +1.140558e-05 | 50.00% | adverse_dominates | 0.00 | 2.37% | 56.89% | 0.000 | 0.253 | -0.060 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +1.317169e-05 | 55.56% | adverse_dominates | 0.00 | 3.05% | 62.08% | 0.000 | 0.348 | -0.350 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +8.497977e-06 | 50.00% | adverse_dominates | 0.00 | 2.39% | 54.90% | 0.000 | 0.259 | -0.104 |

Diagnosis
- dominant_failure_reason_top=adverse_dominates (71.43%)
- top10_mean_npa_core=+2.255089e-05
- top10_mean_pass_rate_core=53.97%

## rank.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +3.733432e-05 | 61.11% | mixed | 0.00 | 2.18% | 56.47% | 0.000 | 0.203 | 0.339 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +3.963248e-05 | 61.11% | mixed | 0.00 | 1.92% | 55.03% | 0.000 | 0.200 | 0.183 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.929304e-05 | 55.56% | mixed | 0.00 | 1.78% | 61.20% | 0.000 | 0.215 | 0.100 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +1.128316e-05 | 50.00% | adverse_dominates | 0.00 | 2.57% | 57.40% | 0.000 | 0.202 | -0.153 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +8.579890e-06 | 50.00% | adverse_dominates | 0.00 | 2.58% | 55.34% | 0.000 | 0.207 | -0.116 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +8.272393e-06 | 50.00% | adverse_dominates | 0.00 | 3.44% | 61.91% | 0.000 | 0.279 | -0.058 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +3.705615e-07 | 50.00% | mixed | 0.00 | 2.02% | 62.07% | 0.000 | 0.216 | 0.117 |

Diagnosis
- dominant_failure_reason_top=mixed (57.14%)
- top10_mean_npa_core=+1.925226e-05
- top10_mean_pass_rate_core=53.97%

## rank.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +2.789179e-05 | 50.00% | mixed | 0.20 | 2.84% | 56.44% | 0.400 | 0.250 | 0.029 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +1.143438e-05 | 50.00% | fees_dominate | 0.20 | 2.53% | 55.49% | 0.400 | 0.250 | -0.231 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +7.313604e-07 | 50.00% | fees_dominate | 0.20 | 2.27% | 59.75% | 0.400 | 0.264 | -0.466 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -2.452259e-06 | 50.00% | fees_dominate | 0.20 | 3.42% | 57.18% | 0.400 | 0.250 | -0.377 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -1.344923e-05 | 50.00% | fees_dominate | 0.20 | 3.47% | 55.18% | 0.400 | 0.256 | -0.437 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.727858e-05 | 44.44% | fees_dominate | 0.20 | 2.54% | 60.37% | 0.400 | 0.265 | -0.365 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -3.696891e-05 | 33.33% | fees_dominate | 0.20 | 4.28% | 62.64% | 0.400 | 0.336 | -0.734 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (85.71%)
- top10_mean_npa_core=-4.298779e-06
- top10_mean_pass_rate_core=46.83%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

