# COMPARE_RANK_RUNS

## RANK_ULTRATIGHT_fee0p5_adv1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +4.009657e-06 | 55.56% | gate_reject | 0.50 | 78.29% | 60.69% | 1.000 | 0.292 | 0.171 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +3.343760e-06 | 55.56% | gate_reject | 0.50 | 80.01% | 61.05% | 1.000 | 0.302 | 0.197 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +8.179368e-07 | 50.00% | gate_reject | 0.50 | 68.03% | 60.69% | 1.000 | 0.314 | 0.111 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -9.772946e-07 | 50.00% | gate_reject | 0.50 | 63.62% | 59.81% | 1.000 | 0.327 | 0.105 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -7.235986e-06 | 44.44% | gate_reject | 0.50 | 75.92% | 60.07% | 1.000 | 0.306 | 0.108 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.328312e-05 | 44.44% | gate_reject | 0.50 | 72.45% | 59.33% | 1.000 | 0.317 | 0.101 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -2.835857e-05 | 44.44% | mixed | 0.50 | 29.81% | 64.88% | 1.000 | 0.334 | 0.344 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-5.954802e-06
- top10_mean_pass_rate_core=49.21%

## RANK_ULTRATIGHT_fee0p7_adv1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +2.132723e-05 | 55.56% | gate_reject | 0.70 | 68.10% | 59.13% | 1.400 | 0.314 | -0.027 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +1.499495e-05 | 50.00% | fees_dominate | 0.70 | 29.81% | 64.59% | 1.400 | 0.333 | -0.001 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +1.186166e-05 | 55.56% | gate_reject | 0.70 | 75.93% | 58.97% | 1.400 | 0.305 | -0.243 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +1.102435e-05 | 55.56% | gate_reject | 0.70 | 63.70% | 58.44% | 1.400 | 0.327 | -0.121 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +5.504268e-06 | 50.00% | gate_reject | 0.70 | 78.24% | 59.45% | 1.400 | 0.292 | -0.254 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +4.476703e-06 | 55.56% | gate_reject | 0.70 | 72.44% | 57.96% | 1.400 | 0.317 | -0.308 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +2.917257e-06 | 50.00% | gate_reject | 0.70 | 80.00% | 59.50% | 1.400 | 0.302 | -0.248 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+1.030092e-05
- top10_mean_pass_rate_core=53.17%

## RANK_ULTRATIGHT_fee0p8_adv1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.229341e-06 | 44.44% | gate_reject | 0.80 | 63.45% | 60.56% | 1.600 | 0.336 | -0.077 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -2.823532e-06 | 44.44% | gate_reject | 0.80 | 67.88% | 61.33% | 1.600 | 0.323 | 0.000 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -5.756277e-06 | 44.44% | gate_reject | 0.80 | 78.17% | 61.16% | 1.600 | 0.299 | -0.107 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -8.071185e-06 | 44.44% | gate_reject | 0.80 | 75.82% | 60.73% | 1.600 | 0.314 | -0.109 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -9.016557e-06 | 44.44% | gate_reject | 0.80 | 72.37% | 60.34% | 1.600 | 0.326 | -0.129 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -1.018614e-05 | 38.89% | gate_reject | 0.80 | 79.85% | 61.41% | 1.600 | 0.307 | -0.210 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.390103e-05 | 38.89% | fees_dominate | 0.80 | 29.56% | 66.44% | 1.600 | 0.350 | -0.117 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-7.283436e-06
- top10_mean_pass_rate_core=42.86%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

