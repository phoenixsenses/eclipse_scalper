# COMPARE_RANK_RUNS

## RANK_ULTRATIGHT_h60_fee0p7.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -6.191185e-05 | 11.11% | gate_reject | 0.70 | 67.55% | 55.56% | 1.400 | 0.306 | -1.322 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -6.442869e-05 | 11.11% | gate_reject | 0.70 | 75.47% | 55.48% | 1.400 | 0.294 | -1.345 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -6.502115e-05 | 11.11% | gate_reject | 0.70 | 63.02% | 55.07% | 1.400 | 0.323 | -1.433 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -6.652392e-05 | 11.11% | gate_reject | 0.70 | 77.90% | 55.73% | 1.400 | 0.280 | -1.353 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -6.788506e-05 | 11.11% | gate_reject | 0.70 | 71.91% | 54.71% | 1.400 | 0.310 | -1.398 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -6.826844e-05 | 11.11% | fees_dominate | 0.70 | 29.71% | 62.28% | 1.400 | 0.338 | -1.269 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -7.019662e-05 | 11.11% | gate_reject | 0.70 | 79.67% | 56.42% | 1.400 | 0.289 | -1.464 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-6.631939e-05
- top10_mean_pass_rate_core=11.11%

## RANK_ULTRATIGHT_h120_fee0p7.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +1.386562e-05 | 50.00% | gate_reject | 0.70 | 63.04% | 60.07% | 1.400 | 0.338 | -0.223 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +7.939589e-06 | 55.56% | fees_dominate | 0.70 | 29.35% | 66.44% | 1.400 | 0.353 | -0.286 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +3.821375e-06 | 50.00% | gate_reject | 0.70 | 67.65% | 60.92% | 1.400 | 0.325 | -0.241 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -2.355324e-05 | 44.44% | gate_reject | 0.70 | 79.68% | 61.52% | 1.400 | 0.309 | -0.311 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -1.447091e-05 | 38.89% | gate_reject | 0.70 | 77.98% | 61.05% | 1.400 | 0.301 | -0.373 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -1.966150e-05 | 38.89% | gate_reject | 0.70 | 75.57% | 60.53% | 1.400 | 0.315 | -0.434 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -2.033961e-05 | 33.33% | gate_reject | 0.70 | 72.03% | 59.27% | 1.400 | 0.328 | -0.461 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-7.485526e-06
- top10_mean_pass_rate_core=44.44%

## RANK_ULTRATIGHT_h240_fee0p7.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.120215e-04 | 33.33% | fees_dominate | 0.70 | 28.22% | 68.47% | 1.400 | 0.347 | -1.709 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -7.836864e-05 | 22.22% | gate_reject | 0.70 | 79.58% | 64.55% | 1.400 | 0.310 | -1.549 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.059528e-04 | 22.22% | gate_reject | 0.70 | 67.42% | 63.50% | 1.400 | 0.322 | -1.662 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.063513e-04 | 22.22% | gate_reject | 0.70 | 62.68% | 63.07% | 1.400 | 0.334 | -1.723 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -1.116246e-04 | 22.22% | gate_reject | 0.70 | 77.86% | 64.35% | 1.400 | 0.302 | -1.634 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -1.155656e-04 | 22.22% | gate_reject | 0.70 | 75.43% | 63.91% | 1.400 | 0.314 | -1.614 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.178617e-04 | 22.22% | gate_reject | 0.70 | 71.80% | 63.36% | 1.400 | 0.327 | -1.693 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-1.068209e-04
- top10_mean_pass_rate_core=23.81%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

