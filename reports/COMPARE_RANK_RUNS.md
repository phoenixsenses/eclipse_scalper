# COMPARE_RANK_RUNS

## RANK_EDGEONLY_FEE0_BUY_21D.json

rows_total=7 top_n=5

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.537155e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.879321e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -2.967889e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -3.364785e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -3.706349e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |

Diagnosis
- dominant_failure_reason_top=adverse_dominates (100.00%)
- top5_mean_npa_core=-2.691100e-05
- top5_mean_pass_rate_core=50.00%

## RANK_EDGEONLY_FEE0_SELL_21D.json

rows_total=7 top_n=5

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -4.740326e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -6.121203e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -7.229492e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -8.091459e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -4.268792e-05 | 44.44% | adverse_dominates | 0.00 | 0.00% | 0.00% | - | - | - |

Diagnosis
- dominant_failure_reason_top=adverse_dominates (100.00%)
- top5_mean_npa_core=-6.090254e-05
- top5_mean_pass_rate_core=48.89%

## RANK_EDGEONLY_FEE0_AUTO_21D.json

rows_total=7 top_n=5

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +8.407132e-06 | 50.00% | adverse_dominates | 0.00 | 0.00% | 53.74% | 0.000 | 0.250 | -0.166 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +7.297650e-06 | 50.00% | adverse_dominates | 0.00 | 0.00% | 60.07% | 0.000 | 0.267 | -0.059 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +1.449272e-06 | 50.00% | mixed | 0.00 | 0.00% | 54.97% | 0.000 | 0.252 | 0.043 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +6.915552e-07 | 50.00% | adverse_dominates | 0.00 | 0.00% | 59.21% | 0.000 | 0.265 | -0.330 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -1.518898e-05 | 50.00% | adverse_dominates | 0.00 | 0.00% | 55.76% | 0.000 | 0.252 | -0.408 |

Diagnosis
- dominant_failure_reason_top=adverse_dominates (80.00%)
- top5_mean_npa_core=+5.313255e-07
- top5_mean_pass_rate_core=50.00%

## RANK_V3_ATTR_21D.json

rows_total=7 top_n=5

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.067118e-04 | 27.78% | fees_dominate | 1.00 | 2.98% | 55.42% | 2.000 | 0.253 | -1.922 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -1.125259e-04 | 16.67% | fees_dominate | 1.00 | 3.33% | 56.06% | 2.000 | 0.255 | -1.973 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.079512e-04 | 16.67% | fees_dominate | 1.00 | 5.42% | 63.80% | 2.000 | 0.348 | -2.291 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.316257e-04 | 5.56% | fees_dominate | 1.00 | 2.82% | 60.59% | 2.000 | 0.272 | -2.118 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -1.307385e-04 | 0.00% | fees_dominate | 0.50 | 3.81% | 54.93% | 2.000 | 0.258 | -2.349 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top5_mean_npa_core=-1.179106e-04
- top5_mean_pass_rate_core=13.33%

## Cross-Run Diagnosis

- BUY/SELL delta (top-5 mean): delta_npa_core=+3.399155e-05, delta_pass_rate_core=+1.11%

