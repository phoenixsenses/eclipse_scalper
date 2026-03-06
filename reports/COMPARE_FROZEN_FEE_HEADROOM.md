# COMPARE_RANK_RUNS

intersect_only=true intersection_count=1

## FROZEN_fee0p5_h120_v3.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +2.155792e-05 | 61.11% | gate_reject | 0.50 | 67.18% | 59.92% | 1.000 | 0.328 | 0.242 |

Diagnosis
- dominant_failure_reason_top=gate_reject (100.00%)
- top50_mean_npa_core=+2.155792e-05
- top50_mean_pass_rate_core=61.11%

## FROZEN_fee0p7_h120_v3.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -4.481449e-05 | 38.89% | gate_reject | 0.70 | 67.31% | 61.48% | 1.400 | 0.330 | -0.663 |

Diagnosis
- dominant_failure_reason_top=gate_reject (100.00%)
- top50_mean_npa_core=-4.481449e-05
- top50_mean_pass_rate_core=38.89%

## FROZEN_fee0p8_h120_v3.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -3.587434e-05 | 33.33% | gate_reject | 0.80 | 67.25% | 59.79% | 1.600 | 0.328 | -0.371 |

Diagnosis
- dominant_failure_reason_top=gate_reject (100.00%)
- top50_mean_npa_core=-3.587434e-05
- top50_mean_pass_rate_core=33.33%

## FROZEN_fee0p9_h120_v3.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.224574e-04 | 33.33% | gate_reject | 0.90 | 67.19% | 61.25% | 1.800 | 0.327 | -1.216 |

Diagnosis
- dominant_failure_reason_top=gate_reject (100.00%)
- top50_mean_npa_core=-1.224574e-04
- top50_mean_pass_rate_core=33.33%

## FROZEN_fee1_h120_v3.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -5.154622e-05 | 27.78% | gate_reject | 1.00 | 67.09% | 60.37% | 2.000 | 0.329 | -1.155 |

Diagnosis
- dominant_failure_reason_top=gate_reject (100.00%)
- top50_mean_npa_core=-5.154622e-05
- top50_mean_pass_rate_core=27.78%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

