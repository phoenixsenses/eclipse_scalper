# COMPARE_RANK_RUNS

intersect_only=true intersection_count=1

## test_cmp_21e2264b_A.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +1.000000e-04 | 60.00% | fees_dominate | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top5_mean_npa_core=+1.000000e-04
- top5_mean_pass_rate_core=60.00%

## test_cmp_21e2264b_B.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +1.000000e-04 | 60.00% | fees_dominate | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top5_mean_npa_core=+1.000000e-04
- top5_mean_pass_rate_core=60.00%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

