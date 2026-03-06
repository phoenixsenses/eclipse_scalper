# COMPARE_RANK_RUNS

## test_cmp_e35c275b_RANK_EDGEONLY_FEE0_BUY_21D.json

rows_total=2 top_n=2

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +1.000000e-04 | 60.00% | fees_dominate | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |
| BTCUSDT | 120 | 0.50 | 2500 | 0.000300 | +5.000000e-05 | 55.00% | fees_dominate | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top2_mean_npa_core=+7.500000e-05
- top2_mean_pass_rate_core=57.50%

## test_cmp_e35c275b_RANK_EDGEONLY_FEE0_SELL_21D.json

rows_total=2 top_n=2

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -5.000000e-05 | 40.00% | adverse_dominates | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |
| BTCUSDT | 120 | 0.50 | 2500 | 0.000300 | -2.000000e-05 | 45.00% | adverse_dominates | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |

Diagnosis
- dominant_failure_reason_top=adverse_dominates (100.00%)
- top2_mean_npa_core=-3.500000e-05
- top2_mean_pass_rate_core=42.50%

## test_cmp_e35c275b_RANK_EDGEONLY_FEE0_AUTO_21D.json

rows_total=1 top_n=1

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +0.000000e+00 | 50.00% | mixed | 1.00 | 10.00% | 50.00% | 1.000 | 1.200 | -0.200 |

Diagnosis
- dominant_failure_reason_top=mixed (100.00%)
- top2_mean_npa_core=+0.000000e+00
- top2_mean_pass_rate_core=50.00%

## Cross-Run Diagnosis

- BUY/SELL delta (top-2 mean): delta_npa_core=+1.100000e-04, delta_pass_rate_core=+15.00%

