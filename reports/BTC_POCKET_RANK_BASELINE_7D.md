# PASSIVE_POCKET_RANKING

candidates=8 ranked=8
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=3 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=14 table_rows_seen=10 rows_with_pass_yes=8 candidates_parsed=8 candidates_unique=8 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -8.315828e-05 | -8.464830e-05 | 42.11% | 0.41 | -0.00019963 | -0.00020273 | -0.00020273 | 0.00% | 0.00% | -8.315828e-05 | -0.00019963 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 42.36% | 2.000 | 0.156 | +0.093 | -2.063 | 0.160 | 0.00 | 0.00% |
| 2 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -9.751892e-05 | -9.909909e-05 | 43.73% | 0.38 | -0.00021830 | -0.00022142 | -0.00022142 | 0.00% | 0.00% | -9.751892e-05 | -0.00021830 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 43.89% | 2.000 | 0.156 | -0.023 | -2.179 | 0.213 | 0.00 | 0.00% |
| 3 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -9.999659e-05 | -1.016080e-04 | 46.22% | 0.46 | -0.00021749 | -0.00022120 | -0.00022120 | 0.00% | 0.00% | -9.999659e-05 | -0.00021749 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 46.37% | 2.000 | 0.197 | -0.069 | -2.266 | 0.217 | 0.00 | 0.00% |
| 4 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.083501e-04 | -1.102904e-04 | 49.30% | 0.41 | -0.00023508 | -0.00023875 | -0.00023875 | 0.00% | 0.00% | -1.083501e-04 | -0.00023508 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 48.70% | 2.000 | 0.199 | -0.196 | -2.395 | 0.314 | 0.00 | 0.00% |
| 5 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.136546e-04 | -1.153028e-04 | 47.23% | 0.39 | -0.00024195 | -0.00024582 | -0.00024582 | 0.00% | 0.00% | -1.136546e-04 | -0.00024195 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 47.21% | 2.000 | 0.179 | -0.197 | -2.376 | 0.248 | 0.00 | 0.00% |
| 6 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.30 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.194236e-04 | -1.217866e-04 | 49.65% | 0.42 | -0.00023603 | -0.00024030 | -0.00024030 | 0.00% | 0.00% | -1.194236e-04 | -0.00023603 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.63% | 2.000 | 0.213 | -0.175 | -2.389 | 0.208 | 0.00 | 0.00% |
| 7 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 8000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.213057e-04 | -1.238239e-04 | 52.82% | 0.17 | -0.00021984 | -0.00022410 | -0.00022410 | 0.00% | 0.00% | -1.213057e-04 | -0.00021984 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 53.01% | 2.000 | 0.215 | -0.299 | -2.514 | 0.593 | 0.00 | 0.00% |
| 8 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 6000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.257641e-04 | -1.273450e-04 | 49.54% | 0.33 | -0.00025828 | -0.00026179 | -0.00026179 | 0.00% | 0.00% | -1.257641e-04 | -0.00025828 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.89% | 2.000 | 0.177 | -0.353 | -2.530 | 0.141 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BTCUSDT | intensity_spike_imbalance_cont | 120 | +3.941532e-06 | +8.472427e-05 | +6.605587e-06 | +0.000000e+00 | -8.738833e-05 | -8.315828e-05 | +4.230050e-06 | 0.00% | 6782 | 6782 | 2873 |
| 2 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.017636e-06 | +8.778125e-05 | +6.847415e-06 | +0.000000e+00 | -9.564630e-05 | -9.751892e-05 | -1.872619e-06 | 0.00% | 6400 | 6400 | 2809 |
| 3 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -3.196602e-06 | +9.273267e-05 | +9.128552e-06 | +0.000000e+00 | -1.050578e-04 | -9.999659e-05 | +5.061230e-06 | 0.00% | 7747 | 7747 | 3592 |
| 4 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -9.541283e-06 | +9.739659e-05 | +9.685704e-06 | +0.000000e+00 | -1.166236e-04 | -1.083501e-04 | +8.273427e-06 | 0.00% | 6914 | 6914 | 3367 |
| 5 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -9.305779e-06 | +9.441281e-05 | +8.447526e-06 | +0.000000e+00 | -1.121661e-04 | -1.136546e-04 | -1.488445e-06 | 0.00% | 6497 | 6497 | 3067 |
| 6 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -8.691797e-06 | +9.926091e-05 | +1.059311e-05 | +0.000000e+00 | -1.185458e-04 | -1.194236e-04 | -8.777447e-07 | 0.00% | 7171 | 7171 | 3559 |
| 7 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.586689e-05 | +1.060109e-04 | +1.139090e-05 | +0.000000e+00 | -1.332687e-04 | -1.213057e-04 | +1.196298e-05 | 0.00% | 2928 | 2928 | 1552 |
| 8 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.761653e-05 | +9.978086e-05 | +8.811334e-06 | +0.000000e+00 | -1.262087e-04 | -1.257641e-04 | +4.445912e-07 | 0.00% | 5476 | 5476 | 2732 |
