# PASSIVE_POCKET_RANKING

candidates=8 ranked=8
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=5 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=14 table_rows_seen=10 rows_with_pass_yes=8 candidates_parsed=8 candidates_unique=8 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -9.676898e-05 | -9.794785e-05 | 41.45% | 0.48 | -0.00023475 | -0.00023769 | -0.00023769 | 0.00% | 0.00% | -9.676898e-05 | -0.00023475 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 41.56% | 2.000 | 0.142 | -0.261 | -2.403 | 0.361 | 0.00 | 0.00% |
| 2 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.034342e-04 | -1.045792e-04 | 42.64% | 0.45 | -0.00024697 | -0.00024962 | -0.00024962 | 0.00% | 0.00% | -1.034342e-04 | -0.00024697 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 42.73% | 2.000 | 0.139 | -0.207 | -2.350 | 0.382 | 0.00 | 0.00% |
| 3 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.036759e-04 | -1.052429e-04 | 45.40% | 0.53 | -0.00023407 | -0.00023766 | -0.00023766 | 0.00% | 0.00% | -1.036759e-04 | -0.00023407 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 45.14% | 2.000 | 0.187 | -0.028 | -2.208 | 0.293 | 0.00 | 0.00% |
| 4 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.077920e-04 | -1.093458e-04 | 46.60% | 0.46 | -0.00022864 | -0.00023201 | -0.00023201 | 0.00% | 0.00% | -1.077920e-04 | -0.00022864 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 46.36% | 2.000 | 0.163 | -0.106 | -2.279 | 0.305 | 0.00 | 0.00% |
| 5 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.30 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.164940e-04 | -1.183569e-04 | 48.61% | 0.48 | -0.00023929 | -0.00024326 | -0.00024326 | 0.00% | 0.00% | -1.164940e-04 | -0.00023929 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 48.58% | 2.000 | 0.199 | -0.085 | -2.277 | 0.297 | 0.00 | 0.00% |
| 6 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.175915e-04 | -1.194510e-04 | 48.19% | 0.47 | -0.00024502 | -0.00024883 | -0.00024883 | 0.00% | 0.00% | -1.175915e-04 | -0.00024502 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 48.14% | 2.000 | 0.185 | -0.235 | -2.420 | 0.290 | 0.00 | 0.00% |
| 7 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 6000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.219976e-04 | -1.235413e-04 | 49.37% | 0.37 | -0.00024778 | -0.00025099 | -0.00025099 | 0.00% | 0.00% | -1.219976e-04 | -0.00024778 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.15% | 2.000 | 0.161 | -0.252 | -2.414 | 0.421 | 0.00 | 0.00% |
| 8 | BTCUSDT | intensity_spike_imbalance_cont | 120 | 0.70 | 8000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.220764e-04 | -1.240853e-04 | 52.07% | 0.20 | -0.00023744 | -0.00024130 | -0.00024130 | 0.00% | 0.00% | -1.220764e-04 | -0.00023744 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 51.17% | 2.000 | 0.201 | -0.227 | -2.436 | 0.410 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.085348e-05 | +8.311715e-05 | +5.906982e-06 | +0.000000e+00 | -9.987761e-05 | -9.676898e-05 | +3.108629e-06 | 0.00% | 9715 | 9714 | 4037 |
| 2 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -8.864915e-06 | +8.546284e-05 | +5.950713e-06 | +0.000000e+00 | -1.002785e-04 | -1.034342e-04 | -3.155726e-06 | 0.00% | 9205 | 9204 | 3933 |
| 3 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.261815e-06 | +9.027661e-05 | +8.435163e-06 | +0.000000e+00 | -9.997358e-05 | -1.036759e-04 | -3.702292e-06 | 0.00% | 10666 | 10665 | 4814 |
| 4 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -4.898505e-06 | +9.271233e-05 | +7.562493e-06 | +0.000000e+00 | -1.051733e-04 | -1.077920e-04 | -2.618682e-06 | 0.00% | 9126 | 9125 | 4230 |
| 5 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -4.115997e-06 | +9.715741e-05 | +9.645703e-06 | +0.000000e+00 | -1.109191e-04 | -1.164940e-04 | -5.574864e-06 | 0.00% | 9816 | 9815 | 4768 |
| 6 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.130288e-05 | +9.627198e-05 | +8.891050e-06 | +0.000000e+00 | -1.164659e-04 | -1.175915e-04 | -1.125544e-06 | 0.00% | 9443 | 9442 | 4545 |
| 7 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.239999e-05 | +9.830777e-05 | +7.916865e-06 | +0.000000e+00 | -1.186246e-04 | -1.219976e-04 | -3.372988e-06 | 0.00% | 7565 | 7564 | 3718 |
| 8 | BTCUSDT | intensity_spike_imbalance_cont | 120 | -1.160418e-05 | +1.023455e-04 | +1.029991e-05 | +0.000000e+00 | -1.242496e-04 | -1.220764e-04 | +2.173213e-06 | 0.00% | 3966 | 3965 | 2029 |
