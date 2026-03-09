# PASSIVE_POCKET_RANKING

candidates=5 ranked=5
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=3 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=11 table_rows_seen=7 rows_with_pass_yes=5 candidates_parsed=5 candidates_unique=5 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=auto gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.123529e-04 | -1.147030e-04 | 50.49% | 0.17 | -0.00023003 | -0.00023490 | -0.00023490 | 0.00% | 0.00% | -1.123529e-04 | -0.00023003 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 50.26% | 2.000 | 0.244 | -0.075 | -2.319 | 0.423 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.140204e-04 | -1.196717e-04 | 59.55% | 0.10 | -0.00019282 | -0.00020180 | -0.00020180 | 0.00% | 0.00% | -1.140204e-04 | -0.00019282 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 60.22% | 2.000 | 0.449 | +0.295 | -2.155 | 0.975 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.369971e-04 | -1.400116e-04 | 49.40% | 0.13 | -0.00028337 | -0.00028911 | -0.00028911 | 0.00% | 0.00% | -1.369971e-04 | -0.00028337 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 50.20% | 2.000 | 0.288 | -0.623 | -2.911 | 0.797 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.543830e-04 | -1.587793e-04 | 54.76% | 0.21 | -0.00028385 | -0.00029193 | -0.00029193 | 0.00% | 0.00% | -1.543830e-04 | -0.00028385 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 54.69% | 2.000 | 0.398 | -0.381 | -2.779 | 0.444 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.021440e-04 | -2.074623e-04 | 61.58% | 0.16 | -0.00031597 | -0.00032427 | -0.00032427 | 0.00% | 0.00% | -2.021440e-04 | -0.00031597 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.88% | 2.000 | 0.408 | -0.835 | -3.243 | 0.603 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -3.792053e-06 | +1.005254e-04 | +1.224000e-05 | +0.000000e+00 | -1.165574e-04 | -1.123529e-04 | +4.204546e-06 | 0.00% | 2855 | 2855 | 1435 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | +1.773585e-05 | +1.204388e-04 | +2.706289e-05 | +0.000000e+00 | -1.297658e-04 | -1.140204e-04 | +1.574547e-05 | 0.00% | 1732 | 1732 | 1043 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.125688e-05 | +1.003967e-04 | +1.447222e-05 | +0.000000e+00 | -1.461258e-04 | -1.369971e-04 | +9.128640e-06 | 0.00% | 2269 | 2269 | 1139 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.084052e-05 | +1.093811e-04 | +2.178245e-05 | +0.000000e+00 | -1.520041e-04 | -1.543830e-04 | -2.378922e-06 | 0.00% | 3571 | 3571 | 1953 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -5.168268e-05 | +1.237631e-04 | +2.524699e-05 | +0.000000e+00 | -2.006928e-04 | -2.021440e-04 | -1.451263e-06 | 0.00% | 2668 | 2668 | 1651 |
