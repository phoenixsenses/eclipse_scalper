# PASSIVE_POCKET_RANKING

candidates=5 ranked=5
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=3 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=11 table_rows_seen=7 rows_with_pass_yes=5 candidates_parsed=5 candidates_unique=5 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.095784e-04 | -1.119188e-04 | 50.81% | 0.17 | -0.00022297 | -0.00022773 | -0.00022773 | 0.00% | 0.00% | -1.095784e-04 | -0.00022297 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 50.32% | 2.000 | 0.239 | -0.026 | -2.265 | 0.230 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.359091e-04 | -1.390328e-04 | 48.26% | 0.14 | -0.00028215 | -0.00028776 | -0.00028776 | 0.00% | 0.00% | -1.359091e-04 | -0.00028215 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.27% | 2.000 | 0.292 | -0.489 | -2.782 | 0.693 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.514283e-04 | -1.568879e-04 | 59.48% | 0.11 | -0.00024777 | -0.00025670 | -0.00025670 | 0.00% | 0.00% | -1.514283e-04 | -0.00024777 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 59.18% | 2.000 | 0.455 | -0.411 | -2.866 | 0.989 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.574660e-04 | -1.616298e-04 | 53.31% | 0.22 | -0.00029687 | -0.00030501 | -0.00030501 | 0.00% | 0.00% | -1.574660e-04 | -0.00029687 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 53.68% | 2.000 | 0.398 | -0.452 | -2.850 | 0.496 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.613155e-04 | -1.662553e-04 | 60.56% | 0.16 | -0.00027785 | -0.00028588 | -0.00028588 | 0.00% | 0.00% | -1.613155e-04 | -0.00027785 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 60.74% | 2.000 | 0.406 | -0.370 | -2.776 | 0.529 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -1.290766e-06 | +1.006342e-04 | +1.204014e-05 | +0.000000e+00 | -1.139652e-04 | -1.095784e-04 | +4.386721e-06 | 0.00% | 2838 | 2838 | 1428 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -2.410650e-05 | +9.854701e-05 | +1.441217e-05 | +0.000000e+00 | -1.370657e-04 | -1.359091e-04 | +1.156588e-06 | 0.00% | 2340 | 2340 | 1153 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -2.434016e-05 | +1.183673e-04 | +2.690699e-05 | +0.000000e+00 | -1.696145e-04 | -1.514283e-04 | +1.818618e-05 | 0.00% | 1766 | 1764 | 1044 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.428123e-05 | +1.073514e-04 | +2.136489e-05 | +0.000000e+00 | -1.529975e-04 | -1.574660e-04 | -4.468503e-06 | 0.00% | 3700 | 3700 | 1986 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -2.244686e-05 | +1.214858e-04 | +2.469080e-05 | +0.000000e+00 | -1.686235e-04 | -1.613155e-04 | +7.307937e-06 | 0.00% | 2746 | 2746 | 1668 |
