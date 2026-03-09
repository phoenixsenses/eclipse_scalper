# PASSIVE_POCKET_RANKING

candidates=5 ranked=5
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=5 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=11 table_rows_seen=7 rows_with_pass_yes=5 candidates_parsed=5 candidates_unique=5 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=baseline gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.271118e-04 | -1.292155e-04 | 45.19% | 0.19 | -0.00027809 | -0.00028269 | -0.00028269 | 0.00% | 0.00% | -1.271118e-04 | -0.00027809 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 44.21% | 2.000 | 0.247 | -0.699 | -2.929 | 0.431 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.296508e-04 | -1.325035e-04 | 46.83% | 0.18 | -0.00027005 | -0.00027574 | -0.00027574 | 0.00% | 0.00% | -1.296508e-04 | -0.00027005 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 46.91% | 2.000 | 0.272 | -0.736 | -3.013 | 0.631 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.501078e-04 | -1.541274e-04 | 53.19% | 0.27 | -0.00027209 | -0.00027941 | -0.00027941 | 0.00% | 0.00% | -1.501078e-04 | -0.00027209 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 53.05% | 2.000 | 0.366 | -0.458 | -2.825 | 0.435 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.507765e-04 | -1.551840e-04 | 61.31% | 0.20 | -0.00025678 | -0.00026434 | -0.00026434 | 0.00% | 0.00% | -1.507765e-04 | -0.00025678 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 62.11% | 2.000 | 0.377 | -0.244 | -2.621 | 0.454 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.812271e-04 | -1.850859e-04 | 59.50% | 0.14 | -0.00030381 | -0.00031131 | -0.00031131 | 0.00% | 0.00% | -1.812271e-04 | -0.00030381 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 59.09% | 2.000 | 0.418 | -0.707 | -3.110 | 0.727 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -3.088044e-05 | +8.841728e-05 | +1.093008e-05 | +0.000000e+00 | -1.302278e-04 | -1.271118e-04 | +3.116013e-06 | 0.00% | 3911 | 3911 | 1729 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.453544e-05 | +9.382114e-05 | +1.275534e-05 | +0.000000e+00 | -1.411119e-04 | -1.296508e-04 | +1.146108e-05 | 0.00% | 3690 | 3690 | 1731 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.432482e-05 | +1.061083e-04 | +1.942891e-05 | +0.000000e+00 | -1.498620e-04 | -1.501078e-04 | -2.458230e-07 | 0.00% | 5468 | 5468 | 2901 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -1.515548e-05 | +1.242126e-04 | +2.339987e-05 | +0.000000e+00 | -1.627680e-04 | -1.507765e-04 | +1.199154e-05 | 0.00% | 3969 | 3969 | 2465 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -4.176410e-05 | +1.181718e-04 | +2.469164e-05 | +0.000000e+00 | -1.846276e-04 | -1.812271e-04 | +3.400480e-06 | 0.00% | 2735 | 2735 | 1616 |
