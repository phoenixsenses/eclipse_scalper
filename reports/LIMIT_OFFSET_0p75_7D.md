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
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.038971e-04 | -1.061177e-04 | 49.32% | 0.17 | -0.00021755 | -0.00022238 | -0.00022238 | 0.00% | 0.00% | -1.038971e-04 | -0.00021755 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.41% | 2.000 | 0.243 | +0.093 | -2.150 | 0.417 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.327655e-04 | -1.353850e-04 | 47.65% | 0.13 | -0.00028882 | -0.00029449 | -0.00029449 | 0.00% | 0.00% | -1.327655e-04 | -0.00028882 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 47.73% | 2.000 | 0.287 | -0.674 | -2.961 | 0.554 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.338950e-04 | -1.390424e-04 | 58.47% | 0.10 | -0.00023629 | -0.00024538 | -0.00024538 | 0.00% | 0.00% | -1.338950e-04 | -0.00023629 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 59.08% | 2.000 | 0.451 | +0.188 | -2.262 | 0.545 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.575538e-04 | -1.616871e-04 | 52.76% | 0.21 | -0.00029991 | -0.00030795 | -0.00030795 | 0.00% | 0.00% | -1.575538e-04 | -0.00029991 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 52.59% | 2.000 | 0.399 | -0.486 | -2.885 | 0.345 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.053939e-04 | -2.104827e-04 | 59.49% | 0.16 | -0.00033742 | -0.00034578 | -0.00034578 | 0.00% | 0.00% | -2.053939e-04 | -0.00033742 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 60.02% | 2.000 | 0.409 | -0.772 | -3.181 | 0.700 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | +4.592043e-06 | +9.881450e-05 | +1.202372e-05 | +0.000000e+00 | -1.062462e-04 | -1.038971e-04 | +2.349126e-06 | 0.00% | 2868 | 2868 | 1417 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.218167e-05 | +9.546455e-05 | +1.368056e-05 | +0.000000e+00 | -1.413268e-04 | -1.327655e-04 | +8.561254e-06 | 0.00% | 2271 | 2271 | 1084 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | +1.112458e-05 | +1.181660e-04 | +2.663106e-05 | +0.000000e+00 | -1.336725e-04 | -1.338950e-04 | -2.225076e-07 | 0.00% | 1723 | 1723 | 1018 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.555105e-05 | +1.051811e-04 | +2.098401e-05 | +0.000000e+00 | -1.517161e-04 | -1.575538e-04 | -5.837649e-06 | 0.00% | 3590 | 3590 | 1888 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -4.635889e-05 | +1.200448e-04 | +2.454406e-05 | +0.000000e+00 | -1.909477e-04 | -2.053939e-04 | -1.444618e-05 | 0.00% | 2679 | 2679 | 1608 |
