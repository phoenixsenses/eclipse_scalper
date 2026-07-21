# PASSIVE_POCKET_RANKING

candidates=1 ranked=1
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=3 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=1 table_rows_seen=1 rows_with_pass_yes=1 candidates_parsed=1 candidates_unique=1 rows_skipped_missing_fields=0
fee_grid=[1.0, 1.5] adverse_mult_grid=[1.0]
pass_threshold=0.500
liquidation_scoring_impact available=True count=1 positive_delta_score_count=1 avg_delta_score_raw_core=+1.000000e-05 avg_delta_npa_core=+4.000000e-06 avg_delta_pass_rate_core=+0.00%
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 60 | 0.50 | 2500 | 0.000250 | 1.600000e-01 | True | True | 100.00% | 100.00% | +1.600000e-05 | +1.600000e-05 | 40.00% | 90.00 | +0.00004000 | +0.00002500 | +0.00002500 | 100.00% | 100.00% | +1.200000e-05 | +0.00003000 | +0.00% | +4.000000e-06 | +0.00001000 | no_fills | 0.00% | 0.00% | 0.000 | 0.000 | +0.000 | +0.000 | 0.000 | 1.50 | 0.00% |

survive_fee1_passrate_ge_0.5=1

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 60 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +0.000000e+00 | +1.600000e-05 | +1.600000e-05 | 0.00% | 0 | 0 | 0 |
