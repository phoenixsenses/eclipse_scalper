# PASSIVE_POCKET_RANKING

candidates=7 ranked=1
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=7 rows_skipped_missing_fields=0
fee_grid=[0.0] adverse_mult_grid=[0.5]
pass_threshold=0.330
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.20 | 2500 | 0.000500 | 5.160786e-01 | True | True | 55.56% | 55.56% | +1.775296e-04 | +1.775296e-04 | 55.98% | 0.08 | +0.00032865 | +0.00032865 | +0.00032865 | 55.56% | 55.56% | +1.415521e-04 | +0.00028295 | +0.00% | +3.597753e-05 | +0.00004570 | mixed | 3.62% | 53.33% | 0.000 | 0.142 | +3.256 | +3.114 | 1.949 | 0.00 | 33.33% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | +1.736627e-04 | +0.000000e+00 | +7.564452e-06 | +0.000000e+00 | +1.660983e-04 | +1.775296e-04 | +1.143135e-05 | 3.62% | 359 | 345 | 184 |
