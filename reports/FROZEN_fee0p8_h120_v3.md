# PASSIVE_POCKET_RANKING

candidates=1 ranked=1
candidate_parse total_rows_seen=26 table_rows_seen=22 rows_with_pass_yes=7 candidates_parsed=7 candidates_unique=1 rows_skipped_missing_fields=0
fee_grid=[0.8] adverse_mult_grid=[1.0]
pass_threshold=0.500
mitigation_profile=anti_adverse_v3 gate_config min_intensity_strong=3500.000000 min_imbalance_strong=0.550000 max_spread_tight=0.000250 max_volatility_extreme=None vol_quantile_reject=0.020000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 horizon_sec_override=120

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | micro_edge_v3_passive_alpha | 120 | 0.40 | 2500 | 0.000300 | 0.000000e+00 | False | False | 33.33% | 33.33% | -3.587434e-05 | -3.587434e-05 | 60.62% | 0.02 | -0.00006180 | -0.00006180 | -0.00006180 | 27.78% | 27.78% | -5.880160e-05 | -0.00009259 | +5.56% | +2.292726e-05 | +0.00003079 | gate_reject | 67.25% | 59.79% | 1.600 | 0.328 | +1.557 | -0.371 | 1.725 | 0.80 | 0.00% |

survive_fee1_passrate_ge_0.5=0
