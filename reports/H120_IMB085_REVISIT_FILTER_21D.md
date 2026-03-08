# PASSIVE_POCKET_RANKING

candidates=5 ranked=5
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=5 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=11 table_rows_seen=7 rows_with_pass_yes=5 candidates_parsed=5 candidates_unique=5 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=event_block_eth_micro_imb085_v1 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.237550e-04 | -1.259533e-04 | 45.78% | 0.20 | -0.00027929 | -0.00028429 | -0.00028429 | 0.00% | 0.00% | -1.237550e-04 | -0.00027929 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 44.07% | 2.000 | 0.250 | -0.486 | -2.716 | 0.396 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.281429e-04 | -1.308383e-04 | 47.95% | 0.18 | -0.00027969 | -0.00028519 | -0.00028519 | 0.00% | 0.00% | -1.281429e-04 | -0.00027969 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 47.90% | 2.000 | 0.273 | -0.492 | -2.765 | 0.435 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.592295e-04 | -1.631476e-04 | 54.47% | 0.27 | -0.00028656 | -0.00029333 | -0.00029333 | 0.00% | 0.00% | -1.592295e-04 | -0.00028656 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 54.55% | 2.000 | 0.365 | -0.383 | -2.752 | 0.391 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.710269e-04 | -1.757248e-04 | 62.01% | 0.19 | -0.00027568 | -0.00028325 | -0.00028325 | 0.00% | 0.00% | -1.710269e-04 | -0.00027568 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 62.76% | 2.000 | 0.376 | -0.518 | -2.898 | 0.596 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.942283e-04 | -1.987555e-04 | 59.77% | 0.14 | -0.00032645 | -0.00033493 | -0.00033493 | 0.00% | 0.00% | -1.942283e-04 | -0.00032645 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 58.48% | 2.000 | 0.416 | -0.718 | -3.145 | 0.521 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.142185e-05 | +8.814070e-05 | +1.099917e-05 | +0.000000e+00 | -1.205617e-04 | -1.237550e-04 | -3.193289e-06 | 0.00% | 3980 | 3980 | 1754 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -2.357519e-05 | +9.580153e-05 | +1.308889e-05 | +0.000000e+00 | -1.324656e-04 | -1.281429e-04 | +4.322675e-06 | 0.00% | 3668 | 3668 | 1757 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.091172e-05 | +1.091077e-04 | +1.993335e-05 | +0.000000e+00 | -1.499527e-04 | -1.592295e-04 | -9.276712e-06 | 0.00% | 5424 | 5424 | 2959 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.250775e-05 | +1.255120e-04 | +2.357825e-05 | +0.000000e+00 | -1.815980e-04 | -1.710269e-04 | +1.057107e-05 | 0.00% | 3955 | 3955 | 2482 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -4.197288e-05 | +1.169620e-04 | +2.431846e-05 | +0.000000e+00 | -1.832534e-04 | -1.942283e-04 | -1.097495e-05 | 0.00% | 2765 | 2765 | 1617 |
