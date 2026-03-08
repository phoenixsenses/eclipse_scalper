# PASSIVE_POCKET_RANKING

candidates=5 ranked=5
statistical bootstrap_ci=False bootstrap_samples=1000 alpha=0.0500 mtc_method=none splits=3 (recommended=5 for 60-day retest)
candidate_parse total_rows_seen=11 table_rows_seen=7 rows_with_pass_yes=5 candidates_parsed=5 candidates_unique=5 rows_skipped_missing_fields=0
fee_grid=[1.0] adverse_mult_grid=[0.8, 1.0, 1.2]
pass_threshold=0.500
liquidation_scoring_impact available=False count=0 positive_delta_score_count=0 avg_delta_score_raw_core=+0.000000e+00 avg_delta_npa_core=+0.000000e+00 avg_delta_pass_rate_core=+0.00%
mitigation_profile=event_block_eth_micro_imb085_v1 gate_config min_intensity_strong=0.000000 min_imbalance_strong=0.000000 max_spread_tight=0.000000 max_volatility_extreme=None vol_quantile_reject=0.010000 scratch_bps=0.0000 scratch_window_sec=0 scratch_taker_fee_bps=0.0000 scratch_slippage_bps=0.0000 passive_max_wait_buckets=0 horizon_sec_override=0

| rank | symbol | rule | horizon | min_imb | min_int | max_spread | score | robust_core | robust_stress | pass_rate_core | pass_rate_stress | npa_core | npa_stress | attempt_fill_rate | attempts_per_min | score_raw_core | score_raw_stress | score_raw_min | baseline_pass_rate_core | baseline_pass_rate_stress | baseline_npa_core | baseline_score_raw_core | delta_pass_rate_core | delta_npa_core | delta_score_raw_core | failure_reason_top | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_raw_return_bps_on_fills | avg_net_return_bps_on_fills | stability_std_bps | best_fee_survive | insufficient_fill_rate |
|---:|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.224297e-04 | -1.249671e-04 | 49.37% | 0.17 | -0.00023644 | -0.00024128 | -0.00024128 | 0.00% | 0.00% | -1.224297e-04 | -0.00023644 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 50.64% | 2.000 | 0.239 | -0.168 | -2.408 | 0.345 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.419315e-04 | -1.475188e-04 | 58.91% | 0.10 | -0.00023168 | -0.00024074 | -0.00024074 | 0.00% | 0.00% | -1.419315e-04 | -0.00023168 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.00% | 2.000 | 0.453 | -0.057 | -2.510 | 0.994 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.565804e-04 | -1.594617e-04 | 48.56% | 0.14 | -0.00031996 | -0.00032585 | -0.00032585 | 0.00% | 0.00% | -1.565804e-04 | -0.00031996 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 49.87% | 2.000 | 0.292 | -0.890 | -3.183 | 1.042 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.769240e-04 | -1.814004e-04 | 54.89% | 0.22 | -0.00031565 | -0.00032363 | -0.00032363 | 0.00% | 0.00% | -1.769240e-04 | -0.00031565 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 55.04% | 2.000 | 0.399 | -0.580 | -2.979 | 0.301 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -2.004023e-04 | -2.053713e-04 | 61.04% | 0.16 | -0.00032603 | -0.00033430 | -0.00033430 | 0.00% | 0.00% | -2.004023e-04 | -0.00032603 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.03% | 2.000 | 0.407 | -0.786 | -3.193 | 0.702 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -8.529426e-06 | +1.012721e-04 | +1.211991e-05 | +0.000000e+00 | -1.219214e-04 | -1.224297e-04 | -5.082342e-07 | 0.00% | 2830 | 2830 | 1433 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.453799e-06 | +1.219989e-04 | +2.765606e-05 | +0.000000e+00 | -1.531087e-04 | -1.419315e-04 | +1.117717e-05 | 0.00% | 1744 | 1741 | 1062 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -4.440638e-05 | +9.974249e-05 | +1.458272e-05 | +0.000000e+00 | -1.587316e-04 | -1.565804e-04 | +2.151143e-06 | 0.00% | 2330 | 2330 | 1162 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -3.191470e-05 | +1.100822e-04 | +2.196541e-05 | +0.000000e+00 | -1.639623e-04 | -1.769240e-04 | -1.296173e-05 | 0.00% | 3650 | 3650 | 2009 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -4.795169e-05 | +1.220599e-04 | +2.486294e-05 | +0.000000e+00 | -1.948745e-04 | -2.004023e-04 | -5.527728e-06 | 0.00% | 2738 | 2738 | 1671 |
