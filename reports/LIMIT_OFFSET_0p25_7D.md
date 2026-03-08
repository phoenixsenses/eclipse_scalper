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
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.85 | 4000 | 0.000150 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.207140e-04 | -1.232162e-04 | 51.09% | 0.17 | -0.00023618 | -0.00024109 | -0.00024109 | 0.00% | 0.00% | -1.207140e-04 | -0.00023618 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 51.44% | 2.000 | 0.243 | -0.155 | -2.399 | 0.398 | 0.00 | 0.00% |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.85 | 6000 | 0.000300 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.353049e-04 | -1.381330e-04 | 51.61% | 0.13 | -0.00027251 | -0.00027828 | -0.00027828 | 0.00% | 0.00% | -1.353049e-04 | -0.00027251 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 51.77% | 2.000 | 0.287 | -0.663 | -2.950 | 0.753 | 0.00 | 0.00% |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.495513e-04 | -1.539369e-04 | 54.51% | 0.21 | -0.00027419 | -0.00028222 | -0.00028222 | 0.00% | 0.00% | -1.495513e-04 | -0.00027419 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 55.17% | 2.000 | 0.396 | -0.390 | -2.785 | 0.406 | 0.00 | 0.00% |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 6000 | 0.000200 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.564579e-04 | -1.612646e-04 | 62.20% | 0.16 | -0.00025047 | -0.00025858 | -0.00025858 | 0.00% | 0.00% | -1.564579e-04 | -0.00025047 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 62.34% | 2.000 | 0.405 | -0.239 | -2.644 | 0.748 | 0.00 | 0.00% |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | 0.50 | 8000 | 0.000250 | 0.000000e+00 | False | False | 0.00% | 0.00% | -1.629367e-04 | -1.684419e-04 | 60.07% | 0.10 | -0.00026574 | -0.00027472 | -0.00027472 | 0.00% | 0.00% | -1.629367e-04 | -0.00026574 | +0.00% | +0.000000e+00 | +0.00000000 | fees_dominate | 0.00% | 61.07% | 2.000 | 0.446 | -0.257 | -2.703 | 0.705 | 0.00 | 0.00% |

survive_fee1_passrate_ge_0.5=0

## Decomposition

| rank | symbol | rule | h | gross_edge_npa | fee_cost_npa | adverse_cost_npa | scratch_cost_npa | net_npa | observed_npa | residual_npa | reject_rate | n_events | n_after_gate | n_filled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -7.995143e-06 | +1.028849e-04 | +1.251821e-05 | +0.000000e+00 | -1.233983e-04 | -1.207140e-04 | +2.684318e-06 | 0.00% | 2877 | 2877 | 1480 |
| 2 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -3.430098e-05 | +1.035367e-04 | +1.485531e-05 | +0.000000e+00 | -1.526930e-04 | -1.353049e-04 | +1.738812e-05 | 0.00% | 2262 | 2262 | 1171 |
| 3 | ETHUSDT | intensity_spike_imbalance_cont | 60 | -2.149002e-05 | +1.103391e-04 | +2.182728e-05 | +0.000000e+00 | -1.536564e-04 | -1.495513e-04 | +4.105044e-06 | 0.00% | 3598 | 3598 | 1985 |
| 4 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -1.488045e-05 | +1.246744e-04 | +2.524585e-05 | +0.000000e+00 | -1.648007e-04 | -1.564579e-04 | +8.342792e-06 | 0.00% | 2687 | 2687 | 1675 |
| 5 | ETHUSDT | intensity_spike_imbalance_cont | 120 | -1.570098e-05 | +1.221392e-04 | +2.724193e-05 | +0.000000e+00 | -1.650821e-04 | -1.629367e-04 | +2.145362e-06 | 0.00% | 1739 | 1739 | 1062 |
