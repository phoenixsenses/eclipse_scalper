# PASSIVE_POCKET_FORWARD_VALIDATION

symbol=ETHUSDT horizon_sec=60 min_imbalance=0.5 min_trade_intensity=2500.0 max_spread=0.00025
seeds=[11, 22, 33, 44, 55] splits=0 min_n=50 min_n_frac=0.003 maker_fee_bps=1.0 passive_adverse_mult=1.0 v2_min_score=0.0 v2_min_persistence=0.0 v2_min_confidence=0.0
effective_min_n_formula=max(min_n=50, ceil(min_n_frac*val_rows)=ceil(0.003*val_rows)); median_frac_component=212 median_effective_min_n=212
gate: min_intensity_strong=0.0 min_imbalance_strong=0.0 max_spread_tight=0.0 max_volatility_extreme=0.0 vol_quantile_reject=0.0
scratch: scratch_bps=0.0 scratch_window_sec=0 scratch_taker_fee_bps=0.0 scratch_slippage_bps=0.0
regime_bucket=none

| seed | split | train_n | val_rows | effective_min_n | filled_n | filled_avg_net | filled_p90_net | filled_win_rate | attempt_fill_rate | net_per_attempt | attempts_per_min | val_before_gate | val_after_gate | fail_reason | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 11 | 1 | 100 | 70632 | 212 | 35 | -0.00010000 | -0.00005000 | 40.00% | 45.00% | -1.000000e-04 | 2.00 | 1200 | 1000 | insufficient_fills | NO |
| 22 | 1 | 100 | 70632 | 212 | 50 | -0.00010000 | -0.00005000 | 40.00% | 45.00% | -1.000000e-04 | 2.00 | 1200 | 1000 | insufficient_fills | NO |

pass_count=0/2
pass_rate=0.00%
insufficient_fill_rate=100.00%
min_n_frac_dominance_rate=100.00%

## Failure Reasons
- insufficient_fills: 2

## Per-Split Capacity
| split | n_seeds | filled_n_mean | attempt_fill_rate_mean | net_per_attempt_mean | attempts_per_min_mean |
|---:|---:|---:|---:|---:|---:|

CAPACITY_WARNING: >50% rows failed due to insufficient fills.

MIN_N_FRAC_WARNING: ceil(min_n_frac*val_rows) exceeded min_n for at least one split/seed row.

## Run Summary
- {'version': 'v1', 'run_type': 'validate_passive_pocket_forward', 'inputs': {}, 'metrics': {'rows_total': 2, 'pass_count': 0, 'pass_rate': 0.0, 'insufficient_fill_rate': 1.0}, 'artifacts': {'md': 'reports\\test_validate_forward_minfrac.md', 'json': 'reports\\test_validate_forward_minfrac.json'}}
