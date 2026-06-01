# PASSIVE_POCKET_FORWARD_VALIDATION

symbol=ETHUSDT horizon_sec=60 min_imbalance=0.3 min_trade_intensity=8000.0 max_spread=0.0002
seeds=[7, 11, 22] splits=4 min_n=20 min_n_frac=0.0001 maker_fee_bps=1.0 passive_adverse_mult=1.2 v2_min_score=0.0 v2_min_persistence=0.0 v2_min_confidence=0.0
effective_min_n_formula=max(min_n=20, ceil(min_n_frac*val_rows)=ceil(0.0001*val_rows)); median_frac_component=20 median_effective_min_n=20
gate: min_intensity_strong=0.0 min_imbalance_strong=0.0 max_spread_tight=0.0 max_volatility_extreme=0.0 vol_quantile_reject=0.0
event_filter: allow=[] block=[] kept_ratio=100.00%
scratch: scratch_bps=0.0 scratch_window_sec=0 scratch_taker_fee_bps=0.0 scratch_slippage_bps=0.0 exec_model=passive_realistic
regime_bucket=none

| seed | split | train_n | val_rows | effective_min_n | filled_n | filled_avg_net | filled_p90_net | filled_win_rate | attempt_fill_rate | net_per_attempt | attempts_per_min | val_before_gate | val_after_gate | fail_reason | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 7 | 1 | 198036 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 7 | 2 | 396072 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 7 | 3 | 594108 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 7 | 4 | 792144 | 198038 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 11 | 1 | 198036 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 11 | 2 | 396072 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 11 | 3 | 594108 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 11 | 4 | 792144 | 198038 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 22 | 1 | 198036 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 22 | 2 | 396072 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 22 | 3 | 594108 | 198036 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |
| 22 | 4 | 792144 | 198038 | 20 | 0 | +0.00000000 | +0.00000000 | 0.00% | 0.00% | +0.000000e+00 | 0.00 | 0 | 0 | insufficient_fills | NO |

pass_count=0/12
pass_rate=0.00%
insufficient_fill_rate=100.00%
min_n_frac_dominance_rate=0.00%

## Failure Reasons
- insufficient_fills: 12

## Per-Split Capacity
| split | n_seeds | filled_n_mean | attempt_fill_rate_mean | net_per_attempt_mean | attempts_per_min_mean |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0.00 | 0.00% | +0.000000e+00 | 0.00 |
| 2 | 3 | 0.00 | 0.00% | +0.000000e+00 | 0.00 |
| 3 | 3 | 0.00 | 0.00% | +0.000000e+00 | 0.00 |
| 4 | 3 | 0.00 | 0.00% | +0.000000e+00 | 0.00 |

CAPACITY_WARNING: >50% rows failed due to insufficient fills.

## Run Summary
- {'version': 'v1', 'run_type': 'validate_passive_pocket_forward', 'inputs': {'db': 'data/microstructure.db', 'symbol': 'ETHUSDT', 'lookback_min': 30240, 'bucket_sec': 1, 'horizon_sec': 60, 'rule': 'micro_edge_v3_passive_alpha', 'side': 'auto', 'min_imbalance': 0.3, 'min_trade_intensity': 8000.0, 'max_spread': 0.0002, 'splits': 5, 'seeds': [7, 11, 22], 'min_n': 20, 'min_n_frac': 0.0001, 'maker_fee_bps': 1.0, 'passive_adverse_mult': 1.2, 'event_allow_lanes': [], 'event_block_lanes': []}, 'metrics': {'rows_total': 12, 'pass_count': 0, 'pass_rate': 0.0, 'insufficient_fill_rate': 1.0, 'event_filter_kept_ratio': 1.0}, 'artifacts': {'md': 'reports\\PTT_REFRESH_21D_B_BASELINE.md', 'json': 'reports\\PTT_REFRESH_21D_B_BASELINE.json'}}
