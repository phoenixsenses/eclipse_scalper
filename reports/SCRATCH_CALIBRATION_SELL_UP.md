# SCRATCH_ANALYSIS

symbol=ETHUSDT side=SELL regime= lookback_min=43200 bucket_sec=1 horizon_sec=120 exec_model=passive_realistic
scratch_taker_fee_bps=0.000 scratch_slippage_bps=0.000
pocket: imb>=0.500 int>=3500 spr<=0.000300

## Baseline
n=0 mean_net=+0.000000e+00 scratch_frac=0.00% horizon_frac=0.00%

## Max Adverse Sweep
| max_adverse_bps | n | mean_net | delta_vs_baseline | scratch_frac |
|---:|---:|---:|---:|---:|
| 2.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 2.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 3.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 3.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 4.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 4.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 5.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 5.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 6.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 6.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 7.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 7.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 8.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 8.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 9.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 9.50 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 10.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |

best_max_adverse_bps=2.00 mean_net=+0.000000e+00

## Trailing Proxy Sweep
| trailing_stop_bps_proxy | n | mean_net | delta_vs_baseline | scratch_frac |
|---:|---:|---:|---:|---:|
| 2.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 3.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 4.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |
| 5.00 | 0 | +0.000000e+00 | +0.000000e+00 | 0.00% |

best_trailing_stop_bps_proxy=2.00 mean_net=+0.000000e+00

## Calibration Notes

- Primary sample too low: baseline_n=0 < min_trades=30. Fallback run executed (regime=NONE, lookback_min=43200).
- Fallback did not improve sample (primary_n=0, fallback_n=0, rc_fallback=0).
- Insufficient calibration sample remains after fallback. Action: increase lookback, relax regime filter, or wait for more data.
