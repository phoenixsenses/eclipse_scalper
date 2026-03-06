# SCRATCH_ANALYSIS

symbol=ETHUSDT side=SELL regime=UP lookback_min=10 bucket_sec=1 horizon_sec=120 exec_model=passive_realistic
scratch_taker_fee_bps=0.000 scratch_slippage_bps=0.000
pocket: imb>=0.500 int>=3000 spr<=0.000300

## Baseline
n=5 mean_net=+1.000000e-04 scratch_frac=20.00% horizon_frac=80.00%

## Max Adverse Sweep
| max_adverse_bps | n | mean_net | delta_vs_baseline | scratch_frac |
|---:|---:|---:|---:|---:|


## Trailing Proxy Sweep
| trailing_stop_bps_proxy | n | mean_net | delta_vs_baseline | scratch_frac |
|---:|---:|---:|---:|---:|


## Run Summary
- `{'version': 'v1', 'run_type': 'backtest_scratch', 'inputs': {'db': 'data/microstructure.db', 'symbol': 'ETHUSDT', 'side': 'SELL', 'regime': 'UP'}, 'metrics': {'baseline_n': 5, 'baseline_mean_net': 0.0001}, 'artifacts': {'json': 'reports\\test_backtest_scratch\\out.json', 'md': 'reports\\test_backtest_scratch\\out.md'}}`
