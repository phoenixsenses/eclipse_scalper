# Daily Execution Calibration (2026-03-05)

- symbol: `ETHUSDT`
- interval_ms: `100`
- lookback_days: `14`
- ok: `1`
- root_cause_enabled: `0`

## Steps
- step_1: rc=0 cmd=`C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe -m tools.calibrate_execution_models --physics data/derived/physics --symbol ETHUSDT --interval-ms 100 --out data/derived/execution_calibration --days 14`
- step_2: rc=0 cmd=`C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe -m tools.execution_e2e_pipeline --sim logs/micro_edge_debug_trades.jsonl --live-db data/paper_trades.db --live-parquet data/live/papertrades_live.parquet`

## Run Summary

- `{'version': 'v1', 'run_type': 'daily_execution_calibration', 'inputs': {'symbol': 'ETHUSDT', 'interval_ms': 100, 'days': 14, 'run_root_cause': 0}, 'metrics': {'ok': True, 'step_count': 2}, 'artifacts': {'json': 'reports\\test_daily_execution_calibration\\2026-03-05_EXEC_CALIBRATION.json', 'md': 'reports\\test_daily_execution_calibration\\2026-03-05_EXEC_CALIBRATION.md'}}`