# Daily Execution Calibration (2026-03-05)

- symbol: `ETHUSDT`
- interval_ms: `100`
- lookback_days: `14`
- ok: `1`
- root_cause_enabled: `1`

## Steps
- step_1: rc=0 cmd=`C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe -m tools.calibrate_execution_models --physics data/derived/physics --symbol ETHUSDT --interval-ms 100 --out data/derived/execution_calibration --days 14`
- step_2: rc=0 cmd=`C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe -m tools.execution_e2e_pipeline --sim logs/micro_edge_debug_trades.jsonl --live-db data/paper_trades.db --live-parquet data/live/papertrades_live.parquet`
- step_3: rc=0 cmd=`C:\Users\Windows 11\AppData\Local\Programs\Python\Python313\python.exe -m tools.live_fill_drift_root_cause --parity-json reports/REPLAY_PARITY_REPORT.json --diag-json reports/EXECUTION_HEALTH.json --tox-json reports/TOXICITY_REPORT.json --audit-json reports/POST_ROLLOUT_AUDIT.json --out-json reports\daily\2026-03-05_LIVE_FILL_DRIFT_ROOT_CAUSE.json --out-md reports\daily\2026-03-05_LIVE_FILL_DRIFT_ROOT_CAUSE.md`

## Root Cause Artifacts
- `reports\daily\2026-03-05_LIVE_FILL_DRIFT_ROOT_CAUSE.md`
- `reports\daily\2026-03-05_LIVE_FILL_DRIFT_ROOT_CAUSE.json`
