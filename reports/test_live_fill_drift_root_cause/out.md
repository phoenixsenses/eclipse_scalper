# LIVE FILL DRIFT ROOT-CAUSE REPORT

- ts_utc: 2026-03-05T19:12:34Z
- overall_status: attention

## Inputs
- parity_json: `reports\test_live_fill_drift_root_cause\parity.json`
- diagnostics_json: `reports\test_live_fill_drift_root_cause\diag.json`
- toxicity_json: `reports\test_live_fill_drift_root_cause\tox.json`
- audit_json: `reports\test_live_fill_drift_root_cause\audit.json`

## Ranked Root Causes
### 1. Insufficient/Noisy Evidence (score=1.400)
Evidence:
- Low diagnostic sample size: rows=20.
- Insufficient simulated fills: sim_count=10.
Actions:
- Increase sample horizon (>=24h and >=50 matched fills) before retuning.
- Run daily calibration pipeline and enforce artifact completeness checks.
- Block parameter updates when audit or coverage checks fail.

## Run Summary
- `{'version': 'v1', 'run_type': 'live_fill_drift_root_cause', 'inputs': {'parity_json': 'reports\\test_live_fill_drift_root_cause\\parity.json', 'diag_json': 'reports\\test_live_fill_drift_root_cause\\diag.json', 'tox_json': 'reports\\test_live_fill_drift_root_cause\\tox.json', 'audit_json': 'reports\\test_live_fill_drift_root_cause\\audit.json', 'run_pipeline': False}, 'metrics': {'overall_status': 'attention', 'cause_count': 1, 'top_score': 1.4}, 'artifacts': {'json': 'reports\\test_live_fill_drift_root_cause\\out.json', 'md': 'reports\\test_live_fill_drift_root_cause\\out.md'}}`
