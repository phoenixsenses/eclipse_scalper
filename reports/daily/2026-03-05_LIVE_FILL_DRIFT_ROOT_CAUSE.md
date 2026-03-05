# LIVE FILL DRIFT ROOT-CAUSE REPORT

- ts_utc: 2026-03-05T14:24:09Z
- overall_status: attention

## Inputs
- parity_json: `reports/REPLAY_PARITY_REPORT.json`
- diagnostics_json: `reports/EXECUTION_HEALTH.json`
- toxicity_json: `reports/TOXICITY_REPORT.json`
- audit_json: `reports/POST_ROLLOUT_AUDIT.json`

## Ranked Root Causes
### 1. Insufficient/Noisy Evidence (score=1.800)
Evidence:
- Low diagnostic sample size: rows=0.
- Post-rollout audit has failing checks.
- Insufficient simulated fills: sim_count=0.
Actions:
- Increase sample horizon (>=24h and >=50 matched fills) before retuning.
- Run daily calibration pipeline and enforce artifact completeness checks.
- Block parameter updates when audit or coverage checks fail.
