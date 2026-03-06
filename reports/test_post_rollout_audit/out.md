# POST ROLLOUT AUDIT

- ts_utc: 2026-03-05T19:12:34Z
- overall_ok: True

## Flags
- EXEC_LATENCY_V2: 0
- QUEUE_MODEL_V2: 0
- EXEC_ENGINE_UNIFIED: 0

## Checks
- diag_rows_ok: 1
- tox_rows_ok: 1
- latency_p95_ok: 1
- fill_rate_ok: 1

## Run Summary
- `{'version': 'v1', 'run_type': 'post_rollout_audit', 'inputs': {'diag_json': 'reports\\test_post_rollout_audit\\diag.json', 'tox_json': 'reports\\test_post_rollout_audit\\tox.json'}, 'metrics': {'overall_ok': True, 'check_count': 4}, 'artifacts': {'json': 'reports\\test_post_rollout_audit\\out.json', 'md': 'reports\\test_post_rollout_audit\\out.md'}}`
