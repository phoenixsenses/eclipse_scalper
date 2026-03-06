# Rollout: Execution V2

## Feature Flags
- `EXEC_LATENCY_V2` (0/1)
- `QUEUE_MODEL_V2` (0/1)
- `EXEC_ENGINE_UNIFIED` (0/1)

## Rollout Steps
1. Canary (1 symbol)
   - Enable only on one symbol/session.
   - Run diagnostics:
     - `python -m tools.execution_diagnostics`
     - `python -m tools.toxicity_report`
2. Limited (2 symbols)
   - Keep same risk limits.
   - Compare to baseline report deltas.
3. Full rollout
   - Enable all 3 flags globally.
   - Run post-check:
     - `python -m tools.post_rollout_audit`

## Rollback
- Set all three flags to `0`.
- Restart service.
- Re-run diagnostics and verify baseline recovery.

## Required Checks
- Fill-rate does not collapse vs baseline.
- Latency p95 stays within budget.
- Toxicity score does not spike abnormally.
- No order-state contract violations in logs.

