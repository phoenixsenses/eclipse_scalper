# Observability Contract (Execution)

Goal: reconstruct one order intent lifecycle end-to-end under failures.

## Required Correlation Fields

Every execution event touching an order intent must include:

- `symbol` (canonical key)
- `correlation_id` (stable per intent)
- `event` (typed event name)
- `ts` (event timestamp)
- `reason` (machine-friendly)
- `code` (normalized error code when relevant)

Optional but recommended:
- `order_id`
- `client_order_id`
- `error_class`
- `attempt` / `tries`
- `mode` (belief mode when available)

## Required Event Families

1. Router placement path
- `order.create.success` / `order.create.failed`
- `order.retry` (one event per retry attempt)

2. Cancel/replace path
- `order.cancel` outcome with idempotent status context
- `order.cancel_replace_giveup` when bounded attempts exhausted

3. Reconcile path
- `reconcile.summary` with mismatch/repair counts and streak
- `execution.belief_state` with debt metrics and posture trace

4. Circuit breaker path
- `kill_switch.halt`
- `kill_switch.clear`
- Escalation events where relevant

## Reconstruction Guarantees

Given telemetry stream + correlation filter:

1. Identify initial intent.
2. Enumerate every retry and classification reason.
3. Determine terminal path:
- created and adopted
- canceled as idempotent success
- canceled with conflict and reconciled
- gave up and escalated to reconcile / halt
4. Explain guard posture active during event window.

## Alerting Guardrails

- Alert when `order.retry` lacks `correlation_id`.
- Alert when repeated retries have no terminal event in bounded window.
- Alert when reconcile mismatch grows while entries still remain unconstrained.
- Alert when halt/clear events occur without nearby belief-state evidence.
