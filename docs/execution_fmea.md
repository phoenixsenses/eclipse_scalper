# Execution FMEA

Failure-mode focus is execution reliability, not strategy quality.

## Top Failure Modes

| Failure mode | Primary detection | Recovery / mitigation | Owner module |
|---|---|---|---|
| Create times out but exchange may have accepted order | Router retry classification + `order.retry` telemetry + reconcile open order evidence | Retry with bounded backoff/jitter and stable correlation identity; stop at max elapsed | `execution/order_router.py` |
| Duplicate `clientOrderId` on retry | Exchange error class `retryable_with_modification` | Rotate bounded variant (length-safe) without unbounded growth; retry within budget | `execution/order_router.py` |
| Cancel after already-filled or unknown order | Unknown-order class + optional `fetch_order` status | Treat as idempotent success unless status indicates open conflict | `execution/order_router.py` |
| Cancel/replace loops under ambiguity | Attempt counters and give-up telemetry | Bounded `cancel_replace_order` attempts; give-up path triggers reconcile | `execution/order_router.py` + `execution/reconcile.py` |
| Partial fill while local state assumes atomic execution | Order status/open-order evidence + position drift on reconcile | Incremental state adoption and protective-order repair; no atomicity assumption | `execution/reconcile.py` + `execution/position_manager.py` |
| Position exists but protection orders missing | Reconcile stop check (`_ensure_protective_stop`) | Repair placement with cooldown to avoid thrash | `execution/reconcile.py` |
| Delayed/contradictory open-orders vs positions snapshots | Mismatch streak + debt symbol map + confidence drop | Belief debt escalation, guard posture tightening, optional kill-switch halt | `execution/reconcile.py` + `execution/belief_controller.py` + `risk/kill_switch.py` |
| Rate-limit storm / exchange overload | Error reason class and repeated retry alerts | Backoff + capped retries + halt escalation on sustained pressure | `execution/order_router.py` + `risk/kill_switch.py` |
| Timestamp skew / recvWindow drift | Error reason `timestamp` + sync diagnostics | Limited retry class and time-sync path; fail safe if persistent | `execution/order_router.py` + exchange bootstrap |
| Local/exchange divergence persists | Reconcile mismatch streak and debt growth | Kill-switch debt trigger and entry freeze; require clean evidence to recover | `risk/kill_switch.py` + `execution/belief_controller.py` |

## RPN Notes (Current Priorities)

Highest operational priority:
- Snapshot contradictions with missing protections.
- Retry storms caused by exchange overload/timeouts.
- Silent observability gaps (missing correlation links).

## Gaps To Continue Hardening

- Add explicit per-order lifecycle completeness checks (intent created -> terminal outcome observed).
- Add bounded stale-order sweep cadence for symbols with repeated unknown-order conflicts.
- Add synthetic chaos replay as CI gate (seeded event stream with expected invariant outcomes).
