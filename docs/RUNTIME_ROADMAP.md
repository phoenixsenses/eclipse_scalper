# Runtime/Ops Improvement Roadmap

Owner: Person 2 (Runtime/Ops)
Branch: `codex/runtime/ops-foundation`
Created: 2026-03-09

## Week 1 — Quick Wins (Safety)

| # | Task | Severity | Effort | Status |
|---|------|----------|--------|--------|
| 1 | CancelledError re-raise in entry_loop.py | Medium | 1-2h | DONE (already clean) |
| 2 | Atomic disk writes audit (shutdown_control, bootstrap) | High | 2-3h | DONE `e6d4400` |
| 3 | Startup validation gates (symbols, hedge mode, API perms) | Medium | 4h | DONE `dec1c4c` |

## Week 2 — High Impact (Risk)

| # | Task | Severity | Effort | Status |
|---|------|----------|--------|--------|
| 4 | Margin/liquidation proximity alerts | High | 2-3h | DONE `059c3e9` |
| 5 | SIGTERM graceful shutdown (cancel orders before exit) | High | 4-5h | DONE (already handled) |
| 6 | Position stuck detection alert (open > TTL with no fills) | Medium | 2h | DONE `059c3e9` |

## Week 3 — Medium Impact (Reliability)

| # | Task | Severity | Effort | Status |
|---|------|----------|--------|--------|
| 7 | Exchange reconnect: add jitter + circuit breaker | Medium | 2h | DONE (already in collector) |
| 8 | Position drift alerts (size/price divergence > 5%) | Medium | 4-5h | DONE (already in reconcile) |
| 9 | Dashboard consolidated risk-metrics endpoint | Low | 5h | DONE |

## Week 4+ — Polish

| # | Task | Severity | Effort | Status |
|---|------|----------|--------|--------|
| 10 | Log rotation manager (compress + archive old JSONL) | Medium | 6-8h | TODO |
| 11 | Prometheus /metrics endpoint | Low | 6h | TODO |
| 12 | WebSocket real-time dashboard | Low | 8+h | TODO |

## Details

### 1. CancelledError re-raise
`execution/entry_loop.py` catches CancelledError but doesn't re-raise in some paths.
This delays graceful shutdown.

### 2. Atomic disk writes
`execution/shutdown_control.py` uses `path.write_text()` directly.
Should use tmp+rename pattern to survive power loss during write.

### 3. Startup validation
No pre-loop check that symbols exist on exchange, hedge mode is on,
API key has create/cancel permissions. Bot can start misconfigured.

### 4. Margin/liquidation alerts
`risk/kill_switch.py` tracks drawdown but not margin ratio.
Need to fetch `marginRatio` from Binance and alert if < 5%.

### 5. SIGTERM graceful shutdown
SIGINT (Ctrl+C) handled but SIGTERM not explicit.
Orchestrator kill could leave open orders orphaned.

### 6. Position stuck detection
Reconcile detects orphans but not positions open > TTL with no fill activity.
Stale capital gets locked without operator awareness.

### 7. Exchange reconnect jitter
Current: fixed exponential backoff. Missing: jitter to avoid thundering herd.

### 8. Position drift alerts
Reconcile corrects state silently. Operator should be alerted when
internal vs exchange position differs by > 5%.

### 9. Dashboard risk-metrics
Frontend makes 5+ requests for risk state. Single consolidated endpoint
would reduce latency and improve operator UX.

### 10. Log rotation
JSONL files grow unbounded. Need weekly compression + 6-month archive policy.

### 11. Prometheus endpoint
Standard /metrics for Grafana/Datadog integration.

### 12. WebSocket
Bidirectional real-time: push metrics, receive control commands.
