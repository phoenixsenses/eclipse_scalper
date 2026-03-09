# Event Lane Gate Shadow Runbook

## Scope

This runbook covers the first narrow live shadow rollout for the event lane gate.

Current scope:
- symbol: `ETHUSDT`
- source: `micro_signal`
- rule family: `micro_edge_v3_passive_alpha`
- horizon: `h=60`
- pocket threshold: `abs(min_imbalance) >= 0.85`
- blocking lanes:
  - `book_proxy_pressure`
  - `volatility_burst`

The gate uses **current bucket tags only**. It does not block from stale summary state.

## Decision Contract

Question:
- Does current-bucket event gating produce sane `allowed` vs `would_block` decisions on live telemetry without false blocks?

Primary metrics:
- `would_block_count`
- `allowed_count`
- `would_block_rate`

Secondary metrics:
- `blocking_lane_counts`
- latest decision payload
- latest bucket imbalance

Success criteria:
- clean current bucket -> `allowed`
- dirty current bucket -> `would_block`
- no false block caused only by stale lane summary state

Failure / freeze criteria:
- repeated `would_block` decisions while latest bucket is clean
- no `allowed` decisions for long clean stretches
- lane counts dominated by obviously stale conditions

Outcome labels:
- `shadow_ok`
- `shadow_noisy`
- `shadow_false_block_risk`

## Environment

Set:

```powershell
$env:ENTRY_EVENT_LANE_GATE_ENABLED="1"
$env:ENTRY_EVENT_LANE_GATE_SHADOW="1"
$env:ENTRY_EVENT_LANE_GATE_DB="data/microstructure.db"
```

Optional:

```powershell
$env:TELEMETRY_PATH="logs/telemetry.jsonl"
```

## What the bot emits

When the narrow pocket is in scope, entry loop emits telemetry event:

- `entry.event_lane_gate`

Decision values:
- `allowed`
- `would_block`
- `blocked` (only if shadow is disabled later)

Relevant payload fields:
- `symbol`
- `rule_name`
- `horizon_sec`
- `shadow`
- `decision`
- `gate_reason`
- `gate_status`
- `blocking_lanes`
- `latest_abs_imbalance`
- `latest_ts_ms`

## Review command

Run:

```powershell
py -3 -m tools.review_event_lane_gate_shadow --telemetry-path logs/telemetry.jsonl --symbol ETHUSDT
```

Expected output:
- `rows_total`
- `allowed_count`
- `would_block_count`
- `allowed_rate`
- `would_block_rate`
- `blocking_lane_counts`
- `latest`

## How to interpret

Good shadow behavior:
- `allowed` appears on clean buckets
- `would_block` appears only when current bucket tags fire
- `blocking_lane_counts` is plausible and not dominated by one obviously stale source

Bad shadow behavior:
- `would_block` spikes while current bucket is clean
- no `allowed` events even though live ETH micro entries are happening
- latest payload contradicts current lane context

## Promotion rule

Do not enable active blocking until:
- shadow telemetry is non-empty
- review output looks stable
- no false block pattern is observed

First active rollout should stay narrow:
- `ETHUSDT`
- `micro_edge_v3_passive_alpha`
- `h=60`

