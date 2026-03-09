# Event Lane Gate Port

## Goal

Port the research-side event lane decision into the live bot with minimal coupling.

Target decision:

- for the target pocket, return one of:
  - `allowed`
  - `blocked`
  - `inactive`
  - `no_data`

Research bridge reference:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py)

## What Must Be Ported

Only the minimal execution-facing logic:

1. load recent rows from `data/microstructure.db`
2. compute current `book_proxy_pressure`
3. compute current `volatility_burst`
4. inspect the latest bucket
5. decide if the target pocket is active
6. decide if active pocket is blocked by current bucket tags

Do not port:

- watchboard
- trend
- suppression
- persistence
- merged banner
- operator brief

Those are monitoring layers, not entry control.

## Recommended Live Helper

Create:

- `execution/event_lane_gate.py`

Recommended functions:

```python
def load_current_event_gate(
    *,
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    pocket_horizon_sec: int,
    pocket_min_abs_imbalance: float,
) -> dict: ...

def should_block_event_gate(
    gate_payload: dict,
    *,
    symbol: str,
    rule_name: str,
    horizon_sec: int,
) -> tuple[bool, str, dict]: ...
```

## Proposed Runtime Contract

Minimum payload shape:

```json
{
  "symbol": "ETHUSDT",
  "pocket": {
    "name": "h60_imb_ge_0p85",
    "horizon_sec": 60,
    "min_abs_imbalance": 0.85,
    "active": true,
    "latest_ts_ms": 0,
    "latest_abs_imbalance": 0.0
  },
  "decision": {
    "status": "allowed",
    "reason": "pocket_active_and_no_blocking_lanes",
    "allow_trade": true,
    "blocked": false,
    "blocking_lanes": []
  },
  "lanes": {
    "book_proxy_pressure": {
      "current": {
        "rule_fired": false,
        "severity": "none"
      }
    },
    "volatility_burst": {
      "current": {
        "rule_fired": false,
        "severity": "none"
      }
    }
  }
}
```

## Gate Semantics

These rules must hold:

### `no_data`

- no usable microstructure rows
- do not silently allow
- configurable behavior:
  - default should be `allow but emit degraded warning` in shadow
  - default should be `block` only if explicitly enabled later

### `inactive`

- pocket itself is not active
- event gate should not be the reason trade is blocked
- upstream signal logic simply continues as usual

### `blocked`

- pocket active
- at least one blocking lane fired on the latest bucket

### `allowed`

- pocket active
- no blocking lane fired on the latest bucket

## Scope Matching

The gate must not apply to every entry.

Initial live scope:

- symbol: `ETHUSDT`
- rule: `micro_edge_v3_passive_alpha`
- horizon: `60`
- min abs imbalance target: `0.85`

Suggested explicit matcher:

```python
def applies_to_live_event_gate(symbol: str, rule_name: str, horizon_sec: int) -> bool:
    return (
        symbol.upper() == "ETHUSDT"
        and rule_name == "micro_edge_v3_passive_alpha"
        and int(horizon_sec) == 60
    )
```

## Where To Hook It

Most practical hook points:

1. after signal creation
2. before order sizing / placement

In current codebase, likely candidates:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py)
  - around the point where `sig` is available and before live order routing

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py)
  - if legacy strategy path remains relevant

## Current Code Reality

The repo does not currently expose a clean isolated “entry decision object”.

That means the safest first implementation is:

1. compute the gate payload in a helper
2. inject one narrow conditional block near signal-to-order transition
3. emit one explicit block reason

## Recommended Env Controls

Add env flags before enabling:

- `ENTRY_EVENT_LANE_GATE_ENABLED`
- `ENTRY_EVENT_LANE_GATE_SHADOW`
- `ENTRY_EVENT_LANE_GATE_DB`
- `ENTRY_EVENT_LANE_GATE_LOOKBACK_MIN`
- `ENTRY_EVENT_LANE_GATE_BUCKET_SEC`

Suggested defaults:

- shadow on first
- enabled off by default

## Logging / Telemetry Requirements

At minimum emit:

- `symbol`
- `rule_name`
- `horizon_sec`
- `gate_status`
- `gate_reason`
- `blocking_lanes`
- `latest_abs_imbalance`
- `latest_ts_ms`

Recommended logical event name:

- `entry.event_lane_gate`

Recommended blocked reason:

- `event_lane_gate_blocked`

## Safe Port Order

1. implement helper
2. shadow mode only
3. compare live decisions vs research CLI
4. activate narrow block
5. review impact

## Hard Constraint

Port only current-bucket lane logic.

Do not accidentally import monitoring policy into execution:

- no watchboard summary
- no lane suppression
- no merged banner logic
- no stale severity-only blocking

That would create false blocks.

## Implementation Recommendation

Best next implementation is a small helper in live repo, not a broad refactor.

Reason:

- current live entry path is large and safety-sensitive
- small helper + narrow insertion point minimizes blast radius
- the research bridge already proved the required contract
