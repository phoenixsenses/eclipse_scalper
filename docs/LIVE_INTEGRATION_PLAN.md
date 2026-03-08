# Live Integration Plan

## Scope

This document describes the minimum path to take the research-side event lane gate into the live bot without touching the ranking research loop at runtime.

Target live behavior:

- live entry flow can query current event lane state from the microstructure DB
- the target pocket can be blocked in real time when event lanes are active
- the gate is explicit, observable, and reversible
- rollout is narrow and scoped to the validated live candidate only

Initial target pocket:

- `ETHUSDT`
- rule family: `micro_edge_v3_passive_alpha`
- horizon: `60`
- event block lanes:
  - `book_proxy_pressure`
  - `volatility_burst`

## Current Wiring

### Live entry path

The live repo does not have a small isolated `entry_decision.py` or `entry_primitives.py`.
Decision flow is currently concentrated in:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py)
- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py)

There are effectively two signal paths:

1. legacy strategy path
- `strategies.eclipse_scalper.scalper_signal`

2. micro signal path
- `core.micro_signal`
- built in `entry_loop.py` via `_build_micro_signal_provider(...)`
- adapted into entry actions by `_micro_signal_to_entry_sig(...)`

### Research-side pocket definition

Pocket-oriented research/live-ish strategy wiring exists in:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\strategies\micro_edge_pocket.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\strategies\micro_edge_pocket.py)

Research validation/ranking wiring exists in:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\validate_passive_pocket_forward.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\validate_passive_pocket_forward.py)
- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\rank_passive_pockets_forward.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\tools\rank_passive_pockets_forward.py)

### Event lane gate bridge

Research already produced a live-facing diagnostic CLI:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py)

That tool currently:

- reads the live microstructure DB
- computes current `book_proxy_pressure` and `volatility_burst`
- checks the latest bucket against the target pocket
- outputs `allowed` vs `blocked`

This is the correct bridge shape.

## Recommended Integration Shape

Do not port research ranking logic into the entry loop.

Do:

1. Keep ranking and validation offline
2. Keep event lane computation synchronous and narrow
3. Add one explicit gate point in live entry

Recommended live gate flow:

1. Entry loop gets a candidate signal
2. Candidate is mapped to a known pocket scope
3. If pocket scope matches the promoted live profile, query event lane state
4. If current bucket is blocked, emit a structured `entry_blocked` reason and stop
5. If current bucket is allowed, continue with existing execution guards

## Narrow Rollout Rule

Do not make event gating global.

Initial rollout should be:

- symbol: `ETHUSDT`
- rule scope: `micro_edge_v3_passive_alpha`
- horizon: `60`
- gate lanes:
  - `book_proxy_pressure`
  - `volatility_burst`

Anything broader is not justified by current evidence.

## Integration Options

### Option A: CLI bridge

Call `tools.check_event_lanes` from live code and parse its JSON output.

Pros:

- lowest coupling
- reuses the existing bridge immediately
- easy to observe and debug

Cons:

- subprocess overhead
- less elegant for high-frequency use

### Option B: port the core gate logic into a small live helper

Create a live helper such as:

- `execution/event_lane_gate.py`

and port only the minimum logic:

- current bucket lane evaluation
- target pocket match
- allow/block decision

Pros:

- no subprocess
- easier operational use in the live loop

Cons:

- code duplication unless carefully isolated

### Recommendation

Start with Option A for readiness testing.
Move to Option B only after behavior is validated in shadow mode.

## Observability Requirements

Live integration is not acceptable unless it emits structured reasons.

Minimum required fields on block:

- symbol
- rule scope
- horizon
- pocket id
- blocking lanes
- latest bucket timestamp
- latest imbalance
- lane severities
- gate decision source

Minimum block reason string:

- `event_lane_gate_blocked`

Secondary details:

- `blocking_lanes=book_proxy_pressure,volatility_burst`

## Rollout Stages

### Stage 0: Analysis

Done here.

### Stage 1: Shadow mode

Add live gate evaluation, but do not block orders yet.
Only emit:

- would-block
- blocking lanes
- bucket context

Goal:

- verify event lane timing is sensible against real live flow
- confirm stale summary does not cause false blocks

### Stage 2: Narrow active gate

Activate blocking only for:

- `ETHUSDT`
- `micro_edge_v3_passive_alpha`
- `h=60`

Keep a kill switch:

- `ENTRY_EVENT_LANE_GATE_ENABLED=1`
- `ENTRY_EVENT_LANE_GATE_SHADOW=0|1`

### Stage 3: Review

Review:

- signal count lost
- blocked count
- false positive suspicion
- realized execution quality after gate

Only then consider broader rollout.

## Critical Design Constraints

1. Gate on current bucket tags, not stale summary state

This is important.
The bridge tool already showed the correct behavior:

- lane summary may be `severe`
- but if current bucket is clean, the trade should still be allowed

2. Do not gate on historical trend/watchboard summaries

Those are for monitoring and operator context, not real-time execution control.

3. Keep the scope explicit

No hidden “if micro signal then gate everything” behavior.

4. Keep the gate reversible by env flag

This is a safety-critical codebase.

## Files To Touch In A Future Implementation

Likely implementation touch points:

- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry_loop.py)
- [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-live-plan\execution\entry.py)

Preferred new helper:

- `execution/event_lane_gate.py`

Optional transitional bridge:

- call [C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py](C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-check-event-lanes\tools\check_event_lanes.py)

## Explicit Non-Goals

Not part of the live integration task:

- ranking in live loop
- event watchboard rendering
- merged banner logic
- broad multi-symbol gating
- BTC rollout
- long-window event-block promotion

## Exit Criteria

Ready for deploy trial means:

1. shadow-mode gate emits correct decisions
2. current-bucket gate matches the research bridge contract
3. live entry loop can block the target pocket cleanly
4. env kill switch exists
5. all blocked entries are observable with explicit reasons
