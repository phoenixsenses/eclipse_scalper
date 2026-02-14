# Belief State Model

Execution runs on local beliefs while exchange state is delayed, partial, and occasionally contradictory.

## State Components

- `belief_debt_sec`: age of unresolved mismatch evidence.
- `belief_debt_symbols`: count of symbols with active unresolved mismatch evidence.
- `mismatch_streak`: rolling mismatch pressure in reconcile.
- `belief_confidence`: derived confidence proxy from debt age against configured reference.
- `debt_growth_per_min`: slope of debt score increase between controller updates.

## Debt Score

`debt_score` is a weighted composition of age + breadth + mismatch persistence:

- age term: `belief_debt_sec / BELIEF_DEBT_REF_SEC`
- breadth term: `belief_debt_symbols * BELIEF_SYMBOL_WEIGHT`
- persistence term: `mismatch_streak * BELIEF_STREAK_WEIGHT`

`debt_growth_per_min` measures score slope and catches abrupt degradation before absolute score limits.

## Mode State Machine

Modes: `GREEN -> YELLOW -> ORANGE -> RED`

- Upgrade transitions:
  - Score/growth thresholds crossed with persistence window.
  - `RED` transition can be immediate on hard breach.
- Downgrade transitions:
  - Require hysteresis threshold and sustained recovery window.
  - Prevents flapping around boundaries.

## Guard Knobs (Outputs)

- `allow_entries`
- `max_notional_usdt`
- `max_leverage`
- `min_entry_conf`
- `entry_cooldown_seconds`
- `max_open_orders_per_symbol`
- `mode`
- `kill_switch_trip`

These knobs are the single policy surface for entry risk posture.

## Control-Loop Semantics

1. Reconcile computes debt evidence.
2. Belief controller maps evidence to mode + knobs.
3. Entry loop/router consume knobs for entry-only constraints.
4. Exits remain ungated by this surface.
5. Telemetry captures decision trace (`mode`, reason, transition, score, growth) for replay.

## Decay and Recovery

- Debt decays when symbols reconcile cleanly.
- Recovery requires sustained clean evidence, not one clean tick.
- Cooldown and hysteresis make behavior boring under noisy streams.
