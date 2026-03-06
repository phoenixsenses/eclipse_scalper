# Research Event Watchboard Trend From History

## Purpose

`event_watchboard_trend_from_history` reads the JSONL snapshot history and builds a trend payload from the last N snapshots.

This is the practical bridge between:

- `event_watchboard_snapshot_append`
- `event_watchboard_trend`

## Tool

- `python -m tools.event_watchboard_trend_from_history`

## Expected Use

Short term:

- operator trend card from real accumulated history
- quick last-N trend checks without passing snapshot file lists manually
- lane-level delta view from persisted snapshots
