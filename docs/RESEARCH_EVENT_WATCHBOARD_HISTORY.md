# Research Event Watchboard History

## Purpose

`event_watchboard_snapshot_append` appends each watchboard snapshot into a JSONL history.

This enables:

- real 1h / 6h / 24h trend slices
- lane switch tracking
- later automation without changing the core watchboard contract

## Tool

- `python -m tools.event_watchboard_snapshot_append`

## Output

- appends one JSON object per snapshot to:
  - `reports/RESEARCH_EVENT_WATCHBOARD_HISTORY.jsonl`
- emits a small append receipt JSON with `run_summary`

## Why This Matters

Trend logic without history is only a contract test.

Trend logic with accumulated snapshots becomes an actual operator signal.
