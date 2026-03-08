# Research Event Watchboard Cycle

## Purpose

`run_research_event_watchboard_cycle` is the operational wrapper for the full event watchboard loop.

It runs:

1. watchboard build
2. snapshot append
3. trend-from-history
4. operator brief build

## Why This Matters

This is the first command in the stack that is directly automation-ready.

Instead of calling three tools manually, the cycle command emits:

- the latest watchboard
- an appended history row
- an updated trend-from-history summary
- an operator-facing brief
- a small cycle receipt
- bounded history growth via `--max-history`

## Tool

- `python -m tools.run_research_event_watchboard_cycle`
