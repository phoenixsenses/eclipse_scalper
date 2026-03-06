# Research Event Operator Brief

## Purpose

`research_event_operator_brief` converts the watchboard and trend outputs into a short operator-facing summary.

## Why This Matters

The watchboard and trend payloads are structured for downstream systems. The operator brief is the compressed version:

- one headline
- one operator note
- explicit severe/stale lane lists

## Tool

- `python -m tools.research_event_operator_brief`
