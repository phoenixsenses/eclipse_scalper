# Runtime Event UI Index

## Purpose

This is the single entry point for Person 2 to implement the research
event-intelligence surface in runtime/dashboard.

Use this document before opening individual handoff docs.

## Implementation Order

1. Overview layer
2. Top banner layer
3. Lane cards and watchlists
4. Lane-specific card refinements

## 1. Overview Layer

Primary runtime issue:

- Issue `#7` `Runtime: add research event watchboard overview`

Use in this order:

- `docs/EVENT_WATCHBOARD_EFFECTIVE_HANDOFF.md`
- `docs/EVENT_WATCHBOARD_EFFECTIVE_CONTRACT.json`
- `docs/EVENT_MERGED_BANNER_POLICY_HANDOFF.md`
- `docs/EVENT_MERGED_BANNER_POLICY_CONTRACT.json`
- `reports/EVENT_WATCHBOARD_EFFECTIVE_REAL.json`
- `reports/EVENT_MERGED_BANNER_POLICY_REAL.json`
- `reports/RESEARCH_EVENT_OPERATOR_BRIEF_REAL.json`

Runtime meaning:

- `effective watchboard` decides primary lane emphasis
- `merged banner policy` decides top header behavior
- `operator brief` gives compressed operator wording

## 2. Top Banner Layer

Preferred source:

- `reports/EVENT_MERGED_BANNER_POLICY_REAL.json`

Rules:

- if `summary.banner_mode = merged`, render one combined top banner
- otherwise use the top lane from `effective watchboard`
- do not replace the lane cards; banner is only the top header

## 3. Lane Cards And Watchlists

Implement lane cards from these issues:

- Issue `#3` liquidation
- Issue `#4` spread stress
- Issue `#5` fill toxicity
- Issue `#6` latency stress
- Issue `#8` return shock
- Issue `#9` volume vacuum
- Issue `#10` volatility burst
- Issue `#11` book proxy pressure

Use each lane's:

- `*_HANDOFF.md`
- `*_CONTRACT.json`
- `*_REAL.json`

## 4. Runtime Display Rules

- use `freshness` before action severity
- use `recommended_action` as the UI hint, not as execution logic
- use `effective_display_mode` for lane emphasis
- use persistence fields to avoid fast banner flipping
- keep degraded lanes visible but lower-emphasis
- do not auto-wire any research event payload into execution

## Current Real Priority Snapshot

Current stack from real artifacts:

- effective top lane: `return_shock`
- merged banner mode: `merged`
- merged focus lanes:
  - `return_shock`
  - `book_proxy_pressure`
  - `volatility_burst`

This means the runtime/dashboard side should start from:

1. merged top banner
2. effective overview card
3. individual lane cards/watchlists

## Practical Branch Guidance

Person 2 should work on:

- base branch: `codex/runtime/ops-foundation`
- task branches:
  - `codex/runtime/event-watchboard-overview`
  - `codex/runtime/event-merged-banner`
  - or one lane-specific branch per issue
