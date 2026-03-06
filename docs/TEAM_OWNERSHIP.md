# Team Ownership

This file is the operational ownership map for two-person parallel work.

## Person 1

Primary lane: `research`

Owned paths:

- `data/`
- `features/`
- `strategies/`
- `tools/`
- `tests/`
- `docs/ROADMAP_PARALLEL_LAYERS.md`

Primary responsibilities:

- microstructure data collection and validation
- canonical integrity and research dataset quality
- feature generation and signal work
- backtest, sweep, calibration, and research reporting
- research test coverage and deterministic tooling

Optional extension areas when coordinated:

- research-facing `scripts/`
- research-facing `docs/`
- selected `config/` changes tied to data or signal behavior

## Person 2

Primary lane: `runtime`

Owned paths:

- `execution/`
- `risk/`
- `bot/`
- `exchanges/`
- `notifications/`
- `dashboard/`
- `monitoring/`

Primary responsibilities:

- order execution and entry/exit flow
- runtime risk controls and kill-switches
- exchange connectivity and runtime resilience
- operations, health checks, alerts, and dashboards

## Shared

These paths require coordination:

- `config/`
- `docs/`
- `README.md`
- `.github/`
- `scripts/`

Shared change rules:

- keep them in dedicated PRs when possible
- mention both lanes in the issue or PR
- avoid mixing shared edits with large feature diffs

## Escalation Rules

- If Person 1 needs `execution/` or `risk/` changes, open a `shared` issue first.
- If Person 2 needs `tools/` or `tests/` changes for runtime validation, keep the change minimal and note the reason in the PR.
- If a task crosses data-to-runtime boundaries, split it into two PRs unless the interface change is trivial.

## Allowed Cross-Lane Touches

Allowed with explicit note in PR:

- Person 1 updating runtime-facing docs
- Person 2 adding narrow tests for runtime fixes
- either person editing `config/` for a coordinated rollout

Not allowed without coordination:

- mixing signal changes and execution rewrites in one PR
- overwriting another lane's experiment artifacts
- silent config changes affecting both lanes
