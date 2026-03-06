# GitHub Collaboration System

This repository now uses a two-lane workflow:

- `research`: Person 1 owns data, feature, strategy, tool, and test work.
- `runtime`: Person 2 owns execution, risk, exchange, dashboard, and monitoring work.
- `shared`: config, docs, workflows, and README changes.

## Branch Model

Long-lived local lane branches:

- `codex/research-mainline`
- `codex/runtime-mainline`
- `codex/shared-mainline`

Short-lived task branches:

- `codex/research/<topic>`
- `codex/runtime/<topic>`
- `codex/shared/<topic>`

Rules:

- Person 1 branches from `codex/research-mainline`.
- Person 2 branches from `codex/runtime-mainline`.
- Shared work branches from `codex/shared-mainline`.
- Do not commit feature work directly on the `*-mainline` branches.
- Do not mix `research` and `runtime` changes in one PR unless the change is deliberately `shared`.
- Do not force-push shared coordination branches.

Current owners:

- Person 1: `@phoenixsenses`
- Person 2: `@emresavaser`

## Pull Request Lanes

Use one PR per lane:

- `research`: data, features, strategies, research tools, tests
- `runtime`: execution, risk, bot, exchanges, notifications, dashboard, monitoring
- `shared`: docs, config, GitHub workflow, README

PR requirements:

- branch name matches lane
- affected paths match lane
- exact validation commands are included
- rollback path is written for runtime changes
- shared paths are called out explicitly

## Review Policy

- `research` PRs: Person 1 self-review plus targeted review if shared paths are touched.
- `runtime` PRs: Person 2 review required before merge.
- `shared` PRs: both lanes review if the change affects both workflows.
- `config`, `.github`, `README.md`, and top-level `docs/` changes should be treated as coordination changes.

## Suggested Labels

- `track:research`
- `track:runtime`
- `track:shared`
- `risk:low`
- `risk:med`
- `risk:high`
- `needs-smoke`
- `needs-data`
- `breaking-config`

## Merge Gates

Minimum merge gate by lane:

- `research`
  - targeted tests
  - `python -m tools.smoke_all --db data/definitely_missing_for_smoke.db`
- `runtime`
  - targeted tests
  - relevant operational smoke
  - rollback note
- `shared`
  - docs or config validation
  - no cross-lane accidental edits

## Shared Path Rules

These paths are not lane-private:

- `config/`
- `docs/`
- `README.md`
- `.github/`
- `scripts/`

If one of these is touched:

- say so in the PR summary
- keep the diff minimal
- avoid bundling unrelated lane work into the same change

## Report and Artifact Hygiene

- Never overwrite active `reports/_runs/*` artifacts from another worktree.
- Prefer lane-specific output names for experiments.
- Keep machine outputs in JSON or JSONL where possible.
- If a tool emits JSON, keep `run_summary` populated.

## Recommended Flow

```powershell
git checkout codex/research-mainline
git checkout -b codex/research/microstructure-alignment
python -m tools.smoke_all --db data/definitely_missing_for_smoke.db
git status
```

After merge, fast-forward the lane branch:

```powershell
git checkout codex/research-mainline
git merge --ff-only codex/research/microstructure-alignment
```
