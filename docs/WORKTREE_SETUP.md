# Worktree Setup

Use separate worktrees so research and runtime do not fight over local state.

## Recommended Layout

Parent folder example:

```text
C:\Users\Windows 11\.vscode\CryptoLion\
  eclipse_scalper\
  eclipse_scalper-research\
  eclipse_scalper-runtime\
  eclipse_scalper-shared\
```

## One-Time Setup

Create the lane branches:

```powershell
git branch codex/research-mainline
git branch codex/runtime-mainline
git branch codex/shared-mainline
```

Create worktrees:

```powershell
git worktree add ..\eclipse_scalper-research codex/research-mainline
git worktree add ..\eclipse_scalper-runtime codex/runtime-mainline
git worktree add ..\eclipse_scalper-shared codex/shared-mainline
```

Or use the helper:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_parallel_worktrees.ps1
```

## Daily Use

Person 1:

```powershell
cd ..\eclipse_scalper-research
git checkout -b codex/research/<topic>
```

Person 2:

```powershell
cd ..\eclipse_scalper-runtime
git checkout -b codex/runtime/<topic>
```

Shared:

```powershell
cd ..\eclipse_scalper-shared
git checkout -b codex/shared/<topic>
```

Or use the helper from inside the relevant worktree:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\new_lane_branch.ps1 -Lane research -Topic microstructure-validation
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\new_lane_branch.ps1 -Lane runtime -Topic dashboard-health-audit
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\new_lane_branch.ps1 -Lane shared -Topic docs-pr-template-update
```

## What Each Folder Means

- `eclipse_scalper-research`
  - checked out on `codex/research-mainline`
  - use this for your data, feature, strategy, tool, and test work
- `eclipse_scalper-runtime`
  - checked out on `codex/runtime-mainline`
  - your friend should use this for execution, risk, exchange, dashboard, and monitoring work
- `eclipse_scalper-shared`
  - checked out on `codex/shared-mainline`
  - use this only for shared docs, config, `.github`, `README`, and scripts

## Practical Rule

- Do not work directly on `codex/research-mainline`, `codex/runtime-mainline`, or `codex/shared-mainline`.
- Start from the correct worktree.
- Create one short-lived branch per task.
- Merge the task branch back into the matching mainline branch after review.

Example:

```powershell
cd C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper-research
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\new_lane_branch.ps1 -Lane research -Topic orderbook-gap-check
```

This creates and checks out:

```text
codex/research/orderbook-gap-check
```

## State Rules

- Keep lane-specific reports in lane-specific filenames.
- Do not share active `reports/_runs/*` output folders between worktrees.
- Do not run destructive cleanup in `reports/`, `logs/`, `state/`, or `data/` from the wrong worktree.
- Use separate terminals and virtualenv activation per worktree if dependencies diverge.

## Verification

Research lane:

```powershell
python -m tools.smoke_all --db data/definitely_missing_for_smoke.db
```

Runtime lane:

```powershell
python -m tools.health_check
python -m tools.ops_smoke --env .env.paper
```
