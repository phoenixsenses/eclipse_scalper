# Runbook: Safe Parallel Work During Long Sweeps

## Scope
This runbook covers safe development while `tools.rank_passive_pockets_forward` sweeps are running. It is additive-only and avoids corrupting active experiment outputs.

## Safety Rules
- Do not mutate `reports/_runs/` outputs produced by active sweeps.
- Do not overwrite `reports/RUN_RANK_SWEEP_REGISTRY.jsonl` from another branch/worktree.
- Do not run destructive cleanup commands on `reports/`, `logs/`, or `state/`.
- Keep research outputs branch-local when possible (new filenames or new run folder).

## Branch Strategy
- Create one feature branch per layer:
  - `feat/data-integrity-*`
  - `feat/signal-guards-*`
  - `feat/passive-diagnostics-*`
  - `feat/reporting-schema-*`
- Rebase frequently on main branch tip.
- Never force-push shared branches used by multiple contributors.

PowerShell:
```powershell
git checkout -b feat/passive-diagnostics-v2
git status
```

## Determinism Policy
- Always provide explicit seeds for split-based tools:
  - `--seeds 7,11,22,33,44,55,66,77,88`
- Keep split count explicit:
  - `--splits 3` or `--splits 6` (no implicit defaults in experiments)
- Keep cost defaults explicit if comparing runs:
  - `--maker-fee-bps-grid ...`
  - `--passive-adverse-mult-grid ...`
- Prefer stable hash-based run folders (`tools/run_rank_sweep.py`).

## Logging Policy
- Structured JSON/JSONL only for machine outputs.
- Required registry fields:
  - `run_id`, `timestamp_utc`, `args`, `outputs`, `summary`
- Use append-only logs for sweep registries.
- Keep stderr/stdout readable and concise; avoid large dumps in normal mode.

## Shared Output Hygiene
- Active sweep outputs:
  - `reports/_runs/<run_id>/rank.json`
  - `reports/_runs/<run_id>/rank.md`
  - `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`
- If you need alternative analysis, write new files:
  - `reports/EXPERIMENT_*`
  - `reports/ANALYSIS_*`
  - `reports/test_*` for test artifacts

## Fast Verification (Smoke Suite)
Run before each commit:
```powershell
python -m tools.smoke_all --db data/microstructure.db
pytest -q tests/test_smoke_all.py
python -m py_compile tools/smoke_all.py tools/run_rank_sweep.py tools/rank_passive_pockets_forward.py tools/validate_passive_pocket_forward.py
```

If DB is unavailable:
```powershell
python -m tools.smoke_all --db data/definitely_missing_for_smoke.db
```
Expected: passes with DB check marked as skipped.

## Parallel Execution Commands (Windows)
Dry-run sweep plan:
```powershell
python -m tools.run_rank_sweep `
  --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md `
  --maker-fee-bps-grid 1.0 `
  --passive-adverse-mult-grid 1.0,1.2,1.5 `
  --vol-quantile-reject-grid 0.01 `
  --dry-run
```

Attribution summary:
```powershell
python -m tools.summarize_rank_attribution --in reports/PASSIVE_POCKET_RANKING.json --top-n 20
```

## Paper Watchdog Identity
- `scripts/start_paper_trading.ps1` starts watchdog as a background process and logs `pid`, not PowerShell `Job.Id`.
- PID registry files:
  - `logs/pids/paper_watchdog.pid`
  - `logs/pids/paper_watchdog.json`
- Startup resolves repo root from script path, so it works from non-repo CWD.
- If watchdog already running and identity matches, startup refuses duplicate launch unless `-ForceRestart` is provided.
- Inspect watchdog with:
```powershell
Get-Process -Id <pid>
```
- Stop watchdog with:
```powershell
Stop-Process -Id <pid>
```

## Health Commands
```powershell
python -m tools.health_check
python -m tools.ops_smoke --env .env.paper
python -m tools.health_cycle_smoke
python -m tools.ingestion_check --db data/microstructure.db --symbols ETHUSDT,BTCUSDT --window-sec 10 --max-lag-sec 5
python -m tools.health_check --max-staleness-sec 15
```
- `health_check` exits `0=ok`, `1=degraded`, `2=halted/missing`.
- `ops_smoke` performs quick deterministic ops checks and redacts sensitive values from output.
- `health_cycle_smoke` runs offline collector simulation and asserts `ok -> degraded -> ok`.
  - Snapshots are written to `logs/health/smoke/`:
    - `overall_snapshot1_ok.json`
    - `overall_snapshot2_degraded.json`
    - `overall_snapshot3_ok.json`
  - Exit codes: `0=success`, `1=assertion/transition failure`, `2=runtime error`.
- `ingestion_check` proves DB is actively ingesting and fresh.
  - Exit codes: `0=OK`, `1=DEGRADED`, `2=ERROR`.
- Paper-trader health gate:
  - Entry loop halts trading decisions when health is missing/stale or collector is disconnected/stale.
  - Escalates to `halted` when reconnect/error thresholds are exceeded.

## Breakage Triage Checklist
- [ ] `tools/smoke_all.py` returns `0`.
- [ ] Targeted tests for touched files pass.
- [ ] No accidental writes to active run folders.
- [ ] `git diff` only includes intended files.
- [ ] JSON outputs still parse and include expected keys.

## If Something Breaks
1. Stop writing to shared report paths.
2. Save failing command and stdout/stderr.
3. Re-run with minimal inputs.
4. Add/adjust regression test first.
5. Patch and validate with smoke + targeted tests.
