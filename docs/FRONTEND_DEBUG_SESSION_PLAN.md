# Frontend Control + Debug Session Plan

## Scope
- Goal: control and diagnose paper-trading runtime from dashboard without changing signal logic.
- Stack: `dashboard/backend` + `dashboard/frontend`.
- Safety model: backend only executes whitelisted debug actions.

## What Is Implemented
- New backend control module: `dashboard/backend/control_actions.py`
  - Whitelisted actions:
    - `validate_env`
    - `preflight_check`
    - `paper_trade_status`
    - `paper_trade_summary`
    - `db_maintenance`
    - `incident_bundle`
  - History log: `logs/dashboard_debug_actions.jsonl`
  - Feature flag: `DASHBOARD_CONTROL_ENABLED` (default enabled).
- New API endpoints in `dashboard/backend/app.py`:
  - `GET /api/debug/actions`
  - `GET /api/debug/history?limit=...`
  - `POST /api/debug/run` with JSON `{ "action": "<name>" }`
- New frontend page:
  - `dashboard/frontend/src/pages/Debug.tsx`
  - Added to router/nav (`/debug`)
  - Includes:
    - action buttons
    - latest command output panel
    - action history table

## Debug Session Workflow

### Session 0: Baseline
1. Open dashboard `Overview`.
2. Confirm runtime is LIVE/DEGRADED (not dead):
   - collector alive
   - last trade age
   - DB growth/size
3. Open `Debug` page.

### Session 1: Environment/Preflight
1. Run `validate_env`.
2. Run `preflight_check`.
3. If any fail:
   - inspect output panel
   - jump to `Logs` page for correlated errors.

### Session 2: Trading Readiness
1. Run `paper_trade_status`.
2. Run `paper_trade_summary`.
3. Validate:
   - active symbol config
   - no persistent `signal not present` due env/wiring
   - expected gate behavior visible in logs.

### Session 3: Storage/Collector Health
1. Run `db_maintenance`.
2. If collector issues suspected, run `incident_bundle`.
3. Review generated artifacts from command output paths.

## Operational Rules
- Keep `DASHBOARD_CONTROL_ENABLED=1` only on trusted local host.
- Use Debug actions for diagnostics; do not run continuous loops from API.
- Preserve signal logic and pocket filters during 60-day collection.
- For incidents, always preserve latest history + logs before restart.

## Fast Triage Matrix
- `ModuleNotFoundError` at bootstrap:
  - run `validate_env`
  - verify venv/python path
- `preflight_check FAIL DB stale`:
  - confirm collector alive + DB write freshness
  - inspect `logs/microstructure_collector.log`
- `conf=0.00 reason=no_match` continuously:
  - this is market-state dependent if thresholds unmet; verify with no-match detail logs.
- unexpected shutdown:
  - use incident bundle
  - inspect shutdown metadata and latest `ENTRY_LOOP_FULL` logs.

## Next Frontend Iteration
- Add dedicated `Controls` section for:
  - start/stop/restart wrapper scripts (strictly local + explicit guard).
- Add structured parser cards for command outputs (not just raw text).
- Add one-click export of latest debug history + log tail bundle.

## Frontend CI Local Smoke

Run from repo root:

```powershell
cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\dashboard\frontend"
cmd /c npm install
cmd /c npm run typecheck
cmd /c npm test
```

If you get npm cache/permission errors (`EACCES`), reopen PowerShell as Administrator and run:

```powershell
cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\dashboard\frontend"
cmd /c npm cache verify
cmd /c npm install
cmd /c npm run typecheck
cmd /c npm test
```
