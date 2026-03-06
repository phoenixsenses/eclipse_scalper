# FRONTEND_ROADMAP

## Current State (Grounded In Repo)

### Repo Areas Inspected
- Frontend: `dashboard/frontend`
- Backend API serving dashboard: `dashboard/backend`
- Dashboard run scripts: `tools/run_dashboard_backend.ps1`, `tools/run_dashboard_frontend.ps1`
- CI workflows: `.github/workflows/ci-tests.yml`, `.github/workflows/telemetry-dashboard.yml`, `.github/workflows/telemetry-smoke.yml`
- Existing research artifacts for sweep companion: `reports/*.md`, `reports/*.json`, `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`

### Frontend Architecture Summary
- Framework: React 18 (`react`, `react-dom`)
- Language: TypeScript (`strict: true`)
- Build tool: Vite 5 (`vite`, `@vitejs/plugin-react`)
- Routing: `react-router-dom` (BrowserRouter + lazy routes)
- State management: local component state only (`useState`, `useEffect`), no global state library
- Data fetching: native `fetch` in `src/api/client.ts` (no retry/abort/cache layer)
- Realtime transport:
  - Polling: Overview (`/api/overview` every 10s, `/api/runtime` every 2s), Trades (every 15s)
  - SSE: Logs page via `EventSource` to `/api/logs/stream`
- Styling: global CSS + inline style objects; no design system/tokens package beyond CSS vars in `src/index.css`

### Frontend File Map
- Entry: `dashboard/frontend/src/main.tsx`
- Router shell: `dashboard/frontend/src/App.tsx`
- Layout/nav: `dashboard/frontend/src/components/Layout.tsx`
- API client: `dashboard/frontend/src/api/client.ts`
- Pages:
  - `dashboard/frontend/src/pages/Overview.tsx`
  - `dashboard/frontend/src/pages/Logs.tsx`
  - `dashboard/frontend/src/pages/Trades.tsx`
  - `dashboard/frontend/src/pages/Settings.tsx`

### Backend API Surface Used By UI
Defined in `dashboard/backend/app.py` and read by `dashboard/frontend/src/api/client.ts`:
- `GET /api/overview`
- `GET /api/runtime`
- `GET /api/scoreboard`
- `GET /api/gates`
- `GET /api/passive-profiles`
- `GET /api/events/regimes`
- `GET /api/events/signals`
- `GET /api/events/stability`
- `GET /api/events/quality`
- `GET /api/logs`
- `GET /api/logs/tail`
- `GET /api/logs/stream` (SSE)
- `GET /api/config`
- `GET /api/health`

### Current Quality/Tooling State
- No frontend test framework configured (`vitest/jest/playwright` absent)
- No frontend lint/format config (`eslint/prettier` absent)
- No frontend CI job (workflows are Python/telemetry focused)
- No OpenAPI-generated frontend types; API responses are all `any`
- Frontend `node_modules/` is committed under `dashboard/frontend/node_modules` (repository hygiene issue)

---

## Top 20 Issues / Opportunities

1. `src/api/client.ts` returns `any` for every endpoint; no contract safety.
2. No centralized error taxonomy (network down vs 500 vs malformed payload).
3. No `AbortController` in polling loops; request overlap/race is possible.
4. Polling is scattered per page; no shared realtime lifecycle policy.
5. SSE reconnect/backoff strategy is not explicit in UI (logs stream silently drops to non-live state).
6. Loading/error/empty states are inconsistent across pages.
7. `Overview.tsx` is monolithic and mixes data shaping + presentation + status logic.
8. Hardcoded regime badge mapping assumes lowercase `trending|volatile|ranging`, but backend sources can emit other labels.
9. UI text encoding artifacts are present (`Loading…`, `●`), indicating encoding/render pipeline issues.
10. No typed runtime schema validation at boundary (unsafe assumptions on backend payload shape).
11. No frontend env contract (`VITE_*`) for API base URL, feature flags, refresh intervals.
12. Vite proxy target is hardcoded (`http://localhost:8765`) and not environment-driven.
13. No route-level error boundary; one thrown render error can blank route subtree.
14. No accessibility baseline (focus states, aria-live for stream status, keyboard semantics audit missing).
15. No responsive layout strategy beyond flex wrap; dashboard density may degrade on smaller widths.
16. No performance instrumentation (render cost, network timing, stale-data age indicator in UI state).
17. Backend models (`dashboard/backend/models.py`) exist but are not enforced in route decorators (`response_model` missing).
18. `dashboard/backend/data_sources.py` performs mixed parsing and formatting logic without explicit versioned response schema.
19. No frontend bundle guardrails (size budgets, chunk analysis, CI build checks).
20. No frontend-specific developer scripts for lint/typecheck/test smoke in one command.

---

## Backend Contract Assumptions Hardcoded In UI

From `dashboard/frontend/src/pages/*.tsx` and `src/api/client.ts`:

- `overview.scoreboard.paper_trading` is boolean-like and present.
- `overview.scoreboard.fills_total|orders_total|blocked_total|kill_switch_trips_total|circuit_breaker_trips_total` are numeric.
- `overview.gates.symbols[]` rows contain `symbol`, `rule_name`, `hit_rate`, `delta_vs_baseline`.
- `overview.recent_regimes[]` rows contain `ts`, `symbol`, `timeframe`, `effective_regime`, `confidence`.
- `runtime.data_freshness.status` is one of `LIVE|DEGRADED|STALE`.
- `runtime.data_freshness.seconds_since_last_trade` is numeric.
- `runtime.collector.alive`, `trades_per_sec_60s`, `mark_per_sec_60s`, `liquidations_per_sec_60s`, `uptime_sec`, `last_log_ts` exist.
- `runtime.database.size_bytes`, `growth_bytes_5min` exist.
- Events endpoints return arrays sortable/reversible as plain JSON rows.
- `signals` rows include `ts_utc`, `symbol`, `regime`, `direction`, `allow_entry`, `confidence`, `reason`.
- `stability` rows include `ts`, `symbol`, `signal_type`, `allowed`, `streak`, `reason`.
- `quality` rows include `ts`, `symbol`, `timeframe`, `severity`, `issues`.
- `logs` endpoint returns `[{name, size_bytes, ...}]`; `log tail` returns `{lines: string[]}`.
- SSE stream emits line payloads as `data: <text>` and occasional keepalive comments.
- `config` endpoint returns rows with `source` values `env_file` and `runtime_env`.
- `health.status` returns string and `ok` means healthy.

These assumptions should be formalized and versioned.

---

## Target State (Production-Ready Dashboard)

- Typed end-to-end contracts (backend response models + frontend TS interfaces + runtime guards).
- Unified data access layer with retry, timeout, cancellation, stale-state handling, and observability.
- Reliable realtime UX under backend downtime (degraded banners, last-success timestamps, auto-reconnect policy).
- Test coverage for critical rendering and contract parsing paths.
- CI checks for frontend build, typecheck, lint, and targeted tests.
- Research Sweep Companion integrated as first-class dashboard features (run list, compare, attribution digest, report links).
- Reproducibility metadata visible in UI (`run_id`, `git_commit`, `seeds`, `params`, source report path).

---

## Minimum Lovable Dashboard (MLD)

MLD based on current UI should include:
- Overview:
  - Runtime status (freshness + collector/db stats)
  - Trading scoreboard cards
  - Top gate rows + regime transitions
  - Last update timestamps and stale-data warning
- Logs:
  - File list
  - Tail viewer
  - Live SSE with reconnect/backoff indicator
- Trades:
  - Signals/stability/quality tabs
  - Symbol filter
  - Error + empty states per tab
- Settings:
  - Health
  - Masked config visibility
  - Runtime overrides
- Research Companion:
  - Latest rank/sweep run list from `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`
  - Top pocket summary table from selected ranking JSON
  - Direct links to report markdown/json artifacts

---

## Phased Roadmap

### Phase 0 - Stabilize Contracts And Runtime Resilience
**Goal**
Establish strict API/UI contracts and eliminate silent runtime failure modes.

**Scope**
Frontend client layer + backend response model enforcement, no page redesign yet.

**Task checklist**
- [ ] Add typed API interfaces in `dashboard/frontend/src/api/types.ts`.
- [ ] Replace `any` return types in `dashboard/frontend/src/api/client.ts`.
- [ ] Add shared fetch wrapper with timeout, abort, retry policy (idempotent GET only).
- [ ] Add `response_model=` to FastAPI routes in `dashboard/backend/app.py` using `dashboard/backend/models.py`.
- [ ] Normalize server error shape for frontend consumption.
- [ ] Add `VITE_API_BASE_URL` support; keep `/api` default.

**Acceptance criteria**
- TypeScript build passes with zero implicit `any` in API layer.
- Backend OpenAPI schema reflects concrete response models.
- Frontend shows explicit degraded/error states on backend outage.

**Risks + mitigations**
- Risk: strict typing reveals inconsistent backend payloads.
- Mitigation: add compatibility mapper in client layer with telemetry logs.

**Complexity**
M

---

### Phase 1 - Realtime Reliability And UX States
**Goal**
Make polling/SSE behavior reliable and observable under failure.

**Scope**
Overview, Trades, Logs pages; shared realtime hooks.

**Task checklist**
- [ ] Introduce `usePollingQuery` hook with abort-on-unmount and interval drift control.
- [ ] Introduce `useLogStream` hook with reconnect backoff + max retries + status enum.
- [ ] Add per-card/page last-success timestamp and stale age label.
- [ ] Add explicit empty/error/loading components reused across pages.
- [ ] Add non-blocking toast/banner for degraded mode.

**Acceptance criteria**
- Dropping backend process yields user-visible stale/degraded status within 5s.
- SSE reconnect attempts visible in Logs page status.
- No React state updates after unmount warnings.

**Risks + mitigations**
- Risk: aggressive polling load.
- Mitigation: central interval config + backoff on repeated failures.

**Complexity**
M

---

### Phase 2 - Frontend Quality Baseline (DX + CI)
**Goal**
Add minimum engineering guardrails for frontend code quality.

**Scope**
Tooling only; no functional changes.

**Task checklist**
- [ ] Add ESLint + TypeScript ruleset for frontend folder.
- [ ] Add Prettier (or keep no formatter but enforce lint style rules).
- [ ] Add `npm` scripts: `typecheck`, `lint`, `test`.
- [ ] Add Vitest + React Testing Library for unit/component tests.
- [ ] Add CI job for frontend build/typecheck/test.
- [ ] Update `.gitignore` and remove committed frontend `node_modules` from VCS.

**Acceptance criteria**
- CI fails on type/lint/test regression in frontend.
- Fresh clone + `npm ci` works without pre-existing node_modules.

**Risks + mitigations**
- Risk: lint rollout noise.
- Mitigation: staged lint rules; start with errors that prevent runtime bugs.

**Complexity**
M

---

### Phase 3 - Research Sweep Companion (Production Utility)
**Goal**
Turn existing report artifacts into actionable dashboard workflows.

**Scope**
New frontend page(s) + lightweight backend endpoints over `reports/` outputs.

**Task checklist**
- [ ] Add backend endpoint to list ranking/sweep runs from `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`.
- [ ] Add endpoint to fetch summarized top pockets from selected rank JSON.
- [ ] Build frontend page `Research` with:
  - [ ] run list filters (rule/symbol/date/mitigation profile)
  - [ ] compare two runs (core/stress pass rate, npa, failure reasons)
  - [ ] links to md/json artifacts
- [ ] Surface reproducibility tags: run_id, git_commit, seeds, splits, fee/adverse grid.

**Acceptance criteria**
- Operator can compare two run_ids in UI without shell.
- Links open exact artifact files for audit.

**Risks + mitigations**
- Risk: large JSON parse cost.
- Mitigation: backend aggregation endpoint with pagination/top-N slicing.

**Complexity**
L

---

### Phase 4 - Architecture Refactor (Incremental, Not Rewrite)
**Goal**
Reduce page complexity and improve maintainability.

**Scope**
Frontend structure and shared modules.

**Task checklist**
- [ ] Introduce feature folders:
  - [ ] `src/features/overview/*`
  - [ ] `src/features/logs/*`
  - [ ] `src/features/trades/*`
  - [ ] `src/features/settings/*`
  - [ ] `src/features/research/*`
- [ ] Move shared primitives to `src/ui/*`.
- [ ] Move API hooks and mappers to `src/data/*`.
- [ ] Add route-level error boundaries.
- [ ] Keep route paths stable during migration.

**Acceptance criteria**
- `Overview.tsx` no longer monolithic (>400 lines split into composable units).
- No route behavior regression.

**Risks + mitigations**
- Risk: refactor churn.
- Mitigation: page-by-page migration with snapshot tests.

**Complexity**
M

---

### Phase 5 - Production Hardening
**Goal**
Make dashboard robust for 24/7 operations and incident response.

**Scope**
Observability, security boundaries, operational UX.

**Task checklist**
- [ ] Add structured frontend telemetry events (no secrets).
- [ ] Add circuit-state UI (backend reachable, stale, degraded, recovering).
- [ ] Add read-only mode banner if backend health not `ok`.
- [ ] Add API response version field and UI compatibility checks.
- [ ] Document env vars and deployment profile for LAN/remote access.

**Acceptance criteria**
- Downtime scenarios are visible, actionable, and auditable in UI.
- No secrets rendered from config endpoint.

**Risks + mitigations**
- Risk: over-instrumentation noise.
- Mitigation: bounded event schema with sampling.

**Complexity**
M

---

## Research Sweep Companion Plan

### Existing Artifacts To Integrate
- Registry: `reports/RUN_RANK_SWEEP_REGISTRY.jsonl`
- Rank outputs: `reports/RANK_*.json`, `reports/PASSIVE_POCKET_RANKING*.json`
- Validation outputs: `reports/FWD_*.md`
- Sweep outputs: `reports/FILTER_SWEEP_*.md`

### UI Capability Roadmap
- Run Explorer:
  - Parse registry lines, show latest N runs
  - Filter by symbol/rule/mitigation profile
- Run Compare:
  - Compare `pass_rate_core`, `pass_rate_stress`, `npa_core`, `score_raw_core`
  - Highlight failure attribution (`fees_dominate`, `adverse_dominates`, `gate_reject`, `mixed`)
- Pocket Drilldown:
  - Show per-pocket gate config and capacity metrics
  - Link to source markdown/json report files

---

## Folder Structure Refactor Plan (Incremental)

Current frontend is small and workable, but refactor is warranted as features expand.

### Proposed Structure
```text
dashboard/frontend/src/
  app/
    router.tsx
    providers.tsx
  data/
    apiClient.ts
    contracts.ts
    mappers.ts
    hooks/
  features/
    overview/
    logs/
    trades/
    settings/
    research/
  ui/
    components/
    states/
  styles/
    tokens.css
    globals.css
```

### Migration Strategy
1. Add new folders without moving existing pages.
2. Extract API layer first (`api/client.ts` -> `data/apiClient.ts`).
3. Extract reusable UI state components.
4. Migrate each page one at a time; keep route paths unchanged.
5. Remove legacy files after parity tests pass.

---

## Proposed API Contract Updates (REST + SSE)

### REST Improvements
- Add response envelope consistency:
```json
{ "ok": true, "data": { ... }, "meta": { "generated_utc": "...", "api_version": "v1" } }
```
- Add stable error envelope:
```json
{ "ok": false, "error": { "code": "UPSTREAM_UNAVAILABLE", "message": "..." } }
```
- Add `run_id`, `git_commit`, and freshness metadata where relevant.
- Add dedicated endpoint for research run registry summary.

### SSE Improvements
Current `/api/logs/stream` emits plain lines only. Add optional typed events:
```text
event: line
id: <offset_or_ts>
data: {"line":"...","file":"..."}

event: status
data: {"state":"reconnecting","retry_ms":1000}
```
Frontend should support both legacy plain-line and typed mode for compatibility.

---

## First PR Plan (Concrete)

### PR Goal
Introduce typed API contracts + shared fetch resilience without changing page behavior.

### Exact files to touch
- `dashboard/frontend/src/api/client.ts`
- `dashboard/frontend/src/api/types.ts` (new)
- `dashboard/frontend/src/api/fetcher.ts` (new)
- `dashboard/frontend/src/pages/Overview.tsx` (minimal type usage + error state normalization)
- `dashboard/backend/app.py` (add `response_model=` where practical)
- `dashboard/backend/models.py` (align/extend models for used endpoints)

### Exact changes
- Replace `any` signatures with typed interfaces.
- Add fetch timeout + explicit error class mapping.
- Preserve existing endpoints and polling cadence.
- Add minimal compatibility mappers where backend fields are optional.

### Commands to run
```powershell
cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\dashboard\frontend"
npm run build

cd "C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper"
python -m py_compile dashboard\backend\app.py dashboard\backend\models.py dashboard\backend\data_sources.py
python -m pytest -q
```

### Local verification checklist
- `http://localhost:5173/` renders Overview with data.
- Kill backend; UI shows degraded/error state instead of hanging silently.
- Restart backend; polling recovers and stale indicators clear.
- Logs page SSE still streams.

---

## Parallel Work Matrix

| Workstream | Owner | Dependency | Notes |
|---|---|---|---|
| Phase 0 typed API contracts | Codex | none | Safe while sweeps run |
| Frontend lint/test setup | friend | none | Separate PR to avoid blocking feature work |
| Research Companion backend endpoint | me | requires sweep results | Needs stable registry schema |
| Research Companion UI | friend | requires sweep results | Start with mock fixture first |
| Realtime hooks refactor | Codex | none | No sweep dependency |
| API envelope/versioning | me | none | Coordinate backend + frontend rollout |
| Artifact path normalization | me | requires data rebuild (optional) | If run layout changes later |

---

## Appendix A - Env Vars (Frontend/Backend)

### Existing
- Backend runtime scripts:
  - `DASHBOARD_PORT`
  - `DASHBOARD_HOST`
  - `LOG_DIR` (consumed in backend data sources)
  - `MICROSTRUCTURE_DB_PATH`
  - `COLLECTOR_LOG_PATH`

### Proposed frontend vars
- `VITE_API_BASE_URL` (default `/api`)
- `VITE_POLL_RUNTIME_MS` (default 2000)
- `VITE_POLL_OVERVIEW_MS` (default 10000)
- `VITE_POLL_TRADES_MS` (default 15000)
- `VITE_ENABLE_RESEARCH_COMPANION` (`0|1`)

---

## Appendix B - Reliability Under Backend Downtime

Mandatory UX behavior:
- Show stale/degraded state within one polling interval.
- Preserve last good payload with timestamp.
- Retry with bounded backoff.
- Never spin forever with blank page.

---

## Appendix C - Known Missing Pieces (Explicit)

- No frontend automated tests currently exist.
- No frontend lint/format enforcement currently exists.
- No frontend CI stage currently exists.
- No typed OpenAPI client generation currently exists.
- No dedicated Research page currently exists.

These are intentional roadmap targets, not assumptions.
