# Eclipse Scalper Dashboard

Real-time monitoring dashboard for the Eclipse Scalper trading system.

## Stack

| Layer    | Technology                 |
|----------|----------------------------|
| Backend  | FastAPI + uvicorn (Python) |
| Frontend | React 18 + Vite + TypeScript |
| Transport | REST + Server-Sent Events  |

## Quick start

Open two terminals from the repo root:

**Terminal 1 — backend**
```powershell
.\tools\run_dashboard_backend.ps1
```
Backend runs at `http://localhost:8765`.
API docs: `http://localhost:8765/api/docs`

**Terminal 2 — frontend**
```powershell
.\tools\run_dashboard_frontend.ps1
```
App runs at `http://localhost:5173`.

**One-command startup (backend + frontend)**
```powershell
.\tools\run_dashboard.ps1
```

## Pages

| Page      | URL          | Description |
|-----------|--------------|-------------|
| Overview  | `/`          | Scoreboard stats, alpha gates, recent regime transitions, preflight & reliability |
| Logs      | `/logs`      | Browse `logs/` files; tail static or stream live via SSE |
| Trades    | `/trades`    | Signal events, stability events, data quality events — filterable by symbol |
| Settings  | `/settings`  | `.env` config viewer (sensitive values masked), runtime overrides, backend health |

## Data sources

All reads are read-only and gracefully degrade (missing files return empty data).

| Source | Used by |
|--------|---------|
| `state/paper_scoreboard.json` | Overview stats |
| `state/micro_edge_gates.json` | Alpha gates table |
| `state/passive_realistic_profiles.json` | Available via `/api/passive-profiles` |
| `logs/alpha_gate.jsonl` | Trades → Signals |
| `logs/signal_stability.jsonl` | Trades → Stability |
| `logs/regime_transitions.jsonl` | Overview + Trades |
| `logs/data_quality.jsonl` | Trades → Quality |
| `logs/exit_quality_summary.json` | Overview |
| `logs/preflight_check.json` | Overview |
| `logs/reliability_gate.txt` | Overview |
| `.env` | Settings (masked) |

## Backend API endpoints

```
GET /api/health
GET /api/overview
GET /api/scoreboard
GET /api/gates
GET /api/passive-profiles
GET /api/events/regimes?limit=100&symbol=ETHUSDT
GET /api/events/signals?limit=100&symbol=ETHUSDT
GET /api/events/stability?limit=100
GET /api/events/quality?limit=100
GET /api/logs
GET /api/logs/tail?file=alpha_gate.jsonl&limit=200
GET /api/logs/stream?file=alpha_gate.jsonl&last_n=50   (SSE)
GET /api/config
```

## File layout

```
dashboard/
  backend/
    __init__.py
    app.py            FastAPI app
    data_sources.py   All file reads
    models.py         Pydantic models
    tailer.py         SSE log streaming
    requirements.txt
  frontend/
    index.html
    package.json
    vite.config.ts
    tsconfig.json
    src/
      main.tsx
      App.tsx
      index.css
      api/client.ts
      components/Layout.tsx
      pages/
        Overview.tsx
        Logs.tsx
        Trades.tsx
        Settings.tsx
tools/
  run_dashboard_backend.ps1
  run_dashboard_frontend.ps1
```
