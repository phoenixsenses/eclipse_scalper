# Eclipse Scalper Dashboard Frontend

## Prerequisites
- Node.js 20 LTS recommended (Node 18+ supported by the current toolchain).
- npm (lockfile is `package-lock.json`).
- Windows PowerShell note: if `npm` is blocked by execution policy, use `npm.cmd` commands.

## Run Backend (Terminal 1)
From repo root:

```powershell
.\tools\run_dashboard_backend.ps1
```

Backend URL: `http://localhost:8765`

## Run Frontend (Terminal 2)
From repo root:

```powershell
cd dashboard\frontend
npm.cmd install
npm.cmd run dev
```

Frontend URL: `http://localhost:5173`

## URLs
- Frontend app: `http://localhost:5173`
- Backend API root: `http://localhost:8765/api`
- Backend docs: `http://localhost:8765/api/docs`

## Environment Configuration
Create `.env` in `dashboard/frontend` (or copy from `.env.example`) to customize dev behavior.

- `VITE_API_BASE`:
  - Browser-facing API base used by frontend code.
  - Default behavior in this project is relative `/api`.
- `VITE_PROXY_TARGET`:
  - Used by Vite dev server proxy only.
  - Default: `http://localhost:8765`.
  - This keeps browser requests relative (`/api/...`) while forwarding to backend in dev.

## Troubleshooting
- PowerShell error: `npm.ps1 cannot be loaded because running scripts is disabled`
  - Use `npm.cmd` instead of `npm`.
  - Or allow scripts for current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

