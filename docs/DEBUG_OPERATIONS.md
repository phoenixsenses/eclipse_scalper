# Debug Operations Playbook

This runbook defines the standard operator flow for dashboard-driven debugging.

## 1. Startup
- Backend API: `python -m dashboard.backend.app`
- Frontend: run dashboard frontend (dev server) and open `/debug`.
- Confirm control is enabled:
  - `DASHBOARD_CONTROL_ENABLED=1`

## 2. Guided Session (Baseline)
- In **Debug > Guided Debug Session**, click `Run Guided Session`.
- Steps executed:
  1. `validate_env`
  2. `preflight_check`
  3. `paper_trade_status`
  4. `incident_bundle`
- If failed, use `Re-run failed step` from Incident Summary.

## 3. Incident-Driven Runbook
- If Incident Summary appears, click `Runbook From Incident`.
- The backend stores incident context in session metadata:
  - `context.file`
  - `context.query`
  - `context.level`
  - `context.source=dashboard_incident`
- Use playbook command chips to copy suggested commands quickly.

## 4. Session Metadata (Tag/Note)
- Load a session from **Recent Sessions**.
- In **Session Metadata**:
  - Set `tag` (examples: `network`, `deps`, `regime`, `data`)
  - Add `note` with key observations
  - Click `Save`
- `Recent Sessions` will show `tag` and `note_preview`.

## 5. Session Compare
- Pick two sessions using `A` and `B`.
- Click `Compare`.
- Optional:
  - Enable `only failed steps`
  - `Export Compare JSON`
  - `Export Compare MD`
  - `Copy Share Link` (query includes `compareA`, `compareB`)

## 6. Timeline Analysis
- For a loaded session, use **Session Timeline**:
  - `step_start`
  - `step_end`
  - `log_snippet`
- Use timeline + snippets to answer:
  - Which step failed first?
  - Did incident signature change between sessions?
  - Are logs consistent with the failing action?

## 7. Auto-Refresh and Alerts
- Selected session auto-refreshes every 15s.
- Alert triggers when:
  - failed step count increases
  - incident type changes
  - snippet count increases
- Dismiss alert after acknowledgement.

## 8. Recommended Operator Workflow
1. Run guided session.
2. Review Incident Summary.
3. Run incident-driven runbook.
4. Tag and note session.
5. Compare with last known good session.
6. Export compare report for handoff.

## 9. API Endpoints Used
- `POST /api/debug/runbook`
- `POST /api/debug/runbook/from-incident`
- `GET /api/debug/sessions`
- `GET /api/debug/sessions/{session_id}`
- `PATCH /api/debug/sessions/{session_id}`
- `GET /api/debug/sessions/{session_id}/timeline`

## 10. Troubleshooting
- `403 Dashboard control disabled`:
  - set `DASHBOARD_CONTROL_ENABLED=1`
- Empty sessions list:
  - verify `logs/debug_sessions/` exists and is writable
- Compare shows no differences:
  - disable `only failed steps` filter
- No timeline rows:
  - open session detail first (`Load`), then wait for refresh

## 11. Security Headers (Frontend -> Backend)
- Write endpoints use these headers from dashboard auth context:
  - `X-Api-Key`
  - `X-Operator`
  - `X-Role`
  - `X-Idempotency-Key` (auto when enabled)
- Recommended:
  - Set auth context in **Settings > Dashboard Auth Context**
  - Use `viewer` for read-only investigations
  - Use `admin` only for policy/bulk/undo operations

## 12. Security Smoke Test
Run end-to-end checks against a running backend:

```powershell
python -m tools.smoke_dashboard_security --base http://127.0.0.1:8000 --api-key "<your_key>"
```

Checks:
- API key gate
- Role gate
- Idempotency replay
- Rate-limit surface validity
- Security audit endpoint reachability
