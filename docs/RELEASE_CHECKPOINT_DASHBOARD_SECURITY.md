# Release Checkpoint: Dashboard Security + Ops Controls

## Scope
- API key gate on write endpoints (`X-Api-Key`)
- Role gate with strict header role clamp (`X-Role`, env max role)
- Rate limit and idempotency protection on critical write routes
- Security audit log stream and dashboard visibility
- Global frontend auth context + settings-driven header management

## New/Updated Environment Variables
- `DASHBOARD_API_KEY`
- `DASHBOARD_CONTROL_ROLE`
- `DASHBOARD_STRICT_HEADER_ROLE`
- `DASHBOARD_CORS_ORIGINS`
- `DASHBOARD_RATE_LIMIT_ENABLED`
- `DASHBOARD_RATE_LIMIT_WINDOW_SEC`
- `DASHBOARD_IDEMPOTENCY_ENABLED`
- `DASHBOARD_IDEMPOTENCY_TTL_SEC`
- `DASHBOARD_OPERATOR`

## Backward Compatibility
- If `DASHBOARD_API_KEY` is empty: no API-key enforcement.
- Role model defaults to env role only if `X-Role` is missing.
- Idempotency behavior only applies when `X-Idempotency-Key` is sent by client.

## Operational Validation
```powershell
python -m tools.smoke_dashboard_security --base http://127.0.0.1:8000 --api-key "<dashboard_api_key>"
```

Expected:
- key gate enforced (when key configured)
- viewer blocked on write routes
- idempotency replay returns same payload
- security audit endpoint returns entries

## Rollback Plan
- Set `DASHBOARD_API_KEY=` (empty) to disable key gate.
- Set `DASHBOARD_RATE_LIMIT_ENABLED=0` to disable write rate limiting.
- Set `DASHBOARD_IDEMPOTENCY_ENABLED=0` to disable replay cache.
- Keep `DASHBOARD_CONTROL_ROLE=admin` if emergency write access is needed.
