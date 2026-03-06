"""FastAPI backend for Eclipse Scalper Dashboard."""
from __future__ import annotations

import os
import time
import json
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .data_sources import (
    LOGS_DIR,
    build_overview,
    list_log_files,
    log_list_last_cache_hit,
    log_tail_last_source,
    read_config_entries,
    read_micro_edge_gates,
    read_passive_profiles,
    read_quality_events,
    read_regime_events,
    read_runtime_status,
    read_live_metrics,
    read_live_monitor_tests_status,
    read_ops_health,
    read_connectivity_diag,
    read_supervisor_status,
    read_scoreboard,
    read_signal_events,
    read_stability_events,
    tail_log_file,
)
from .control_actions import (
    append_incident_audit,
    bulk_update_incidents,
    build_session_timeline,
    control_enabled,
    get_auto_runbook_policy,
    get_macro_preset,
    get_runbook_session,
    list_incidents,
    list_actions,
    list_runbook_sessions,
    preview_bulk_update_incidents,
    read_incident_audit,
    read_history,
    run_auto_runbook_once,
    run_action,
    run_runbook_for_incident,
    run_runbook,
    run_runbook_with_context,
    set_auto_runbook_policy,
    set_macro_preset,
    undo_last_incident_action,
    update_incident,
    update_runbook_session,
)
from .tailer import tail_sse
from .models import (
    ControlActionInfo,
    ControlActionRequest,
    ControlActionResult,
    ControlHistoryItem,
    IncidentActionRequest,
    IncidentBulkActionRequest,
    IncidentBulkPreviewRequest,
    IncidentBulkPreviewResult,
    IncidentAuditEvent,
    IncidentInboxItem,
    AutoRunbookPolicy,
    AutoRunbookResult,
    MacroPresetConfig,
    RunbookContextRequest,
    RunbookRequest,
    RunbookSessionPatchRequest,
    SessionTimelineEvent,
    RunbookSessionDetail,
    RunbookSessionSummary,
    ConfigEntry,
    DataQualityEvent,
    HealthResponse,
    OpsHealthResponse,
    DiagConnectivityResponse,
    LogFile,
    LogTailResponse,
    MicroEdgeGatesResponse,
    OverviewResponse,
    PassiveProfilesResponse,
    RegimeEvent,
    RuntimeResponse,
    LiveMetricsResponse,
    LiveMonitorTestsStatusResponse,
    Scoreboard,
    SignalEvent,
    StabilityEvent,
    SecurityAuditEvent,
)

_SECURITY_AUDIT_PATH = LOGS_DIR / "dashboard_security_audit.jsonl"
_RATE_STATE: dict[str, list[float]] = {}
_IDEMPOTENCY_STATE: dict[str, tuple[float, object]] = {}
_APP_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_TEST_SCRIPT = _APP_REPO_ROOT / "tools" / "run_live_monitor_tests.ps1"
_LIVE_TEST_RUNNER_LOG = _APP_REPO_ROOT / "logs" / "live_monitor_tests_runner.log"


def _cors_origins() -> list[str]:
    raw = (os.environ.get("DASHBOARD_CORS_ORIGINS") or "").strip()
    if not raw:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [x.strip() for x in raw.split(",") if x.strip()]

app = FastAPI(
    title="Eclipse Scalper Dashboard API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_dirs() -> None:
    """Create required directories on dashboard boot."""
    for d in (LOGS_DIR, LOGS_DIR / "health"):
        d.mkdir(parents=True, exist_ok=True)


def _role_rank(role: str) -> int:
    return {"viewer": 1, "operator": 2, "admin": 3}.get((role or "viewer").strip().lower(), 1)


def _control_role() -> str:
    return (os.environ.get("DASHBOARD_CONTROL_ROLE", "admin") or "admin").strip().lower()


def _strict_header_role() -> bool:
    raw = (os.environ.get("DASHBOARD_STRICT_HEADER_ROLE", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _effective_role(req: Request) -> str:
    env_role = _control_role()
    hdr_role = (req.headers.get("x-role") or "").strip().lower()
    if hdr_role not in {"viewer", "operator", "admin"}:
        return env_role
    if _strict_header_role():
        return hdr_role if _role_rank(hdr_role) <= _role_rank(env_role) else env_role
    return hdr_role


def _require_role(min_role: str, req: Request) -> None:
    if _role_rank(_effective_role(req)) < _role_rank(min_role):
        _append_security_audit(req, "role_denied", f"need={min_role} have={_effective_role(req)}")
        raise HTTPException(status_code=403, detail=f"Requires role: {min_role}")


def _guard_write(req: Request, min_role: str, rate_bucket: str, rate_max_ops: int) -> None:
    _require_api_key(req)
    _require_role(min_role, req)
    _check_rate_limit(req, rate_bucket, rate_max_ops)


def _request_audit_meta(req: Request) -> dict:
    operator = (req.headers.get("x-operator") or os.environ.get("DASHBOARD_OPERATOR") or "local").strip() or "local"
    return {
        "operator": operator,
        "role": _effective_role(req),
        "ip": (req.client.host if req.client else None),
        "user_agent": req.headers.get("user-agent"),
        "request_id": req.headers.get("x-request-id"),
    }


def _append_security_audit(req: Request, kind: str, detail: str = "") -> None:
    row = {
        "ts": time.time(),
        "kind": kind,
        "path": req.url.path,
        "method": req.method,
        **_request_audit_meta(req),
        "detail": detail[:500],
    }
    try:
        _SECURITY_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_SECURITY_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _read_security_audit(limit: int = 100) -> list[dict]:
    if limit <= 0:
        return []
    rows: list[dict] = []
    try:
        with open(_SECURITY_AUDIT_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return rows[-limit:]


def _api_key_required() -> bool:
    return bool((os.environ.get("DASHBOARD_API_KEY") or "").strip())


def _require_api_key(req: Request) -> None:
    if not _api_key_required():
        return
    expected = (os.environ.get("DASHBOARD_API_KEY") or "").strip()
    actual = (req.headers.get("x-api-key") or "").strip()
    if not actual or actual != expected:
        _append_security_audit(req, "auth_failed", "invalid api key")
        raise HTTPException(status_code=401, detail="Invalid API key")


def _rate_limit_enabled() -> bool:
    raw = (os.environ.get("DASHBOARD_RATE_LIMIT_ENABLED", "0") or "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _check_rate_limit(req: Request, bucket: str, max_ops: int) -> None:
    if not _rate_limit_enabled():
        return
    window_sec = max(5, int((os.environ.get("DASHBOARD_RATE_LIMIT_WINDOW_SEC") or "60").strip() or "60"))
    now = time.time()
    key = f"{bucket}:{(req.client.host if req.client else '-') }:{(req.headers.get('x-operator') or '-')}"
    arr = [ts for ts in _RATE_STATE.get(key, []) if now - ts <= window_sec]
    if len(arr) >= max_ops:
        _RATE_STATE[key] = arr
        _append_security_audit(req, "rate_limited", f"bucket={bucket} max={max_ops} window={window_sec}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    arr.append(now)
    _RATE_STATE[key] = arr


def _idempotency_enabled() -> bool:
    raw = (os.environ.get("DASHBOARD_IDEMPOTENCY_ENABLED", "1") or "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _idempotency_replay(req: Request, scope: str) -> object | None:
    if not _idempotency_enabled():
        return None
    key = (req.headers.get("x-idempotency-key") or "").strip()
    if not key:
        return None
    ttl_sec = max(60, int((os.environ.get("DASHBOARD_IDEMPOTENCY_TTL_SEC") or "600").strip() or "600"))
    now = time.time()
    # purge expired
    expired = [k for k, (ts, _) in _IDEMPOTENCY_STATE.items() if now - ts > ttl_sec]
    for k in expired:
        _IDEMPOTENCY_STATE.pop(k, None)
    storage_key = f"{scope}:{key}"
    hit = _IDEMPOTENCY_STATE.get(storage_key)
    if hit is None:
        return None
    _append_security_audit(req, "idempotency_replay", f"scope={scope}")
    return hit[1]


def _idempotency_store(req: Request, scope: str, payload: object) -> None:
    if not _idempotency_enabled():
        return
    key = (req.headers.get("x-idempotency-key") or "").strip()
    if not key:
        return
    _IDEMPOTENCY_STATE[f"{scope}:{key}"] = (time.time(), payload)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Overview
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/overview", response_model=OverviewResponse)
async def get_overview():
    return build_overview()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Scoreboard
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/scoreboard", response_model=Scoreboard)
async def get_scoreboard():
    return read_scoreboard()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Gates & profiles
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/gates", response_model=MicroEdgeGatesResponse)
async def get_gates():
    return read_micro_edge_gates()


@app.get("/api/passive-profiles", response_model=PassiveProfilesResponse)
async def get_passive_profiles():
    return read_passive_profiles()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Events / log streams
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/events/regimes", response_model=list[RegimeEvent])
async def get_regime_events(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = Query(None),
):
    return read_regime_events(limit=limit, symbol=symbol)


@app.get("/api/events/signals", response_model=list[SignalEvent])
async def get_signal_events(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = Query(None),
):
    return read_signal_events(limit=limit, symbol=symbol)


@app.get("/api/events/stability", response_model=list[StabilityEvent])
async def get_stability_events(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = Query(None),
):
    return read_stability_events(limit=limit, symbol=symbol)


@app.get("/api/events/quality", response_model=list[DataQualityEvent])
async def get_quality_events(
    limit: int = Query(100, ge=1, le=1000),
    symbol: Optional[str] = Query(None),
):
    return read_quality_events(limit=limit, symbol=symbol)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Logs
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/logs", response_model=list[LogFile])
async def get_log_files(response: Response):
    t0 = time.perf_counter()
    rows = list_log_files()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Logs-List-Ms"] = f"{dt_ms:.2f}"
    response.headers["X-Logs-Cache-Hit"] = "1" if log_list_last_cache_hit() else "0"
    return rows


@app.get("/api/logs/tail", response_model=LogTailResponse)
async def get_log_tail(
    response: Response,
    file: str = Query(..., description="Filename in logs/ directory"),
    limit: int = Query(200, ge=1, le=2000),
):
    t0 = time.perf_counter()
    lines = tail_log_file(file, limit=limit)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    response.headers["X-Logs-Tail-Ms"] = f"{dt_ms:.2f}"
    response.headers["X-Logs-Tail-Source"] = log_tail_last_source()
    return {"file": file, "lines": lines}


@app.get("/api/logs/stream")
async def stream_log(
    file: str = Query(..., description="Filename in logs/ directory"),
    last_n: int = Query(50, ge=0, le=500),
):
    """Server-Sent Events stream for a log file."""
    safe_name = Path(file).name
    path = (LOGS_DIR / safe_name).resolve()
    try:
        path.relative_to(LOGS_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal rejected")

    async def generator():
        async for chunk in tail_sse(path, last_n=last_n):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Config
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/config", response_model=list[ConfigEntry])
async def get_config():
    return read_config_entries()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Runtime status
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/runtime", response_model=RuntimeResponse)
async def get_runtime():
    return read_runtime_status()


@app.get("/api/live/metrics", response_model=LiveMetricsResponse)
async def get_live_metrics():
    return read_live_metrics()

@app.get("/api/live/tests/status", response_model=LiveMonitorTestsStatusResponse)
async def get_live_tests_status(limit: int = Query(80, ge=10, le=400)):
    return read_live_monitor_tests_status(limit=limit)


@app.post("/api/live/tests/run", response_model=dict)
async def post_live_tests_run(request: Request):
    _guard_write(request, "operator", "live_tests_run", 12)
    if not _LIVE_TEST_SCRIPT.exists():
        raise HTTPException(status_code=404, detail=f"Missing script: {_LIVE_TEST_SCRIPT}")

    _LIVE_TEST_RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(_LIVE_TEST_SCRIPT),
    ]
    try:
        with open(_LIVE_TEST_RUNNER_LOG, "a", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_APP_REPO_ROOT),
                stdout=fh,
                stderr=fh,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start live tests: {exc}")

    out = {
        "ok": True,
        "started": True,
        "pid": proc.pid,
        "script": str(_LIVE_TEST_SCRIPT),
        "runner_log": str(_LIVE_TEST_RUNNER_LOG),
        "command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\tools\\run_live_monitor_tests.ps1",
    }
    append_incident_audit({"kind": "live_tests_run", "action": "run_live_monitor_tests", **_request_audit_meta(request)})
    return out
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Health
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/api/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "api_version": app.version}


@app.get("/api/ops/health", response_model=OpsHealthResponse)
async def get_ops_health():
    return read_ops_health()


@app.get("/api/diag/connectivity", response_model=DiagConnectivityResponse)
async def get_diag_connectivity():
    return read_connectivity_diag()


@app.get("/api/ops/supervisor", response_model=dict)
async def get_ops_supervisor():
    return read_supervisor_status()


@app.get("/ops/supervisor", response_model=dict, include_in_schema=False)
async def get_ops_supervisor_alias():
    return read_supervisor_status()


# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Debug control
# Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

@app.get("/api/debug/actions", response_model=list[ControlActionInfo])
async def get_debug_actions():
    return list_actions()


@app.get("/api/debug/history", response_model=list[ControlHistoryItem])
async def get_debug_history(limit: int = Query(30, ge=1, le=200)):
    return read_history(limit=limit)


@app.post("/api/debug/run", response_model=ControlActionResult)
async def post_debug_run(req: ControlActionRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "debug_run", 30)
    try:
        out = run_action(req.action)
        append_incident_audit({"kind": "debug_run", "action": req.action, **_request_audit_meta(request)})
        return out
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action}")


@app.post("/api/debug/runbook", response_model=RunbookSessionDetail)
async def post_debug_runbook(req: RunbookRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "runbook", 12)
    try:
        out = run_runbook(req.actions or None)
        append_incident_audit({"kind": "runbook", "action": "runbook", **_request_audit_meta(request)})
        return out
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown action in runbook: {exc}")


@app.post("/api/debug/runbook/from-incident", response_model=RunbookSessionDetail)
async def post_debug_runbook_from_incident(req: RunbookContextRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "runbook_incident", 12)
    try:
        context = {
            "file": req.file or "",
            "query": req.query or "",
            "level": req.level or "",
            "source": "dashboard_incident",
        }
        out = run_runbook_with_context(req.actions or None, context=context)
        append_incident_audit({"kind": "runbook_incident", "action": "runbook_from_incident", **_request_audit_meta(request)})
        return out
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown action in runbook: {exc}")


@app.get("/api/debug/sessions", response_model=list[RunbookSessionSummary])
async def get_debug_sessions(limit: int = Query(30, ge=1, le=200)):
    return list_runbook_sessions(limit=limit)


@app.get("/api/debug/sessions/{session_id}", response_model=RunbookSessionDetail)
async def get_debug_session_detail(session_id: str):
    try:
        return get_runbook_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@app.patch("/api/debug/sessions/{session_id}", response_model=RunbookSessionDetail)
async def patch_debug_session(session_id: str, req: RunbookSessionPatchRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "session_patch", 40)
    try:
        out = update_runbook_session(session_id, tag=req.tag, note=req.note)
        append_incident_audit({"kind": "session_patch", "action": "session_patch", "incident_id": session_id, **_request_audit_meta(request)})
        return out
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@app.get("/api/debug/sessions/{session_id}/timeline", response_model=list[SessionTimelineEvent])
async def get_debug_session_timeline(session_id: str):
    try:
        session = get_runbook_session(session_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return build_session_timeline(session)


@app.get("/api/debug/incidents", response_model=list[IncidentInboxItem])
async def get_debug_incidents(limit: int = Query(50, ge=1, le=500)):
    return list_incidents(limit=limit)


@app.patch("/api/debug/incidents/{incident_id}", response_model=dict)
async def patch_debug_incident(incident_id: str, req: IncidentActionRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "incident_patch", 40)
    out = update_incident(
        incident_id=incident_id,
        action=req.action,
        incident_type=req.incident_type,
        snooze_minutes=req.snooze_minutes or 60,
    )
    append_incident_audit(
        {
            "kind": "incident_action",
            "action": req.action,
            "incident_id": incident_id,
            "incident_type": req.incident_type,
            **_request_audit_meta(request),
        }
    )
    return out


@app.post("/api/debug/incidents/bulk", response_model=dict)
async def post_debug_incidents_bulk(req: IncidentBulkActionRequest, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "admin", "incident_bulk", 8)
    cached = _idempotency_replay(request, "incident_bulk")
    if cached is not None:
        return cached
    out = bulk_update_incidents(
        action=req.action,
        incident_type=req.incident_type,
        status_scope=req.status_scope or "active",
    )
    append_incident_audit({"kind": "incident_bulk", "action": req.action, "incident_type": req.incident_type, **_request_audit_meta(request)})
    _idempotency_store(request, "incident_bulk", out)
    return out


@app.post("/api/debug/incidents/bulk/preview", response_model=IncidentBulkPreviewResult)
async def post_debug_incidents_bulk_preview(req: IncidentBulkPreviewRequest):
    return preview_bulk_update_incidents(
        incident_type=req.incident_type,
        status_scope=req.status_scope or "active",
    )


@app.post("/api/debug/incidents/undo", response_model=dict)
async def post_debug_incidents_undo(request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "admin", "incident_undo", 8)
    cached = _idempotency_replay(request, "incident_undo")
    if cached is not None:
        return cached
    out = undo_last_incident_action()
    append_incident_audit({"kind": "incident_undo", "action": "undo", **_request_audit_meta(request)})
    _idempotency_store(request, "incident_undo", out)
    return out


@app.get("/api/debug/incidents/audit", response_model=list[IncidentAuditEvent])
async def get_debug_incidents_audit(limit: int = Query(50, ge=1, le=500)):
    return read_incident_audit(limit=limit)


@app.post("/api/debug/incidents/{incident_id}/runbook", response_model=RunbookSessionDetail)
async def post_debug_incident_runbook(incident_id: str, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "operator", "incident_runbook", 12)
    cached = _idempotency_replay(request, f"incident_runbook:{incident_id}")
    if cached is not None:
        return cached
    try:
        out = run_runbook_for_incident(incident_id)
        append_incident_audit({"kind": "incident_runbook", "action": "incident_runbook", "incident_id": incident_id, **_request_audit_meta(request)})
        _idempotency_store(request, f"incident_runbook:{incident_id}", out)
        return out
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")


@app.get("/api/debug/incidents-policy", response_model=AutoRunbookPolicy)
async def get_debug_incident_policy():
    return get_auto_runbook_policy()


@app.patch("/api/debug/incidents-policy", response_model=AutoRunbookPolicy)
async def patch_debug_incident_policy(req: AutoRunbookPolicy, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "admin", "incident_policy", 20)
    out = set_auto_runbook_policy(enabled=req.enabled, min_level=req.min_level, cooldown_sec=req.cooldown_sec)
    append_incident_audit({"kind": "incident_policy", "action": "policy_patch", **_request_audit_meta(request)})
    return out


@app.post("/api/debug/incidents/auto-run", response_model=AutoRunbookResult)
async def post_debug_incident_auto_run(request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "admin", "incident_auto_run", 12)
    cached = _idempotency_replay(request, "incident_auto_run")
    if cached is not None:
        return cached
    out = run_auto_runbook_once()
    append_incident_audit({"kind": "incident_auto_run", "action": "auto_run", **_request_audit_meta(request)})
    _idempotency_store(request, "incident_auto_run", out)
    return out


@app.get("/api/debug/macro-preset", response_model=MacroPresetConfig)
async def get_debug_macro_preset():
    return get_macro_preset()


@app.patch("/api/debug/macro-preset", response_model=MacroPresetConfig)
async def patch_debug_macro_preset(req: MacroPresetConfig, request: Request):
    if not control_enabled():
        raise HTTPException(status_code=403, detail="Dashboard control disabled")
    _guard_write(request, "admin", "macro_preset_patch", 20)
    meta = _request_audit_meta(request)
    out = set_macro_preset(
        preset=req.preset,
        ack_filtered=req.ackFiltered,
        auto_run=req.autoRun,
        export_md=req.exportMd,
        refresh=req.refresh,
        owner=req.owner or str(meta.get("operator") or "local"),
    )
    append_incident_audit({"kind": "macro_preset", "action": "macro_preset_patch", **meta})
    return out


@app.get("/api/debug/security-audit", response_model=list[SecurityAuditEvent])
async def get_debug_security_audit(limit: int = Query(100, ge=1, le=500)):
    return _read_security_audit(limit=limit)

