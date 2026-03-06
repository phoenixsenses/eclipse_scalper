"""Whitelisted debug/control actions for dashboard frontend."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_sources import LOGS_DIR, REPO_ROOT


@dataclass(frozen=True)
class ActionSpec:
    action: str
    command: list[str]
    timeout_sec: int
    description: str


def _py_module(module_name: str) -> list[str]:
    return [sys.executable, "-m", module_name]


ACTION_SPECS: dict[str, ActionSpec] = {
    "validate_env": ActionSpec(
        action="validate_env",
        command=_py_module("tools.validate_env"),
        timeout_sec=45,
        description="Run environment validation checks.",
    ),
    "preflight_check": ActionSpec(
        action="preflight_check",
        command=_py_module("tools.preflight_check"),
        timeout_sec=60,
        description="Run preflight data/readiness checks.",
    ),
    "paper_trade_status": ActionSpec(
        action="paper_trade_status",
        command=_py_module("tools.paper_trade_status"),
        timeout_sec=45,
        description="Collect current paper trading status.",
    ),
    "paper_trade_summary": ActionSpec(
        action="paper_trade_summary",
        command=_py_module("tools.paper_trade_summary"),
        timeout_sec=60,
        description="Generate paper trade summary.",
    ),
    "db_maintenance": ActionSpec(
        action="db_maintenance",
        command=_py_module("tools.db_maintenance"),
        timeout_sec=180,
        description="Run DB maintenance/checkpoint/backup tasks.",
    ),
    "incident_bundle": ActionSpec(
        action="incident_bundle",
        command=_py_module("tools.incident_bundle"),
        timeout_sec=180,
        description="Generate incident diagnostics bundle.",
    ),
    "dashboard_logs_bundle": ActionSpec(
        action="dashboard_logs_bundle",
        command=_py_module("tools.dashboard_logs_bundle"),
        timeout_sec=60,
        description="Export dashboard backend/supervisor log bundle.",
    ),
    "restart_dashboard_backend": ActionSpec(
        action="restart_dashboard_backend",
        command=_py_module("tools.dashboard_backend_restart_once"),
        timeout_sec=45,
        description="Restart dashboard backend process once and wait for health.",
    ),
}

_HISTORY_PATH = LOGS_DIR / "dashboard_debug_actions.jsonl"
_SESSION_DIR = LOGS_DIR / "debug_sessions"
_INCIDENT_STATE_PATH = LOGS_DIR / "debug_incident_state.json"
_AUTO_POLICY_PATH = LOGS_DIR / "debug_auto_policy.json"
_INCIDENT_UNDO_PATH = LOGS_DIR / "debug_incident_undo.json"
_INCIDENT_AUDIT_PATH = LOGS_DIR / "debug_incident_audit.jsonl"
_MACRO_PRESET_PATH = LOGS_DIR / "debug_macro_preset.json"
_UNDO_WINDOW_SEC = 60
DEFAULT_RUNBOOK_STEPS = ["validate_env", "preflight_check", "paper_trade_status", "incident_bundle"]
DEFAULT_MACRO_PRESET = {
    "preset": "full",
    "ackFiltered": True,
    "autoRun": True,
    "exportMd": True,
    "refresh": True,
    "owner": "local",
    "updated_ts": 0.0,
}


def control_enabled() -> bool:
    raw = os.environ.get("DASHBOARD_CONTROL_ENABLED", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def list_actions() -> list[dict[str, Any]]:
    return [
        {
            "action": spec.action,
            "description": spec.description,
            "timeout_sec": spec.timeout_sec,
        }
        for spec in ACTION_SPECS.values()
    ]


def _append_history(payload: dict[str, Any]) -> None:
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def read_history(limit: int = 30) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    items: list[dict[str, Any]] = []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items.append(obj)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    return items[-limit:]


def run_action(action: str) -> dict[str, Any]:
    spec = ACTION_SPECS.get(action)
    if spec is None:
        raise KeyError(action)

    started = time.time()
    history_item: dict[str, Any] = {
        "ts": started,
        "action": action,
        "command": spec.command,
    }
    try:
        proc = subprocess.run(
            spec.command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec.timeout_sec,
            check=False,
        )
        ended = time.time()
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        result = {
            "action": action,
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "duration_sec": round(ended - started, 3),
            "output": output[-20000:],
            "started_ts": started,
            "ended_ts": ended,
        }
        history_item.update(
            {
                "ok": result["ok"],
                "exit_code": result["exit_code"],
                "duration_sec": result["duration_sec"],
            }
        )
        _append_history(history_item)
        return result
    except subprocess.TimeoutExpired as exc:
        ended = time.time()
        output = (exc.stdout or "") + ("\n" + exc.stderr if exc.stderr else "")
        result = {
            "action": action,
            "ok": False,
            "exit_code": -1,
            "duration_sec": round(ended - started, 3),
            "output": (output + f"\n[dashboard] timeout after {spec.timeout_sec}s")[-20000:],
            "started_ts": started,
            "ended_ts": ended,
        }
        history_item.update(
            {
                "ok": False,
                "exit_code": -1,
                "duration_sec": result["duration_sec"],
                "error": "timeout",
            }
        )
        _append_history(history_item)
        return result


def _detect_incident_hint(source: str) -> dict[str, Any] | None:
    t = (source or "").lower()
    if not t.strip():
        return None
    if "modulenotfounderror" in t or "no module named" in t:
        return {
            "type": "dependency_missing",
            "title": "Dependency missing",
            "detail": "Python module/dependency missing at runtime.",
            "file": "paper_trading.log",
            "query": "ModuleNotFoundError",
            "level": "ERROR",
            "suggested_command": ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
            "confidence": 0.95,
        }
    if "requesttimeout" in t or "exchange timeout" in t:
        return {
            "type": "exchange_timeout",
            "title": "Exchange timeout",
            "detail": "Exchange/network request timed out during bootstrap or runtime.",
            "file": "paper_trading.log",
            "query": "RequestTimeout",
            "level": "WARNING",
            "suggested_command": "python -m tools.incident_bundle",
            "confidence": 0.85,
        }
    if "db stale" in t or "preflight fail" in t:
        return {
            "type": "data_freshness",
            "title": "Data freshness issue",
            "detail": "Collector lag or DB freshness gate failed.",
            "file": "microstructure_collector.log",
            "query": "stale",
            "level": "WARNING",
            "suggested_command": "python -m tools.preflight_check",
            "confidence": 0.90,
        }
    if "regime_mismatch" in t or "regime=unknown" in t:
        return {
            "type": "regime_gate",
            "title": "Regime gating block",
            "detail": "Regime gate blocked entries or regime state unresolved.",
            "file": "paper_trading.log",
            "query": "REGIME",
            "level": "INFO",
            "suggested_command": "python -m tools.paper_trade_status",
            "confidence": 0.75,
        }
    if "no_match" in t or "signal not present" in t:
        return {
            "type": "signal_no_match",
            "title": "Signal no-match",
            "detail": "Market state did not satisfy pocket thresholds at evaluation time.",
            "file": "paper_trading.log",
            "query": "no_match_detail",
            "level": "INFO",
            "suggested_command": "python -m tools.paper_trade_summary",
            "confidence": 0.75,
        }
    if "shutdown" in t or "offline" in t:
        return {
            "type": "shutdown_event",
            "title": "Shutdown event",
            "detail": "Runtime reported shutdown/offline transition.",
            "file": "paper_trading.log",
            "query": "SHUTDOWN",
            "level": "CRITICAL",
            "suggested_command": "python -m tools.incident_bundle",
            "confidence": 0.90,
        }
    return None


def _collect_log_snippets(file_name: str, query: str, max_hits: int = 5, context: int = 1) -> list[dict[str, Any]]:
    if not file_name or not query:
        return []
    path = (LOGS_DIR / Path(file_name).name).resolve()
    try:
        path.relative_to(LOGS_DIR.resolve())
    except Exception:
        return []
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except Exception:
        return []

    q = query.lower()
    hits: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if q not in line.lower():
            continue
        start = max(0, idx - context)
        end = min(len(lines), idx + context + 1)
        hits.append(
            {
                "file": file_name,
                "line_no": idx + 1,
                "match": line,
                "context_before": lines[start:idx],
                "context_after": lines[idx + 1 : end],
            }
        )
        if len(hits) >= max_hits:
            break
    return hits


def run_runbook(actions: list[str] | None = None) -> dict[str, Any]:
    return run_runbook_with_context(actions=actions, context=None)


def run_runbook_with_context(actions: list[str] | None = None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = actions or list(DEFAULT_RUNBOOK_STEPS)
    started = time.time()
    step_results: list[dict[str, Any]] = []
    failed_action: str | None = None

    for action in steps:
        result = run_action(action)
        step_results.append(result)
        if not result.get("ok", False):
            failed_action = action
            break

    ended = time.time()
    merged_output = "\n".join(str(x.get("output", "")) for x in step_results if x.get("output"))
    incident = _detect_incident_hint(merged_output)
    snippets = _collect_log_snippets(
        file_name=(incident or {}).get("file", ""),
        query=(incident or {}).get("query", ""),
        max_hits=5,
        context=1,
    )
    session_id = "session_" + datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    payload = {
        "session_id": session_id,
        "started_ts": started,
        "ended_ts": ended,
        "duration_sec": round(ended - started, 3),
        "ok": failed_action is None,
        "failed_action": failed_action,
        "steps": step_results,
        "incident_hint": incident,
        "log_snippets": snippets,
        "context": context or {},
        "tag": "",
        "note": "",
        "updated_ts": ended,
    }
    try:
        _SESSION_DIR.mkdir(parents=True, exist_ok=True)
        with open(_SESSION_DIR / f"{session_id}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return payload


def list_runbook_sessions(limit: int = 30) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    try:
        if not _SESSION_DIR.exists():
            return []
        files = sorted(_SESSION_DIR.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for p in files[:limit]:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                obj = json.load(f)
            rows.append(
                {
                    "session_id": obj.get("session_id") or p.stem,
                    "started_ts": obj.get("started_ts"),
                    "ended_ts": obj.get("ended_ts"),
                    "duration_sec": obj.get("duration_sec"),
                    "ok": bool(obj.get("ok", False)),
                    "failed_action": obj.get("failed_action"),
                    "incident_type": (obj.get("incident_hint") or {}).get("type"),
                    "tag": obj.get("tag") or "",
                    "note_preview": (obj.get("note") or "")[:120],
                }
            )
        except Exception:
            continue
    return rows


def get_runbook_session(session_id: str) -> dict[str, Any]:
    safe = Path(session_id).name
    p = (_SESSION_DIR / f"{safe}.json").resolve()
    try:
        p.relative_to(_SESSION_DIR.resolve())
    except Exception:
        raise FileNotFoundError(safe)
    if not p.exists():
        raise FileNotFoundError(safe)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    obj.setdefault("context", {})
    obj.setdefault("tag", "")
    obj.setdefault("note", "")
    obj.setdefault("updated_ts", obj.get("ended_ts"))
    return obj


def _load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        if not path.exists():
            return dict(default)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            out = dict(default)
            out.update(obj)
            return out
    except Exception:
        pass
    return dict(default)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _incident_state() -> dict[str, Any]:
    return _load_json(
        _INCIDENT_STATE_PATH,
        {
            "acked_ids": [],
            "resolved_ids": [],
            "acked_meta": {},
            "resolved_meta": {},
            "snoozed_types": {},
            "muted_types": {},
        },
    )


def _write_incident_state(state: dict[str, Any]) -> None:
    _save_json(_INCIDENT_STATE_PATH, state)


def _incident_operator() -> str:
    return os.environ.get("DASHBOARD_OPERATOR", "local").strip() or "local"


def _append_incident_audit(payload: dict[str, Any]) -> None:
    row = {
        "ts": time.time(),
        "operator": _incident_operator(),
        **payload,
    }
    try:
        _INCIDENT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INCIDENT_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def append_incident_audit(payload: dict[str, Any]) -> None:
    _append_incident_audit(payload)


def read_incident_audit(limit: int = 50) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with open(_INCIDENT_AUDIT_PATH, "r", encoding="utf-8", errors="replace") as f:
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


def _record_incident_undo(state_before: dict[str, Any], action_payload: dict[str, Any]) -> None:
    payload = {
        "ts": time.time(),
        "expires_ts": time.time() + _UNDO_WINDOW_SEC,
        "state_before": state_before,
        "action": action_payload,
    }
    _save_json(_INCIDENT_UNDO_PATH, payload)


def _level_rank(level: str | None) -> int:
    m = {"INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    return m.get((level or "WARNING").upper(), 2)


def list_incidents(limit: int = 50) -> list[dict[str, Any]]:
    state = _incident_state()
    acked = set(state.get("acked_ids") or [])
    resolved = set(state.get("resolved_ids") or [])
    snoozed_types = state.get("snoozed_types") or {}
    muted_types = state.get("muted_types") or {}
    acked_meta = state.get("acked_meta") or {}
    resolved_meta = state.get("resolved_meta") or {}
    rows: list[dict[str, Any]] = []
    for s in list_runbook_sessions(limit=max(1, limit * 2)):
        sid = s.get("session_id")
        if not sid:
            continue
        detail = get_runbook_session(str(sid))
        hint = detail.get("incident_hint") or {}
        if not hint and detail.get("ok", False):
            continue
        incident_type = str(hint.get("type") or "unknown")
        incident_id = str(sid)
        status = "new"
        if incident_id in resolved:
            status = "resolved"
        elif incident_id in acked:
            status = "ack"
        rows.append(
            {
                "incident_id": incident_id,
                "session_id": str(sid),
                "ts": detail.get("ended_ts") or detail.get("started_ts"),
                "type": incident_type,
                "title": hint.get("title") or "Unknown incident",
                "level": hint.get("level") or "WARNING",
                "file": hint.get("file"),
                "query": hint.get("query"),
                "status": status,
                "snoozed_until": snoozed_types.get(incident_type),
                "muted": bool(muted_types.get(incident_type, False)),
                "failed_action": detail.get("failed_action"),
                "ack_ts": acked_meta.get(incident_id),
                "resolved_ts": resolved_meta.get(incident_id),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def update_incident(incident_id: str, action: str, incident_type: str | None = None, snooze_minutes: int = 60) -> dict[str, Any]:
    state = _incident_state()
    prev_state = json.loads(json.dumps(state))
    acked = set(state.get("acked_ids") or [])
    resolved = set(state.get("resolved_ids") or [])
    snoozed_types = dict(state.get("snoozed_types") or {})
    muted_types = dict(state.get("muted_types") or {})
    acked_meta = dict(state.get("acked_meta") or {})
    resolved_meta = dict(state.get("resolved_meta") or {})
    if action == "ack":
        acked.add(incident_id)
        acked_meta[incident_id] = time.time()
    elif action == "resolve":
        resolved.add(incident_id)
        resolved_meta[incident_id] = time.time()
    elif action == "snooze_type" and incident_type:
        snoozed_types[incident_type] = time.time() + max(1, int(snooze_minutes)) * 60
    elif action == "mute_type" and incident_type:
        muted_types[incident_type] = True
    elif action == "unmute_type" and incident_type:
        muted_types[incident_type] = False
    state["acked_ids"] = sorted(acked)
    state["resolved_ids"] = sorted(resolved)
    state["snoozed_types"] = snoozed_types
    state["muted_types"] = muted_types
    state["acked_meta"] = acked_meta
    state["resolved_meta"] = resolved_meta
    _record_incident_undo(
        prev_state,
        {
            "kind": "single",
            "incident_id": incident_id,
            "action": action,
            "incident_type": incident_type,
            "snooze_minutes": snooze_minutes,
        },
    )
    _write_incident_state(state)
    _append_incident_audit(
        {
            "kind": "single",
            "incident_id": incident_id,
            "action": action,
            "incident_type": incident_type,
        }
    )
    return state


def bulk_update_incidents(action: str, incident_type: str | None = None, status_scope: str = "active") -> dict[str, Any]:
    state = _incident_state()
    prev_state = json.loads(json.dumps(state))
    acked = set(state.get("acked_ids") or [])
    resolved = set(state.get("resolved_ids") or [])
    acked_meta = dict(state.get("acked_meta") or {})
    resolved_meta = dict(state.get("resolved_meta") or {})
    targets = list_incidents(limit=1000)
    count = 0
    for inc in targets:
        iid = str(inc.get("incident_id") or "")
        if not iid:
            continue
        if incident_type and str(inc.get("type") or "") != incident_type:
            continue
        if status_scope != "all" and str(inc.get("status") or "") == "resolved":
            continue
        if action == "ack":
            if iid not in acked:
                acked.add(iid)
                acked_meta[iid] = time.time()
                count += 1
        elif action == "resolve":
            if iid not in resolved:
                resolved.add(iid)
                resolved_meta[iid] = time.time()
                count += 1
    state["acked_ids"] = sorted(acked)
    state["resolved_ids"] = sorted(resolved)
    state["acked_meta"] = acked_meta
    state["resolved_meta"] = resolved_meta
    _record_incident_undo(
        prev_state,
        {
            "kind": "bulk",
            "action": action,
            "incident_type": incident_type,
            "status_scope": status_scope,
            "updated": count,
        },
    )
    _write_incident_state(state)
    _append_incident_audit(
        {
            "kind": "bulk",
            "action": action,
            "incident_type": incident_type,
            "status_scope": status_scope,
            "updated": count,
        }
    )
    return {"ok": True, "updated": count, "action": action, "incident_type": incident_type, "status_scope": status_scope}


def preview_bulk_update_incidents(incident_type: str | None = None, status_scope: str = "active") -> dict[str, Any]:
    targets = list_incidents(limit=1000)
    eligible = 0
    for inc in targets:
        if incident_type and str(inc.get("type") or "") != incident_type:
            continue
        if status_scope != "all" and str(inc.get("status") or "") == "resolved":
            continue
        eligible += 1
    return {
        "ok": True,
        "eligible": eligible,
        "incident_type": incident_type,
        "status_scope": status_scope,
    }


def undo_last_incident_action() -> dict[str, Any]:
    try:
        if not _INCIDENT_UNDO_PATH.exists():
            return {"ok": False, "reason": "no_undo_available"}
        with open(_INCIDENT_UNDO_PATH, "r", encoding="utf-8", errors="replace") as f:
            payload = json.load(f)
    except Exception:
        return {"ok": False, "reason": "undo_read_failed"}
    expires_ts = float(payload.get("expires_ts") or 0.0)
    if time.time() > expires_ts:
        return {"ok": False, "reason": "undo_window_expired"}
    prev_state = payload.get("state_before")
    if not isinstance(prev_state, dict):
        return {"ok": False, "reason": "invalid_undo_payload"}
    _write_incident_state(prev_state)
    _append_incident_audit({"kind": "undo", "action": payload.get("action")})
    return {"ok": True, "reason": "restored", "action": payload.get("action")}


def get_auto_runbook_policy() -> dict[str, Any]:
    return _load_json(
        _AUTO_POLICY_PATH,
        {
            "enabled": False,
            "min_level": "WARNING",
            "cooldown_sec": 900,
            "last_run_ts_by_type": {},
        },
    )


def set_auto_runbook_policy(enabled: bool | None = None, min_level: str | None = None, cooldown_sec: int | None = None) -> dict[str, Any]:
    p = get_auto_runbook_policy()
    if enabled is not None:
        p["enabled"] = bool(enabled)
    if min_level is not None:
        p["min_level"] = str(min_level).upper()
    if cooldown_sec is not None:
        p["cooldown_sec"] = max(60, int(cooldown_sec))
    _save_json(_AUTO_POLICY_PATH, p)
    return p


def run_auto_runbook_once() -> dict[str, Any]:
    p = get_auto_runbook_policy()
    if not p.get("enabled", False):
        return {"ran": False, "reason": "policy_disabled", "incident_id": None, "session_id": None}
    min_rank = _level_rank(str(p.get("min_level") or "WARNING"))
    cooldown_sec = int(p.get("cooldown_sec") or 900)
    last = dict(p.get("last_run_ts_by_type") or {})
    now = time.time()
    for inc in list_incidents(limit=50):
        if inc.get("status") == "resolved":
            continue
        itype = str(inc.get("type") or "unknown")
        if inc.get("muted"):
            continue
        snoozed_until = float(inc.get("snoozed_until") or 0.0)
        if snoozed_until > now:
            continue
        if _level_rank(str(inc.get("level") or "WARNING")) < min_rank:
            continue
        last_ts = float(last.get(itype) or 0.0)
        if now - last_ts < cooldown_sec:
            continue
        session = run_runbook_with_context(
            actions=list(DEFAULT_RUNBOOK_STEPS),
            context={
                "file": inc.get("file") or "",
                "query": inc.get("query") or "",
                "level": inc.get("level") or "",
                "source": "auto_runbook_policy",
                "incident_id": inc.get("incident_id"),
            },
        )
        last[itype] = now
        p["last_run_ts_by_type"] = last
        _save_json(_AUTO_POLICY_PATH, p)
        return {"ran": True, "reason": "executed", "incident_id": inc.get("incident_id"), "session_id": session.get("session_id")}
    return {"ran": False, "reason": "no_eligible_incident", "incident_id": None, "session_id": None}


def run_runbook_for_incident(incident_id: str) -> dict[str, Any]:
    for inc in list_incidents(limit=100):
        if str(inc.get("incident_id")) != str(incident_id):
            continue
        return run_runbook_with_context(
            actions=list(DEFAULT_RUNBOOK_STEPS),
            context={
                "file": inc.get("file") or "",
                "query": inc.get("query") or "",
                "level": inc.get("level") or "",
                "source": "incident_inbox_manual",
                "incident_id": incident_id,
            },
        )
    raise FileNotFoundError(str(incident_id))


def get_macro_preset() -> dict[str, Any]:
    p = _load_json(_MACRO_PRESET_PATH, DEFAULT_MACRO_PRESET)
    p.setdefault("owner", "local")
    p.setdefault("updated_ts", 0.0)
    return p


def set_macro_preset(
    preset: str | None = None,
    ack_filtered: bool | None = None,
    auto_run: bool | None = None,
    export_md: bool | None = None,
    refresh: bool | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    p = get_macro_preset()
    if preset is not None:
        p["preset"] = str(preset)
    if ack_filtered is not None:
        p["ackFiltered"] = bool(ack_filtered)
    if auto_run is not None:
        p["autoRun"] = bool(auto_run)
    if export_md is not None:
        p["exportMd"] = bool(export_md)
    if refresh is not None:
        p["refresh"] = bool(refresh)
    p["owner"] = str(owner or _incident_operator())[:64]
    p["updated_ts"] = time.time()
    _save_json(_MACRO_PRESET_PATH, p)
    return p


def update_runbook_session(session_id: str, tag: str | None = None, note: str | None = None) -> dict[str, Any]:
    safe = Path(session_id).name
    p = (_SESSION_DIR / f"{safe}.json").resolve()
    try:
        p.relative_to(_SESSION_DIR.resolve())
    except Exception:
        raise FileNotFoundError(safe)
    if not p.exists():
        raise FileNotFoundError(safe)
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    if tag is not None:
        obj["tag"] = str(tag)[:64]
    if note is not None:
        obj["note"] = str(note)[:4000]
    obj["updated_ts"] = time.time()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return obj


def build_session_timeline(session: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in session.get("steps") or []:
        rows.append(
            {
                "ts": s.get("started_ts"),
                "kind": "step_start",
                "title": f"Step start: {s.get('action', '-')}",
                "detail": "",
                "status": "running",
                "action": s.get("action"),
            }
        )
        rows.append(
            {
                "ts": s.get("ended_ts"),
                "kind": "step_end",
                "title": f"Step end: {s.get('action', '-')}",
                "detail": f"exit={s.get('exit_code')} duration={s.get('duration_sec')}",
                "status": "ok" if s.get("ok") else "fail",
                "action": s.get("action"),
            }
        )
    for sn in session.get("log_snippets") or []:
        rows.append(
            {
                "ts": session.get("ended_ts"),
                "kind": "log_snippet",
                "title": f"Log match: {sn.get('file', '-')}",
                "detail": sn.get("match", ""),
                "status": "info",
                "action": None,
                "line_no": sn.get("line_no"),
            }
        )
    rows.sort(key=lambda x: float(x.get("ts") or 0.0))
    return rows
