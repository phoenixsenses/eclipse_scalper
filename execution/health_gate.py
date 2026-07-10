from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from tools.health_state import write_component_health, utc_now_iso
except Exception:  # pragma: no cover
    write_component_health = None  # type: ignore
    utc_now_iso = None  # type: ignore


@dataclass
class GateState:
    degraded_since: float = 0.0
    halted: bool = False
    last_reason: str = ""
    halt_until_ts: float = 0.0
    last_ingestion_check_ts: float = 0.0
    last_ingestion_ok: Optional[bool] = None
    last_ingestion_reason: str = ""


@dataclass
class GateDecision:
    allow: bool
    reason: str
    state: str
    collector_connected: Optional[bool]
    collector_lag_sec: Optional[int]
    reconnects_last_5m: int
    errors_last_5m: int
    halt_until_ts: float = 0.0
    ingestion_checked: bool = False
    ingestion_ok: Optional[bool] = None
    ingestion_reason: str = ""


def _parse_ts(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    text = str(ts).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def load_overall_health(path: str | Path = "logs/health/overall.json") -> Dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _paper_contract_fields() -> Dict[str, Any]:
    env_profile = str(os.environ.get("SCALPER_ENV_PROFILE", "") or "").strip().lower()
    dry_run = _env_truthy("SCALPER_DRY_RUN", False)
    paper_profile_active = dry_run or env_profile == "paper"
    paper_execution_mode = str(os.environ.get("PAPER_EXECUTION_MODE", "") or "").strip().lower()
    allow_live_private_api = _env_truthy("PAPER_ALLOW_LIVE_PRIVATE_API", False)
    binance_testnet = _env_truthy("BINANCE_TESTNET", False)
    private_key_present = bool(str(os.environ.get("BINANCE_API_KEY", "") or os.environ.get("BINANCE_KEY", "") or "").strip())
    private_secret_present = bool(str(os.environ.get("BINANCE_API_SECRET", "") or os.environ.get("BINANCE_SECRET", "") or "").strip())
    private_api_present = private_key_present and private_secret_present

    startup_contract_safe = True
    startup_contract_reason = ""
    if env_profile and env_profile != "paper":
        startup_contract_safe = False
        startup_contract_reason = "env_profile_not_paper"
    elif not paper_profile_active:
        startup_contract_safe = False
        startup_contract_reason = "paper_profile_inactive"
    elif allow_live_private_api:
        startup_contract_safe = False
        startup_contract_reason = "paper_allows_live_private_api"
    elif private_api_present and not binance_testnet:
        startup_contract_safe = False
        startup_contract_reason = "paper_private_api_non_testnet"

    return {
        "env_profile": env_profile or None,
        "paper_profile_active": paper_profile_active,
        "paper_execution_mode": paper_execution_mode or None,
        "binance_testnet": binance_testnet,
        "paper_allow_live_private_api": allow_live_private_api,
        "startup_contract_safe": startup_contract_safe,
        "startup_contract_reason": startup_contract_reason,
    }


def evaluate_health_gate(
    health_obj: Dict[str, Any] | None,
    gate_state: GateState,
    *,
    now_ts: Optional[float] = None,
    max_health_staleness_sec: int = 15,
    max_collector_lag_sec: int = 30,
    max_reconnects_5m: int = 10,
    max_errors_5m: int = 10,
    max_degraded_sec: int = 120,
    halt_cooldown_sec: int = 60,
    use_ingestion_check: bool = False,
    ingestion_probe: Optional[Callable[[], tuple[bool, str]]] = None,
    ingestion_check_cooldown_sec: int = 10,
) -> GateDecision:
    now = float(time.time() if now_ts is None else now_ts)
    if gate_state.halted and gate_state.halt_until_ts > now:
        return GateDecision(
            False,
            "reconnect_escalation_cooldown",
            "halted",
            None,
            None,
            0,
            0,
            halt_until_ts=float(gate_state.halt_until_ts),
            ingestion_checked=False,
            ingestion_ok=gate_state.last_ingestion_ok,
            ingestion_reason=gate_state.last_ingestion_reason,
        )
    if gate_state.halted and gate_state.halt_until_ts > 0 and now >= gate_state.halt_until_ts:
        gate_state.halted = False
        gate_state.halt_until_ts = 0.0
    if not isinstance(health_obj, dict):
        gate_state.halted = True
        gate_state.last_reason = "health_missing"
        return GateDecision(False, "health_missing", "halted", None, None, 0, 0, halt_until_ts=0.0)
    ts = _parse_ts(health_obj.get("ts_utc"))
    if ts is None or (now - ts) > float(max_health_staleness_sec):
        gate_state.halted = True
        gate_state.last_reason = "health_stale"
        return GateDecision(False, "health_stale", "halted", None, None, 0, 0, halt_until_ts=0.0)

    state = str(health_obj.get("state") or "").lower().strip()
    components = health_obj.get("components") if isinstance(health_obj.get("components"), dict) else {}
    collector = components.get("collector") if isinstance(components.get("collector"), dict) else {}
    connected = collector.get("connected")
    lag_val = collector.get("progress_lag_sec")
    lag_sec = int(lag_val) if isinstance(lag_val, (int, float)) else None
    reconnects = int(collector.get("reconnects_last_5m", 0) or 0)
    errors = int(collector.get("errors_last_5m", 0) or 0)

    # Escalation paths
    if reconnects >= int(max_reconnects_5m) or errors >= int(max_errors_5m):
        gate_state.halted = True
        gate_state.last_reason = "reconnect_escalation"
        gate_state.halt_until_ts = now + max(0.0, float(halt_cooldown_sec))
        return GateDecision(
            False,
            "reconnect_escalation",
            "halted",
            connected if isinstance(connected, bool) else None,
            lag_sec,
            reconnects,
            errors,
            halt_until_ts=float(gate_state.halt_until_ts),
        )

    degraded_reason = ""
    if state != "ok":
        degraded_reason = "overall_not_ok"
    elif connected is not True:
        degraded_reason = "collector_disconnected"
    elif lag_sec is None or lag_sec > int(max_collector_lag_sec):
        degraded_reason = "data_stale"

    if degraded_reason:
        if gate_state.degraded_since <= 0:
            gate_state.degraded_since = now
        if (now - gate_state.degraded_since) >= float(max_degraded_sec):
            gate_state.halted = True
            gate_state.last_reason = "reconnect_escalation"
            gate_state.halt_until_ts = now + max(0.0, float(halt_cooldown_sec))
            return GateDecision(
                False,
                "reconnect_escalation",
                "halted",
                connected if isinstance(connected, bool) else None,
                lag_sec,
                reconnects,
                errors,
                halt_until_ts=float(gate_state.halt_until_ts),
            )
        gate_state.last_reason = degraded_reason
        return GateDecision(False, degraded_reason, "degraded", connected if isinstance(connected, bool) else None, lag_sec, reconnects, errors, halt_until_ts=float(gate_state.halt_until_ts))

    ingestion_checked = False
    ingestion_ok = gate_state.last_ingestion_ok
    ingestion_reason = gate_state.last_ingestion_reason
    if use_ingestion_check and callable(ingestion_probe):
        if (now - float(gate_state.last_ingestion_check_ts or 0.0)) >= float(max(1, int(ingestion_check_cooldown_sec))):
            ingestion_checked = True
            gate_state.last_ingestion_check_ts = now
            try:
                ok, reason = ingestion_probe()
            except Exception as e:
                ok, reason = False, f"probe_error:{type(e).__name__}"
            gate_state.last_ingestion_ok = bool(ok)
            gate_state.last_ingestion_reason = str(reason or "")
            ingestion_ok = gate_state.last_ingestion_ok
            ingestion_reason = gate_state.last_ingestion_reason
        if gate_state.last_ingestion_ok is False:
            gate_state.last_reason = "ingestion_stalled"
            return GateDecision(
                False,
                "ingestion_stalled",
                "degraded",
                connected if isinstance(connected, bool) else None,
                lag_sec,
                reconnects,
                errors,
                halt_until_ts=float(gate_state.halt_until_ts),
                ingestion_checked=ingestion_checked,
                ingestion_ok=ingestion_ok,
                ingestion_reason=ingestion_reason,
            )

    gate_state.degraded_since = 0.0
    gate_state.halted = False
    gate_state.last_reason = ""
    return GateDecision(
        True,
        "",
        "ok",
        connected if isinstance(connected, bool) else None,
        lag_sec,
        reconnects,
        errors,
        halt_until_ts=float(gate_state.halt_until_ts),
        ingestion_checked=ingestion_checked,
        ingestion_ok=ingestion_ok,
        ingestion_reason=ingestion_reason,
    )


def write_paper_trader_health(
    decision: GateDecision,
    reason: str,
    *,
    root: str | Path = "logs/health",
) -> None:
    """Writes only this component's own dedicated file
    (logs/health/paper_trader.json). logs/health/overall.json is owned
    solely by tools/heartbeat_watchdog.py, which reads this file directly
    each cycle -- see reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md.
    This function must never read-modify-write overall.json: two writers
    doing read-merge-write against the same file is a lost-update race no
    atomic file replacement can prevent."""
    if not callable(write_component_health):
        return
    status = "ok" if decision.allow else ("halted" if decision.state == "halted" else "degraded")
    contract_fields = _paper_contract_fields()
    payload = {
        "status": status,
        "connected": decision.allow,
        "last_progress_ts_utc": (utc_now_iso() if callable(utc_now_iso) else None),
        "progress_lag_sec": decision.collector_lag_sec,
        "reconnects_last_5m": int(decision.reconnects_last_5m),
        "errors_last_5m": int(decision.errors_last_5m),
        "reason": reason,
        **contract_fields,
    }
    try:
        write_component_health("paper_trader", payload, root=root)
    except Exception:
        return
