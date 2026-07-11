from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.liquidation_silence_policy import (
    SEVERITY_GREEN,
    SEVERITY_RED,
    SEVERITY_UNKNOWN,
    SEVERITY_YELLOW,
    compose_with_overall_severity,
)
from tools.native_ws_health_policy import evaluate_policy, read_source_freshness

ROOT = Path(__file__).resolve().parents[1]
LOG_HEALTH = ROOT / "logs" / "health"
REPORTS = ROOT / "reports"
DEFAULT_DB_PATH = ROOT / "data" / "microstructure.db"

# logs/health/overall.json is the canonical operational-health payload read by
# execution/health_gate.py, execution/entry_loop.py, dashboard/backend, and
# ami/host_health/observation.py. Every one of those consumers expects
# state in {"ok", "degraded", "halted"} (lowercase) -- never the RED/YELLOW/
# GREEN vocabulary used by reports/WATCHDOG_STATUS.json and logs/health/
# watchdog.json, which are a separate, additive, operator-facing pair of
# outputs owned by this same process. See CANONICAL_OVERALL_STATE_MAP.
STATE_OK = "ok"
STATE_DEGRADED = "degraded"
STATE_HALTED = "halted"
CANONICAL_OVERALL_STATE_MAP = {"GREEN": STATE_OK, "YELLOW": STATE_DEGRADED, "RED": STATE_HALTED}
# execution/health_gate.py's default ENTRY_HEALTH_MAX_STALENESS_SEC is 15s
# (execution/entry_loop.py:1217). The watchdog's --interval-sec must stay
# comfortably below that or the live-safety gate will see overall.json's
# ts_utc as stale and halt regardless of actual health.
#
# Measured 2026-07-10 against the live production watchdog (--interval-sec
# 10, 9 consecutive real inter-write deltas): actual cycle time was ~11.03s,
# not the nominal 10s -- evaluation itself (chiefly the ~1s
# python_process_running() PowerShell Get-CimInstance spawns) adds ~1.03s
# per cycle. That leaves only ~15-11.03=~3.97s of real margin (~1.36x, not
# the ~1.5x a naive 15/10 would suggest) before a consumer sees stale
# health. Reduced to 5s here: evaluation cost is the same ~1.03s regardless
# of --interval-sec, so real cycle time becomes ~6s, giving ~9s of real
# margin (~2.5x) -- a materially safer buffer against jitter/scheduling
# pauses, at the cost of roughly double the PowerShell subprocess spawns
# (still cheap relative to the 5s budget). See
# reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md Part D
# for the full measurement and boundary-test evidence
# (tests/runtime/test_health_gate_unit.py).
CANONICAL_STALENESS_BUDGET_SEC = 15
DEFAULT_INTERVAL_SEC = 5


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_z() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_ts(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        if " " in text and text.endswith("UTC"):
            text = text.replace(" UTC", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def age_sec(raw: Any, now: Optional[datetime] = None) -> Optional[float]:
    dt = parse_ts(raw)
    if dt is None:
        return None
    return max(0.0, ((now or utc_now()) - dt).total_seconds())


def read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp_{path.name}_{time.time_ns()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def python_process_running(needle: str) -> bool:
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return False
        payload = json.loads(proc.stdout)
        items = payload if isinstance(payload, list) else [payload]
        return any(needle in str(item.get("CommandLine") or "") for item in items if isinstance(item, dict))
    except Exception:
        return False


def component_fresh(payload: Dict[str, Any], max_age_sec: int, now: datetime) -> bool:
    ts_age = age_sec(payload.get("ts_utc"), now)
    return ts_age is not None and ts_age <= max_age_sec


# Components tools/heartbeat_watchdog.py does not itself evaluate but folds
# into canonical overall.json every cycle, read fresh from each owner's own
# dedicated file -- never preserved from a previous overall.json. Their
# presence in THIS dict is purely for surfacing each payload verbatim under
# overall.json's components[] -- inclusion here alone never sets top-level
# state, so a stale or missing entry here can never mask, nor fabricate, a
# RED/YELLOW verdict via _read_optional_components. (Top-level state is driven
# only by evaluate()'s critical/warning lists -- collector/bookticker/native-WS
# always, and liquidation_silence additionally, the latter folded in evaluate()
# via a SEPARATE fresh read of its own severity through
# tools.liquidation_silence_policy.compose_with_overall_severity, with the same
# never-downgrade precedence as native_ws; see that fold below.) Each entry:
# filename under logs/health/, and the module that owns writing it.
OPTIONAL_COMPONENT_FILES: Dict[str, str] = {
    "paper_trader": "paper_trader.json",  # owner: execution/health_gate.py
    "replay": "replay.json",  # owner: tools/replay_slice.py
    # owner: tools/liquidation_silence_detector.py (disabled-by-default one-shot
    # detector; its severity is additionally folded into top-level state in
    # evaluate() -- see the compose_with_overall_severity block there).
    "liquidation_silence": "liquidation_silence.json",
}


def _read_optional_components(log_health: Path) -> Dict[str, Any]:
    """Reads each optional component's own dedicated file fresh, every cycle.
    Missing or corrupt -> omitted entirely (read_json returns {} for either
    case, and an empty payload is never inserted); never fabricated, never
    carried forward from a prior overall.json. A present-but-stale
    component's own ts_utc is copied through verbatim -- readers can compute
    its age themselves -- so rewriting overall.json can never make a stale
    component look fresher than it is.
    """
    out: Dict[str, Any] = {}
    for name, filename in OPTIONAL_COMPONENT_FILES.items():
        payload = read_json(log_health / filename)
        if payload:
            out[name] = payload
    return out


def build_canonical_overall(
    *,
    overall: str,
    issues: list,
    collector_component: Dict[str, Any],
    bookticker_component: Dict[str, Any],
    native_ws_policy: Dict[str, Any],
    runtime_mode: Optional[str],
    now_iso: str,
    log_health: Path = LOG_HEALTH,
) -> Dict[str, Any]:
    """The single canonical logs/health/overall.json payload for one
    evaluation cycle. tools/heartbeat_watchdog.py is the sole authoritative
    writer of this file (see
    reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md).
    Every component is sourced fresh this cycle: collector/bookticker/
    watchdog from this process's own evaluation, everything else (see
    OPTIONAL_COMPONENT_FILES) read directly from its owner's dedicated file.
    Nothing is ever read back from a previous overall.json -- an unowned
    component that appeared in a stale prior overall.json is never silently
    carried forward once its writer stops running. runtime_mode falling back
    to "paper" (never to a remembered past mode) is deliberate: an unknown
    current mode must never be allowed to echo a possibly-stale "live".
    """
    components: Dict[str, Any] = _read_optional_components(log_health)
    components["collector"] = collector_component
    components["bookticker"] = bookticker_component
    components["watchdog"] = {
        "status": STATE_OK,
        "connected": True,
        "last_progress_ts_utc": now_iso,
        "progress_lag_sec": 0,
        "reconnects_last_5m": 0,
        "errors_last_5m": 0,
    }

    return {
        "ts_utc": now_iso,
        "mode": runtime_mode or "paper",
        "state": CANONICAL_OVERALL_STATE_MAP.get(overall, STATE_HALTED),
        "reason": "; ".join(issues),
        "reasons": list(issues),
        "components": components,
        "native_ws_status": native_ws_policy["status"],
        "native_ws_reasons": native_ws_policy["reasons"],
        "native_websocket": native_ws_policy["native_websocket"],
        "rest_fallback": native_ws_policy["rest_fallback"],
        "source_freshness": native_ws_policy["source_freshness"],
        "native_ws_thresholds": native_ws_policy["thresholds"],
    }


def evaluate(
    max_age_sec: int = 180,
    expect_bookticker: bool = True,
    expect_detector: bool = True,
    expect_runtime: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    now = utc_now()
    # Resolved against the current ROOT (not snapshotted as a default arg) so
    # tests that monkeypatch ROOT for isolation don't fall through to the real
    # production database.
    resolved_db_path = db_path if db_path is not None else (ROOT / "data" / "microstructure.db")
    collector = read_json(LOG_HEALTH / "collector.json")
    collector_heartbeat = read_json(ROOT / "logs" / "collector_heartbeat.json")
    bookticker = read_json(LOG_HEALTH / "bookticker.json")
    runtime = read_json(ROOT / "logs" / "runtime_launcher_status.json")

    collector_fresh = component_fresh(collector, max_age_sec, now)
    collector_progress_age = age_sec(collector.get("last_progress_ts_utc"), now)
    collector_process = python_process_running("data.microstructure_collector")
    collector_progressing = collector_progress_age is not None and collector_progress_age <= max_age_sec
    collector_healthy = bool(
        collector_process
        and collector_fresh
        and collector_progressing
        and str(collector.get("status") or "").lower() == "ok"
        and bool(collector.get("required_streams_progressing", True))
    )

    bookticker_fresh = component_fresh(bookticker, max_age_sec, now)
    tick_age = bookticker.get("last_tick_event_age_sec")
    try:
        tick_age_f = float(tick_age) if tick_age is not None else None
    except Exception:
        tick_age_f = None
    bookticker_process = python_process_running("bookticker_collector")
    bookticker_healthy = bool(
        bookticker_process
        and bookticker_fresh
        and str(bookticker.get("status") or "").lower() == "ok"
        and bool(bookticker.get("connected", False))
        and (tick_age_f is None or tick_age_f <= max_age_sec)
    )

    runtime_fresh = component_fresh(runtime, max_age_sec, now)
    runtime_mode = runtime.get("active_mode") or runtime.get("requested_mode")
    runtime_healthy = bool(
        runtime_fresh
        and str(runtime.get("status") or "").lower() == "ready"
        and str(runtime_mode or "").lower() in {"live", "paper"}
        and not bool(runtime.get("fallback_to_paper", False))
    )

    detector_healthy = python_process_running("tools/run_detector.py")

    source_freshness = read_source_freshness(resolved_db_path, now=now.timestamp())
    native_ws_policy = evaluate_policy(
        collector_heartbeat=collector_heartbeat,
        collector_component=collector,
        collector_process_alive=collector_process,
        source_freshness=source_freshness,
        now=now,
    )

    critical = []
    warning = []
    if not collector_process:
        critical.append("collector_process_down")
    elif not collector_healthy:
        warning.append("collector_degraded")
    if expect_bookticker and not bookticker_process:
        critical.append("bookticker_process_down")
    elif expect_bookticker and not bookticker_healthy:
        warning.append("bookticker_degraded")
    if expect_runtime and not runtime_healthy:
        warning.append("runtime_not_ready")
    if expect_detector and not detector_healthy:
        warning.append("detector_not_seen")

    # Merge the native-WS-aware policy into the existing RED/YELLOW/GREEN enum
    # (deterministic precedence RED > YELLOW > GREEN, same as the block below):
    # native RED always forces top-level RED; native DEGRADED raises top-level
    # GREEN to at least YELLOW; native GREEN never downgrades an existing
    # RED/YELLOW caused by something else. This is what prevents the exact
    # 2026-07-06..07-10 incident class from ever reading as top-level GREEN
    # again -- previously only the separate, additive "native_ws_status" field
    # carried this signal and "overall" (the field existing consumers such as
    # execution/belief_controller.py actually read) was untouched by it.
    if native_ws_policy["status"] == "RED":
        critical.append("native_ws_red:" + ",".join(native_ws_policy["reasons"]))
    elif native_ws_policy["status"] == "DEGRADED":
        warning.append("native_ws_degraded:" + ",".join(native_ws_policy["reasons"]))

    if critical:
        overall = "RED"
        severity = "critical"
    elif warning:
        overall = "YELLOW"
        severity = "warning"
    else:
        overall = "GREEN"
        severity = "info"

    # Fold the liquidation-silence detector's own severity axis into the
    # top-level RED/YELLOW/GREEN enum with the SAME never-downgrade precedence
    # already applied to native_ws above, via the detector policy's canonical
    # compose_with_overall_severity(): detector RED forces top-level RED,
    # detector YELLOW/UNKNOWN raise top-level to at least YELLOW, detector GREEN
    # never downgrades a RED/YELLOW verdict already in force. The detector owns
    # and writes its own dedicated component file
    # (logs/health/liquidation_silence.json); it is read fresh here every cycle
    # -- never fabricated, never carried forward from a prior overall.json.
    # Until the (disabled-by-default) detector process is actually run, this
    # file is absent -> severity defaults to GREEN -> compose is a pure no-op,
    # so wiring it in can never by itself change any verdict. When present its
    # non-green severity is also recorded in critical/warning (hence in
    # issues/reasons and errors_last_5m) exactly like native_ws.
    liq_silence = read_json(LOG_HEALTH / "liquidation_silence.json")
    liq_silence_severity = str((liq_silence or {}).get("severity") or SEVERITY_GREEN).upper()
    if liq_silence_severity not in {SEVERITY_GREEN, SEVERITY_YELLOW, SEVERITY_RED, SEVERITY_UNKNOWN}:
        liq_silence_severity = SEVERITY_GREEN
    composed_overall = compose_with_overall_severity(overall, liq_silence_severity)
    if liq_silence_severity != SEVERITY_GREEN:
        liq_reasons = liq_silence.get("reason_codes") if isinstance(liq_silence, dict) else None
        detail = ",".join(str(r) for r in liq_reasons) if isinstance(liq_reasons, list) and liq_reasons else ""
        reason_code = "liquidation_silence_" + liq_silence_severity.lower() + ((":" + detail) if detail else "")
        if composed_overall == "RED":
            critical.append(reason_code)
        else:
            warning.append(reason_code)
    overall = composed_overall
    severity = "critical" if overall == "RED" else ("warning" if overall == "YELLOW" else "info")

    out = {
        "ts_utc": utc_now_z(),
        "overall": overall,
        "collector_healthy": collector_healthy,
        "collector_alive": collector_process,
        "collector_status": collector.get("status"),
        "collector_progress_lag_sec": collector_progress_age,
        "collector_transport_connected": collector.get("transport_connected"),
        "collector_required_streams_progressing": collector.get("required_streams_progressing"),
        "detector_healthy": detector_healthy,
        "bookticker_healthy": bookticker_healthy,
        "bookticker_alive": bookticker_process,
        "bookticker_tick_age_sec": tick_age_f,
        "runtime_healthy": runtime_healthy,
        "runtime_mode": runtime_mode,
        "watchdog_expected": True,
        "expect_bookticker": bool(expect_bookticker),
        "expect_detector": bool(expect_detector),
        "expect_runtime": bool(expect_runtime),
        "dq": "GREEN" if not critical else None,
        "severity": severity,
        "issues": critical + warning,
        # Additive fields below -- native-WS-aware status, independent of the
        # REST-blended "overall" above. See native_ws_policy for precedence
        # and reason codes; existing "overall"/"issues"/etc. are unchanged.
        "native_ws_status": native_ws_policy["status"],
        "native_ws_reasons": native_ws_policy["reasons"],
        "native_websocket": native_ws_policy["native_websocket"],
        "rest_fallback": native_ws_policy["rest_fallback"],
        "source_freshness": native_ws_policy["source_freshness"],
        "native_ws_thresholds": native_ws_policy["thresholds"],
    }
    health = {
        "status": "ok" if overall == "GREEN" else "degraded",
        "connected": overall == "GREEN",
        "last_progress_ts_utc": out["ts_utc"],
        "progress_lag_sec": 0,
        "reconnects_last_5m": 0,
        "errors_last_5m": len(critical + warning),
        "overall": overall,
        "runtime_mode": runtime_mode,
        "detector_healthy": detector_healthy,
        "collector_healthy": collector_healthy,
        "bookticker_healthy": bookticker_healthy,
        "ts_utc": out["ts_utc"],
        "issues": critical + warning,
        "native_ws_status": native_ws_policy["status"],
        "native_ws_reasons": native_ws_policy["reasons"],
    }
    overall_canonical = build_canonical_overall(
        overall=overall,
        issues=critical + warning,
        collector_component=collector,
        bookticker_component=bookticker,
        native_ws_policy=native_ws_policy,
        runtime_mode=runtime_mode,
        now_iso=out["ts_utc"],
        # LOG_HEALTH resolved here (call time, current module global) rather
        # than relying on build_canonical_overall's own default arg, so tests
        # that monkeypatch hw.LOG_HEALTH for isolation are honored correctly.
        log_health=LOG_HEALTH,
    )
    return {"report": out, "health": health, "overall": overall_canonical}


def run_once(
    max_age_sec: int,
    expect_bookticker: bool = True,
    expect_detector: bool = True,
    expect_runtime: bool = True,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    result = evaluate(
        max_age_sec=max_age_sec,
        expect_bookticker=expect_bookticker,
        expect_detector=expect_detector,
        expect_runtime=expect_runtime,
        db_path=db_path,
    )
    # All three outputs are derived from the same in-memory evaluation result
    # in this one cycle, so they cannot disagree in severity with each other.
    atomic_write(REPORTS / "WATCHDOG_STATUS.json", result["report"])
    atomic_write(LOG_HEALTH / "watchdog.json", result["health"])
    atomic_write(LOG_HEALTH / "overall.json", result["overall"])
    return result["report"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Eclipse heartbeat watchdog")
    parser.add_argument("--interval-sec", type=int, default=DEFAULT_INTERVAL_SEC)
    parser.add_argument("--max-age-sec", type=int, default=180)
    parser.add_argument("--expect-bookticker", action="store_true")
    parser.add_argument("--expect-detector", action="store_true")
    parser.add_argument("--expect-runtime", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB_PATH))
    args = parser.parse_args()
    while True:
        report = run_once(
            max_age_sec=max(30, int(args.max_age_sec)),
            expect_bookticker=bool(args.expect_bookticker),
            expect_detector=bool(args.expect_detector),
            expect_runtime=bool(args.expect_runtime),
            db_path=Path(str(args.db_path)),
        )
        print(json.dumps(report, ensure_ascii=True, separators=(",", ":")), flush=True)
        if args.once:
            return 0
        time.sleep(max(5, int(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
