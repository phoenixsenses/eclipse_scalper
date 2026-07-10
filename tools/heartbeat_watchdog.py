from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from tools.native_ws_health_policy import evaluate_policy, read_source_freshness

ROOT = Path(__file__).resolve().parents[1]
LOG_HEALTH = ROOT / "logs" / "health"
REPORTS = ROOT / "reports"
DEFAULT_DB_PATH = ROOT / "data" / "microstructure.db"


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
    return {"report": out, "health": health}


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
    atomic_write(REPORTS / "WATCHDOG_STATUS.json", result["report"])
    atomic_write(LOG_HEALTH / "watchdog.json", result["health"])
    return result["report"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Eclipse heartbeat watchdog")
    parser.add_argument("--interval-sec", type=int, default=30)
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
