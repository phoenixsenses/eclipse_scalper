from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_TOP_LEVEL_KEYS = ("ts_utc", "mode", "state", "components")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline health-cycle smoke: assert collector health transitions ok -> degraded -> ok."
    )
    p.add_argument("--db-path", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--cycle-sec", type=float, default=10.0)
    p.add_argument("--down-sec", type=float, default=4.0)
    p.add_argument("--max-seconds", type=float, default=22.0)
    p.add_argument("--stats-interval", type=float, default=2.0)
    p.add_argument("--out-dir", default="logs/health/smoke")
    p.add_argument("--health-file", default="logs/health/overall.json")
    p.add_argument("--poll-interval-sec", type=float, default=0.5)
    return p


def _read_health(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _collector_state(obj: Dict[str, Any]) -> tuple[str, Optional[bool]]:
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    collector = comps.get("collector") if isinstance(comps.get("collector"), dict) else {}
    status = str(collector.get("status") or "").lower()
    connected = collector.get("connected")
    if isinstance(connected, bool):
        return status, connected
    return status, None


def _validate_snapshot(obj: Dict[str, Any], expected_state: str) -> tuple[bool, str]:
    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in obj:
            return False, f"missing_key:{key}"
    state = str(obj.get("state") or "").lower()
    if state != expected_state:
        return False, f"state_mismatch:expected={expected_state}:actual={state}"
    c_status, c_connected = _collector_state(obj)
    if expected_state == "ok":
        if c_status != "ok" or c_connected is not True:
            return False, f"collector_mismatch_ok:status={c_status}:connected={c_connected}"
    elif expected_state == "degraded":
        if c_status != "degraded" or c_connected is not False:
            return False, f"collector_mismatch_degraded:status={c_status}:connected={c_connected}"
    return True, "ok"


def _extract_reconnects(obj: Dict[str, Any]) -> int:
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    collector = comps.get("collector") if isinstance(comps.get("collector"), dict) else {}
    try:
        return int(collector.get("reconnects_last_5m", 0))
    except Exception:
        return 0


def _cleanup_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def run_smoke(args: argparse.Namespace) -> int:
    out_dir = Path(str(args.out_dir))
    health_file = Path(str(args.health_file))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("overall_snapshot*.json"):
        f.unlink(missing_ok=True)

    pid_file = Path("logs/pids/paper_watchdog.pid")
    meta_file = Path("logs/pids/paper_watchdog.json")
    pid_exists_before = pid_file.exists()
    meta_exists_before = meta_file.exists()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "data.microstructure_collector",
        "--symbols",
        str(args.symbols),
        "--db-path",
        str(args.db_path),
        "--stats-interval",
        str(args.stats_interval),
        "--simulate-connection",
        "--simulate-cycle-sec",
        str(args.cycle_sec),
        "--simulate-down-sec",
        str(args.down_sec),
        "--simulate-max-seconds",
        str(args.max_seconds),
    ]

    proc: Optional[subprocess.Popen] = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        end_ts = time.time() + float(args.max_seconds) + 5.0
        last_obj: Optional[Dict[str, Any]] = None
        phase = 0
        reconnect_marks: list[int] = []
        snapshots: list[Path] = []
        file_names = [
            "overall_snapshot1_ok.json",
            "overall_snapshot2_degraded.json",
            "overall_snapshot3_ok.json",
        ]
        expected_states = ["ok", "degraded", "ok"]

        while time.time() < end_ts:
            obj = _read_health(health_file)
            if isinstance(obj, dict):
                last_obj = obj
                want = expected_states[phase]
                ok, reason = _validate_snapshot(obj, want)
                if ok:
                    snap_path = out_dir / file_names[phase]
                    snap_path.write_text(json.dumps(obj, ensure_ascii=True), encoding="utf-8")
                    snapshots.append(snap_path)
                    reconnect_marks.append(_extract_reconnects(obj))
                    phase += 1
                    if phase >= 3:
                        break
            if proc.poll() is not None and phase < 3:
                break
            time.sleep(max(0.1, float(args.poll_interval_sec)))

        if phase < 3:
            last_state = None if last_obj is None else last_obj.get("state")
            print(f"health_cycle_smoke fail transition_timeout phase={phase} last_state={last_state}")
            return 1

        if not (reconnect_marks[0] <= reconnect_marks[1] <= reconnect_marks[2]):
            print(
                "health_cycle_smoke fail reconnects_non_monotonic "
                f"values={reconnect_marks}"
            )
            return 1

        if (not pid_exists_before and pid_file.exists()) or (not meta_exists_before and meta_file.exists()):
            print("health_cycle_smoke fail watchdog_registry_created_unexpectedly")
            return 1

        print("health_cycle_smoke ok")
        for p in snapshots:
            print(f"- snapshot={p}")
        return 0
    except AssertionError as e:
        print(f"health_cycle_smoke fail assertion={e}")
        return 1
    except Exception as e:
        print(f"health_cycle_smoke error runtime={type(e).__name__}:{e}")
        return 2
    finally:
        _cleanup_process(proc)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())

