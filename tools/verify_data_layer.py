from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.data_layer_probe import is_heartbeat_fresh, parse_heartbeat_json


def _powershell_json(command: str) -> Any:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "powershell failed")
    out = (proc.stdout or "").strip()
    if not out:
        return []
    return json.loads(out)


def _list_python_processes() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    # Prefer psutil when available (works well in restricted shells).
    try:
        import psutil  # type: ignore

        rows: List[Dict[str, Any]] = []
        for proc in psutil.process_iter(["name", "pid", "cmdline"]):
            name = str((proc.info or {}).get("name") or "").lower()
            if name not in {"python.exe", "python"}:
                continue
            cmdline = (proc.info or {}).get("cmdline") or []
            cl = " ".join(str(x) for x in cmdline if x is not None)
            rows.append({"ProcessId": int((proc.info or {}).get("pid")), "CommandLine": cl})
        return rows, None
    except Exception:
        pass

    # Fallback: CIM
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    try:
        obj = _powershell_json(cmd)
        if isinstance(obj, dict):
            return [obj], None
        if isinstance(obj, list):
            return obj, None
        return [], "process_scan_empty"
    except Exception as exc:
        return [], f"process_scan_failed:{exc}"


def _list_python_processes_from_meta(pid_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mapping = [
        pid_root / "microstructure_collector.json",
        pid_root / "event_diary.json",
    ]
    for meta_path in mapping:
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
            pid = int((payload or {}).get("pid") or 0)
            cmd = str((payload or {}).get("cmdline_sig") or "")
            if pid <= 0 or not cmd:
                continue
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", f"Get-Process -Id {pid} | Out-Null"],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                rows.append({"ProcessId": pid, "CommandLine": cmd})
        except Exception:
            continue
    return rows


def _find_by_substring(rows: List[Dict[str, Any]], needle: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        cl = str(r.get("CommandLine") or "")
        if needle in cl:
            out.append({"pid": int(r.get("ProcessId")), "command_line": cl})
    return out


def _file_stats(path: Path) -> Tuple[int, float]:
    st = path.stat()
    return int(st.st_size), float(st.st_mtime)


def verify(
    db_path: Path,
    csv_path: Path,
    wait_sec: int,
    min_db_growth_bytes: int,
) -> Tuple[bool, Dict[str, str]]:
    details: Dict[str, str] = {}
    fresh_window_sec = max(60, int(wait_sec) * 30)

    procs, scan_error = _list_python_processes()
    if not procs:
        meta_rows = _list_python_processes_from_meta(db_path.parent.parent / "logs" / "pids")
        if meta_rows:
            procs = meta_rows
            scan_error = (f"{scan_error};" if scan_error else "") + "process_scan_meta_fallback"
    micro = _find_by_substring(procs, "data.microstructure_collector")
    diary = _find_by_substring(procs, "data.event_diary")
    details["microstructure_collector_count"] = str(len(micro))
    details["event_diary_count"] = str(len(diary))
    if scan_error:
        details["process_scan"] = scan_error

    if not db_path.exists():
        details["db"] = f"missing:{db_path}"
        return False, details
    if not csv_path.exists():
        details["csv"] = f"missing:{csv_path}"
        return False, details

    db_size_0, db_mtime_0 = _file_stats(db_path)
    csv_size_0, csv_mtime_0 = _file_stats(csv_path)
    time.sleep(max(1, int(wait_sec)))
    db_size_1, db_mtime_1 = _file_stats(db_path)
    csv_size_1, csv_mtime_1 = _file_stats(csv_path)

    db_growth = db_size_1 - db_size_0
    csv_mtime_delta = csv_mtime_1 - csv_mtime_0
    details["db_size_before"] = str(db_size_0)
    details["db_size_after"] = str(db_size_1)
    details["db_growth_bytes"] = str(db_growth)
    details["csv_mtime_before"] = str(int(csv_mtime_0))
    details["csv_mtime_after"] = str(int(csv_mtime_1))
    details["csv_mtime_delta_sec"] = str(int(csv_mtime_delta))

    now = time.time()
    details["csv_age_sec"] = str(int(max(0.0, now - csv_mtime_1)))
    details["db_age_sec"] = str(int(max(0.0, now - db_mtime_1)))

    heartbeat_path = db_path.parent.parent / "logs" / "collector_heartbeat.json"
    heartbeat_fresh = False
    if heartbeat_path.exists():
        try:
            hb = parse_heartbeat_json(heartbeat_path.read_text(encoding="utf-8", errors="ignore"))
            if hb:
                heartbeat_fresh = is_heartbeat_fresh(hb, fresh_sec=fresh_window_sec, now_ts=now)
        except Exception:
            heartbeat_fresh = False
    collector_data_live = db_growth >= int(min_db_growth_bytes) or heartbeat_fresh or (max(0.0, now - db_mtime_1) <= fresh_window_sec)
    diary_data_live = csv_mtime_delta > 0 or (max(0.0, now - csv_mtime_1) <= fresh_window_sec)
    details["collector_data_live"] = str(bool(collector_data_live)).lower()
    details["diary_data_live"] = str(bool(diary_data_live)).lower()
    details["heartbeat_fresh"] = str(bool(heartbeat_fresh)).lower()

    if not micro:
        details["proc_micro"] = "not_running"
    if not diary:
        details["proc_diary"] = "not_running"
    if db_growth < int(min_db_growth_bytes):
        details["db_growth_check"] = f"insufficient_growth(min={min_db_growth_bytes})"
    if csv_mtime_delta <= 0:
        details["csv_update_check"] = "mtime_not_updated"

    diary_live = bool(diary) or diary_data_live
    details["diary_live"] = str(bool(diary_live)).lower()
    if bool(micro) and diary_live and collector_data_live and diary_data_live:
        details["overall_status"] = "running"
        ok = True
    elif (bool(micro) and collector_data_live) or diary_live or diary_data_live:
        details["overall_status"] = "degraded"
        ok = False
    else:
        details["overall_status"] = "stopped"
        ok = False
    return ok, details


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify microstructure data layer processes and writes.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--wait-sec", type=int, default=5)
    p.add_argument("--min-db-growth-bytes", type=int, default=1)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ok, details = verify(
            db_path=Path(str(args.db)),
            csv_path=Path(str(args.csv)),
            wait_sec=int(args.wait_sec),
            min_db_growth_bytes=int(args.min_db_growth_bytes),
        )
    except Exception as exc:
        print(f"FAIL data_layer_not_running err={exc}")
        return 3

    status = "OK data_layer_running" if ok else "FAIL data_layer_not_running"
    print(status)
    for k in sorted(details):
        print(f"{k}={details[k]}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
