from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.check_data_ready import check_db_fresh


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _powershell_json(command: str) -> Any:
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "powershell command failed")
    payload = (proc.stdout or "").strip()
    if not payload:
        return []
    return json.loads(payload)


def find_collector_processes(match_substring: str = "data.microstructure_collector") -> List[Dict[str, Any]]:
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
        "| Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    )
    obj = _powershell_json(cmd)
    items = obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
    out: List[Dict[str, Any]] = []
    for item in items:
        pid = item.get("ProcessId")
        cl = str(item.get("CommandLine") or "")
        if not pid or not cl:
            continue
        if match_substring not in cl:
            continue
        out.append({"pid": int(pid), "command_line": cl})
    return out


def _append_watchdog_log(
    log_path: Path,
    action: str,
    pids: List[int],
    start_cmd: str,
    details: Dict[str, Any],
) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "pids": pids,
            "start_cmd": start_cmd,
            "details": details,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        # best-effort only
        return


def _kill_pids(pids: List[int]) -> None:
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, text=True, check=False)


def _restart_collector(start_cmd: str) -> None:
    # New console window, detached. Use cmd /k so operators can see startup output.
    subprocess.Popen(
        ["powershell", "-NoProfile", "-Command", f"Start-Process powershell -ArgumentList '-NoProfile -Command {start_cmd}'"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watchdog for microstructure collector (manual operator tool).")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--fresh-sec", type=int, default=120)
    p.add_argument("--kill-after-sec", type=int, default=600)
    p.add_argument(
        "--start-cmd",
        default="python -W ignore -u -m data.microstructure_collector --symbols BTCUSDT,ETHUSDT",
    )
    p.add_argument("--apply", action="store_true", help="Apply kill/restart actions. Default is dry-run.")
    p.add_argument("--log", default="logs/micro_collector_watchdog.log")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    db_path = Path(str(args.db))
    symbols = _parse_symbols(args.symbols)
    if not db_path.exists():
        print(f"FAIL watchdog db_missing path={db_path}")
        return 3
    try:
        conn = sqlite3.connect(str(db_path))
    except Exception as exc:
        print(f"FAIL watchdog db_open_failed path={db_path} err={exc}")
        return 3

    try:
        ok, details, table_diags = check_db_fresh(conn, symbols=symbols, fresh_sec=int(args.fresh_sec))
    except Exception as exc:
        print(f"FAIL watchdog freshness_error err={exc}")
        return 3
    finally:
        conn.close()

    for key in sorted(details):
        print(f"{key}={details[key]}")

    if ok:
        print("OK watchdog data_fresh")
        return 0

    table_ages = [d.age_sec for d in table_diags if d.age_sec is not None]
    max_age = max(table_ages) if table_ages else None
    print(f"STALE watchdog max_age_sec={max_age if max_age is not None else '-'}")

    try:
        matches = find_collector_processes("data.microstructure_collector")
    except Exception as exc:
        print(f"FAIL watchdog process_scan_error err={exc}")
        return 3
    pids = [int(m["pid"]) for m in matches]
    print(f"collector_match_count={len(matches)}")
    for m in matches:
        print(f"collector_pid={m['pid']} cmd={m['command_line']}")

    kill_after_sec = int(max(1, args.kill_after_sec))
    should_restart = max_age is not None and max_age >= kill_after_sec
    if not should_restart:
        print(f"STALE watchdog below_kill_threshold kill_after_sec={kill_after_sec}")
        return 2

    if not args.apply:
        print("DRY_RUN watchdog would_kill_and_restart=1")
        print(f"would_kill_pids={pids}")
        print(f"would_start_cmd={args.start_cmd}")
        return 2

    try:
        if pids:
            _kill_pids(pids)
        _restart_collector(str(args.start_cmd))
        _append_watchdog_log(
            log_path=Path(str(args.log)),
            action="restart",
            pids=pids,
            start_cmd=str(args.start_cmd),
            details={"max_age_sec": max_age, "fresh_sec": int(args.fresh_sec)},
        )
        print("OK watchdog restart_launched=1")
        return 0
    except Exception as exc:
        _append_watchdog_log(
            log_path=Path(str(args.log)),
            action="restart_failed",
            pids=pids,
            start_cmd=str(args.start_cmd),
            details={"error": str(exc), "max_age_sec": max_age},
        )
        print(f"FAIL watchdog restart_error err={exc}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
