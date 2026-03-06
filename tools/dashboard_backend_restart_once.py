from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _stop_pid(pid: int) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], check=False, capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_path = repo_root / "runtime" / "dashboard_backend.json"
    backend_script = repo_root / "tools" / "run_dashboard_backend.ps1"

    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("DASHBOARD_PORT", "8765"))

    if runtime_path.exists():
        try:
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            host = str(payload.get("host") or host)
            port = int(payload.get("port") or port)
            pid = payload.get("pid")
            if isinstance(pid, int) and pid > 0:
                _stop_pid(pid)
                time.sleep(0.4)
        except Exception:
            pass

    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(backend_script),
    ]
    # Detached launch; backend script itself starts uvicorn foreground in this child shell.
    subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    health_url = f"http://{host}:{port}/api/health"
    for _ in range(30):
        if _health_ok(health_url):
            print(f"restart_dashboard_backend ok url={health_url}")
            return 0
        time.sleep(0.5)

    print(f"restart_dashboard_backend fail url={health_url}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
