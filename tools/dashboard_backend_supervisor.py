from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dashboard backend supervisor")
    p.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8765")))
    p.add_argument("--health-path", default="/api/health")
    p.add_argument("--check-interval", type=float, default=5.0)
    p.add_argument("--restart-backoff", type=float, default=3.0)
    p.add_argument("--max-restarts-per-hour", type=int, default=8)
    p.add_argument("--fail-threshold", type=int, default=3)
    return p


def _health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _log(msg: str, log_file: Path) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _start_backend(repo_root: Path) -> subprocess.Popen[str]:
    script = repo_root / "tools" / "run_dashboard_backend.ps1"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ]
    return subprocess.Popen(cmd, cwd=str(repo_root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    log_file = repo_root / "logs" / "dashboard_backend_supervisor.log"
    health_url = f"http://{args.host}:{args.port}{args.health_path}"

    stop = False

    def _handle(_sig: int, _frm) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle)

    restarts: deque[float] = deque()
    proc: subprocess.Popen[str] | None = _start_backend(repo_root)
    _log(f"started backend pid={proc.pid} url={health_url}", log_file)
    fail_count = 0
    detached_monitoring = False

    while not stop:
        now = time.time()
        while restarts and now - restarts[0] > 3600:
            restarts.popleft()

        alive = (proc is not None) and (proc.poll() is None)
        ok = _health_ok(health_url)

        if not alive:
            rc = proc.returncode if proc is not None else None
            if ok:
                # Wrapper may exit with rc=0 when backend is already up; this is healthy.
                fail_count = 0
                if not detached_monitoring:
                    _log(f"backend launcher exited rc={rc} but health is OK; monitoring existing backend", log_file)
                    detached_monitoring = True
                proc = None
            else:
                fail_count += 1
                _log(f"backend exited rc={rc} fail_count={fail_count}", log_file)
        elif not ok:
            fail_count += 1
            _log(f"health check failed fail_count={fail_count}", log_file)
        else:
            fail_count = 0
            detached_monitoring = False

        if fail_count >= args.fail_threshold:
            if len(restarts) >= args.max_restarts_per_hour:
                _log("restart cap reached; stopping supervisor", log_file)
                break
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            time.sleep(max(0.5, args.restart_backoff))
            proc = _start_backend(repo_root)
            restarts.append(time.time())
            _log(f"restarted backend pid={proc.pid} restarts_1h={len(restarts)}", log_file)
            fail_count = 0
            detached_monitoring = False

        time.sleep(max(1.0, args.check_interval))

    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    _log("supervisor stopped", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
