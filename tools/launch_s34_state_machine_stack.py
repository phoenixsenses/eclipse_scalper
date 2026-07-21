from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def launch(name: str, args: list[str]) -> int:
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = (log_dir / f"{name}.out.log").open("ab", buffering=0)
    err = (log_dir / f"{name}.err.log").open("ab", buffering=0)
    proc = subprocess.Popen(
        [PYTHON, "-W", "ignore", "-u", "-m", *args],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=out,
        stderr=err,
        close_fds=True,
    )
    return int(proc.pid)


def main() -> int:
    live_pid = launch(
        "s34_state_machine_live_executor",
        ["tools.s34_state_machine_live_executor", "--live", "--confirm-live-orders"],
    )
    shadow_pid = launch(
        "s34_state_machine_shadow_runner",
        ["tools.s34_realtime_shadow_runner"],
    )
    print(f"state_machine_live_pid={live_pid}")
    print(f"state_machine_shadow_pid={shadow_pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
