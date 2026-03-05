from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path


def test_health_check_stale_exit_2() -> None:
    base = Path("eclipse_scalper/localtests/health_check_stale") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    health = base / "overall.json"
    payload = {
        "ts_utc": "2000-01-01T00:00:00Z",
        "mode": "paper",
        "state": "ok",
        "components": {"collector": {"status": "ok", "connected": True}},
    }
    health.write_text(json.dumps(payload), encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "tools.health_check",
        "--health",
        str(health),
        "--max-staleness-sec",
        "15",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert res.returncode == 2
    assert "health_stale" in (res.stdout + res.stderr)

