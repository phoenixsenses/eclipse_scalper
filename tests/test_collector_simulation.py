from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def test_simulate_connection_transitions() -> None:
    base = Path("eclipse_scalper/localtests/collector_sim") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    db_path = base / "sim_micro.db"
    hb_path = base / "collector_heartbeat.json"
    cmd = [
        sys.executable,
        "-m",
        "data.microstructure_collector",
        "--symbols",
        "ETHUSDT",
        "--db-path",
        str(db_path),
        "--heartbeat-path",
        str(hb_path),
        "--stats-interval",
        "2",
        "--simulate-connection",
        "--simulate-cycle-sec",
        "6",
        "--simulate-down-sec",
        "2",
        "--simulate-max-seconds",
        "8",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        assert res.returncode == 0, res.stderr or res.stdout
        out = res.stdout
        assert "[SIM] state=connected" in out
        assert "[SIM] state=disconnected" in out
        assert out.count("[SIM] state=connected") >= 2
        assert hb_path.exists()
        payload = json.loads(hb_path.read_text(encoding="utf-8"))
        assert "connected" in payload
        assert "current_backoff_seconds" in payload
    finally:
        shutil.rmtree(base, ignore_errors=True)
