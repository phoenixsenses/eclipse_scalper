from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import daily_execution_calibration as dec


def test_daily_execution_calibration_writes_run_summary(monkeypatch) -> None:
    out_dir = Path("reports/test_daily_execution_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd):
        return {"cmd": cmd, "rc": 0, "stdout_tail": "ok", "stderr_tail": ""}

    monkeypatch.setattr(dec, "_run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--symbol",
            "ETHUSDT",
            "--report-dir",
            str(out_dir),
            "--run-root-cause",
            "0",
        ],
    )
    assert dec.main() == 0
    out_json = next(out_dir.glob("*_EXEC_CALIBRATION.json"))
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "daily_execution_calibration"
    assert payload["run_summary"]["metrics"]["step_count"] == 2
