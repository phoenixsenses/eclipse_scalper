from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_artifacts as va


def test_validate_artifacts_writes_run_summary(monkeypatch) -> None:
    out_json = Path("reports/test_validate_artifacts/out.json")
    out_md = Path("reports/test_validate_artifacts/out.md")
    cal = Path("reports/test_validate_artifacts/cal.json")
    exe = Path("reports/test_validate_artifacts/exe.json")
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text("{}", encoding="utf-8")
    exe.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(va, "validate_calibration_file", lambda path: (True, [], {"kind": "cal"}))
    monkeypatch.setattr(va, "validate_execution_file", lambda path: (True, [], {"kind": "exe"}))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--calibration",
            str(cal),
            "--execution",
            str(exe),
            "--out-json",
            str(out_json),
            "--out-report",
            str(out_md),
        ],
    )
    rc = va.main()
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "validate_artifacts"
    assert payload["run_summary"]["metrics"]["ok"] is True
    assert payload["run_summary"]["artifacts"]["json"].endswith("out.json")
    assert payload["run_summary"]["artifacts"]["report"].endswith("out.md")
    assert "## Run Summary" in out_md.read_text(encoding="utf-8")
