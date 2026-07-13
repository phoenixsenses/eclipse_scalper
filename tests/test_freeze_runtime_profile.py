from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import freeze_runtime_profile as frp


def test_freeze_runtime_profile_writes_run_summary(monkeypatch, tmp_path) -> None:
    out_json = tmp_path / "test_freeze_runtime_profile" / "out.json"
    out_md = tmp_path / "test_freeze_runtime_profile" / "out.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCALPER_DRY_RUN", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--out-json", str(out_json), "--out-md", str(out_md), "--write-lock"],
    )
    assert frp.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "freeze_runtime_profile"
