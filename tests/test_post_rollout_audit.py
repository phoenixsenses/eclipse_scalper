from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import post_rollout_audit as pra


def test_post_rollout_audit_writes_run_summary(monkeypatch, tmp_path) -> None:
    base = tmp_path / "test_post_rollout_audit"
    base.mkdir(parents=True, exist_ok=True)
    diag = base / "diag.json"
    tox = base / "tox.json"
    out_json = base / "out.json"
    out_md = base / "out.md"
    diag.write_text(json.dumps({"rows": 10, "fill_rate": 0.2, "latency_fill_delay_sec_p95": 2.0}), encoding="utf-8")
    tox.write_text(json.dumps({"rows": 10, "sides": {"buy": {"rows": 5}}}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--diag-json", str(diag), "--tox-json", str(tox), "--out-json", str(out_json), "--out-md", str(out_md)],
    )
    assert pra.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "post_rollout_audit"
    assert payload["run_summary"]["metrics"]["check_count"] >= 1
