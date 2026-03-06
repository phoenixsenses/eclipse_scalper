from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import execution_e2e_pipeline as p


def test_pipeline_run_helper_smoke() -> None:
    out = p._run(["python", "-c", "print('ok')"])
    assert "cmd" in out and "rc" in out


def test_pipeline_write_json(tmp_path=None) -> None:
    root = Path("localtests") / "exec_e2e_pipeline"
    root.mkdir(parents=True, exist_ok=True)
    f = root / "x.json"
    p._write(f, {"ok": True})
    assert f.exists()


def test_execution_e2e_pipeline_writes_run_summary(monkeypatch) -> None:
    out_json = Path("reports/test_execution_e2e_pipeline/out.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def _fake_run(cmd):
        return {"cmd": cmd, "rc": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(p, "_run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--sim", "logs/x.jsonl", "--live-db", "data/paper_trades.db", "--live-parquet", "data/live/papertrades_live.parquet", "--out-json", str(out_json)],
    )
    assert p.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "execution_e2e_pipeline"
    assert payload["run_summary"]["metrics"]["step_count"] == 4
