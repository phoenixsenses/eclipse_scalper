from __future__ import annotations

from pathlib import Path

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

