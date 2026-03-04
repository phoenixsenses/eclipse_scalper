from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from tools import incident_bundle as ib


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"incident_bundle_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_incident_bundle_collects_logs_and_run_artifacts(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        (tmp / "logs").mkdir(parents=True, exist_ok=True)
        (tmp / "data" / "live").mkdir(parents=True, exist_ok=True)
        (tmp / "data" / "runs" / "alpha" / "run_20260304_010101").mkdir(parents=True, exist_ok=True)

        (tmp / "logs" / "runtime.log").write_text("line1\nline2\n", encoding="utf-8")
        (tmp / "logs" / "last_shutdown.json").write_text('{"reason":"x"}\n', encoding="utf-8")
        (tmp / "data" / "live" / "status.json").write_text('{"ok":1}\n', encoding="utf-8")
        (tmp / "data" / "runs" / "alpha" / "run_20260304_010101" / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp / "data" / "runs" / "alpha" / "run_20260304_010101" / "pointers.json").write_text("{}", encoding="utf-8")

        out_root = tmp / "reports" / "incidents"
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--out-root",
                str(out_root),
                "--logs-dir",
                str(tmp / "logs"),
                "--run-root",
                str(tmp / "data" / "runs" / "alpha"),
                "--tail-lines",
                "20",
            ],
        )
        assert ib.main() == 0

        bundles = sorted(out_root.glob("incident_*"))
        assert bundles, "bundle directory not created"
        summary = json.loads((bundles[-1] / "summary.json").read_text(encoding="utf-8"))
        assert "bundle_dir" in summary
        assert (bundles[-1] / "latest_runtime_tail.log").exists()
        assert (bundles[-1] / "summary.md").exists()
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

