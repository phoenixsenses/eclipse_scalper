from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from tools import run_full_sweep as rfs


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"run_full_sweep_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_run_full_sweep_builds_rank_command_and_manifest(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_dir = tmp / "runs"
        calls: list[list[str]] = []

        def _fake_call(cmd, env=None):  # type: ignore[no-untyped-def]
            calls.append(list(cmd))
            return 0

        monkeypatch.setattr(rfs.subprocess, "call", _fake_call)
        monkeypatch.setattr(
            "sys.argv",
            [
                "x",
                "--candidates-md",
                "reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md",
                "--symbols",
                "ETHUSDT",
                "--sides",
                "sell",
                "--fees",
                "0.5",
                "--adverse",
                "1.0",
                "--horizons",
                "120",
                "--workers",
                "1",
                "--output-dir",
                str(out_dir),
            ],
        )
        rc = rfs.main()
        assert rc == 0
        assert len(calls) == 1
        cmd = calls[0]
        assert "tools.rank_passive_pockets_forward" in cmd
        assert "--candidates-md" in cmd
        assert "reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md" in cmd
        mf = out_dir / "manifest.json"
        assert mf.exists()
        obj = json.loads(mf.read_text(encoding="utf-8"))
        jobs = obj.get("jobs", [])
        assert len(jobs) == 1
        assert int(jobs[0].get("rc", 1)) == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

