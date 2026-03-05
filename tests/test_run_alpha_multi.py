from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import run_alpha_multi as tool


def _mk_local_tmp() -> Path:
    p = (Path("localtests") / f"run_alpha_multi_{uuid.uuid4().hex[:8]}").resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def test_run_alpha_multi_creates_symbol_runs_deterministic(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out = tmp / "out"

        def _fake_main() -> int:
            argv = list(sys.argv)
            run_dir = Path(argv[argv.index("--run-dir") + 1])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
            (run_dir / "pointers.json").write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
            return 0

        monkeypatch.setattr(tool.alpha_pipeline, "main", _fake_main)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_alpha_multi",
                "--symbols",
                "ETHUSDT,BTCUSDT",
                "--out-root",
                str(out),
                "--quick",
            ],
        )
        assert tool.main() == 0
        multi_runs = sorted([p for p in out.iterdir() if p.is_dir() and p.name.startswith("multi_")])
        assert multi_runs
        m = json.loads((multi_runs[-1] / "manifest.json").read_text(encoding="utf-8"))
        syms = sorted([str(x.get("symbol", "")) for x in m.get("runs", [])])
        assert syms == ["BTCUSDT", "ETHUSDT"]
        for r in m.get("runs", []):
            assert Path(str(r.get("run_dir", ""))).exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

