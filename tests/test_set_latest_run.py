from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from tools.set_latest_run import set_latest_run
except ModuleNotFoundError:  # pragma: no cover
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.set_latest_run import set_latest_run


def _mk_run(path: Path, with_stability: bool = False, with_stability_all: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metrics.json").write_text(json.dumps({"pnl_net_sum": 1.0}, sort_keys=True) + "\n", encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"strategy": "baseline"}, sort_keys=True) + "\n", encoding="utf-8")
    (path / "summary.md").write_text("# ok\n", encoding="utf-8")
    if with_stability:
        (path / "stability.csv").write_text("k,v\na,1\n", encoding="utf-8")
        (path / "stability_up.csv").write_text("k,v\na,1\n", encoding="utf-8")
        (path / "stability_down.csv").write_text("k,v\na,1\n", encoding="utf-8")
    if with_stability_all:
        (path / "stability_all.csv").write_text("k,v\na,1\n", encoding="utf-8")


def test_set_latest_run_copies_files_and_is_idempotent() -> None:
    base = Path("eclipse_scalper/localtests/set_latest") / uuid.uuid4().hex
    run = base / "run"
    latest = base / "latest"
    _mk_run(run)
    out = set_latest_run(run, latest, copy_mode="copy", overwrite=True)
    assert out == latest
    assert (latest / "metrics.json").exists()
    assert (latest / "config.json").exists()
    assert (latest / "run_dir.txt").exists()
    first = (latest / "metrics.json").read_text(encoding="utf-8")
    out2 = set_latest_run(run, latest, copy_mode="copy", overwrite=True)
    assert out2 == latest
    second = (latest / "metrics.json").read_text(encoding="utf-8")
    assert first == second


def test_set_latest_run_include_glob_copies_stability_files() -> None:
    base = Path("eclipse_scalper/localtests/set_latest") / uuid.uuid4().hex
    run = base / "run"
    latest = base / "latest"
    _mk_run(run, with_stability=True)
    set_latest_run(run, latest, copy_mode="copy", overwrite=True, include_glob=["stability*.csv"])
    assert (latest / "stability.csv").exists()
    assert (latest / "stability_up.csv").exists()
    assert (latest / "stability_down.csv").exists()


def test_set_latest_run_extra_missing_warns_or_strict_fails() -> None:
    base = Path("eclipse_scalper/localtests/set_latest") / uuid.uuid4().hex
    run = base / "run"
    latest = base / "latest"
    _mk_run(run)
    warnings: list[str] = []
    set_latest_run(run, latest, copy_mode="copy", overwrite=True, extra="stability.csv", warnings_out=warnings)
    assert warnings
    failed = False
    try:
        set_latest_run(run, latest, copy_mode="copy", overwrite=True, extra="stability.csv", strict_extra=True)
    except FileNotFoundError:
        failed = True
    assert failed is True


def test_set_latest_run_stability_all_compat_writes_stability_csv() -> None:
    base = Path("eclipse_scalper/localtests/set_latest") / uuid.uuid4().hex
    run = base / "run"
    latest = base / "latest"
    _mk_run(run, with_stability=False, with_stability_all=True)
    set_latest_run(run, latest, copy_mode="copy", overwrite=True)
    assert (latest / "stability_all.csv").exists()
    assert (latest / "stability.csv").exists()


def test_set_latest_run_print_env_includes_stability_lines() -> None:
    base = Path("eclipse_scalper/localtests/set_latest") / uuid.uuid4().hex
    run = base / "run"
    latest = base / "latest"
    _mk_run(run, with_stability=True)
    cmd = [
        sys.executable,
        "-m",
        "tools.set_latest_run",
        "--run-dir",
        str(run),
        "--latest-dir",
        str(latest),
        "--include-glob",
        "stability*.csv",
        "--print-env",
        "--enable-alpha-gate",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + "\n" + res.stderr
    out = res.stdout
    assert "set ALPHA_GATE_METRICS_PATH=" in out
    assert "set ALPHA_GATE_ENABLED=1" in out
    assert "set ALPHA_GATE_STABILITY_PATH=" in out
    assert "set ALPHA_GATE_STABILITY_UP_PATH=" in out
    assert "set ALPHA_GATE_STABILITY_DOWN_PATH=" in out
