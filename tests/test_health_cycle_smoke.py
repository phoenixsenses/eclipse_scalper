from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path


try:
    from tools.health_cycle_smoke import _validate_snapshot, run_smoke
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.health_cycle_smoke import _validate_snapshot, run_smoke

from tools import heartbeat_watchdog as hw


def _stub_process_alive(monkeypatch) -> None:
    """The real simulated-collector subprocess genuinely is alive for this
    test's whole duration -- its own self-reported collector.json status is
    what drives the ok/degraded/ok transition, not OS-level process
    detection. hw.python_process_running() otherwise spawns a real
    PowerShell Get-CimInstance call on every evaluation cycle (~hundreds of
    ms), which starves this test's fast polling loop of enough samples to
    reliably catch the simulated collector's brief down_sec window."""
    monkeypatch.setattr(hw, "python_process_running", lambda needle: True)


def _sample(state: str, connected: bool, reconnects: int) -> dict:
    return {
        "ts_utc": "2026-03-01T00:00:00+00:00",
        "mode": "paper",
        "state": state,
        "components": {
            "collector": {
                "status": state,
                "connected": connected,
                "reconnects_last_5m": reconnects,
                "errors_last_5m": 0,
            }
        },
    }


def test_validate_snapshot_rules() -> None:
    ok_obj = _sample("ok", True, 0)
    deg_obj = _sample("degraded", False, 1)
    valid, reason = _validate_snapshot(ok_obj, "ok")
    assert valid, reason
    valid, reason = _validate_snapshot(deg_obj, "degraded")
    assert valid, reason
    bad = _sample("ok", False, 0)
    valid, _ = _validate_snapshot(bad, "ok")
    assert not valid


def test_run_smoke_success_short_cycle(monkeypatch) -> None:
    """Fully isolated: --root and --db-path both point inside a disposable
    temp directory, so this test never reads or writes the real repo's
    logs/health/overall.json, never touches data/microstructure.db, and
    never depends on the real production heartbeat_watchdog process being
    alive -- run_smoke() produces canonical overall.json itself, in-process,
    scoped to --root. Same result whether or not the real Eclipse runtime
    is running. See
    reports/research/s34/CANONICAL_OPERATIONAL_HEALTH_2026-07-10.md Part C.
    """
    base = Path("eclipse_scalper/localtests/health_cycle_smoke") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    out_dir = base / "out"
    isolated_root = base / "isolated_root"
    db_path = base / "isolated_db" / "microstructure.db"
    args = argparse.Namespace(
        db_path=str(db_path),
        symbols="ETHUSDT",
        cycle_sec=6.0,
        down_sec=2.0,
        max_seconds=10.0,
        stats_interval=1.0,
        out_dir=str(out_dir),
        root=str(isolated_root),
        poll_interval_sec=0.2,
        seed_market_data=True,
    )
    _stub_process_alive(monkeypatch)
    try:
        rc = run_smoke(args)
        assert rc == 0
        for name in (
            "overall_snapshot1_ok.json",
            "overall_snapshot2_degraded.json",
            "overall_snapshot3_ok.json",
        ):
            p = out_dir / name
            assert p.exists()
            json.loads(p.read_text(encoding="utf-8"))

        assert (isolated_root / "logs" / "health" / "overall.json").exists()  # the file it actually wrote
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_run_smoke_creates_no_files_in_real_repo_health_dir(monkeypatch) -> None:
    """Isolation proof that stays valid even while the real, live Eclipse
    watchdog is running and continuously rewriting its own files in place:
    checks the *set of filenames* under the real logs/health/ directory
    (immune to the live process's in-place content/mtime changes to files
    it already owns), not file contents -- an isolated run must never add
    (or remove) anything there."""
    real_health_dir = Path("logs/health")
    names_before = set(p.name for p in real_health_dir.iterdir()) if real_health_dir.exists() else set()

    base = Path("eclipse_scalper/localtests/health_cycle_smoke") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    try:
        args = argparse.Namespace(
            db_path=str(base / "isolated_db" / "microstructure.db"),
            symbols="ETHUSDT",
            cycle_sec=2.0,
            down_sec=1.0,
            max_seconds=1.5,
            stats_interval=1.0,
            out_dir=str(base / "out"),
            root=str(base / "isolated_root"),
            poll_interval_sec=0.2,
            seed_market_data=True,
        )
        _stub_process_alive(monkeypatch)
        run_smoke(args)  # regardless of pass/fail/timeout, isolation must hold

        names_after = set(p.name for p in real_health_dir.iterdir()) if real_health_dir.exists() else set()
        assert names_after == names_before
    finally:
        shutil.rmtree(base, ignore_errors=True)

