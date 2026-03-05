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


def test_run_smoke_success_short_cycle() -> None:
    base = Path("eclipse_scalper/localtests/health_cycle_smoke") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    out_dir = base / "out"
    args = argparse.Namespace(
        db_path="data/microstructure.db",
        symbols="ETHUSDT",
        cycle_sec=6.0,
        down_sec=2.0,
        max_seconds=10.0,
        stats_interval=1.0,
        out_dir=str(out_dir),
        health_file="logs/health/overall.json",
        poll_interval_sec=0.2,
    )
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
    finally:
        shutil.rmtree(base, ignore_errors=True)

