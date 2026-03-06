from __future__ import annotations

from pathlib import Path

from tools import run_execution_canary as c


def test_render_md_contains_checks() -> None:
    md = c._render_md(
        {
            "ts_utc": "2026-03-05T00:00:00Z",
            "overall_ok": True,
            "symbol": "ETHUSDT",
            "max_cycles": 1,
            "checks": {"a": True, "b": False},
            "steps": [{"rc": 0, "cmd": ["python", "-V"]}],
        }
    )
    assert "overall_ok" in md
    assert "Checks" in md


def test_safe_read_json_missing() -> None:
    p = Path("localtests") / "missing_canary.json"
    if p.exists():
        p.unlink()
    out = c._safe_read_json(p)
    assert out == {}

