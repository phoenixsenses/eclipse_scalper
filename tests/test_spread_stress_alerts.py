from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import spread_stress_alerts as ssa


def test_tag_rows_detects_spread_stress() -> None:
    rows = [
        {"ts_ms": 1.0, "spread": 0.001, "trade_intensity": 100.0, "ret_1": 0.0},
        {"ts_ms": 2.0, "spread": 0.020, "trade_intensity": 1.0, "ret_1": -0.001},
        {"ts_ms": 3.0, "spread": 0.015, "trade_intensity": 2.0, "ret_1": 0.001},
    ]
    tags = ssa._tag_rows(rows)
    assert len(tags) == 3
    assert any(bool(r["rule_fired"]) for r in tags)


def test_main_writes_json_and_md(monkeypatch) -> None:
    monkeypatch.setattr(
        ssa,
        "_load_rows",
        lambda db, symbol, lookback_min, bucket_sec: [
            {"ts_ms": 1.0, "spread": 0.001, "trade_intensity": 100.0, "ret_1": 0.0},
            {"ts_ms": 2.0, "spread": 0.020, "trade_intensity": 1.0, "ret_1": -0.001},
            {"ts_ms": 3.0, "spread": 0.015, "trade_intensity": 2.0, "ret_1": 0.001},
        ],
    )
    out_dir = Path("localtests/test_spread_stress_alerts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "out.json"
    out_md = out_dir / "out.md"
    rc = ssa.main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "spread_stress_alerts"
    assert payload["summary"]["tagged_count"] >= 1
    assert out_md.exists()
