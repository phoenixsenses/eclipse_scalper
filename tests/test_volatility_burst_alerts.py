from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import volatility_burst_alerts as vba


def test_tag_rows_detects_volatility_burst() -> None:
    rows = [
        {"ts_ms": 1.0, "spread": 0.0002, "trade_intensity": 100.0, "ret_1": 0.0001},
        {"ts_ms": 2.0, "spread": 0.0002, "trade_intensity": 500.0, "ret_1": -0.0100},
        {"ts_ms": 3.0, "spread": 0.0002, "trade_intensity": 450.0, "ret_1": 0.0080},
    ]
    tags = vba._tag_rows(rows)
    assert len(tags) == 3
    assert any(bool(r["rule_fired"]) for r in tags)


def test_main_writes_json_and_md(monkeypatch) -> None:
    monkeypatch.setattr(
        vba,
        "_load_rows",
        lambda db, symbol, lookback_min, bucket_sec: [
            {"ts_ms": 1.0, "spread": 0.0002, "trade_intensity": 100.0, "ret_1": 0.0001},
            {"ts_ms": 2.0, "spread": 0.0002, "trade_intensity": 500.0, "ret_1": -0.0100},
            {"ts_ms": 3.0, "spread": 0.0002, "trade_intensity": 450.0, "ret_1": 0.0080},
        ],
    )
    out_dir = Path("localtests/test_volatility_burst_alerts")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "out.json"
    out_md = out_dir / "out.md"
    rc = vba.main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["lane"] == "volatility_burst"
    assert payload["run_summary"]["run_type"] == "volatility_burst_alerts"
    assert payload["summary"]["tagged_count"] >= 1
    assert out_md.exists()
