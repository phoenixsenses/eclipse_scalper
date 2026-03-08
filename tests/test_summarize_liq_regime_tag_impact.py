from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_liq_regime_tag_impact as mod


def test_summarize_liq_regime_tag_impact_writes_json(capsys):
    root = Path("localtests") / "summarize_liq_regime_tag_impact"
    root.mkdir(parents=True, exist_ok=True)
    in_path = root / f"in_{uuid.uuid4().hex[:8]}.json"
    out_path = root / f"out_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "liquidation_regime_tag_impact": {
            "discovery": {
                "available": True,
                "tagged": {"n": 4, "avg_net": 0.0012, "p90_net": 0.0018},
                "normal": {"n": 8, "avg_net": 0.0004, "p90_net": 0.0010},
            },
            "validation": {
                "available": True,
                "tagged": {"n": 3, "avg_net": -0.0001, "p90_net": 0.0003},
                "normal": {"n": 9, "avg_net": 0.0002, "p90_net": 0.0007},
            },
        }
    }
    try:
        in_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        code = mod.main(["--in", str(in_path), "--out-json", str(out_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "discovery" in out
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["run_summary"]["run_type"] == "summarize_liq_regime_tag_impact"
        assert saved["discovery"]["delta_avg_net"] == pytest.approx(0.0008)
        assert saved["validation"]["delta_avg_net"] == pytest.approx(-0.0003)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def test_summarize_liq_regime_tag_impact_falls_back_to_debug_rows():
    root = Path("localtests") / "summarize_liq_regime_tag_impact"
    root.mkdir(parents=True, exist_ok=True)
    debug_path = root / f"debug_{uuid.uuid4().hex[:8]}.jsonl"
    in_path = root / f"in_{uuid.uuid4().hex[:8]}.json"
    out_path = root / f"out_{uuid.uuid4().hex[:8]}.json"
    debug_rows = [
        {
            "ts_bucket": 1,
            "gross_ret": 0.0020,
            "cost": 0.0010,
            "net_ret": 0.0010,
            "exec_model": "taker",
            "horizon_sec": 30,
            "spread": 0.010,
            "trade_intensity": 6.0,
            "ret_1": -0.0030,
            "liq_imbalance": 0.85,
            "liq_rate_per_sec": 30.0,
            "imbalance": 0.2,
        },
        {
            "ts_bucket": 2,
            "gross_ret": 0.0018,
            "cost": 0.0010,
            "net_ret": 0.0008,
            "exec_model": "taker",
            "horizon_sec": 30,
            "spread": 0.040,
            "trade_intensity": 1.0,
            "ret_1": 0.0,
            "liq_imbalance": 0.0,
            "liq_rate_per_sec": 0.0,
            "imbalance": 0.0,
        },
        {
            "ts_bucket": 3,
            "gross_ret": 0.0022,
            "cost": 0.0010,
            "net_ret": 0.0012,
            "exec_model": "taker",
            "horizon_sec": 30,
            "spread": 0.011,
            "trade_intensity": 6.2,
            "ret_1": -0.0025,
            "liq_imbalance": 0.90,
            "liq_rate_per_sec": 32.0,
            "imbalance": 0.1,
        },
        {
            "ts_bucket": 4,
            "gross_ret": 0.0015,
            "cost": 0.0010,
            "net_ret": 0.0005,
            "exec_model": "taker",
            "horizon_sec": 30,
            "spread": 0.042,
            "trade_intensity": 1.2,
            "ret_1": 0.0,
            "liq_imbalance": 0.0,
            "liq_rate_per_sec": 0.0,
            "imbalance": 0.0,
        },
    ]
    payload = {
        "debug": str(debug_path),
        "discover_frac": 0.5,
        "liquidation_regime_tag_impact": {
            "discovery": {"available": False, "tagged": {"n": 0, "avg_net": 0.0, "p90_net": 0.0}, "normal": {"n": 0, "avg_net": 0.0, "p90_net": 0.0}},
            "validation": {"available": False, "tagged": {"n": 0, "avg_net": 0.0, "p90_net": 0.0}, "normal": {"n": 0, "avg_net": 0.0, "p90_net": 0.0}},
        },
    }
    try:
        debug_path.write_text("\n".join(json.dumps(r) for r in debug_rows) + "\n", encoding="utf-8")
        in_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        code = mod.main(["--in", str(in_path), "--out-json", str(out_path)])
        assert code == 0
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        assert saved["discovery"]["available"] is True
        assert saved["validation"]["available"] is True
        assert saved["discovery"]["tagged"]["n"] >= 1
    finally:
        debug_path.unlink(missing_ok=True)
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
