from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import liquidation_regime_tagger as lrt


def test_tag_rows_marks_fired_rows() -> None:
    rows = [
        {"ts_ms": 1.0, "mid": 100.0, "spread": 0.001, "imbalance": 0.0, "trade_intensity": 10.0, "ret_1": -0.001, "liq_count": 1.0, "liq_qty": 5.0, "liq_imbalance": 0.9, "liq_rate_per_sec": 20.0},
        {"ts_ms": 2.0, "mid": 100.1, "spread": 0.050, "imbalance": 0.0, "trade_intensity": 1.0, "ret_1": 0.0, "liq_count": 0.0, "liq_qty": 0.0, "liq_imbalance": 0.0, "liq_rate_per_sec": 0.0},
    ]
    tagged = lrt._tag_rows(rows, "high_liq_reversal_regime")
    assert len(tagged) == 2
    assert tagged[0]["tag"] in {"high_liq_reversal", "normal"}
    assert "rule_fired" in tagged[0]


def test_main_writes_json_and_md(monkeypatch) -> None:
    monkeypatch.setattr(
        lrt,
        "_load_rows",
        lambda db, symbol, lookback_min, bucket_sec: [
            {"ts_ms": 1.0, "mid": 100.0, "spread": 0.001, "imbalance": 0.0, "trade_intensity": 10.0, "ret_1": -0.001, "liq_count": 1.0, "liq_qty": 5.0, "liq_imbalance": 0.9, "liq_rate_per_sec": 20.0},
            {"ts_ms": 2.0, "mid": 100.1, "spread": 0.050, "imbalance": 0.0, "trade_intensity": 1.0, "ret_1": 0.0, "liq_count": 0.0, "liq_qty": 0.0, "liq_imbalance": 0.0, "liq_rate_per_sec": 0.0},
        ],
    )
    monkeypatch.setattr(
        lrt,
        "_tag_rows",
        lambda rows, rule_name: [
            {"ts_ms": 1, "tag": "high_liq_reversal", "rule_fired": True, "trade_intensity": 10.0, "spread": 0.001, "ret_1": -0.001, "liq_imbalance": 0.9, "liq_rate_per_sec": 20.0},
            {"ts_ms": 2, "tag": "normal", "rule_fired": False, "trade_intensity": 1.0, "spread": 0.05, "ret_1": 0.0, "liq_imbalance": 0.0, "liq_rate_per_sec": 0.0},
        ],
    )
    out_dir = Path("reports/test_liquidation_regime_tagger")
    out_json = out_dir / "out.json"
    out_md = out_dir / "out.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--db",
            "data/microstructure.db",
            "--symbol",
            "ETHUSDT",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
    )
    rc = lrt.main()
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["run_summary"]["run_type"] == "liquidation_regime_tagger"
    assert payload["summary"]["tagged_count"] == 1
    assert out_md.exists()
