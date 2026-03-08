from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_rank_event_filter_set as mod


def test_summarize_rank_event_filter_set_payload() -> None:
    root = Path("localtests") / "rank_event_filter_set_summary"
    root.mkdir(parents=True, exist_ok=True)
    baseline_json = root / f"baseline_{uuid.uuid4().hex[:8]}.json"
    filtered_json = root / f"filtered_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"summary_{uuid.uuid4().hex[:8]}.json"
    try:
        baseline_json.write_text(
            json.dumps(
                {
                    "ranking": [
                        {"symbol": "ETHUSDT", "rule": "r1", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500, "max_spread": 0.0002, "npa_core": -0.0002, "score_raw_core": -0.0003, "attempt_fill_rate": 0.6},
                        {"symbol": "BTCUSDT", "rule": "r1", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500, "max_spread": 0.0002, "npa_core": -0.0003, "score_raw_core": -0.0004, "attempt_fill_rate": 0.5},
                    ]
                }
            ),
            encoding="utf-8",
        )
        filtered_json.write_text(
            json.dumps(
                {
                    "ranking": [
                        {"symbol": "ETHUSDT", "rule": "r1", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500, "max_spread": 0.0002, "npa_core": -0.0001, "score_raw_core": -0.0002, "attempt_fill_rate": 0.58, "event_filter_kept_ratio": 0.7},
                        {"symbol": "BTCUSDT", "rule": "r1", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500, "max_spread": 0.0002, "npa_core": -0.00035, "score_raw_core": -0.00045, "attempt_fill_rate": 0.48, "event_filter_kept_ratio": 0.7},
                    ]
                }
            ),
            encoding="utf-8",
        )
        built = mod.build_payload(
            baseline_json=str(baseline_json),
            filtered_json=str(filtered_json),
            out_json=str(out_json),
        )
        assert built["common_count"] == 2
        assert built["improved_count"] == 1
        assert built["degraded_count"] == 1
        assert built["best_tradeoff_row"]["symbol"] == "ETHUSDT"
        assert built["run_summary"]["run_type"] == "summarize_rank_event_filter_set"
    finally:
        baseline_json.unlink(missing_ok=True)
        filtered_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
