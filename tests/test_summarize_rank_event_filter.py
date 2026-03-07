from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_rank_event_filter as mod


def test_summarize_rank_event_filter_payload() -> None:
    root = Path("localtests") / "rank_event_filter_summary"
    root.mkdir(parents=True, exist_ok=True)
    baseline_json = root / f"baseline_{uuid.uuid4().hex[:8]}.json"
    filtered_json = root / f"filtered_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"summary_{uuid.uuid4().hex[:8]}.json"
    try:
        baseline_json.write_text(
            json.dumps(
                {
                    "ranking": [
                        {
                            "symbol": "ETHUSDT",
                            "rule": "micro_edge_v3_passive_alpha",
                            "npa_core": -0.0002,
                            "score_raw_core": -0.0003,
                            "attempt_fill_rate": 0.65,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        filtered_json.write_text(
            json.dumps(
                {
                    "ranking": [
                        {
                            "symbol": "ETHUSDT",
                            "rule": "micro_edge_v3_passive_alpha",
                            "npa_core": -0.0001,
                            "score_raw_core": -0.0001,
                            "attempt_fill_rate": 0.60,
                            "event_filter_kept_ratio": 0.74,
                            "event_block_lanes": ["book_proxy_pressure", "volatility_burst"],
                        }
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
        assert built["delta"]["npa_core"] > 0.0
        assert built["filtered_top"]["event_filter_kept_ratio"] == 0.74
        assert built["recommendation"] == "test_event_block_v1_in_rank_pipeline"
        assert built["run_summary"]["run_type"] == "summarize_rank_event_filter"
    finally:
        baseline_json.unlink(missing_ok=True)
        filtered_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
