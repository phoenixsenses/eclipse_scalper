from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_event_signal_bridge as mod


def test_summarize_event_signal_bridge_payload() -> None:
    root = Path("localtests") / "event_signal_bridge"
    root.mkdir(parents=True, exist_ok=True)
    forward_json = root / f"forward_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"summary_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "event_lane_context_impact": {
            "discovery": {
                "available": True,
                "rows_total": 10,
                "lane_count": 5,
                "by_lane": {
                    "volume_vacuum": {"tagged_n": 2, "delta_avg_net": 0.0010, "delta_p90_net": 0.0005},
                    "return_shock": {"tagged_n": 3, "delta_avg_net": -0.0002, "delta_p90_net": 0.0001},
                },
            },
            "validation": {
                "available": True,
                "rows_total": 8,
                "lane_count": 5,
                "by_lane": {
                    "return_shock": {"tagged_n": 2, "delta_avg_net": 0.0001, "delta_p90_net": 0.0002},
                    "book_proxy_pressure": {"tagged_n": 2, "delta_avg_net": -0.0004, "delta_p90_net": -0.0005},
                },
            },
        }
    }
    try:
        forward_json.write_text(json.dumps(payload), encoding="utf-8")
        built = mod.build_payload(forward_json=str(forward_json), out_json=str(out_json))
        assert built["discovery"]["best_positive_lane"]["lane"] == "volume_vacuum"
        assert built["validation"]["best_positive_lane"]["lane"] == "return_shock"
        assert built["validation"]["worst_negative_lane"]["lane"] == "book_proxy_pressure"
        assert built["recommendation"] == "test_event_conditioned_filter"
        assert built["run_summary"]["run_type"] == "summarize_event_signal_bridge"
    finally:
        forward_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
