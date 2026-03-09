from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import evaluate_event_conditioned_filter as mod


def test_evaluate_event_conditioned_filter_payload() -> None:
    root = Path("localtests") / "event_conditioned_filter"
    root.mkdir(parents=True, exist_ok=True)
    bridge_json = root / f"bridge_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"filter_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "discovery": {
            "ranked": [
                {"lane": "volume_vacuum", "tagged_n": 4, "delta_avg_net": 0.0011, "delta_p90_net": 0.0010},
                {"lane": "book_proxy_pressure", "tagged_n": 5, "delta_avg_net": -0.0008, "delta_p90_net": -0.0006},
            ]
        },
        "validation": {
            "ranked": [
                {"lane": "return_shock", "tagged_n": 6, "delta_avg_net": 0.0002, "delta_p90_net": 0.0005},
                {"lane": "spread_stress", "tagged_n": 3, "delta_avg_net": -0.0004, "delta_p90_net": -0.0011},
                {"lane": "book_proxy_pressure", "tagged_n": 5, "delta_avg_net": -0.0002, "delta_p90_net": -0.0003},
            ]
        },
    }
    try:
        bridge_json.write_text(json.dumps(payload), encoding="utf-8")
        built = mod.build_payload(bridge_json=str(bridge_json), out_json=str(out_json), min_tagged_n=3)
        assert built["summary"]["primary_allow_lane"] == "return_shock"
        assert built["summary"]["tentative_allow_lane"] == "volume_vacuum"
        assert built["summary"]["block_lane_count"] == 2
        assert built["summary"]["recommendation"] == "test_allow_and_block_filters"
        assert built["filter_candidate"]["allow_lanes"] == ["return_shock"]
        assert built["filter_candidate"]["block_lanes"][0]["lane"] == "spread_stress"
        assert built["run_summary"]["run_type"] == "evaluate_event_conditioned_filter"
    finally:
        bridge_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
