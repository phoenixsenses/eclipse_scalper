from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import evaluate_event_conditioned_forward_grid as mod


def test_evaluate_event_conditioned_forward_grid_payload() -> None:
    root = Path("localtests") / "event_conditioned_forward_grid"
    root.mkdir(parents=True, exist_ok=True)
    debug_jsonl = root / f"debug_{uuid.uuid4().hex[:8]}.jsonl"
    filter_json = root / f"filter_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"grid_{uuid.uuid4().hex[:8]}.json"

    rows = []
    for i in range(40):
        hot = (i % 6 == 0)
        rows.append(
            {
                "ts_bucket": i + 1,
                "gross_ret": 0.002 if hot else 0.0010,
                "cost": 0.001,
                "net_ret": 0.001 if hot else -0.0002,
                "exec_model": "taker",
                "horizon_sec": 30,
                "spread": 0.00005 if hot else 0.00025,
                "trade_intensity": 1800 if hot else 350,
                "ret_1": 0.003 if hot else 0.0001,
                "imbalance": 0.1 if hot else -0.8,
                "regime_spread_bin": ">p75" if not hot else "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": ">p75" if hot else "p25-50",
                "regime_imb_bin": "+[0.3,0.5)" if hot else "-[0.7,0.9)",
            }
        )
    filter_payload = {
        "filter_candidate": {
            "allow_lanes": ["return_shock"],
            "tentative_allow_lanes": ["volume_vacuum"],
            "block_lanes": [{"lane": "book_proxy_pressure"}],
        }
    }
    try:
        debug_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        filter_json.write_text(json.dumps(filter_payload), encoding="utf-8")
        built = mod.build_payload(debug_jsonl=str(debug_jsonl), filter_json=str(filter_json), out_json=str(out_json))
        assert built["variant_count"] == 5
        assert built["best_variant"]
        assert len(built["rows"]) == 5
        assert built["run_summary"]["run_type"] == "evaluate_event_conditioned_forward_grid"
    finally:
        debug_jsonl.unlink(missing_ok=True)
        filter_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
