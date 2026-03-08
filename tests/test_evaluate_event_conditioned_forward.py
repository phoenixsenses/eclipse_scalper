from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import evaluate_event_conditioned_forward as mod


def test_evaluate_event_conditioned_forward_payload() -> None:
    root = Path("localtests") / "event_conditioned_forward"
    root.mkdir(parents=True, exist_ok=True)
    debug_jsonl = root / f"debug_{uuid.uuid4().hex[:8]}.jsonl"
    filter_json = root / f"filter_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"eval_{uuid.uuid4().hex[:8]}.json"

    rows = []
    for i in range(30):
        ret_1 = 0.003 if i % 5 == 0 else 0.0001
        rows.append(
            {
                "ts_bucket": i + 1,
                "gross_ret": 0.0020 if i % 5 == 0 else 0.0011,
                "cost": 0.0010,
                "net_ret": 0.0010 if i % 5 == 0 else 0.0001,
                "exec_model": "taker",
                "horizon_sec": 30,
                "spread": 0.00005 if i % 5 == 0 else 0.0003,
                "trade_intensity": 2000 if i % 5 == 0 else 400,
                "ret_1": ret_1,
                "imbalance": 0.2 if i % 5 == 0 else -0.8,
                "regime_spread_bin": ">p75" if i % 5 else "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": ">p75" if i % 5 == 0 else "p25-50",
                "regime_imb_bin": "+[0.3,0.5)" if i % 5 == 0 else "-[0.7,0.9)",
            }
        )

    filter_payload = {
        "filter_candidate": {
            "allow_lanes": ["return_shock"],
            "block_lanes": [{"lane": "book_proxy_pressure"}],
        }
    }
    try:
        debug_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        filter_json.write_text(json.dumps(filter_payload), encoding="utf-8")
        built = mod.build_payload(debug_jsonl=str(debug_jsonl), filter_json=str(filter_json), out_json=str(out_json))
        assert built["allow_lanes"] == ["return_shock"]
        assert built["block_lanes"] == ["book_proxy_pressure"]
        assert built["validation"]["filtered"]["n"] <= built["validation"]["baseline"]["n"]
        assert built["run_summary"]["run_type"] == "evaluate_event_conditioned_forward"
    finally:
        debug_jsonl.unlink(missing_ok=True)
        filter_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
