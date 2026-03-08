from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_liq_tag_signal_behavior as mod


def test_summarize_liq_tag_signal_behavior_writes_json():
    root = Path("localtests") / "summarize_liq_tag_signal_behavior"
    root.mkdir(parents=True, exist_ok=True)
    debug_path = root / f"debug_{uuid.uuid4().hex[:8]}.jsonl"
    out_path = root / f"out_{uuid.uuid4().hex[:8]}.json"
    rows = [
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
    ]
    try:
        debug_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        code = mod.main(["--debug", str(debug_path), "--out-json", str(out_path)])
        assert code == 0
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["run_summary"]["run_type"] == "summarize_liq_tag_signal_behavior"
        assert payload["overall"]["rows_total"] == 2
        assert payload["overall"]["normal"]["n"] >= 1
    finally:
        debug_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

