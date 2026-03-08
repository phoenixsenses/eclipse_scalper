from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import summarize_event_conditioned_forward_grid as mod


def test_summarize_event_conditioned_forward_grid_payload() -> None:
    root = Path("localtests") / "event_conditioned_forward_grid_summary"
    root.mkdir(parents=True, exist_ok=True)
    grid_json = root / f"grid_{uuid.uuid4().hex[:8]}.json"
    out_json = root / f"summary_{uuid.uuid4().hex[:8]}.json"
    payload = {
        "rows": [
            {"variant": "primary_allow_block", "validation_delta_avg_net": 0.0012, "validation_kept_ratio": 0.03},
            {"variant": "block_only", "validation_delta_avg_net": 0.0001, "validation_kept_ratio": 0.74},
            {"variant": "baseline_like", "validation_delta_avg_net": 0.0, "validation_kept_ratio": 1.0},
        ]
    }
    try:
        grid_json.write_text(json.dumps(payload), encoding="utf-8")
        built = mod.build_payload(grid_json=str(grid_json), out_json=str(out_json))
        assert built["best_quality_variant"]["variant"] == "primary_allow_block"
        assert built["best_tradeoff_variant"]["variant"] == "block_only"
        assert built["recommendation"] == "test_tradeoff_variant_in_rank_pipeline"
        assert built["run_summary"]["run_type"] == "summarize_event_conditioned_forward_grid"
    finally:
        grid_json.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
