from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import analyze_micro_edge_regimes as mod


def test_analyze_regimes_grouping_and_flags():
    root = Path(__file__).resolve().parents[1] / "logs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test_analyze_micro_edge_regimes.jsonl"
    rows = [
        {
            "ts_bucket": 1,
            "gross_ret": 0.0020,
            "cost": 0.0010,
            "net_ret": 0.0010,
            "exec_model": "taker",
            "horizon_sec": 30,
            "regime_spread_bin": "<=p25",
            "regime_intensity_bin": ">p75",
            "regime_vol_bin": "p25-50",
            "regime_imb_bin": "+[0.5,0.7)",
        },
        {
            "ts_bucket": 2,
            "gross_ret": 0.0018,
            "cost": 0.0010,
            "net_ret": 0.0008,
            "exec_model": "taker",
            "horizon_sec": 30,
            "regime_spread_bin": "<=p25",
            "regime_intensity_bin": ">p75",
            "regime_vol_bin": "p25-50",
            "regime_imb_bin": "+[0.5,0.7)",
        },
        {
            "ts_bucket": 3,
            "gross_ret": -0.0002,
            "cost": 0.0010,
            "net_ret": -0.0012,
            "exec_model": "taker",
            "horizon_sec": 30,
            "regime_spread_bin": ">p75",
            "regime_intensity_bin": "<=p25",
            "regime_vol_bin": ">p75",
            "regime_imb_bin": "-[0.5,0.7)",
        },
        {
            "ts_bucket": 4,
            "gross_ret": -0.0001,
            "cost": 0.0010,
            "net_ret": -0.0011,
            "exec_model": "taker",
            "horizon_sec": 30,
            "regime_spread_bin": ">p75",
            "regime_intensity_bin": "<=p25",
            "regime_vol_bin": ">p75",
            "regime_imb_bin": "-[0.5,0.7)",
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    loaded = mod.load_debug_rows(path)
    out = mod.analyze(
        loaded,
        group_fields=["regime_spread_bin", "regime_intensity_bin", "regime_vol_bin", "regime_imb_bin"],
        min_n=2,
    )
    assert len(out) == 2
    neg_group = [r for r in out if bool(r["p90_net_negative"])]
    assert len(neg_group) == 1
    pos_group = [r for r in out if not bool(r["p90_net_negative"])][0]
    assert abs(float(pos_group["break_even_bps_total"]) - (((0.0020 + 0.0018) / 2.0) * 10000.0)) < 1e-9

