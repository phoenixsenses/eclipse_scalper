from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import calibrate_capacity_thresholds as cct


def test_calibrate_capacity_thresholds_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        cct,
        "_parse_candidates_from_md",
        lambda p, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 60, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
                {"symbol": "BTCUSDT", "horizon_sec": 120, "min_imbalance": 0.4, "min_trade_intensity": 3000.0, "max_spread": 0.00030},
            ],
            {},
        ),
    )

    def _fake_validate(**kwargs):
        frac = float(kwargs.get("min_n_frac", 0.0))
        base = 0.00002 if frac <= 0.0005 else -0.00001
        insuff = 0.0 if frac <= 0.0005 else 1.0
        per = [
            {"attempts_per_min": 1.5, "attempt_fill_rate": 0.45, "net_per_attempt": base, "pass": base > 0},
            {"attempts_per_min": 2.0, "attempt_fill_rate": 0.50, "net_per_attempt": base, "pass": base > 0},
        ]
        return {
            "pass_rate": 0.5 if base > 0 else 0.0,
            "insufficient_fill_rate": insuff,
            "per_combo": per,
        }

    monkeypatch.setattr(cct, "validate_pocket_forward", _fake_validate)

    out_md = Path("reports/test_capacity_calib.md")
    out_json = Path("reports/test_capacity_calib.json")
    argv = [
        "x",
        "--candidates-md",
        "reports/dummy.md",
        "--min-n-frac-grid",
        "0.0005,0.003",
        "--out-md",
        str(out_md),
        "--out-json",
        str(out_json),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = cct.main()
    assert rc == 0
    txt = out_md.read_text(encoding="utf-8")
    assert "capacity_pass_pct" in txt
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 2
    assert float(data["rows"][0]["median_net_per_attempt"]) > float(data["rows"][1]["median_net_per_attempt"])
