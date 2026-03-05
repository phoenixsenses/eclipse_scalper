from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import triage_capacity as tc


def test_triage_capacity_outputs(monkeypatch) -> None:
    monkeypatch.setattr(
        tc,
        "_parse_candidates_from_md",
        lambda p, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {},
        ),
    )

    monkeypatch.setattr(
        tc,
        "validate_pocket_forward",
        lambda **kwargs: {
            "per_combo": [
                {"attempts_per_min": 0.08, "effective_min_n": 60, "filled_n": 45, "fail_reason": "ok"},
                {"attempts_per_min": 0.10, "effective_min_n": 62, "filled_n": 40, "fail_reason": "insufficient_fills"},
            ]
        },
    )

    out_json = Path("reports/test_triage_capacity.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--out-json",
            str(out_json),
        ],
    )
    rc = tc.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(data["rows"]) == 1
    row = data["rows"][0]
    assert "effective_min_n_median" in row
    assert "prob_fail_insufficient_fills" in row


def test_triage_capacity_gate_config_and_determinism(monkeypatch) -> None:
    monkeypatch.setattr(
        tc,
        "_parse_candidates_from_md",
        lambda p, debug=False: (
            [
                {"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025},
            ],
            {},
        ),
    )

    calls = []

    def _fake_validate(**kwargs):
        calls.append(
            {
                "max_volatility_extreme": float(kwargs.get("max_volatility_extreme", 0.0) or 0.0),
                "vol_quantile_reject": float(kwargs.get("vol_quantile_reject", 0.0) or 0.0),
            }
        )
        return {
            "per_combo": [
                {"attempts_per_min": 0.08, "effective_min_n": 60, "filled_n": 45, "fail_reason": "ok"},
                {"attempts_per_min": 0.10, "effective_min_n": 62, "filled_n": 40, "fail_reason": "insufficient_fills"},
            ]
        }

    monkeypatch.setattr(tc, "validate_pocket_forward", _fake_validate)

    out_a = Path("reports/test_triage_capacity_gate_a.json")
    out_b = Path("reports/test_triage_capacity_gate_b.json")
    argv = [
        "x",
        "--candidates-md",
        "reports/dummy.md",
        "--out-json",
        str(out_a),
        "--mitigation-profile",
        "anti_adverse_v3",
        "--vol-quantile-reject",
        "0.01",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = tc.main()
    assert rc == 0
    data_a = json.loads(out_a.read_text(encoding="utf-8"))
    assert data_a["gate_config"]["mitigation_profile"] == "anti_adverse_v3"
    assert abs(float(data_a["gate_config"]["vol_quantile_reject"]) - 0.01) < 1e-12
    assert data_a["gate_config"]["max_volatility_extreme"] is None

    # Run again with same settings; output should be deterministic.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--out-json",
            str(out_b),
            "--mitigation-profile",
            "anti_adverse_v3",
            "--vol-quantile-reject",
            "0.01",
        ],
    )
    rc = tc.main()
    assert rc == 0
    data_b = json.loads(out_b.read_text(encoding="utf-8"))
    assert data_a["gate_config"] == data_b["gate_config"]
    assert data_a["rows"] == data_b["rows"]
    assert calls, "validate_pocket_forward was not called"
    assert all(abs(c["vol_quantile_reject"] - 0.01) < 1e-12 for c in calls)


def test_triage_capacity_max_volatility_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(
        tc,
        "_parse_candidates_from_md",
        lambda p, debug=False: (
            [{"symbol": "ETHUSDT", "horizon_sec": 120, "min_imbalance": 0.5, "min_trade_intensity": 2500.0, "max_spread": 0.00025}],
            {},
        ),
    )
    seen = {"max_vol": []}

    def _fake_validate(**kwargs):
        seen["max_vol"].append(float(kwargs.get("max_volatility_extreme", 0.0) or 0.0))
        return {"per_combo": [{"attempts_per_min": 0.1, "effective_min_n": 20, "filled_n": 20, "fail_reason": "ok"}]}

    monkeypatch.setattr(tc, "validate_pocket_forward", _fake_validate)
    out_json = Path("reports/test_triage_capacity_max_vol.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--candidates-md",
            "reports/dummy.md",
            "--out-json",
            str(out_json),
            "--mitigation-profile",
            "anti_adverse_v2",
            "--max-volatility-extreme",
            "0.004",
        ],
    )
    rc = tc.main()
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert abs(float(data["gate_config"]["max_volatility_extreme"]) - 0.004) < 1e-12
    assert abs(seen["max_vol"][0] - 0.004) < 1e-12
