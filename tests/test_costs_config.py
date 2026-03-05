from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _reload_costs_module():
    import config.costs as costs

    return importlib.reload(costs)


def test_default_maker_fee_from_env_fallback(monkeypatch) -> None:
    monkeypatch.delenv("MAKER_FEE_BPS", raising=False)
    costs = _reload_costs_module()
    assert float(costs.DEFAULT_MAKER_FEE_BPS) == 1.0


def test_default_maker_fee_from_env_negative_rebate(monkeypatch) -> None:
    monkeypatch.setenv("MAKER_FEE_BPS", "-0.25")
    costs = _reload_costs_module()
    assert float(costs.DEFAULT_MAKER_FEE_BPS) == -0.25


def test_default_maker_fee_invalid_env_uses_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MAKER_FEE_BPS", "not-a-number")
    costs = _reload_costs_module()
    assert float(costs.DEFAULT_MAKER_FEE_BPS) == 1.0


def test_validate_parser_uses_cost_default(monkeypatch) -> None:
    monkeypatch.setenv("MAKER_FEE_BPS", "-0.75")
    import config.costs as costs
    import tools.validate_passive_pocket_forward as vf

    importlib.reload(costs)
    importlib.reload(vf)
    monkeypatch.setattr(sys, "argv", ["x"])
    args = vf._args()
    assert float(args.maker_fee_bps) == -0.75


def test_passive_fill_allows_negative_fee_rebate(monkeypatch) -> None:
    monkeypatch.setenv("MAKER_FEE_BPS", "-1.0")
    import config.costs as costs
    import execution.passive_execution_simulator as pes

    importlib.reload(costs)
    importlib.reload(pes)
    monkeypatch.setattr(pes, "_u01_from_key", lambda seed, key: 0.0)
    out = pes.simulate_passive_fill(
        event={
            "event_id": "evt-1",
            "symbol": "ETHUSDT",
            "side": "LONG",
            "entry_price": 100.0,
            "future_mids": [99.4, 99.3, 99.2],
        },
        horizon_sec=30,
        features={
            "spread": 0.01,  # 1%
            "trade_intensity": 2000.0,
            "vol_proxy": 0.001,
            "imbalance_for_fill": 0.7,
        },
            params={
                "seed": 7,
                "maker_fee_bps": -1.0,
                "base_touch": 1.0,
                "base_full_cond_touch": 1.0,
                "base_adverse_bps": 0.0,
            "adverse_means": {},
            "touch_rates": {},
            "full_rates": {},
            "edges": {},
        },
    )
    assert bool(out["filled"]) is True
    assert float(out["effective_cost_bps"]) < 0.0
