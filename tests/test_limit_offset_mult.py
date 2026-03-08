"""Tests for limit_offset_mult parameter wiring through the passive fill chain."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.passive_execution_simulator import _event_touch_and_depth
from tools import micro_edge_backtest as mb


# ---------------------------------------------------------------------------
# Unit tests: _event_touch_and_depth
# ---------------------------------------------------------------------------


def test_touch_long_tighter_mult_fills_sooner() -> None:
    """LONG: mult=0.25 limit is closer to entry → fills on a smaller dip."""
    ep = 100.0
    sp = 0.001  # 0.1% spread

    # Price dips by 0.03% then recovers — just enough for mult=0.25 but not 0.5
    future_mids = [100.0, 99.975, 99.98, 100.0]

    touched_tight, _, idx_tight = _event_touch_and_depth(
        side="LONG",
        entry_price=ep,
        spread_ratio=sp,
        future_mids=future_mids,
        limit_offset_mult=0.25,
    )
    touched_wide, _, idx_wide = _event_touch_and_depth(
        side="LONG",
        entry_price=ep,
        spread_ratio=sp,
        future_mids=future_mids,
        limit_offset_mult=0.50,
    )

    # mult=0.25: limit = 100*(1 - 0.25*0.001) = 99.975 → touches at index 1
    assert touched_tight is True
    assert idx_tight == 1

    # mult=0.50: limit = 100*(1 - 0.50*0.001) = 99.95 → price never reaches it
    assert touched_wide is False


def test_touch_short_tighter_mult_fills_sooner() -> None:
    """SHORT: mult=0.25 limit is closer to entry → fills on a smaller rally."""
    ep = 100.0
    sp = 0.001

    # Price rallies by 0.03% then falls back — enough for mult=0.25 but not 0.5
    future_mids = [100.0, 100.025, 100.02, 100.0]

    touched_tight, _, idx_tight = _event_touch_and_depth(
        side="SHORT",
        entry_price=ep,
        spread_ratio=sp,
        future_mids=future_mids,
        limit_offset_mult=0.25,
    )
    touched_wide, _, idx_wide = _event_touch_and_depth(
        side="SHORT",
        entry_price=ep,
        spread_ratio=sp,
        future_mids=future_mids,
        limit_offset_mult=0.50,
    )

    # mult=0.25: limit = 100*(1 + 0.25*0.001) = 100.025 → touches at index 1
    assert touched_tight is True
    assert idx_tight == 1

    # mult=0.50: limit = 100*(1 + 0.50*0.001) = 100.05 → price never reaches it
    assert touched_wide is False


def test_touch_no_movement_never_fills() -> None:
    """Flat price → no fill regardless of mult."""
    future_mids = [100.0] * 10

    for mult in [0.1, 0.25, 0.5, 0.75, 1.0]:
        touched, _, _ = _event_touch_and_depth(
            side="LONG",
            entry_price=100.0,
            spread_ratio=0.001,
            future_mids=future_mids,
            limit_offset_mult=mult,
        )
        assert touched is False, f"Expected no fill for flat price at mult={mult}"


def test_touch_zero_mult_fills_immediately() -> None:
    """mult=0.0: limit = entry price exactly → fills on first tick at or below entry."""
    ep = 100.0
    future_mids = [100.0, 99.99, 99.98]

    touched, depth, idx = _event_touch_and_depth(
        side="LONG",
        entry_price=ep,
        spread_ratio=0.001,
        future_mids=future_mids,
        limit_offset_mult=0.0,
    )

    # limit = 100*(1 - 0) = 100.0 → first tick at 100.0 satisfies px <= limit
    assert touched is True
    assert idx == 0


# ---------------------------------------------------------------------------
# Integration test: fill rate monotonically decreases as mult increases
# ---------------------------------------------------------------------------


def test_fill_rate_decreases_as_mult_increases() -> None:
    """
    With tighter limit (lower mult), more signals get filled.
    Using a LONG scenario with modest dips: mult=0.25 fills more than mult=0.75.
    """
    # Build rows with a gentle trending-up market (price rises modestly each bar)
    # For LONG: fills when price DROPS. With a gently rising market,
    # only small dips occur, so tighter limits (0.25) fill more than wide (0.75).
    rows = []
    for i in range(300):
        # Price oscillates: rises slowly with periodic small dips
        mid = 100.0 + i * 0.001 + (-0.02 if i % 10 == 5 else 0.0)
        rows.append({
            "ts_ms": float(1000 + i),
            "mid": mid,
            "spread": 0.001,
            "trade_intensity": 5000.0,
            "micro_volatility": 0.001,
            "imbalance": 0.6,
            "ret_1": 0.0,
            "volume": 1.0,
            "liquidity_proxy": 1.0,
        })

    base_params = {
        "seed": 42,
        "base_touch": 1.0,
        "base_full_cond_touch": 1.0,
        "base_adverse_bps": 0.2,
        "maker_fee_bps": 0.5,
        "edges": {
            "spread": [0.0005, 0.0015],
            "trade_intensity": [1000.0, 8000.0],
            "vol_proxy": [0.0005, 0.0015],
            "imbalance_for_fill": [0.4, 0.8],
        },
        "touch_rates": {k: [1, 1, 1] for k in ["spread", "trade_intensity", "vol_proxy", "imbalance_for_fill"]},
        "full_rates": {k: [1, 1, 1] for k in ["spread", "trade_intensity", "vol_proxy", "imbalance_for_fill"]},
        "adverse_means": {k: [0.2, 0.2, 0.2] for k in ["spread", "trade_intensity", "vol_proxy", "imbalance_for_fill"]},
        "queue_competition_strength": 0.0,
        "adverse_toxicity_strength": 0.0,
    }

    def _run(mult: float) -> dict:
        return mb.simulate_rule_trades(
            rows=rows,
            rule_name="intensity_spike_imbalance_cont",
            side="LONG",
            thresholds={},
            labels=None,
            hold_buckets=10,
            cooldown_buckets=0,
            fee_bps=0.0,
            slip_bps=0.0,
            exec_model="passive_realistic",
            passive_params={**base_params, "limit_offset_mult": mult},
        )

    sim_tight = _run(0.25)
    sim_wide = _run(0.75)

    tight_fills = int(sim_tight.get("filled_only_metrics", {}).get("n_filled", 0))
    wide_fills = int(sim_wide.get("filled_only_metrics", {}).get("n_filled", 0))

    # Tighter limit → fills easier → more fills
    assert tight_fills >= wide_fills, (
        f"Expected tight (0.25) fills >= wide (0.75) fills, "
        f"got tight={tight_fills} wide={wide_fills}"
    )
