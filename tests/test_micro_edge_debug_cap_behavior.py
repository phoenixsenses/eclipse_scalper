from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.micro_edge_backtest as backtest_mod


def _rows(n: int = 20):
    out = []
    mid = 100.0
    for i in range(n):
        mid *= 1.0005
        out.append(
            {
                "ts_ms": float(i * 1000),
                "mid": float(mid),
                "imbalance": 0.8,
                "trade_intensity": 5000.0,
                "spread": 0.0002,
                "ret_1": 0.0001,
            }
        )
    return out


def test_debug_samples_zero_is_no_cap(monkeypatch):
    written: list[dict] = []

    def _fake_append(path, rec):
        written.append(json.loads(json.dumps(rec)))

    monkeypatch.setattr(backtest_mod, "append_jsonl", _fake_append)
    sim = backtest_mod.simulate_rule_trades(
        rows=_rows(25),
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={"imb_q10": -1.0, "imb_q90": 1.0, "int_q90": 0.0, "spr_q90": 1.0},
        labels=None,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=4.0,
        slip_bps=2.0,
        debug_samples=0,
        debug_symbol="BTCUSDT",
        debug_out_path=Path("dummy.jsonl"),
    )
    assert int(sim["debug_stats"]["debug_written"]) == len(written)
    assert len(written) > 2
    info = backtest_mod.debug_cap_info(0, len(written))
    assert info["debug_out_capped"] is False
    assert info["debug_samples_limit"] == 0


def test_debug_samples_cap_applies(monkeypatch):
    written: list[dict] = []

    def _fake_append(path, rec):
        written.append(rec)

    monkeypatch.setattr(backtest_mod, "append_jsonl", _fake_append)
    sim = backtest_mod.simulate_rule_trades(
        rows=_rows(30),
        rule_name="intensity_spike_imbalance_cont",
        side="LONG",
        thresholds={"imb_q10": -1.0, "imb_q90": 1.0, "int_q90": 0.0, "spr_q90": 1.0},
        labels=None,
        hold_buckets=1,
        cooldown_buckets=0,
        fee_bps=4.0,
        slip_bps=2.0,
        debug_samples=2,
        debug_symbol="BTCUSDT",
        debug_out_path=Path("dummy.jsonl"),
    )
    assert int(sim["debug_stats"]["debug_written"]) == 2
    assert len(written) == 2
    info = backtest_mod.debug_cap_info(2, len(written))
    assert info["debug_out_capped"] is True

