from __future__ import annotations

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_passive_pocket_forward as vf


def test_validate_pocket_forward_deterministic(monkeypatch) -> None:
    vf._ROWS_CACHE.clear()
    monkeypatch.setattr(vf.time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(vf, "_load_symbol_trades_and_marks", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        vf,
        "build_bucket_features",
        lambda *args, **kwargs: [
            {"ts_ms": float(i), "mid": 100.0 + i * 0.001, "spread": 0.0001, "trade_intensity": 3000.0, "micro_volatility": 0.001, "imbalance": 0.6, "ret_1": 0.0}
            for i in range(1000)
        ],
    )
    monkeypatch.setattr(vf, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(vf, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(vf, "build_passive_calibration_samples", lambda **kwargs: [{"spread": 0.0001, "trade_intensity": 3000.0, "vol_proxy": 0.001, "imbalance_for_fill": 0.6, "touched": True, "full_proxy": True, "adverse_bps": 0.1, "depth": 0.6}])
    monkeypatch.setattr(vf, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed), "maker_fee_bps": float(maker_fee_bps), "passive_adverse_mult": 1.0})
    monkeypatch.setattr(vf, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(vf, "resolve_symbol_profile", lambda profiles, symbol: {})

    def _sim(**kwargs):
        seed = int(kwargs["passive_params"].get("seed", 0))
        val = ((seed % 7) - 3) * 1e-5
        return {
            "filled_only_metrics": {"n": 60, "avg_net": val, "p90_net": 0.0002, "win_rate": 0.55},
            "attempt_level_metrics": {"fill_rate": 0.5},
        }

    monkeypatch.setattr(vf, "simulate_rule_trades", _sim)

    a = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11,22,33",
        min_n=50,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
    )
    b = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11,22,33",
        min_n=50,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
    )
    assert a["rows_total"] == b["rows_total"]
    assert a["pass_count"] == b["pass_count"]
    assert a["per_combo"] == b["per_combo"]


def test_effective_min_n_with_fraction(monkeypatch) -> None:
    vf._ROWS_CACHE.clear()
    monkeypatch.setattr(vf.time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(vf, "_load_symbol_trades_and_marks", lambda *args, **kwargs: ([], []))
    # 40k rows => split val rows around 10k for splits=4
    monkeypatch.setattr(
        vf,
        "build_bucket_features",
        lambda *args, **kwargs: [
            {"ts_ms": float(i), "mid": 100.0 + i * 0.001, "spread": 0.0001, "trade_intensity": 3000.0, "micro_volatility": 0.001, "imbalance": 0.6, "ret_1": 0.0}
            for i in range(40000)
        ],
    )
    monkeypatch.setattr(vf, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(vf, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(vf, "build_passive_calibration_samples", lambda **kwargs: [{"spread": 0.0001, "trade_intensity": 3000.0, "vol_proxy": 0.001, "imbalance_for_fill": 0.6, "touched": True, "full_proxy": True, "adverse_bps": 0.1, "depth": 0.6}])
    monkeypatch.setattr(vf, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed), "maker_fee_bps": float(maker_fee_bps), "passive_adverse_mult": 1.0})
    monkeypatch.setattr(vf, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(vf, "resolve_symbol_profile", lambda profiles, symbol: {})
    monkeypatch.setattr(
        vf,
        "simulate_rule_trades",
        lambda **kwargs: {"filled_only_metrics": {"n": 40, "avg_net": 0.0002, "p90_net": 0.0003, "win_rate": 0.55}, "attempt_level_metrics": {"fill_rate": 0.5}},
    )
    res = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11",
        min_n=30,
        min_n_frac=0.01,
        maker_fee_bps=0.5,
    )
    assert res["rows_total"] > 0
    assert all(int(r["effective_min_n"]) == max(30, int(math.ceil(0.01 * int(r["val_n_rows"])))) for r in res["per_combo"])
    assert all(str(r["fail_reason"]) == "insufficient_fills" for r in res["per_combo"])


def test_net_per_attempt_deterministic(monkeypatch) -> None:
    """net_per_attempt must be identical across two fresh calls with same inputs (DAT-03)."""
    vf._ROWS_CACHE.clear()
    monkeypatch.setattr(vf.time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(vf, "_load_symbol_trades_and_marks", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        vf,
        "build_bucket_features",
        lambda *args, **kwargs: [
            {"ts_ms": float(i), "mid": 100.0 + i * 0.001, "spread": 0.0001, "trade_intensity": 3000.0, "micro_volatility": 0.001, "imbalance": 0.6, "ret_1": 0.0}
            for i in range(1000)
        ],
    )
    monkeypatch.setattr(vf, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(vf, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(vf, "build_passive_calibration_samples", lambda **kwargs: [{"spread": 0.0001, "trade_intensity": 3000.0, "vol_proxy": 0.001, "imbalance_for_fill": 0.6, "touched": True, "full_proxy": True, "adverse_bps": 0.1, "depth": 0.6}])
    monkeypatch.setattr(vf, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed), "maker_fee_bps": float(maker_fee_bps), "passive_adverse_mult": 1.0})
    monkeypatch.setattr(vf, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(vf, "resolve_symbol_profile", lambda profiles, symbol: {})

    def _sim_with_npa(**kwargs):
        seed = int(kwargs["passive_params"].get("seed", 0))
        val = ((seed % 7) - 3) * 1e-5
        n_att = 80 + (seed % 5)   # deterministic from seed — no wall-clock
        return {
            "filled_only_metrics": {"n": 60, "avg_net": val, "p90_net": 0.0002, "win_rate": 0.55},
            "attempt_level_metrics": {"fill_rate": 0.5, "n_attempts": n_att, "net_per_attempt": val * 0.5},
        }

    monkeypatch.setattr(vf, "simulate_rule_trades", _sim_with_npa)

    common_kwargs = dict(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11,22,33",
        min_n=50,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
    )
    a = vf.validate_pocket_forward(**common_kwargs)
    vf._ROWS_CACHE.clear()
    b = vf.validate_pocket_forward(**common_kwargs)

    assert a["per_combo"] == b["per_combo"], "per_combo must be identical on repeated calls (DAT-03)"
    for row in a["per_combo"]:
        assert "net_per_attempt" in row, "net_per_attempt field must be present in per_combo rows"
        assert "val_attempts" in row, "val_attempts field must be present in per_combo rows"
        assert "val_filled" in row, "val_filled field must be present in per_combo rows"
        assert "attempts_per_min" in row, "attempts_per_min field must be present in per_combo rows"
        assert row["attempts_per_min"] >= 0.0, "attempts_per_min must be non-negative"


def test_effective_min_n_without_fraction(monkeypatch) -> None:
    vf._ROWS_CACHE.clear()
    monkeypatch.setattr(vf.time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(vf, "_load_symbol_trades_and_marks", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        vf,
        "build_bucket_features",
        lambda *args, **kwargs: [
            {"ts_ms": float(i), "mid": 100.0 + i * 0.001, "spread": 0.0001, "trade_intensity": 3000.0, "micro_volatility": 0.001, "imbalance": 0.6, "ret_1": 0.0}
            for i in range(40000)
        ],
    )
    monkeypatch.setattr(vf, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(vf, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(vf, "build_passive_calibration_samples", lambda **kwargs: [{"spread": 0.0001, "trade_intensity": 3000.0, "vol_proxy": 0.001, "imbalance_for_fill": 0.6, "touched": True, "full_proxy": True, "adverse_bps": 0.1, "depth": 0.6}])
    monkeypatch.setattr(vf, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed), "maker_fee_bps": float(maker_fee_bps), "passive_adverse_mult": 1.0})
    monkeypatch.setattr(vf, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(vf, "resolve_symbol_profile", lambda profiles, symbol: {})
    monkeypatch.setattr(
        vf,
        "simulate_rule_trades",
        lambda **kwargs: {"filled_only_metrics": {"n": 40, "avg_net": 0.0002, "p90_net": 0.0003, "win_rate": 0.55}, "attempt_level_metrics": {"fill_rate": 0.5}},
    )
    res = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11",
        min_n=30,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
    )
    assert res["rows_total"] > 0
    assert all(int(r["effective_min_n"]) == 30 for r in res["per_combo"])
    assert all(str(r["fail_reason"]) == "ok" for r in res["per_combo"])


def test_main_prints_min_n_frac_dominance_warning(monkeypatch, capsys) -> None:
    fake = {
        "rows_total": 2,
        "pass_count": 0,
        "pass_rate": 0.0,
        "insufficient_fill_rate": 1.0,
        "min_n_frac_dominance_rate": 1.0,
        "frac_min_component_median": 212,
        "effective_min_n_median": 212,
        "per_split": [],
        "per_combo": [
            {
                "seed": 11,
                "split": 1,
                "train_n": 100,
                "val_n_rows": 70632,
                "frac_min_component": 212,
                "effective_min_n": 212,
                "filled_n": 35,
                "filled_avg_net": -0.0001,
                "filled_p90_net": -0.00005,
                "filled_win_rate": 0.4,
                "attempt_fill_rate": 0.45,
                "net_per_attempt": -0.0001,
                "attempts_per_min": 2.0,
                "val_attempts": 1000,
                "val_attempts_before_gate": 1200,
                "val_attempts_after_gate": 1000,
                "fail_reason": "insufficient_fills",
                "pass": False,
            },
            {
                "seed": 22,
                "split": 1,
                "train_n": 100,
                "val_n_rows": 70632,
                "frac_min_component": 212,
                "effective_min_n": 212,
                "filled_n": 50,
                "filled_avg_net": -0.0001,
                "filled_p90_net": -0.00005,
                "filled_win_rate": 0.4,
                "attempt_fill_rate": 0.45,
                "net_per_attempt": -0.0001,
                "attempts_per_min": 2.0,
                "val_attempts": 1000,
                "val_attempts_before_gate": 1200,
                "val_attempts_after_gate": 1000,
                "fail_reason": "insufficient_fills",
                "pass": False,
            },
        ],
    }
    monkeypatch.setattr(vf, "validate_pocket_forward", lambda **kwargs: fake)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--out-md",
            "reports/test_validate_forward_minfrac.md",
            "--min-n",
            "50",
            "--min-n-frac",
            "0.003",
        ],
    )
    rc = vf.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "effective_min_n formula" in out
    assert "WARNING min_n_frac dominates" in out


def test_regime_bucket_is_deterministic(monkeypatch) -> None:
    vf._ROWS_CACHE.clear()
    monkeypatch.setattr(vf.time, "time", lambda: 1700000000.0)
    monkeypatch.setattr(vf, "_load_symbol_trades_and_marks", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(
        vf,
        "build_bucket_features",
        lambda *args, **kwargs: [
            {
                "ts_ms": float(i),
                "mid": 100.0 + i * 0.001,
                "spread": 0.0001 + (i % 4) * 0.0001,
                "trade_intensity": 2000.0 + (i % 5) * 100.0,
                "micro_volatility": 0.001 + (i % 3) * 0.0002,
                "imbalance": 0.6,
                "ret_1": 0.0001,
            }
            for i in range(1000)
        ],
    )
    monkeypatch.setattr(vf, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(vf, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(vf, "build_passive_calibration_samples", lambda **kwargs: [])
    monkeypatch.setattr(vf, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed)})
    monkeypatch.setattr(vf, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(vf, "resolve_symbol_profile", lambda profiles, symbol: {})
    monkeypatch.setattr(
        vf,
        "simulate_rule_trades",
        lambda **kwargs: {
            "filled_only_metrics": {"n": 60, "avg_net": 0.0001, "p90_net": 0.0002, "win_rate": 0.55},
            "attempt_level_metrics": {"fill_rate": 0.5, "n_attempts": 4, "net_per_attempt": 0.0001},
            "attempt_rows": [
                {"signal_idx": 1, "filled": True, "net_return": 0.0002},
                {"signal_idx": 2, "filled": False, "net_return": 0.0},
                {"signal_idx": 3, "filled": True, "net_return": 0.0001},
                {"signal_idx": 4, "filled": True, "net_return": 0.0002},
            ],
        },
    )
    a = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11,22",
        min_n=30,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
        regime_bucket="spread_q",
    )
    b = vf.validate_pocket_forward(
        db="data/microstructure.db",
        symbol="ETHUSDT",
        lookback_min=100,
        bucket_sec=1,
        horizon_sec=60,
        rule="r",
        side="auto",
        min_imbalance=0.5,
        min_trade_intensity=2500,
        max_spread=0.00025,
        splits=4,
        seeds="11,22",
        min_n=30,
        min_n_frac=0.0,
        maker_fee_bps=0.5,
        regime_bucket="spread_q",
    )
    assert a["per_regime"] == b["per_regime"]
    assert len(a["per_regime"]) >= 1
