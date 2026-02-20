from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import sweep_passive_realistic_filters as sweep


class _DummyConn:
    def close(self) -> None:
        return None


def _rows(n: int = 300) -> list[dict[str, float]]:
    out = []
    for i in range(n):
        out.append(
            {
                "ts_ms": float(i),
                "mid": 100.0 + i * 0.001,
                "spread": 0.0002,
                "trade_intensity": 3000.0,
                "imbalance": 0.6,
                "ret_1": 0.0001,
                "micro_volatility": 0.001,
            }
        )
    return out


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr(sweep.sqlite3, "connect", lambda *a, **k: _DummyConn())
    monkeypatch.setattr(sweep, "_parse_symbols", lambda raw: ["ETHUSDT"])
    monkeypatch.setattr(sweep, "_load_symbol_trades_and_marks", lambda *a, **k: ([], []))
    monkeypatch.setattr(sweep, "build_bucket_features", lambda *a, **k: _rows())
    monkeypatch.setattr(sweep, "enrich_rows_with_v2", lambda rows, **k: rows)
    monkeypatch.setattr(sweep, "compute_rule_thresholds", lambda rows: {})
    monkeypatch.setattr(sweep, "compute_regime_bins", lambda rows: {"vol": (0.0, 0.0, 0.001), "intensity": (0.0, 0.0, 2500.0)})
    monkeypatch.setattr(sweep, "load_passive_profiles", lambda path: {})
    monkeypatch.setattr(sweep, "resolve_symbol_profile", lambda profiles, symbol: {})
    monkeypatch.setattr(sweep, "build_passive_calibration_samples", lambda **kwargs: [])
    monkeypatch.setattr(sweep, "calibrate_passive_model", lambda samples, maker_fee_bps, seed: {"seed": int(seed)})
    monkeypatch.setattr(
        sweep,
        "simulate_rule_trades",
        lambda **kwargs: {
            "filled_only_metrics": {"n": 100, "avg_net": 0.0002, "p90_net": 0.0003, "win_rate": 0.55},
            "attempt_level_metrics": {"n_attempts": 150, "fill_rate": 0.5, "net_per_attempt": 0.00012},
        },
    )


def _run(monkeypatch, min_n: int, out_md: str) -> str:
    _patch_common(monkeypatch)
    calls: list[float] = []

    def _fake_validate(**kwargs):
        calls.append(float(kwargs.get("min_n", 0) or 0.0))
        if int(min_n) > 100:
            insuff = 1.0
            afr = 0.45
            npa = 0.00012
        else:
            insuff = 0.0
            afr = 0.45
            npa = 0.00012
        return {
            "insufficient_fill_rate": insuff,
            "per_combo": [
                {
                    "val_attempts": 120,
                    "val_filled": 55,
                    "attempt_fill_rate": afr,
                    "attempts_per_min": 1.8,
                    "net_per_attempt": npa,
                }
            ],
        }

    monkeypatch.setattr(sweep, "validate_pocket_forward", _fake_validate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--db",
            "data/microstructure.db",
            "--symbols",
            "ETHUSDT",
            "--horizons",
            "60",
            "--min-imbalance-grid",
            "0.5",
            "--min-trade-intensity-grid",
            "2500",
            "--max-spread-grid",
            "0.0005",
            "--rule",
            "micro_edge_v2_passive_alpha",
            "--min-validation-n",
            "30",
            "--splits",
            "4",
            "--seeds",
            "11,22",
            "--min-n",
            str(min_n),
            "--min-n-frac",
            "0.0",
            "--min-attempt-fill-rate",
            "0.10",
            "--max-insufficient-fill-rate",
            "0.50",
            "--top-k",
            "5",
            "--out-md",
            out_md,
        ],
    )
    rc = sweep.main()
    assert rc == 0
    assert calls, "validate_pocket_forward mock was not called"
    return Path(out_md).read_text(encoding="utf-8")


def test_sweep_pass_shrinks_when_min_n_increased(monkeypatch) -> None:
    low = _run(monkeypatch, min_n=50, out_md="reports/test_sweep_capacity_low.md")
    assert " | YES | YES |" in low  # cap_ok, pass
    high = _run(monkeypatch, min_n=500, out_md="reports/test_sweep_capacity_high.md")
    assert " | NO | NO |" in high


def test_sweep_capacity_columns_present(monkeypatch) -> None:
    txt = _run(monkeypatch, min_n=50, out_md="reports/test_sweep_capacity_cols.md")
    assert "cap_attempt_fill_rate" in txt
    assert "insufficient_fill_rate" in txt
    assert "cap_ok" in txt
