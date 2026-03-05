from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

try:
    from execution.alpha_gate import evaluate_alpha_gate_from_env
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from execution.alpha_gate import evaluate_alpha_gate_from_env


def _write_stability(path: Path, *, pos_frac: float, score: float, worst: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "slices_count,pos_slices_count,pos_slices_frac,pnl_net_sum_total,pnl_net_sum_mean,pnl_net_sum_median,pnl_net_sum_std,pnl_net_sum_min,pnl_net_sum_max,worst_pnl_net_per_fill,fill_rate_mean,stability_score,combined_score,regime\n"
        f"4,2,{pos_frac},1.0,0.25,0.2,0.1,-0.2,0.5,{worst},0.4,{score},{score},all\n",
        encoding="utf-8",
    )


def _write_metrics(path: Path, pnl: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{"
            f"\"fills_count\":10,\"decisions_count\":20,\"pnl_net_sum\":{pnl},"
            "\"fee_dominates_count\":1,\"spread_cost_est_sum\":0.0"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_alpha_gate_both_regime_blocks_up_score_low(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_both_regime") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        _write_metrics(Path("runs/latest/metrics.json"), pnl=1.0)
        _write_stability(Path("runs/latest/stability.csv"), pos_frac=0.8, score=0.2, worst=-0.01)
        _write_stability(Path("runs/latest/stability_up.csv"), pos_frac=0.8, score=-0.2, worst=-0.01)
        _write_stability(Path("runs/latest/stability_down.csv"), pos_frac=0.8, score=0.2, worst=-0.01)
        monkeypatch.setenv("ALPHA_GATE_MODE", "both_regime")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        monkeypatch.setenv("ALPHA_GATE_MIN_STABILITY_SCORE_UP", "0.0")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is True
        assert dec.reason == "alpha_regime_up_score_low"
    finally:
        os.chdir(old)


def test_alpha_gate_both_regime_uses_last_good_fallback(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_both_regime") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        _write_metrics(Path("runs/latest/metrics.json"), pnl=1.0)
        _write_stability(Path("runs/latest/stability.csv"), pos_frac=0.8, score=0.2, worst=-0.01)
        _write_stability(Path("runs/last_good/stability_up.csv"), pos_frac=0.8, score=0.2, worst=-0.01)
        _write_stability(Path("runs/last_good/stability_down.csv"), pos_frac=0.8, score=0.2, worst=-0.01)
        monkeypatch.setenv("ALPHA_GATE_MODE", "both_regime")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is False
        assert dec.details.get("regime_up_fallback_used") is True
        assert dec.details.get("regime_down_fallback_used") is True
    finally:
        os.chdir(old)

