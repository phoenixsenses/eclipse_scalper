from __future__ import annotations

import json
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def test_alpha_gate_uses_last_good_when_latest_missing(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_fallback") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        _write_json(Path("runs/last_good/metrics.json"), {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": 1.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.0})
        monkeypatch.setenv("ALPHA_GATE_METRICS_PATH", "runs/latest/metrics.json")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is False
        assert dec.details.get("metrics_fallback_used") is True
    finally:
        os.chdir(old)


def test_alpha_gate_uses_last_good_when_latest_stale(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_fallback") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        _write_json(Path("runs/latest/metrics.json"), {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": -1.0, "fee_dominates_count": 10, "spread_cost_est_sum": 1.0})
        old_ts = time.time() - 10_000
        os.utime("runs/latest/metrics.json", (old_ts, old_ts))
        _write_json(Path("runs/last_good/metrics.json"), {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": 2.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.0})
        monkeypatch.setenv("ALPHA_GATE_METRICS_PATH", "runs/latest/metrics.json")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "10")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is False
        assert dec.details.get("metrics_fallback_used") is True
    finally:
        os.chdir(old)


def test_alpha_gate_updates_last_good_only_when_ok(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_fallback") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        _write_json(Path("runs/latest/metrics.json"), {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": 2.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.0})
        _write_json(Path("runs/latest/config.json"), {"strategy": "x"})
        monkeypatch.setenv("ALPHA_GATE_METRICS_PATH", "runs/latest/metrics.json")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        dec_ok = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec_ok.blocked is False
        assert Path("runs/last_good/metrics.json").exists()
        m_ok = Path("runs/last_good/metrics.json").read_text(encoding="utf-8")

        _write_json(Path("runs/latest/metrics.json"), {"fills_count": 10, "decisions_count": 20, "pnl_net_sum": -2.0, "fee_dominates_count": 10, "spread_cost_est_sum": 1.0})
        dec_bad = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec_bad.blocked is True
        m_after = Path("runs/last_good/metrics.json").read_text(encoding="utf-8")
        assert m_after == m_ok
    finally:
        os.chdir(old)


def test_alpha_gate_uses_last_good_stability_when_latest_stale(monkeypatch) -> None:
    base = Path("eclipse_scalper/localtests/alpha_fallback") / uuid.uuid4().hex
    base.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(base)
    try:
        Path("runs/latest").mkdir(parents=True, exist_ok=True)
        Path("runs/latest/stability.csv").write_text(
            "slices_count,pos_slices_count,pos_slices_frac,pnl_net_sum_total,pnl_net_sum_mean,pnl_net_sum_median,pnl_net_sum_std,pnl_net_sum_min,pnl_net_sum_max,worst_pnl_net_per_fill,fill_rate_mean,stability_score\n"
            "2,0,0.0,-1,-0.5,-0.5,0.1,-0.6,-0.4,-0.2,0.1,-0.2\n",
            encoding="utf-8",
        )
        old_ts = time.time() - 10_000
        os.utime("runs/latest/stability.csv", (old_ts, old_ts))
        Path("runs/last_good").mkdir(parents=True, exist_ok=True)
        Path("runs/last_good/stability.csv").write_text(
            "slices_count,pos_slices_count,pos_slices_frac,pnl_net_sum_total,pnl_net_sum_mean,pnl_net_sum_median,pnl_net_sum_std,pnl_net_sum_min,pnl_net_sum_max,worst_pnl_net_per_fill,fill_rate_mean,stability_score\n"
            "3,3,1.0,1,0.333,0.3,0.1,0.2,0.5,-0.01,0.4,0.2\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("ALPHA_GATE_MODE", "stability")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_PATH", "runs/latest/stability.csv")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "10")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is False
        assert dec.details.get("stability_fallback_used") is True
        assert dec.details.get("stability_path_used") == "runs/last_good/stability.csv"
    finally:
        os.chdir(old)
