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


def _write_stability_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(row.keys())
    lines = [",".join(headers), ",".join(str(row[h]) for h in headers)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_alpha_gate_stability_mode_blocks_on_low_pos_slice_frac(monkeypatch) -> None:
    tmp_path = Path("eclipse_scalper/localtests/alpha_stability") / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        _write_stability_csv(
            Path("runs/latest/stability.csv"),
            {
                "slices_count": "4",
                "pos_slices_count": "1",
                "pos_slices_frac": "0.25",
                "pnl_net_sum_total": "-1.0",
                "pnl_net_sum_mean": "-0.25",
                "pnl_net_sum_median": "-0.2",
                "pnl_net_sum_std": "0.1",
                "pnl_net_sum_min": "-0.4",
                "pnl_net_sum_max": "0.0",
                "worst_pnl_net_per_fill": "-0.01",
                "fill_rate_mean": "0.3",
                "stability_score": "-0.2",
            },
        )
        monkeypatch.setenv("ALPHA_GATE_MODE", "stability")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_PATH", "runs/latest/stability.csv")
        monkeypatch.setenv("ALPHA_GATE_MIN_POS_SLICES_FRAC", "0.5")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is True
        assert dec.reason == "alpha_unstable"
        assert dec.details.get("alpha_gate_mode") == "stability"
    finally:
        os.chdir(old)


def test_alpha_gate_both_mode_requires_metrics_and_stability(monkeypatch) -> None:
    tmp_path = Path("eclipse_scalper/localtests/alpha_stability") / uuid.uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        Path("runs/latest").mkdir(parents=True, exist_ok=True)
        Path("runs/latest/metrics.json").write_text(
            '{"fills_count": 10, "decisions_count": 20, "pnl_net_sum": 2.0, "fee_dominates_count": 1, "spread_cost_est_sum": 0.0}\n',
            encoding="utf-8",
        )
        _write_stability_csv(
            Path("runs/latest/stability.csv"),
            {
                "slices_count": "4",
                "pos_slices_count": "3",
                "pos_slices_frac": "0.75",
                "pnl_net_sum_total": "1.0",
                "pnl_net_sum_mean": "0.25",
                "pnl_net_sum_median": "0.2",
                "pnl_net_sum_std": "0.1",
                "pnl_net_sum_min": "-0.1",
                "pnl_net_sum_max": "0.5",
                "worst_pnl_net_per_fill": "-0.02",
                "fill_rate_mean": "0.4",
                "stability_score": "0.1",
            },
        )
        monkeypatch.setenv("ALPHA_GATE_MODE", "both")
        monkeypatch.setenv("ALPHA_GATE_METRICS_PATH", "runs/latest/metrics.json")
        monkeypatch.setenv("ALPHA_GATE_STABILITY_PATH", "runs/latest/stability.csv")
        monkeypatch.setenv("ALPHA_GATE_MAX_WORST_PNL_NET_PER_FILL", "0.01")
        monkeypatch.setenv("ALPHA_GATE_MAX_STALENESS_SEC", "999999")
        dec = evaluate_alpha_gate_from_env(now_ts=time.time())
        assert dec.blocked is True
        assert dec.reason == "alpha_tail_risk"
        assert dec.details.get("alpha_gate_mode") == "both"
    finally:
        os.chdir(old)
