from __future__ import annotations

import json
import uuid
from pathlib import Path

try:
    from tools.compare_runs import compare_runs
except ModuleNotFoundError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.compare_runs import compare_runs


def _mk_run(path: Path, cfg: dict, met: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(cfg, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(met, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_compare_runs_detects_config_and_metrics_delta() -> None:
    base = Path("eclipse_scalper/localtests/compare_runs") / uuid.uuid4().hex
    a = base / "a"
    b = base / "b"
    _mk_run(
        a,
        {"strategy": "baseline", "execution_sim": {"spread_bps": 0.0}},
        {"pnl_net_sum": 1.5, "fills_count": 10, "spread_cost_est_sum": 0.0},
    )
    _mk_run(
        b,
        {"strategy": "baseline", "execution_sim": {"spread_bps": 2.0}},
        {"pnl_net_sum": 1.0, "fills_count": 8, "spread_cost_est_sum": 0.4},
    )
    report = compare_runs(a, b, top_k=10)
    assert isinstance(report["a_run_dir"], str)
    cfg_keys = [x["key"] for x in report["config_diff"]]
    assert "execution_sim.spread_bps" in cfg_keys
    md = {x["key"]: x for x in report["metrics_diff"]}
    assert "pnl_net_sum" in md
    assert abs(float(md["pnl_net_sum"]["delta"]) - (-0.5)) < 1e-9
    assert any("pnl_net_sum" in h for h in report["highlights"])
