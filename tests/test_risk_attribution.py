from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.microphys.risk.attribution import build_risk_attribution
from tools import risk_attribution as ra


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"risk_attr_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_build_risk_attribution_basic() -> None:
    trades = pd.DataFrame(
        [
            {"side": "buy", "filled": True, "risk_reason": "OK", "pnl_net_notional": 1.0, "_entry_ts_ms": 1000},
            {"side": "sell", "filled": False, "risk_reason": "RISK_CAP_EXPOSURE", "pnl_net_notional": -0.5, "_entry_ts_ms": 1100},
        ]
    )
    gating = pd.DataFrame(
        [
            {"ts_ms": 1000, "active_expert_id": 2},
            {"ts_ms": 1100, "active_expert_id": -1},
        ]
    )
    out = build_risk_attribution(trades, gating_df=gating)
    assert set(out.keys()) == {"by_side", "by_fill", "by_reason", "by_expert"}
    assert int(out["by_side"]["count"].sum()) == 2
    assert int(out["by_expert"]["count"].sum()) == 2


def test_risk_attribution_tool_writes_outputs(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    cwd = Path.cwd()
    try:
        monkeypatch.chdir(tmp)
        live = tmp / "data" / "live"
        live.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"side": "buy", "filled": True, "risk_reason": "OK", "pnl_net_notional": 2.0, "_entry_ts_ms": 2000},
                {"side": "buy", "filled": True, "risk_reason": "OK", "pnl_net_notional": -1.0, "_entry_ts_ms": 2100},
            ]
        ).to_parquet(live / "papertrades_live.parquet", index=False)
        pd.DataFrame([{"ts_ms": 2000, "active_expert_id": 0}, {"ts_ms": 2100, "active_expert_id": 0}]).to_parquet(
            live / "gating_live.parquet", index=False
        )
        out_md = tmp / "reports" / "risk_attr.md"
        out_dir = tmp / "data" / "derived" / "risk_attribution"
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--live-root", str(live), "--out-md", str(out_md), "--out-dir", str(out_dir)],
        )
        assert ra.main() == 0
        assert out_md.exists()
        assert (out_dir / "by_side.parquet").exists()
        assert (out_dir / "by_expert.parquet").exists()
    finally:
        monkeypatch.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

