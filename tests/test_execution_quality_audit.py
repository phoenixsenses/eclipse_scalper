from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from tools import execution_quality_audit as eqa


def _mk_local_tmp() -> Path:
    p = Path("localtests") / f"exec_quality_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def test_execution_quality_audit_skip_missing(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        out_md = tmp / "out.md"
        out_json = tmp / "out.json"
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--in-parquet", str(tmp / "missing.parquet"), "--out-md", str(out_md), "--out-json", str(out_json)],
        )
        assert eqa.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "skip"
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_execution_quality_audit_builds_metrics(monkeypatch) -> None:
    tmp = _mk_local_tmp()
    try:
        inp = tmp / "papertrades.parquet"
        out_md = tmp / "audit.md"
        out_json = tmp / "audit.json"
        df = pd.DataFrame(
            [
                {
                    "entry_ts_utc": "2026-03-04T10:00:00Z",
                    "side": "buy",
                    "execution_model": "simple",
                    "risk_reason": "OK",
                    "filled": True,
                    "ttl_expired": False,
                    "fill_delay_bars": 0,
                    "entry_price": 100.0,
                    "fill_price": 100.01,
                    "pnl_net_notional": 1.2,
                    "pnl_gross_notional": 1.4,
                    "fee_notional": 0.2,
                },
                {
                    "entry_ts_utc": "2026-03-04T10:00:01Z",
                    "side": "sell",
                    "execution_model": "maker_hazard",
                    "risk_reason": "RISK_CAP_EXPOSURE",
                    "filled": False,
                    "ttl_expired": True,
                    "fill_delay_bars": 5,
                    "entry_price": 99.5,
                    "fill_price": 99.5,
                    "pnl_net_notional": -0.3,
                    "pnl_gross_notional": -0.2,
                    "fee_notional": 0.1,
                },
            ]
        )
        df.to_parquet(inp, index=False)
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--in-parquet", str(inp), "--out-md", str(out_md), "--out-json", str(out_json), "--last-n", "100"],
        )
        assert eqa.main() == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["status"] == "ok"
        assert int(payload["rows"]) == 2
        assert "buy" in payload["by_side"]
        assert "maker_hazard" in payload["by_execution_model"]
        assert out_md.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

