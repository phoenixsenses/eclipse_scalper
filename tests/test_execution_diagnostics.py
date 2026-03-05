from __future__ import annotations

import pandas as pd

from tools.execution_diagnostics import compute_execution_diagnostics
from tools.post_rollout_audit import build_audit
from tools.toxicity_report import build_toxicity_report


def test_execution_diagnostics_core_metrics() -> None:
    df = pd.DataFrame(
        [
            {"filled": 1, "fill_delay_sec": 2.0, "pnl_bps": 0.8, "max_adverse_bps": 1.2, "side": "buy"},
            {"filled": 0, "fill_delay_sec": 10.0, "pnl_bps": -0.2, "max_adverse_bps": 2.0, "side": "sell"},
            {"filled": 1, "fill_delay_sec": 4.0, "pnl_bps": 0.4, "max_adverse_bps": 0.9, "side": "buy"},
        ]
    )
    d = compute_execution_diagnostics(df)
    assert int(d["rows"]) == 3
    assert 0.0 <= float(d["fill_rate"]) <= 1.0
    assert float(d["latency_fill_delay_sec_p95"]) >= float(d["latency_fill_delay_sec_p50"])


def test_toxicity_and_post_rollout_audit() -> None:
    df = pd.DataFrame(
        [
            {"side": "buy", "pnl_bps": 0.5, "max_adverse_bps": 1.0},
            {"side": "sell", "pnl_bps": -0.1, "max_adverse_bps": 2.0},
        ]
    )
    tox = build_toxicity_report(df)
    assert int(tox["rows"]) == 2
    assert "buy" in tox["sides"]
    diag = {"rows": 2, "fill_rate": 0.3, "latency_fill_delay_sec_p95": 5.0}
    audit = build_audit(diag, tox)
    assert "flags" in audit and "checks" in audit
    assert isinstance(bool(audit["overall_ok"]), bool)

