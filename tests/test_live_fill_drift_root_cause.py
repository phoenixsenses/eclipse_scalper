from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.live_fill_drift_root_cause import rank_root_causes


def test_rank_root_causes_latency_and_queue() -> None:
    parity = {
        "sim_count": 120,
        "matched_count": 50,
        "match_rate_vs_sim": 0.42,
        "fill_rate_delta": 0.22,
        "mean_fill_delay_delta_sec": 2.3,
        "mean_adverse_bps_delta": 1.6,
    }
    diag = {
        "rows": 200,
        "latency_fill_delay_sec_p95": 12.5,
        "toxicity_score": 1.4,
    }
    tox = {"toxicity_score": 1.4}
    audit = {"overall_ok": False}

    causes = rank_root_causes(parity=parity, diag=diag, tox=tox, audit=audit)
    assert len(causes) >= 2
    names = [c.name for c in causes]
    assert "Latency Modeling Drift" in names
    assert "Queue/Hazard Miscalibration" in names


def test_rank_root_causes_insufficient_evidence() -> None:
    parity = {
        "sim_count": 10,
        "matched_count": 2,
        "match_rate_vs_sim": 0.2,
    }
    diag = {"rows": 5}
    tox = {}
    audit = {"overall_ok": False}

    causes = rank_root_causes(parity=parity, diag=diag, tox=tox, audit=audit)
    assert len(causes) >= 1
    assert any(c.name == "Insufficient/Noisy Evidence" for c in causes)
