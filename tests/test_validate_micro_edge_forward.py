from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_micro_edge_forward as mod


def test_forward_validation_flags_collapse(capsys: pytest.CaptureFixture[str]):
    root = Path(__file__).resolve().parents[1] / "logs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test_validate_micro_edge_forward.jsonl"

    rows = []
    # Discovery (first 6): positive net for one regime pocket.
    for i in range(6):
        rows.append(
            {
                "ts_bucket": i + 1,
                "gross_ret": 0.002,
                "cost": 0.001,
                "net_ret": 0.001,
                "exec_model": "taker",
                "horizon_sec": 30,
                "regime_spread_bin": "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": "p25-50",
                "regime_imb_bin": "+[0.5,0.7)",
            }
        )
    # Validation (last 4): same pocket collapses.
    for i in range(4):
        rows.append(
            {
                "ts_bucket": 100 + i,
                "gross_ret": 0.0002,
                "cost": 0.001,
                "net_ret": -0.0008,
                "exec_model": "taker",
                "horizon_sec": 30,
                "regime_spread_bin": "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": "p25-50",
                "regime_imb_bin": "+[0.5,0.7)",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    code = mod.main(
        [
            "--debug",
            str(path),
            "--discover-frac",
            "0.6",
            "--top-k",
            "1",
            "--min-n",
            "2",
            "--min-select-frac",
            "0.01",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "FLAG validation_collapse_detected" in out


def test_forward_validation_relax_min_n(capsys: pytest.CaptureFixture[str]):
    root = Path(__file__).resolve().parents[1] / "logs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "test_validate_micro_edge_forward_relax.jsonl"
    rows = []
    # 30 rows total, one regime group -> strict min-n=50 should relax to floor=20.
    for i in range(30):
        rows.append(
            {
                "ts_bucket": i + 1,
                "gross_ret": 0.0015,
                "cost": 0.0010,
                "net_ret": 0.0005 if i < 18 else -0.0002,
                "exec_model": "taker",
                "horizon_sec": 60,
                "regime_spread_bin": "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": "p25-50",
                "regime_imb_bin": "+[0.5,0.7)",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    code = mod.main(
        [
            "--debug",
            str(path),
            "--discover-frac",
            "0.6",
            "--top-k",
            "1",
            "--min-n-discovery",
            "50",
            "--min-n-validation",
            "50",
            "--relax-floor",
            "20",
            "--min-select-frac",
            "0.01",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "RELAX min_n_discovery from 50 to 20" in out
    assert "RELAX min_n_validation from 50 to 20" in out


def test_collapse_balanced_no_flip_no_dual_drop():
    collapse, flags, _ = mod.evaluate_collapse_flags(
        disc_sm={"avg_net": 0.00040, "p90_net": 0.00060, "n": 100},
        valid_sm={"avg_net": 0.00031, "p90_net": 0.00051, "n": 90},
        disc_frac=0.20,
        valid_frac=0.15,
        avg_eps=0.0002,
        p90_eps=0.0002,
        select_ratio=0.5,
        n_ratio=0.5,
        mode="balanced",
    )
    assert flags["p90_sign_flip"] is False
    assert collapse is False


def test_collapse_balanced_on_p90_sign_flip():
    collapse, flags, _ = mod.evaluate_collapse_flags(
        disc_sm={"avg_net": 0.00040, "p90_net": 0.00030, "n": 100},
        valid_sm={"avg_net": 0.00010, "p90_net": -0.00001, "n": 100},
        disc_frac=0.20,
        valid_frac=0.20,
        avg_eps=0.0002,
        p90_eps=0.0002,
        select_ratio=0.5,
        n_ratio=0.5,
        mode="balanced",
    )
    assert flags["p90_sign_flip"] is True
    assert collapse is True


def test_collapse_balanced_on_avg_and_p90_drop():
    collapse, flags, _ = mod.evaluate_collapse_flags(
        disc_sm={"avg_net": 0.00050, "p90_net": 0.00060, "n": 100},
        valid_sm={"avg_net": 0.00020, "p90_net": 0.00030, "n": 90},
        disc_frac=0.20,
        valid_frac=0.16,
        avg_eps=0.0002,
        p90_eps=0.0002,
        select_ratio=0.5,
        n_ratio=0.5,
        mode="balanced",
    )
    assert flags["avg_net_drop"] is True
    assert flags["p90_drop"] is True
    assert collapse is True
