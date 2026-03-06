from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import validate_micro_edge_forward as mod


def _mk_jsonl_path(tag: str) -> Path:
    root = Path("localtests") / "micro_edge_forward"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{tag}_{uuid.uuid4().hex[:8]}.jsonl"


def test_forward_validation_flags_collapse(capsys: pytest.CaptureFixture[str]):
    path = _mk_jsonl_path("collapse")

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
    try:
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
    finally:
        path.unlink(missing_ok=True)


def test_forward_validation_relax_min_n(capsys: pytest.CaptureFixture[str]):
    path = _mk_jsonl_path("relax")
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
    try:
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
    finally:
        path.unlink(missing_ok=True)


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


def test_forward_validation_writes_json_with_liquidation_impact():
    root = Path("localtests") / "micro_edge_forward"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"forward_liq_{uuid.uuid4().hex[:8]}.jsonl"
    out_json = root / f"forward_liq_report_{uuid.uuid4().hex[:8]}.json"
    rows = []
    for i in range(8):
        rows.append(
            {
                "ts_bucket": i + 1,
                "gross_ret": 0.0020,
                "cost": 0.0010,
                "net_ret": 0.0010 if i < 4 else 0.0003,
                "exec_model": "taker",
                "horizon_sec": 30,
                "regime_spread_bin": "<=p25",
                "regime_intensity_bin": ">p75",
                "regime_vol_bin": "p25-50",
                "regime_imb_bin": "+[0.5,0.7)",
                "v2_liq_reversal_signal": 0.9 if i in (1, 2, 5, 6) else 0.0,
                "liq_rate_per_sec": 12.0 if i in (1, 2, 5, 6) else 0.0,
                "liq_imbalance": 0.8 if i in (1, 2, 5, 6) else 0.0,
            }
        )
    try:
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        code = mod.main(
            [
                "--debug",
                str(path),
                "--discover-frac",
                "0.5",
                "--top-k",
                "1",
                "--min-n",
                "2",
                "--out-json",
                str(out_json),
            ]
        )
        assert code == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["run_summary"]["run_type"] == "validate_micro_edge_forward"
        assert payload["liquidation_impact"]["discovery"]["available"] is True
        assert payload["liquidation_impact"]["validation"]["available"] is True
        assert payload["liquidation_regime_tag_impact"]["discovery"]["available"] is True
        assert "tagged" in payload["liquidation_regime_tag_impact"]["validation"]
    finally:
        path.unlink(missing_ok=True)
        out_json.unlink(missing_ok=True)
