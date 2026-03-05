from __future__ import annotations

import json
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compare_rank_runs as crr


def _mk_row(symbol: str, npa: float, pass_rate: float, reason: str, **kw):
    row = {
        "symbol": symbol,
        "horizon_sec": 120,
        "min_imbalance": 0.5,
        "min_trade_intensity": 2500.0,
        "max_spread": 0.0003,
        "npa_core": npa,
        "pass_rate_core": pass_rate,
        "failure_reason_top": reason,
        "best_fee_survive": 1.0,
        "gate_reject_ratio": 0.1,
        "fill_rate_after_gate": 0.5,
        "avg_fee_bps": 1.0,
        "avg_adverse_bps_on_fills": 1.2,
        "avg_net_return_bps_on_fills": -0.2,
    }
    row.update(kw)
    return row


def test_compare_rank_runs_writes_md_and_buy_sell_delta(monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    buy = reports / f"test_cmp_{suffix}_RANK_EDGEONLY_FEE0_BUY_21D.json"
    sell = reports / f"test_cmp_{suffix}_RANK_EDGEONLY_FEE0_SELL_21D.json"
    auto = reports / f"test_cmp_{suffix}_RANK_EDGEONLY_FEE0_AUTO_21D.json"
    out_md = reports / f"test_cmp_{suffix}_COMPARE_RANK_RUNS.md"

    buy.write_text(
        json.dumps(
            {
                "ranking": [
                    _mk_row("ETHUSDT", 0.00010, 0.60, "fees_dominate"),
                    _mk_row("BTCUSDT", 0.00005, 0.55, "fees_dominate"),
                ]
            }
        ),
        encoding="utf-8",
    )
    sell.write_text(
        json.dumps(
            {
                "ranking": [
                    _mk_row("ETHUSDT", -0.00005, 0.40, "adverse_dominates"),
                    _mk_row("BTCUSDT", -0.00002, 0.45, "adverse_dominates"),
                ]
            }
        ),
        encoding="utf-8",
    )
    auto.write_text(
        json.dumps({"ranking": [_mk_row("ETHUSDT", 0.0, 0.5, "mixed")]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--ins",
            f"{buy},{sell},{auto}",
            "--top-n",
            "2",
            "--out-md",
            str(out_md),
        ],
    )
    rc = crr.main()
    assert rc == 0
    text = out_md.read_text(encoding="utf-8")
    assert f"## {buy.name}" in text
    assert f"## {sell.name}" in text
    assert "Cross-Run Diagnosis" in text
    assert "BUY/SELL delta" in text
    assert "delta_npa_core" in text
    assert "failure_reason_top" in text
    assert "gate_reject_ratio" in text


def test_compare_rank_runs_missing_file_returns_2(monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    out_md = reports / f"test_cmp_{suffix}_COMPARE_RANK_RUNS.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "--ins",
            str(reports / f"test_cmp_{suffix}_missing.json"),
            "--out-md",
            str(out_md),
        ],
    )
    rc = crr.main()
    assert rc == 2


def test_compare_rank_runs_intersect_only_subset(monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    a = reports / f"test_cmp_{suffix}_A.json"
    b = reports / f"test_cmp_{suffix}_B.json"
    out_md = reports / f"test_cmp_{suffix}_INTERSECT.md"

    common = _mk_row("ETHUSDT", 0.00010, 0.60, "fees_dominate", horizon_sec=120)
    only_a = _mk_row("BTCUSDT", 0.00005, 0.55, "mixed", horizon_sec=60)
    only_b = _mk_row("BTCUSDT", 0.00004, 0.50, "mixed", horizon_sec=120)
    a.write_text(json.dumps({"ranking": [common, only_a]}), encoding="utf-8")
    b.write_text(json.dumps({"ranking": [common, only_b]}), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--ins", f"{a},{b}", "--top-n", "5", "--intersect-only", "--out-md", str(out_md)],
    )
    rc = crr.main()
    assert rc == 0
    text = out_md.read_text(encoding="utf-8")
    assert "intersect_only=true intersection_count=1" in text
    assert "ETHUSDT" in text
    # non-intersection pockets should be gone
    assert text.count("BTCUSDT") == 0


def test_compare_rank_runs_intersect_only_empty_returns_2(monkeypatch):
    suffix = uuid.uuid4().hex[:8]
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    a = reports / f"test_cmp_{suffix}_A_empty.json"
    b = reports / f"test_cmp_{suffix}_B_empty.json"
    out_md = reports / f"test_cmp_{suffix}_INTERSECT_EMPTY.md"
    a.write_text(json.dumps({"ranking": [_mk_row("ETHUSDT", 0.00010, 0.60, "fees_dominate", horizon_sec=120)]}), encoding="utf-8")
    b.write_text(json.dumps({"ranking": [_mk_row("BTCUSDT", 0.00010, 0.60, "fees_dominate", horizon_sec=120)]}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["x", "--ins", f"{a},{b}", "--intersect-only", "--out-md", str(out_md)],
    )
    rc = crr.main()
    assert rc == 2
