"""Tests for tools/analyze_cost_breakdown.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tools.analyze_cost_breakdown import _analyze_pocket, _parse_eval_key, _fit_linear, main


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def _make_pocket(npa_map: dict) -> dict:
    """Build a minimal pocket dict with the given (fee, adv) → npa mapping."""
    evals = {}
    for (fee, adv), npa in npa_map.items():
        evals[f"fee={fee:.3f}|adv={adv:.3f}"] = {"median_net_per_attempt": npa}
    return {
        "symbol": "ETHUSDT",
        "horizon_sec": 120,
        "min_imbalance": 0.5,
        "min_trade_intensity": 2500.0,
        "max_spread": 0.0003,
        "rule": "micro_edge_v3_passive_alpha",
        "evals": evals,
    }


# ─────────────────────────────────────────────
# _parse_eval_key
# ─────────────────────────────────────────────

def test_parse_eval_key_valid():
    assert _parse_eval_key("fee=1.000|adv=1.200") == (1.0, 1.2)
    assert _parse_eval_key("fee=0.500|adv=0.800") == (0.5, 0.8)


def test_parse_eval_key_invalid():
    assert _parse_eval_key("bad_key") is None
    assert _parse_eval_key("") is None


# ─────────────────────────────────────────────
# _fit_linear
# ─────────────────────────────────────────────

def test_fit_linear_recovers_known_model():
    # NPA = 0.0005 - 0.0001*fee - 0.0002*adv
    b0, b1, b2 = 0.0005, -0.0001, -0.0002
    pts = [
        (f, a, b0 + b1 * f + b2 * a)
        for f in (0.5, 1.0, 1.5)
        for a in (1.0, 1.2)
    ]
    fit, quality = _fit_linear(pts)
    assert fit is not None
    assert quality == "ok"
    assert abs(fit[0] - b0) < 1e-8
    assert abs(fit[1] - b1) < 1e-8
    assert abs(fit[2] - b2) < 1e-8


def test_fit_linear_too_few_points():
    fit, q = _fit_linear([(1.0, 1.0, 0.0), (1.5, 1.0, -0.1)])
    assert fit is None
    assert "too_few" in q


# ─────────────────────────────────────────────
# _analyze_pocket — schema keys
# ─────────────────────────────────────────────

def test_analyze_pocket_schema_keys():
    pocket = _make_pocket({
        (0.5, 1.0): 0.00005,
        (1.0, 1.0): -0.00001,
        (1.5, 1.0): -0.00007,
        (0.5, 1.2): -0.00002,
        (1.0, 1.2): -0.00008,
        (1.5, 1.2): -0.00014,
    })
    result = _analyze_pocket(pocket)
    required = [
        "symbol", "horizon_sec", "raw_edge_proxy_bps",
        "fee_sensitivity", "adv_sensitivity",
        "break_even_fee_bps", "break_even_adv_mult",
        "gap_to_breakeven_fee_bps", "gap_to_breakeven_adv_mult",
        "dominator", "current_npa_bps", "per_eval",
    ]
    for k in required:
        assert k in result, f"Missing key: {k}"


def test_analyze_pocket_no_evals():
    pocket = {"symbol": "ETHUSDT", "evals": {}}
    result = _analyze_pocket(pocket)
    assert result.get("error") == "no_evals"
    assert result["per_eval"] == []


# ─────────────────────────────────────────────
# Break-even math validation
# ─────────────────────────────────────────────

def test_break_even_fee_correct():
    """Verify break_even_fee_bps satisfies NPA=0 in the linear model."""
    # Model: NPA = 0.00060 - 0.00020*fee - 0.00010*adv
    # At adv=1.0: 0 = 0.00060 - 0.00020*be_fee - 0.00010
    #             be_fee = (0.00060 - 0.00010) / 0.00020 = 2.50
    pocket = _make_pocket({
        (f, a): 0.00060 - 0.00020 * f - 0.00010 * a
        for f in (0.5, 1.0, 2.0, 2.5)
        for a in (0.8, 1.0, 1.2)
    })
    result = _analyze_pocket(pocket)
    be = result.get("break_even_fee_bps")
    assert be is not None
    assert abs(be - 2.50) < 0.05, f"Expected ~2.50, got {be}"


def test_break_even_adv_correct():
    """Verify break_even_adv_mult satisfies NPA=0."""
    # Model: NPA = 0.00030 - 0.00010*fee - 0.00020*adv
    # At fee=1.0: 0 = 0.00030 - 0.00010 - 0.00020*be_adv
    #             be_adv = (0.00030 - 0.00010) / 0.00020 = 1.0
    pocket = _make_pocket({
        (f, a): 0.00030 - 0.00010 * f - 0.00020 * a
        for f in (0.5, 1.0, 1.5)
        for a in (0.5, 1.0, 1.5)
    })
    result = _analyze_pocket(pocket)
    be = result.get("break_even_adv_mult")
    assert be is not None
    assert abs(be - 1.0) < 0.05, f"Expected ~1.0, got {be}"


def test_dominator_fee():
    # b1*fee >> b2*adv → fee dominates
    pocket = _make_pocket({
        (f, a): 0.001 - 0.002 * f - 0.0001 * a
        for f in (0.5, 1.0, 2.0)
        for a in (1.0, 1.2)
    })
    result = _analyze_pocket(pocket)
    assert result["dominator"] == "fee"


def test_dominator_adverse():
    # b2*adv >> b1*fee → adverse dominates
    pocket = _make_pocket({
        (f, a): 0.001 - 0.0001 * f - 0.002 * a
        for f in (0.5, 1.0, 1.5)
        for a in (0.5, 1.0, 2.0)
    })
    result = _analyze_pocket(pocket)
    assert result["dominator"] == "adverse"


# ─────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────

def test_main_smoke():
    """Smoke: run main() on a small fixture JSON → outputs written, schema correct."""
    pocket = _make_pocket({
        (0.5, 1.0): 0.00005,
        (1.0, 1.0): -0.00001,
        (1.5, 1.0): -0.00007,
        (0.5, 1.2): -0.00002,
        (1.0, 1.2): -0.00008,
        (1.5, 1.2): -0.00014,
    })
    rank_data = {"count": 1, "ranking": [pocket]}

    tmp_path = Path("localtests") / f"cost_breakdown_{uuid.uuid4().hex[:8]}"
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        rank_json = tmp_path / "rank.json"
        rank_json.write_text(json.dumps(rank_data), encoding="utf-8")

        out_json = tmp_path / "cost.json"
        out_md = tmp_path / "cost.md"

        sys.argv = [
            "analyze_cost_breakdown",
            "--rank-json", str(rank_json),
            "--out-json", str(out_json),
            "--out-md", str(out_md),
        ]
        rc = main()
        assert rc == 0

        result = json.loads(out_json.read_text(encoding="utf-8"))
        assert result["tool"] == "analyze_cost_breakdown"
        assert result["n_pockets"] == 1
        assert result["run_summary"]["run_type"] == "analyze_cost_breakdown"
        assert result["run_summary"]["metrics"]["n_pockets"] == 1
        assert "pockets" in result
        assert "closest_to_breakeven_fee" in result
        assert "closest_to_breakeven_adv" in result

        p = result["pockets"][0]
        assert p["symbol"] == "ETHUSDT"
        assert "break_even_fee_bps" in p
        assert "break_even_adv_mult" in p

        md = out_md.read_text(encoding="utf-8")
        assert "COST BREAKDOWN" in md
        assert "dominator" in md
    finally:
        for f in (tmp_path / "rank.json", tmp_path / "cost.json", tmp_path / "cost.md"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass
        try:
            tmp_path.rmdir()
        except Exception:
            pass


def test_main_missing_file():
    tmp = Path("localtests") / f"cost_breakdown_missing_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = [
            "analyze_cost_breakdown",
            "--rank-json", str(tmp / "nonexistent.json"),
        ]
        rc = main()
        assert rc == 2
    finally:
        try:
            tmp.rmdir()
        except Exception:
            pass
