"""Tests for s34_risk_alerter — observation-only, no live actions."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_risk_alerter import build_message


def _make_audit(margin_env: float = 29.8, budget: float = 0.3,
                worst_fill: float = -175.7, nominal: float = 150.0,
                atomic: str = "NOT_ATOMIC_ENTRY_THEN_STOP_AFTER_FILL_DETECTION",
                tail_rate: float = 18.7) -> dict:
    return {
        "live_env": {
            "margin_usdt": margin_env,
            "max_budget_margin_usdt": budget,
            "notional_usdt": margin_env * 40,
            "max_budget_notional_usdt": budget * 40,
            "leverage": 40.0,
        },
        "gap_through": {
            "current_stop_nominal_bps": nominal,
            "current_stop_research_max_loss_bps": worst_fill,
            "gap_plus_fee_bps": abs(worst_fill) - nominal,
        },
        "atomicity_audit": {
            "finding": atomic,
            "poll_sec": 2.0,
        },
        "tail_frequency": {
            "large_loss_rate": tail_rate,
            "probabilities": {
                "at_least_one_tail_in_3_trades": 46.3,
                "at_least_one_tail_in_5_trades": 64.6,
            },
        },
        "stop_budget_math": {"planned_stop_loss_pct_equity_research": 59.7},
        "protective_stop_research": {"best_t3r_variant": {"variant": "fixed_sl_150"}},
        "recommendations": ["Reduce margin."],
    }


def _write_json_tmp(data: dict) -> Path:
    """Write dict to a temp file, return its path."""
    td = tempfile.mkdtemp()
    p = Path(td) / "audit.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _run_evaluate(monkeypatch, audit_data: dict, live_data: dict | None = None,
                  kill_exists: bool = False, ledger_lines: list[str] | None = None):
    import tools.s34_risk_alerter as mod
    audit_path = _write_json_tmp(audit_data)
    live_path  = _write_json_tmp(live_data or {})
    monkeypatch.setattr(mod, "EXECUTION_AUDIT", audit_path)
    monkeypatch.setattr(mod, "LIVE_STATE", live_path)
    monkeypatch.setattr(mod, "SHADOW_LEDGER", Path("/nonexistent_ledger_xyz"))
    if kill_exists:
        kf = _write_json_tmp({})
        monkeypatch.setattr(mod, "KILL_SWITCH", kf)
    else:
        monkeypatch.setattr(mod, "KILL_SWITCH", Path("/nonexistent_kill_xyz"))
    from tools.s34_risk_alerter import evaluate_conditions
    return evaluate_conditions()


def test_oversize_condition_fires(monkeypatch):
    """99x oversize triggers OVERSIZE condition."""
    result = _run_evaluate(monkeypatch, _make_audit(margin_env=29.8, budget=0.3))
    assert result["conditions"]["OVERSIZE"] is True
    assert result["oversize_x"] > 10.0


def test_no_oversize_when_size_ok(monkeypatch):
    """1x oversize does not trigger OVERSIZE."""
    result = _run_evaluate(monkeypatch, _make_audit(margin_env=0.3, budget=0.3))
    assert result["conditions"]["OVERSIZE"] is False


def test_sl_gap_condition_fires(monkeypatch):
    """worst_fill -175.7bps > nominal 150bps triggers SL_GAP."""
    result = _run_evaluate(monkeypatch, _make_audit(worst_fill=-175.7, nominal=150.0))
    assert result["conditions"]["SL_GAP"] is True


def test_no_sl_gap_when_within_nominal(monkeypatch):
    """worst_fill -100bps < nominal 150bps does NOT trigger SL_GAP."""
    result = _run_evaluate(monkeypatch, _make_audit(worst_fill=-100.0, nominal=150.0))
    assert result["conditions"]["SL_GAP"] is False


def test_atomicity_condition_fires(monkeypatch):
    """NOT_ATOMIC finding triggers ATOMICITY."""
    result = _run_evaluate(monkeypatch, _make_audit(atomic="NOT_ATOMIC_ENTRY_THEN_STOP_AFTER_FILL_DETECTION"))
    assert result["conditions"]["ATOMICITY"] is True


def test_no_atomicity_when_atomic(monkeypatch):
    """ATOMIC finding does not trigger ATOMICITY."""
    result = _run_evaluate(monkeypatch, _make_audit(atomic="ATOMIC_BRACKET"))
    assert result["conditions"]["ATOMICITY"] is False


def test_kill_trip_condition_fires(monkeypatch):
    """KILL_SWITCH file presence triggers KILL_TRIP."""
    result = _run_evaluate(monkeypatch, _make_audit(), kill_exists=True)
    assert result["conditions"]["KILL_TRIP"] is True


def test_kill_trip_inactive_when_no_file(monkeypatch):
    """No KILL_SWITCH file means KILL_TRIP is False."""
    result = _run_evaluate(monkeypatch, _make_audit(), kill_exists=False)
    assert result["conditions"]["KILL_TRIP"] is False


def test_build_message_contains_triggered(monkeypatch):
    """build_message includes TRIGGERED section when conditions fire."""
    result = _run_evaluate(monkeypatch, _make_audit())
    msg = build_message(result["conditions"], result["details"], result["oversize_x"])
    assert "TRIGGERED" in msg
    assert "OVERSIZE" in msg
    assert "ATOMICITY" in msg


def test_build_message_all_clear():
    """build_message shows ALL_CLEAR when no conditions fire."""
    conds = {k: False for k in ("OVERSIZE", "ATOMICITY", "SL_GAP", "KILL_TRIP", "TAIL_EVENT")}
    details = {k: "ok" for k in conds}
    msg = build_message(conds, details, 0.0)
    assert "ALL_CLEAR" in msg
    assert "TRIGGERED" not in msg


def test_build_message_oversize_recommendation(monkeypatch):
    """build_message includes resize recommendation when OVERSIZE triggers."""
    result = _run_evaluate(monkeypatch, _make_audit(margin_env=29.8, budget=0.3))
    msg = build_message(result["conditions"], result["details"], result["oversize_x"])
    assert "MARGIN" in msg.upper() or "budget" in msg.lower()
