"""S34 Cascade Navigation Dashboard v1.

A point-in-time NAVIGATION / PERMISSION layer for the S34 liquidation-cascade
family. This is NOT a trade trigger. It answers: given everything knowable at
`as_of_ts` (and nothing after it), what is the current cascade state, is the
regime/execution supportive, and does the family have *proven* edge -- so a
human/operator can decide whether to even look for an entry.

Design contract:
- Every feature is computed from rows with ts <= as_of_ts only. The partial
  cascade snapshot is run through `assert_feature_set_available`, so any
  accidental lookahead raises instead of silently leaking (the exact failure
  mode that contaminated the old shadow-paper evidence).
- "Cascade momentum" (is the cascade physically still growing?) is reported
  separately from "rule edge" (does this family earn money in the clean,
  holdout-validated recheck?). Strong momentum with no proven edge => no
  permission. That separation is the whole point.

Panels:
  1. Anchor Health     - running notional, distance to thresholds, elapsed, rate, accel, knowable flag
  2. Cascade Phase     - forming/accelerating/decelerating/exhaustion/dead + shape + momentum/fade scores
  3. Regime Compass    - BTC alignment, day trend, session, day range (vol), funding
  4. Execution Reality - book staleness, spread, expected entry adverse, no-fill risk
  5. Rule Confidence   - clean recheck verdict, calibration/holdout median/mean/T3R, evidence quality
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    MarkIndex,
    book_at,
    iso_ms,
    load_liquidations,
    r1,
)
from tools.s34_feature_availability import (
    FeatureClass,
    FeatureValue,
    assert_feature_set_available,
)
from ami.storage.union_reader import open_union_ro, as_of_row  # noqa: E402

DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
OUT_JSON = OUT_DIR / "S34_CASCADE_NAVIGATION_DASHBOARD.json"
OUT_MD = OUT_DIR / "S34_CASCADE_NAVIGATION_DASHBOARD.md"
MANAGEMENT_FRAGMENT = OUT_DIR / "S34_CASCADE_NAVIGATION_MANAGEMENT_FRAGMENT.json"
V9_RISK_REVIEW = OUT_DIR / "S34_V9_KILL_FORWARD_REVIEW.json"
V11_GOVERNANCE = OUT_DIR / "S34_V11_FORWARD_GOVERNANCE.json"
V02_SHADOW_MIRROR_BRIEF = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
V02_H4_SHADOW_FRAGMENT = OUT_DIR / "S34_V02_H4_SHADOW_DASHBOARD_FRAGMENT.json"
SIZING_SHADOW_PAPER = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER.json"
EXECUTION_AUDIT = OUT_DIR / "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.json"
RECHECK_GLOB = "S34_KNOWABLE_ANCHOR_ROUTE_RECHECK*.json"
VALIDATED_SIGNALS_REGISTRY = OUT_DIR / "S34_VALIDATED_SIGNALS_REGISTRY.json"
REALTIME_SHADOW_STATE = ROOT / "reports" / "shadow" / "s34_state_machine_shadow_state.json"
REALTIME_SHADOW_LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
LIVE_EXECUTOR_STATE = ROOT / "runtime" / "s34_v_engine_live_state.json"
ENV_PATH = ROOT / ".env"

THRESHOLDS_USD = (50_000.0, 100_000.0, 200_000.0, 500_000.0, 1_000_000.0)
EXECMGMT_DEFAULT_EQUITY_USDT = 35.0
EXECMGMT_TAIL_BUDGET_PCT = 1.0
EXECMGMT_WORST_REAL_FILL_BPS = -175.7
EXECMGMT_DEFAULT_LEVERAGE = 40.0


@dataclass(frozen=True)
class Lane:
    symbol: str
    liq_side: str
    direction: str
    family: str


UNIVERSE: tuple[Lane, ...] = (
    Lane("ETHUSDT", "BUY", "LONG", "ETH_BUY"),
    Lane("ETHUSDT", "SELL", "SHORT", "ETH_SELL"),
    Lane("SOLUSDT", "BUY", "LONG", "SOL_BUY"),
    Lane("SOLUSDT", "SELL", "SHORT", "SOL_SELL"),
    Lane("BTCUSDT", "BUY", "LONG", "BTC_BUY_DISTRIBUTED"),
    Lane("BTCUSDT", "SELL", "SHORT", "BTC_SELL"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_label(ts_ms: int) -> str:
    hour = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).hour
    if 0 <= hour < 7:
        return "ASIA"
    if 7 <= hour < 13:
        return "EUROPE"
    if 13 <= hour < 21:
        return "US"
    return "OFF_HOURS"


def load_mark_index_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> MarkIndex:
    """Bounded mark loader -- avoids pulling full per-symbol history from the large DB."""
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    return MarkIndex(rows)


def prior_return_bps(marks: MarkIndex, ts_ms: int, window_sec: int) -> float | None:
    return marks.ret_bps(int(ts_ms) - int(window_sec) * 1000, int(ts_ms))


def latest_liq_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute(
        "SELECT MAX(ts_ms) FROM liquidations WHERE symbol=?", (symbol,)
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def day_start_ms(ts_ms: int) -> int:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
    start = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    return int(start.timestamp() * 1000)


# Funding prints roughly every 8h, so nothing within a day means the source is dead,
# not that funding is unchanged. Bounded for the same reason section 189 bounded the
# monitor's copy: an unbounded as-of silently returns a months-old value as current.
FUNDING_LOOKBACK_MS = 24 * 3_600_000


def funding_rate_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    """Funding at-or-before `ts_ms`, or None if the source has gone quiet.

    Uses `as_of_row` rather than a plain `ORDER BY ts_ms DESC LIMIT 1`. Through the
    union view that form is the failure the union_reader docstring measures at 7560x:
    SQLite can only merge an ordered compound when the ORDER BY term is a RESULT
    column, and this query selects only `funding_rate`, so the whole compound gets
    materialised and sorted to return one row. The section 258/259 sweep fixed the
    shadow runner's copy of this probe and missed this one.
    """
    try:
        row = as_of_row(
            conn, "mark_prices", "funding_rate",
            where="symbol=? AND ts_ms<=? AND ts_ms>=? AND funding_rate IS NOT NULL",
            params=(symbol, int(ts_ms), int(ts_ms) - FUNDING_LOOKBACK_MS),
        )
    except (sqlite3.OperationalError, ValueError):
        return None
    return float(row[1]) if row and row[1] is not None else None


def load_management_fragment(path: Path = MANAGEMENT_FRAGMENT) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "MISSING",
            "permission": "UNKNOWN",
            "recommendation": "run tools/s34_v_engine_forward_management_monitor.py",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}


def load_v9_review(path: Path = V9_RISK_REVIEW) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    kill = payload.get("kill_criteria_redesign") or {}
    tick = payload.get("tick_level_atomicity_scan") or {}
    policy = payload.get("forward_review_decision_gate") or {}
    rec = kill.get("recommended_kill_rule") or {}
    return {
        "status": "ACTIVE",
        "recommended_kill": rec.get("name"),
        "kill_logic": rec.get("logic"),
        "tick_atomicity_status": tick.get("status"),
        "tick_worst_adverse_bps": tick.get("worst_tick_adverse_bps"),
        "forward_gate": policy.get("status"),
    }


def load_v11_governance(path: Path = V11_GOVERNANCE) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    integ = payload.get("forward_sample_integrity") or {}
    gov = payload.get("operator_governance") or {}
    truth = payload.get("execution_truth_ledger") or {}
    fee = payload.get("fee_tier_verification") or {}
    return {
        "status": "ACTIVE",
        "forward_integrity": integ.get("status"),
        "valid_forward_n": integ.get("valid_forward_n"),
        "operator_governance": gov.get("status"),
        "oversize_vs_balanced": gov.get("oversize_vs_balanced"),
        "execution_truth": truth.get("status"),
        "fee_tier": fee.get("status"),
    }


def load_v02_shadow_mirror(path: Path = V02_SHADOW_MIRROR_BRIEF) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    protocol = payload.get("protocol") or {}
    ledger = payload.get("ledger") or {}
    overall = payload.get("overall") or {}
    recent = payload.get("recent") or {}
    overall_summary = overall.get("summary") or {}
    recent_summary = recent.get("summary") or {}
    latest = payload.get("latest_observations") or []
    latest_signal = None
    if latest:
        latest_signal = (latest[-1] or {}).get("signal_utc")
    return {
        "status": "ACTIVE",
        "protocol_id": protocol.get("id"),
        "permission": protocol.get("permission"),
        "decision": protocol.get("decision"),
        "live_rule_match": protocol.get("live_rule_match"),
        "ledger_rows": ledger.get("rows_total"),
        "rows_added_this_run": ledger.get("rows_added_this_run"),
        "overall_n": overall_summary.get("n"),
        "overall_sum_bps": overall_summary.get("sum_bps"),
        "overall_median_bps": overall_summary.get("median_bps"),
        "overall_t3r_bps": overall_summary.get("top3_winner_removed_sum_bps"),
        "recent_n": recent_summary.get("n"),
        "recent_sum_bps": recent_summary.get("sum_bps"),
        "recent_median_bps": recent_summary.get("median_bps"),
        "recent_t3r_bps": recent_summary.get("top3_winner_removed_sum_bps"),
        "kill_triggered": (recent.get("kill_check") or {}).get("triggered"),
        "latest_signal_utc": latest_signal,
    }


def load_v02_h4_shadow(path: Path = V02_H4_SHADOW_FRAGMENT) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    return {
        "status": payload.get("status") or "ACTIVE",
        "protocol_id": payload.get("protocol_id"),
        "h2_sum_bps": payload.get("h2_sum_bps"),
        "h4_sum_bps": payload.get("h4_sum_bps"),
        "h4_t3r_bps": payload.get("h4_t3r_bps"),
        "cross_policy_sum_bps": payload.get("cross_policy_sum_bps"),
        "sl150_touch_count": payload.get("sl150_touch_count"),
        "queue_status": payload.get("queue_status"),
        "decision": payload.get("decision"),
    }


def load_sizing_shadow_paper(path: Path = SIZING_SHADOW_PAPER) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    modes = payload.get("modes") or {}
    return {
        "status": payload.get("status") or "ACTIVE",
        "rule_id": payload.get("rule_id"),
        "source_shadow_rows": payload.get("source_shadow_rows"),
        "current_env": modes.get("CURRENT_ENV") or {},
        "balanced": modes.get("BALANCED") or {},
        "survival": modes.get("SURVIVAL") or {},
        "stop_assisted": modes.get("STOP_ASSISTED") or {},
    }


def load_execution_audit(path: Path = EXECUTION_AUDIT) -> dict[str, Any]:
    """Load execution-management audit: sizing, stop, atomicity, tail frequency."""
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    env = payload.get("live_env") or {}
    gap = payload.get("gap_through") or {}
    atom = payload.get("atomicity_audit") or {}
    tail = payload.get("tail_frequency") or {}
    sb = payload.get("stop_budget_math") or {}
    ps = payload.get("protective_stop_research") or {}
    recs = payload.get("recommendations") or []
    margin_env = float(env.get("margin_usdt") or 0.0)
    budget_margin = float(env.get("max_budget_margin_usdt") or 0.0)
    oversize = round(margin_env / budget_margin, 1) if budget_margin > 0 else None
    return {
        "status": "ACTIVE" if env else "MISSING",
        "margin_env_usdt": margin_env,
        "notional_env_usdt": float(env.get("notional_usdt") or 0.0),
        "budget_margin_usdt": budget_margin,
        "budget_notional_usdt": float(env.get("max_budget_notional_usdt") or 0.0),
        "oversize_multiple": oversize,
        "leverage": float(env.get("leverage") or 40.0),
        "stop_nominal_bps": float(gap.get("current_stop_nominal_bps") or 150.0),
        "stop_worst_fill_bps": float(gap.get("current_stop_research_max_loss_bps") or -175.7),
        "stop_gap_bps": float(gap.get("gap_plus_fee_bps") or 0.0),
        "stop_protection": "PARTIAL_GAP_THROUGH",
        "stop_pct_equity": float(sb.get("planned_stop_loss_pct_equity_research") or 0.0),
        "atomicity": str(atom.get("finding") or "UNKNOWN"),
        "poll_sec": float(atom.get("poll_sec") or 2.0),
        "tail_rate_pct": float(tail.get("large_loss_rate") or 0.0),
        "tail_in_3_pct": float((tail.get("probabilities") or {}).get("at_least_one_tail_in_3_trades") or 0.0),
        "tail_in_5_pct": float((tail.get("probabilities") or {}).get("at_least_one_tail_in_5_trades") or 0.0),
        "best_stop_variant": (ps.get("best_t3r_variant") or {}).get("variant"),
        "recommendations": recs,
    }


def load_env_values(path: Path = ENV_PATH) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _float_from_any(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_float(values: dict[str, Any], keys: tuple[str, ...], default: float | None = None) -> tuple[float | None, str | None]:
    for key in keys:
        val = _float_from_any(values.get(key))
        if val is not None:
            return val, key
    return default, None


def load_execmgmt_panel(
    *,
    env_path: Path = ENV_PATH,
    sizing_path: Path = SIZING_SHADOW_PAPER,
    audit_path: Path = EXECUTION_AUDIT,
) -> dict[str, Any]:
    """Read-only sizing recommendation panel. It never mutates env/config/order state."""
    env = load_env_values(env_path)
    sizing = {}
    audit = {}
    if sizing_path.exists():
        try:
            sizing = json.loads(sizing_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sizing = {}
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit = {}

    audit_env = audit.get("live_env") if isinstance(audit.get("live_env"), dict) else {}
    leverage, leverage_source = _first_float(
        env,
        ("S34_LIVE_MAX_LEVERAGE", "LEVERAGE"),
        _float_from_any(audit_env.get("leverage"), EXECMGMT_DEFAULT_LEVERAGE),
    )
    leverage = float(leverage or EXECMGMT_DEFAULT_LEVERAGE)

    margin, margin_source = _first_float(
        env,
        ("ORDER_MARGIN_USD", "S34_LIVE_MARGIN_USDT", "FIXED_MARGIN_USDT"),
        _float_from_any(audit_env.get("margin_usdt")),
    )
    current_notional, notional_source = _first_float(
        env,
        ("ORDER_NOTIONAL_USD", "ORDER_NOTIONAL_USDT", "S34_LIVE_ORDER_NOTIONAL_USD", "S34_LIVE_NOTIONAL_USDT", "FIXED_NOTIONAL_USDT"),
        None,
    )
    if current_notional is None and margin is not None:
        current_notional = float(margin) * leverage
        notional_source = f"{margin_source or 'margin'}*leverage"
    if current_notional is None:
        current_notional = _float_from_any(audit_env.get("notional_usdt"), 0.0) or 0.0
        notional_source = "execution_audit.live_env.notional_usdt"
    if margin is None and current_notional is not None and leverage > 0:
        margin = float(current_notional) / leverage
        margin_source = "notional/leverage"

    margin_pct_val, _ = _first_float(env, ("S34_LIVE_MARGIN_PCT_ETH", "S34_LIVE_MARGIN_PCT"), 85.0)
    margin_pct_val = float(margin_pct_val or 85.0)

    equity_default = _float_from_any(sizing.get("equity_assumption_usdt"), EXECMGMT_DEFAULT_EQUITY_USDT)
    equity, equity_source = _first_float(
        env,
        ("ORDER_EQUITY_USD", "ACCOUNT_EQUITY_USD", "ACCOUNT_EQUITY_USDT", "S34_ACCOUNT_EQUITY_USDT", "S34_LIVE_EQUITY_USDT"),
        None,
    )
    if equity is None and current_notional and leverage > 0 and margin_pct_val > 0:
        equity = float(current_notional) / leverage / (margin_pct_val / 100.0)
        equity_source = f"derived_notional/lev/margin_pct({margin_pct_val:.0f}%)"
    if equity is None:
        equity = equity_default
        equity_source = "static_default"
    equity = float(equity or EXECMGMT_DEFAULT_EQUITY_USDT)

    tail_budget_usdt = equity * EXECMGMT_TAIL_BUDGET_PCT / 100.0
    worst_fill_abs_bps = abs(EXECMGMT_WORST_REAL_FILL_BPS)
    tail_budget_notional = tail_budget_usdt / (worst_fill_abs_bps / 10_000.0) if worst_fill_abs_bps > 0 else 0.0
    oversize = float(current_notional) / tail_budget_notional if tail_budget_notional > 0 else None
    recommended_margin = tail_budget_notional / leverage if leverage > 0 else None
    urgent = bool(oversize is not None and oversize > 10.0)

    return {
        "status": "ACTIVE",
        "mode": "READ_ONLY_RECOMMENDATION_NO_ACTION",
        "equity_usdt": round(equity, 4),
        "equity_source": equity_source or "sizing_shadow_default",
        "tail_budget_pct": EXECMGMT_TAIL_BUDGET_PCT,
        "tail_budget_usdt": round(tail_budget_usdt, 4),
        "worst_real_fill_bps": EXECMGMT_WORST_REAL_FILL_BPS,
        "tail_budget_notional_usdt": round(tail_budget_notional, 4),
        "current_notional_usdt": round(float(current_notional or 0.0), 4),
        "current_notional_source": notional_source,
        "current_margin_usdt": round(float(margin or 0.0), 4),
        "current_margin_source": margin_source,
        "leverage": round(leverage, 4),
        "leverage_source": leverage_source or "default_or_audit",
        "oversize_multiple": round(oversize, 1) if oversize is not None else None,
        "urgent": urgent,
        "flag": "URGENT_OVERSIZE_GT_10X" if urgent else "OK",
        "recommended_max_margin_usdt": round(recommended_margin, 4) if recommended_margin is not None else None,
        "read": "display/recommendation only; no order/config/env mutation",
    }


def load_stopprot_panel(
    *,
    env_path: Path = ENV_PATH,
    audit_path: Path = EXECUTION_AUDIT,
) -> dict[str, Any]:
    env = load_env_values(env_path)
    audit = {}
    if audit_path.exists():
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit = {}
    gap = audit.get("gap_through") if isinstance(audit.get("gap_through"), dict) else {}
    atom = audit.get("atomicity_audit") if isinstance(audit.get("atomicity_audit"), dict) else {}
    nominal, nominal_source = _first_float(
        env,
        ("S34_V_ENGINE_LIVE_STOP_BPS", "S34_V_ENGINE_EMERGENCY_STOP_BPS", "STOP_BPS"),
        _float_from_any(gap.get("current_stop_nominal_bps"), 150.0),
    )
    nominal = float(nominal or 150.0)
    worst_fill = _float_from_any(gap.get("current_stop_research_max_loss_bps"), EXECMGMT_WORST_REAL_FILL_BPS)
    worst_fill = float(worst_fill or EXECMGMT_WORST_REAL_FILL_BPS)
    gap_bps = max(0.0, abs(worst_fill) - abs(nominal))
    partial = abs(worst_fill) > abs(nominal)
    poll_sec = _float_from_any(atom.get("poll_sec"), 2.0)
    return {
        "status": "ACTIVE",
        "mode": "READ_ONLY_WARNING_NO_ACTION",
        "nominal_stop_bps": round(nominal, 4),
        "nominal_stop_source": nominal_source or "audit_or_default",
        "worst_real_fill_bps": round(worst_fill, 4),
        "gap_through_bps": round(gap_bps, 4),
        "gap_through_flag": "PARTIAL_PROTECTION" if partial else "FULL_WITHIN_AUDIT_SAMPLE",
        "atomicity": "ENTRY_THEN_STOP_POLL_LOOP",
        "unprotected_window_sec": round(float(poll_sec or 2.0), 4),
        "read": "entry fill to stop placement has an approximately 2s unprotected poll-loop window; warning only",
    }


def load_validated_signals(path: Path = VALIDATED_SIGNALS_REGISTRY) -> dict[str, Any]:
    """Load the permutation-validated signal registry (fifth-wave research results)."""
    if not path.exists():
        return {"status": "MISSING"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "error": str(exc)}
    sigs = payload.get("signals", [])
    armed   = [s for s in sigs if s.get("live_status") == "ARMED"]
    shadow  = [s for s in sigs if s.get("live_status") == "SHADOW_PAPER_TRACKING"]
    return {
        "status": payload.get("status", "UNKNOWN"),
        "n_signals_total": len(sigs),
        "n_armed_live": len(armed),
        "n_shadow_paper": len(shadow),
        "armed_ids": [s["id"] for s in armed],
        "shadow_ids": [s["id"] for s in shadow],
        "best_hold_wr": max((s.get("hold_wr", 0) for s in sigs), default=0),
        "best_hold_t3r": max((s.get("hold_t3r", 0) for s in sigs), default=0),
        "portfolio_hold_n": (payload.get("combined_portfolio") or {}).get("hold_n"),
        "portfolio_hold_wr": (payload.get("combined_portfolio") or {}).get("hold_wr"),
        "generated_utc": payload.get("generated_utc"),
        "signals": sigs,
    }


def load_silence_gate_status(
    executor_state_path: Path = LIVE_EXECUTOR_STATE,
) -> dict[str, Any]:
    """Read live executor state and extract silence gate tracking."""
    if not executor_state_path.exists():
        return {"status": "EXECUTOR_STATE_MISSING"}
    try:
        state = json.loads(executor_state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "EXECUTOR_STATE_INVALID"}
    active = state.get("active")
    status = state.get("status") or {}
    if not isinstance(active, dict):
        return {
            "status": "NO_ACTIVE_POSITION",
            "executor_mode": status.get("mode"),
            "executor_rule": status.get("rule"),
            "silence_gate_armed": True,
            "active_status": None,
        }
    gate_resolved = active.get("silence_gate_resolved")
    return {
        "status": "POSITION_ACTIVE",
        "executor_mode": status.get("mode"),
        "active_status": active.get("status"),
        "anchor_ts_ms": active.get("anchor_ts_ms"),
        "anchor_utc": active.get("anchor_utc"),
        "fill_detected_at_utc": active.get("fill_detected_at_utc"),
        "exit_due_ts_ms": active.get("exit_due_ts_ms"),
        "silence_gate_armed": True,
        "silence_gate_resolved": gate_resolved,
        "silence_confirmed_at_utc": active.get("silence_confirmed_at_utc"),
        "noisy_cascade_ts_ms": active.get("noisy_cascade_ts_ms"),
        "exit_reason": active.get("exit_reason"),
    }


def load_realtime_shadow_bucket(
    state_path: Path = REALTIME_SHADOW_STATE,
    ledger_path: Path = REALTIME_SHADOW_LEDGER,
) -> dict[str, Any]:
    """Load shadow paper state + compute P&L from ledger."""
    result: dict[str, Any] = {"status": "ACTIVE"}
    # State
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            positions = state.get("positions", {})
            open_pos  = {k: v for k, v in positions.items()
                         if v.get("status") not in {None, "EXPIRED_NO_BTC", "EXPIRED_SILENCE"}}
            result["open_n"] = len(open_pos)
            result["open_positions"] = [
                {"id": k, "signal": v.get("signal"), "direction": v.get("direction"),
                 "status": v.get("status"), "score": v.get("score"),
                 "sync_k": v.get("sync_k")}
                for k, v in list(open_pos.items())[:5]
            ]
            result["pnl"] = state.get("pnl", {})
        except Exception:
            result["open_n"] = 0
    else:
        result["open_n"] = 0

    # Ledger stats
    if ledger_path.exists():
        by_sig: dict[str, list[float]] = {}
        close_rows: list[dict] = []
        n_events = 0
        with ledger_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("event") == "CLOSE":
                    n_events += 1
                    sig = r.get("signal", "?")
                    net = r.get("net_bps")
                    if net is not None:
                        by_sig.setdefault(sig, []).append(float(net))
                    close_rows.append(r)
        result["closed_trades_n"] = n_events
        sig_stats = {}
        for sig, vals in by_sig.items():
            wins = sum(1 for v in vals if v > 0)
            sig_stats[sig] = {
                "n": len(vals), "wr": round(wins/len(vals), 3) if vals else 0,
                "total_net": round(sum(vals), 1),
                "avg_net": round(sum(vals)/len(vals), 1) if vals else 0,
            }
        result["signal_stats"] = sig_stats

        def _long_base(r: dict) -> bool:
            return (r.get("signal") == "LONG_SILENCE"
                    and str(r.get("close_reason") or "") == "TIME_EXIT"
                    and float(r.get("sync_k") or 0) < 200_000)

        def _cand(rows: list, fn) -> dict:
            matched = [r for r in rows if fn(r)]
            nets = [float(r["net_bps"]) for r in matched if r.get("net_bps") is not None]
            wins = sum(1 for v in nets if v > 0)
            return {"n": len(matched), "wr": round(wins / len(nets), 3) if nets else 0,
                    "avg_bps": round(sum(nets) / len(nets), 1) if nets else 0}

        result["candidate_buckets"] = [
            {"name": "C_score_relax",      "label": "score>=2 (btc7d removed)",      "mc_p": 0.018, "stats": _cand(close_rows, lambda r: _long_base(r) and float(r.get("score") or 0) >= 2)},
            {"name": "C_btc7d500",         "label": "btc7d in (-500,0) [live only]", "mc_p": 0.240, "stats": _cand(close_rows, lambda r: _long_base(r) and r.get("btc7d_bps") is not None and -500 < float(r.get("btc7d_bps") or 0) < 0)},
            {"name": "C_freq_btc4h",       "label": "btc4h<0 no btc7d [live only]",  "mc_p": 0.375, "stats": _cand(close_rows, lambda r: _long_base(r) and r.get("btc4h_bps") is not None and float(r.get("btc4h_bps") or 0) < 0)},
            {"name": "C_double_cascade",   "label": "density_24h>=1+prebuildup>=2 [live only]", "mc_p": 0.001, "stats": _cand(close_rows, lambda r: _long_base(r) and r.get("density_24h") is not None and r.get("prebuildup") is not None and int(r.get("density_24h") or 0) >= 1 and int(r.get("prebuildup") or 0) >= 2)},
            {"name": "C_btc_falling_fast", "label": "btc5m<-20bps [live only]",      "mc_p": 0.001, "stats": _cand(close_rows, lambda r: _long_base(r) and r.get("btc5m_bps") is not None and float(r.get("btc5m_bps") or 0) < -20)},
            {"name": "C_failed_cascade",   "label": "failed_cascade==True [live only]", "mc_p": 0.006, "stats": _cand(close_rows, lambda r: _long_base(r) and r.get("failed_cascade") is True)},
        ]
    else:
        result["closed_trades_n"] = 0
        result["signal_stats"] = {}
        result["candidate_buckets"] = []
    return result


def window_notional(rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    return sum(float(r["notional"]) for r in rows if int(start_ms) <= int(r["ts_ms"]) <= int(end_ms))


# ---------------------------------------------------------------------------
# Panel 1 + 2: partial cascade snapshot (knowable at as_of only)
# ---------------------------------------------------------------------------

def partial_cascade_state(
    rows: list[dict[str, Any]], as_of_ts: int, *, accel_window_sec: int
) -> dict[str, Any]:
    rows = [r for r in rows if int(r["ts_ms"]) <= int(as_of_ts)]
    if not rows:
        return {
            "liq_count": 0,
            "running_notional": 0.0,
            "first_ts_ms": None,
            "last_ts_ms": None,
            "elapsed_since_first_sec": 0.0,
            "gap_since_last_sec": None,
            "running_rate": 0.0,
            "running_accel": 0.0,
            "single_liq_dominance": 0.0,
            "max_single_notional": 0.0,
        }
    first_ts = int(rows[0]["ts_ms"])
    last_ts = int(rows[-1]["ts_ms"])
    total = sum(float(r["notional"]) for r in rows)
    max_single = max(float(r["notional"]) for r in rows)
    elapsed = max(0.0, (as_of_ts - first_ts) / 1000.0)
    aw = float(accel_window_sec)
    cur_rate = window_notional(rows, as_of_ts - int(aw) * 1000, as_of_ts) / max(1.0, aw)
    prev_rate = window_notional(rows, as_of_ts - 2 * int(aw) * 1000, as_of_ts - int(aw) * 1000) / max(1.0, aw)
    return {
        "liq_count": len(rows),
        "running_notional": total,
        "first_ts_ms": first_ts,
        "last_ts_ms": last_ts,
        "elapsed_since_first_sec": elapsed,
        "gap_since_last_sec": max(0.0, (as_of_ts - last_ts) / 1000.0),
        "running_rate": total / max(1.0, (last_ts - first_ts) / 1000.0),
        "running_accel": cur_rate - prev_rate,
        "single_liq_dominance": (max_single / total * 100.0) if total > 0 else 0.0,
        "max_single_notional": max_single,
    }


def cascade_features(state: dict[str, Any], as_of_ts: int) -> list[FeatureValue]:
    """All RUNNING_CLUSTER and knowable at as_of by construction."""
    t = int(as_of_ts)
    keys = (
        "running_notional",
        "liq_count",
        "running_rate",
        "running_accel",
        "single_liq_dominance",
        "elapsed_since_first_sec",
    )
    return [FeatureValue(k, state.get(k), t, FeatureClass.RUNNING_CLUSTER) for k in keys]


def anchor_health_panel(state: dict[str, Any]) -> dict[str, Any]:
    notional = float(state["running_notional"])
    next_threshold = None
    distance = None
    for th in THRESHOLDS_USD:
        if notional < th:
            next_threshold = th
            distance = th - notional
            break
    crossed = [th for th in THRESHOLDS_USD if notional >= th]
    return {
        "running_notional": r1(notional),
        "highest_threshold_crossed": (max(crossed) if crossed else None),
        "next_threshold": next_threshold,
        "distance_to_next_usd": r1(distance) if distance is not None else None,
        "liq_count": int(state["liq_count"]),
        "elapsed_since_first_sec": r1(state["elapsed_since_first_sec"]),
        "gap_since_last_sec": r1(state["gap_since_last_sec"]) if state["gap_since_last_sec"] is not None else None,
        "liq_rate_usd_per_sec": r1(state["running_rate"]),
        "acceleration_usd_per_sec": r1(state["running_accel"]),
        "knowable": True,  # asserted upstream; here as an explicit pilot flag
    }


def cascade_phase_panel(state: dict[str, Any], *, dead_gap_sec: float) -> dict[str, Any]:
    count = int(state["liq_count"])
    notional = float(state["running_notional"])
    accel = float(state["running_accel"])
    gap = state["gap_since_last_sec"]
    dominance = float(state["single_liq_dominance"])

    if count == 0 or (gap is not None and gap > dead_gap_sec):
        phase = "DEAD"
    elif notional < THRESHOLDS_USD[0]:
        phase = "FORMING"
    elif accel > 0.0:
        phase = "ACCELERATING"
    else:
        phase = "DECELERATING"
    if phase == "DECELERATING" and notional >= THRESHOLDS_USD[2] and (gap or 0.0) >= 10.0:
        phase = "EXHAUSTION"

    shape = "one_shot_dominant" if dominance >= 50.0 else "distributed"

    # market-state scores in [0,1] -- physical cascade behaviour, NOT edge.
    # materiality damps sub-threshold "dust" cascades so a $20 liq cannot read as high momentum.
    materiality = min(1.0, notional / THRESHOLDS_USD[0])
    momentum = 0.0
    if phase in ("ACCELERATING", "FORMING"):
        raw = 0.4 + max(0.0, accel) / max(1.0, max(notional, THRESHOLDS_USD[0]) / 60.0)
        if gap is not None and gap <= 5.0:
            raw += 0.2
        momentum = min(1.0, raw) * materiality
    fade = 0.0
    if phase in ("DECELERATING", "EXHAUSTION"):
        fade = 0.5 + (0.3 if dominance >= 50.0 else 0.0) + (0.2 if phase == "EXHAUSTION" else 0.0)
        fade = min(1.0, fade)
    return {
        "phase": phase,
        "shape": shape,
        "single_liq_dominance_pct": r1(dominance),
        "continuation_momentum_score": r1(momentum),
        "fade_momentum_score": r1(fade),
    }


# ---------------------------------------------------------------------------
# Panel 3: regime compass
# ---------------------------------------------------------------------------

def regime_compass_panel(
    conn: sqlite3.Connection, lane: Lane, marks, btc_marks, as_of_ts: int, *, btc_window_sec: int
) -> dict[str, Any]:
    day0 = day_start_ms(as_of_ts)
    day_open = marks.at_or_after(day0)
    cur = marks.at_or_before(as_of_ts)
    day_trend_bps = None
    if day_open and cur and float(day_open[1]) > 0:
        day_trend_bps = (float(cur[1]) - float(day_open[1])) / float(day_open[1]) * 10_000.0
    day_slice = marks.slice_range(day0, as_of_ts)
    day_range_bps = None
    if day_slice and day_open and float(day_open[1]) > 0:
        hi = max(px for _, px in day_slice)
        lo = min(px for _, px in day_slice)
        day_range_bps = (hi - lo) / float(day_open[1]) * 10_000.0
    btc_ret = btc_marks.ret_bps(as_of_ts - int(btc_window_sec) * 1000, as_of_ts)
    # alignment: for LONG we want BTC up; for SHORT we want BTC down.
    aligned = None
    if btc_ret is not None:
        aligned = (btc_ret > 0) if lane.direction == "LONG" else (btc_ret < 0)
    return {
        "session": session_label(as_of_ts),
        "day_trend_bps": r1(day_trend_bps) if day_trend_bps is not None else None,
        "day_range_bps": r1(day_range_bps) if day_range_bps is not None else None,
        "btc_return_bps": r1(btc_ret) if btc_ret is not None else None,
        "btc_aligned_with_direction": aligned,
        "funding_rate": funding_rate_at(conn, lane.symbol, as_of_ts),
    }


# ---------------------------------------------------------------------------
# Panel 4: execution reality
# ---------------------------------------------------------------------------

def execution_reality_panel(
    conn: sqlite3.Connection, lane: Lane, as_of_ts: int, *, max_book_staleness_sec: int
) -> dict[str, Any]:
    book = book_at(conn, lane.symbol, as_of_ts, max_book_staleness_sec)
    if not book:
        return {
            "book_available": False,
            "no_fill_risk": "HIGH",
            "book_staleness_sec": None,
            "spread_bps": None,
            "expected_entry_adverse_bps": None,
        }
    spread_bps = (book.ask - book.bid) / book.mid * 10_000.0 if book.mid > 0 else None
    return {
        "book_available": True,
        "no_fill_risk": "LOW" if book.staleness_ms <= max_book_staleness_sec * 1000 else "MED",
        "book_staleness_sec": r1(book.staleness_ms / 1000.0),
        "spread_bps": r1(spread_bps) if spread_bps is not None else None,
        # crossing the spread on a taker entry costs ~half the spread vs mid.
        "expected_entry_adverse_bps": r1(spread_bps / 2.0) if spread_bps is not None else None,
    }


# ---------------------------------------------------------------------------
# Panel 5: rule confidence (from clean holdout-validated recheck)
# ---------------------------------------------------------------------------

def load_rule_confidence(recheck_json: Path) -> dict[str, list[dict[str, Any]]]:
    if not recheck_json.exists():
        return {}
    payload = json.loads(recheck_json.read_text(encoding="utf-8"))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for res in payload.get("results", []):
        family = str(res.get("family"))
        cal = (res.get("calibration") or {}).get("summary") or {}
        hold = (res.get("holdout") or {}).get("summary") or {}
        by_family.setdefault(family, []).append(
            {
                "rule_name": res.get("rule_name"),
                "verdict": res.get("verdict"),
                "cal_median_bps": cal.get("median_bps"),
                "cal_mean_bps": cal.get("mean_bps"),
                "cal_t3r_cum_bps": cal.get("top3_winner_removed_cum_bps"),
                "hold_median_bps": hold.get("median_bps"),
                "hold_mean_bps": hold.get("mean_bps"),
                "hold_t3r_cum_bps": hold.get("top3_winner_removed_cum_bps"),
                "hold_filled_n": (res.get("holdout") or {}).get("filled_n"),
                "no_fill_rate_cal": (res.get("calibration") or {}).get("no_fill_rate"),
            }
        )
    return by_family


def rule_confidence_panel(
    family: str,
    confidence: dict[str, list[dict[str, Any]]],
    validated_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = confidence.get(family, [])
    paper_candidates = [r for r in rules if r.get("verdict") == "PAPER_CANDIDATE"]
    tested = [r for r in rules if r.get("verdict") in ("RESEARCH_ONLY", "PAPER_CANDIDATE")]
    if not rules:
        edge = "UNKNOWN"
        evidence = "NO_CLEAN_RECHECK"
    elif paper_candidates:
        edge = "PROVEN"
        evidence = "CLEAN_HOLDOUT_VALIDATED"
    elif not tested:
        edge = "UNKNOWN"
        evidence = "ONLY_THIN_OR_BLOCKED_ROUTES"
    else:
        thin = any((r.get("hold_median_bps") or -1) > 0 for r in tested)
        edge = "THIN" if thin else "NONE"
        evidence = "CLEAN_BUT_FAILS_GATE"

    # Override with fifth-wave validated signals registry when available
    validated_sigs: list[dict[str, Any]] = []
    if validated_registry and validated_registry.get("status") not in {"MISSING", "INVALID"}:
        validated_sigs = [s for s in validated_registry.get("signals", [])
                          if s.get("family") == family]
    if validated_sigs:
        armed = [s for s in validated_sigs if s.get("live_status") == "ARMED"]
        edge = "PROVEN_5TH_WAVE"
        evidence = f"PERM_OOS_PASS_{len(validated_sigs)}_SIGNALS"
        paper_candidates_v = [{"rule_name": s["id"], "verdict": "PAPER_CANDIDATE",
                                "hold_t3r_cum_bps": s.get("hold_t3r"),
                                "hold_wr": s.get("hold_wr"),
                                "hold_filled_n": s.get("hold_n"),
                                "live_status": s.get("live_status")}
                               for s in validated_sigs]
        paper_candidates = paper_candidates or paper_candidates_v

    return {
        "family_edge": edge,
        "evidence_quality": evidence,
        "paper_candidate_count": len(paper_candidates),
        "validated_signal_count": len(validated_sigs),
        "rules": rules,
        "validated_signals": validated_sigs,
    }


# ---------------------------------------------------------------------------
# Exploratory V-engine observation (not a permission grant)
# ---------------------------------------------------------------------------

def exploratory_v_fade_panel(lane: Lane, state: dict[str, Any], marks: MarkIndex, as_of_ts: int) -> dict[str, Any]:
    protocol = "S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D"
    base = {
        "protocol": protocol,
        "status": "INACTIVE",
        "permission": None,
        "observation_only": True,
        "reason": "lane_not_applicable",
        "maker_direction": None,
        "vdepth_bps": None,
        "prior_4h_bps": None,
        "rule": {
            "symbol": "ETHUSDT",
            "liq_side": "SELL",
            "threshold_usd": 200_000.0,
            "vdepth_min_bps": 28.0,
            "vdepth_max_bps": 40.0,
            "prior_4h_lt_bps": -50.0,
            "maker_offset_bps": 20.0,
            "cross_margin_bps": 2.0,
            "horizon_sec": 7200,
        },
    }
    if lane.symbol != "ETHUSDT" or lane.liq_side != "SELL":
        return base
    base["maker_direction"] = "LONG"
    first_ts = state.get("first_ts_ms")
    if first_ts is None or int(state.get("liq_count") or 0) <= 0:
        base.update({"reason": "no_active_sell_cascade"})
        return base
    if float(state.get("running_notional") or 0.0) < 200_000.0:
        base.update({"reason": "below_200k_running_notional"})
        return base
    first = marks.at_or_after(int(first_ts))
    cur = marks.at_or_before(int(as_of_ts))
    if not first or not cur or float(first[1]) <= 0.0:
        base.update({"reason": "missing_mark_for_vdepth"})
        return base
    vdepth = (float(first[1]) - float(cur[1])) / float(first[1]) * 10_000.0
    prior4h = prior_return_bps(marks, as_of_ts, 4 * 3600)
    base["vdepth_bps"] = r1(vdepth)
    base["prior_4h_bps"] = r1(prior4h) if prior4h is not None else None
    if not (28.0 <= float(vdepth) < 40.0):
        base.update({"reason": "vdepth_outside_28_40"})
        return base
    if prior4h is None or not (float(prior4h) < -50.0):
        base.update({"reason": "prior4h_not_down_enough"})
        return base
    base.update(
        {
            "status": "ACTIVE",
            "permission": "EXPLORATORY_V_FADE_V0_1",
            "reason": "frozen_exploratory_conditions_met_observe_only",
        }
    )
    return base


# ---------------------------------------------------------------------------
# Navigation verdict
# ---------------------------------------------------------------------------

def navigation_verdict(panels: dict[str, Any]) -> dict[str, Any]:
    phase = panels["cascade_phase"]["phase"]
    momentum = float(panels["cascade_phase"]["continuation_momentum_score"] or 0.0)
    fade = float(panels["cascade_phase"]["fade_momentum_score"] or 0.0)
    edge = panels["rule_confidence"]["family_edge"]
    exec_ok = panels["execution_reality"]["book_available"] and (
        panels["execution_reality"]["no_fill_risk"] != "HIGH"
    )

    # Permission requires PROVEN edge. Market momentum alone never grants it.
    if edge == "PROVEN" and exec_ok and momentum >= 0.5 and phase in ("ACCELERATING", "FORMING"):
        permission = "CONTINUATION_VIABLE"
        action = f"new {panels['_direction']} allowed (proven edge + live momentum); size per risk"
    elif edge == "PROVEN" and exec_ok and fade >= 0.7 and phase == "EXHAUSTION":
        permission = "FADE_VIABLE"
        action = f"counter-{panels['_direction']} fade candidate (proven edge + exhaustion)"
    else:
        permission = "OBSERVE_ONLY"
        reasons = []
        if edge != "PROVEN":
            reasons.append(f"edge={edge}")
        if not exec_ok:
            reasons.append("execution_degraded")
        if phase in ("DEAD",):
            reasons.append("no_active_cascade")
        action = f"no new entry ({', '.join(reasons) or 'conditions not met'}); observe"
    return {
        "permission": permission,
        "action": action,
        "continuation_momentum": r1(momentum),
        "fade_momentum": r1(fade),
        "blocked_by_edge": edge != "PROVEN",
    }


def evaluate_lane(
    conn: sqlite3.Connection,
    lane: Lane,
    confidence: dict[str, list[dict[str, Any]]],
    btc_marks,
    *,
    as_of_ts: int,
    bucket_sec: int,
    accel_window_sec: int,
    dead_gap_sec: float,
    btc_window_sec: int,
    max_book_staleness_sec: int,
    validated_registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bucket_ms = int(bucket_sec) * 1000
    bucket_start = (int(as_of_ts) // bucket_ms) * bucket_ms
    rows = load_liquidations(conn, lane.symbol, lane.liq_side, bucket_start, as_of_ts)
    state = partial_cascade_state(rows, as_of_ts, accel_window_sec=accel_window_sec)

    # Structural anti-lookahead guard: every cascade feature must be knowable at as_of.
    assert_feature_set_available(
        cascade_features(state, as_of_ts), as_of_ts, context=f"navigation:{lane.family}"
    )

    marks_start = min(day_start_ms(as_of_ts), int(as_of_ts) - 4 * 3600 * 1000 - 300_000)
    marks = load_mark_index_range(conn, lane.symbol, marks_start, as_of_ts)
    panels = {
        "_direction": lane.direction,
        "anchor_health": anchor_health_panel(state),
        "cascade_phase": cascade_phase_panel(state, dead_gap_sec=dead_gap_sec),
        "regime_compass": regime_compass_panel(
            conn, lane, marks, btc_marks, as_of_ts, btc_window_sec=btc_window_sec
        ),
        "execution_reality": execution_reality_panel(
            conn, lane, as_of_ts, max_book_staleness_sec=max_book_staleness_sec
        ),
        "rule_confidence": rule_confidence_panel(lane.family, confidence, validated_registry),
        "exploratory_v_fade": exploratory_v_fade_panel(lane, state, marks, as_of_ts),
    }
    verdict = navigation_verdict(panels)
    panels.pop("_direction", None)
    return {
        "lane": {"symbol": lane.symbol, "liq_side": lane.liq_side, "direction": lane.direction, "family": lane.family},
        "as_of_ts_ms": int(as_of_ts),
        "as_of_utc": iso_ms(as_of_ts),
        "panels": panels,
        "verdict": verdict,
    }


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"S34 CASCADE NAVIGATION DASHBOARD  |  as_of={report['as_of_utc']}")
    lines.append("=" * 78)
    mgmt = report.get("execution_management") or {}
    if mgmt:
        lines.append(
            "MGMT   : "
            f"size={mgmt.get('size_status')} kill={mgmt.get('kill_status')} "
            f"regime={mgmt.get('regime_status')} "
            f"margin env/budget={mgmt.get('env_planned_margin_usdt')}/{mgmt.get('max_tail_budget_margin_usdt')} "
            f"oversize={mgmt.get('oversize_multiple')}x "
            f"rec={mgmt.get('recommendation')}"
        )
    v9 = report.get("risk_review_v9") or {}
    if v9 and v9.get("status") == "ACTIVE":
        lines.append(
            "RISKV9 : "
            f"kill={v9.get('recommended_kill')} "
            f"tick={v9.get('tick_atomicity_status')} worst={v9.get('tick_worst_adverse_bps')}bps "
            f"gate={v9.get('forward_gate')}"
        )
    v11 = report.get("forward_governance_v11") or {}
    if v11 and v11.get("status") == "ACTIVE":
        lines.append(
            "GOVV11 : "
            f"fwd={v11.get('forward_integrity')} validN={v11.get('valid_forward_n')} "
            f"decision={v11.get('operator_governance')} "
            f"truth={v11.get('execution_truth')} fee={v11.get('fee_tier')}"
        )
    mirror = report.get("v02_shadow_mirror") or {}
    if mirror and mirror.get("status") == "ACTIVE":
        lines.append(
            "SHADOW : "
            f"{mirror.get('permission')} decision={mirror.get('decision')} "
            f"live_match={mirror.get('live_rule_match')} rows={mirror.get('ledger_rows')} "
            f"recentN={mirror.get('recent_n')} recentSum={mirror.get('recent_sum_bps')}bps "
            f"kill={mirror.get('kill_triggered')}"
        )
    h4_shadow = report.get("v02_h4_shadow") or {}
    if h4_shadow and h4_shadow.get("status") == "ACTIVE":
        lines.append(
            "H4SHAD : "
            f"h2Sum={h4_shadow.get('h2_sum_bps')}bps "
            f"h4Sum={h4_shadow.get('h4_sum_bps')}bps "
            f"h4T3R={h4_shadow.get('h4_t3r_bps')}bps "
            f"crossPolicy={h4_shadow.get('cross_policy_sum_bps')}bps "
            f"sl150Touches={h4_shadow.get('sl150_touch_count')} "
            f"{h4_shadow.get('decision')}"
        )
    sizing = report.get("sizing_shadow_paper") or {}
    if sizing and sizing.get("status") == "SHADOW_PAPER_SIZING_ONLY_NO_ORDER":
        cur = sizing.get("current_env") or {}
        bal = sizing.get("balanced") or {}
        surv = sizing.get("survival") or {}
        lines.append(
            "SIZE   : "
            f"current pnl={cur.get('sum_pnl_usdt')} end={cur.get('ending_equity_usdt')} "
            f"balanced pnl={bal.get('sum_pnl_usdt')} end={bal.get('ending_equity_usdt')} "
            f"survival pnl={surv.get('sum_pnl_usdt')} end={surv.get('ending_equity_usdt')} "
            "shadow_only"
        )
    execmgmt = report.get("execmgmt") or {}
    if execmgmt and execmgmt.get("status") == "ACTIVE":
        lines.append(
            "EXECMGMT: "
            f"notional={execmgmt.get('current_notional_usdt')}usdt "
            f"tail_notional={execmgmt.get('tail_budget_notional_usdt')}usdt "
            f"oversize={execmgmt.get('oversize_multiple')}x "
            f"rec_margin={execmgmt.get('recommended_max_margin_usdt')}usdt "
            f"flag={execmgmt.get('flag')} read_only"
        )
    stopprot = report.get("stopprot") or {}
    if stopprot and stopprot.get("status") == "ACTIVE":
        lines.append(
            "STOPPROT: "
            f"nominal={stopprot.get('nominal_stop_bps')}bps "
            f"worst_fill={stopprot.get('worst_real_fill_bps')}bps "
            f"gap={stopprot.get('gap_through_bps')}bps "
            f"flag={stopprot.get('gap_through_flag')} "
            f"atomicity={stopprot.get('atomicity')}~{stopprot.get('unprotected_window_sec')}s"
        )
    # ── Silence gate status (live executor) ──────────────────────────────────
    sg = report.get("silence_gate") or {}
    sgate_resolved = sg.get("silence_gate_resolved")
    if sg.get("status") == "POSITION_ACTIVE":
        gate_label = {
            "SILENCE": "SILENCE_CONFIRMED_4H_HOLD",
            "NOISY":   "NOISY_EARLY_EXIT_TRIGGERED",
            None:      "MONITORING_IN_PROGRESS",
        }.get(sgate_resolved, sgate_resolved or "MONITORING")
        lines.append(
            "SGATE  : ARMED active_pos=True"
            f" gate={gate_label}"
            f" anchor={sg.get('anchor_utc','?')[:19]}"
            f" mode={sg.get('executor_mode','?')}"
        )
    else:
        lines.append(
            f"SGATE  : ARMED active_pos=False mode={sg.get('executor_mode','?')}"
            " (silence gate will activate on next cascade entry)"
        )
    # ── Real-time shadow bucket ────────────────────────────────────────────────
    sb = report.get("shadow_bucket") or {}
    if sb:
        sig_stats = sb.get("signal_stats") or {}
        stat_parts = []
        for sig_key, stat in sig_stats.items():
            stat_parts.append(f"{sig_key.replace('_','')[:6]}:N={stat['n']}WR={stat['wr']:.0%}tot={stat['total_net']:+.0f}")
        stat_str = " | ".join(stat_parts) if stat_parts else "no_closed_trades_yet"
        lines.append(
            f"SBUCK  : open={sb.get('open_n',0)} closed={sb.get('closed_trades_n',0)}"
            f" signals=LONG_SILENCE+SHORT_NEITHER+SHORT_NOISY"
            f" [{stat_str}]"
        )
        for cb in sb.get("candidate_buckets") or []:
            s = cb.get("stats") or {}
            n = s.get("n", 0)
            wr_str = f"{s['wr']:.0%}" if n > 0 else "-"
            avg_str = f"{s['avg_bps']:+.1f}bps" if n > 0 else "-"
            lines.append(
                f"  CAND  [{cb['name']}] N={n} WR={wr_str} avg={avg_str}"
                f" mc_p={cb.get('mc_p','?')} | {cb.get('label','')}"
            )
    # ── Validated signals registry ───────────────────────────────────────────
    vr = report.get("validated_registry") or {}
    if vr and vr.get("status") not in {"MISSING", "INVALID", None}:
        lines.append(
            f"VALREG : {vr.get('n_signals_total',0)} signals"
            f" armed={vr.get('n_armed_live',0)} shadow={vr.get('n_shadow_paper',0)}"
            f" best_wr={vr.get('best_hold_wr',0):.0%}"
            f" portfolio_n={vr.get('portfolio_hold_n')} portfolio_wr={vr.get('portfolio_hold_wr',0):.0%}"
            f" status={vr.get('status','?')}"
        )
    # ── Execution-management audit (sizing / stop / atomicity / tail) ──────────
    ea = report.get("execution_audit") or {}
    if ea.get("status") == "ACTIVE":
        oversize = ea.get("oversize_multiple")
        oversize_str = f"{oversize}x" if oversize else "?"
        atom_short = "NOT_ATOMIC" if "NOT_ATOMIC" in str(ea.get("atomicity", "")) else str(ea.get("atomicity", "?"))
        urgent = "URGENT_RESIZE" if (oversize or 0) > 10 else "OK"
        lines.append(
            "EXECAUD : "
            f"margin_env={ea.get('margin_env_usdt')}usdt budget={ea.get('budget_margin_usdt')}usdt "
            f"oversize={oversize_str} leverage={ea.get('leverage')}x "
            f"atomicity={atom_short} tail={ea.get('tail_rate_pct')}% "
            f"{urgent}"
        )
        lines.append(
            "STOPAUD : "
            f"stop_nom={ea.get('stop_nominal_bps')}bps worst_fill={ea.get('stop_worst_fill_bps')}bps "
            f"GAP_THROUGH={ea.get('stop_gap_bps')}bps equity_risk={ea.get('stop_pct_equity')}% "
            f"tail_in3={ea.get('tail_in_3_pct')}% tail_in5={ea.get('tail_in_5_pct')}% "
            f"best_stop={ea.get('best_stop_variant')}"
        )
    for lane in report["lanes"]:
        p = lane["panels"]
        ah, cp, rc, ex, rcf, evf = (
            p["anchor_health"],
            p["cascade_phase"],
            p["regime_compass"],
            p["execution_reality"],
            p["rule_confidence"],
            p["exploratory_v_fade"],
        )
        v = lane["verdict"]
        L = lane["lane"]
        lines.append("")
        lines.append(f"{L['symbol']} {L['liq_side']} ({L['direction']})  [{L['family']}]")
        lines.append(
            f"  Anchor : notional={ah['running_notional']} next={ah['next_threshold']} "
            f"dist={ah['distance_to_next_usd']} n={ah['liq_count']} "
            f"elapsed={ah['elapsed_since_first_sec']}s gap={ah['gap_since_last_sec']}s "
            f"accel={ah['acceleration_usd_per_sec']} knowable={ah['knowable']}"
        )
        lines.append(
            f"  Phase  : {cp['phase']} / {cp['shape']} dom={cp['single_liq_dominance_pct']}% "
            f"cont_mom={cp['continuation_momentum_score']} fade_mom={cp['fade_momentum_score']}"
        )
        lines.append(
            f"  Regime : session={rc['session']} day_trend={rc['day_trend_bps']}bps "
            f"day_range={rc['day_range_bps']}bps btc={rc['btc_return_bps']}bps aligned={rc['btc_aligned_with_direction']}"
        )
        lines.append(
            f"  Exec   : book={ex['book_available']} stale={ex['book_staleness_sec']}s "
            f"spread={ex['spread_bps']}bps entry_adverse~{ex['expected_entry_adverse_bps']}bps no_fill={ex['no_fill_risk']}"
        )
        lines.append(
            f"  Rule   : edge={rcf['family_edge']} evidence={rcf['evidence_quality']} "
            f"paper_candidates={rcf['paper_candidate_count']}"
        )
        if evf["status"] == "ACTIVE":
            lines.append(
                f"  VEng   : {evf['permission']} maker={evf['maker_direction']} "
                f"vdepth={evf['vdepth_bps']} prior4h={evf['prior_4h_bps']} observe_only={evf['observation_only']}"
            )
        lines.append(f"  >> {v['permission']}: {v['action']}")
    lines.append("")
    return "\n".join(lines)


def render_md(report: dict[str, Any]) -> str:
    mgmt = report.get("execution_management") or {}
    lines = [
        "# S34 Cascade Navigation Dashboard",
        "",
        f"Generated: `{report['generated_at_utc']}`  |  as_of: `{report['as_of_utc']}`",
        "",
        "NAVIGATION / PERMISSION layer -- not a trade trigger. Every feature is knowable at `as_of` "
        "(asserted via feature-availability). `family_edge` comes from the clean holdout-validated route recheck; "
        "market momentum never grants permission on its own.",
        "",
        "| Lane | Phase | Cont/Fade Mom | Session | DayTrend | BTC Aligned | Spread | Edge | V-Engine | Permission |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for lane in report["lanes"]:
        p = lane["panels"]
        cp, rc, ex, rcf, evf, v, L = (
            p["cascade_phase"],
            p["regime_compass"],
            p["execution_reality"],
            p["rule_confidence"],
            p["exploratory_v_fade"],
            lane["verdict"],
            lane["lane"],
        )
        lines.append(
            f"| {L['symbol']} {L['liq_side']} | {cp['phase']}/{cp['shape']} | "
            f"{cp['continuation_momentum_score']}/{cp['fade_momentum_score']} | {rc['session']} | "
            f"{rc['day_trend_bps']} | {rc['btc_aligned_with_direction']} | {ex['spread_bps']} | "
            f"{rcf['family_edge']} | {evf['permission'] or evf['status']} | {v['permission']} |"
        )
    lines.append("")
    if mgmt:
        lines.extend(
            [
                "## Execution Management",
                "",
                "| Size | Kill | Regime | Env Margin | Tail Budget Margin | Stop Budget Margin | Oversize | Recommendation |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
                (
                    f"| {mgmt.get('size_status')} | {mgmt.get('kill_status')} | {mgmt.get('regime_status')} | "
                    f"{mgmt.get('env_planned_margin_usdt')} | {mgmt.get('max_tail_budget_margin_usdt')} | "
                    f"{mgmt.get('max_stop_budget_margin_usdt')} | {mgmt.get('oversize_multiple')} | "
                    f"{mgmt.get('recommendation')} |"
                ),
                "",
            ]
        )
    execmgmt = report.get("execmgmt") or {}
    stopprot = report.get("stopprot") or {}
    if execmgmt and execmgmt.get("status") == "ACTIVE":
        lines.extend(
            [
                "## EXECMGMT Sizing",
                "",
                "| Equity | Tail Budget | Current Notional | Tail-Budget Notional | Oversize | Recommended Max Margin | Flag | Mode |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                (
                    f"| {execmgmt.get('equity_usdt')} | {execmgmt.get('tail_budget_usdt')} | "
                    f"{execmgmt.get('current_notional_usdt')} | {execmgmt.get('tail_budget_notional_usdt')} | "
                    f"{execmgmt.get('oversize_multiple')} | {execmgmt.get('recommended_max_margin_usdt')} | "
                    f"{execmgmt.get('flag')} | {execmgmt.get('mode')} |"
                ),
                "",
            ]
        )
    if stopprot and stopprot.get("status") == "ACTIVE":
        lines.extend(
            [
                "## STOPPROT Stop Protection",
                "",
                "| Nominal Stop | Worst Real Fill | Gap-Through | Flag | Atomicity | Unprotected Window | Mode |",
                "| ---: | ---: | ---: | --- | --- | ---: | --- |",
                (
                    f"| {stopprot.get('nominal_stop_bps')} | {stopprot.get('worst_real_fill_bps')} | "
                    f"{stopprot.get('gap_through_bps')} | {stopprot.get('gap_through_flag')} | "
                    f"{stopprot.get('atomicity')} | {stopprot.get('unprotected_window_sec')} | "
                    f"{stopprot.get('mode')} |"
                ),
                "",
            ]
        )
    v9 = report.get("risk_review_v9") or {}
    if v9 and v9.get("status") == "ACTIVE":
        lines.extend(
            [
                "## Risk Review v9",
                "",
                "| Kill Rule | Tick Atomicity | Worst Gap | Forward Gate |",
                "| --- | --- | ---: | --- |",
                f"| {v9.get('recommended_kill')} | {v9.get('tick_atomicity_status')} | {v9.get('tick_worst_adverse_bps')} | {v9.get('forward_gate')} |",
                "",
            ]
        )
    v11 = report.get("forward_governance_v11") or {}
    if v11 and v11.get("status") == "ACTIVE":
        lines.extend(
            [
                "## Forward Governance v11",
                "",
                "| Forward Integrity | Valid Forward N | Operator Governance | Oversize vs Balanced | Execution Truth | Fee Tier |",
                "| --- | ---: | --- | ---: | --- | --- |",
                (
                    f"| {v11.get('forward_integrity')} | {v11.get('valid_forward_n')} | "
                    f"{v11.get('operator_governance')} | {v11.get('oversize_vs_balanced')} | "
                    f"{v11.get('execution_truth')} | {v11.get('fee_tier')} |"
                ),
                "",
            ]
        )
    mirror = report.get("v02_shadow_mirror") or {}
    if mirror and mirror.get("status") == "ACTIVE":
        lines.extend(
            [
                "## V0.2 Shadow Mirror",
                "",
                "| Protocol | Permission | Decision | Live Match | Rows | Recent N | Recent Sum | Recent Median | Recent T3R | Kill | Latest Signal |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
                (
                    f"| {mirror.get('protocol_id')} | {mirror.get('permission')} | "
                    f"{mirror.get('decision')} | {mirror.get('live_rule_match')} | "
                    f"{mirror.get('ledger_rows')} | {mirror.get('recent_n')} | "
                    f"{mirror.get('recent_sum_bps')} | {mirror.get('recent_median_bps')} | "
                    f"{mirror.get('recent_t3r_bps')} | {mirror.get('kill_triggered')} | "
                    f"{mirror.get('latest_signal_utc')} |"
                ),
                "",
            ]
        )
    h4_shadow = report.get("v02_h4_shadow") or {}
    if h4_shadow and h4_shadow.get("status") == "ACTIVE":
        lines.extend(
            [
                "## V0.2 H4 Shadow Management",
                "",
                "| Protocol | Decision | H2 Sum | H4 Sum | H4 T3R | Cross Policy Sum | SL150 Touches | Queue |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
                (
                    f"| {h4_shadow.get('protocol_id')} | {h4_shadow.get('decision')} | "
                    f"{h4_shadow.get('h2_sum_bps')} | {h4_shadow.get('h4_sum_bps')} | "
                    f"{h4_shadow.get('h4_t3r_bps')} | {h4_shadow.get('cross_policy_sum_bps')} | "
                    f"{h4_shadow.get('sl150_touch_count')} | {h4_shadow.get('queue_status')} |"
                ),
                "",
            ]
        )
    sizing = report.get("sizing_shadow_paper") or {}
    if sizing and sizing.get("status") == "SHADOW_PAPER_SIZING_ONLY_NO_ORDER":
        lines.extend(
            [
                "## Sizing Shadow Paper",
                "",
                "| Mode | N | Notional | Margin | Lev | Sum bps | PnL USDT | End Equity | Max DD % |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, key in (("CURRENT_ENV", "current_env"), ("BALANCED", "balanced"), ("SURVIVAL", "survival")):
            row = sizing.get(key) or {}
            lines.append(
                f"| {label} | {row.get('n')} | {row.get('notional_usdt')} | {row.get('margin_usdt')} | "
                f"{row.get('leverage')} | {row.get('sum_bps')} | {row.get('sum_pnl_usdt')} | "
                f"{row.get('ending_equity_usdt')} | {row.get('max_drawdown_pct_equity')} |"
            )
        lines.extend(["", "Sizing shadow is observation only; it does not place orders or change live size.", ""])
    return "\n".join(lines)


def resolve_recheck_json(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    matches = sorted(glob.glob(str(OUT_DIR / RECHECK_GLOB)), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(matches[0]) if matches else (OUT_DIR / "S34_KNOWABLE_ANCHOR_ROUTE_RECHECK.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 cascade navigation / permission dashboard (point-in-time, no lookahead).")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--as-of-ts-ms", type=int, default=None, help="Point-in-time cutoff; default = latest liquidation in DB.")
    p.add_argument("--as-of-utc", default="", help="ISO cutoff (overrides --as-of-ts-ms).")
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--dead-gap-sec", type=float, default=120.0)
    p.add_argument("--btc-window-sec", type=int, default=900)
    p.add_argument("--max-book-staleness-sec", type=int, default=5)
    p.add_argument("--recheck-json", default=None)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    p.add_argument("--md-out", type=Path, default=OUT_MD)
    p.add_argument("--quiet", action="store_true")
    # Canonical operator web UI (read-only, GET/HEAD only). Default invocation
    # remains the CLI report; --serve opts into the read-only web surface.
    p.add_argument("--serve", action="store_true",
                   help="Serve the read-only canonical operator dashboard (GET/HEAD only).")
    p.add_argument("--serve-host", default="127.0.0.1", help="Loopback host for --serve (default 127.0.0.1).")
    p.add_argument("--serve-port", type=int, default=8770, help="Port for --serve (default 8770).")
    return p.parse_args(argv)


def _run_serve_mode(args) -> int:
    """Dispatch to the read-only web operator dashboard. Imported lazily so the
    existing CLI report path has zero new dependencies."""
    from dashboard.backend.server import serve as _serve
    return _serve(host=args.serve_host, port=args.serve_port)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if getattr(args, "serve", False):
        # Web serve mode is explicitly separate from CLI report mode.
        return _run_serve_mode(args)
    confidence = load_rule_confidence(resolve_recheck_json(args.recheck_json))
    validated_registry = load_validated_signals()
    # default path -> union (spans live+frozen post-rotation); explicit --db override -> that file verbatim
    _db_conn = (open_union_ro() if str(args.db) == str(DEFAULT_DB)
                else sqlite3.connect(f"file:{args.db}?mode=ro", uri=True))
    with _db_conn as conn:
        if args.as_of_utc:
            text = args.as_of_utc.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            as_of_ts = int(datetime.fromisoformat(text).timestamp() * 1000)
        elif args.as_of_ts_ms is not None:
            as_of_ts = int(args.as_of_ts_ms)
        else:
            candidates = [latest_liq_ts(conn, lane.symbol) for lane in UNIVERSE]
            candidates = [c for c in candidates if c is not None]
            if not candidates:
                print("No liquidation data found.", file=sys.stderr)
                return 1
            as_of_ts = max(candidates)
        btc_marks = load_mark_index_range(
            conn, "BTCUSDT", as_of_ts - (int(args.btc_window_sec) + 120) * 1000, as_of_ts
        )
        lanes = [
            evaluate_lane(
                conn,
                lane,
                confidence,
                btc_marks,
                as_of_ts=as_of_ts,
                bucket_sec=int(args.bucket_sec),
                accel_window_sec=int(args.accel_window_sec),
                dead_gap_sec=float(args.dead_gap_sec),
                btc_window_sec=int(args.btc_window_sec),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
                validated_registry=validated_registry,
            )
            for lane in UNIVERSE
        ]
    report = {
        "generated_at_utc": utc_now(),
        "as_of_ts_ms": int(as_of_ts),
        "as_of_utc": iso_ms(as_of_ts),
        "execution_management": load_management_fragment(),
        "risk_review_v9": load_v9_review(),
        "forward_governance_v11": load_v11_governance(),
        "v02_shadow_mirror": load_v02_shadow_mirror(),
        "v02_h4_shadow": load_v02_h4_shadow(),
        "sizing_shadow_paper": load_sizing_shadow_paper(),
        "silence_gate": load_silence_gate_status(),
        "shadow_bucket": load_realtime_shadow_bucket(),
        "validated_registry": validated_registry,
        "execution_audit": load_execution_audit(),
        "execmgmt": load_execmgmt_panel(),
        "stopprot": load_stopprot_panel(),
        "config": {
            "bucket_sec": int(args.bucket_sec),
            "accel_window_sec": int(args.accel_window_sec),
            "dead_gap_sec": float(args.dead_gap_sec),
            "btc_window_sec": int(args.btc_window_sec),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "recheck_json": str(resolve_recheck_json(args.recheck_json)),
        },
        "lanes": lanes,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.md_out.write_text(render_md(report), encoding="utf-8")
    if not args.quiet:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
