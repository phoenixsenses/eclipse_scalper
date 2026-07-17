"""Panel — EXECMGMT / STOPPROT (read-only), self-contained port of the
already-accepted panel logic (commit 051468c0) plus F-DASH-01 closure.

The computation mirrors ``tools/s34_cascade_navigation_dashboard.py``'s accepted
``load_execmgmt_panel`` / ``load_stopprot_panel`` exactly, but is reimplemented
here so the dashboard package stays self-contained (the canonical entrypoint
module has an unguarded dependency on an untracked research script and cannot be
imported in a clean checkout). Only numeric sizing/leverage/stop keys are read
from ``.env`` — never secrets.

F-DASH-01: ``-175.7 bps`` worst-real-fill is explicitly labelled as
*Historical reference / research worst-fill evidence* with source path +
timestamp, and as a *fallback historical reference* when the live audit artifact
is absent, so it can never be mistaken for active config / a current runtime limit.
"""
from __future__ import annotations

from typing import Any

from dashboard.backend.readers import load_env_numeric, load_json
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

AUDIT_REL = ("reports", "research", "s34", "S34_V_ENGINE_EXECUTION_MANAGEMENT_AUDIT.json")
SIZING_REL = ("reports", "research", "s34", "S34_V_ENGINE_SIZING_SHADOW_PAPER.json")

# Accepted constants (identical to the committed canonical panel).
DEFAULT_EQUITY_USDT = 35.0
TAIL_BUDGET_PCT = 1.0
WORST_REAL_FILL_BPS = -175.7          # historical research worst-fill evidence
DEFAULT_LEVERAGE = 40.0
WORST_FILL_DEFINITION = "Historical reference / research worst-fill evidence (bps, negative = loss)"

_LEVERAGE_KEYS = ("S34_LIVE_MAX_LEVERAGE", "LEVERAGE")
_MARGIN_KEYS = ("ORDER_MARGIN_USD", "S34_LIVE_MARGIN_USDT", "FIXED_MARGIN_USDT")
_NOTIONAL_KEYS = ("ORDER_NOTIONAL_USD", "ORDER_NOTIONAL_USDT", "S34_LIVE_ORDER_NOTIONAL_USD",
                  "S34_LIVE_NOTIONAL_USDT", "FIXED_NOTIONAL_USDT")
_EQUITY_KEYS = ("ORDER_EQUITY_USD", "ACCOUNT_EQUITY_USD", "ACCOUNT_EQUITY_USDT",
                "S34_ACCOUNT_EQUITY_USDT", "S34_LIVE_EQUITY_USDT")
_MARGIN_PCT_KEYS = ("S34_LIVE_MARGIN_PCT_ETH", "S34_LIVE_MARGIN_PCT")
_STOP_KEYS = ("S34_V_ENGINE_LIVE_STOP_BPS", "S34_V_ENGINE_EMERGENCY_STOP_BPS", "STOP_BPS")

_ALL_NUMERIC = tuple(dict.fromkeys(
    _LEVERAGE_KEYS + _MARGIN_KEYS + _NOTIONAL_KEYS + _EQUITY_KEYS + _MARGIN_PCT_KEYS + _STOP_KEYS
))


def _first(env: dict[str, float], keys, default):
    """Return (value, source_key). ``default`` must already be a (value, label)
    tuple; it is returned as-is when no key matches."""
    for k in keys:
        if k in env:
            return env[k], k
    return default


def _num(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _compute(env: dict[str, float], sizing: dict, audit: dict) -> tuple[dict, dict, bool]:
    audit_env = audit.get("live_env") if isinstance(audit.get("live_env"), dict) else {}
    leverage, lev_src = _first(env, _LEVERAGE_KEYS, (_num(audit_env.get("leverage"), DEFAULT_LEVERAGE), "audit_or_default"))
    leverage = float(leverage or DEFAULT_LEVERAGE)
    margin, margin_src = _first(env, _MARGIN_KEYS, (_num(audit_env.get("margin_usdt")), "audit"))
    notional, notional_src = _first(env, _NOTIONAL_KEYS, (None, None))
    if notional is None and margin is not None:
        notional = float(margin) * leverage
        notional_src = f"{margin_src or 'margin'}*leverage"
    if notional is None:
        notional = _num(audit_env.get("notional_usdt"), 0.0) or 0.0
        notional_src = "execution_audit.live_env.notional_usdt"
    if margin is None and notional is not None and leverage > 0:
        margin = float(notional) / leverage
        margin_src = "notional/leverage"
    margin_pct, _ = _first(env, _MARGIN_PCT_KEYS, (85.0, "default"))
    margin_pct = float(margin_pct or 85.0)
    equity_default = _num(sizing.get("equity_assumption_usdt"), DEFAULT_EQUITY_USDT)
    equity, equity_src = _first(env, _EQUITY_KEYS, (None, None))
    if equity is None and notional and leverage > 0 and margin_pct > 0:
        equity = float(notional) / leverage / (margin_pct / 100.0)
        equity_src = f"derived_notional/lev/margin_pct({margin_pct:.0f}%)"
    if equity is None:
        equity, equity_src = equity_default, "static_default"
    equity = float(equity or DEFAULT_EQUITY_USDT)

    tail_budget_usdt = equity * TAIL_BUDGET_PCT / 100.0
    worst_abs = abs(WORST_REAL_FILL_BPS)
    tail_notional = tail_budget_usdt / (worst_abs / 10_000.0) if worst_abs > 0 else 0.0
    oversize = float(notional) / tail_notional if tail_notional > 0 else None
    rec_margin = tail_notional / leverage if leverage > 0 else None
    urgent = bool(oversize is not None and oversize > 10.0)
    em = {
        "status": "ACTIVE", "mode": "READ_ONLY_RECOMMENDATION_NO_ACTION",
        "equity_usdt": round(equity, 4), "equity_source": equity_src or "sizing_shadow_default",
        "tail_budget_pct": TAIL_BUDGET_PCT, "tail_budget_usdt": round(tail_budget_usdt, 4),
        "worst_real_fill_bps": WORST_REAL_FILL_BPS,
        "tail_budget_notional_usdt": round(tail_notional, 4),
        "current_notional_usdt": round(float(notional or 0.0), 4), "current_notional_source": notional_src,
        "current_margin_usdt": round(float(margin or 0.0), 4), "current_margin_source": margin_src,
        "leverage": round(leverage, 4), "leverage_source": lev_src or "default_or_audit",
        "oversize_multiple": round(oversize, 1) if oversize is not None else None,
        "urgent": urgent, "flag": "URGENT_OVERSIZE_GT_10X" if urgent else "OK",
        "recommended_max_margin_usdt": round(rec_margin, 4) if rec_margin is not None else None,
    }

    gap = audit.get("gap_through") if isinstance(audit.get("gap_through"), dict) else {}
    atom = audit.get("atomicity_audit") if isinstance(audit.get("atomicity_audit"), dict) else {}
    nominal, nominal_src = _first(env, _STOP_KEYS, (_num(gap.get("current_stop_nominal_bps"), 150.0), "audit_or_default"))
    nominal = float(nominal or 150.0)
    audit_worst = gap.get("current_stop_research_max_loss_bps")
    worst_fill = _num(audit_worst, WORST_REAL_FILL_BPS)
    worst_fill = float(worst_fill or WORST_REAL_FILL_BPS)
    gap_bps = max(0.0, abs(worst_fill) - abs(nominal))
    poll_sec = _num(atom.get("poll_sec"), 2.0)
    sp = {
        "status": "ACTIVE", "mode": "READ_ONLY_WARNING_NO_ACTION",
        "nominal_stop_bps": round(nominal, 4), "nominal_stop_source": nominal_src or "audit_or_default",
        "worst_real_fill_bps": round(worst_fill, 4), "gap_through_bps": round(gap_bps, 4),
        "gap_through_flag": "PARTIAL_PROTECTION" if abs(worst_fill) > abs(nominal) else "FULL_WITHIN_AUDIT_SAMPLE",
        "atomicity": "ENTRY_THEN_STOP_POLL_LOOP", "unprotected_window_sec": round(float(poll_sec or 2.0), 4),
    }
    worst_from_fallback = audit_worst is None
    return em, sp, worst_from_fallback


def build(ctx: DashboardContext) -> PanelViewModel:
    audit_path = ctx.p(*AUDIT_REL)
    sizing_path = ctx.p(*SIZING_REL)
    env = load_env_numeric(ctx.env_path, _ALL_NUMERIC)
    audit_data, audit_ps, audit_mtime = load_json(audit_path)
    sizing_data, _sps, _smt = load_json(sizing_path)
    audit = audit_data if isinstance(audit_data, dict) else {}
    sizing = sizing_data if isinstance(sizing_data, dict) else {}

    em, sp, worst_from_fallback = _compute(env, sizing, audit)

    vm = PanelViewModel(
        key="execmgmt_stopprot",
        title="EXECMGMT / STOPPROT",
        source_path=str(audit_path),
        read_timestamp=ctx.now,
        source_timestamp=audit_mtime,
        freshness_threshold_seconds=ctx.threshold("execution_state"),
        parse_status=audit_ps if audit_ps == ParseStatus.MISSING else ParseStatus.OK,
        provenance_status=ProvenanceStatus.HISTORICAL if worst_from_fallback else ProvenanceStatus.DIRECT,
    )
    worst_label = ("fallback historical reference (audit artifact absent)"
                   if worst_from_fallback else "research worst-fill from execution-management audit")
    vm.fields = {
        "execmgmt": em,
        "stopprot": sp,
        "worst_real_fill_bps_value": sp["worst_real_fill_bps"],
        "worst_real_fill_bps_definition": WORST_FILL_DEFINITION,
        "worst_real_fill_bps_provenance": worst_label,
        "worst_real_fill_bps_source_path": str(audit_path),
        "worst_real_fill_bps_source_timestamp": audit_mtime,
        "worst_real_fill_bps_is_active_config": False,
        "note": "read-only recommendation/warning; no order/config/env mutation",
    }
    vm.add_reason("worst_fill_is_historical_reference_not_active_config")
    if worst_from_fallback:
        vm.add_reason("worst_fill_fallback_historical_reference")
        vm.trust_state = TrustState.INFERRED
    urgent = bool(em.get("urgent"))
    vm.severity = Severity.MEDIUM if urgent else Severity.OK
    if urgent:
        vm.add_reason("execmgmt_oversize_urgent")
    vm.status = "ACTIVE"
    vm.summary = (
        f"EXECMGMT oversize={em.get('oversize_multiple')}x flag={em.get('flag')} · "
        f"STOPPROT worst_fill={sp['worst_real_fill_bps']}bps "
        f"({'fallback ' if worst_from_fallback else ''}historical) gap={sp['gap_through_bps']}bps"
    )
    return vm
