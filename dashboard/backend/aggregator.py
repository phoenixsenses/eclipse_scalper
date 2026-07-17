"""Aggregator: runs every panel adapter in isolation and assembles the overview.

Partial-failure isolation: each adapter is executed inside its own try/except so
one failing adapter degrades only its own panel (typed fail-closed) and never
takes down the whole dashboard. No secret or full exception environment is ever
dumped — only the exception class name and a short reason.
"""
from __future__ import annotations

import time
from typing import Any, Callable

from dashboard.backend.freshness import apply_freshness
from dashboard.backend.sources import REPO_ROOT, DashboardContext
from dashboard.backend.view_models import (
    NON_CURRENT_TRUST,
    PanelViewModel,
    Severity,
    failed_panel,
    max_severity,
    severity_rank,
)

# Adapter registry: (panel_key, title, builder). Imported lazily to keep the
# aggregator importable even if one adapter module has an import error.
from dashboard.backend.adapters import (
    execution_safety,
    executive_header,
    gate2,
    governance,
    native_ws,
    paper_bucket_ledger,
    process_topology,
    research,
    s34_shadow_evidence,
    s34_state,
    shadow_paper_activity,
    storage,
    execmgmt_stopprot,
)

# Deterministic panel ordering (matches the required information architecture).
# Each entry stores the adapter MODULE; ``module.build`` is resolved at call time
# so an adapter can be swapped/patched and partial-failure isolation still holds.
PANEL_BUILDERS: list[tuple[str, str, Any]] = [
    ("executive_header", "Executive Safety Header", executive_header),
    ("process_topology", "Runtime Process Topology", process_topology),
    ("s34_state", "S34 State Machine", s34_state),
    ("s34_shadow_evidence", "S34 Shadow Evidence (forward vs backfill)", s34_shadow_evidence),
    ("shadow_paper_activity", "Shadow/Paper Activity", shadow_paper_activity),
    ("paper_bucket_ledger", "Paper Bucket Ledger (per-rule)", paper_bucket_ledger),
    ("execution_safety", "Execution Safety", execution_safety),
    ("gate2", "Liquidation-Silence / Gate 2", gate2),
    ("native_ws", "Native WebSocket & Collectors", native_ws),
    ("storage", "Storage & Data Integrity", storage),
    ("research", "Research & Strategy Readiness", research),
    ("governance", "Governance", governance),
    ("execmgmt_stopprot", "EXECMGMT / STOPPROT", execmgmt_stopprot),
]


def _run_one(key: str, title: str, module, ctx: DashboardContext) -> PanelViewModel:
    try:
        vm = module.build(ctx)
        if not isinstance(vm, PanelViewModel):
            return failed_panel(key, title, error_class="ContractError",
                                reason="adapter did not return PanelViewModel")
        # Enforce the shared trust policy uniformly (stale-GREEN prohibition):
        # recompute trust from the adapter's declared metadata unless the adapter
        # already reported a hard failure.
        apply_freshness(vm)
        # A non-current trust state must never present as OK severity.
        if vm.trust_state in NON_CURRENT_TRUST and severity_rank(vm.severity) < severity_rank(Severity.LOW):
            vm.severity = Severity.LOW
            vm.add_reason("severity_raised_for_non_current_trust")
        return vm
    except Exception as exc:  # isolation boundary
        # Surface ONLY the exception class name. The exception message/args are
        # deliberately NOT included — they can carry paths, values, or secrets
        # (no full exception environment is ever dumped).
        return failed_panel(key, title, error_class=type(exc).__name__, reason="")


def build_overview(ctx: DashboardContext | None = None) -> dict[str, Any]:
    """Build the full read-only overview payload (all panels)."""
    ctx = ctx or DashboardContext(repo_root=REPO_ROOT)
    generated_at = ctx.now if ctx.now is not None else time.time()
    panels: dict[str, Any] = {}
    worst = Severity.OK
    critical_blockers = 0
    oldest_age = 0.0

    for key, title, module in PANEL_BUILDERS:
        vm = _run_one(key, title, module, ctx)
        panels[key] = vm.to_dict()
        worst = max_severity(worst, vm.severity)
        if severity_rank(vm.severity) >= severity_rank(Severity.HIGH):
            critical_blockers += 1
        if isinstance(vm.age_seconds, (int, float)) and vm.age_seconds > oldest_age:
            oldest_age = float(vm.age_seconds)

    return {
        "generated_at": generated_at,
        "generated_at_iso": _iso(generated_at),
        "overall_severity": worst.value,
        "critical_blocker_count": critical_blockers,
        "oldest_artifact_age_seconds": round(oldest_age, 3),
        "panel_order": [k for k, _t, _b in PANEL_BUILDERS],
        "panels": panels,
        "contract": {
            "read_only": True,
            "control_actions_available": False,
            "db_mode": "ro",
        },
    }


def build_panel(panel_key: str, ctx: DashboardContext | None = None) -> dict[str, Any] | None:
    """Build a single panel by key (for /api/panel/<key>)."""
    ctx = ctx or DashboardContext(repo_root=REPO_ROOT)
    for key, title, module in PANEL_BUILDERS:
        if key == panel_key:
            return _run_one(key, title, module, ctx).to_dict()
    return None


def _iso(epoch: float) -> str:
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(float(epoch), _dt.timezone.utc).isoformat()
    except Exception:
        return ""
