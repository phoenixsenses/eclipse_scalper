"""Panel 1 — Executive Safety Header (always-visible, sticky).

Aggregates the highest-signal safety facts: live-executor/scheduler posture,
branch/HEAD, canonical governance state, watchdog, oldest artifact age. When the
live executor is observed running, the header is NOT rendered GREEN — it carries
an explicit safety warning severity.
"""
from __future__ import annotations

import time

from dashboard.backend.canonical_state import extract_open_findings, read_canonical_state, read_git_identity
from dashboard.backend.process_identity import scan_topology
from dashboard.backend.readers import load_env_numeric
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

POSTURE_KEYS = ("S34_LIVE_TRADING_ENABLED", "SCALPER_DRY_RUN")


def build(ctx: DashboardContext) -> PanelViewModel:
    read_ts = ctx.now if ctx.now is not None else time.time()
    topo = scan_topology(ctx.proc_snapshot)
    state = read_canonical_state(ctx.system_state)
    git = read_git_identity(ctx.repo_root)
    findings = extract_open_findings(ctx.system_state)["counts"]
    env_numeric = load_env_numeric(ctx.env_path, POSTURE_KEYS)

    live_exec = topo["live_executor_count"]
    scheduler = topo["scheduler_count"]
    watchdog = topo["watchdog_pids"]
    open_high = findings.get("CRITICAL", 0) + findings.get("HIGH", 0) + findings.get("MEDIUM", 0)

    vm = PanelViewModel(
        key="executive_header",
        title="Executive Safety Header",
        source_path="aggregate:process+git+system_state",
        read_timestamp=read_ts,
        source_timestamp=read_ts,
        freshness_threshold_seconds=ctx.threshold("process_topology"),
        provenance_status=ProvenanceStatus.DERIVED,
    )

    # Severity: live executor ON is the dominant safety escalation.
    if live_exec > 0:
        sev = Severity.HIGH
        vm.add_reason("LIVE_EXECUTOR_RUNNING")
    elif open_high > 0:
        sev = Severity.MEDIUM
        vm.add_reason("open_findings_present")
    else:
        sev = Severity.OK

    if not topo.get("psutil_available", False) and ctx.proc_snapshot is None:
        vm.trust_state = TrustState.INFERRED
        vm.add_reason("process_scan_unavailable")

    vm.fields = {
        "mode": "READ_ONLY_OPERATOR_DASHBOARD",
        "live_executor": "ON" if live_exec > 0 else "OFF",
        "live_executor_expected": "OFF",
        "live_executor_ok": live_exec == 0,
        "scheduler_state": "RUNNING" if scheduler > 0 else "STOPPED",
        "shadow_paper_mode": True,
        "branch": git.get("branch"),
        "head_short": git.get("head_short"),
        "canonical_governance_state": state.get("canonical_state"),
        "latest_section": state.get("latest_section_number"),
        "watchdog_pids": watchdog,
        "watchdog_ok": len(watchdog) >= 1,
        "last_refresh_epoch": read_ts,
        "open_finding_counts": findings,
        "critical_high_medium_open": open_high,
        "env_posture_numeric": env_numeric,
        "control_actions_available": False,
    }
    vm.severity = sev
    vm.status = "ACTIVE"
    vm.summary = (
        f"{'⚠ LIVE EXECUTOR ON ' if live_exec > 0 else 'live_exec=OFF '}"
        f"scheduler={'ON' if scheduler > 0 else 'OFF'} "
        f"watchdog={'ok' if watchdog else 'MISSING'} "
        f"{git.get('branch')}@{git.get('head_short')} "
        f"open(C/H/M/L)={findings.get('CRITICAL',0)}/{findings.get('HIGH',0)}/{findings.get('MEDIUM',0)}/{findings.get('LOW',0)}"
    )
    return vm
