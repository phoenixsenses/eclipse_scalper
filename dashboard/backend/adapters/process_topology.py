"""Panel 2 — Runtime Process Topology (read-only)."""
from __future__ import annotations

import time

from dashboard.backend.process_identity import scan_topology
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)


def build(ctx: DashboardContext) -> PanelViewModel:
    topo = scan_topology(ctx.proc_snapshot)
    read_ts = ctx.now if ctx.now is not None else topo["read_timestamp"]

    vm = PanelViewModel(
        key="process_topology",
        title="Runtime Process Topology",
        source_path="psutil:live",
        read_timestamp=read_ts,
        source_timestamp=read_ts,
        freshness_threshold_seconds=ctx.threshold("process_topology"),
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if not topo.get("psutil_available", False) and ctx.proc_snapshot is None:
        vm.status = "DEGRADED"
        vm.parse_status = ParseStatus.MISSING
        vm.trust_state = TrustState.MISSING
        vm.severity = Severity.MEDIUM
        vm.summary = "psutil unavailable — process topology cannot be observed (fail-closed)"
        vm.add_reason("psutil_unavailable")
        return vm

    roles = topo["roles"]
    vm.rows = roles
    missing = [r for r in roles if r["health"] == "MISSING"]
    duplicate = [r for r in roles if r["health"] == "DUPLICATE"]
    unexpected_live = [r for r in roles if r["health"] == "RUNNING_UNEXPECTED"]
    orphans = topo.get("orphans", [])

    vm.fields = {
        "scheduler_count": topo["scheduler_count"],
        "live_executor_count": topo["live_executor_count"],
        "watchdog_pids": topo["watchdog_pids"],
        "total_python_procs": topo["total_python_procs"],
        "missing_roles": [r["key"] for r in missing],
        "duplicate_roles": [r["key"] for r in duplicate],
        "unexpected_live_roles": [r["key"] for r in unexpected_live],
        "orphan_count": len(orphans),
    }
    vm.raw_evidence = {"orphans": orphans[:20]}

    sev = Severity.OK
    if unexpected_live:
        sev = Severity.HIGH
        vm.add_reason("live_executor_or_off_role_running")
    if duplicate:
        sev = Severity.HIGH if sev == Severity.OK else sev
        vm.add_reason("duplicate_role")
    if missing:
        sev = Severity.MEDIUM if sev == Severity.OK else sev
        vm.add_reason("missing_expected_role")
    if orphans:
        vm.add_reason("orphan_eclipse_process")

    vm.severity = sev
    vm.status = "ACTIVE"
    vm.summary = (
        f"{len([r for r in roles if r['actual_count'] > 0])} roles running; "
        f"scheduler={topo['scheduler_count']} live_executor={topo['live_executor_count']} "
        f"missing={len(missing)} duplicate={len(duplicate)} orphans={len(orphans)}"
    )
    return vm
