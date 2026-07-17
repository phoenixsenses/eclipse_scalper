"""Panel 9 — Governance (read-only, bounded SYSTEM_STATE.md + git plumbing)."""
from __future__ import annotations

from dashboard.backend.canonical_state import (
    extract_open_findings,
    read_canonical_state,
    read_git_identity,
)
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

# Known foreign-owned tracked paths preserved in the source working copy. Shown
# for operator awareness; provenance is INFERRED (not computed via git here).
KNOWN_FOREIGN_OWNED = (
    ".claude/settings.local.json",
    "runtime/dashboard_backend.json",
    "tests/test_native_ws_health_policy.py",
    "tools/native_ws_health_policy.py",
)


def build(ctx: DashboardContext) -> PanelViewModel:
    state = read_canonical_state(ctx.system_state)
    findings = extract_open_findings(ctx.system_state)
    git = read_git_identity(ctx.repo_root)

    ps = ParseStatus(state.get("parse_status", "malformed")) if state.get("parse_status") in (
        "ok", "missing", "malformed", "empty") else ParseStatus.MALFORMED

    vm = PanelViewModel(
        key="governance",
        title="Governance",
        source_path=state.get("source_path"),
        read_timestamp=state.get("read_timestamp"),
        source_timestamp=state.get("source_timestamp"),
        freshness_threshold_seconds=ctx.threshold("governance"),
        parse_status=ps,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if ps == ParseStatus.MISSING:
        vm.status = "MISSING"
        vm.severity = Severity.MEDIUM
        vm.trust_state = TrustState.MISSING
        vm.summary = "SYSTEM_STATE.md not found (fail-closed)"
        vm.add_reason("system_state_missing")
        return vm

    counts = findings["counts"]
    open_high = counts.get("CRITICAL", 0) + counts.get("HIGH", 0)

    vm.fields = {
        "canonical_state": state.get("canonical_state"),
        "latest_section_number": state.get("latest_section_number"),
        "latest_section_title": state.get("latest_section_title"),
        "findings_counts": counts,
        "branch": git.get("branch"),
        "head_short": git.get("head_short"),
        "detached": git.get("detached"),
        "ahead_behind": git.get("ahead_behind"),  # None => not computed w/o git binary
        "ahead_behind_provenance": "inferred_unavailable_no_subprocess",
        "known_foreign_owned": list(KNOWN_FOREIGN_OWNED),
        "dirty_partition_provenance": "inferred_not_computed_without_git_plumbing",
        "tail_truncated": state.get("tail_truncated"),
    }
    vm.rows = findings["findings"]
    vm.raw_evidence = {"git_parse_status": git.get("parse_status")}

    if open_high > 0:
        vm.severity = Severity.HIGH
        vm.add_reason("open_high_or_critical_finding")
    elif counts.get("MEDIUM", 0) > 0:
        vm.severity = Severity.MEDIUM
        vm.add_reason("open_medium_finding")
    else:
        vm.severity = Severity.OK

    vm.status = "ACTIVE"
    vm.summary = (
        f"§{state.get('latest_section_number')} · {state.get('canonical_state') or 'state?'} · "
        f"branch={git.get('branch')}@{git.get('head_short')} · "
        f"findings C{counts.get('CRITICAL',0)}/H{counts.get('HIGH',0)}/M{counts.get('MEDIUM',0)}/L{counts.get('LOW',0)}"
    )
    return vm
