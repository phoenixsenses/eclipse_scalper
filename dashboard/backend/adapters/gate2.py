"""Panel 5 — Liquidation-Silence / Gate 2 (read-only)."""
from __future__ import annotations

from dashboard.backend.process_identity import scan_topology
from dashboard.backend.readers import load_json, parse_iso, stat_mtime
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

# Candidate detector artifacts (first existing wins). All read-only, fail-closed.
DETECTOR_CANDIDATES = (
    ("runtime", "liquidation_silence_detector.json"),
    ("runtime", "s34_liquidation_silence.json"),
    ("reports", "research", "s34", "LIQUIDATION_SILENCE_DETECTOR_STATE.json"),
)


def _first_existing(ctx: DashboardContext):
    for rel in DETECTOR_CANDIDATES:
        p = ctx.p(*rel)
        if p.exists():
            return p
    return None


def build(ctx: DashboardContext) -> PanelViewModel:
    topo = scan_topology(ctx.proc_snapshot)
    scheduler_count = topo["scheduler_count"]
    path = _first_existing(ctx)

    vm = PanelViewModel(
        key="gate2",
        title="Liquidation-Silence / Gate 2",
        source_path=str(path) if path else "runtime/liquidation_silence_detector.json",
        read_timestamp=ctx.now,
        freshness_threshold_seconds=ctx.threshold("gate2_detector"),
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if path is None:
        vm.status = "NOT_ACTIVATED"
        vm.parse_status = ParseStatus.MISSING
        vm.trust_state = TrustState.MISSING
        vm.severity = Severity.INFO
        vm.fields = {
            "detector_artifact": "absent",
            "scheduler_process_count": scheduler_count,
            "accepted_but_not_activated": True,
            "continuity_status": "MISSING",
            "literal_launcher_validation": "PENDING (F-LAUNCHER-STOP-01)",
            "scheduled_task_status": "unknown",
            "permanent_activation_authorization": "not_recorded",
        }
        vm.summary = (
            "Gate 2 detector not activated (no runtime artifact); "
            f"scheduler_process_count={scheduler_count}; ACCEPTED_BUT_NOT_ACTIVATED"
        )
        vm.add_reason("detector_artifact_absent")
        vm.add_reason("accepted_but_not_activated")
        return vm

    data, ps, mtime = load_json(path)
    if ps != ParseStatus.OK or not isinstance(data, dict):
        vm.status = "MALFORMED"
        vm.parse_status = ParseStatus.MALFORMED
        vm.trust_state = TrustState.MALFORMED
        vm.severity = Severity.MEDIUM
        vm.summary = "Gate 2 detector artifact unparseable (fail-closed)"
        vm.add_reason("detector_artifact_malformed")
        return vm

    evaluated_at = parse_iso(data.get("evaluated_at") or data.get("evaluated_utc"))
    vm.source_timestamp = evaluated_at or mtime
    continuity = str(data.get("continuity") or data.get("continuity_status") or "UNKNOWN")
    vm.fields = {
        "detector_health": data.get("health") or data.get("status"),
        "evaluated_at": data.get("evaluated_at") or data.get("evaluated_utc"),
        "tracked_symbols": data.get("tracked_symbols") or data.get("symbols"),
        "scheduler_process_count": scheduler_count,
        "scheduler_pid": topo.get("roles") and next(
            (r["pids"] for r in topo["roles"] if r["key"] == "liquidation_silence_scheduler"), []),
        "bom_status": data.get("bom_status"),
        "provenance_status": data.get("provenance"),
        "continuity_status": continuity if continuity in ("CONTINUOUS", "REPLACED_OR_RESTARTED", "MISSING") else "UNKNOWN",
        "last_successful_cycle": data.get("last_successful_cycle"),
        "cadence_summary": data.get("cadence"),
        "literal_launcher_validation": "PENDING (F-LAUNCHER-STOP-01)",
        "scheduled_task_status": data.get("scheduled_task_status", "unknown"),
        "permanent_activation_authorization": data.get("activation_authorization", "not_recorded"),
    }
    if scheduler_count == 0:
        vm.severity = Severity.MEDIUM
        vm.add_reason("scheduler_not_running")
    else:
        vm.severity = Severity.OK
    vm.status = "ACTIVE"
    vm.summary = (
        f"detector={vm.fields['detector_health']} continuity={vm.fields['continuity_status']} "
        f"scheduler={scheduler_count} launcher_validation=PENDING"
    )
    return vm
