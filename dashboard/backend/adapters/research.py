"""Panel 8 — Research & Strategy Readiness (read-only).

Research candidates are surfaced as evidence only and are NEVER presented as
execution-approved.
"""
from __future__ import annotations

from dashboard.backend.readers import load_json, parse_iso
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

REGISTRY_REL = ("reports", "research", "s34", "S34_VALIDATED_SIGNALS_REGISTRY.json")


def build(ctx: DashboardContext) -> PanelViewModel:
    path = ctx.p(*REGISTRY_REL)
    data, ps, mtime = load_json(path)

    vm = PanelViewModel(
        key="research",
        title="Research & Strategy Readiness",
        source_path=str(path),
        read_timestamp=ctx.now,
        source_timestamp=parse_iso(data.get("generated_utc")) if isinstance(data, dict) else mtime,
        freshness_threshold_seconds=ctx.threshold("research"),
        parse_status=ps,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if ps == ParseStatus.MISSING:
        vm.status = "MISSING"
        vm.trust_state = TrustState.MISSING
        vm.severity = Severity.INFO
        vm.summary = "validated signals registry absent (no actionable candidate surfaced)"
        vm.fields = {"no_actionable_candidate": True, "execution_approved": False}
        vm.add_reason("registry_missing")
        return vm
    if ps != ParseStatus.OK or not isinstance(data, dict):
        vm.status = "MALFORMED"
        vm.parse_status = ParseStatus.MALFORMED
        vm.trust_state = TrustState.MALFORMED
        vm.severity = Severity.MEDIUM
        vm.summary = "validated signals registry unparseable (fail-closed)"
        vm.add_reason("registry_malformed")
        return vm

    signals = data.get("signals")
    sig_list = signals if isinstance(signals, list) else (list(signals.values()) if isinstance(signals, dict) else [])
    armed = [s for s in sig_list if isinstance(s, dict) and str(s.get("status", "")).upper() in ("ARMED", "ELIGIBLE")]
    shadow = [s for s in sig_list if isinstance(s, dict) and str(s.get("status", "")).upper() in ("SHADOW", "SHADOW_ONLY", "OBSERVE")]

    vm.fields = {
        "registry_status": data.get("status"),
        "generated_utc": data.get("generated_utc"),
        "signal_count": len(sig_list),
        "shadow_only_count": len(shadow),
        "execution_eligible_count": len(armed),
        "no_actionable_candidate": len(armed) == 0,
        "execution_approved": False,  # research is never execution-approved here
        "fee_bps": data.get("fee_bps"),
        "holdout_period": data.get("holdout_period"),
        "canonical_evidence_path": str(path),
    }
    vm.rows = [
        {
            "name": (s.get("name") or s.get("rule_name") or s.get("id")),
            "status": s.get("status"),
            "shadow_only": str(s.get("status", "")).upper() not in ("ARMED", "ELIGIBLE"),
        }
        for s in sig_list if isinstance(s, dict)
    ][:20]
    vm.severity = Severity.OK
    vm.status = "ACTIVE"
    vm.summary = (
        f"registry={data.get('status')} signals={len(sig_list)} "
        f"shadow_only={len(shadow)} eligible={len(armed)} (research evidence only, NOT execution-approved)"
    )
    return vm
