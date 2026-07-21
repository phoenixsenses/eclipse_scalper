"""Panel — Liquidation / Stream Anomaly Monitor (read-only).

Surfaces the situational anomaly flags produced by ``tools/liq_anomaly_monitor.py``
(funding / OI / basis / vol / liquidation / cross-asset / book). These are
situational-awareness flags ONLY — NOT predictions and NOT an execution edge
(this microstructure has no fee-surviving edge; CV-AUC~0.55). Read straight from
the bounded JSON artifact; no DB access, no control actions.
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

ARTIFACT_REL = ("reports", "research", "s34", "LIQ_ANOMALY_MONITOR.json")

_SEV_MAP = {
    "ok": Severity.OK,
    "info": Severity.INFO,
    "low": Severity.LOW,
    "medium": Severity.MEDIUM,
    "high": Severity.HIGH,
    "critical": Severity.CRITICAL,
}


def build(ctx: DashboardContext) -> PanelViewModel:
    path = ctx.p(*ARTIFACT_REL)
    data, ps, mtime = load_json(path)

    vm = PanelViewModel(
        key="liq_anomaly",
        title="Liquidation / Stream Anomaly Monitor",
        source_path=str(path),
        read_timestamp=ctx.now,
        source_timestamp=parse_iso(data.get("generated_utc")) if isinstance(data, dict) else mtime,
        freshness_threshold_seconds=ctx.threshold("liq_anomaly"),
        parse_status=ps,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if ps == ParseStatus.MISSING:
        vm.status = "MISSING"
        vm.trust_state = TrustState.MISSING
        vm.severity = Severity.INFO
        vm.summary = "anomaly monitor artifact absent (liq_anomaly_monitor not running)"
        vm.add_reason("artifact_missing")
        return vm
    if ps != ParseStatus.OK or not isinstance(data, dict):
        vm.status = "MALFORMED"
        vm.parse_status = ParseStatus.MALFORMED
        vm.trust_state = TrustState.MALFORMED
        vm.severity = Severity.MEDIUM
        vm.summary = "anomaly monitor artifact unparseable (fail-closed)"
        vm.add_reason("artifact_malformed")
        return vm

    flags = data.get("flags") if isinstance(data.get("flags"), list) else []
    overall = str(data.get("overall_severity", "ok")).lower()
    active = [f for f in flags if isinstance(f, dict) and str(f.get("severity", "ok")).lower() not in ("ok", "info")]
    stale = [f for f in flags if isinstance(f, dict) and not f.get("fresh", True)]

    vm.fields = {
        "as_of_utc": data.get("as_of_utc"),
        "symbol": data.get("symbol"),
        "overall_severity": overall,
        "flag_count": len(flags),
        "active_anomaly_count": len(active),
        "stale_source_count": len(stale),
        "predictor": False,  # situational only — NOT an edge (AUC~0.55)
        "kind": data.get("kind"),
    }
    vm.rows = [
        {
            "name": f.get("name"),
            "value": f.get("value"),
            "severity": f.get("severity"),
            "fresh": f.get("fresh", True),
            "note": f.get("note", ""),
        }
        for f in flags if isinstance(f, dict)
    ][:30]
    vm.severity = _SEV_MAP.get(overall, Severity.INFO)
    vm.status = "ACTIVE"
    active_names = ", ".join(str(f.get("name")) for f in active[:5]) or "none"
    vm.summary = (
        f"overall={overall} active={len(active)}/{len(flags)} "
        f"({active_names}){' stale_srcs=' + str(len(stale)) if stale else ''} — situational only, NOT a predictor"
    )
    if stale:
        vm.add_reason("some_sources_stale")
    return vm
