"""Panel 6 — Native WebSocket & Collectors (read-only).

IMPORTANT: this adapter reads ONLY runtime health artifacts. It deliberately does
NOT import, copy, cherry-pick, or otherwise depend on the separately-owned dirty
``tools/native_ws_health_policy.py`` package. Any policy semantics referenced are
derived from committed HEAD behaviour and the runtime artifact itself; the panel
surfaces the real observed values and labels the policy provenance explicitly.
"""
from __future__ import annotations

from dashboard.backend.process_identity import scan_topology
from dashboard.backend.readers import load_json, parse_iso
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

WS_HEALTH_CANDIDATES = (
    ("runtime", "native_ws_health.json"),
    ("reports", "native_ws_health.json"),
    ("logs", "native_ws_health.json"),
)


def _first_existing(ctx: DashboardContext):
    for rel in WS_HEALTH_CANDIDATES:
        p = ctx.p(*rel)
        if p.exists():
            return p
    return None


def build(ctx: DashboardContext) -> PanelViewModel:
    topo = scan_topology(ctx.proc_snapshot)
    collector_roles = [r for r in topo["roles"] if "collector" in r["key"] or r["key"] == "collector_supervisor"]
    path = _first_existing(ctx)

    vm = PanelViewModel(
        key="native_ws",
        title="Native WebSocket & Collectors",
        source_path=str(path) if path else "runtime/native_ws_health.json",
        read_timestamp=ctx.now,
        freshness_threshold_seconds=ctx.threshold("native_ws"),
        provenance_status=ProvenanceStatus.DIRECT,
    )
    vm.fields = {
        "policy_provenance": "committed_HEAD_behaviour (dirty native_ws_health_policy package NOT absorbed)",
        "collector_roles": [
            {"key": r["key"], "actual": r["actual_count"], "health": r["health"], "duplicate": r["duplicate_count"]}
            for r in collector_roles
        ],
    }

    if path is None:
        vm.status = "NO_ARTIFACT"
        vm.parse_status = ParseStatus.MISSING
        vm.trust_state = TrustState.MISSING
        # Collectors may still be up (process topology) even without a health json.
        running = [r for r in collector_roles if r["actual_count"] > 0]
        vm.severity = Severity.INFO if running else Severity.MEDIUM
        vm.summary = (
            f"native-ws health artifact absent (fail-closed); "
            f"{len(running)}/{len(collector_roles)} collector roles running (from process scan)"
        )
        vm.add_reason("ws_health_artifact_absent")
        return vm

    data, ps, mtime = load_json(path)
    if ps != ParseStatus.OK or not isinstance(data, dict):
        vm.status = "MALFORMED"
        vm.parse_status = ParseStatus.MALFORMED
        vm.trust_state = TrustState.MALFORMED
        vm.severity = Severity.MEDIUM
        vm.summary = "native-ws health artifact unparseable (fail-closed)"
        vm.add_reason("ws_health_artifact_malformed")
        return vm

    last_msg = parse_iso(data.get("last_message_utc") or data.get("last_message_ts"))
    vm.source_timestamp = last_msg or mtime
    vm.fields.update({
        "connected": data.get("connected"),
        "last_message_utc": data.get("last_message_utc"),
        "message_age_sec": data.get("message_age_sec"),
        "reconnect_state": data.get("reconnect_state"),
        "backoff_sec": data.get("backoff_sec"),
        "last_error": (str(data.get("last_error"))[:200] if data.get("last_error") else None),
        "fallback_state": data.get("fallback_state"),
        "rest_fallback_active": data.get("rest_fallback_active"),
        "degraded": data.get("degraded"),
    })
    degraded = bool(data.get("degraded"))
    vm.severity = Severity.MEDIUM if degraded else Severity.OK
    if degraded:
        vm.add_reason("ws_degraded")
    vm.status = "ACTIVE"
    vm.summary = (
        f"connected={data.get('connected')} msg_age={data.get('message_age_sec')}s "
        f"fallback={data.get('fallback_state')} degraded={degraded}"
    )
    return vm
