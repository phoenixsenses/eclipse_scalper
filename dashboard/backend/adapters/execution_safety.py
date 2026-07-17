"""Panel 4 — Execution Safety (read-only). Explicitly NO CONTROL ACTIONS."""
from __future__ import annotations

from dashboard.backend.process_identity import scan_topology
from dashboard.backend.readers import load_env_numeric, load_json
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

LIVE_STATE_REL = ("runtime", "s34_v_engine_live_state.json")
# Numeric-only env keys relevant to execution posture (never secrets).
POSTURE_KEYS = (
    "S34_LIVE_TRADING_ENABLED", "SCALPER_DRY_RUN", "S34_LIVE_MAX_LEVERAGE",
    "S34_V_ENGINE_LIVE_STOP_BPS", "ORDER_NOTIONAL_USD",
)


def build(ctx: DashboardContext) -> PanelViewModel:
    topo = scan_topology(ctx.proc_snapshot)
    live_exec_count = topo["live_executor_count"]

    live_path = ctx.p(*LIVE_STATE_REL)
    data, ps, mtime = load_json(live_path)
    env_numeric = load_env_numeric(ctx.env_path, POSTURE_KEYS)

    vm = PanelViewModel(
        key="execution_safety",
        title="Execution Safety",
        source_path=str(live_path),
        read_timestamp=ctx.now,
        source_timestamp=mtime,
        freshness_threshold_seconds=ctx.threshold("execution_state"),
        parse_status=ps if ps != ParseStatus.MISSING else ParseStatus.MISSING,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    state = data if isinstance(data, dict) else {}
    orders = state.get("orders") if isinstance(state.get("orders"), (list, dict)) else None
    pending = state.get("pending") if isinstance(state.get("pending"), (list, dict)) else None

    def _count(x):
        if isinstance(x, dict):
            return len(x)
        if isinstance(x, list):
            return len(x)
        return None

    # Live executor running is a HIGH safety signal (default posture is OFF).
    if live_exec_count > 0:
        sev = Severity.HIGH
        vm.add_reason("live_executor_running")
    else:
        sev = Severity.OK

    if ps == ParseStatus.MISSING:
        # No live-state artifact: executor almost certainly off; not itself an error.
        vm.trust_state = TrustState.MISSING
        vm.add_reason("live_state_artifact_absent")

    vm.fields = {
        "NO_CONTROL_ACTIONS_AVAILABLE": True,
        "live_executor_process_count": live_exec_count,
        "live_executor_expected": 0,
        "live_state_status": state.get("status"),
        "live_state_active": state.get("active"),
        "pending_order_count": _count(pending),
        "open_order_count": _count(orders),
        "processed_event_ids": _count(state.get("processed_event_ids")),
        "reconciliation": (state.get("reconciliation") if isinstance(state.get("reconciliation"), dict) else None),
        # F-NEW-03 micro-signal cancellation is a fire-and-forget-then-drained
        # background task inside the entry loop; it has no persisted artifact, so
        # we do NOT fabricate a state — we report observability honestly.
        "micro_signal_cancel_observability": "no_persisted_artifact",
        "env_posture_numeric": env_numeric,
    }
    vm.severity = sev
    vm.status = "ACTIVE"
    vm.summary = (
        f"NO CONTROL ACTIONS AVAILABLE · live_executor={live_exec_count} (expected 0) · "
        f"pending={_count(pending)} open={_count(orders)}"
    )
    return vm
