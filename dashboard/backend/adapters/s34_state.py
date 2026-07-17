"""Panel 3 — S34 State Machine (primary panel, read-only)."""
from __future__ import annotations

from dashboard.backend import shadow_observatory as obs
from dashboard.backend.readers import iter_jsonl_tail, load_json, parse_iso
from dashboard.backend.sources import DashboardContext
from dashboard.backend.view_models import (
    ParseStatus,
    PanelViewModel,
    ProvenanceStatus,
    Severity,
    TrustState,
)

STATE_REL = ("reports", "shadow", "s34_state_machine_shadow_state.json")
LEDGER_REL = ("reports", "shadow", "s34_state_machine_shadow.jsonl")
BOUNDED_HISTORY = 15


def build(ctx: DashboardContext) -> PanelViewModel:
    state_path = ctx.p(*STATE_REL)
    ledger_path = ctx.p(*LEDGER_REL)
    data, ps, mtime = load_json(state_path)

    vm = PanelViewModel(
        key="s34_state",
        title="S34 State Machine",
        source_path=str(state_path),
        read_timestamp=ctx.now,
        source_timestamp=parse_iso(data.get("updated_utc")) if isinstance(data, dict) else mtime,
        freshness_threshold_seconds=ctx.threshold("s34_state"),
        parse_status=ps,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    if ps == ParseStatus.MISSING:
        vm.status = "MISSING"
        vm.severity = Severity.MEDIUM
        vm.trust_state = TrustState.MISSING
        vm.summary = "S34 shadow state artifact not found (fail-closed)"
        vm.add_reason("state_artifact_missing")
        return vm
    if ps == ParseStatus.MALFORMED or not isinstance(data, dict):
        vm.status = "MALFORMED"
        vm.severity = Severity.MEDIUM
        vm.trust_state = TrustState.MALFORMED
        vm.parse_status = ParseStatus.MALFORMED
        vm.summary = "S34 shadow state artifact unparseable (fail-closed)"
        vm.add_reason("state_artifact_malformed")
        return vm

    positions = data.get("positions") if isinstance(data.get("positions"), dict) else {}
    closed_ids = data.get("closed_ids") if isinstance(data.get("closed_ids"), dict) else {}
    processed = data.get("processed") if isinstance(data.get("processed"), dict) else {}

    # Bounded recent transition history from the ledger tail.
    rows, lps, _lmt = iter_jsonl_tail(ledger_path)
    closes = [r for r in rows if str(r.get("event")) == "CLOSE"]
    recent = closes[-BOUNDED_HISTORY:]
    last_close = closes[-1] if closes else None

    # Liveness is reconciled against the runner rather than read off the ledger:
    # a lifecycle with no terminal event that the runner no longer tracks is an
    # orphan, and counting it as open would overstate exposure. See §140.
    lifecycles = obs.build_lifecycles(rows)
    live = obs.reconcile_liveness(lifecycles, positions)

    open_pos = [pid for pid, p in positions.items() if isinstance(p, dict)]
    vm.fields = {
        "mode": data.get("mode"),
        "rule_name": data.get("rule_name"),
        "updated_utc": data.get("updated_utc"),
        "open_positions": len(open_pos),
        "open_position_ids": open_pos[:10],
        "closed_ids_count": len(closed_ids),
        "processed_count": len(processed),
        "duplicate_close_protection": "closed_ids_tracked" if closed_ids else "no_closed_ids_yet",
        # Liveness reconciliation (ledger vs runner).
        "exposed_positions": live["exposed_n"],
        "waiting_positions": live["waiting_n"],
        "ledger_orphans": live["orphaned_n"],
        "runner_positions_absent_from_ledger": len(live["untracked_by_ledger"]),
        # The producer's state.json["pnl"] block is deliberately NOT surfaced
        # here: it sums backfilled (simulated) closes together with forward ones,
        # so reporting it verbatim presents simulation as observed evidence. The
        # split lives in the s34_shadow_evidence panel, recomputed from the ledger.
        "pnl_source": "see s34_shadow_evidence panel (forward/backfill split, recomputed from ledger)",
        "closed_trades_in_ledger": len(closes),
        "last_close_signal": (last_close or {}).get("signal") if last_close else None,
        "last_close_reason": (last_close or {}).get("close_reason") if last_close else None,
        "last_close_net_bps": (last_close or {}).get("net_bps") if last_close else None,
        "ledger_parse_status": lps.value,
    }
    vm.rows = [
        {
            "id": r.get("id"),
            "signal": r.get("signal"),
            "direction": r.get("direction"),
            "close_reason": r.get("close_reason"),
            "net_bps": r.get("net_bps"),
            "score": r.get("score"),
            "closed_utc": r.get("closed_utc"),
            "status": r.get("status"),
        }
        for r in reversed(recent)
    ]
    vm.status = "ACTIVE"
    vm.severity = Severity.OK
    if live["orphaned_n"]:
        # The ledger is missing terminal events it should have written.
        vm.severity = Severity.LOW
        vm.add_reason("ledger_orphans_no_terminal_event")
    if live["untracked_by_ledger"]:
        vm.severity = Severity.LOW
        vm.add_reason("runner_positions_absent_from_ledger")
    vm.summary = (
        f"mode={data.get('mode')} rule={data.get('rule_name')} exposed={live['exposed_n']} "
        f"waiting={live['waiting_n']} orphans={live['orphaned_n']} closed={len(closes)} "
        f"dup_close_guard={'on' if closed_ids else 'pending'}"
    )
    return vm
