"""S34 Shadow Evidence — per-route forward PnL, backfill held apart (read-only).

The runner's ``state.json["pnl"]`` block is not the source here. That block sums
backfilled (simulated) closes together with forward ones, so surfacing it
verbatim reports simulation as observed evidence. This panel recomputes every
route from the ledger and keeps the two populations in separate tables.

Routes under ``MIN_EVIDENCE_N`` forward closes report INSUFFICIENT_N instead of
a win rate: a WR of 1.00 over n=1 reads as a decision if printed as a number.

No route is ever truncated away. See SYSTEM_STATE §140.
"""
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


def _row(route: str, stats: dict, population: str) -> dict:
    return {
        "route": route,
        "population": population,
        "n": stats["n"],
        "total_net_bps": stats["total_net_bps"],
        "avg_net_bps": stats["avg_net_bps"],
        "wr": stats["wr_display"],
        "best_bps": stats["best_bps"],
        "worst_bps": stats["worst_bps"],
    }


def build(ctx: DashboardContext) -> PanelViewModel:
    state_path = ctx.p(*STATE_REL)
    ledger_path = ctx.p(*LEDGER_REL)
    data, ps, mtime = load_json(state_path)

    vm = PanelViewModel(
        key="s34_shadow_evidence",
        title="S34 Shadow Evidence (forward vs backfill)",
        source_path=str(ledger_path),
        read_timestamp=ctx.now,
        source_timestamp=parse_iso(data.get("updated_utc")) if isinstance(data, dict) else mtime,
        freshness_threshold_seconds=ctx.threshold("s34_shadow_evidence"),
        parse_status=ps,
        provenance_status=ProvenanceStatus.DIRECT,
    )

    rows, lps, lmt = iter_jsonl_tail(ledger_path)
    if lps == ParseStatus.MISSING:
        vm.status = "MISSING"
        vm.severity = Severity.MEDIUM
        vm.trust_state = TrustState.MISSING
        vm.parse_status = ParseStatus.MISSING
        vm.summary = "S34 shadow ledger not found (fail-closed)"
        vm.add_reason("ledger_missing")
        return vm

    ev = obs.evidence(rows)
    fwd, bf = ev["forward"], ev["backfill"]

    vm.fields = {
        "forward_closes": fwd["totals"]["n"],
        "forward_total_net_bps": fwd["totals"]["total_net_bps"],
        "forward_routes": fwd["totals"]["routes"],
        "backfill_closes": bf["totals"]["n"],
        "backfill_total_net_bps": bf["totals"]["total_net_bps"],
        "backfill_label": bf["label"],
        "closes_total_in_ledger": ev["closes_total"],
        "min_evidence_n": obs.MIN_EVIDENCE_N,
        "fee_bps": obs.FEE_BPS,
        "evidence_note": (
            "Forward and backfill are never summed. state.json['pnl'] is not read: "
            "it merges both populations."
        ),
        "ledger_parse_status": lps.value,
    }

    # Forward first (the evidence), then backfill (the simulation), each
    # ordered by n. Population is a column so the two can never be read as one
    # table by accident.
    vm.rows = [_row(r, s, "FORWARD") for r, s in fwd["by_route"].items()]
    vm.rows += [_row(r, s, "BACKFILL_SIMULATED") for r, s in bf["by_route"].items()]

    if not vm.rows:
        vm.status = "NO_CLOSES"
        vm.severity = Severity.INFO
        vm.summary = "No closed trades in ledger yet"
        return vm

    thin = [r for r, s in fwd["by_route"].items() if not s["sufficient_n"]]
    vm.status = "ACTIVE"
    vm.severity = Severity.OK
    if thin:
        vm.add_reason("routes_below_min_evidence_n")
    vm.summary = (
        f"forward n={fwd['totals']['n']} net={fwd['totals']['total_net_bps']:+.1f}bps "
        f"across {fwd['totals']['routes']} routes ({len(thin)} below n={obs.MIN_EVIDENCE_N}) | "
        f"backfill n={bf['totals']['n']} SIMULATED_NOT_EVIDENCE"
    )
    return vm
