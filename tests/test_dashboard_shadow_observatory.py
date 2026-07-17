"""Tests for the S34 shadow observatory analysis + its two canonical panels.

These target epistemic invariants, not markup:

  * backfilled closes never enter forward evidence, and never sum with it;
  * the runner's state.json["pnl"] block is never the evidence source;
  * NO route is ever truncated away (the pre-§140 adapter head-sliced the pnl
    keys, which — because the producer writes sort_keys=True — deterministically
    dropped every SHORT_* route rather than a random tail);
  * routes under MIN_EVIDENCE_N report INSUFFICIENT_N instead of a win rate;
  * a ledger lifecycle the runner no longer tracks is an orphan, not exposure;
  * absent/malformed inputs fail closed, never render as healthy.

Fixtures are built under tmp_path; the production ledger is never touched.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.backend import shadow_observatory as obs  # noqa: E402
from dashboard.backend.adapters import s34_shadow_evidence, s34_state  # noqa: E402
from dashboard.backend.sources import DashboardContext  # noqa: E402
from dashboard.backend.view_models import Severity  # noqa: E402

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)
T0_MS = int(NOW.timestamp() * 1000)


def _open(pid: str, signal: str, *, status: str = "OPEN", backfill: bool = False, **extra):
    entered = not status.startswith("MONITORING")
    row = {
        "id": pid,
        "signal": signal,
        "direction": "LONG",
        "event": "OPEN",
        "status": status,
        "anchor_ts_ms": T0_MS,
        "entry_ts_ms": T0_MS if entered else None,
        "entry_price": 1000.0 if entered else None,
        "opened_utc": NOW.isoformat(),
    }
    if backfill:
        row["backfill"] = True
    row.update(extra)
    return row


def _close(pid: str, signal: str, net: float, *, backfill: bool = False, reason: str = "TIME_EXIT"):
    row = _open(pid, signal, status=f"CLOSED_{reason}", backfill=backfill)
    row.update(
        {
            "event": "CLOSE",
            "close_reason": reason,
            "exit_price": 1001.0,
            "exit_ts_ms": T0_MS + 60_000,
            "outcome_bps": net + obs.FEE_BPS,
            "net_bps": net,
        }
    )
    return row


def _ctx(tmp_path: Path, rows: list[dict], *, state: dict | None = None) -> DashboardContext:
    shadow = tmp_path / "reports" / "shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    (shadow / "s34_state_machine_shadow.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    payload = {
        "mode": "OBSERVE_ONLY_NO_ORDER",
        "rule_name": "TEST_RULE",
        "updated_utc": NOW.isoformat(),
        "positions": {},
        "closed_ids": {},
        "processed": {},
        "pnl": {},
    }
    if state is not None:
        payload.update(state)
    (shadow / "s34_state_machine_shadow_state.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    return DashboardContext(repo_root=tmp_path, now=NOW.timestamp())


# ------------------------------------------------------------ evidence split


def test_backfill_never_enters_forward_evidence(tmp_path):
    rows = [
        _close("F1", "ROUTE_A", 10.0),
        _close("B1", "ROUTE_A", 900.0, backfill=True),
        _close("B2", "ROUTE_A", 900.0, backfill=True),
    ]
    ev = obs.evidence(rows)

    assert ev["forward"]["totals"]["n"] == 1
    assert ev["forward"]["totals"]["total_net_bps"] == 10.0
    assert ev["backfill"]["totals"]["n"] == 2
    assert ev["backfill"]["totals"]["total_net_bps"] == 1800.0
    assert ev["backfill"]["label"] == "SIMULATED_NOT_EVIDENCE"


def test_evidence_panel_keeps_populations_in_separate_rows(tmp_path):
    ctx = _ctx(tmp_path, [_close("F1", "ROUTE_A", 10.0), _close("B1", "ROUTE_A", 900.0, backfill=True)])
    vm = s34_shadow_evidence.build(ctx)

    pops = {r["population"] for r in vm.rows}
    assert pops == {"FORWARD", "BACKFILL_SIMULATED"}
    assert vm.fields["forward_closes"] == 1
    assert vm.fields["backfill_closes"] == 1
    assert vm.fields["forward_total_net_bps"] == 10.0


def test_state_json_pnl_block_is_never_the_evidence_source(tmp_path):
    """state.json's pnl mixes forward and backfilled closes; both panels ignore it."""
    poisoned = {"pnl": {"ROUTE_A": {"n": 999, "total_net": 12345.0, "wins": 999, "wr": 1.0}}}
    ctx = _ctx(tmp_path, [_close("F1", "ROUTE_A", -20.0)], state=poisoned)

    ev_vm = s34_shadow_evidence.build(ctx)
    assert ev_vm.fields["forward_total_net_bps"] == -20.0
    assert "12345" not in json.dumps(ev_vm.to_dict())

    state_vm = s34_state.build(ctx)
    assert "pnl_summary" not in state_vm.fields
    assert "12345" not in json.dumps(state_vm.to_dict())


# ------------------------------------------------------- no silent truncation


def test_no_route_is_truncated_away(tmp_path):
    """Regression: the pre-§140 adapter head-sliced state.json's sorted pnl keys.

    Because the producer writes sort_keys=True, a head-slice drops SHORT_* routes
    deterministically -- it is a systematic bias toward LONG routes, not a
    random tail. Every route with a close must appear.
    """
    routes = [f"LONG_R{i:02d}" for i in range(9)] + ["SHORT_NOISY", "SHORT_NEITHER"]
    rows = [_close(f"F{i}", r, 10.0) for i, r in enumerate(routes)]
    ctx = _ctx(tmp_path, rows)
    vm = s34_shadow_evidence.build(ctx)

    reported = {r["route"] for r in vm.rows if r["population"] == "FORWARD"}
    assert reported == set(routes)
    assert "SHORT_NOISY" in reported
    assert "SHORT_NEITHER" in reported
    assert vm.fields["forward_routes"] == len(routes)


def test_losing_short_route_is_visible(tmp_path):
    """The routes the old head-slice hid were the SHORT ones; they must show."""
    rows = [_close(f"F{i}", "SHORT_NOISY", -9.4) for i in range(46)]
    rows += [_close(f"G{i}", "BUY_FADE_SHORT_H45_SL75", 7.1) for i in range(74)]
    vm = s34_shadow_evidence.build(_ctx(tmp_path, rows))

    short = next(r for r in vm.rows if r["route"] == "SHORT_NOISY")
    assert short["n"] == 46
    assert short["total_net_bps"] < 0


# ------------------------------------------------------------ INSUFFICIENT_N


def test_route_under_min_evidence_n_reports_insufficient(tmp_path):
    stats = obs.route_stats([50.0] * (obs.MIN_EVIDENCE_N - 1))
    assert stats["sufficient_n"] is False
    assert stats["wr"] is None
    assert stats["wr_display"] == "INSUFFICIENT_N"


def test_route_at_min_evidence_n_reports_a_win_rate(tmp_path):
    nets = [50.0 if i % 2 == 0 else -50.0 for i in range(obs.MIN_EVIDENCE_N)]
    stats = obs.route_stats(nets)
    assert stats["sufficient_n"] is True
    assert stats["wr"] == 0.5


def test_thin_route_never_reports_a_numeric_wr_in_the_panel(tmp_path):
    vm = s34_shadow_evidence.build(_ctx(tmp_path, [_close("F1", "SMALL", 63.6)]))
    row = next(r for r in vm.rows if r["route"] == "SMALL")

    assert row["wr"] == "INSUFFICIENT_N"
    assert "100.0%" not in json.dumps(vm.to_dict())
    assert "routes_below_min_evidence_n" in vm.reason_codes


# ---------------------------------------------------- liveness reconciliation


def test_lifecycle_untracked_by_runner_is_an_orphan_not_exposure(tmp_path):
    ctx = _ctx(tmp_path, [_open("P1", "ROUTE_A", status="OPEN")], state={"positions": {}})
    vm = s34_state.build(ctx)

    assert vm.fields["ledger_orphans"] == 1
    assert vm.fields["exposed_positions"] == 0
    assert vm.severity == Severity.LOW
    assert "ledger_orphans_no_terminal_event" in vm.reason_codes


def test_runner_tracked_position_is_exposed(tmp_path):
    ctx = _ctx(tmp_path, [_open("P1", "ROUTE_A", status="OPEN")], state={"positions": {"P1": {"id": "P1"}}})
    vm = s34_state.build(ctx)

    assert vm.fields["exposed_positions"] == 1
    assert vm.fields["ledger_orphans"] == 0
    assert vm.severity == Severity.OK


def test_monitoring_status_carries_no_exposure(tmp_path):
    ctx = _ctx(
        tmp_path,
        [_open("P1", "ROUTE_A", status="MONITORING_PROP")],
        state={"positions": {"P1": {"id": "P1"}}},
    )
    vm = s34_state.build(ctx)

    assert vm.fields["waiting_positions"] == 1
    assert vm.fields["exposed_positions"] == 0


def test_runner_position_absent_from_ledger_is_surfaced(tmp_path):
    ctx = _ctx(tmp_path, [], state={"positions": {"GHOST": {"id": "GHOST"}}})
    vm = s34_state.build(ctx)

    assert vm.fields["runner_positions_absent_from_ledger"] == 1
    assert "runner_positions_absent_from_ledger" in vm.reason_codes


def test_closed_lifecycle_is_not_live(tmp_path):
    rows = [_open("C1", "ROUTE_A"), _close("C1", "ROUTE_A", 10.0)]
    live = obs.reconcile_liveness(obs.build_lifecycles(rows), {})

    assert live["exposed_n"] == 0
    assert live["orphaned_n"] == 0


def test_last_event_wins_when_status_changes(tmp_path):
    rows = [_open("P1", "R", status="MONITORING_PROP"), _open("P1", "R", status="OPEN")]
    live = obs.reconcile_liveness(obs.build_lifecycles(rows), {"P1": {}})

    assert live["exposed_n"] == 1
    assert live["waiting_n"] == 0


# ------------------------------------------------------------- fail-closed


def test_evidence_panel_fails_closed_when_ledger_absent(tmp_path):
    (tmp_path / "reports" / "shadow").mkdir(parents=True)
    ctx = DashboardContext(repo_root=tmp_path, now=NOW.timestamp())
    vm = s34_shadow_evidence.build(ctx)

    assert vm.status == "MISSING"
    assert vm.severity == Severity.MEDIUM
    assert "ledger_missing" in vm.reason_codes


def test_state_panel_fails_closed_when_state_absent(tmp_path):
    (tmp_path / "reports" / "shadow").mkdir(parents=True)
    ctx = DashboardContext(repo_root=tmp_path, now=NOW.timestamp())
    vm = s34_state.build(ctx)

    assert vm.status == "MISSING"
    assert "state_artifact_missing" in vm.reason_codes


def test_empty_ledger_reports_no_closes(tmp_path):
    vm = s34_shadow_evidence.build(_ctx(tmp_path, []))
    assert vm.status == "NO_CLOSES"
    assert vm.rows == []


# ---------------------------------------------------------------- funnel


def test_funnel_attributes_expiry_and_entry(tmp_path):
    expire = _open("E1", "ROUTE_A", status="MONITORING_BTC")
    expire = {**expire, "event": "EXPIRE", "status": "EXPIRED_NO_BTC"}
    rows = [
        _open("E1", "ROUTE_A", status="MONITORING_BTC"),
        expire,
        _open("C1", "ROUTE_A"),
        _close("C1", "ROUTE_A", 10.0),
    ]
    f = obs.funnel(obs.build_lifecycles(rows))

    assert f["by_route"]["ROUTE_A"]["opened"] == 2
    assert f["by_route"]["ROUTE_A"]["entered"] == 1
    assert f["by_route"]["ROUTE_A"]["expired"] == 1
    assert f["expire_reasons"]["EXPIRED_NO_BTC"] == 1


# -------------------------------------------------------------- observers


def test_observers_are_observe_only(tmp_path):
    rows = [{"id": "F1", "signal": "R", "event": "PROFIT_LOCK_TRIGGER", "status": "OPEN"}]
    o = obs.observers(rows)

    assert o["profit_lock"]["mode"] == "OBSERVE_ONLY_NO_ORDER"
    assert o["profit_lock"]["trigger_events"] == 1
    assert o["bd_first_buy50"]["mode"] == "OBSERVE_ONLY_NO_ORDER"
    assert o["scalein"]["mode"] == "OBSERVE_ONLY_NO_ORDER"
