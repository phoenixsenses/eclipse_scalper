"""Tests — dashboard paper_bucket_ledger panel (read-only, per-rule).

Covers the three properties that make this panel safe rather than merely
useful: fail-closed sourcing, the non-poolable pre_v2/v2 split, and the
deprecated-bucket flag. Uses a synthetic DB; never touches the live
data/s34_intelligence.db.
"""
from __future__ import annotations

import sqlite3

import pytest

from dashboard.backend.adapters import paper_bucket_ledger as P
from dashboard.backend.sources import DashboardContext

B = P.MIN_GAP_V2_BOUNDARY_MS


def _mk_db(path, trades):
    """trades: (trade_id, rule, status, entry_ts_ms, net_bps, exit_reason)"""
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE s34_trades (trade_id TEXT, signal_id TEXT, rule_name TEXT, symbol TEXT,"
        " direction TEXT, status TEXT, opened_at_utc TEXT, entry_ts_ms INTEGER, entry_price REAL,"
        " tp_price REAL, sl_price REAL, be_trigger_price REAL);"
        "CREATE TABLE s34_outcomes (trade_id TEXT, signal_id TEXT, rule_name TEXT,"
        " outcome_ts_utc TEXT, exit_ts_ms INTEGER, exit_reason TEXT, gross_bps REAL,"
        " entry_adverse_bps REAL, exit_adverse_bps REAL, spread_cost_bps REAL,"
        " fee_cost_bps REAL, net_bps REAL);"
        "CREATE TABLE s34_rejected_signals (decision_id TEXT, signal_id TEXT, trade_id TEXT,"
        " rejected_ts_utc TEXT, signal_ts_ms INTEGER, rule_name TEXT, reason TEXT,"
        " context_json TEXT);"
    )
    for tid, rule, status, ets, net, reason in trades:
        conn.execute(
            "INSERT INTO s34_trades (trade_id, rule_name, symbol, direction, status, entry_ts_ms)"
            " VALUES (?,?,?,?,?,?)", (tid, rule, "ETHUSDT", "LONG", status, ets))
        if status == "CLOSED":
            conn.execute(
                "INSERT INTO s34_outcomes (trade_id, rule_name, exit_ts_ms, exit_reason, net_bps)"
                " VALUES (?,?,?,?,?)", (tid, rule, ets + 1000, reason, net))
    conn.commit()
    conn.close()


def _ctx(tmp_path, monkeypatch, trades):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    _mk_db(root / "data" / "s34_intelligence.db", trades)
    return DashboardContext(repo_root=root, now=1_784_000_000.0)


def test_missing_db_is_fail_closed_not_zero_trades(tmp_path):
    root = tmp_path / "empty"
    (root / "data").mkdir(parents=True)
    vm = P.build(DashboardContext(repo_root=root, now=1.0))
    assert vm.status == "UNAVAILABLE"
    assert vm.rows == []
    # The whole point: absence of the source must never render as "0 trades".
    assert "NOT evidence of zero trades" in vm.fields["note"]
    assert vm.severity is not P.Severity.OK


def test_unreadable_db_is_fail_closed(tmp_path):
    root = tmp_path / "bad"
    (root / "data").mkdir(parents=True)
    (root / "data" / "s34_intelligence.db").write_bytes(b"this is not a sqlite file")
    vm = P.build(DashboardContext(repo_root=root, now=1.0))
    assert vm.status == "ERROR"
    assert vm.rows == []
    assert "NOT reported as zero trades" in vm.fields["note"]


def test_pre_v2_and_v2_are_split_and_never_pooled(tmp_path, monkeypatch):
    # 2 winners before the boundary, 2 losers after: pooled WR would be 50%
    # and hide that the two eras point opposite ways.
    ctx = _ctx(tmp_path, monkeypatch, [
        ("t1", "R_A", "CLOSED", B - 10_000, 100.0, "TP"),
        ("t2", "R_A", "CLOSED", B - 5_000, 50.0, "TP"),
        ("t3", "R_A", "CLOSED", B + 5_000, -40.0, "SL"),
        ("t4", "R_A", "CLOSED", B + 10_000, -60.0, "SL"),
    ])
    vm = P.build(ctx)
    row = next(r for r in vm.rows if r["bucket"] == "R_A")
    assert row["stats"]["all"]["n"] == 4
    assert row["stats"]["pre_v2"] == {"n": 2, "wins": 2, "win_rate": 100.0,
                                      "avg_net_bps": 75.0, "total_net_bps": 150.0}
    assert row["stats"]["v2"] == {"n": 2, "wins": 0, "win_rate": 0.0,
                                  "avg_net_bps": -50.0, "total_net_bps": -100.0}
    assert "MUST NOT be pooled" in vm.fields["pooling_warning"]


def test_boundary_trade_belongs_to_v2_not_pre_v2(tmp_path, monkeypatch):
    """Exact-boundary timestamp is v2 (>=), matching the runner's activation ts."""
    ctx = _ctx(tmp_path, monkeypatch, [("t1", "R_A", "CLOSED", B, 10.0, "TP")])
    row = P.build(ctx).rows[0]
    assert row["stats"]["v2"]["n"] == 1
    assert row["stats"]["pre_v2"]["n"] == 0


def test_deprecated_bucket_flagged_and_excluded_from_live_total(tmp_path, monkeypatch):
    dep = sorted(P.DEPRECATED_RULES)[0]
    ctx = _ctx(tmp_path, monkeypatch, [
        (dep, dep, "CLOSED", B - 1000, 500.0, "TP"),      # big historical winner
        ("t2", "R_LIVE", "CLOSED", B - 1000, -20.0, "SL"),
    ])
    vm = P.build(ctx)
    drow = next(r for r in vm.rows if r["bucket"] == dep)
    assert drow["deprecated"] is True
    assert drow["deprecated_reason"] == P.DEPRECATED_REASON
    # Pooled total is flattered by the deprecated winner; ex-deprecated is not.
    assert vm.fields["overall"]["all"]["total_net_bps"] == 480.0
    assert vm.fields["overall_excluding_deprecated"]["all"]["total_net_bps"] == -20.0
    assert vm.fields["deprecated_bucket_count"] == 1


def test_open_positions_surfaced_and_not_counted_as_closed(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, [
        ("t1", "R_A", "CLOSED", B - 1000, 10.0, "TP"),
        ("t2", "R_A", "OPEN", B - 500, None, None),
    ])
    vm = P.build(ctx)
    row = next(r for r in vm.rows if r["bucket"] == "R_A")
    assert row["stats"]["all"]["n"] == 1, "OPEN must not enter closed-trade stats"
    assert row["open_now"] == 1
    assert vm.fields["open_virtual_positions"] == 1
    assert vm.severity is P.Severity.INFO


def test_missing_net_bps_is_not_treated_as_zero(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, [("t1", "R_A", "CLOSED", B - 1000, None, "TIME")])
    row = P.build(ctx).rows[0]
    s = row["stats"]["all"]
    assert s["n"] == 1
    assert s["win_rate"] is None and s["total_net_bps"] is None
    assert "not treated as zero" in s["note"]


def test_panel_declares_no_control_actions(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, [("t1", "R_A", "CLOSED", B - 1000, 10.0, "TP")])
    vm = P.build(ctx)
    assert vm.fields["no_control_actions_available"] is True
    assert "not independent cycles" in vm.fields["n_semantics"].lower()


def test_reject_reasons_are_grouped_per_bucket(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, monkeypatch, [("t1", "R_A", "CLOSED", B - 1000, 10.0, "TP")])
    db = ctx.p("data", "s34_intelligence.db")
    conn = sqlite3.connect(db)
    conn.executemany(
        "INSERT INTO s34_rejected_signals (rule_name, reason) VALUES (?,?)",
        [("R_A", "REGIME_FILTER"), ("R_A", "REGIME_FILTER"), ("R_A", "MAX_OPEN_TRADES")],
    )
    conn.commit()
    conn.close()
    row = next(r for r in P.build(ctx).rows if r["bucket"] == "R_A")
    assert row["reject_reasons"] == {"REGIME_FILTER": 2, "MAX_OPEN_TRADES": 1}


def test_registered_in_aggregator_panel_list():
    from dashboard.backend import aggregator
    keys = [k for k, _title, _mod in aggregator.PANEL_BUILDERS]
    assert "paper_bucket_ledger" in keys
