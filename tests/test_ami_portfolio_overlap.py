"""Tests — portfolio overlap observer (whitepaper §73.1 / Part IX).

Covers the interval law, the episode adjustment, the exclusion contract (a bad
timestamp must be dropped and counted, never defaulted), and the refusal to
emit a correlation coefficient or a control action.
"""
from __future__ import annotations

import sqlite3

import pytest

from ami.portfolio import overlap as P

H = 3_600_000
T0 = 1_780_000_000_000


def _db(rows):
    """rows: (trade_id, rule, symbol, entry_ts, exit_ts, net_bps, exit_reason)"""
    c = sqlite3.connect(":memory:")
    c.executescript(
        "CREATE TABLE s34_trades (trade_id TEXT, signal_id TEXT, rule_name TEXT, symbol TEXT,"
        " direction TEXT, entry_ts_ms INTEGER);"
        "CREATE TABLE s34_signals (signal_id TEXT, signal_ts_ms INTEGER);"
        "CREATE TABLE s34_outcomes (trade_id TEXT, exit_ts_ms INTEGER, net_bps REAL,"
        " exit_reason TEXT);"
    )
    for tid, rule, sym, ent, ex, net, reason in rows:
        sid = "S-" + tid
        c.execute("INSERT INTO s34_trades VALUES (?,?,?,?,'LONG',?)", (tid, sid, rule, sym, ent))
        c.execute("INSERT INTO s34_signals VALUES (?,?)", (sid, ent))
        c.execute("INSERT INTO s34_outcomes VALUES (?,?,?,?)", (tid, ex, net, reason))
    c.commit()
    return c


def test_overlapping_intervals_are_a_pair():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + 2 * H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 3 * H, -20.0, "SL")])
    r = P.analyze(c)
    assert r["pair_n"] == 1
    assert r["symbol_pair_concentration"] == {("ETHUSDT", "SOLUSDT"): 1}


def test_touching_intervals_are_not_concurrent():
    """B opens exactly when A closes: there is no instant of joint exposure."""
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 2 * H, -20.0, "SL")])
    assert P.analyze(c)["pair_n"] == 0


def test_disjoint_intervals_are_not_concurrent():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + 2 * H, T0 + 3 * H, -20.0, "SL")])
    assert P.analyze(c)["pair_n"] == 0


def test_joint_loss_and_joint_stop_are_counted_separately():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + 2 * H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 3 * H, -20.0, "TIME")])
    r = P.analyze(c)
    assert r["joint_loss_pairs"] == 1, "both lost"
    assert r["joint_stop_pairs"] == 0, "only one exited via SL"
    assert r["joint_loss_total_bps"] == -30.0


def test_mixed_outcome_pair_is_neither_joint_loss_nor_joint_win():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + 2 * H, +10.0, "TP"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 3 * H, -20.0, "SL")])
    r = P.analyze(c)
    assert r["joint_loss_pairs"] == 0
    assert r["joint_win_pairs"] == 0


def test_episode_gate_separates_distant_trades():
    """A >4h quiet gap starts a new independent episode (canonical-v1 rule)."""
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + 10 * H, T0 + 11 * H, -20.0, "SL")])
    r = P.analyze(c)
    assert r["episode_n"] == 2
    assert r["pair_n"] == 0


def test_pair_n_inflates_relative_to_episodes():
    """The whole point of episode adjustment: 3 mutually-overlapping trades in
    ONE episode produce 3 pairs. Reporting 3 as independent would triple-count
    a single market event."""
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + 3 * H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 3 * H, -20.0, "SL"),
             ("C", "R3", "BTCUSDT", T0 + H, T0 + 3 * H, -30.0, "SL")])
    r = P.analyze(c)
    assert r["pair_n"] == 3
    assert r["episodes_with_overlap_n"] == 1, "3 pairs, 1 independent episode"


def test_zero_entry_timestamp_is_excluded_and_counted_not_defaulted():
    """31 of the real ledger's 265 trades carry entry_ts_ms=0. Treating those
    as a 1970 timestamp chains them into a fake mega-episode."""
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -10.0, "SL"),
             ("BAD", "R2", "SOLUSDT", 0, T0 + H, -20.0, "SL")])
    r = P.analyze(c)
    assert r["trade_n"] == 1
    assert r["excluded"] == {"MISSING_ENTRY_TS": 1}
    assert r["excluded_n"] == 1


def test_missing_net_bps_is_excluded_not_zeroed():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -10.0, "SL"),
             ("BAD", "R2", "SOLUSDT", T0, T0 + H, None, "TIME")])
    r = P.analyze(c)
    assert r["excluded"] == {"MISSING_NET_BPS": 1}
    assert r["trade_n"] == 1


def test_non_positive_duration_is_excluded():
    c = _db([("BAD", "R1", "ETHUSDT", T0 + H, T0, -10.0, "SL")])
    r = P.analyze(c)
    assert r["status"] == "NO_USABLE_TRADES"
    assert "NOT evidence of zero overlap" in r["note"]


def test_empty_ledger_is_not_reported_as_zero_overlap():
    c = _db([])
    r = P.analyze(c)
    assert r["status"] == "NO_USABLE_TRADES"


def test_refuses_correlation_coefficient_and_control_actions():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + 2 * H, -10.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + H, T0 + 3 * H, -20.0, "SL")])
    r = P.analyze(c)
    assert "returns_correlation" in r["refused"]
    assert "capital_path_ergodicity" in r["refused"]
    assert r["no_control_actions_available"] is True
    assert "NOT an independent-observation" in r["n_semantics"]


def test_worst_episode_is_the_most_negative_not_the_largest():
    c = _db([("A", "R1", "ETHUSDT", T0, T0 + H, -50.0, "SL"),
             ("B", "R2", "SOLUSDT", T0 + 10 * H, T0 + 11 * H, +80.0, "TP"),
             ("C", "R3", "BTCUSDT", T0 + 10 * H, T0 + 11 * H, +80.0, "TP")])
    r = P.analyze(c)
    assert r["worst_episode"]["net_bps"] == -50.0
    assert r["worst_episode"]["trades"] == 1
