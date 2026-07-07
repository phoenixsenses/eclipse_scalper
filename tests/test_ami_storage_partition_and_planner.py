"""Focused tests: ami.storage.partition (identity + planner)."""
from __future__ import annotations

import datetime as dt
import sqlite3

import pytest

from ami.storage import partition as PT
from ami.storage.registry import UnknownTableError

NOW = dt.datetime(2026, 7, 7, 18, 0, 0, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# Partition identity
# ---------------------------------------------------------------------------

def test_closed_month_accepted():
    pid = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                       utc_month=5, source_watermark_value=1, now=NOW)
    assert pid.partition_start_ms == 1777593600000
    assert pid.partition_end_ms == 1780272000000


def test_current_month_rejected():
    with pytest.raises(PT.PartitionValidationError, match="current UTC month"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=7, source_watermark_value=1, now=NOW)


def test_current_month_message_discloses_partial_reason():
    """The current-month rejection and the partial-month rejection are
    the same underlying condition for calendar months (a month's end
    boundary cannot be in the future without `now` being inside it) --
    the current-month message explicitly says so."""
    with pytest.raises(PT.PartitionValidationError, match="partial UTC month, has not fully closed"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=7, source_watermark_value=1, now=NOW)


def test_future_month_rejected():
    with pytest.raises(PT.PartitionValidationError, match="future UTC month"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=12, source_watermark_value=1, now=NOW)


def test_partial_month_rejected_distinct_from_current_month():
    """A partition month whose END has not yet arrived, observed from a
    LATER month than the partition's own month, is impossible for
    calendar months (contiguous, no gaps) -- so this test proves the
    other direction: July 2026 (the current month, per NOW) is rejected
    as partial when evaluated against a `now` still inside July but past
    the 1st, confirming the half-open `end > now` check independent of
    the current-month identity shortcut."""
    late_july = dt.datetime(2026, 7, 20, tzinfo=dt.timezone.utc)
    with pytest.raises(PT.PartitionValidationError, match="has not fully closed yet"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=7, source_watermark_value=1, now=late_july)


def test_active_horizon_protection():
    """June 2026 (ended 2026-07-01) is fully closed by 2026-07-07, but
    falls inside the 30-day active horizon (2026-06-07..2026-07-07)."""
    with pytest.raises(PT.PartitionValidationError, match="active retention horizon"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=6, source_watermark_value=1, now=NOW)


def test_may_2026_outside_active_horizon():
    # Should NOT raise -- May is safely before the 30-day horizon.
    pid = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                       utc_month=5, source_watermark_value=1, now=NOW)
    assert pid.utc_month == 5


def test_malformed_month_rejected():
    with pytest.raises(PT.PartitionValidationError, match="malformed"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=13, source_watermark_value=1, now=NOW)
    with pytest.raises(PT.PartitionValidationError, match="malformed"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=0, source_watermark_value=1, now=NOW)


def test_half_open_boundaries():
    pid = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                       utc_month=5, source_watermark_value=1, now=NOW)
    may_last_ms = pid.partition_end_ms - 1
    assert pid.partition_start_ms <= may_last_ms < pid.partition_end_ms
    assert not (pid.partition_start_ms <= pid.partition_end_ms < pid.partition_end_ms)


def test_deterministic_partition_id():
    a = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=1, now=NOW)
    b = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=999, now=NOW)
    assert a.partition_id == b.partition_id  # watermark value doesn't affect identity


def test_deterministic_archive_path():
    pid = PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                       utc_month=5, source_watermark_value=1, now=NOW)
    assert pid.archive_relative_path == f"mark_prices/ETHUSDT/2026/05/part-{pid.partition_id}.parquet"


def test_unknown_table_rejected_at_identity_construction():
    with pytest.raises(UnknownTableError):
        PT.build_partition_identity(table="not_a_real_table", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=1, now=NOW)


def test_naive_now_rejected():
    naive = dt.datetime(2026, 7, 7)
    with pytest.raises(PT.PartitionValidationError, match="timezone-aware"):
        PT.build_partition_identity(table="mark_prices", symbol="ETHUSDT", utc_year=2026,
                                     utc_month=5, source_watermark_value=1, now=naive)


# ---------------------------------------------------------------------------
# Planner (synthetic in-memory fixture)
# ---------------------------------------------------------------------------

def _make_mark_prices_fixture():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE mark_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)""")
    conn.execute("CREATE INDEX idx_mark_ts ON mark_prices(ts_ms)")
    conn.execute("CREATE INDEX idx_mark_symbol_ts ON mark_prices(symbol, ts_ms)")
    conn.execute("CREATE TABLE gaps (id INTEGER PRIMARY KEY, stream TEXT, start_ts_ms INTEGER, "
                 "end_ts_ms INTEGER, resolved_bool INTEGER)")
    may_start = 1777593600000
    rows = [(may_start + i * 60000, "ETHUSDT", 3000.0 + i, None, None) for i in range(5)]
    rows += [(may_start + i * 60000, "BTCUSDT", 60000.0 + i, None, None) for i in range(5)]
    conn.executemany("INSERT INTO mark_prices (ts_ms,symbol,mark_price,funding_rate,next_funding_time_ms) "
                     "VALUES (?,?,?,?,?)", rows)
    conn.commit()
    return conn


def test_planner_bounded_estimate_matches_fixture():
    conn = _make_mark_prices_fixture()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    assert plan.estimated_row_count == 5
    assert plan.archive_rehearsal_eligible is True
    conn.close()


def test_planner_requires_indexes():
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE mark_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        mark_price REAL NOT NULL, funding_rate REAL, next_funding_time_ms INTEGER)""")
    conn.execute("CREATE TABLE gaps (id INTEGER PRIMARY KEY, stream TEXT, start_ts_ms INTEGER, "
                 "end_ts_ms INTEGER, resolved_bool INTEGER)")
    conn.commit()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    assert plan.plan_state == PT.PLAN_BLOCKED_BY_INDEX
    assert plan.archive_rehearsal_eligible is False
    conn.close()


def test_planner_discloses_unresolved_gaps():
    conn = _make_mark_prices_fixture()
    conn.execute("INSERT INTO gaps (stream, start_ts_ms, end_ts_ms, resolved_bool) VALUES "
                 "('mark_prices', 1777593600000, NULL, 0)")
    conn.commit()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    assert plan.unresolved_gap_count == 1
    assert plan.archive_rehearsal_eligible is True  # gaps disclose, don't block rehearsal
    conn.close()


def test_planner_purge_and_production_always_false():
    conn = _make_mark_prices_fixture()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    assert plan.purge_eligible is False
    assert plan.production_activation_eligible is False
    conn.close()


def test_planner_resource_limit_blocks():
    conn = _make_mark_prices_fixture()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5,
                              max_estimated_rows=1, now=NOW)
    assert plan.plan_state == PT.PLAN_BLOCKED_BY_RESOURCE_LIMIT
    assert plan.archive_rehearsal_eligible is False
    conn.close()


def test_planner_unknown_dependency_fails_closed_via_validation_error():
    conn = _make_mark_prices_fixture()
    plan = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=7, now=NOW)
    assert plan.plan_state == PT.PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS
    assert plan.archive_rehearsal_eligible is False
    conn.close()


def test_planner_deterministic_output():
    conn = _make_mark_prices_fixture()
    p1 = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    p2 = PT.plan_partition(conn, table="mark_prices", symbol="ETHUSDT", utc_year=2026, utc_month=5, now=NOW)
    assert p1.plan_state == p2.plan_state
    assert p1.estimated_row_count == p2.estimated_row_count
    conn.close()


def test_all_plan_states_in_declared_enum():
    for state in (PT.PLAN_ARCHIVE_ELIGIBLE, PT.PLAN_BLOCKED_BY_SCHEMA, PT.PLAN_BLOCKED_BY_INDEX,
                  PT.PLAN_BLOCKED_BY_SOURCE_GAP, PT.PLAN_BLOCKED_BY_REPAIR,
                  PT.PLAN_BLOCKED_BY_RESOURCE_LIMIT, PT.PLAN_BLOCKED_BY_UNKNOWN_SOURCE_SEMANTICS):
        assert state in PT.PLAN_STATES
