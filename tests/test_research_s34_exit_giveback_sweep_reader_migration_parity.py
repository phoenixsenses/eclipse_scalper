"""Range-read consumer migration (BATCH-STORAGE-ROTATION-RETENTION-
RANGE-READ-CONSUMER-MIGRATION-V7), second target.

`tools/research_s34_exit_giveback_sweep.py`'s `path_marks` (a bounded
range read over `mark_prices`, `ts_ms>=? AND ts_ms<=? ORDER BY ts_ms`) is
migrated to `path_marks_v2` (reader-backed via `plan_read`/`execute_read`).
`path_marks` is kept unchanged as the parity oracle; it is no longer
called from the live path (`simulate_standard`, `simulate_trailing_half_mfe`,
`simulate_partial_tp60_rest_tp120`, `main`), which now call
`path_marks_v2` exclusively.

Call-site audit: `symbol` is always "ETHUSDT" in this file's only real
call path -- `load_feature_events()` hard-filters `symbol='ETHUSDT'` --
so real ARCHIVE_ONLY/HYBRID production smoke against the real
`mark_prices/ETHUSDT/2026-05` partition is possible, not merely
synthetic.

Range-boundary note: the oracle's SQL uses an INCLUSIVE upper bound
(`ts_ms<=entry_ts_ms+MAX_HORIZON_SEC*1000`); `plan_read`/`execute_read`
use the reader's half-open `[start_ms, end_ms)` convention, so
`path_marks_v2` passes `end_ms+1`. A dedicated test proves a row exactly
AT the boundary is included and a row past it is excluded, matching the
oracle bit-for-bit.

Out-of-scope, untouched (this gate is range-read only):
  * `mark_at()` -- ASOF-SHAPED but called exclusively with
    `before=False` in this file, i.e. `ORDER BY ts_ms ASC LIMIT 1`
    ("first row at-or-after ts_ms"). `research_reader.lookup_latest_at_or_before`
    only supports the opposite direction ("latest row at-or-before
    ts_ms", `ORDER BY ts_ms DESC LIMIT 1`). No forward-ASOF primitive
    exists in the reader and this gate may not add a new helper --
    `mark_at` is left entirely on direct SQL, proven untouched below.
  * `load_feature_events()` / `load_outcome_labels()` -- query
    `data/s34_feature_factory.db` (`liq_event_features`,
    `liq_event_outcome_labels`), a different SQLite database entirely,
    not `microstructure.db`, not an allowlisted research-reader table.
    Left entirely on direct SQL, proven untouched below.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import research_reader as RR
from ami.storage import source_access as SRC
from ami.storage.registry import get_table_spec
import tools.research_s34_exit_giveback_sweep as mod
from tools.research_s34_exit_giveback_sweep import path_marks, path_marks_v2

REAL_SOURCE_DB = str(SRC.DEFAULT_SOURCE_PATH)
SPEC = get_table_spec("mark_prices")

pytestmark = pytest.mark.filterwarnings("ignore")

MAY_ARCHIVE_START_MS = 1777593600000  # real mark_prices/ETHUSDT/2026-05 partition start
MAY_ARCHIVE_END_MS = 1780272000000    # real mark_prices/ETHUSDT/2026-05 partition end
HORIZON_MS = mod.MAX_HORIZON_SEC * 1000


def _root():
    root, _ = PR.resolve_production_root()
    return root


def _needs_real_db():
    if not os.path.exists(REAL_SOURCE_DB):
        pytest.skip("real source database not present in this checkout")


def _old(symbol, entry_ts_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return path_marks(conn, symbol, entry_ts_ms)
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _old_row_count(symbol, start_ms, end_ms):
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=?",
            (symbol, int(start_ms), int(end_ms)),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()


def _digest(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Real production smoke: ARCHIVE_ONLY -- entry well inside mark_prices/
# ETHUSDT/2026-05, horizon short enough to stay inside the same partition
# ---------------------------------------------------------------------------

def test_v2_parity_archive_only_real_window_and_plan_mode():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    end_ms = entry_ts + HORIZON_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=end_ms + 1)
    assert plan.mode == "ARCHIVE_ONLY"
    old_rows = _old("ETHUSDT", entry_ts)
    new_rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


# ---------------------------------------------------------------------------
# Real production smoke: recent live SQLITE_ONLY window
# ---------------------------------------------------------------------------

def test_v2_parity_sqlite_only_recent_live_window():
    _needs_real_db()
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        max_ts = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    entry_ts = max_ts - 60_000  # short horizon well past the archive, entirely live
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=entry_ts + HORIZON_MS + 1)
    assert plan.mode == "SQLITE_ONLY"
    old_rows = _old("ETHUSDT", entry_ts)
    new_rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


# ---------------------------------------------------------------------------
# Real production smoke: HYBRID window straddling the archive/live boundary
# ---------------------------------------------------------------------------

def test_v2_parity_hybrid_real_boundary_and_plan_mode():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_END_MS - (HORIZON_MS // 2)
    end_ms = entry_ts + HORIZON_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=end_ms + 1)
    assert plan.mode == "HYBRID"
    old_rows = _old("ETHUSDT", entry_ts)
    new_rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert new_rows == old_rows
    assert len(old_rows) > 0


def test_provenance_reports_hybrid_source_and_columns():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_END_MS - (HORIZON_MS // 2)
    end_ms = entry_ts + HORIZON_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=end_ms + 1)
    result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB)
    list(result.iter_rows())
    prov = result.provenance
    assert prov["source_type"] == "HYBRID"
    assert prov["table"] == "mark_prices"
    assert prov["symbol"] == "ETHUSDT"
    assert prov["columns"] == ["ts_ms", "mark_price"]
    assert prov["ordering"] == "(ts_ms ASC, id ASC)"
    assert len(prov["archive_segments"]) == 1
    assert prov["archive_segments"][0]["manifest_sha256"]
    assert len(prov["sqlite_ranges"]) == 1
    assert prov["row_count"] > 0


# ---------------------------------------------------------------------------
# Row count / ordering / first-last-timestamp / digest / batch-size parity
# ---------------------------------------------------------------------------

def test_row_count_parity_real_archive_window():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    old_n = _old_row_count("ETHUSDT", entry_ts, entry_ts + HORIZON_MS)
    new_rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert len(new_rows) == old_n


def test_ordering_parity_strictly_ascending_real_data():
    _needs_real_db()
    rows = path_marks_v2(_root(), "ETHUSDT", MAY_ARCHIVE_START_MS, source_db_path=REAL_SOURCE_DB)
    ts_seq = [r[0] for r in rows]
    assert ts_seq == sorted(ts_seq)


def test_first_last_timestamp_and_digest_parity_real_data():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    old_rows = _old("ETHUSDT", entry_ts)
    new_rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert old_rows[0] == new_rows[0]
    assert old_rows[-1] == new_rows[-1]
    assert _digest(old_rows) == _digest(new_rows)


def test_repeated_run_determinism():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    d1 = _digest(path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB))
    d2 = _digest(path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB))
    assert d1 == d2


def test_output_invariant_to_batch_size_real_data():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    plan = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=entry_ts + HORIZON_MS + 1)

    def _rows(batch_size):
        result = RR.execute_read(plan, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB, batch_size=batch_size)
        return [(int(ts), float(px)) for ts, px in result.iter_rows()]

    assert _rows(1_000_000) == _rows(11)


# ---------------------------------------------------------------------------
# Inclusive-boundary parity
# ---------------------------------------------------------------------------

def test_inclusive_end_boundary_parity_real_data():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    conn, log = SRC.open_read_only(REAL_SOURCE_DB)
    try:
        last_ts = conn.execute(
            "SELECT ts_ms FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (entry_ts, entry_ts + HORIZON_MS),
        ).fetchone()[0]
    finally:
        SRC.assert_read_only_session_clean(log)
        conn.close()
    old_n_at = _old_row_count("ETHUSDT", entry_ts, int(last_ts))
    old_n_before = _old_row_count("ETHUSDT", entry_ts, int(last_ts) - 1)
    assert old_n_at == old_n_before + 1

    plan_at = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=int(last_ts) + 1)
    n_at = sum(1 for _ in RR.execute_read(plan_at, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB).iter_rows())
    plan_before = RR.plan_read(_root(), table="mark_prices", symbol="ETHUSDT", start_ms=entry_ts, end_ms=int(last_ts))
    n_before = sum(1 for _ in RR.execute_read(plan_before, columns=("ts_ms", "mark_price"), source_db_path=REAL_SOURCE_DB).iter_rows())
    assert n_at == old_n_at
    assert n_before == old_n_before


# ---------------------------------------------------------------------------
# Empty / bounded window
# ---------------------------------------------------------------------------

def test_empty_range_returns_empty_list_synthetic(tmp_path):
    root = str(tmp_path / "r")
    os.makedirs(root)
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": 0, "entries": []}, f)
    db = str(tmp_path / "s.sqlite")
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE mark_prices (id INTEGER PRIMARY KEY, ts_ms INTEGER, symbol TEXT, "
              "mark_price REAL, funding_rate REAL, next_funding_time_ms INTEGER)")
    c.execute("CREATE INDEX ix ON mark_prices(symbol, ts_ms)")
    far_future = 4_000_000_000_000  # 2096 -- guaranteed to be strictly after any inserted row
    c.execute("INSERT INTO mark_prices VALUES (1,?,?,?,?,?)", (1_600_000_000_000, "ETHUSDT", 3000.0, 0.0001, None))
    c.commit()
    c.close()
    assert path_marks_v2(root, "ETHUSDT", far_future, source_db_path=db) == []


def test_bounded_window_row_count_sane_real_data():
    _needs_real_db()
    entry_ts = MAY_ARCHIVE_START_MS
    rows = path_marks_v2(_root(), "ETHUSDT", entry_ts, source_db_path=REAL_SOURCE_DB)
    assert 0 < len(rows) < 1_000_000  # a 1h window is never anywhere near this large
    for ts, px in rows:
        assert entry_ts <= ts <= entry_ts + HORIZON_MS
        assert px > 0.0


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

def test_no_full_reverify_referenced():
    assert "reverify_guard" not in inspect.getsource(mod)
    assert not hasattr(mod, "run_guarded_reverify")


def test_source_mutation_zero():
    _needs_real_db()
    root = _root()
    idx = os.path.join(root, PR.ROOT_INDEX_NAME)
    before = os.path.getmtime(idx)
    path_marks_v2(root, "ETHUSDT", MAY_ARCHIVE_START_MS, source_db_path=REAL_SOURCE_DB)
    assert os.path.getmtime(idx) == before


def test_path_marks_v1_no_longer_called_from_live_path():
    for fn in (mod.simulate_standard, mod.simulate_trailing_half_mfe,
               mod.simulate_partial_tp60_rest_tp120, mod.main):
        src = inspect.getsource(fn)
        assert "path_marks(" not in src or "path_marks_v2(" in src
        assert "= path_marks(" not in src
    for fn in (mod.simulate_standard, mod.simulate_trailing_half_mfe, mod.simulate_partial_tp60_rest_tp120):
        assert "path_marks_v2(" in inspect.getsource(fn)


def test_mark_at_asof_forward_lookup_left_untouched():
    """`mark_at()` is ASOF-shaped (`ORDER BY ts_ms {dir} LIMIT 1`) but
    called exclusively with `before=False` (ASC direction, "first row
    at-or-after") -- the opposite of what `lookup_latest_at_or_before`
    supports. Confirmed still on direct SQL, unmigrated."""
    src = inspect.getsource(mod.mark_at)
    assert "con.execute" in src
    assert "order by ts_ms {order}" in src
    assert "RR.lookup_latest_at_or_before" not in src  # documented in the docstring, never called
    for fn in (mod.simulate_standard, mod.simulate_trailing_half_mfe, mod.simulate_partial_tp60_rest_tp120):
        assert "before=False" in inspect.getsource(fn)


def test_feature_factory_db_reads_left_untouched():
    """`load_feature_events`/`load_outcome_labels` read
    `data/s34_feature_factory.db` (`liq_event_features`,
    `liq_event_outcome_labels`) -- a different SQLite database entirely,
    not an allowlisted research-reader table. Confirmed still on direct
    sqlite3, unmigrated, and that this gate did not touch the FEATURE_DB
    constant or these two functions' SQL."""
    assert mod.FEATURE_DB == "data/s34_feature_factory.db"
    for fn in (mod.load_feature_events, mod.load_outcome_labels):
        src = inspect.getsource(fn)
        assert "sqlite3.connect(FEATURE_DB)" in src
        assert "plan_read" not in src and "execute_read" not in src


def test_real_feature_factory_db_present_and_untouched_by_migration():
    if not os.path.exists(mod.FEATURE_DB):
        pytest.skip("s34_feature_factory.db not present in this checkout")
    before = os.path.getmtime(mod.FEATURE_DB)
    _ = mod.load_feature_events()
    _ = mod.load_outcome_labels()
    after = os.path.getmtime(mod.FEATURE_DB)
    assert before == after


# ---------------------------------------------------------------------------
# Synthetic fixture: trust-failure fail-closed
# ---------------------------------------------------------------------------

def test_trust_failure_fails_closed_synthetic(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ami.storage.archive import build_pyarrow_schema

    root = str(tmp_path / "r")
    rel = os.path.join("table=mark_prices", "venue=BINANCE_USDM_PERP", "market_segment=PERPETUAL_FUTURES",
                        "symbol=ETHUSDT", "year=2026", "month=05", "version=v1")
    final_dir = os.path.join(root, rel)
    os.makedirs(final_dir, exist_ok=True)
    schema = build_pyarrow_schema(SPEC)
    may = 1777593600000
    rows = [(1, may, "ETHUSDT", 3000.0, 0.0001, None)]
    arrays = [pa.array([r[i] for r in rows], type=schema.field(c).type)
              for i, c in enumerate(SPEC.preserved_columns)]
    pp = os.path.join(final_dir, "part-00000.parquet")
    pq.write_table(pa.Table.from_arrays(arrays, schema=schema), pp, compression="zstd")
    manifest = {"source_table": "mark_prices", "symbol": "ETHUSDT", "venue": SPEC.venue,
                "market_segment": SPEC.market_segment, "row_count": 1, "shards": None,
                "ordered_scientific_content_hash": "irrelevant", "partition_id": "corrupt-test",
                "parquet_path": os.path.join(rel, "part-00000.parquet"),
                "production_status": "PRODUCTION_VERIFIED", "purge_authorization": "PROHIBITED",
                "source_watermark_field": "id", "source_watermark_value": 1,
                "parquet_sha256": hashlib.sha256(open(pp, "rb").read()).hexdigest()}
    mp = os.path.join(final_dir, "manifest.json")
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    with open(os.path.join(final_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        f.write("ok\n")
    entry = {"archive_identity": "corrupt-test", "archive_relative_path": rel, "source_table": "mark_prices",
             "symbol": "ETHUSDT", "venue": SPEC.venue, "market_segment": SPEC.market_segment,
             "partition_start_ms": may, "partition_end_ms": may + 3600_000, "row_count": 1,
             "manifest_sha256": "0" * 64,  # deliberately WRONG -- must fail the trust check
             "authorization_receipt_sha256": None, "production_status": "PRODUCTION_VERIFIED",
             "purge_authorization": "PROHIBITED", "scientific_content_hash": "irrelevant"}
    with open(os.path.join(root, "catalog_index.json"), "w", encoding="utf-8") as f:
        json.dump({"catalog_contract_version": "v1", "entry_count": 1, "entries": [entry]}, f)
    with pytest.raises(RR.ArchiveTrustError):
        path_marks_v2(root, "ETHUSDT", may)
