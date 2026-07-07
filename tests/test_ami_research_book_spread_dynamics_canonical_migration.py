"""BATCH-BOOK-SPREAD-DYNAMICS-CANONICAL-MIGRATION-V1 (M-0036) -- focused tests.

DISPOSABLE_DB_ONLY: every test runs against a disposable COPY of the real
canonical.sqlite (never the real path) + a READ-ONLY connection to the frozen
retained rehearsal database. Outcome-blind throughout.

Run: pytest tests/test_ami_research_book_spread_dynamics_canonical_migration.py --basetemp <scratchpad> -p no:cacheprovider
"""
from __future__ import annotations

import shutil
import sqlite3

import pytest

from ami.research import book_spread_dynamics_canonical_migration as MIG
from ami.warehouse import schema as wschema
from ami.warehouse.schema import DEFAULT_PATH as REAL_CANONICAL_PATH

FROZEN_SOURCE = "D:/eclipse_scalper/.runtime_temp/spread_rehearsal_v1/rehearsal_run1.sqlite"
FROZEN_ROOT = "33c4f4be3233aad399d72fc525601c7eecb2eb6ab235ecd4070ba640701c6e31"
FROZEN_COMPONENT = {
    "ordered_anchor_manifest": "a77a8daf2a8d198d775436674a20a9bd5328dc071e2883938b7c331c17c534bb",
    "exact_feature_manifest": "b1eb902f5b3d1ea0f19b4b60d0ad999907a042b228adf506bbe09800a81e155b",
    "exclusion_manifest": "0694e43300710e1204c1b23643d9eacb9f10188c21aa0ceda572c28229cc8449",
    "cycle_membership_manifest": "e692ff1c8ce37b54a3349a501a38bd44f24865e75a51accc81c7e97399d29e18",
    "representative_manifest": "edadf5972cbbdddb0efa1db8234473ee089972f504d3bfbfafbae508238db246",
}
EXPECTED = {
    "ami_book_spread_change_windowed_flow": 196,
    "ami_book_spread_change_window_quality_v1": 324,
    "ami_book_spread_change_exclusions": 128,
}


def _disposable(tmp_path):
    dst = tmp_path / "canon.sqlite"
    shutil.copy2(REAL_CANONICAL_PATH, dst)
    conn = sqlite3.connect(dst)
    conn.execute("PRAGMA foreign_keys=ON")
    wschema.init_schema(conn)  # additive; brings the disposable copy to v14
    return conn


def _source():
    c = sqlite3.connect(f"file:{FROZEN_SOURCE}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=ON")
    return c


# ---------------------------------------------------------------------------
# Identity / constant enforcement
# ---------------------------------------------------------------------------

def test_migration_constants():
    assert MIG.MIGRATION_ID == "M-0036"
    assert MIG.FORMULA_VERSION == "BOOK_SPREAD_CHANGE_BPS_W300_V1"
    assert MIG.ROW_ACCOUNTING_ROOT == FROZEN_ROOT
    assert MIG.SPECIFICATION_HASH == "ea611121291c63136860d57926389520de571ce6615bed2e1a3627e51442a212"


def test_schema_version_is_14():
    assert wschema.CANONICAL_SCHEMA_VERSION == 14


# ---------------------------------------------------------------------------
# Counts + identities on first run
# ---------------------------------------------------------------------------

def test_first_run_inserts_expected_counts(tmp_path):
    conn = _disposable(tmp_path)
    pre = MIG.canonical_counts(conn)
    real_applied = pre == EXPECTED  # branch-aware: once the real migration lands, a copy already has the rows
    src = _source()
    try:
        result = MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    for t, n in EXPECTED.items():
        assert result[t]["inserted"] + result[t]["noop_identical"] == n
        if real_applied:
            assert result[t]["noop_identical"] == n
        else:
            assert result[t]["inserted"] == n and result[t]["noop_identical"] == 0
    assert MIG.canonical_counts(conn) == EXPECTED
    # integrity + FK clean
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_accounting_identities_after_migration(tmp_path):
    conn = _disposable(tmp_path)
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    q = dict(conn.execute("SELECT source_quality_class, COUNT(*) FROM ami_book_spread_change_window_quality_v1 "
                          "GROUP BY source_quality_class").fetchall())
    assert q == {"EXACT_RECONSTRUCTABLE": 196, "STALE_SOURCE": 22, "UNAVAILABLE_BEFORE_COLLECTION": 106}
    x = dict(conn.execute("SELECT source_quality_class, COUNT(*) FROM ami_book_spread_change_exclusions "
                          "GROUP BY source_quality_class").fetchall())
    assert x == {"STALE_SOURCE": 22, "UNAVAILABLE_BEFORE_COLLECTION": 106}
    reps = conn.execute("SELECT COUNT(*) FROM ami_book_spread_change_windowed_flow WHERE is_cycle_representative=1").fetchone()[0]
    cycles = conn.execute("SELECT COUNT(DISTINCT cycle_id) FROM ami_book_spread_change_windowed_flow").fetchone()[0]
    assert reps == cycles == 97
    # exact anchors never in exclusions; excluded never in feature table
    overlap = conn.execute(
        "SELECT COUNT(*) FROM ami_book_spread_change_windowed_flow f "
        "JOIN ami_book_spread_change_exclusions x ON x.anchor_id=f.anchor_id").fetchone()[0]
    assert overlap == 0
    conn.close()


def test_canonical_replay_reproduces_frozen_manifest_hashes(tmp_path):
    conn = _disposable(tmp_path)
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    replay = MIG.canonical_replay_hashes(conn)
    assert replay == FROZEN_COMPONENT
    conn.close()


# ---------------------------------------------------------------------------
# Idempotency + conflict + no-mutation
# ---------------------------------------------------------------------------

def test_second_run_is_noop_identical(tmp_path):
    conn = _disposable(tmp_path)
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
        src.close()
        src = _source()
        r2 = MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    for t, n in EXPECTED.items():
        assert r2[t]["inserted"] == 0 and r2[t]["noop_identical"] == n
    assert MIG.canonical_counts(conn) == EXPECTED
    conn.close()


def test_conflict_nonidentical_raises(tmp_path):
    conn = _disposable(tmp_path)
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    # corrupt one migrated feature row, then re-migrate from the unmodified source
    fid = conn.execute("SELECT feature_id FROM ami_book_spread_change_windowed_flow LIMIT 1").fetchone()[0]
    conn.execute("UPDATE ami_book_spread_change_windowed_flow SET spread_change_bps_w300 = spread_change_bps_w300 + 1.0 "
                 "WHERE feature_id=?", (fid,))
    conn.commit()
    src = _source()
    try:
        with pytest.raises(MIG.ConflictNonIdentical):
            MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    conn.close()


def test_protected_tables_unchanged_by_migration(tmp_path):
    conn = _disposable(tmp_path)
    pre = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
           for t in ("ami_events", "ami_signal_lifecycle", "ami_cycles", "experiment_registry",
                     "experiment_results", "ami_absorption_impact_windowed_flow", "ami_cvd_windowed_flow")}
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    post = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in pre}
    assert pre == post
    conn.close()


# ---------------------------------------------------------------------------
# Schema constraints (CHECK/UNIQUE/FK) reject bad rows
# ---------------------------------------------------------------------------

def test_feature_table_rejects_non_exact_quality(tmp_path):
    conn = _disposable(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ami_book_spread_change_windowed_flow (feature_id, formula_version, anchor_id, "
                     "cycle_id, signal_birth_ts, symbol, venue, market_segment, quote_currency, direction, "
                     "current_target_ts, current_quote_id, current_quote_ts, current_quote_age_ms, current_bid, "
                     "current_ask, current_mid, current_spread_bps, historical_target_ts, historical_quote_id, "
                     "historical_quote_ts, historical_quote_age_ms, historical_bid, historical_ask, historical_mid, "
                     "historical_spread_bps, spread_change_bps_w300, source_quality_class, known_at_ts, "
                     "feature_available_ts, is_cycle_representative, specification_hash, row_accounting_root, "
                     "migration_id, input_manifest_id, created_ms) VALUES "
                     "('BSF-x','BOOK_SPREAD_CHANGE_BPS_W300_V1','A','C',1000,'ETHUSDT','BINANCE_USDM_PERP',"
                     "'PERPETUAL_FUTURES','USDT','LONG',1000,1,1000,0,1.0,1.1,1.05,1.0,700,1,700,0,1.0,1.1,1.05,1.0,"
                     "0.5,'STALE_SOURCE',1000,1000,0,'s','" + FROZEN_ROOT + "','M-0036','m',1)")
    conn.close()


def test_feature_table_rejects_wrong_root(tmp_path):
    conn = _disposable(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ami_book_spread_change_windowed_flow (feature_id, formula_version, anchor_id, "
                     "cycle_id, signal_birth_ts, symbol, venue, market_segment, quote_currency, direction, "
                     "current_target_ts, current_quote_id, current_quote_ts, current_quote_age_ms, current_bid, "
                     "current_ask, current_mid, current_spread_bps, historical_target_ts, historical_quote_id, "
                     "historical_quote_ts, historical_quote_age_ms, historical_bid, historical_ask, historical_mid, "
                     "historical_spread_bps, spread_change_bps_w300, source_quality_class, known_at_ts, "
                     "feature_available_ts, is_cycle_representative, specification_hash, row_accounting_root, "
                     "migration_id, input_manifest_id, created_ms) VALUES "
                     "('BSF-x','BOOK_SPREAD_CHANGE_BPS_W300_V1','A','C',1000,'ETHUSDT','BINANCE_USDM_PERP',"
                     "'PERPETUAL_FUTURES','USDT','LONG',1000,1,1000,0,1.0,1.1,1.05,1.0,700,1,700,0,1.0,1.1,1.05,1.0,"
                     "0.5,'EXACT_RECONSTRUCTABLE',1000,1000,0,'s','WRONGROOT','M-0036','m',1)")
    conn.close()


def test_exclusion_table_rejects_exact(tmp_path):
    conn = _disposable(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ami_book_spread_change_exclusions (exclusion_id, formula_version, anchor_id, "
                     "cycle_id, source_quality_class, exclusion_reason, exclusion_endpoint, "
                     "exclusion_precedence_position, current_quality_status, historical_quality_status, "
                     "current_quote_age_ms, historical_quote_age_ms, specification_hash, row_accounting_root, "
                     "migration_id, input_manifest_id, created_ms) VALUES "
                     "('BSX-x','BOOK_SPREAD_CHANGE_BPS_W300_V1','A','C','EXACT_RECONSTRUCTABLE','r','both',4,"
                     "'EXACT_RECONSTRUCTABLE','EXACT_RECONSTRUCTABLE',0,0,'s','" + FROZEN_ROOT + "','M-0036','m',1)")
    conn.close()


def test_feature_table_unique_anchor_formula(tmp_path):
    conn = _disposable(tmp_path)
    src = _source()
    try:
        MIG.run_canonical_migration(conn, src)
    finally:
        src.close()
    row = conn.execute("SELECT anchor_id FROM ami_book_spread_change_windowed_flow LIMIT 1").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO ami_book_spread_change_windowed_flow (feature_id, formula_version, anchor_id, "
                     "cycle_id, signal_birth_ts, symbol, venue, market_segment, quote_currency, direction, "
                     "current_target_ts, current_quote_id, current_quote_ts, current_quote_age_ms, current_bid, "
                     "current_ask, current_mid, current_spread_bps, historical_target_ts, historical_quote_id, "
                     "historical_quote_ts, historical_quote_age_ms, historical_bid, historical_ask, historical_mid, "
                     "historical_spread_bps, spread_change_bps_w300, source_quality_class, known_at_ts, "
                     "feature_available_ts, is_cycle_representative, specification_hash, row_accounting_root, "
                     "migration_id, input_manifest_id, created_ms) VALUES "
                     "('BSF-dup2','BOOK_SPREAD_CHANGE_BPS_W300_V1',?,'C',1000,'ETHUSDT','BINANCE_USDM_PERP',"
                     "'PERPETUAL_FUTURES','USDT','LONG',1000,1,1000,0,1.0,1.1,1.05,1.0,700,1,700,0,1.0,1.1,1.05,1.0,"
                     "0.5,'EXACT_RECONSTRUCTABLE',1000,1000,0,'s','" + FROZEN_ROOT + "','M-0036','m',1)", (row,))
    conn.close()


def test_no_outcome_columns_in_destination_tables(tmp_path):
    conn = _disposable(tmp_path)
    for t in EXPECTED:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        for bad in ("endpoint_return_bps", "mfe_bps", "mae_bps"):
            assert bad not in cols
        for altwin in ("w60", "w600", "w1800", "w3600"):
            assert not any(altwin in c.lower() for c in cols)
    conn.close()
