"""Focused tests: ami.storage.production_activation (manual-only
production archive activation) + policy activation states + streaming
export/restore.

Engine tests use disposable roots under pytest `tmp_path`. The three real
production partitions (mark_prices rehearsal + agg_trades + book_ticker
activation) are read-only-inspected by the acceptance tests at the bottom.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3

import pytest

from ami.storage import production as PR
from ami.storage import production_activation as PA
from ami.storage import policy as POL
from ami.storage.partition import build_partition_identity
from ami.storage.registry import get_table_spec

NOW = dt.datetime(2026, 7, 8, 18, 0, 0, tzinfo=dt.timezone.utc)


def _agg_fixture(n_eth=8, n_btc=10):
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE agg_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER NOT NULL, symbol TEXT NOT NULL,
        price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL, is_buyer_maker INTEGER NOT NULL)""")
    conn.execute("CREATE INDEX idx_trade_ts ON agg_trades(ts_ms)")
    conn.execute("CREATE INDEX idx_trade_symbol_ts ON agg_trades(symbol, ts_ms)")
    conn.execute("CREATE TABLE gaps (id INTEGER PRIMARY KEY, stream TEXT, start_ts_ms INTEGER, "
                 "end_ts_ms INTEGER, resolved_bool INTEGER)")
    # Feb 2026 window: [1769904000000, 1772323200000); data "starts" mid-month
    feb = 1770076800000
    rows = []
    for i in range(n_eth):
        rows.append((feb + i * 60000, "ETHUSDT", 3000.0 + i, 1.0, 3000.0 + i, i % 2))
    for i in range(n_btc):
        rows.append((feb + i * 60000, "BTCUSDT", 60000.0 + i, 0.1, 6000.0, i % 2))
    conn.executemany("INSERT INTO agg_trades (ts_ms,symbol,price,quantity,notional,is_buyer_maker) "
                     "VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


def _receipt_for(conn, root, table="agg_trades"):
    plan = PA.preflight_partition(conn, root=root, table=table, now=NOW)
    return plan, PA.issue_gate_authorized_receipt(plan, root=root, approver="gate", justification="test")


# ---------------------------------------------------------------------------
# Policy: manual-only activation states
# ---------------------------------------------------------------------------

def test_manual_only_states_are_separate_named_values():
    states = POL.production_activation_states()
    assert states["manual_production_archive_creation"] == "ENABLED"
    assert states["general_unrestricted_activation"] == "DISABLED"
    assert states["scheduler"] == "DISABLED"
    assert states["purge"] == "DISABLED"
    assert states["vacuum"] == "DISABLED"
    assert states["source_retention"] == "REQUIRED"


def test_general_activation_flag_still_false():
    assert POL.GENERAL_PRODUCTION_ACTIVATION_ENABLED is False


# ---------------------------------------------------------------------------
# Authorization receipt (Phase 3)
# ---------------------------------------------------------------------------

def _sample_partition():
    return build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                    source_watermark_value=100, now=NOW)


def _sample_receipt(**over):
    p = _sample_partition()
    spec = get_table_spec("agg_trades")
    kw = dict(partition=p, spec=spec, archive_version="v1", root="D:/eclipse_scalper/data/archives/raw_v1",
              source_schema_hash="ssh", archive_schema_hash="ash", max_source_rows=100,
              max_source_bytes=1 << 30, max_output_bytes=1 << 30, approver="gate", justification="t")
    kw.update(over)
    return PA.build_authorization_receipt(**kw)


def test_receipt_verifies_against_matching_plan():
    r = _sample_receipt()
    PA.verify_authorization_receipt(r, partition=_sample_partition(), archive_version="v1",
                                    root="D:/eclipse_scalper/data/archives/raw_v1",
                                    source_schema_hash="ssh", archive_schema_hash="ash")


def test_receipt_action_is_create_only():
    assert _sample_receipt()["action"] == "CREATE_PRODUCTION_ARCHIVE_ONLY"


def test_receipt_prohibits_purge_scheduler_vacuum():
    r = _sample_receipt()
    assert r["purge_authorization"] == "PROHIBITED"
    assert r["scheduler_authorization"] == "PROHIBITED"
    assert r["vacuum_authorization"] == "PROHIBITED"
    assert r["source_retention_requirement"] == "SOURCE_MUST_REMAIN_PRESENT"


def test_receipt_self_hash_present_and_valid():
    r = _sample_receipt()
    assert r["receipt_sha256"] == PA._receipt_self_hash(r)


def test_wrong_table_receipt_rejected():
    r = _sample_receipt()
    other = build_partition_identity(table="book_ticker", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                     source_watermark_value=100, now=NOW)
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.verify_authorization_receipt(r, partition=other, archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


def test_wrong_watermark_receipt_rejected():
    r = _sample_receipt()
    other = build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                     source_watermark_value=999, now=NOW)
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.verify_authorization_receipt(r, partition=other, archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


def test_wrong_root_receipt_rejected():
    r = _sample_receipt()
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.verify_authorization_receipt(r, partition=_sample_partition(), archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1_other",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


def test_schema_drift_receipt_rejected():
    r = _sample_receipt()
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.verify_authorization_receipt(r, partition=_sample_partition(), archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="DIFFERENT", archive_schema_hash="ash")


def test_altered_receipt_rejected():
    r = _sample_receipt()
    r["max_authorized_source_rows"] = 999999  # tamper without recomputing self-hash
    with pytest.raises(PA.AuthorizationReceiptRejected, match="self-hash"):
        PA.verify_authorization_receipt(r, partition=_sample_partition(), archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


def test_expired_receipt_rejected():
    r = _sample_receipt(expiry_ms=1)  # far in the past
    with pytest.raises(PA.AuthorizationReceiptRejected, match="expired"):
        PA.verify_authorization_receipt(r, partition=_sample_partition(), archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


def test_missing_receipt_rejected():
    with pytest.raises(PA.AuthorizationReceiptRejected, match="no authorization"):
        PA.verify_authorization_receipt({}, partition=_sample_partition(), archive_version="v1",
                                        root="D:/eclipse_scalper/data/archives/raw_v1",
                                        source_schema_hash="ssh", archive_schema_hash="ash")


# ---------------------------------------------------------------------------
# Catalog lock (Phase 5)
# ---------------------------------------------------------------------------

def test_lock_atomic_acquisition(tmp_path):
    root = str(tmp_path / "raw_v1")
    os.makedirs(root)
    lock = PA.acquire_catalog_lock(root, job_identity="j1", timeout_sec=2)
    assert PA.catalog_lock_status(root)["state"] == "HELD"
    PA.release_catalog_lock(lock)
    assert PA.catalog_lock_status(root)["state"] == "ABSENT"


def test_second_acquisition_conflicts(tmp_path):
    root = str(tmp_path / "raw_v1")
    os.makedirs(root)
    lock = PA.acquire_catalog_lock(root, job_identity="j1", timeout_sec=2)
    with pytest.raises(PA.CatalogLockConflict):
        PA.acquire_catalog_lock(root, job_identity="j2", timeout_sec=1)
    PA.release_catalog_lock(lock)


def test_non_owner_cannot_release(tmp_path):
    root = str(tmp_path / "raw_v1")
    os.makedirs(root)
    lock = PA.acquire_catalog_lock(root, job_identity="j1", timeout_sec=2)
    fake = {**lock, "owner_token": "WRONG_TOKEN"}
    with pytest.raises(PA.CatalogLockConflict):
        PA.release_catalog_lock(fake)
    PA.release_catalog_lock(lock)  # real owner still can


def test_unreadable_lock_classified_repair_required(tmp_path):
    root = str(tmp_path / "raw_v1")
    os.makedirs(root)
    lock_path = PA._lock_path(root)
    with open(lock_path, "w") as f:
        f.write("{ not valid json")
    status = PA.catalog_lock_status(root)
    assert status["state"] == "CATALOG_LOCK_REPAIR_REQUIRED"
    # not auto-deleted
    assert os.path.exists(lock_path)


def test_lock_absent_when_no_file(tmp_path):
    root = str(tmp_path / "raw_v1")
    os.makedirs(root)
    assert PA.catalog_lock_status(root)["state"] == "ABSENT"


# ---------------------------------------------------------------------------
# Candidate selection (Phase 8)
# ---------------------------------------------------------------------------

def test_candidate_selects_smallest_positive_symbol():
    conn = _agg_fixture(n_eth=8, n_btc=10)
    sel = PA.select_candidate_partition(conn, table="agg_trades", now=NOW)
    assert sel["chosen_symbol"] == "ETHUSDT"  # 8 < 10, SOL=0 excluded
    assert sel["chosen_row_count"] == 8
    conn.close()


def test_candidate_lexicographic_tie_break():
    conn = _agg_fixture(n_eth=10, n_btc=10)  # tie
    sel = PA.select_candidate_partition(conn, table="agg_trades", now=NOW)
    assert sel["chosen_symbol"] == "BTCUSDT"  # lexicographically first among equal counts
    conn.close()


def test_candidate_earliest_eligible_month():
    conn = _agg_fixture()
    sel = PA.select_candidate_partition(conn, table="agg_trades", now=NOW)
    assert (sel["chosen_year"], sel["chosen_month"]) == (2026, 2)
    conn.close()


def test_candidate_excludes_zero_count_symbol():
    conn = _agg_fixture(n_eth=5, n_btc=0)
    sel = PA.select_candidate_partition(conn, table="agg_trades", now=NOW)
    assert sel["chosen_symbol"] == "ETHUSDT"
    conn.close()


def test_candidate_records_all_considered():
    conn = _agg_fixture()
    sel = PA.select_candidate_partition(conn, table="agg_trades", now=NOW)
    assert len(sel["symbol_counts"]) == 3
    assert any(c["count"] == 0 for c in sel["symbol_counts"])  # SOL disclosed as 0
    conn.close()


# ---------------------------------------------------------------------------
# Resource guards (Phase 10)
# ---------------------------------------------------------------------------

def test_resource_check_rejects_low_free_space():
    fake_plan = {"table": "agg_trades", "estimated_source_bytes": 1000, "estimated_parquet_bytes": 1000}
    with pytest.raises(PA.ResourceLimitExceeded, match="free space"):
        PA.check_resource_limits(free_bytes=10 * 1024 ** 3, plans=[fake_plan])


def test_resource_check_rejects_oversized_partition():
    fake_plan = {"table": "book_ticker", "estimated_source_bytes": 200 * 1024 ** 3,
                "estimated_parquet_bytes": 1000}
    with pytest.raises(PA.ResourceLimitExceeded):
        PA.check_resource_limits(free_bytes=600 * 1024 ** 3, plans=[fake_plan])


def test_resource_check_passes_within_limits():
    plans = [{"table": "agg_trades", "estimated_source_bytes": 2 * 1024 ** 3, "estimated_parquet_bytes": 1 * 1024 ** 3},
             {"table": "book_ticker", "estimated_source_bytes": 10 * 1024 ** 3, "estimated_parquet_bytes": 8 * 1024 ** 3}]
    result = PA.check_resource_limits(free_bytes=1000 * 1024 ** 3, plans=plans)
    assert result["projected_free_bytes"] > 400 * 1024 ** 3


def test_no_hidden_force_override_in_signature():
    import inspect
    sig = inspect.signature(PA.check_resource_limits)
    assert "force" not in sig.parameters


# ---------------------------------------------------------------------------
# Six-file authorized publication (Phase 12-14) on disposable roots
# ---------------------------------------------------------------------------

def test_authorized_publish_creates_six_files(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    plan, receipt = _receipt_for(conn, root)
    result = PA.publish_authorized_production_partition(
        conn, root=root, partition=plan["partition"], spec=plan["spec"], archive_version="v1",
        receipt=receipt, job_identity="job-1", source_schema_hash=plan["source_schema_hash"],
        export_cutoff="2026-01-01T00:00:00Z")
    assert result.status == "PUBLISHED"
    for name in (PR.PARQUET_NAME, PR.MANIFEST_NAME, PA.AUTHORIZATION_RECEIPT_NAME,
                 PR.CATALOG_ENTRY_NAME, PR.SUCCESS_NAME):
        assert os.path.exists(os.path.join(result.final_partition_dir, name))
    assert result.reverification_mismatch_count == 0
    conn.close()


def test_authorized_publish_rejects_mismatched_receipt(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    plan, receipt = _receipt_for(conn, root)
    receipt["source_watermark_value"] = 999999  # drift; self-hash now invalid
    with pytest.raises(PA.AuthorizationReceiptRejected):
        PA.publish_authorized_production_partition(
            conn, root=root, partition=plan["partition"], spec=plan["spec"], archive_version="v1",
            receipt=receipt, job_identity="job-1", source_schema_hash=plan["source_schema_hash"],
            export_cutoff="x")
    conn.close()


def test_authorized_catalog_entry_has_receipt_fields(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    plan, receipt = _receipt_for(conn, root)
    result = PA.publish_authorized_production_partition(
        conn, root=root, partition=plan["partition"], spec=plan["spec"], archive_version="v1",
        receipt=receipt, job_identity="job-1", source_schema_hash=plan["source_schema_hash"],
        export_cutoff="x")
    with open(os.path.join(result.final_partition_dir, PR.CATALOG_ENTRY_NAME)) as f:
        entry = json.load(f)
    assert entry["activation_era"] is True
    assert entry["action"] == "CREATE_PRODUCTION_ARCHIVE_ONLY"
    assert "authorization_receipt_sha256" in entry
    assert entry["authorization_identity"] == receipt["authorization_identity"]
    conn.close()


def test_publish_lock_released_after_completion(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    plan, receipt = _receipt_for(conn, root)
    PA.publish_authorized_production_partition(
        conn, root=root, partition=plan["partition"], spec=plan["spec"], archive_version="v1",
        receipt=receipt, job_identity="job-1", source_schema_hash=plan["source_schema_hash"],
        export_cutoff="x")
    assert PA.catalog_lock_status(root)["state"] == "ABSENT"
    conn.close()


def test_reverify_authorized_detects_missing_receipt(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    plan, receipt = _receipt_for(conn, root)
    result = PA.publish_authorized_production_partition(
        conn, root=root, partition=plan["partition"], spec=plan["spec"], archive_version="v1",
        receipt=receipt, job_identity="job-1", source_schema_hash=plan["source_schema_hash"],
        export_cutoff="x")
    os.remove(os.path.join(result.final_partition_dir, PA.AUTHORIZATION_RECEIPT_NAME))
    mismatches = PA.reverify_authorized_partition(result.final_partition_dir, plan["partition"])
    assert mismatches > 0
    conn.close()


# ---------------------------------------------------------------------------
# Gate driver: idempotency + never v2 (Phase 17)
# ---------------------------------------------------------------------------

def test_gate_publish_then_noop(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    r1 = PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                                   justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    assert r1["status"] == "PUBLISHED"
    r2 = PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                                   justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    assert r2["status"] == "NOOP_IDENTICAL_PRODUCTION_ARCHIVE"
    assert r2["reverification_mismatch_count"] == 0
    conn.close()


def test_gate_publish_never_creates_v2(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                              justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                              justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    version_dirs = []
    for _, dirs, _ in os.walk(root):
        version_dirs.extend(d for d in dirs if d.startswith("version="))
    assert version_dirs == ["version=v1"]
    conn.close()


def test_gate_publish_idempotent_rerun_does_not_rewrite(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    r1 = PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                                   justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    parquet = os.path.join(r1["final_partition_dir"], PR.PARQUET_NAME)
    mtime = os.path.getmtime(parquet)
    PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                              justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    assert os.path.getmtime(parquet) == mtime
    conn.close()


# ---------------------------------------------------------------------------
# Root index concurrency-safe rebuild (Phase 6)
# ---------------------------------------------------------------------------

def test_root_index_rebuild_under_lock(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                              justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    idx_path, sha = PA.rebuild_root_index_under_lock(root, job_identity="rebuild-job")
    assert os.path.exists(idx_path)
    assert PA.catalog_lock_status(root)["state"] == "ABSENT"  # released
    conn.close()


def test_root_index_rebuild_deterministic(tmp_path):
    conn = _agg_fixture()
    root = str(tmp_path / "raw_v1")
    PA.gate_publish_partition(conn, root=root, table="agg_trades", approver="gate",
                              justification="t", now=NOW, free_bytes=1000 * 1024 ** 3)
    idx1 = PR.build_root_catalog_index(root)
    idx2 = PR.build_root_catalog_index(root)
    assert idx1["index_self_hash"] == idx2["index_self_hash"]
    assert idx1["entry_count"] == 1
    conn.close()


# ---------------------------------------------------------------------------
# Streaming export / restore parity (verifies the RAM-bounded path)
# ---------------------------------------------------------------------------

def test_streaming_export_matches_fetchall_hash(tmp_path):
    from ami.storage.archive import stream_export_to_parquet, fetch_partition_rows, canonical_row_hash
    conn = _agg_fixture()
    spec = get_table_spec("agg_trades")
    partition = build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                         source_watermark_value=8, now=NOW)
    out = str(tmp_path / "s.parquet")
    r = stream_export_to_parquet(conn, spec, partition, out, batch_size=3, max_output_bytes=10 * 1024 * 1024)
    fetch_rows = fetch_partition_rows(conn, spec, partition)
    assert r["scientific_content_hash"] == canonical_row_hash(fetch_rows)
    assert r["row_count"] == len(fetch_rows)
    conn.close()


def test_streaming_restore_parity(tmp_path):
    from ami.storage.archive import stream_export_to_parquet
    from ami.storage.restorer import stream_restore_slice
    conn = _agg_fixture()
    spec = get_table_spec("agg_trades")
    partition = build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                         source_watermark_value=8, now=NOW)
    parquet = str(tmp_path / "s.parquet")
    r = stream_export_to_parquet(conn, spec, partition, parquet, batch_size=3, max_output_bytes=10 * 1024 * 1024)
    dest = str(tmp_path / ".runtime_temp" / "restored.sqlite")
    manifest = {"ordered_scientific_content_hash": r["scientific_content_hash"]}
    result = stream_restore_slice(destination_path=dest, spec=spec, parquet_path=parquet,
                                  manifest=manifest, expected_scientific_hash=r["scientific_content_hash"],
                                  batch_size=3)
    assert result.row_count == r["row_count"]
    assert result.scientific_content_hash == r["scientific_content_hash"]
    conn.close()


def test_streaming_export_rejects_output_cap(tmp_path):
    from ami.storage.archive import stream_export_to_parquet, ExportValidationError
    conn = _agg_fixture()
    spec = get_table_spec("agg_trades")
    partition = build_partition_identity(table="agg_trades", symbol="ETHUSDT", utc_year=2026, utc_month=2,
                                         source_watermark_value=8, now=NOW)
    with pytest.raises(ExportValidationError):
        stream_export_to_parquet(conn, spec, partition, str(tmp_path / "s.parquet"),
                                 batch_size=3, max_output_bytes=1)  # absurdly small cap
    conn.close()


# ---------------------------------------------------------------------------
# CLI surface (Phase 22)
# ---------------------------------------------------------------------------

def test_cli_has_activation_commands():
    from ami.storage import cli as CLI
    parser = CLI.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    cmds = set(sub.choices.keys())
    for c in ("production-plan", "production-archive-authorized", "production-verify",
             "production-catalog-rebuild", "production-health"):
        assert c in cmds


def test_cli_still_has_no_forbidden_commands():
    from ami.storage import cli as CLI
    parser = CLI.build_parser()
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    cmds = set(sub.choices.keys())
    for forbidden in ("purge", "delete", "vacuum", "schedule", "activate-production",
                     "archive-all", "archive-range", "compact", "stop-collector", "restart-collector"):
        assert forbidden not in cmds


def test_cli_archive_authorized_requires_receipt_path():
    from ami.storage import cli as CLI
    with pytest.raises(SystemExit):
        CLI.parse_args(["production-archive-authorized"])  # missing --receipt-path


# ---------------------------------------------------------------------------
# Acceptance: real production archive (read-only inspection)
# ---------------------------------------------------------------------------

REAL_ROOT = "D:/eclipse_scalper/data/archives/raw_v1"
REAL_MARK_PRICES_DIR = (REAL_ROOT + "/table=mark_prices/venue=BINANCE_USDM_PERP/"
                        "market_segment=PERPETUAL_FUTURES/symbol=ETHUSDT/year=2026/month=05/version=v1")


def test_real_mark_prices_still_immutable_no_receipt():
    """The rehearsal-era mark_prices partition must remain 5-file (no
    retroactive authorization receipt) and byte-identical."""
    assert os.path.isdir(REAL_MARK_PRICES_DIR)
    assert not os.path.exists(os.path.join(REAL_MARK_PRICES_DIR, PA.AUTHORIZATION_RECEIPT_NAME))
    assert PR._sha256_file(os.path.join(REAL_MARK_PRICES_DIR, PR.PARQUET_NAME)) == \
        "6f91914400dcbe84b662c9260a24f9e5eb7f56b2d9db34adfde55a53af8e900f"


def test_real_production_health_all_disabled():
    from ami.storage.health import scan_production_archive_health
    h = scan_production_archive_health(REAL_ROOT)
    assert h["general_unrestricted_activation"] == "DISABLED"
    assert h["scheduler"] == "DISABLED"
    assert h["purge"] == "DISABLED"
    assert h["vacuum"] == "DISABLED"
    assert h["manual_production_archive_creation"] == "ENABLED"
