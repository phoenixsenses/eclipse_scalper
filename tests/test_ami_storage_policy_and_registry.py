"""Focused tests: ami.storage.policy + ami.storage.registry."""
from __future__ import annotations

import pytest

from ami.storage import policy as P
from ami.storage import registry as R


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

def test_default_policy_valid():
    assert P.DEFAULT_POLICY.active_raw_retention_days == 30
    assert P.DEFAULT_POLICY.archive_format == "PARQUET"
    assert P.DEFAULT_POLICY.compression == "ZSTD"
    assert P.DEFAULT_POLICY.partition_timezone == "UTC"


def test_30_day_minimum_enforced():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(active_raw_retention_days=14)
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(active_raw_retention_days=29)
    P.StoragePolicy(active_raw_retention_days=30)  # boundary: must pass
    P.StoragePolicy(active_raw_retention_days=45)  # above minimum: always permitted


def test_utc_only_partitioning():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(partition_timezone="America/New_York")


def test_parquet_only_archive_format():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(archive_format="CSV")
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(archive_format="JSONL")


def test_zstd_only_compression():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(compression="GZIP")
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(compression="SNAPPY")


def test_deletion_disabled_no_override():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(automatic_purge_enabled=True)


def test_purge_disabled_no_override():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(partial_month_purge_allowed=True)


def test_vacuum_disabled_no_override():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(automatic_vacuum_enabled=True)


def test_production_activation_disabled_no_override():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(production_activation_enabled=True)


def test_scheduler_disabled_no_override():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(scheduler_activation_enabled=True)


def test_unknown_policy_version_rejected():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(policy_version="v2")


def test_manifest_and_verification_requirements_cannot_be_disabled():
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(require_manifest_before_publication=False)
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(require_verification_before_publication=False)
    with pytest.raises(P.PolicyValidationError):
        P.StoragePolicy(require_purge_authorization=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_exactly_three_allowlisted_tables():
    assert R.allowlisted_tables() == ("agg_trades", "book_ticker", "mark_prices")


def test_unknown_table_rejected():
    with pytest.raises(R.UnknownTableError):
        R.get_table_spec("liquidations")
    with pytest.raises(R.UnknownTableError):
        R.get_table_spec("some_arbitrary_table_name")


def test_required_columns_present_for_all_tables():
    for table in R.allowlisted_tables():
        spec = R.get_table_spec(table)
        assert spec.stable_ordering_field in spec.preserved_columns
        assert spec.symbol_field in spec.preserved_columns
        assert spec.partition_ts_field in spec.preserved_columns


def test_stable_ordering_is_id_for_all_tables():
    for table in R.allowlisted_tables():
        assert R.get_table_spec(table).stable_ordering_field == "id"
        assert R.get_table_spec(table).primary_key == "id"


def test_exact_source_to_archive_mappings():
    spec = R.get_table_spec("mark_prices")
    assert spec.archive_types["id"] == "int64"
    assert spec.archive_types["ts_ms"] == "int64"
    assert spec.archive_types["symbol"] == "string"
    assert spec.archive_types["mark_price"] == "double"


def test_no_float_conversion_of_integer_columns():
    for table in R.allowlisted_tables():
        spec = R.get_table_spec(table)
        assert spec.archive_types[spec.stable_ordering_field] == "int64"
        assert spec.archive_types[spec.partition_ts_field] == "int64"


def test_research_dependency_metadata_present():
    for table in R.allowlisted_tables():
        spec = R.get_table_spec(table)
        assert isinstance(spec.research_dependencies, tuple)


def test_purge_default_prohibited_for_all_tables():
    for table in R.allowlisted_tables():
        assert R.get_table_spec(table).purge_default == "PROHIBITED"
        assert R.get_table_spec(table).archive_eligible is True


def test_venue_and_market_segment_consistent():
    for table in R.allowlisted_tables():
        spec = R.get_table_spec(table)
        assert spec.venue == "BINANCE_USDM_PERP"
        assert spec.market_segment == "PERPETUAL_FUTURES"


def test_mark_prices_matches_the_accepted_dry_run_schema():
    """Cross-check against the frozen dry-run schema dict
    (storage_rotation_retention_disposable_dry_run_v1.build_parquet_schema_dict)."""
    from ami.governance import storage_rotation_retention_disposable_dry_run_v1 as D
    frozen = D.build_parquet_schema_dict()
    spec = R.get_table_spec("mark_prices")
    for col, meta in frozen.items():
        assert spec.archive_types[col] == meta["parquet_type"]
        assert (col in spec.nullable_columns) == meta["nullable"]


def test_agg_trades_expected_columns():
    spec = R.get_table_spec("agg_trades")
    assert set(spec.preserved_columns) == {"id", "ts_ms", "symbol", "price", "quantity",
                                            "notional", "is_buyer_maker"}
    assert spec.nullable_columns == ()


def test_book_ticker_expected_columns():
    spec = R.get_table_spec("book_ticker")
    assert set(spec.preserved_columns) == {
        "id", "ts_ms", "symbol", "bid_price", "bid_qty", "ask_price", "ask_qty",
        "mid_price", "spread_pct", "book_imbalance", "bid_depth_usd"}
    assert spec.nullable_columns == ("bid_depth_usd",)
