"""Source-table registry (Phase 3): an allowlist of exactly the three
archive-eligible raw high-frequency tables accepted in the storage
readiness batch (`f65545ee`) -- `mark_prices`, `agg_trades`,
`book_ticker`. This is an allowlist, not a discovery mechanism: no code
in this package ever archives a table that is not registered here, and
an unregistered table name raises `UnknownTableError` rather than being
silently accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SOURCE_DATABASE = "microstructure.db"
VENUE = "BINANCE_USDM_PERP"
MARKET_SEGMENT = "PERPETUAL_FUTURES"
QUOTE_CURRENCY = "USDT"


class UnknownTableError(Exception):
    """Raised when a caller requests a table not present in
    SOURCE_TABLE_REGISTRY -- the registry is a strict allowlist."""


@dataclass
class SourceTableSpec:
    table: str
    source_database: str
    producer: str
    primary_key: str
    partition_ts_field: str
    receipt_ts_field: str
    exchange_ts_field: str | None
    symbol_field: str
    venue: str
    market_segment: str
    quote_currency: str
    preserved_columns: tuple[str, ...]
    nullable_columns: tuple[str, ...]
    source_types: dict[str, str]
    archive_types: dict[str, str]
    stable_ordering_field: str
    expected_indexes: tuple[str, ...]
    gap_ledger_stream: str | None
    repair_layer: str | None
    research_dependencies: tuple[str, ...]
    archive_eligible: bool = True
    purge_default: str = "PROHIBITED"
    direct_read_supported: bool = True
    minimal_restore_supported: bool = True


SOURCE_TABLE_REGISTRY: dict[str, SourceTableSpec] = {
    "mark_prices": SourceTableSpec(
        table="mark_prices", source_database=SOURCE_DATABASE,
        producer="mark-price/funding collector",
        primary_key="id", partition_ts_field="ts_ms", receipt_ts_field="ts_ms",
        exchange_ts_field=None, symbol_field="symbol",
        venue=VENUE, market_segment=MARKET_SEGMENT, quote_currency=QUOTE_CURRENCY,
        preserved_columns=("id", "ts_ms", "symbol", "mark_price", "funding_rate", "next_funding_time_ms"),
        nullable_columns=("funding_rate", "next_funding_time_ms"),
        source_types={"id": "INTEGER", "ts_ms": "INTEGER", "symbol": "TEXT", "mark_price": "REAL",
                       "funding_rate": "REAL", "next_funding_time_ms": "INTEGER"},
        archive_types={"id": "int64", "ts_ms": "int64", "symbol": "string", "mark_price": "double",
                        "funding_rate": "double", "next_funding_time_ms": "int64"},
        stable_ordering_field="id",
        expected_indexes=("idx_mark_ts", "idx_mark_symbol_ts"),
        gap_ledger_stream="mark_prices", repair_layer=None,
        research_dependencies=("FUNDING_LEVEL_VELOCITY", "FAM_SPOT_PERP_BASIS_REVERSAL"),
    ),
    "agg_trades": SourceTableSpec(
        table="agg_trades", source_database=SOURCE_DATABASE,
        producer="trade-tick collector",
        primary_key="id", partition_ts_field="ts_ms", receipt_ts_field="ts_ms",
        exchange_ts_field=None, symbol_field="symbol",
        venue=VENUE, market_segment=MARKET_SEGMENT, quote_currency=QUOTE_CURRENCY,
        preserved_columns=("id", "ts_ms", "symbol", "price", "quantity", "notional", "is_buyer_maker"),
        nullable_columns=(),
        source_types={"id": "INTEGER", "ts_ms": "INTEGER", "symbol": "TEXT", "price": "REAL",
                       "quantity": "REAL", "notional": "REAL", "is_buyer_maker": "INTEGER"},
        archive_types={"id": "int64", "ts_ms": "int64", "symbol": "string", "price": "double",
                        "quantity": "double", "notional": "double", "is_buyer_maker": "int64"},
        stable_ordering_field="id",
        expected_indexes=("idx_trade_ts", "idx_trade_symbol_ts"),
        gap_ledger_stream="agg_trades", repair_layer="ami_agg_trades_repaired (canonical)",
        research_dependencies=("execution_fill_studies_future", "FAM_CASCADE_ABSORPTION_IMPACT_historical"),
    ),
    "book_ticker": SourceTableSpec(
        table="book_ticker", source_database=SOURCE_DATABASE,
        producer="L1 book-ticker collector",
        primary_key="id", partition_ts_field="ts_ms", receipt_ts_field="ts_ms",
        exchange_ts_field=None, symbol_field="symbol",
        venue=VENUE, market_segment=MARKET_SEGMENT, quote_currency=QUOTE_CURRENCY,
        preserved_columns=("id", "ts_ms", "symbol", "bid_price", "bid_qty", "ask_price", "ask_qty",
                            "mid_price", "spread_pct", "book_imbalance", "bid_depth_usd"),
        nullable_columns=("bid_depth_usd",),
        source_types={"id": "INTEGER", "ts_ms": "INTEGER", "symbol": "TEXT", "bid_price": "REAL",
                       "bid_qty": "REAL", "ask_price": "REAL", "ask_qty": "REAL", "mid_price": "REAL",
                       "spread_pct": "REAL", "book_imbalance": "REAL", "bid_depth_usd": "REAL"},
        archive_types={"id": "int64", "ts_ms": "int64", "symbol": "string", "bid_price": "double",
                        "bid_qty": "double", "ask_price": "double", "ask_qty": "double",
                        "mid_price": "double", "spread_pct": "double", "book_imbalance": "double",
                        "bid_depth_usd": "double"},
        stable_ordering_field="id",
        expected_indexes=("idx_bt_symbol_ts", "idx_bt_ts"),
        gap_ledger_stream=None, repair_layer=None,
        research_dependencies=("FAM_BOOK_SPREAD_DYNAMICS_LONG",),
    ),
}


def get_table_spec(table: str) -> SourceTableSpec:
    if table not in SOURCE_TABLE_REGISTRY:
        raise UnknownTableError(
            f"table {table!r} is not in the archive-eligible allowlist "
            f"({sorted(SOURCE_TABLE_REGISTRY)})")
    return SOURCE_TABLE_REGISTRY[table]


def allowlisted_tables() -> tuple[str, ...]:
    return tuple(sorted(SOURCE_TABLE_REGISTRY))
