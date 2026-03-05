# DB Schema Report

- DB: `data\microstructure.db`
- Tables: 3

## Likely Core Tables

- trades: `agg_trades`
- book: `mark_prices`
- liquidations: `liquidations`

## Table Summary

| table | rows | ts_col | min_ts | max_ts |
|---|---:|---|---:|---:|
| `agg_trades` | 56344284 | `ts_ms` | 1771165587.967 | 1772425617.354 |
| `liquidations` | 39235 | `ts_ms` | 1771165818.195 | 1772425615.115 |
| `mark_prices` | 2514044 | `ts_ms` | 1771165588.0 | 1772425617.001 |

## agg_trades

- rows: 56344284
- timestamp candidates: `ts_ms`
- chosen timestamp: `ts_ms`

### Columns

| name | type | notnull | pk |
|---|---|---:|---:|
| `id` | `INTEGER` | 0 | 1 |
| `ts_ms` | `INTEGER` | 1 | 0 |
| `symbol` | `TEXT` | 1 | 0 |
| `price` | `REAL` | 1 | 0 |
| `quantity` | `REAL` | 1 | 0 |
| `notional` | `REAL` | 1 | 0 |
| `is_buyer_maker` | `INTEGER` | 1 | 0 |

### Indexes

- `idx_trade_symbol_ts` unique=0 columns=[symbol, ts_ms]
- `idx_trade_ts` unique=0 columns=[ts_ms]

## liquidations

- rows: 39235
- timestamp candidates: `ts_ms, trade_time_ms`
- chosen timestamp: `ts_ms`

### Columns

| name | type | notnull | pk |
|---|---|---:|---:|
| `id` | `INTEGER` | 0 | 1 |
| `ts_ms` | `INTEGER` | 1 | 0 |
| `symbol` | `TEXT` | 1 | 0 |
| `side` | `TEXT` | 1 | 0 |
| `price` | `REAL` | 1 | 0 |
| `quantity` | `REAL` | 1 | 0 |
| `notional` | `REAL` | 1 | 0 |
| `trade_time_ms` | `INTEGER` | 1 | 0 |

### Indexes

- `idx_liq_symbol_ts` unique=0 columns=[symbol, ts_ms]
- `idx_liq_ts` unique=0 columns=[ts_ms]

## mark_prices

- rows: 2514044
- timestamp candidates: `ts_ms`
- chosen timestamp: `ts_ms`

### Columns

| name | type | notnull | pk |
|---|---|---:|---:|
| `id` | `INTEGER` | 0 | 1 |
| `ts_ms` | `INTEGER` | 1 | 0 |
| `symbol` | `TEXT` | 1 | 0 |
| `mark_price` | `REAL` | 1 | 0 |
| `funding_rate` | `REAL` | 0 | 0 |
| `next_funding_time_ms` | `INTEGER` | 0 | 0 |

### Indexes

- `idx_mark_symbol_ts` unique=0 columns=[symbol, ts_ms]
- `idx_mark_ts` unique=0 columns=[ts_ms]

