# Liquidation Transport Restoration - 2026-06-06

## Summary

Liquidation transport was restored by moving the Binance Futures WebSocket collector from the legacy unrouted path to the routed market path:

- Previous: `wss://fstream.binance.com/stream?streams=...`
- Current: `wss://fstream.binance.com/market/stream?streams=...`

The collector now subscribes to global liquidation stream `!forceOrder@arr` plus symbol-level `aggTrade` and `markPrice@1s` streams.

## Evidence

Public WebSocket probes before the code change:

| Endpoint | Frames |
| --- | ---: |
| `wss://fstream.binance.com/ws/!forceOrder@arr` | 0 |
| `wss://fstream.binance.com/ws/ethusdt@forceOrder` | 0 |
| `wss://fstream.binance.com/ws/btcusdt@forceOrder` | 0 |
| `wss://fstream.binance.com/ws/ethusdt@bookTicker` | 2813 |

Public WebSocket probes on routed `/market` path:

| Endpoint | Window | Frames |
| --- | ---: | ---: |
| `wss://fstream.binance.com/market/ws/ethusdt@markPrice@1s` | 13s | 7 |
| `wss://fstream.binance.com/market/ws/ethusdt@aggTrade` | 13s | 38 |
| `wss://fstream.binance.com/market/ws/!forceOrder@arr` | 30s | 10 |

Post-restart collector status:

```text
collector_supervisor_alive=True
collector_alive=True
event_diary_alive=True
heartbeat_watchdog_alive=True
collector_connected=True
collector_rest_fallback_active=False
collector_rows_written_since_start={"agg_trades":14337,"mark_prices":891,"liquidations":120}
collector_liquidation_transport_available=True
watchdog_overall=GREEN
```

DB verification over the first post-restart window:

```text
LIQ_LAST_10M_TOTAL: 137
ETHUSDT: 7
BTCUSDT: 5
SOLUSDT: 2
```

## Code Changes

- `data/microstructure_collector.py`
  - `BINANCE_WS` changed to `wss://fstream.binance.com/market/stream`.
  - Added `liquidation_stream_mode`, default `all_market_arr`.
  - Added CLI flag `--liquidation-stream-mode`.
  - Parser now recognizes `!forceOrder@arr` because the stream name does not contain `@forceOrder`.

- `tests/test_microstructure_rest_fallback.py`
  - Added parser coverage for `!forceOrder@arr`.
  - Added URL strategy tests for routed `/market/stream`.

## Verification

Focused tests:

```text
10 passed
```

Runtime validation:

- Collector restarted cleanly.
- WebSocket data is flowing without REST fallback active.
- Liquidation rows are being persisted to `data/microstructure.db`.

## Remaining Caveat

This restores the transport layer. It does not by itself prove S34 signal quality. S34 still needs forward validation with the restored live liquidation feed.
