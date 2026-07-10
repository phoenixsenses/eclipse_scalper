# Liquidation Transport Restoration - 2026-07-10

## Summary

The Binance Futures WebSocket collector regressed back onto the legacy unrouted path
in commit `5cda3122` (2026-07-03), silently reverting the 2026-06-06 fix
(`LIQUIDATION_TRANSPORT_RESTORED_2026-06-06.md`) on an incorrect premise. The
regression became live on the first collector restart after that commit
(2026-07-06T10:32:57Z) and ran for ~4 days, 1 hour before detection.

- Broken (regressed): `wss://fstream.binance.com/stream?streams=...`
- Restored: `wss://fstream.binance.com/market/stream?streams=...`

`!forceOrder@arr` (all-market liquidation mode), REST fallback, and all other
subscription/consumer logic were left untouched.

## Root Cause

`data/microstructure_collector.py:411` hard-coded `BINANCE_WS =
"wss://fstream.binance.com/stream"`. Independent non-Python
(`System.Net.WebSockets.ClientWebSocket`) validation proved this endpoint completes
the TLS/WS handshake but delivers zero application frames, while `/market/stream`
delivers frames within ~1s. Handshake-success was not sufficient evidence of health;
confirming this required treating handshake and application-frame receipt as
separate signals throughout the investigation.

## Evidence

Non-Python routed-endpoint probe (this session, sequential, one fresh
`ClientWebSocket` per test, 30s cap or 10 frames):

| Endpoint | Handshake | App frames | Time to first frame |
| --- | --- | ---: | ---: |
| `wss://fstream.binance.com/stream?streams=...` (legacy, regressed) | Success | 0 | never (30s timeout) |
| `wss://fstream.binance.com/market/stream?streams=!forceOrder@arr/...` | Success | 10 | 0.89s |
| `wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s` | Success | 10 | 1.68s |
| `wss://fstream.binance.com/public/ws/btcusdt@bookTicker` | Success | 10 | 0.94s |

Failure window reconstructed from `logs/microstructure_collector.supervised.out.log`
(3,089 parsed connect-cycles): daily median connect-to-connect duration collapsed
from 4,954s-44,197s (healthy baseline, 06-30/07-01/07-04) to exactly 111.0s with
zero sessions over 600s, starting precisely 2026-07-06T10:32:59Z and continuing
unchanged through 2026-07-10T11:16:50Z (last failed cycle before this fix) - 2,306
of 2,307 reconnects in the final 72h all failed with
`stall_timeout_no_messages>45s`.

Post-restart collector status (supervisor-mediated restart of the `MicroCollector`
child only; `scripts/collector_supervisor.py`, PID 15676 -> 9236, parent 23052
unchanged):

```text
collector_connected=True
collector_transport_connected=True
collector_last_error=
collector_health_status=ok
collector_rows_written_since_start={"agg_trades":23304,"mark_prices":3600,"liquidations":345}
watchdog_overall=GREEN
s34_v_engine_live_executor_alive=False
s34_state_machine_live_executor_alive=False
```

DB verification (read-only, `ORDER BY rowid DESC LIMIT 1`):

```text
liquidations last row before fix:  rowid=1328472  ts=2026-07-06T10:06:39.307Z  (frozen ~4d)
liquidations first new row after:  2026-07-10T11:24:37.685Z  (EVAAUSDT SELL)
liquidations rowid ~20 min later:  1328846 (+374 rows since restart)
```

Stats-loop confirmation (native WS parse counters, in-process, two consecutive
5-minute prints with zero reconnects between them):

```text
[11:22:45] Trades: 21/s (6,202 total)  | Mark: 3/s (900 total)  | Liqs: 0.3/s (105 total)
[11:27:45] Trades: 17/s (11,263 total) | Mark: 3/s (1,800 total)| Liqs: 0.3/s (181 total)
```

## Code Changes

- `data/microstructure_collector.py`
  - `BINANCE_WS` restored to `wss://fstream.binance.com/market/stream` (one line).
  - No other logic, schema, migration, checkpoint, consumer, or executor code changed.

- `tests/test_microstructure_rest_fallback.py`
  - Corrected 2 assertions that had encoded the regressed `/stream` expectation.
  - Added 4 regression tests: pinned `BINANCE_WS` value, no duplicate
    `/market/market/stream` construction in either liquidation mode, exact
    constructed-URL shape for `all_market_arr`, and `BINANCE_REST` unchanged.

## Verification

Focused + adjacent tests, run twice independently (implementation pass and a
separate acceptance-review pass), `--basetemp` scratchpad, `-p no:cacheprovider`:

```text
12 passed, 0 failed (both runs)
```

## Downstream Consumer Recovery (no manual intervention)

`s34_v_engine_v02_shadow_mirror` (PID 11672, untouched throughout) had been frozen
since 2026-07-06T09:46:46Z waiting on liquidation data. Once real liquidations
resumed, its checkpoint began advancing entirely on its own, in the bounded
6-hour (`BOOTSTRAP_CHUNK_SEC`) steps the code already implements:

```text
closed_before_ts_ms: 1783327800000 (2026-07-06T08:50:00Z, frozen 4 days)
             -> 1783349400000 (2026-07-06T14:50:00Z)   [+1 tick]
             -> 1783457400000 (2026-07-07T20:50:00Z)   [+5 ticks, ~15 min later]
```

No checkpoint file was manually edited.

## Permanent Data Gap

Liquidation rows between **2026-07-06T10:06:39Z** and **2026-07-10T11:24:37Z**
(~4 days, 1h 18m) are **permanently unrecoverable**. `data/microstructure_collector.py`
has no REST fallback path for `forceOrder`/liquidations (only `aggTrade` and
`markPrice` are REST-covered), and Binance's public REST API does not offer
historical liquidation backfill. No attempt was made to reconstruct or fabricate
this interval.

## Process / Executor Safety

- Live executors (`s34_v_engine_live_executor`, `s34_state_machine_live_executor`):
  confirmed OFF before, during, and after this change.
- No duplicate `data.microstructure_collector` process was introduced (single PID
  throughout each check).
- No full runtime restart was performed or required - only the `MicroCollector`
  child was cycled, via the existing supervisor's own respawn mechanism.

## Remaining Caveat

This is the second time this exact endpoint has regressed (2026-06-06, then
2026-07-03->07-06). A narrowly-scoped monitoring follow-up is recommended
separately (native-WS-specific staleness age, liquidation-source age, and a
`degraded` overall state when native WebSocket is down even if REST-covered
tables stay fresh) so a third recurrence is caught within minutes rather than
days - not implemented as part of this batch, per instruction.
