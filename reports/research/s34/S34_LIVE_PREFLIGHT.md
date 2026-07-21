# S34 Live Preflight

- generated_at_utc: `2026-06-26T22:12:32.307407+00:00`
- mode: `READ_ONLY_PREFLIGHT_NO_ORDERS`
- config: `$40.00` margin x `40` = `$1600.00` notional
- 1 bps value: `$0.1600`

## Environment

| Check | Value |
|---|---:|
| API keys present | False |
| Testnet | False |
| Dry run | True |
| Live trading enabled | False |
| Safety | SAFE_DISABLED |
| Status | FAIL |

## Stream Health

| Table | Last UTC | Age sec | Rows 1h | Status |
|---|---:|---:|---:|---:|
| liquidations | 2026-06-26T22:12:24.934000+00:00 | +7.4 | 631 | PASS |
| book_ticker | 2026-06-26T22:12:33.330000+00:00 | -1.0 | 2084304 | PASS |
| mark_prices | 2026-06-26T22:12:33+00:00 | -0.7 | 10803 | PASS |
| agg_trades | 2026-06-26T22:12:32.997000+00:00 | -0.7 | 72465 | PASS |

## Rule Math And Geometry

| Rule | N 7d | WR | Median bps | Cum bps | Worst bps | Median $ | Worst $ | RR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | 15 | 80.0% | +53.46 | +640.43 | -51.64 | +8.55 | -8.26 | +1.50 | PASS |
| `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 24 | 62.5% | +48.45 | +640.40 | -63.16 | +7.75 | -10.11 | +1.50 | PASS |

## Overall

- math_geometry_pass: `True`
- streams_pass: `True`
- env_ready: `False`
- live_ready: `False`
- live_armed: `False`

No orders are placed by this script.
