# S34 Live Preflight

- generated_at_utc: `2026-06-26T22:13:58.940313+00:00`
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
| liquidations | 2026-06-26T22:13:45.265000+00:00 | +13.7 | 629 | PASS |
| book_ticker | 2026-06-26T22:13:59.671000+00:00 | -0.7 | 2084011 | PASS |
| mark_prices | 2026-06-26T22:13:59+00:00 | -0.1 | 10803 | PASS |
| agg_trades | 2026-06-26T22:13:59.869000+00:00 | -0.9 | 72534 | PASS |

## Rule Math And Geometry

| Rule | N 7d | WR | Median bps | Cum bps | Worst bps | Median $ | Worst $ | RR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | 19 | 78.9% | +52.33 | +743.26 | -51.64 | +8.37 | -8.26 | +1.50 | PASS |
| `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 24 | 62.5% | +48.45 | +640.40 | -63.16 | +7.75 | -10.11 | +1.50 | PASS |

## Overall

- math_geometry_pass: `True`
- streams_pass: `True`
- env_ready: `False`
- live_ready: `False`
- live_armed: `False`

No orders are placed by this script.
