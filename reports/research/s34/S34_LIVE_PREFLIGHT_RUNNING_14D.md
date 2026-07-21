# S34 Live Preflight

- generated_at_utc: `2026-06-26T22:30:32.050535+00:00`
- mode: `READ_ONLY_PREFLIGHT_NO_ORDERS`
- config: `$40.00` margin x `40` = `$1600.00` notional
- 1 bps value: `$0.1600`

## Environment

| Check | Value |
|---|---:|
| API keys present | True |
| Testnet | False |
| Dry run | False |
| Live trading enabled | True |
| Safety | ARMED |
| Status | PASS |

## Stream Health

| Table | Last UTC | Age sec | Rows 1h | Status |
|---|---:|---:|---:|---:|
| liquidations | 2026-06-26T22:30:30.112000+00:00 | +1.9 | 642 | PASS |
| book_ticker | 2026-06-26T22:30:29.680000+00:00 | +2.4 | 2075779 | PASS |
| mark_prices | 2026-06-26T22:30:33+00:00 | -0.9 | 10801 | PASS |
| agg_trades | 2026-06-26T22:30:32.819000+00:00 | -0.8 | 70971 | PASS |

## Rule Math And Geometry

| Rule | N window | WR | Median bps | Cum bps | Worst bps | Median $ | Worst $ | RR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | 19 | 78.9% | +52.33 | +743.26 | -51.64 | +8.37 | -8.26 | +1.50 | PASS |
| `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | 24 | 62.5% | +48.45 | +640.40 | -63.16 | +7.75 | -10.11 | +1.50 | PASS |

## Overall

- math_geometry_pass: `True`
- streams_pass: `True`
- env_ready: `True`
- live_ready: `True`
- live_armed: `True`

No orders are placed by this script.
