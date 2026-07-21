# S34 Feature Factory Phase 2 — Multi-Symbol Expansion

Generated: 2026-06-26T13:55:07.200600+00:00

Appends ETH SELL, SOL BUY, SOL SELL, BTC BUY, BTC SELL events to `data/s34_feature_factory.db`.
Uses UPSERT — existing ETH BUY rows are not modified.

DB size after: 1.4 MB

## Results by Symbol-Side

### BTCUSDT BUY
- Threshold: 1,000,000
- Events: 127
- Labels: 254  no-fill: 0
- Date range: 2026-02-17 → 2026-06-26
- Runtime: 89.0s

| Route | N | Median | WR | TP | SL | BE | TIME |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60 | 127 | +22.2 | 54% | 57 | 17 | 40 | 13 |
| SHORT_DELAY0_TP40_CONTROL | 127 | -48.4 | 18% | 20 | 96 | 0 | 11 |

### BTCUSDT SELL
- Threshold: 1,000,000
- Events: 113
- Labels: 339  no-fill: 0
- Date range: 2026-02-18 → 2026-06-26
- Runtime: 28.2s

| Route | N | Median | WR | TP | SL | BE | TIME |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP40_CONTROL | 113 | -48.2 | 19% | 16 | 81 | 3 | 13 |
| SHORT_DELAY0_TP40 | 113 | +32.2 | 74% | 82 | 18 | 0 | 13 |
| SHORT_DELAY0_TP60 | 113 | +25.1 | 55% | 50 | 18 | 22 | 23 |

### ETHUSDT SELL
- Threshold: 500,000
- Events: 222
- Labels: 666  no-fill: 0
- Date range: 2026-02-16 → 2026-06-26
- Runtime: 88.0s

| Route | N | Median | WR | TP | SL | BE | TIME |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP40_CONTROL | 222 | -48.7 | 19% | 37 | 164 | 10 | 11 |
| SHORT_DELAY0_TP60 | 222 | +52.2 | 60% | 123 | 40 | 39 | 20 |
| SHORT_DELAY0_TP80 | 222 | -8.1 | 48% | 85 | 40 | 66 | 31 |

### SOLUSDT BUY
- Threshold: 100,000
- Events: 104
- Labels: 208  no-fill: 0
- Date range: 2026-04-20 → 2026-06-26
- Runtime: 6.4s

| Route | N | Median | WR | TP | SL | BE | TIME |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60 | 104 | +52.1 | 57% | 54 | 14 | 30 | 6 |
| SHORT_DELAY0_TP40_CONTROL | 104 | -49.0 | 19% | 19 | 76 | 0 | 9 |

### SOLUSDT SELL
- Threshold: 100,000
- Events: 105
- Labels: 315  no-fill: 0
- Date range: 2026-04-18 → 2026-06-26
- Runtime: 3.6s

| Route | N | Median | WR | TP | SL | BE | TIME |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP40_CONTROL | 105 | -48.4 | 24% | 23 | 74 | 5 | 3 |
| SHORT_DELAY0_TP40 | 105 | +32.4 | 71% | 75 | 25 | 0 | 5 |
| SHORT_DELAY0_TP60 | 105 | +26.0 | 53% | 51 | 25 | 19 | 10 |

## Verification

```sql
SELECT symbol, liq_side, COUNT(*) FROM liq_event_features GROUP BY symbol, liq_side;
```

## Note

All features in `liq_event_features` are signal-time only (no lookahead).
Route outcomes live exclusively in `liq_event_outcome_labels`.
_Read-only research DB expansion. No runner, config, or pre-reg changes made._
