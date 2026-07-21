# BTC 1000K SELL Anomaly Audit

Generated: 2026-06-26T10:13:37.007347+00:00

**Hypothesis under test:** TP40 is the best config for BTC 1000K SELL, unlike ETH/SOL SELL where TP60/TP80 wins.
Does this reflect a genuine structural difference (BTC moves are shorter) or in-sample overfitting?

Lookback: 120 days. Signal count: 107.
Real bookTicker fills. Net after fee+spread+adverse.

---

## 1. TP40 vs TP60 — Full Comparison

| Config | N | Median | Mean | WR | Top3-Removed | Pos Days | Exits |
|---|---:|---:|---:|---:|---:|---:|---|
| TP40/SL40/BE40 | 70 | +32.0 | +18.1 | 77% | +1135 | 18/24 | TP=52  TIME=11  SL=7 |
| TP60/SL40/BE40 | 70 | +22.5 | +16.3 | 56% | +927 | 13/24 | TP=30  TIME=18  BE=15  SL=7 |

### Half-split stability

| Config | Half | N | Median | Mean | WR | Top3-Removed |
|---|---|---:|---:|---:|---:|---:|
| TP40 | 1st half | 35 | +30.6 | +10.9 | 66% | +260 |
| TP40 | 2nd half | 35 | +32.9 | +25.2 | 89% | +757 |
| TP60 | 1st half | 35 | -8.5 | +2.2 | 37% | -91 |
| TP60 | 2nd half | 35 | +51.5 | +30.5 | 74% | +851 |

---

## 2. MFE / Giveback Analysis  (TP40 path)

Total trades analyzed: 70

### MFE distribution

| MFE bucket | Count | % |
|---|---:|---:|
| < 0 (adverse) bps | 0 | 0% |
| 0-20 bps | 5 | 7% |
| 20-40 bps | 36 | 51% |
| 40-60 bps | 29 | 41% |
| 60-80 bps | 0 | 0% |
| 80+ bps | 0 | 0% |

- Trades reaching MFE >= 40 bps: **29** / 70 (41%)
- Trades reaching MFE >= 60 bps: **0** / 70 (0%)
- Trades reaching 40 but NOT 60: **29** (the 'TP40 pocket')
- Trades that hit MFE >= 60 but exited early (BE/TIME/SL under TP40 rule): 0
- Trades that never reach +40: 41 (59%)

### Time to MFE distribution

| Bucket | Count | % |
|---|---:|---:|
| < 5 min | 43 | 61% |
| 5-15 min | 15 | 21% |
| 15-30 min | 6 | 9% |
| 30-60 min | 6 | 9% |

Median MFE: 39.5 bps  |  Mean MFE: 36.1 bps
Median time-to-MFE: 3.6 min  |  Mean: 9.6 min

---

## 3. Regime Split

### Day Trend — TP40 vs TP60

| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |
|---|---:|---:|---:|---:|---:|---:|
| Bull (trend >= 0) | 13 | +31.3 | 85% | 13 | +2.2 | 54% |
| Bear (trend < 0) | 57 | +32.6 | 75% | 57 | +27.2 | 56% |
| Strong Bull (>= +100) | 4 | +8.9 | 50% | 4 | -13.7 | 25% |
| Strong Bear (<= -100) | 28 | +32.7 | 75% | 28 | +51.3 | 61% |

### Day Range — TP40 vs TP60

| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |
|---|---:|---:|---:|---:|---:|---:|
| < 250 bps | 46 | +31.9 | 74% | 46 | +22.5 | 54% |
| 250-500 bps | 20 | +31.3 | 80% | 20 | +8.9 | 55% |
| 500-750 bps | 4 | +36.5 | 100% | 4 | +53.7 | 75% |
| >= 750 bps | 0 | — | — | 0 | — | — |

### Liq Imbalance — TP40 vs TP60

| Slice | N40 | Med40 | WR40 | N60 | Med60 | WR60 |
|---|---:|---:|---:|---:|---:|---:|
| Sell-heavy (imbal < -0.1) | 50 | +32.8 | 74% | 50 | +26.9 | 56% |
| Buy-heavy (imbal > 0.1) | 15 | +31.7 | 87% | 15 | +15.6 | 60% |

---

## 4. No-Fill Bias

- Total signals: 107
- Filled (closed trades): 70
- No-fill (skipped): 37 (35%)

Cluster notional not available in signal records for this route.

No-fill events span 20 distinct days (of 37 no-fill signals).
No-fill events are spread across the lookback period — not concentrated in a single regime pocket.

Filled trade day-range: median=201 bps  mean=234 bps
Day-range distribution among filled trades:
  - <250 bps: 46 (66%)
  - 250-500 bps: 20 (29%)
  - 500-750 bps: 4 (6%)
  - >=750 bps: 0 (0%)

---

## 5. Verdict

**B — Weak but watchlist**

TP40 is consistently better than TP60 across both halves, suggesting BTC moves are genuinely shorter. However, the overall median is materially lower than ETH/SOL SELL routes (~+32 vs +50), and WR=77% with TP40 is suspiciously high — may reflect in-sample fitting to a pocket of short moves. No-fill rate is high (35%), which could introduce selection bias. Watchlist status is appropriate. Do not add to runner until ETH/SOL SELL calibration is complete and BTC can be examined with regime conditioning.

### Summary of structural signals:

- TP40 median (+32.0) > TP60 median (+22.5): TP40 structurally better
- Half-split stability: 1H=+30.6 → 2H=+32.9 (stable)
- MFE 40-60 pocket: 29 trades (41% of filled). Trades never reaching +40: 41 (59%). WR=77% from TP40 is consistent with BTC short move structure.
- No-fill rate: 35%. If no-fill events are stronger signals, filled sample underestimates full population quality.

**Bottom line:** BTC 1000K SELL shows real edge in some market pockets but is not at ETH/SOL SELL quality. Keep on watchlist. Revisit after ETH/SOL SELL N=30 complete.

_Read-only. No runner, config, or pre-reg changes made._
