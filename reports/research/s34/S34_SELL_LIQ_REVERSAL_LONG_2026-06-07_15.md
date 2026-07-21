# S34 SELL Liquidation Reversal Long Replay

Date: 2026-06-16

Question: if SELL liquidations are not clean continuation shorts, do they work better as capitulation/reversal LONG entries?

Model: simplified mark-price replay, flat 8 bps round trip, no real bid/ask live fill parity.

## Top Results

| Rank | Candidate | N | Days | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | TP Touch | SL Touch | BE Hit | Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SELL_REVERSAL_LONG 300000 TP40 DELAY300s | 12 | 4 | +15.79 | +32.34 | +189.50 | 75.0% | +33.31 | -15.98 | 75.0% | 16.7% | 75.0% | {'TP': 9, 'BE': 0, 'SL': 2, 'TIME': 1} |
| 2 | SELL_REVERSAL_LONG 200000 TP40 DELAY300s | 33 | 4 | +9.24 | +32.33 | +305.06 | 66.7% | +32.15 | -20.90 | 66.7% | 24.2% | 69.7% | {'TP': 22, 'BE': 1, 'SL': 8, 'TIME': 2} |
| 3 | SELL_REVERSAL_LONG 300000 TP60 DELAY300s | 12 | 4 | +8.29 | +11.96 | +99.46 | 50.0% | +39.61 | -16.06 | 16.7% | 16.7% | 75.0% | {'TP': 2, 'BE': 3, 'SL': 2, 'TIME': 5} |
| 4 | SELL_REVERSAL_LONG 200000 TP60 DELAY300s | 33 | 4 | +4.48 | -8.09 | +147.79 | 42.4% | +40.44 | -20.94 | 27.3% | 24.2% | 69.7% | {'TP': 9, 'BE': 9, 'SL': 8, 'TIME': 7} |
| 5 | SELL_REVERSAL_LONG 300000 TP40 DELAY60s | 12 | 4 | +2.04 | +19.88 | +24.43 | 58.3% | +29.99 | -25.32 | 41.7% | 25.0% | 66.7% | {'TP': 5, 'BE': 1, 'SL': 3, 'TIME': 3} |
| 6 | SELL_REVERSAL_LONG 300000 TP80 DELAY120s | 12 | 4 | +1.55 | -6.60 | +18.56 | 41.7% | +35.07 | -23.48 | 16.7% | 33.3% | 50.0% | {'TP': 2, 'BE': 1, 'SL': 4, 'TIME': 5} |
| 7 | SELL_REVERSAL_LONG 200000 TP40 DELAY60s | 33 | 4 | +1.06 | +17.52 | +34.88 | 51.5% | +31.95 | -20.78 | 45.5% | 24.2% | 66.7% | {'TP': 15, 'BE': 5, 'SL': 8, 'TIME': 5} |
| 8 | SELL_REVERSAL_LONG 100000 TP40 DELAY300s | 73 | 4 | +0.50 | +32.02 | +36.78 | 50.7% | +29.10 | -23.35 | 50.7% | 28.8% | 63.0% | {'TP': 37, 'BE': 8, 'SL': 21, 'TIME': 7} |
| 9 | SELL_REVERSAL_LONG 100000 TP40 DELAY60s | 73 | 4 | +0.17 | -8.05 | +12.37 | 47.9% | +31.14 | -18.37 | 45.2% | 23.3% | 65.8% | {'TP': 33, 'BE': 13, 'SL': 17, 'TIME': 10} |
| 10 | SELL_REVERSAL_LONG 200000 TP60 DELAY60s | 33 | 4 | +0.03 | -8.50 | +1.13 | 36.4% | +38.05 | -20.78 | 27.3% | 24.2% | 66.7% | {'TP': 9, 'BE': 10, 'SL': 8, 'TIME': 6} |
| 11 | SELL_REVERSAL_LONG 50000 TP40 DELAY60s | 141 | 4 | -0.35 | -8.05 | -49.77 | 46.8% | +30.93 | -18.79 | 45.4% | 24.8% | 65.2% | {'TP': 64, 'BE': 26, 'SL': 35, 'TIME': 16} |
| 12 | SELL_REVERSAL_LONG 100000 TP60 DELAY60s | 73 | 4 | -0.83 | -8.62 | -60.45 | 34.2% | +36.99 | -18.48 | 24.7% | 23.3% | 65.8% | {'TP': 18, 'BE': 23, 'SL': 17, 'TIME': 15} |
| 13 | SELL_REVERSAL_LONG 300000 TP60 DELAY120s | 12 | 4 | -1.56 | -6.60 | -18.71 | 41.7% | +31.96 | -23.48 | 16.7% | 33.3% | 50.0% | {'TP': 2, 'BE': 1, 'SL': 4, 'TIME': 5} |
| 14 | SELL_REVERSAL_LONG 50000 TP60 DELAY60s | 141 | 4 | -1.86 | -8.59 | -261.66 | 33.3% | +36.94 | -18.86 | 22.0% | 24.8% | 65.2% | {'TP': 31, 'BE': 45, 'SL': 35, 'TIME': 30} |
| 15 | SELL_REVERSAL_LONG 300000 TP60 DELAY60s | 12 | 4 | -2.20 | -8.25 | -26.39 | 41.7% | +34.39 | -25.32 | 16.7% | 25.0% | 66.7% | {'TP': 2, 'BE': 3, 'SL': 3, 'TIME': 4} |
| 16 | SELL_REVERSAL_LONG 300000 TP80 DELAY300s | 12 | 4 | -2.28 | -8.48 | -27.31 | 33.3% | +39.78 | -16.06 | 0.0% | 16.7% | 75.0% | {'TP': 0, 'BE': 5, 'SL': 2, 'TIME': 5} |
| 17 | SELL_REVERSAL_LONG 300000 TP120 DELAY300s | 12 | 4 | -2.28 | -8.48 | -27.31 | 33.3% | +39.78 | -16.06 | 0.0% | 16.7% | 75.0% | {'TP': 0, 'BE': 5, 'SL': 2, 'TIME': 5} |
| 18 | SELL_REVERSAL_LONG 200000 TP80 DELAY120s | 33 | 4 | -2.33 | -12.57 | -76.82 | 33.3% | +39.25 | -23.37 | 21.2% | 36.4% | 51.5% | {'TP': 7, 'BE': 6, 'SL': 12, 'TIME': 8} |
| 19 | SELL_REVERSAL_LONG 100000 TP60 DELAY300s | 73 | 4 | -2.82 | -8.32 | -205.67 | 32.9% | +35.54 | -23.37 | 21.9% | 28.8% | 63.0% | {'TP': 16, 'BE': 21, 'SL': 21, 'TIME': 15} |
| 20 | SELL_REVERSAL_LONG 100000 TP40 DELAY0s | 73 | 4 | -2.95 | -8.17 | -215.67 | 42.5% | +29.29 | -18.34 | 39.7% | 27.4% | 67.1% | {'TP': 29, 'BE': 18, 'SL': 20, 'TIME': 6} |

## Verdict

## Day Split For Main Pocket

Candidate: `SELL_REVERSAL_LONG 200000 TP40 DELAY300s`

| Day | N | Cum Net | Mean Net | Median Net |
|---|---:|---:|---:|---:|
| 2026-06-07 | 10 | -165.76 | -16.58 | -48.47 |
| 2026-06-11 | 5 | +165.01 | +33.00 | +33.04 |
| 2026-06-14 | 5 | +81.57 | +16.31 | +32.02 |
| 2026-06-15 | 13 | +224.24 | +17.25 | +32.43 |

Outlier check for the same candidate: full cum `+305.06`; removing the top 3 winners leaves `+202.70`, so it is not only one or two prints. The weakness is regime split: 2026-06-07 is sharply negative while the other three days are positive.

## Verdict

SELL liquidation reversal LONG is more interesting than immediate SELL->SHORT continuation, but it is still not ready for active paper. The best pocket is delayed mean reversion: wait 300 seconds after a >=200K ETH SELL liquidation, then LONG with TP40/SL40/BE30. It works on 2026-06-11, 06-14, and 06-15, but fails badly on 2026-06-07. This needs a regime discriminator before it can become an exploratory rule.
