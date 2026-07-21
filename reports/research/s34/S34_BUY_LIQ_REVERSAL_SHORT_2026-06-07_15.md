# S34 BUY Liquidation Reversal Short Replay

Date: 2026-06-16

Question: besides the active BUY-liq momentum LONG, is there an exhaustion/reversal SHORT after large BUY liquidations?

Model: simplified mark-price replay, flat 8 bps round trip, no real bid/ask live fill parity.

## Top Results

| Rank | Candidate | N | Days | Mean Net | Median Net | Cum Net | WR | Mean MFE | Mean MAE | TP Touch | SL Touch | BE Hit | Exits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | BUY_REVERSAL_SHORT 500000 TP40 DELAY600s | 59 | 4 | -7.77 | -9.40 | -458.64 | 33.9% | +27.09 | -23.11 | 32.2% | 23.7% | 57.6% | {'TP': 19, 'BE': 15, 'SL': 14, 'TIME': 11} |
| 2 | BUY_REVERSAL_SHORT 300000 TP40 DELAY600s | 101 | 4 | -8.32 | -9.43 | -840.75 | 34.7% | +27.15 | -24.45 | 33.7% | 25.7% | 56.4% | {'TP': 34, 'BE': 23, 'SL': 26, 'TIME': 18} |
| 3 | BUY_REVERSAL_SHORT 200000 TP40 DELAY600s | 152 | 4 | -9.20 | -9.46 | -1398.61 | 34.2% | +27.13 | -25.17 | 33.6% | 28.9% | 57.2% | {'TP': 51, 'BE': 36, 'SL': 44, 'TIME': 21} |
| 4 | BUY_REVERSAL_SHORT 100000 TP40 DELAY600s | 273 | 4 | -11.18 | -9.51 | -3050.91 | 33.7% | +26.96 | -26.03 | 32.6% | 34.8% | 54.6% | {'TP': 89, 'BE': 58, 'SL': 95, 'TIME': 31} |
| 5 | BUY_REVERSAL_SHORT 500000 TP60 DELAY600s | 59 | 4 | -12.73 | -9.53 | -751.03 | 16.9% | +30.75 | -23.12 | 13.6% | 23.7% | 57.6% | {'TP': 8, 'BE': 25, 'SL': 14, 'TIME': 12} |
| 6 | BUY_REVERSAL_SHORT 500000 TP80 DELAY600s | 59 | 4 | -14.67 | -9.53 | -865.38 | 13.6% | +32.16 | -23.12 | 5.1% | 23.7% | 57.6% | {'TP': 3, 'BE': 27, 'SL': 14, 'TIME': 15} |
| 7 | BUY_REVERSAL_SHORT 500000 TP120 DELAY600s | 59 | 4 | -16.43 | -9.53 | -969.62 | 11.9% | +32.92 | -23.12 | 0.0% | 23.7% | 57.6% | {'TP': 0, 'BE': 28, 'SL': 14, 'TIME': 17} |
| 8 | BUY_REVERSAL_SHORT 50000 TP40 DELAY600s | 434 | 4 | -12.23 | -9.82 | -5308.69 | 33.2% | +26.24 | -26.34 | 31.8% | 37.6% | 53.5% | {'TP': 138, 'BE': 89, 'SL': 163, 'TIME': 44} |
| 9 | BUY_REVERSAL_SHORT 200000 TP60 DELAY600s | 152 | 4 | -12.36 | -9.82 | -1877.97 | 19.1% | +31.51 | -25.20 | 17.8% | 28.9% | 57.2% | {'TP': 27, 'BE': 59, 'SL': 44, 'TIME': 22} |
| 10 | BUY_REVERSAL_SHORT 200000 TP80 DELAY600s | 152 | 4 | -13.88 | -9.82 | -2110.12 | 15.8% | +33.81 | -25.20 | 7.9% | 28.9% | 57.2% | {'TP': 12, 'BE': 64, 'SL': 44, 'TIME': 32} |
| 11 | BUY_REVERSAL_SHORT 200000 TP120 DELAY600s | 152 | 4 | -16.55 | -9.82 | -2514.92 | 14.5% | +35.31 | -25.20 | 0.0% | 28.9% | 57.2% | {'TP': 0, 'BE': 66, 'SL': 44, 'TIME': 42} |
| 12 | BUY_REVERSAL_SHORT 300000 TP60 DELAY600s | 101 | 4 | -13.87 | -9.91 | -1401.23 | 15.8% | +31.00 | -24.47 | 13.9% | 25.7% | 56.4% | {'TP': 14, 'BE': 42, 'SL': 26, 'TIME': 19} |
| 13 | BUY_REVERSAL_SHORT 300000 TP80 DELAY600s | 101 | 4 | -15.19 | -9.91 | -1533.91 | 13.9% | +32.66 | -24.47 | 5.9% | 25.7% | 56.4% | {'TP': 6, 'BE': 44, 'SL': 26, 'TIME': 25} |
| 14 | BUY_REVERSAL_SHORT 300000 TP120 DELAY600s | 101 | 4 | -17.58 | -9.91 | -1775.61 | 11.9% | +33.31 | -24.47 | 0.0% | 25.7% | 56.4% | {'TP': 0, 'BE': 46, 'SL': 26, 'TIME': 29} |
| 15 | BUY_REVERSAL_SHORT 100000 TP60 DELAY600s | 273 | 4 | -16.08 | -10.08 | -4390.55 | 17.2% | +30.75 | -26.05 | 13.6% | 34.8% | 54.6% | {'TP': 37, 'BE': 103, 'SL': 95, 'TIME': 38} |
| 16 | BUY_REVERSAL_SHORT 100000 TP80 DELAY600s | 273 | 4 | -17.53 | -10.08 | -4785.64 | 14.3% | +32.44 | -26.06 | 5.5% | 34.8% | 54.6% | {'TP': 15, 'BE': 111, 'SL': 95, 'TIME': 52} |
| 17 | BUY_REVERSAL_SHORT 100000 TP120 DELAY600s | 273 | 4 | -19.04 | -10.08 | -5199.11 | 13.6% | +33.45 | -26.06 | 0.4% | 34.8% | 54.6% | {'TP': 1, 'BE': 113, 'SL': 95, 'TIME': 64} |
| 18 | BUY_REVERSAL_SHORT 50000 TP60 DELAY600s | 434 | 4 | -16.11 | -10.24 | -6991.57 | 18.7% | +30.18 | -26.36 | 14.1% | 37.6% | 53.5% | {'TP': 61, 'BE': 152, 'SL': 163, 'TIME': 58} |
| 19 | BUY_REVERSAL_SHORT 50000 TP80 DELAY600s | 434 | 4 | -17.15 | -10.35 | -7444.20 | 16.4% | +31.98 | -26.37 | 6.2% | 37.6% | 53.5% | {'TP': 27, 'BE': 162, 'SL': 163, 'TIME': 82} |
| 20 | BUY_REVERSAL_SHORT 50000 TP120 DELAY600s | 434 | 4 | -19.25 | -10.43 | -8356.45 | 15.2% | +33.09 | -26.37 | 0.5% | 37.6% | 53.5% | {'TP': 2, 'BE': 167, 'SL': 163, 'TIME': 102} |
| 21 | BUY_REVERSAL_SHORT 500000 TP80 DELAY300s | 59 | 4 | -3.08 | -11.35 | -181.66 | 33.9% | +39.35 | -25.74 | 27.1% | 40.7% | 45.8% | {'TP': 16, 'BE': 8, 'SL': 24, 'TIME': 11} |
| 22 | BUY_REVERSAL_SHORT 500000 TP120 DELAY300s | 59 | 4 | -6.23 | -11.35 | -367.53 | 23.7% | +47.85 | -25.77 | 13.6% | 40.7% | 45.8% | {'TP': 8, 'BE': 14, 'SL': 24, 'TIME': 13} |
| 23 | BUY_REVERSAL_SHORT 500000 TP60 DELAY300s | 59 | 4 | -7.85 | -11.35 | -463.33 | 33.9% | +33.91 | -25.74 | 28.8% | 40.7% | 45.8% | {'TP': 17, 'BE': 8, 'SL': 24, 'TIME': 10} |
| 24 | BUY_REVERSAL_SHORT 500000 TP40 DELAY300s | 59 | 4 | -10.46 | -11.35 | -617.21 | 42.4% | +27.25 | -25.70 | 40.7% | 40.7% | 45.8% | {'TP': 24, 'BE': 3, 'SL': 24, 'TIME': 8} |
| 25 | BUY_REVERSAL_SHORT 300000 TP80 DELAY300s | 101 | 4 | -9.01 | -18.21 | -910.49 | 29.7% | +35.85 | -28.47 | 22.8% | 46.5% | 42.6% | {'TP': 23, 'BE': 15, 'SL': 47, 'TIME': 16} |

## Day Split For Top Candidate

Candidate: `BUY_REVERSAL_SHORT 500000 TP40 DELAY600s`

| Day | N | Cum Net | Mean Net | Median Net |
|---|---:|---:|---:|---:|
| 2026-06-07 | 22 | +262.95 | +11.95 | +32.49 |
| 2026-06-11 | 4 | -82.93 | -20.73 | -28.64 |
| 2026-06-14 | 11 | -97.41 | -8.86 | -8.08 |
| 2026-06-15 | 22 | -541.26 | -24.60 | -29.61 |

## Verdict

BUY liquidation reversal SHORT is not viable in this window. Even the best-ranked candidate is negative after costs:

- `BUY_REVERSAL_SHORT 500000 TP40 DELAY600s`
- N `59`, days `4`
- mean net `-7.77 bps`
- median net `-9.40 bps`
- cumulative net `-458.64 bps`

The grid says the same thing across thresholds: waiting longer helps reduce immediate squeeze damage, but not enough to overcome fees and adverse continuation. This supports the current architecture: BUY liquidation is primarily a LONG/momentum signal in the observed data, not a SHORT exhaustion signal.

Decision: kill BUY-liq reversal SHORT for now. Do not add it to live paper.
