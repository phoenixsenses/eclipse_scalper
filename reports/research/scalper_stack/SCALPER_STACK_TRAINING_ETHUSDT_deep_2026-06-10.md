# Scalper Stack Training — ETHUSDT (2026-06-10)

- seed: 42 (deterministic; no randomness used)
- config: fee=5.0 bps/side, slip=1.0 bps/side, folds=4, min_train_trades=30
- data: 139823 one-minute bars, 122 gaps > 2 min, range 2026-02-15 14:26 → 2026-06-10 11:51 UTC
- grid: 432 configs, eligible after train filters: 12

Selection uses folds 1..N-1 (train) only. The final fold is holdout —
reported for the top configs but never used for ranking.

| rank | ema | buf | sep | hold | delta | cool | exit | side | train N | train bps | train WR | hold N | hold bps | hold WR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9/21 | 0.5 | 0.6 | 4 | strong2.0 | 15 | h120 | long | 108 | 7.59 | 43% | 42 | -35.22 | 29% |
| 2 | 9/21 | 0.75 | 0.6 | 4 | strong2.0 | 15 | h120 | long | 104 | 6.72 | 40% | 40 | -34.26 | 28% |
| 3 | 9/21 | 1.0 | 0.6 | 4 | strong2.0 | 15 | h120 | long | 110 | 5.77 | 38% | 41 | -41.92 | 27% |
| 4 | 5/13 | 0.5 | 0.6 | 4 | strong2.0 | 15 | h120 | long | 141 | 1.86 | 42% | 51 | -24.07 | 27% |
| 5 | 5/13 | 0.75 | 0.6 | 4 | strong2.0 | 15 | h120 | long | 139 | 0.44 | 40% | 51 | -23.03 | 25% |
| 6 | 9/21 | 0.5 | 0.6 | 4 | strong2.0 | 15 | h120 | both | 200 | 0.02 | 46% | 96 | -24.88 | 38% |
| 7 | 9/21 | 0.75 | 0.6 | 4 | strong2.0 | 15 | h120 | both | 196 | -0.31 | 45% | 92 | -22.90 | 37% |
| 8 | 9/21 | 0.5 | 0.6 | 4 | strong1.5 | 15 | h120 | long | 148 | -0.34 | 41% | 58 | -20.76 | 31% |
| 9 | 5/13 | 0.75 | 0.4 | 4 | strong2.0 | 15 | h120 | short | 194 | -0.38 | 45% | 62 | 25.53 | 42% |
| 10 | 9/21 | 1.0 | 0.6 | 4 | strong1.5 | 15 | h120 | long | 146 | -1.15 | 38% | 55 | -24.61 | 31% |
