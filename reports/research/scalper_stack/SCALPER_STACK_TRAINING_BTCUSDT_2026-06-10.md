# Scalper Stack Training — BTCUSDT (2026-06-10)

- seed: 42 (deterministic; no randomness used)
- config: fee=5.0 bps/side, slip=1.0 bps/side, folds=4, min_train_trades=30
- data: 134584 one-minute bars, 126 gaps > 2 min, range 2026-02-15 14:26 → 2026-06-10 11:39 UTC
- grid: 2592 configs, eligible after train filters: 0

Selection uses folds 1..N-1 (train) only. The final fold is holdout —
reported for the top configs but never used for ranking.

| rank | ema | buf | sep | hold | delta | cool | exit | train N | train bps | train WR | hold N | hold bps | hold WR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

**No config passed the train filters** (min trades + fold consistency).
