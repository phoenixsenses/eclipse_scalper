# S34 Phase 2 Calculator Validation

Generated: 2026-06-26T14:11:42.154667+00:00

Train/test split: 70% / 30% by event timestamp (strict temporal, no leakage).
KNN: K=20 (adaptive: min(20, train_n//5)), default weights from research_s34_cluster_geometry_features.py.
Auto-weights: correlation-based reweighting on train set. Preliminary: test N < 30.

## Summary — Primary Route KNN Uplift Ranking

| Combo | Route | N_test | Base Median | Realized Median | Pred Median | Uplift | Dir Acc | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTCUSDT BUY | LONG_DELAY0_TP60 | 39 | -8.1 | +52.3 | -8.1 | +60.4 | 44% | 40.4 |
| SOLUSDT SELL | SHORT_DELAY0_TP60 | 32 | +10.3 | +52.1 | +19.6 | +41.9 | 41% | 47.7 |
| SOLUSDT BUY | LONG_DELAY0_TP60 | 32 | +12.7 | +52.5 | +27.6 | +39.7 | 66% | 33.0 |
| ETHUSDT SELL | SHORT_DELAY0_TP60 | 67 | +37.7 | +52.4 | +52.4 | +14.7 | 75% | 22.0 |
| BTCUSDT SELL | SHORT_DELAY0_TP40 | 34 | +32.1 | +32.4 | +32.1 | +0.3 | 68% | 18.9 |
| ETHUSDT BUY | LONG_DELAY0_TP60 | 135 | +20.4 | -8.1 | +7.1 | -28.5 | 59% | 31.7 |

\* = preliminary (test N < 30)

> **Uplift** = realized_median(test) - base_rate_median(train). Positive = KNN selected a better-than-average subset.
> **Dir Acc** = fraction of test events where sign(KNN prediction) == sign(realized outcome).
> **MAE** = mean absolute error between per-event KNN prediction and realized outcome.

---

## BTCUSDT BUY

- Total events: 127  Train: 88  Test: 39
- Train ends: 2026-06-11  Test starts: 2026-06-11
- Primary route: `LONG_DELAY0_TP60`

### LONG_DELAY0_TP60 **(primary)**

Train N=88  Test N=39  K=17

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -8.1 | -8.1 | -8.1 |
| Realized median | — | +52.3 | +52.3 |
| MAE | — | 40.4 | 38.0 |
| Direction accuracy | — | 44% | 44% |
| Uplift vs base-rate | — | +60.4 | — |
| Base-rate WR | 46% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 8 | +28.6 | +52.5 | 88% |
| neutral | 31 | -8.1 | +52.3 | 71% |
| negative | 0 | NA | NA | NA |

### SHORT_DELAY0_TP40_CONTROL

Train N=88  Test N=39  K=17

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -48.3 | -48.5 | -48.5 |
| Realized median | — | -48.5 | -48.5 |
| MAE | — | 9.6 | 10.4 |
| Direction accuracy | — | 90% | 90% |
| Uplift vs base-rate | — | -0.2 | — |
| Base-rate WR | 22% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 0 | NA | NA | NA |
| negative | 39 | -48.4 | -48.5 | 10% |

## BTCUSDT SELL

- Total events: 113  Train: 79  Test: 34
- Train ends: 2026-06-17  Test starts: 2026-06-17
- Primary route: `SHORT_DELAY0_TP40`

### LONG_DELAY0_TP40_CONTROL

Train N=79  Test N=34  K=15

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -48.1 | -48.1 | -48.1 |
| Realized median | — | -48.4 | -48.4 |
| MAE | — | 15.8 | 15.2 |
| Direction accuracy | — | 94% | 91% |
| Uplift vs base-rate | — | -0.2 | — |
| Base-rate WR | 25% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 7 | -0.4 | -48.1 | NA |
| negative | 27 | -48.1 | -48.5 | 4% |

### SHORT_DELAY0_TP40 **(primary)**

Train N=79  Test N=34  K=15

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +32.1 | +32.1 | +32.1 |
| Realized median | — | +32.4 | +32.4 |
| MAE | — | 18.9 | 16.7 |
| Direction accuracy | — | 68% | 71% |
| Uplift vs base-rate | — | +0.3 | — |
| Base-rate WR | 66% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 24 | +32.1 | +32.4 | 92% |
| neutral | 4 | -0.3 | +32.7 | 100% |
| negative | 6 | -20.0 | +32.5 | 100% |

### SHORT_DELAY0_TP60

Train N=79  Test N=34  K=15

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -8.1 | -8.2 | -8.2 |
| Realized median | — | +52.2 | +52.2 |
| MAE | — | 45.0 | 39.9 |
| Direction accuracy | — | 38% | 50% |
| Uplift vs base-rate | — | +60.3 | — |
| Base-rate WR | 46% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 6 | +25.5 | +39.6 | 67% |
| neutral | 22 | -8.2 | +52.3 | 82% |
| negative | 6 | -20.0 | +36.3 | 67% |

## ETHUSDT BUY

- Total events: 450  Train: 315  Test: 135
- Train ends: 2026-04-17  Test starts: 2026-04-17
- Primary route: `LONG_DELAY0_TP60`

### LONG_DELAY0_TP60 **(primary)**

Train N=315  Test N=135  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +20.4 | +7.1 | -8.0 |
| Realized median | — | -8.1 | -8.1 |
| MAE | — | 31.7 | 30.9 |
| Direction accuracy | — | 59% | 62% |
| Uplift vs base-rate | — | -28.5 | — |
| Base-rate WR | 51% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 55 | +41.3 | +52.3 | 62% |
| neutral | 80 | -8.3 | -8.5 | 34% |
| negative | 0 | NA | NA | NA |

### LONG_DELAY60_TP120

Train N=315  Test N=135  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -8.8 | -9.2 | -9.2 |
| Realized median | — | -8.6 | -8.6 |
| MAE | — | 37.5 | 37.2 |
| Direction accuracy | — | 70% | 70% |
| Uplift vs base-rate | — | +0.2 | — |
| Base-rate WR | 28% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 121 | -9.0 | -8.6 | 31% |
| negative | 14 | -20.5 | -11.5 | 14% |

### SHORT_DELAY0_TP40_CONTROL

Train N=315  Test N=135  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -48.5 | -48.4 | -48.3 |
| Realized median | — | -48.3 | -48.3 |
| MAE | — | 26.4 | 25.8 |
| Direction accuracy | — | 74% | 71% |
| Uplift vs base-rate | — | +0.2 | — |
| Base-rate WR | 22% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 7 | -8.6 | -11.4 | 14% |
| negative | 128 | -48.4 | -48.4 | 27% |

## ETHUSDT SELL

- Total events: 222  Train: 155  Test: 67
- Train ends: 2026-06-11  Test starts: 2026-06-11
- Primary route: `SHORT_DELAY0_TP60`

### LONG_DELAY0_TP40_CONTROL

Train N=155  Test N=67  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -48.7 | -48.8 | -48.9 |
| Realized median | — | -48.9 | -48.9 |
| MAE | — | 15.8 | 15.4 |
| Direction accuracy | — | 87% | 87% |
| Uplift vs base-rate | — | -0.2 | — |
| Base-rate WR | 21% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 0 | NA | NA | NA |
| negative | 67 | -48.8 | -48.9 | 13% |

### SHORT_DELAY0_TP60 **(primary)**

Train N=155  Test N=67  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +37.7 | +52.4 | +52.4 |
| Realized median | — | +52.4 | +52.4 |
| MAE | — | 22.0 | 21.8 |
| Direction accuracy | — | 75% | 75% |
| Uplift vs base-rate | — | +14.7 | — |
| Base-rate WR | 54% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 53 | +52.4 | +52.6 | 81% |
| neutral | 14 | -5.6 | -8.2 | 43% |
| negative | 0 | NA | NA | NA |

### SHORT_DELAY0_TP80

Train N=155  Test N=67  K=20

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -8.3 | -2.4 | +10.4 |
| Realized median | — | +51.1 | +51.1 |
| MAE | — | 39.1 | 38.1 |
| Direction accuracy | — | 64% | 70% |
| Uplift vs base-rate | — | +59.4 | — |
| Base-rate WR | 43% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 26 | +45.9 | +72.3 | 81% |
| neutral | 41 | -8.2 | -8.2 | 44% |
| negative | 0 | NA | NA | NA |

## SOLUSDT BUY

- Total events: 104  Train: 72  Test: 32
- Train ends: 2026-06-19  Test starts: 2026-06-20
- Primary route: `LONG_DELAY0_TP60`

### LONG_DELAY0_TP60 **(primary)**

Train N=72  Test N=32  K=14

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +12.7 | +27.6 | +52.4 |
| Realized median | — | +52.5 | +52.5 |
| MAE | — | 33.0 | 30.6 |
| Direction accuracy | — | 66% | 66% |
| Uplift vs base-rate | — | +39.7 | — |
| Base-rate WR | 53% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 25 | +38.3 | +52.4 | 68% |
| neutral | 7 | -2.6 | +53.6 | 57% |
| negative | 0 | NA | NA | NA |

### SHORT_DELAY0_TP40_CONTROL

Train N=72  Test N=32  K=14

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -49.0 | -49.1 | -49.1 |
| Realized median | — | -49.2 | -49.2 |
| MAE | — | 22.5 | 22.5 |
| Direction accuracy | — | 75% | 75% |
| Uplift vs base-rate | — | -0.1 | — |
| Base-rate WR | 17% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 0 | NA | NA | NA |
| negative | 32 | -49.1 | -49.2 | 25% |

## SOLUSDT SELL

- Total events: 105  Train: 73  Test: 32
- Train ends: 2026-06-19  Test starts: 2026-06-20
- Primary route: `SHORT_DELAY0_TP60`

### LONG_DELAY0_TP40_CONTROL

Train N=73  Test N=32  K=14

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | -48.3 | -48.3 | -48.3 |
| Realized median | — | -48.7 | -48.7 |
| MAE | — | 18.1 | 19.5 |
| Direction accuracy | — | 78% | 78% |
| Uplift vs base-rate | — | -0.4 | — |
| Base-rate WR | 25% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 0 | NA | NA | NA |
| neutral | 0 | NA | NA | NA |
| negative | 32 | -48.3 | -48.7 | 22% |

### SHORT_DELAY0_TP40

Train N=73  Test N=32  K=14

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +32.3 | +32.4 | +32.4 |
| Realized median | — | +32.7 | +32.7 |
| MAE | — | 18.7 | 18.7 |
| Direction accuracy | — | 78% | 78% |
| Uplift vs base-rate | — | +0.4 | — |
| Base-rate WR | 68% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 32 | +32.4 | +32.7 | 78% |
| neutral | 0 | NA | NA | NA |
| negative | 0 | NA | NA | NA |

### SHORT_DELAY0_TP60 **(primary)**

Train N=73  Test N=32  K=14

| | Base-rate | KNN (default) | KNN (auto-w) |
|---|---:|---:|---:|
| Predicted median | +10.3 | +19.6 | +6.9 |
| Realized median | — | +52.1 | +52.1 |
| MAE | — | 47.7 | 43.4 |
| Direction accuracy | — | 41% | 53% |
| Uplift vs base-rate | — | +41.9 | — |
| Base-rate WR | 51% | — | — |

Calibration (KNN default, threshold 15.0 bps):

| Predicted bucket | N | Pred Median | Realized Median | WR |
|---|---:|---:|---:|---:|
| positive | 18 | +21.9 | +22.0 | 50% |
| neutral | 14 | -8.8 | +52.4 | 71% |
| negative | 0 | NA | NA | NA |

---

## Verdict

**Calculator adds value (uplift > +5 bps, confirmed):**
- BTCUSDT BUY / LONG_DELAY0_TP60: uplift=+60.4 bps, dir_acc=44%
- SOLUSDT SELL / SHORT_DELAY0_TP60: uplift=+41.9 bps, dir_acc=41%
- SOLUSDT BUY / LONG_DELAY0_TP60: uplift=+39.7 bps, dir_acc=66%
- ETHUSDT SELL / SHORT_DELAY0_TP60: uplift=+14.7 bps, dir_acc=75%

**Treat as base-rate only (uplift <= 0, confirmed):**
- ETHUSDT BUY / LONG_DELAY0_TP60: uplift=-28.5 bps

_Read-only validation. No runner, config, or pre-reg changes made._
