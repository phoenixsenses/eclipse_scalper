# BTC SELL KNN Variance Check

Generated: 2026-06-26T15:22:21+00:00
Combo: BTCUSDT SELL  |  Route: SHORT_DELAY0_TP40
N events: 113  |  Train: 79 (70%)  |  Test: 34 (30%)

## Question

Does K=50 recency KNN add real discriminative power, or is the 0.94 dir_acc
just majority-class alignment (predicting always-positive and test is 94% positive)?

## Results

| Config | DirAcc KNN | DirAcc Constant | KNN Gain | MAE | PredStd | PredRange | UniquePreds |
|---|---:|---:|---:|---:|---:|---:|---:|
| K=20 euclidean | 71% | 94% | -23.5 pp | 19.9 | 22.5 | 57.8 | 11/34 |
| K=50 recency | 94% | 94% | +0.0 pp | 5.7 | 0.0 | 0.1 | 1/34 |

**Constant baseline**: always predict sign(train_median). If train_median > 0, predict positive.
**KNN Gain**: dir_acc(KNN) - dir_acc(constant) — how much the model adds over naive prediction.

## Verdict: `NOT_USEFUL_BASE_RATE_ALIGNMENT`

Prediction std=0.0 bps (< 5 bps) and KNN gain=+0.0 pp over constant baseline. K=50 predictions converge to train median — majority-class alignment confirmed. Keep BTC SELL as BASE_RATE_ONLY.

## Recommendation: keep BASE_RATE_ONLY

- BTC SELL tag remains `BASE_RATE_ONLY_PENDING_VARIANCE_CHECK` until clearly supported.
- Review when test N grows (currently N_test=34).
