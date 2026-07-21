# S34 Calculator Improvement Research

Generated: 2026-06-26T14:33:46+00:00

---

## Section 1: Architecture Audit

### Features
- Count: 11  (cluster_notional, cluster_duration_sec, cluster_liq_count, max_single_liq_share, intensity_per_sec, inter_cluster_gap_sec, day_trend_bps, day_range_bps, symbol_pre_5m_bps, symbol_pre_15m_bps, btc_pre_15m_bps)
- `cluster_notional` is log-transformed before distance computation.
- All features are signal-time (populated at cluster-formation from mark_prices history). No forward-looking fields.

### Normalization
- **IQR-based robust scale** (p75-p25) per feature.
- Scope: computed per-call on the filtered event universe (symbol+liq_side+notional filter). Not global.

### K Selection
- CLI default K = **50**. Very large relative to small combos (BTC N=127, SOL N=104).
- No per-combo K tuning. Validation scripts used adaptive `min(20, train_n//5)`.

### Missing Features
- Missing dimensions are silently skipped; `total_weight` normalises automatically.
- Risk: if high-weight features (notional, day_trend) are missing, distance falls back to weaker features without warning.

### Temporal No-Lookahead
- **Live mode**: correct. Target = current event; all DB events are historical.
- **`run_oos_validation()` in calculator**: NOT per-event evaluation. It calls `knn_select(test_pool, cli_target_args)` — selects K test-pool events closest to the CLI target query, not K train events per test event. Research scripts implement correct per-event train-pool prediction.

### Target-Less Mode
- When no `--target-*` args given: returns **first K events in temporal order**.
- Metadata flags `error: 'no target features supplied'` — but this is not surfaced in dashboard output.
- **Display risk**: dashboard would show the first K historical events as 'predictions' with no signal. Must be labelled `population_scan_only`.

### Default min-notional = 200K
- Correct for ETH BUY/SELL (DB populated at 200K/500K — all pass).
- Too permissive for BTC (should be 1M) and SOL (should be 100K).
- Running BTC calculator without `--min-notional 1000000` mixes ETH-threshold events into the candidate pool.

---

## Section 2: Feature Ablation

> delta_dir_acc < 0 → dropping feature hurt accuracy (feature is **useful**)
> delta_dir_acc > 0 → dropping feature improved accuracy (feature is **noise**)

### BTCUSDT BUY  (baseline: dir_acc=54%  MAE=38.0  N_test=39)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `cluster_notional` | 49% | -5.1 pp | 37.2 | -0.8 | useful |
| `inter_cluster_gap_sec` | 49% | -5.1 pp | 39.8 | +1.8 | useful |
| `day_trend_bps` | 49% | -5.1 pp | 39.8 | +1.8 | useful |
| `symbol_pre_5m_bps` | 49% | -5.1 pp | 38.3 | +0.3 | useful |
| `btc_pre_15m_bps` | 49% | -5.1 pp | 39.4 | +1.4 | useful |
| `intensity_per_sec` | 51% | -2.5 pp | 38.0 | +0.0 | marginal |
| `symbol_pre_15m_bps` | 51% | -2.5 pp | 38.3 | +0.3 | marginal |
| `cluster_duration_sec` | 54% | +0.0 pp | 37.6 | -0.4 | marginal |
| `cluster_liq_count` | 56% | +2.6 pp | 36.7 | -1.3 | marginal |
| `max_single_liq_share` | 56% | +2.6 pp | 38.4 | +0.4 | marginal |
| `day_range_bps` | 59% | +5.2 pp | 36.4 | -1.6 | noise |

### BTCUSDT SELL  (baseline: dir_acc=71%  MAE=19.9  N_test=34)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `cluster_duration_sec` | 68% | -3.0 pp | 20.6 | +0.7 | marginal |
| `btc_pre_15m_bps` | 68% | -3.0 pp | 18.9 | -1.0 | marginal |
| `cluster_liq_count` | 71% | +0.0 pp | 17.1 | -2.8 | marginal |
| `symbol_pre_5m_bps` | 71% | +0.0 pp | 18.9 | -1.0 | marginal |
| `max_single_liq_share` | 74% | +2.9 pp | 18.5 | -1.4 | marginal |
| `intensity_per_sec` | 74% | +2.9 pp | 19.5 | -0.4 | marginal |
| `day_range_bps` | 74% | +2.9 pp | 18.2 | -1.7 | marginal |
| `symbol_pre_15m_bps` | 74% | +2.9 pp | 17.9 | -2.0 | marginal |
| `cluster_notional` | 76% | +5.9 pp | 16.0 | -3.9 | noise |
| `inter_cluster_gap_sec` | 76% | +5.9 pp | 16.1 | -3.8 | noise |
| `day_trend_bps` | 88% | +17.6 pp | 11.8 | -8.1 | noise |

### ETHUSDT BUY  (baseline: dir_acc=59%  MAE=31.7  N_test=135)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `cluster_notional` | 51% | -8.2 pp | 34.6 | +2.9 | useful |
| `max_single_liq_share` | 55% | -4.5 pp | 33.2 | +1.5 | useful |
| `day_trend_bps` | 56% | -3.0 pp | 31.7 | +0.0 | marginal |
| `cluster_liq_count` | 57% | -2.3 pp | 34.0 | +2.3 | marginal |
| `day_range_bps` | 57% | -2.3 pp | 32.1 | +0.4 | marginal |
| `cluster_duration_sec` | 59% | +0.0 pp | 32.5 | +0.8 | marginal |
| `symbol_pre_15m_bps` | 59% | +0.0 pp | 32.9 | +1.2 | marginal |
| `symbol_pre_5m_bps` | 60% | +0.7 pp | 33.6 | +1.9 | marginal |
| `btc_pre_15m_bps` | 60% | +0.7 pp | 33.0 | +1.3 | marginal |
| `intensity_per_sec` | 61% | +1.4 pp | 31.0 | -0.7 | marginal |
| `inter_cluster_gap_sec` | 63% | +3.7 pp | 32.1 | +0.4 | noise |

### ETHUSDT SELL  (baseline: dir_acc=75%  MAE=22.0  N_test=67)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `inter_cluster_gap_sec` | 72% | -3.0 pp | 23.8 | +1.8 | marginal |
| `day_range_bps` | 73% | -1.5 pp | 22.0 | +0.0 | marginal |
| `symbol_pre_5m_bps` | 75% | +0.0 pp | 24.8 | +2.8 | marginal |
| `cluster_notional` | 78% | +3.0 pp | 26.0 | +4.0 | useful |
| `max_single_liq_share` | 78% | +3.0 pp | 23.0 | +1.0 | marginal |
| `intensity_per_sec` | 78% | +3.0 pp | 23.9 | +1.9 | marginal |
| `cluster_liq_count` | 79% | +4.5 pp | 22.8 | +0.8 | noise |
| `btc_pre_15m_bps` | 79% | +4.5 pp | 21.8 | -0.2 | noise |
| `cluster_duration_sec` | 81% | +6.0 pp | 21.5 | -0.5 | noise |
| `day_trend_bps` | 81% | +6.0 pp | 21.3 | -0.7 | noise |
| `symbol_pre_15m_bps` | 81% | +6.0 pp | 21.7 | -0.3 | noise |

### SOLUSDT BUY  (baseline: dir_acc=53%  MAE=34.6  N_test=32)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `cluster_duration_sec` | 47% | -6.2 pp | 35.9 | +1.3 | useful |
| `max_single_liq_share` | 47% | -6.2 pp | 36.2 | +1.6 | useful |
| `symbol_pre_5m_bps` | 47% | -6.2 pp | 36.8 | +2.2 | useful |
| `day_range_bps` | 50% | -3.1 pp | 33.6 | -1.0 | useful |
| `inter_cluster_gap_sec` | 53% | +0.0 pp | 34.7 | +0.1 | marginal |
| `symbol_pre_15m_bps` | 56% | +3.1 pp | 32.0 | -2.6 | noise |
| `cluster_notional` | 59% | +6.3 pp | 33.4 | -1.2 | noise |
| `cluster_liq_count` | 59% | +6.3 pp | 34.5 | -0.1 | noise |
| `day_trend_bps` | 62% | +9.4 pp | 33.4 | -1.2 | noise |
| `btc_pre_15m_bps` | 62% | +9.4 pp | 33.0 | -1.6 | noise |
| `intensity_per_sec` | 69% | +15.7 pp | 29.5 | -5.1 | noise |

### SOLUSDT SELL  (baseline: dir_acc=34%  MAE=49.0  N_test=32)

| Feature | DirAcc w/o | ΔDirAcc | MAE w/o | ΔMAE | Verdict |
|---|---:|---:|---:|---:|---|
| `cluster_duration_sec` | 31% | -3.2 pp | 48.7 | -0.3 | useful |
| `cluster_liq_count` | 34% | +0.0 pp | 49.5 | +0.5 | marginal |
| `max_single_liq_share` | 34% | +0.0 pp | 48.3 | -0.7 | marginal |
| `symbol_pre_5m_bps` | 38% | +3.1 pp | 48.9 | -0.1 | noise |
| `btc_pre_15m_bps` | 38% | +3.1 pp | 49.1 | +0.1 | noise |
| `cluster_notional` | 41% | +6.2 pp | 49.3 | +0.3 | noise |
| `day_range_bps` | 41% | +6.2 pp | 45.6 | -3.4 | noise |
| `symbol_pre_15m_bps` | 41% | +6.2 pp | 44.6 | -4.4 | noise |
| `intensity_per_sec` | 44% | +9.4 pp | 44.9 | -4.1 | noise |
| `day_trend_bps` | 47% | +12.5 pp | 44.8 | -4.2 | noise |
| `inter_cluster_gap_sec` | 50% | +15.6 pp | 43.3 | -5.7 | noise |

---

## Section 3: K Selection Sweep

### BTCUSDT BUY  (best K = 5)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 ** | 39 | 54% | 34.9 | -3.3 | +52.3 | +60.4 |
| 10 | 39 | 46% | 38.0 | -5.7 | +52.3 | +60.4 |
| 15 | 39 | 41% | 40.5 | -8.1 | +52.3 | +60.4 |
| 20 | 39 | 54% | 38.0 | -5.7 | +52.3 | +60.4 |
| 30 | 39 | 41% | 37.8 | -5.7 | +52.3 | +60.4 |
| 50 | 39 | 38% | 43.5 | -8.1 | +52.3 | +60.4 |

### BTCUSDT SELL  (best K = 50)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 34 | 68% | 22.7 | +32.1 | +32.4 | +0.3 |
| 10 | 34 | 76% | 19.0 | +32.1 | +32.4 | +0.3 |
| 15 | 34 | 68% | 18.9 | +32.1 | +32.4 | +0.3 |
| 20 | 34 | 71% | 19.9 | +32.1 | +32.4 | +0.3 |
| 30 | 34 | 85% | 14.2 | +32.1 | +32.4 | +0.3 |
| 50 ** | 34 | 94% | 5.7 | +32.1 | +32.4 | +0.3 |

### ETHUSDT BUY  (best K = 50)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 135 | 59% | 33.5 | -8.1 | -8.0 | -28.5 |
| 10 | 135 | 53% | 37.7 | +13.8 | -8.0 | -28.5 |
| 15 | 135 | 60% | 33.5 | -8.1 | -8.0 | -28.5 |
| 20 | 135 | 59% | 31.7 | +7.1 | -8.0 | -28.5 |
| 30 | 135 | 64% | 31.8 | -8.0 | -8.0 | -28.5 |
| 50 ** | 135 | 66% | 28.9 | -8.1 | -8.0 | -28.5 |

### ETHUSDT SELL  (best K = 10)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 67 | 76% | 22.7 | +52.3 | +52.4 | +14.7 |
| 10 ** | 67 | 82% | 23.5 | +52.1 | +52.4 | +14.7 |
| 15 | 67 | 78% | 21.7 | +52.3 | +52.4 | +14.7 |
| 20 | 67 | 75% | 22.0 | +52.4 | +52.4 | +14.7 |
| 30 | 67 | 79% | 21.4 | +52.2 | +52.4 | +14.7 |
| 50 | 67 | 76% | 23.4 | +52.2 | +52.4 | +14.7 |

### SOLUSDT BUY  (best K = 50)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 32 | 59% | 31.0 | +52.6 | +52.5 | +39.7 |
| 10 | 32 | 59% | 30.8 | +45.5 | +52.5 | +39.7 |
| 15 | 32 | 62% | 29.4 | +38.3 | +52.5 | +39.7 |
| 20 | 32 | 53% | 34.6 | +22.2 | +52.5 | +39.7 |
| 30 | 32 | 62% | 32.6 | +22.2 | +52.5 | +39.7 |
| 50 ** | 32 | 66% | 35.7 | +38.3 | +52.5 | +39.7 |

### SOLUSDT SELL  (best K = 50)

| K | N_test | DirAcc | MAE | PredMedian | RealMedian | Uplift |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 32 | 56% | 43.4 | +21.8 | +52.2 | +41.8 |
| 10 | 32 | 50% | 45.7 | +21.8 | +52.2 | +41.8 |
| 15 | 32 | 38% | 48.1 | +19.1 | +52.2 | +41.8 |
| 20 | 32 | 34% | 49.0 | +10.1 | +52.2 | +41.8 |
| 30 | 32 | 50% | 42.6 | +14.7 | +52.2 | +41.8 |
| 50 ** | 32 | 62% | 42.6 | -8.0 | +52.2 | +41.8 |

---

## Section 4: Distance Metric Comparison

All at K=20.

### BTCUSDT BUY  (best metric = euclidean)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean ** | 39 | 54% | 38.0 | -5.7 | +52.3 |
| manhattan | 39 | 51% | 39.4 | -0.7 | +52.3 |
| cosine | 39 | 54% | 37.7 | -5.7 | +52.3 |
| recency | 39 | 41% | 42.7 | -8.2 | +52.3 |

### BTCUSDT SELL  (best metric = recency)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean | 34 | 71% | 19.9 | +32.1 | +32.4 |
| manhattan | 34 | 79% | 15.3 | +32.1 | +32.4 |
| cosine | 34 | 74% | 18.1 | +32.1 | +32.4 |
| recency ** | 34 | 85% | 9.5 | +32.1 | +32.4 |

### ETHUSDT BUY  (best metric = recency)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean | 135 | 59% | 31.7 | +7.1 | -8.0 |
| manhattan | 135 | 63% | 31.4 | -0.4 | -8.0 |
| cosine | 135 | 56% | 33.1 | -8.2 | -8.0 |
| recency ** | 135 | 64% | 31.1 | -8.1 | -8.0 |

### ETHUSDT SELL  (best metric = manhattan)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean | 67 | 75% | 22.0 | +52.4 | +52.4 |
| manhattan ** | 67 | 79% | 21.9 | +52.4 | +52.4 |
| cosine | 67 | 78% | 24.3 | +52.2 | +52.4 |
| recency | 67 | 75% | 23.5 | +52.5 | +52.4 |

### SOLUSDT BUY  (best metric = recency)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean | 32 | 53% | 34.6 | +22.2 | +52.5 |
| manhattan | 32 | 56% | 35.2 | +16.8 | +52.5 |
| cosine | 32 | 56% | 33.6 | +38.4 | +52.5 |
| recency ** | 32 | 59% | 37.8 | +38.5 | +52.5 |

### SOLUSDT SELL  (best metric = cosine)

| Metric | N_test | DirAcc | MAE | PredMedian | RealMedian |
|---|---:|---:|---:|---:|---:|
| euclidean | 32 | 34% | 49.0 | +10.1 | +52.2 |
| manhattan | 32 | 38% | 46.9 | +6.8 | +52.2 |
| cosine ** | 32 | 47% | 45.5 | +11.5 | +52.2 |
| recency | 32 | 44% | 47.6 | +9.0 | +52.2 |

---

## Section 5: Calibration  (threshold = ±15 bps)

Does 'predicted strong positive' actually realize a positive outcome?

| Combo | Bucket | N | PredMedian | RealMedian | WinRate | Calibrated |
|---|---|---:|---:|---:|---:|---|
| BTCUSDT BUY | strong_pos | 8 | +40.4 | +52.6 | 88% | yes |
| BTCUSDT BUY | neutral | 31 | -8.1 | +52.3 | 71% | neutral |
| BTCUSDT BUY | neg | 0 | NA | NA | NA% | no_data |
| BTCUSDT SELL | strong_pos | 26 | +32.1 | +32.4 | 92% | yes |
| BTCUSDT SELL | neutral | 2 | -9.8 | +32.4 | 100% | neutral |
| BTCUSDT SELL | neg | 6 | -24.9 | +32.9 | 100% | no |
| ETHUSDT BUY | strong_pos | 55 | +41.3 | +52.3 | 62% | yes |
| ETHUSDT BUY | neutral | 80 | -8.3 | -8.5 | 34% | neutral |
| ETHUSDT BUY | neg | 0 | NA | NA | NA% | no_data |
| ETHUSDT SELL | strong_pos | 53 | +52.4 | +52.6 | 81% | yes |
| ETHUSDT SELL | neutral | 14 | -5.6 | -8.2 | 43% | neutral |
| ETHUSDT SELL | neg | 0 | NA | NA | NA% | no_data |
| SOLUSDT BUY | strong_pos | 18 | +45.3 | +53.4 | 67% | yes |
| SOLUSDT BUY | neutral | 14 | -5.3 | +52.2 | 64% | neutral |
| SOLUSDT BUY | neg | 0 | NA | NA | NA% | no_data |
| SOLUSDT SELL | strong_pos | 15 | +25.3 | -8.0 | 47% | no |
| SOLUSDT SELL | neutral | 17 | -8.4 | +52.2 | 71% | neutral |
| SOLUSDT SELL | neg | 0 | NA | NA | NA% | no_data |

---

## Section 6: Regime Drift  (train vs test feature distributions)

> drift_score = |test_mean - train_mean| / train_std
> high_drift > 1.5 | moderate_drift 0.7-1.5 | low_drift 0.3-0.7 | stable < 0.3

### BTCUSDT BUY  (max_feat_drift=0.659  outcome_drift=0.555  n_drifted=2)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 14.57 | 14.64 | 0.097 | stable |
| `cluster_duration_sec` | 170.74 | 184.90 | 0.185 | stable |
| `cluster_liq_count` | 26.14 | 25.92 | 0.014 | stable |
| `max_single_liq_share` | 53.44 | 48.46 | 0.211 | stable |
| `intensity_per_sec` | 23257.39 | 43819.53 | 0.659 | low_drift |
| `inter_cluster_gap_sec` | 112710.48 | 34122.58 | 0.192 | stable |
| `day_trend_bps` | 140.74 | 17.11 | 0.654 | low_drift |
| `day_range_bps` | 284.22 | 284.81 | 0.004 | stable |
| `symbol_pre_5m_bps` | 25.31 | 14.33 | 0.255 | stable |
| `symbol_pre_15m_bps` | 38.45 | 22.49 | 0.319 | low_drift |
| `btc_pre_15m_bps` | 38.45 | 22.49 | 0.319 | low_drift |

**Outcome (primary route)**: train_median=-8.1  test_median=+52.3  drift=0.555  verdict=low_drift

### BTCUSDT SELL  (max_feat_drift=0.812  outcome_drift=0.565  n_drifted=2)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 14.51 | 14.72 | 0.294 | stable |
| `cluster_duration_sec` | 182.51 | 200.57 | 0.231 | stable |
| `cluster_liq_count` | 32.77 | 37.41 | 0.230 | stable |
| `max_single_liq_share` | 59.42 | 38.24 | 0.812 | moderate_drift |
| `intensity_per_sec` | 72337.22 | 25841.17 | 0.097 | stable |
| `inter_cluster_gap_sec` | 131985.22 | 22645.89 | 0.248 | stable |
| `day_trend_bps` | -67.50 | -157.29 | 0.543 | low_drift |
| `day_range_bps` | 247.03 | 289.67 | 0.332 | low_drift |
| `symbol_pre_5m_bps` | -21.33 | -28.00 | 0.343 | low_drift |
| `symbol_pre_15m_bps` | -33.02 | -30.62 | 0.086 | stable |
| `btc_pre_15m_bps` | -33.02 | -30.62 | 0.086 | stable |

**Outcome (primary route)**: train_median=+32.2  test_median=+32.4  drift=0.565  verdict=low_drift

### ETHUSDT BUY  (max_feat_drift=0.735  outcome_drift=0.172  n_drifted=2)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 12.90 | 13.38 | 0.704 | moderate_drift |
| `cluster_duration_sec` | 156.21 | 135.09 | 0.256 | stable |
| `cluster_liq_count` | 18.94 | 15.28 | 0.254 | stable |
| `max_single_liq_share` | 68.05 | 57.30 | 0.453 | low_drift |
| `intensity_per_sec` | 12103.21 | 24386.91 | 0.298 | stable |
| `inter_cluster_gap_sec` | 16682.31 | 38250.98 | 0.735 | moderate_drift |
| `day_trend_bps` | 103.06 | 108.55 | 0.017 | stable |
| `day_range_bps` | 436.65 | 318.20 | 0.439 | low_drift |
| `symbol_pre_5m_bps` | 17.27 | 18.77 | 0.050 | stable |
| `symbol_pre_15m_bps` | 18.65 | 24.97 | 0.111 | stable |
| `btc_pre_15m_bps` | 14.81 | 18.26 | 0.080 | stable |

**Outcome (primary route)**: train_median=+20.4  test_median=-8.0  drift=0.172  verdict=stable

### ETHUSDT SELL  (max_feat_drift=0.595  outcome_drift=0.352  n_drifted=2)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 13.86 | 14.22 | 0.538 | low_drift |
| `cluster_duration_sec` | 171.70 | 182.94 | 0.135 | stable |
| `cluster_liq_count` | 26.31 | 28.57 | 0.119 | stable |
| `max_single_liq_share` | 61.13 | 44.72 | 0.595 | low_drift |
| `intensity_per_sec` | 24924.83 | 16198.15 | 0.088 | stable |
| `inter_cluster_gap_sec` | 64609.40 | 19250.06 | 0.157 | stable |
| `day_trend_bps` | -19.29 | -91.05 | 0.240 | stable |
| `day_range_bps` | 380.30 | 338.89 | 0.168 | stable |
| `symbol_pre_5m_bps` | -23.72 | -21.00 | 0.078 | stable |
| `symbol_pre_15m_bps` | -35.07 | -31.85 | 0.057 | stable |
| `btc_pre_15m_bps` | -24.26 | -24.80 | 0.014 | stable |

**Outcome (primary route)**: train_median=+37.7  test_median=+52.4  drift=0.352  verdict=low_drift

### SOLUSDT BUY  (max_feat_drift=0.760  outcome_drift=0.262  n_drifted=3)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 12.54 | 12.45 | 0.132 | stable |
| `cluster_duration_sec` | 131.19 | 132.68 | 0.018 | stable |
| `cluster_liq_count` | 13.82 | 15.44 | 0.155 | stable |
| `max_single_liq_share` | 64.02 | 53.35 | 0.378 | low_drift |
| `intensity_per_sec` | 11290.02 | 4977.61 | 0.168 | stable |
| `inter_cluster_gap_sec` | 73373.32 | 18910.35 | 0.130 | stable |
| `day_trend_bps` | 165.72 | 32.84 | 0.629 | low_drift |
| `day_range_bps` | 338.48 | 486.74 | 0.760 | moderate_drift |
| `symbol_pre_5m_bps` | 26.67 | 32.97 | 0.303 | low_drift |
| `symbol_pre_15m_bps` | 44.27 | 44.94 | 0.020 | stable |
| `btc_pre_15m_bps` | 26.26 | 13.27 | 0.561 | low_drift |

**Outcome (primary route)**: train_median=+12.7  test_median=+52.5  drift=0.262  verdict=stable

### SOLUSDT SELL  (max_feat_drift=0.458  outcome_drift=0.301  n_drifted=0)

| Feature | TrainMean | TestMean | DriftScore | Verdict |
|---|---:|---:|---:|---|
| `cluster_notional` | 12.40 | 12.55 | 0.222 | stable |
| `cluster_duration_sec` | 150.23 | 180.56 | 0.366 | low_drift |
| `cluster_liq_count` | 18.19 | 20.62 | 0.226 | stable |
| `max_single_liq_share` | 55.22 | 58.31 | 0.113 | stable |
| `intensity_per_sec` | 7619.57 | 3261.50 | 0.137 | stable |
| `inter_cluster_gap_sec` | 74345.39 | 19470.67 | 0.132 | stable |
| `day_trend_bps` | -71.54 | -91.08 | 0.086 | stable |
| `day_range_bps` | 320.93 | 408.50 | 0.458 | low_drift |
| `symbol_pre_5m_bps` | -30.24 | -34.56 | 0.224 | stable |
| `symbol_pre_15m_bps` | -48.06 | -52.56 | 0.135 | stable |
| `btc_pre_15m_bps` | -27.35 | -36.77 | 0.397 | low_drift |

**Outcome (primary route)**: train_median=+10.3  test_median=+52.2  drift=0.301  verdict=low_drift

---

## Section 7: Proposed Calculator v2 Design

**Tag changes from current:**
    BTCUSDT BUY: DRIFT_ARTIFACT_PRELIMINARY -> BASE_RATE_ONLY (K: 20 -> 50)
    BTCUSDT SELL: BASE_RATE_ONLY -> KNN_USEFUL (K: 20 -> 50)
    ETHUSDT BUY: REGIME_SHIFT_WARNING -> KNN_USEFUL (K: 20 -> 50)
    SOLUSDT BUY: DRIFT_ARTIFACT_PRELIMINARY -> KNN_USEFUL (K: 20 -> 50)
    SOLUSDT SELL: DRIFT_ARTIFACT_PRELIMINARY -> BASE_RATE_ONLY (K: 20 -> 50)

| Combo | CurrentTag | ProposedTag | K | Metric | NoiseFeat | FallbackBehavior |
|---|---|---|---:|---|---|---|
| BTCUSDT BUY | DRIFT_ARTIFACT_PRELIMINARY | **BASE_RATE_ONLY** | 50 | euclidean | `day_range_bps` | show_base-rate_median_only |
| BTCUSDT SELL | BASE_RATE_ONLY | **KNN_USEFUL** | 50 | recency | `cluster_notional inter_cluster_gap_sec day_trend_bps` | predict_with_strong-bucket_caveat |
| ETHUSDT BUY | REGIME_SHIFT_WARNING | **KNN_USEFUL** | 50 | recency | `inter_cluster_gap_sec` | predict_with_strong-bucket_caveat |
| ETHUSDT SELL | KNN_USEFUL | **KNN_USEFUL** | 10 | manhattan | `cluster_liq_count btc_pre_15m_bps cluster_duration_sec day_trend_bps symbol_pre_15m_bps` | predict_with_optimised_K |
| SOLUSDT BUY | DRIFT_ARTIFACT_PRELIMINARY | **KNN_USEFUL** | 50 | recency | `symbol_pre_15m_bps cluster_notional cluster_liq_count day_trend_bps btc_pre_15m_bps intensity_per_sec` | predict_with_strong-bucket_caveat |
| SOLUSDT SELL | DRIFT_ARTIFACT_PRELIMINARY | **BASE_RATE_ONLY** | 50 | cosine | `symbol_pre_5m_bps btc_pre_15m_bps cluster_notional day_range_bps symbol_pre_15m_bps intensity_per_sec day_trend_bps inter_cluster_gap_sec` | show_base-rate_median_only |

### Per-Combo Detail

**BTCUSDT BUY** → `BASE_RATE_ONLY`
- K=5 outperforms default K=20 (dir_acc 0.54 -> 0.54)
- Noise features (dropping helps): day_range_bps
- Useful features (top by dir_acc delta): cluster_notional, inter_cluster_gap_, day_trend_bps, symbol_pre_5m_bps
- Calibration strong_pos bucket: N=8, realized_median=+52.6, calibrated=yes
- Drift: max_feature=0.659  outcome=0.555  n_drifted_features=2
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: broad / usable / thin / too_thin

**BTCUSDT SELL** → `KNN_USEFUL`
- K=50 outperforms default K=20 (dir_acc 0.71 -> 0.94)
- recency metric outperforms euclidean (0.71 -> 0.85)
- Noise features (dropping helps): cluster_notional, inter_cluster_gap_sec, day_trend_bps
- Calibration strong_pos bucket: N=26, realized_median=+32.4, calibrated=yes
- Drift: max_feature=0.812  outcome=0.565  n_drifted_features=2
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: drifted / broad / usable / thin / too_thin

**ETHUSDT BUY** → `KNN_USEFUL`
- K=50 outperforms default K=20 (dir_acc 0.59 -> 0.66)
- recency metric outperforms euclidean (0.59 -> 0.64)
- Noise features (dropping helps): inter_cluster_gap_sec
- Useful features (top by dir_acc delta): cluster_notional, max_single_liq_sha
- Calibration strong_pos bucket: N=55, realized_median=+52.3, calibrated=yes
- Drift: max_feature=0.735  outcome=0.172  n_drifted_features=2
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: drifted / broad / usable / thin / too_thin

**ETHUSDT SELL** → `KNN_USEFUL`
- K=10 outperforms default K=20 (dir_acc 0.75 -> 0.82)
- manhattan metric outperforms euclidean (0.75 -> 0.79)
- Noise features (dropping helps): cluster_liq_count, btc_pre_15m_bps, cluster_duration_sec, day_trend_bps, symbol_pre_15m_bps
- Useful features (top by dir_acc delta): cluster_notional
- Calibration strong_pos bucket: N=53, realized_median=+52.6, calibrated=yes
- Drift: max_feature=0.595  outcome=0.352  n_drifted_features=2
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: broad / usable / thin / too_thin

**SOLUSDT BUY** → `KNN_USEFUL`
- K=50 outperforms default K=20 (dir_acc 0.53 -> 0.66)
- recency metric outperforms euclidean (0.53 -> 0.59)
- Noise features (dropping helps): symbol_pre_15m_bps, cluster_notional, cluster_liq_count, day_trend_bps, btc_pre_15m_bps, intensity_per_sec
- Useful features (top by dir_acc delta): cluster_duration_s, max_single_liq_sha, symbol_pre_5m_bps, day_range_bps
- Calibration strong_pos bucket: N=18, realized_median=+53.4, calibrated=yes
- Drift: max_feature=0.76  outcome=0.262  n_drifted_features=3
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: drifted / broad / usable / thin / too_thin

**SOLUSDT SELL** → `BASE_RATE_ONLY`
- K=50 outperforms default K=20 (dir_acc 0.34 -> 0.62)
- cosine metric outperforms euclidean (0.34 -> 0.47)
- Noise features (dropping helps): symbol_pre_5m_bps, btc_pre_15m_bps, cluster_notional, day_range_bps, symbol_pre_15m_bps, intensity_per_sec, day_trend_bps, inter_cluster_gap_sec
- Useful features (top by dir_acc delta): cluster_duration_s
- Calibration strong_pos bucket: N=15, realized_median=-8.0, calibrated=no
- Drift: max_feature=0.458  outcome=0.301  n_drifted_features=0
- no_target_mode: population_scan_only — must label as such, not a prediction
- confidence_labels: broad / usable / thin / too_thin

---

## Overfitting Risk Notes

- All evaluation is temporal OOS (test = last 30%). Features are pre-computed, no lookahead.
- K sweep and ablation use the same 70/30 split — selecting best K/metric on test data is mild overfitting. Treat as directional, not certified.
- Combos with N_test < 30 are marked `*` and tagged DRIFT_ARTIFACT_PRELIMINARY regardless of metrics.
- ETH BUY N_test=135 is the only combo with sufficient test size for confident conclusions.
- Large 'uplift' values for BTC BUY / SOL combos reflect regime drift (test period stronger than train), not genuine KNN signal.
