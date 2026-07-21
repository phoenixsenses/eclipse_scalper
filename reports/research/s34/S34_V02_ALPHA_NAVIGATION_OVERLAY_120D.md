# S34 V02 Alpha Navigation Overlay

Generated: `2026-06-29T18:44:25.862937+00:00`
Scope: `{'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'days': 120, 'start_utc': '2026-03-01T18:34:52.001000+00:00', 'end_utc': '2026-06-29T18:34:52.001000+00:00', 'maker_fee_bps': -0.5, 'taker_fee_bps': 3.05, 'cross_margin_bps': 2.0, 'buy_spike_threshold': 200000.0, 'buy_extreme_threshold': 893164.1, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## 1. Live-Like V02 Fill Set

- all anchors: `11`
- filled: `11`
- no maker fill: `0`
- baseline 2h: `{'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}`

## 2. BUY Spike Overlay

- buy_spike_pre_5m: `{'False': {'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}}`
- buy_spike_pre_15m: `{'False': {'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}}`
- buy_spike_pre_30m: `{'False': {'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}}`
- buy_spike_post_1m: `{'False': {'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}}`
- buy_spike_post_5m: `{'False': {'n': 10, 'sum': 1033.5, 'mean': 103.3, 'median': 64.9, 'win_rate': 1.0, 't3r': 347.0, 'min': 15.0, 'max': 302.2}, 'True': {'n': 1, 'sum': 44.2, 'mean': 44.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 44.2, 'min': 44.2, 'max': 44.2}}`
- buy_spike_post_15m: `{'False': {'n': 8, 'sum': 812.4, 'mean': 101.6, 'median': 37.4, 'win_rate': 1.0, 't3r': 126.0, 'min': 15.0, 'max': 302.2}, 'True': {'n': 3, 'sum': 265.2, 'mean': 88.4, 'median': 83.6, 'win_rate': 1.0, 't3r': 265.2, 'min': 44.2, 'max': 137.4}}`

## 3. Scalp Horizon Decomposition

- 15s: `{'n': 11, 'sum': -43.0, 'mean': -3.9, 'median': 0.7, 'win_rate': 0.636, 't3r': -50.9, 'min': -18.7, 'max': 3.8}`
- 30s: `{'n': 11, 'sum': -2.4, 'mean': -0.2, 'median': 2.1, 'win_rate': 0.636, 't3r': -40.3, 'min': -19.5, 'max': 23.6}`
- 60s: `{'n': 11, 'sum': -38.2, 'mean': -3.5, 'median': -3.3, 'win_rate': 0.273, 't3r': -92.9, 'min': -37.7, 'max': 38.4}`
- 2m: `{'n': 11, 'sum': -38.1, 'mean': -3.5, 'median': -0.3, 'win_rate': 0.364, 't3r': -93.2, 'min': -64.8, 'max': 35.6}`
- 5m: `{'n': 11, 'sum': -84.1, 'mean': -7.6, 'median': -7.6, 'win_rate': 0.455, 't3r': -175.8, 'min': -73.8, 'max': 56.3}`
- 15m: `{'n': 11, 'sum': -42.4, 'mean': -3.9, 'median': 3.3, 'win_rate': 0.545, 't3r': -155.2, 'min': -60.6, 'max': 59.3}`
- 60m: `{'n': 11, 'sum': 834.7, 'mean': 75.9, 'median': 50.8, 'win_rate': 0.818, 't3r': 248.0, 'min': -48.8, 'max': 257.0}`
- 2h: `{'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}`

## 4. MFE/MAE Path

`{'mfe': {'n': 11, 'sum': 1806.5, 'mean': 164.2, 'median': 128.6, 'win_rate': 1.0, 't3r': 915.5, 'min': 51.4, 'max': 334.4}, 'mae': {'n': 11, 'sum': -607.5, 'mean': -55.2, 'median': -19.0, 'win_rate': 0.0, 't3r': -597.0, 'min': -147.2, 'max': -2.2}, 'mfe_time_sec_median': 3959.0, 'mae_time_sec_median': 670.0}`

## 5. Danger / Navigation Tags

- nav_high_fill: `{'False': {'n': 8, 'sum': 799.9, 'mean': 100.0, 'median': 64.9, 'win_rate': 1.0, 't3r': 205.6, 'min': 15.0, 'max': 302.2}, 'True': {'n': 3, 'sum': 277.7, 'mean': 92.6, 'median': 28.6, 'win_rate': 1.0, 't3r': 277.7, 'min': 19.7, 'max': 229.5}}`
- nav_high_holds_5m: `{'False': {'n': 6, 'sum': 636.9, 'mean': 106.1, 'median': 110.5, 'win_rate': 1.0, 't3r': 115.2, 'min': 15.0, 'max': 229.5}, 'True': {'n': 5, 'sum': 440.8, 'mean': 88.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 48.2, 'min': 19.7, 'max': 302.2}}`
- liquidity_thin: `{'False': {'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}}`
- book_support: `{'False': {'n': 2, 'sum': 273.7, 'mean': 136.8, 'median': 136.8, 'win_rate': 1.0, 't3r': 273.7, 'min': 44.2, 'max': 229.5}, 'True': {'n': 9, 'sum': 804.0, 'mean': 89.3, 'median': 46.2, 'win_rate': 1.0, 't3r': 209.7, 'min': 15.0, 'max': 302.2}}`
- exhaustion_risk: `{'False': {'n': 5, 'sum': 459.3, 'mean': 91.9, 'median': 44.2, 'win_rate': 1.0, 't3r': 48.2, 'min': 19.7, 'max': 229.5}, 'True': {'n': 6, 'sum': 618.4, 'mean': 103.1, 'median': 64.9, 'win_rate': 1.0, 't3r': 77.8, 'min': 15.0, 'max': 302.2}}`
- squeeze_active: `{'False': {'n': 6, 'sum': 636.9, 'mean': 106.1, 'median': 110.5, 'win_rate': 1.0, 't3r': 115.2, 'min': 15.0, 'max': 229.5}, 'True': {'n': 5, 'sum': 440.8, 'mean': 88.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 48.2, 'min': 19.7, 'max': 302.2}}`
- rebound_confirmed_5m: `{'False': {'n': 10, 'sum': 1033.5, 'mean': 103.3, 'median': 64.9, 'win_rate': 1.0, 't3r': 347.0, 'min': 15.0, 'max': 302.2}, 'True': {'n': 1, 'sum': 44.2, 'mean': 44.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 44.2, 'min': 44.2, 'max': 44.2}}`
- nav_recommendation: `{'BASELINE': {'n': 2, 'sum': 366.9, 'mean': 183.4, 'median': 183.4, 'win_rate': 1.0, 't3r': 366.9, 'min': 137.4, 'max': 229.5}, 'HOLD_ALLOWED': {'n': 1, 'sum': 44.2, 'mean': 44.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 44.2, 'min': 44.2, 'max': 44.2}, 'SCALP_ONLY': {'n': 2, 'sum': 48.2, 'mean': 24.1, 'median': 24.1, 'win_rate': 1.0, 't3r': 48.2, 'min': 19.7, 'max': 28.6}, 'SCALP_OR_REDUCE': {'n': 6, 'sum': 618.4, 'mean': 103.1, 'median': 64.9, 'win_rate': 1.0, 't3r': 77.8, 'min': 15.0, 'max': 302.2}}`

## 6. State Sequence Anatomy

- LMHHHM: `{'n': 2, 'sum': 31.6, 'mean': 15.8, 'median': 15.8, 'win_rate': 1.0, 't3r': 31.6, 'min': 15.0, 'max': 16.6}`
- LLLLHH: `{'n': 1, 'sum': 83.6, 'mean': 83.6, 'median': 83.6, 'win_rate': 1.0, 't3r': 83.6, 'min': 83.6, 'max': 83.6}`
- HHHMHH: `{'n': 1, 'sum': 28.6, 'mean': 28.6, 'median': 28.6, 'win_rate': 1.0, 't3r': 28.6, 'min': 28.6, 'max': 28.6}`
- LLMHMH: `{'n': 1, 'sum': 46.2, 'mean': 46.2, 'median': 46.2, 'win_rate': 1.0, 't3r': 46.2, 'min': 46.2, 'max': 46.2}`
- MMHHHH: `{'n': 1, 'sum': 44.2, 'mean': 44.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 44.2, 'min': 44.2, 'max': 44.2}`
- LMLLHM: `{'n': 1, 'sum': 154.8, 'mean': 154.8, 'median': 154.8, 'win_rate': 1.0, 't3r': 154.8, 'min': 154.8, 'max': 154.8}`
- HMMMMM: `{'n': 1, 'sum': 229.5, 'mean': 229.5, 'median': 229.5, 'win_rate': 1.0, 't3r': 229.5, 'min': 229.5, 'max': 229.5}`
- MMHMHL: `{'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}`
- MLLHHM: `{'n': 1, 'sum': 19.7, 'mean': 19.7, 'median': 19.7, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 19.7}`
- LHMHHH: `{'n': 1, 'sum': 302.2, 'mean': 302.2, 'median': 302.2, 'win_rate': 1.0, 't3r': 302.2, 'min': 302.2, 'max': 302.2}`

## 7. Shadow Management Policies

- baseline_2h: `{'n': 11, 'sum': 1077.7, 'mean': 98.0, 'median': 46.2, 'win_rate': 1.0, 't3r': 391.2, 'min': 15.0, 'max': 302.2}`
- scalp_or_reduce_5m: `{'n': 11, 'sum': 313.7, 'mean': 28.5, 'median': 13.0, 'win_rate': 0.545, 't3r': -109.5, 'min': -73.8, 'max': 229.5}`
- confirmed_hold_else_15m: `{'n': 11, 'sum': -1.5, 'mean': -0.1, 'median': 9.6, 'win_rate': 0.545, 't3r': -135.8, 'min': -60.6, 'max': 59.3}`
- danger_exit_1m_else_2h: `{'n': 11, 'sum': 456.7, 'mean': 41.5, 'median': 19.7, 'win_rate': 0.636, 't3r': 45.6, 'min': -37.7, 'max': 229.5}`

## 8. Interpretation

Scalp horizons do not improve robust T3R versus baseline 2h. Navigation management policies are not yet better than baseline.