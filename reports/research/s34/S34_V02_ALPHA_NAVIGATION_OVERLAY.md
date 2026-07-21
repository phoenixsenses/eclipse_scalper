# S34 V02 Alpha Navigation Overlay

Generated: `2026-06-29T18:34:38.142169+00:00`
Scope: `{'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'days': 30, 'start_utc': '2026-05-30T18:31:57.001000+00:00', 'end_utc': '2026-06-29T18:31:57.001000+00:00', 'maker_fee_bps': -0.5, 'taker_fee_bps': 3.05, 'cross_margin_bps': 2.0, 'buy_spike_threshold': 302236.6, 'buy_extreme_threshold': 1847540.9, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## 1. Live-Like V02 Fill Set

- all anchors: `7`
- filled: `7`
- no maker fill: `0`
- baseline 2h: `{'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}`

## 2. BUY Spike Overlay

- buy_spike_pre_5m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- buy_spike_pre_15m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- buy_spike_pre_30m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_1m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_5m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_15m: `{'False': {'n': 6, 'sum': 796.5, 'mean': 132.8, 'median': 100.5, 'win_rate': 1.0, 't3r': 110.1, 'min': 19.7, 'max': 302.2}, 'True': {'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}}`

## 3. Scalp Horizon Decomposition

- 15s: `{'n': 7, 'sum': -28.5, 'mean': -4.1, 'median': 0.5, 'win_rate': 0.571, 't3r': -36.1, 'min': -15.2, 'max': 3.8}`
- 30s: `{'n': 7, 'sum': 6.4, 'mean': 0.9, 'median': 5.0, 'win_rate': 0.571, 't3r': -31.0, 'min': -19.5, 'max': 23.6}`
- 60s: `{'n': 7, 'sum': 8.3, 'mean': 1.2, 'median': -3.7, 'win_rate': 0.429, 't3r': -46.4, 'min': -19.3, 'max': 38.4}`
- 2m: `{'n': 7, 'sum': 18.3, 'mean': 2.6, 'median': -0.3, 'win_rate': 0.429, 't3r': -33.7, 'min': -17.7, 'max': 35.6}`
- 5m: `{'n': 7, 'sum': 65.9, 'mean': 9.4, 'median': 13.0, 'win_rate': 0.571, 't3r': -24.0, 'min': -20.3, 'max': 56.3}`
- 15m: `{'n': 7, 'sum': 96.9, 'mean': 13.8, 'median': 9.6, 'win_rate': 0.714, 't3r': -7.1, 'min': -12.4, 'max': 59.3}`
- 60m: `{'n': 7, 'sum': 753.3, 'mean': 107.6, 'median': 123.2, 'win_rate': 0.857, 't3r': 166.6, 'min': -47.5, 'max': 257.0}`
- 2h: `{'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}`

## 4. MFE/MAE Path

`{'mfe': {'n': 7, 'sum': 1436.1, 'mean': 205.2, 'median': 237.9, 'win_rate': 1.0, 't3r': 545.1, 'min': 62.5, 'max': 334.4}, 'mae': {'n': 7, 'sum': -304.9, 'mean': -43.6, 'median': -18.4, 'win_rate': 0.0, 't3r': -285.3, 'min': -147.2, 'max': -2.2}, 'mfe_time_sec_median': 3959.0, 'mae_time_sec_median': 75.0}`

## 5. Danger / Navigation Tags

- nav_high_fill: `{'False': {'n': 5, 'sum': 684.7, 'mean': 136.9, 'median': 137.4, 'win_rate': 1.0, 't3r': 90.4, 'min': 44.2, 'max': 302.2}, 'True': {'n': 2, 'sum': 249.2, 'mean': 124.6, 'median': 124.6, 'win_rate': 1.0, 't3r': 249.2, 'min': 19.7, 'max': 229.5}}`
- nav_high_holds_5m: `{'False': {'n': 3, 'sum': 521.6, 'mean': 173.9, 'median': 154.8, 'win_rate': 1.0, 't3r': 521.6, 'min': 137.4, 'max': 229.5}, 'True': {'n': 4, 'sum': 412.3, 'mean': 103.1, 'median': 45.2, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 302.2}}`
- liquidity_thin: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- book_support: `{'False': {'n': 2, 'sum': 273.7, 'mean': 136.8, 'median': 136.8, 'win_rate': 1.0, 't3r': 273.7, 'min': 44.2, 'max': 229.5}, 'True': {'n': 5, 'sum': 660.2, 'mean': 132.0, 'median': 137.4, 'win_rate': 1.0, 't3r': 65.9, 'min': 19.7, 'max': 302.2}}`
- exhaustion_risk: `{'False': {'n': 4, 'sum': 430.8, 'mean': 107.7, 'median': 90.8, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 229.5}, 'True': {'n': 3, 'sum': 503.2, 'mean': 167.7, 'median': 154.8, 'win_rate': 1.0, 't3r': 503.2, 'min': 46.2, 'max': 302.2}}`
- squeeze_active: `{'False': {'n': 3, 'sum': 521.6, 'mean': 173.9, 'median': 154.8, 'win_rate': 1.0, 't3r': 521.6, 'min': 137.4, 'max': 229.5}, 'True': {'n': 4, 'sum': 412.3, 'mean': 103.1, 'median': 45.2, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 302.2}}`
- rebound_confirmed_5m: `{'False': {'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}}`
- nav_recommendation: `{'BASELINE': {'n': 2, 'sum': 366.9, 'mean': 183.4, 'median': 183.4, 'win_rate': 1.0, 't3r': 366.9, 'min': 137.4, 'max': 229.5}, 'SCALP_ONLY': {'n': 2, 'sum': 63.9, 'mean': 31.9, 'median': 31.9, 'win_rate': 1.0, 't3r': 63.9, 'min': 19.7, 'max': 44.2}, 'SCALP_OR_REDUCE': {'n': 3, 'sum': 503.2, 'mean': 167.7, 'median': 154.8, 'win_rate': 1.0, 't3r': 503.2, 'min': 46.2, 'max': 302.2}}`

## 6. State Sequence Anatomy

- LLMHMH: `{'n': 1, 'sum': 46.2, 'mean': 46.2, 'median': 46.2, 'win_rate': 1.0, 't3r': 46.2, 'min': 46.2, 'max': 46.2}`
- MMHHHH: `{'n': 1, 'sum': 44.2, 'mean': 44.2, 'median': 44.2, 'win_rate': 1.0, 't3r': 44.2, 'min': 44.2, 'max': 44.2}`
- LMLLHM: `{'n': 1, 'sum': 154.8, 'mean': 154.8, 'median': 154.8, 'win_rate': 1.0, 't3r': 154.8, 'min': 154.8, 'max': 154.8}`
- HMMMMM: `{'n': 1, 'sum': 229.5, 'mean': 229.5, 'median': 229.5, 'win_rate': 1.0, 't3r': 229.5, 'min': 229.5, 'max': 229.5}`
- MMHMHL: `{'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}`
- MLLHHM: `{'n': 1, 'sum': 19.7, 'mean': 19.7, 'median': 19.7, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 19.7}`
- LHMHHH: `{'n': 1, 'sum': 302.2, 'mean': 302.2, 'median': 302.2, 'win_rate': 1.0, 't3r': 302.2, 'min': 302.2, 'max': 302.2}`

## 7. Shadow Management Policies

- baseline_2h: `{'n': 7, 'sum': 933.9, 'mean': 133.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 247.5, 'min': 19.7, 'max': 302.2}`
- scalp_or_reduce_5m: `{'n': 7, 'sum': 435.2, 'mean': 62.2, 'median': 15.7, 'win_rate': 0.714, 't3r': 12.0, 'min': -9.1, 'max': 229.5}`
- confirmed_hold_else_15m: `{'n': 7, 'sum': 96.9, 'mean': 13.8, 'median': 9.6, 'win_rate': 0.714, 't3r': -7.1, 'min': -12.4, 'max': 59.3}`
- danger_exit_1m_else_2h: `{'n': 7, 'sum': 471.5, 'mean': 67.4, 'median': 38.4, 'win_rate': 0.857, 't3r': 60.5, 'min': -3.7, 'max': 229.5}`

## 8. Interpretation

Scalp horizons do not improve robust T3R versus baseline 2h. Navigation management policies are not yet better than baseline.