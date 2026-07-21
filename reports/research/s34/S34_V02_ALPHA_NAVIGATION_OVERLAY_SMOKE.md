# S34 V02 Alpha Navigation Overlay

Generated: `2026-06-29T18:31:56.373475+00:00`
Scope: `{'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'days': 7, 'start_utc': '2026-06-22T18:31:06+00:00', 'end_utc': '2026-06-29T18:31:06+00:00', 'maker_fee_bps': -0.5, 'taker_fee_bps': 3.05, 'cross_margin_bps': 2.0, 'buy_spike_threshold': 320411.5, 'buy_extreme_threshold': 1728163.2, 'note': 'Research-only. No live executor/config/order logic touched.'}`

## 1. Live-Like V02 Fill Set

- all anchors: `3`
- filled: `3`
- no maker fill: `0`
- baseline 2h: `{'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}`

## 2. BUY Spike Overlay

- buy_spike_pre_5m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- buy_spike_pre_15m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- buy_spike_pre_30m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_1m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_5m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- buy_spike_post_15m: `{'False': {'n': 2, 'sum': 321.9, 'mean': 160.9, 'median': 160.9, 'win_rate': 1.0, 't3r': 321.9, 'min': 19.7, 'max': 302.2}, 'True': {'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}}`

## 3. Scalp Horizon Decomposition

- 15s: `{'n': 3, 'sum': -28.5, 'mean': -9.5, 'median': -15.2, 'win_rate': 0.333, 't3r': -28.5, 'min': -15.2, 'max': 2.0}`
- 30s: `{'n': 3, 'sum': -9.0, 'mean': -3.0, 'median': -13.1, 'win_rate': 0.333, 't3r': -9.0, 'min': -19.5, 'max': 23.6}`
- 60s: `{'n': 3, 'sum': 5.8, 'mean': 1.9, 'median': -13.3, 'win_rate': 0.333, 't3r': 5.8, 'min': -19.3, 'max': 38.4}`
- 2m: `{'n': 3, 'sum': 4.9, 'mean': 1.6, 'median': -12.9, 'win_rate': 0.333, 't3r': 4.9, 'min': -17.7, 'max': 35.6}`
- 5m: `{'n': 3, 'sum': 26.9, 'mean': 9.0, 'median': -9.1, 'win_rate': 0.333, 't3r': 26.9, 'min': -20.3, 'max': 56.3}`
- 15m: `{'n': 3, 'sum': 68.9, 'mean': 23.0, 'median': 21.9, 'win_rate': 0.667, 't3r': 68.9, 'min': -12.4, 'max': 59.3}`
- 60m: `{'n': 3, 'sum': 382.7, 'mean': 127.6, 'median': 173.2, 'win_rate': 0.667, 't3r': 382.7, 'min': -47.5, 'max': 257.0}`
- 2h: `{'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}`

## 4. MFE/MAE Path

`{'mfe': {'n': 3, 'sum': 664.0, 'mean': 221.3, 'median': 237.9, 'win_rate': 1.0, 't3r': 664.0, 'min': 91.7, 'max': 334.4}, 'mae': {'n': 3, 'sum': -261.6, 'mean': -87.2, 'median': -100.7, 'win_rate': 0.0, 't3r': -261.6, 'min': -147.2, 'max': -13.7}, 'mfe_time_sec_median': 3789.0, 'mae_time_sec_median': 575.0}`

## 5. Danger / Navigation Tags

- nav_high_fill: `{'False': {'n': 2, 'sum': 439.5, 'mean': 219.8, 'median': 219.8, 'win_rate': 1.0, 't3r': 439.5, 'min': 137.4, 'max': 302.2}, 'True': {'n': 1, 'sum': 19.7, 'mean': 19.7, 'median': 19.7, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 19.7}}`
- nav_high_holds_5m: `{'False': {'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}, 'True': {'n': 2, 'sum': 321.9, 'mean': 160.9, 'median': 160.9, 'win_rate': 1.0, 't3r': 321.9, 'min': 19.7, 'max': 302.2}}`
- liquidity_thin: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- book_support: `{'True': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- exhaustion_risk: `{'False': {'n': 2, 'sum': 157.1, 'mean': 78.5, 'median': 78.5, 'win_rate': 1.0, 't3r': 157.1, 'min': 19.7, 'max': 137.4}, 'True': {'n': 1, 'sum': 302.2, 'mean': 302.2, 'median': 302.2, 'win_rate': 1.0, 't3r': 302.2, 'min': 302.2, 'max': 302.2}}`
- squeeze_active: `{'False': {'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}, 'True': {'n': 2, 'sum': 321.9, 'mean': 160.9, 'median': 160.9, 'win_rate': 1.0, 't3r': 321.9, 'min': 19.7, 'max': 302.2}}`
- rebound_confirmed_5m: `{'False': {'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}}`
- nav_recommendation: `{'BASELINE': {'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}, 'SCALP_ONLY': {'n': 1, 'sum': 19.7, 'mean': 19.7, 'median': 19.7, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 19.7}, 'SCALP_OR_REDUCE': {'n': 1, 'sum': 302.2, 'mean': 302.2, 'median': 302.2, 'win_rate': 1.0, 't3r': 302.2, 'min': 302.2, 'max': 302.2}}`

## 6. State Sequence Anatomy

- MMHMHL: `{'n': 1, 'sum': 137.4, 'mean': 137.4, 'median': 137.4, 'win_rate': 1.0, 't3r': 137.4, 'min': 137.4, 'max': 137.4}`
- MLLHHM: `{'n': 1, 'sum': 19.7, 'mean': 19.7, 'median': 19.7, 'win_rate': 1.0, 't3r': 19.7, 'min': 19.7, 'max': 19.7}`
- LHMHHH: `{'n': 1, 'sum': 302.2, 'mean': 302.2, 'median': 302.2, 'win_rate': 1.0, 't3r': 302.2, 'min': 302.2, 'max': 302.2}`

## 7. Shadow Management Policies

- baseline_2h: `{'n': 3, 'sum': 459.2, 'mean': 153.1, 'median': 137.4, 'win_rate': 1.0, 't3r': 459.2, 'min': 19.7, 'max': 302.2}`
- scalp_or_reduce_5m: `{'n': 3, 'sum': 184.6, 'mean': 61.5, 'median': 56.3, 'win_rate': 0.667, 't3r': 184.6, 'min': -9.1, 'max': 137.4}`
- confirmed_hold_else_15m: `{'n': 3, 'sum': 68.9, 'mean': 23.0, 'median': 21.9, 'win_rate': 0.667, 't3r': 68.9, 'min': -12.4, 'max': 59.3}`
- danger_exit_1m_else_2h: `{'n': 3, 'sum': 195.5, 'mean': 65.2, 'median': 38.4, 'win_rate': 1.0, 't3r': 195.5, 'min': 19.7, 'max': 137.4}`

## 8. Interpretation

Scalp horizons do not improve robust T3R versus baseline 2h. Navigation management policies are not yet better than baseline.