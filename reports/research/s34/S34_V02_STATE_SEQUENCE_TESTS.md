# S34 V02 State Sequence Tests

Generated: `2026-06-29T18:11:14.375445+00:00`
Scope: ETHUSDT last `30` days, BUY spike events `114`, fee `6.1` bps.
BUY spike threshold: `302236.6` notional/min.

## Candidate Groups

| group | N | 15m fee-net sum | median | WR | T3R | hold sum | hold T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL_BUY_SPIKE | 114 | 1727.7 | 3.79 | 0.535 | 1148.6 | 637.9 | 162.7 |
| NAV_HIGH_BUY_SPIKE | 72 | 1035.7 | 1.34 | 0.514 | 468.5 | 395.1 | -80.1 |
| NAV_HIGH_PERSIST_3OF5 | 50 | 354.7 | -5.32 | 0.48 | -136.1 | 187.5 | -154.3 |
| NAV_RISING_BUY_SPIKE | 46 | 722.8 | -0.88 | 0.5 | 214.6 | 147.2 | -295.2 |
| NAV_HIGH_RISING | 40 | 605.6 | -2.16 | 0.5 | 97.5 | 97.4 | -320.8 |
| PRICE_IMPULSE_POSITIVE | 114 | 1727.7 | 3.79 | 0.535 | 1148.6 | 637.9 | 162.7 |
| NAV_HIGH_AND_IMPULSE | 72 | 1035.7 | 1.34 | 0.514 | 468.5 | 395.1 | -80.1 |
| EXHAUSTION_RISK | 80 | 1514.8 | 8.86 | 0.562 | 935.7 | 790.0 | 314.7 |
| CLEAN_NOT_EXHAUSTION | 34 | 212.9 | -5.32 | 0.471 | -97.5 | -118.0 | -276.3 |

## Scalp vs Swing

- {'n': 110, 'corr_5m_120m': 0.392, 'corr_15m_120m': 0.603, 'sign_matrix_15m_vs_120m': {'both_pos': 43, 'scalp_pos_swing_neg': 19, 'scalp_neg_swing_pos': 13, 'both_neg': 35}}

## Entry Delay, Exits, Promotion

### ALL_BUY_SPIKE
- delays 15m fee-net: `{'0m': {'n': 114, 'sum': 1727.7, 'mean': 15.16, 'median': 3.79, 'win_rate': 0.535, 't3r': 1148.6, 'min': -138.78, 'max': 213.07}, '1m': {'n': 114, 'sum': 339.0, 'mean': 2.97, 'median': -9.02, 'win_rate': 0.412, 't3r': -152.1, 'min': -142.33, 'max': 175.02}, '2m': {'n': 114, 'sum': 330.4, 'mean': 2.9, 'median': -6.39, 'win_rate': 0.439, 't3r': -161.4, 'min': -137.37, 'max': 176.34}, '5m': {'n': 114, 'sum': 206.2, 'mean': 1.81, 'median': -3.93, 'win_rate': 0.482, 't3r': -334.7, 'min': -122.37, 'max': 188.68}}`
- invalidation 15m fee-net: `{'n': 114, 'sum': 1349.8, 'mean': 11.84, 'median': 5.77, 'win_rate': 0.614, 't3r': 876.4, 'min': -40.56, 'max': 191.55}`, exits `{'TIME': 1, 'NAV_LOW': 110, 'BID_THIN': 2, 'BTC_DUMPING': 1}`

### NAV_HIGH_BUY_SPIKE
- delays 15m fee-net: `{'0m': {'n': 72, 'sum': 1035.7, 'mean': 14.39, 'median': 1.34, 'win_rate': 0.514, 't3r': 468.5, 'min': -138.78, 'max': 213.07}, '1m': {'n': 72, 'sum': 115.8, 'mean': 1.61, 'median': -9.02, 'win_rate': 0.417, 't3r': -358.4, 'min': -142.33, 'max': 175.02}, '2m': {'n': 72, 'sum': 106.9, 'mean': 1.49, 'median': -8.51, 'win_rate': 0.403, 't3r': -384.9, 'min': -137.37, 'max': 176.34}, '5m': {'n': 72, 'sum': 166.5, 'mean': 2.31, 'median': -4.52, 'win_rate': 0.472, 't3r': -374.3, 'min': -122.37, 'max': 188.68}}`
- invalidation 15m fee-net: `{'n': 72, 'sum': 826.1, 'mean': 11.47, 'median': 5.52, 'win_rate': 0.611, 't3r': 403.4, 'min': -36.51, 'max': 191.55}`, exits `{'TIME': 1, 'NAV_LOW': 69, 'BID_THIN': 1, 'BTC_DUMPING': 1}`

### NAV_HIGH_AND_IMPULSE
- delays 15m fee-net: `{'0m': {'n': 72, 'sum': 1035.7, 'mean': 14.39, 'median': 1.34, 'win_rate': 0.514, 't3r': 468.5, 'min': -138.78, 'max': 213.07}, '1m': {'n': 72, 'sum': 115.8, 'mean': 1.61, 'median': -9.02, 'win_rate': 0.417, 't3r': -358.4, 'min': -142.33, 'max': 175.02}, '2m': {'n': 72, 'sum': 106.9, 'mean': 1.49, 'median': -8.51, 'win_rate': 0.403, 't3r': -384.9, 'min': -137.37, 'max': 176.34}, '5m': {'n': 72, 'sum': 166.5, 'mean': 2.31, 'median': -4.52, 'win_rate': 0.472, 't3r': -374.3, 'min': -122.37, 'max': 188.68}}`
- invalidation 15m fee-net: `{'n': 72, 'sum': 826.1, 'mean': 11.47, 'median': 5.52, 'win_rate': 0.611, 't3r': 403.4, 'min': -36.51, 'max': 191.55}`, exits `{'TIME': 1, 'NAV_LOW': 69, 'BID_THIN': 1, 'BTC_DUMPING': 1}`

### CLEAN_NOT_EXHAUSTION
- delays 15m fee-net: `{'0m': {'n': 34, 'sum': 212.9, 'mean': 6.26, 'median': -5.32, 'win_rate': 0.471, 't3r': -97.5, 'min': -98.31, 'max': 131.2}, '1m': {'n': 34, 'sum': -79.3, 'mean': -2.33, 'median': -4.74, 'win_rate': 0.441, 't3r': -380.4, 'min': -125.02, 'max': 125.08}, '2m': {'n': 34, 'sum': 18.9, 'mean': 0.56, 'median': -3.92, 'win_rate': 0.471, 't3r': -293.2, 'min': -101.51, 'max': 157.71}, '5m': {'n': 34, 'sum': 150.3, 'mean': 4.42, 'median': 1.02, 'win_rate': 0.529, 't3r': -208.5, 'min': -85.1, 'max': 188.68}}`
- invalidation 15m fee-net: `{'n': 34, 'sum': -40.1, 'mean': -1.18, 'median': 0.31, 'win_rate': 0.5, 't3r': -161.2, 'min': -36.51, 'max': 52.4}`, exits `{'TIME': 1, 'NAV_LOW': 32, 'BID_THIN': 1, 'BTC_DUMPING': 0}`

## Promotion Screen

- candidates passing N>=40, full sum/T3R>0, hold sum/T3R>0: `[{'name': 'ALL_BUY_SPIKE', 'n': 114, 'full_15m_fee_net': {'n': 114, 'sum': 1727.7, 'mean': 15.16, 'median': 3.79, 'win_rate': 0.535, 't3r': 1148.6, 'min': -138.78, 'max': 213.07}, 'hold_15m_fee_net': {'n': 57, 'sum': 637.9, 'mean': 11.19, 'median': 1.65, 'win_rate': 0.526, 't3r': 162.7, 'min': -116.92, 'max': 213.07}}, {'name': 'PRICE_IMPULSE_POSITIVE', 'n': 114, 'full_15m_fee_net': {'n': 114, 'sum': 1727.7, 'mean': 15.16, 'median': 3.79, 'win_rate': 0.535, 't3r': 1148.6, 'min': -138.78, 'max': 213.07}, 'hold_15m_fee_net': {'n': 57, 'sum': 637.9, 'mean': 11.19, 'median': 1.65, 'win_rate': 0.526, 't3r': 162.7, 'min': -116.92, 'max': 213.07}}, {'name': 'EXHAUSTION_RISK', 'n': 80, 'full_15m_fee_net': {'n': 80, 'sum': 1514.8, 'mean': 18.94, 'median': 8.86, 'win_rate': 0.562, 't3r': 935.7, 'min': -138.78, 'max': 213.07}, 'hold_15m_fee_net': {'n': 40, 'sum': 790.0, 'mean': 19.75, 'median': 8.86, 'win_rate': 0.6, 't3r': 314.7, 'min': -116.92, 'max': 213.07}}]`

## Notes

- Research-only. Live executor/config/order logic untouched.
- This is still a one-minute proxy. A passing candidate would need tick-level execution and forward paper before live.