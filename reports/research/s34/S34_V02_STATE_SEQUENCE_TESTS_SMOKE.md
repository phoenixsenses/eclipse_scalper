# S34 V02 State Sequence Tests

Generated: `2026-06-29T18:08:28.065090+00:00`
Scope: ETHUSDT last `7` days, BUY spike events `52`, fee `6.1` bps.
BUY spike threshold: `320411.5` notional/min.

## Candidate Groups

| group | N | 15m fee-net sum | median | WR | T3R | hold sum | hold T3R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL_BUY_SPIKE | 52 | 767.4 | 6.58 | 0.577 | 292.2 | 491.9 | 16.7 |
| NAV_HIGH_BUY_SPIKE | 35 | 459.1 | 1.65 | 0.543 | -16.1 | 514.4 | 39.2 |
| NAV_HIGH_PERSIST_3OF5 | 25 | 233.7 | 1.65 | 0.56 | -108.1 | 236.9 | -47.7 |
| NAV_RISING_BUY_SPIKE | 20 | 324.9 | 4.1 | 0.55 | -117.5 | 157.6 | -136.4 |
| NAV_HIGH_RISING | 17 | 202.3 | 3.28 | 0.529 | -215.9 | 204.1 | -89.9 |
| PRICE_IMPULSE_POSITIVE | 52 | 767.4 | 6.58 | 0.577 | 292.2 | 491.9 | 16.7 |
| NAV_HIGH_AND_IMPULSE | 35 | 459.1 | 1.65 | 0.543 | -16.1 | 514.4 | 39.2 |
| EXHAUSTION_RISK | 35 | 847.6 | 27.28 | 0.657 | 372.4 | 618.6 | 143.4 |
| CLEAN_NOT_EXHAUSTION | 17 | -80.2 | -5.44 | 0.412 | -248.1 | -41.3 | -136.4 |

## Scalp vs Swing

- {'n': 48, 'corr_5m_120m': 0.424, 'corr_15m_120m': 0.58, 'sign_matrix_15m_vs_120m': {'both_pos': 20, 'scalp_pos_swing_neg': 9, 'scalp_neg_swing_pos': 5, 'both_neg': 14}}

## Entry Delay, Exits, Promotion

### ALL_BUY_SPIKE
- delays 15m fee-net: `{'0m': {'n': 52, 'sum': 767.4, 'mean': 14.76, 'median': 6.58, 'win_rate': 0.577, 't3r': 292.2, 'min': -116.92, 'max': 213.07}, '1m': {'n': 52, 'sum': 287.3, 'mean': 5.53, 'median': -4.28, 'win_rate': 0.481, 't3r': -131.2, 'min': -125.02, 'max': 175.02}, '2m': {'n': 52, 'sum': 337.9, 'mean': 6.5, 'median': -4.93, 'win_rate': 0.481, 't3r': -137.7, 'min': -101.51, 'max': 176.34}, '5m': {'n': 52, 'sum': 171.1, 'mean': 3.29, 'median': 2.54, 'win_rate': 0.519, 't3r': -198.5, 'min': -116.25, 'max': 176.39}}`
- invalidation 15m fee-net: `{'n': 52, 'sum': 537.7, 'mean': 10.34, 'median': 0.06, 'win_rate': 0.5, 't3r': 115.0, 'min': -37.83, 'max': 191.55}`, exits `{'TIME': 0, 'NAV_LOW': 49, 'BID_THIN': 2, 'BTC_DUMPING': 1}`

### NAV_HIGH_BUY_SPIKE
- delays 15m fee-net: `{'0m': {'n': 35, 'sum': 459.1, 'mean': 13.12, 'median': 1.65, 'win_rate': 0.543, 't3r': -16.1, 'min': -116.92, 'max': 213.07}, '1m': {'n': 35, 'sum': 133.6, 'mean': 3.82, 'median': -6.86, 'win_rate': 0.486, 't3r': -285.0, 'min': -125.02, 'max': 175.02}, '2m': {'n': 35, 'sum': 132.8, 'mean': 3.79, 'median': -8.45, 'win_rate': 0.429, 't3r': -307.5, 'min': -101.51, 'max': 176.34}, '5m': {'n': 35, 'sum': -18.7, 'mean': -0.53, 'median': -4.03, 'win_rate': 0.486, 't3r': -355.2, 'min': -116.25, 'max': 176.39}}`
- invalidation 15m fee-net: `{'n': 35, 'sum': 459.6, 'mean': 13.13, 'median': 1.05, 'win_rate': 0.514, 't3r': 37.0, 'min': -34.39, 'max': 191.55}`, exits `{'TIME': 0, 'NAV_LOW': 33, 'BID_THIN': 1, 'BTC_DUMPING': 1}`

### NAV_HIGH_AND_IMPULSE
- delays 15m fee-net: `{'0m': {'n': 35, 'sum': 459.1, 'mean': 13.12, 'median': 1.65, 'win_rate': 0.543, 't3r': -16.1, 'min': -116.92, 'max': 213.07}, '1m': {'n': 35, 'sum': 133.6, 'mean': 3.82, 'median': -6.86, 'win_rate': 0.486, 't3r': -285.0, 'min': -125.02, 'max': 175.02}, '2m': {'n': 35, 'sum': 132.8, 'mean': 3.79, 'median': -8.45, 'win_rate': 0.429, 't3r': -307.5, 'min': -101.51, 'max': 176.34}, '5m': {'n': 35, 'sum': -18.7, 'mean': -0.53, 'median': -4.03, 'win_rate': 0.486, 't3r': -355.2, 'min': -116.25, 'max': 176.39}}`
- invalidation 15m fee-net: `{'n': 35, 'sum': 459.6, 'mean': 13.13, 'median': 1.05, 'win_rate': 0.514, 't3r': 37.0, 'min': -34.39, 'max': 191.55}`, exits `{'TIME': 0, 'NAV_LOW': 33, 'BID_THIN': 1, 'BTC_DUMPING': 1}`

### CLEAN_NOT_EXHAUSTION
- delays 15m fee-net: `{'0m': {'n': 17, 'sum': -80.2, 'mean': -4.72, 'median': -5.44, 'win_rate': 0.412, 't3r': -248.1, 'min': -98.31, 'max': 82.82}, '1m': {'n': 17, 'sum': -190.4, 'mean': -11.2, 'median': -6.86, 'win_rate': 0.471, 't3r': -321.3, 'min': -125.02, 'max': 56.06}, '2m': {'n': 17, 'sum': -82.9, 'mean': -4.88, 'median': -4.34, 'win_rate': 0.471, 't3r': -261.0, 'min': -101.51, 'max': 91.04}, '5m': {'n': 17, 'sum': -80.2, 'mean': -4.72, 'median': -9.68, 'win_rate': 0.412, 't3r': -224.1, 'min': -61.47, 'max': 80.16}}`
- invalidation 15m fee-net: `{'n': 17, 'sum': -53.3, 'mean': -3.14, 'median': -8.11, 'win_rate': 0.412, 't3r': -109.8, 'min': -34.39, 'max': 38.67}`, exits `{'TIME': 0, 'NAV_LOW': 16, 'BID_THIN': 1, 'BTC_DUMPING': 0}`

## Promotion Screen

- candidates passing N>=40, full sum/T3R>0, hold sum/T3R>0: `[{'name': 'ALL_BUY_SPIKE', 'n': 52, 'full_15m_fee_net': {'n': 52, 'sum': 767.4, 'mean': 14.76, 'median': 6.58, 'win_rate': 0.577, 't3r': 292.2, 'min': -116.92, 'max': 213.07}, 'hold_15m_fee_net': {'n': 26, 'sum': 491.9, 'mean': 18.92, 'median': 6.58, 'win_rate': 0.577, 't3r': 16.7, 'min': -104.46, 'max': 213.07}}, {'name': 'PRICE_IMPULSE_POSITIVE', 'n': 52, 'full_15m_fee_net': {'n': 52, 'sum': 767.4, 'mean': 14.76, 'median': 6.58, 'win_rate': 0.577, 't3r': 292.2, 'min': -116.92, 'max': 213.07}, 'hold_15m_fee_net': {'n': 26, 'sum': 491.9, 'mean': 18.92, 'median': 6.58, 'win_rate': 0.577, 't3r': 16.7, 'min': -104.46, 'max': 213.07}}]`

## Notes

- Research-only. Live executor/config/order logic untouched.
- This is still a one-minute proxy. A passing candidate would need tick-level execution and forward paper before live.