# S34 Liquidation Swing Event (beta-controlled, 1h-48h)

Generated: `2026-06-28T17:55:40.386997+00:00`  |  cost `6.1`bps, holdout `0.3`

`signal_diff` = median(raw return | BUY-liq) - median(raw | SELL-liq); beta cancels. >0 CONTINUATION, <0 REVERSAL. `cont_net` = combined continuation-signed net P&L (LONG after BUY-liq, SHORT after SELL-liq), beta-cancelling, after cost. Note: long horizons overlap (not independent); read the BUY-vs-SELL difference and cal/hold stability, not single windows.

## ETHUSDT 200K  (BUY events=547, SELL events=570)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -5.2 | 9.1 | -14.3 | REVERSAL | -13.7 | 44.5 | -15.4 | -9.6 |
| 2h | -6.6 | 11.5 | -18.1 | REVERSAL | -14.4 | 44.9 | -17.5 | -9.1 |
| 4h | -3.4 | 15.6 | -19.0 | REVERSAL | -14.8 | 46.0 | -12.9 | -24.9 |
| 8h | -2.5 | 7.9 | -10.4 | REVERSAL | -11.1 | 48.0 | -3.8 | -27.3 |
| 12h | -12.7 | 8.6 | -21.3 | REVERSAL | -17.1 | 47.4 | -10.2 | -30.6 |
| 24h | -11.1 | -1.8 | -9.3 | REVERSAL | -9.9 | 49.5 | -5.6 | -17.6 |
| 48h | 0.9 | 17.5 | -16.6 | REVERSAL | -17.6 | 49.4 | -23.0 | -10.8 |

## ETHUSDT 500K  (BUY events=215, SELL events=226)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -19.7 | 11.5 | -31.2 | REVERSAL | -20.4 | 41.4 | -24.5 | -8.6 |
| 2h | -31.6 | 12.0 | -43.6 | REVERSAL | -27.6 | 40.2 | -36.0 | -10.3 |
| 4h | -18.1 | 24.1 | -42.2 | REVERSAL | -26.8 | 43.0 | -25.7 | -27.7 |
| 8h | -18.0 | 4.9 | -22.9 | REVERSAL | -18.4 | 45.5 | -19.4 | -13.2 |
| 12h | -59.2 | 10.7 | -69.9 | REVERSAL | -27.8 | 44.0 | -43.0 | 9.4 |
| 24h | -73.8 | -11.9 | -61.9 | REVERSAL | -26.0 | 46.9 | -27.8 | -12.4 |
| 48h | -76.3 | -2.9 | -73.4 | REVERSAL | -39.7 | 46.9 | -46.7 | -1.3 |

## ETHUSDT 1000K  (BUY events=120, SELL events=118)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -21.1 | 7.9 | -29.0 | REVERSAL | -20.0 | 38.0 | -19.4 | -21.9 |
| 2h | -19.0 | 15.3 | -34.3 | REVERSAL | -22.8 | 40.1 | -22.7 | -22.8 |
| 4h | -22.2 | 33.7 | -55.9 | REVERSAL | -29.5 | 42.2 | -28.3 | -39.8 |
| 8h | -13.4 | 4.1 | -17.5 | REVERSAL | -16.4 | 46.0 | -14.8 | -23.8 |
| 12h | -50.4 | -2.1 | -48.3 | REVERSAL | -20.9 | 46.6 | -21.4 | -12.6 |
| 24h | -97.4 | -37.3 | -60.1 | REVERSAL | -36.8 | 46.2 | -36.8 | -33.9 |
| 48h | -106.1 | -45.9 | -60.2 | REVERSAL | -26.3 | 48.3 | -32.3 | 27.8 |

## SOLUSDT 200K  (BUY events=78, SELL events=62)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -22.8 | 2.7 | -25.5 | REVERSAL | -17.5 | 43.2 | -17.0 | -20.8 |
| 2h | -14.1 | 12.4 | -26.5 | REVERSAL | -19.4 | 43.2 | -17.6 | -31.2 |
| 4h | -7.1 | 23.6 | -30.7 | REVERSAL | -18.0 | 43.9 | -10.2 | -40.7 |
| 8h | 3.1 | 42.6 | -39.5 | REVERSAL | -25.9 | 45.7 | -27.5 | 5.0 |
| 12h | 19.8 | 61.9 | -42.1 | REVERSAL | -29.6 | 46.7 | -28.2 | -31.9 |
| 24h | 73.4 | -8.9 | 82.3 | CONTINUATION | 14.0 | 56.0 | 26.5 | -0.8 |
| 48h | 117.9 | 72.6 | 45.3 | CONTINUATION | -5.6 | 51.9 | -26.8 | 91.2 |

## SOLUSDT 500K  (BUY events=29, SELL events=23)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -13.6 | 10.2 | -23.8 | REVERSAL | -18.0 | 48.1 | -18.0 | -27.7 |
| 2h | -13.3 | -4.3 | -9.0 | REVERSAL | -7.9 | 48.1 | -7.9 | -3.5 |
| 4h | -2.2 | 55.9 | -58.1 | REVERSAL | -45.1 | 38.5 | -31.1 | -83.5 |
| 8h | 21.2 | 104.5 | -83.3 | REVERSAL | -19.4 | 44.2 | -12.9 | -90.9 |
| 12h | 16.6 | 105.8 | -89.2 | REVERSAL | -89.0 | 44.2 | -49.3 | -130.7 |
| 24h | 59.8 | 97.4 | -37.6 | REVERSAL | -4.0 | 50.0 | 16.5 | -99.9 |
| 48h | 59.0 | 311.5 | -252.5 | REVERSAL | -61.2 | 45.1 | -62.2 | 31.2 |

## SOLUSDT 1000K  (BUY events=13, SELL events=12)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -64.0 | 70.7 | -134.7 | REVERSAL | -76.4 | 32.0 | -70.1 | -104.5 |
| 2h | -58.4 | 9.7 | -68.1 | REVERSAL | -29.9 | 40.0 | -29.9 | -65.1 |
| 4h | -74.1 | 96.0 | -170.1 | REVERSAL | -81.7 | 36.0 | -80.2 | -213.7 |
| 8h | -36.4 | 132.4 | -168.8 | REVERSAL | -62.4 | 36.0 | -40.3 | -210.0 |
| 12h | -99.9 | 189.7 | -289.6 | REVERSAL | -109.9 | 36.0 | -106.0 | -235.2 |
| 24h | -7.9 | -12.9 | 5.0 | CONTINUATION | -6.5 | 48.0 | 95.0 | -126.1 |
| 48h | -99.3 | 239.6 | -338.9 | REVERSAL | -155.3 | 41.7 | -74.0 | -749.6 |

## BTCUSDT 200K  (BUY events=640, SELL events=615)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -5.4 | 4.3 | -9.7 | REVERSAL | -11.1 | 45.1 | -10.5 | -11.7 |
| 2h | -5.9 | 2.8 | -8.7 | REVERSAL | -10.1 | 46.8 | -11.3 | -7.8 |
| 4h | -2.8 | 3.3 | -6.1 | REVERSAL | -9.1 | 48.4 | -6.2 | -20.9 |
| 8h | 1.3 | 4.7 | -3.4 | REVERSAL | -8.2 | 49.3 | -2.6 | -26.4 |
| 12h | -6.3 | -0.6 | -5.7 | REVERSAL | -8.2 | 49.8 | -2.3 | -18.0 |
| 24h | 12.5 | -0.2 | 12.7 | CONTINUATION | 2.0 | 51.5 | 6.4 | -3.8 |
| 48h | 16.1 | 34.0 | -17.9 | REVERSAL | -14.1 | 49.1 | -12.8 | -19.5 |

## BTCUSDT 500K  (BUY events=261, SELL events=226)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -6.3 | 6.5 | -12.8 | REVERSAL | -12.5 | 45.1 | -13.6 | -11.6 |
| 2h | -3.4 | 7.8 | -11.2 | REVERSAL | -11.9 | 46.1 | -14.1 | -6.2 |
| 4h | 2.6 | 19.3 | -16.7 | REVERSAL | -10.7 | 48.0 | -7.4 | -20.9 |
| 8h | -5.0 | 16.3 | -21.3 | REVERSAL | -16.2 | 45.8 | -8.6 | -36.0 |
| 12h | -6.8 | 1.8 | -8.6 | REVERSAL | -10.8 | 47.9 | -9.8 | -20.0 |
| 24h | 7.2 | 4.0 | 3.2 | CONTINUATION | -7.4 | 49.8 | -7.6 | -6.2 |
| 48h | 14.3 | 47.2 | -32.9 | REVERSAL | -17.5 | 48.5 | -10.5 | -22.0 |

## BTCUSDT 1000K  (BUY events=128, SELL events=114)

| Horizon | BUY raw med | SELL raw med | signal_diff | dir | cont_net | cont_win% | cal_net | hold_net |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1h | -10.6 | 6.4 | -17.0 | REVERSAL | -15.2 | 44.0 | -16.1 | -2.9 |
| 2h | -10.1 | 10.9 | -21.0 | REVERSAL | -17.0 | 44.0 | -20.7 | 4.8 |
| 4h | -11.5 | 6.4 | -17.9 | REVERSAL | -16.9 | 45.6 | -13.5 | -18.0 |
| 8h | -19.1 | 8.6 | -27.7 | REVERSAL | -17.8 | 46.5 | -13.3 | -27.0 |
| 12h | -24.2 | -14.7 | -9.5 | REVERSAL | -10.3 | 48.5 | -11.2 | 0.8 |
| 24h | -12.1 | -11.7 | -0.4 | REVERSAL | -5.7 | 50.2 | -6.6 | 19.6 |
| 48h | -38.0 | 52.4 | -90.4 | REVERSAL | -53.4 | 44.1 | -75.0 | 40.1 |
