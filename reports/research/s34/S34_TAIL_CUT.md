# S34 Tail-Cut Conditioning

Generated: `2026-06-28T21:44:09.175288+00:00`  |  ETHUSDT SELL deep-V>= 28.0bps, 200K, 4h, fee 3.05/side, resume 200K, btc_veto 30.0bps

Events: 54  |  resumed-after-entry: 50  |  btc-vetoed: 33

Knowable interventions vs the -330 tail. A real cut: much better MAX_LOSS and positive T3R on BOTH splits.

| Variant | split | N | sum | med | win% | max_loss | T3R |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_4h | cal | 35 | -379.3 | -4.1 | 48.6 | -291.8 | -883.0 |
| baseline_4h | hold | 16 | 657.9 | 19.4 | 56.2 | -338.0 | -289.8 |
| event_stop | cal | 37 | -410.1 | -15.7 | 16.2 | -68.1 | -765.2 |
| event_stop | hold | 17 | -208.1 | -16.5 | 11.8 | -38.6 | -291.4 |
| btc_filter_4h | cal | 13 | -673.3 | -14.7 | 30.8 | -291.8 | -886.8 |
| btc_filter_4h | hold | 6 | 157.2 | 7.3 | 50.0 | -137.4 | -276.0 |
| event_stop+btc_filter | cal | 14 | -413.8 | -24.8 | 0.0 | -68.1 | -394.7 |
| event_stop+btc_filter | hold | 7 | 81.3 | -16.5 | 28.6 | -38.6 | -103.2 |

## Reverter vs continuer separation (big losers, baseline net < -100)
- big losers N=8; of them resumed=8, btc-failed=4, either-flag=8 (100.0% caught by a knowable flag)
