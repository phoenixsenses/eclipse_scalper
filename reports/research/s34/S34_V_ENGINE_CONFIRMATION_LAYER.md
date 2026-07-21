# S34 V Engine Confirmation Layer

Generated: `2026-06-28T19:25:39.865627+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only test. Confirmation is known 15 minutes after maker fill, so this is not an entry-time filter.

Baseline closed-fill original: N=19 sum=876.1 med=37.0 T3R=348.4

## Variant Table

| Condition | Pass | Filter original | Failed original | Kill@15 hold | Delayed@15 entry |
| --- | ---: | --- | --- | --- | --- |
| `anchor_reclaimed_15m` | 14/19 | N=14 sum=846.2 med=39.4 T3R=399.9 | N=5 sum=29.9 med=-16.2 T3R=-181.9 | N=19 sum=534.8 med=30.4 T3R=88.5 | N=14 sum=477.4 med=12.7 T3R=110.0 |
| `btc_not_down_continues` | 16/19 | N=16 sum=957.2 med=39.4 T3R=435.9 | N=3 sum=-81.1 med=-16.2 T3R=-81.1 | N=19 sum=656.2 med=32.3 T3R=134.9 | N=16 sum=586.7 med=12.7 T3R=139.9 |
| `candle15_bull_reclaim` | 11/19 | N=11 sum=747.9 med=44.6 T3R=301.6 | N=8 sum=128.2 med=28.3 T3R=-141.5 | N=19 sum=350.1 med=26.1 T3R=-96.2 | N=11 sum=310.5 med=5.2 T3R=-24.8 |
| `anchor_and_btc` | 14/19 | N=14 sum=846.2 med=39.4 T3R=399.9 | N=5 sum=29.9 med=-16.2 T3R=-181.9 | N=19 sum=534.8 med=30.4 T3R=88.5 | N=14 sum=477.4 med=12.7 T3R=110.0 |
| `anchor_and_candle15` | 11/19 | N=11 sum=747.9 med=44.6 T3R=301.6 | N=8 sum=128.2 med=28.3 T3R=-141.5 | N=19 sum=350.1 med=26.1 T3R=-96.2 | N=11 sum=310.5 med=5.2 T3R=-24.8 |
| `btc_and_candle15` | 11/19 | N=11 sum=747.9 med=44.6 T3R=301.6 | N=8 sum=128.2 med=28.3 T3R=-141.5 | N=19 sum=350.1 med=26.1 T3R=-96.2 | N=11 sum=310.5 med=5.2 T3R=-24.8 |
| `all3` | 11/19 | N=11 sum=747.9 med=44.6 T3R=301.6 | N=8 sum=128.2 med=28.3 T3R=-141.5 | N=19 sum=350.1 med=26.1 T3R=-96.2 | N=11 sum=310.5 med=5.2 T3R=-24.8 |

## Best Practical Read

- `btc_not_down_continues` kill@15: N=19 sum=656.2 med=32.3 T3R=134.9; filter-original: N=16 sum=957.2 med=39.4 T3R=435.9; delayed@15: N=16 sum=586.7 med=12.7 T3R=139.9
- `anchor_reclaimed_15m` kill@15: N=19 sum=534.8 med=30.4 T3R=88.5; filter-original: N=14 sum=846.2 med=39.4 T3R=399.9; delayed@15: N=14 sum=477.4 med=12.7 T3R=110.0
- `anchor_and_btc` kill@15: N=19 sum=534.8 med=30.4 T3R=88.5; filter-original: N=14 sum=846.2 med=39.4 T3R=399.9; delayed@15: N=14 sum=477.4 med=12.7 T3R=110.0
- `candle15_bull_reclaim` kill@15: N=19 sum=350.1 med=26.1 T3R=-96.2; filter-original: N=11 sum=747.9 med=44.6 T3R=301.6; delayed@15: N=11 sum=310.5 med=5.2 T3R=-24.8
- `anchor_and_candle15` kill@15: N=19 sum=350.1 med=26.1 T3R=-96.2; filter-original: N=11 sum=747.9 med=44.6 T3R=301.6; delayed@15: N=11 sum=310.5 med=5.2 T3R=-24.8

## Fill Source Check

- `anchor_reclaimed_15m` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 14, 'not_confirmed': 5}`
- `btc_not_down_continues` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 16, 'not_confirmed': 3}`
- `candle15_bull_reclaim` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 11, 'not_confirmed': 8}`
- `anchor_and_btc` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 14, 'not_confirmed': 5}`
- `anchor_and_candle15` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 11, 'not_confirmed': 8}`
- `btc_and_candle15` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 11, 'not_confirmed': 8}`
- `all3` exit15 sources `{'book_ticker': 19}`, delayed15 sources `{'book_ticker': 11, 'not_confirmed': 8}`

## all3 Cards

Passed observations:
- 2026-06-26T13:18:54.877000+00:00 net=299.7 ret15=61.9 btc=btc_down_then_stable candle15=bull_reclaim
- 2026-06-16T04:31:11.525000+00:00 net=74.7 ret15=16.6 btc=btc_down_then_stable candle15=bull_reclaim
- 2026-06-18T15:57:31.634000+00:00 net=71.9 ret15=27.1 btc=btc_down_then_stable candle15=bull_reclaim
- 2026-06-25T15:03:23.104000+00:00 net=59.1 ret15=44.1 btc=btc_down_then_stable candle15=bull_reclaim
- 2026-06-12T15:56:42.488000+00:00 net=53.7 ret15=76.2 btc=btc_supportive candle15=bull_reclaim
- 2026-04-21T14:57:31.255000+00:00 net=44.6 ret15=36.2 btc=btc_down_then_stable candle15=bull_reclaim
- 2026-06-21T11:18:26.629000+00:00 net=37.0 ret15=34.8 btc=btc_supportive candle15=bull_reclaim
- 2026-06-26T10:39:33.530000+00:00 net=32.3 ret15=40.1 btc=btc_down_then_stable candle15=bull_reclaim

Failed observations:
- 2026-06-23T07:59:44.477000+00:00 net=-146.0 ret15=-211.9 btc=btc_down_continues candle15=bear_followthrough
- 2026-06-16T02:27:55.467000+00:00 net=-35.9 ret15=-0.7 btc=btc_supportive candle15=hammer_reversal
- 2026-06-25T16:32:03.169000+00:00 net=-16.2 ret15=-19.4 btc=btc_down_continues candle15=bear_followthrough
- 2026-04-20T14:08:39.155000+00:00 net=27.9 ret15=-40.3 btc=btc_softening candle15=bear_followthrough
- 2026-04-20T14:41:11.191000+00:00 net=28.7 ret15=-39.6 btc=btc_softening candle15=bear_followthrough
- 2026-06-17T01:17:01.753000+00:00 net=41.7 ret15=6.6 btc=btc_supportive candle15=neutral
- 2026-04-16T13:52:14.594000+00:00 net=81.1 ret15=-53.1 btc=btc_down_continues candle15=bear_followthrough
- 2026-06-26T02:48:30.475000+00:00 net=146.9 ret15=1.8 btc=btc_down_then_stable candle15=hammer_reversal
