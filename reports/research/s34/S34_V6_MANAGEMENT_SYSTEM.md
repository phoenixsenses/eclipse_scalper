# S34 v6 Management System

Generated: `2026-06-29T06:34:25.256987+00:00`

`RESEARCH_RISK_ONLY_NO_LIVE_ORDER_LOGIC_CHANGE` - no entry filter, live order logic, or executor setting was changed.

## 1. Tail-Aware Sizing

- historical min bps: `-507.2`
- stress tail abs bps: `634.0`
- equity assumption: `$35.0`
- current 85% margin style stress loss: `{'margin_usdt': 29.8, 'notional_usdt': 1190.0, 'stress_tail_loss_usdt': 75.5, 'stress_tail_loss_pct_equity': 215.6}`
- Kelly note: Kelly fraction forced to 0 for promotion because edge is unvalidated; use tail-budget sizing only.

| Risk % equity | Risk USDT | Max notional | Max margin @ leverage |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.1 | 1.4 | 0.0 |
| 0.5 | 0.2 | 2.8 | 0.1 |
| 1.0 | 0.3 | 5.5 | 0.1 |
| 2.0 | 0.7 | 11.0 | 0.3 |
| 5.0 | 1.8 | 27.6 | 0.7 |

## 2. Defensive Dissipation Observer

- status: `SHADOW_ONLY_NO_ORDER_CHANGE`
- primary reference: `{'config_id': 'tau120_dual_and_replQ50_decelQ50', 'cuts': {'replenish_cut': 10.7903, 'decel_cut': 0.4737}, 'hold_delta': 1961.7, 'cal_delta': -2872.2, 'read': 'tail cut in holdout, calibration worsened; not an execution rule'}`
- hard rule: Never alter live orders from this observer without operator sign-off and forward validation.

## 3. Regime-Degradation Monitor

- status: `DATA_INSUFFICIENT`
- closed_n: `11`
- all closed: N=11 sum=1081.6 mean=98.3 med=46.3 T3R=402.6 WR=1.0 maxL=13.3
- last5: N=5 sum=846.1 mean=169.2 med=152.3 T3R=167.1 WR=1.0 maxL=17.2
- last10: N=10 sum=1000.5 mean=100.0 med=44.0 T3R=321.5 WR=1.0 maxL=13.3
- triggers: `[]`

## 4. Failure-Mode Classifier

- status: `DESCRIPTIVE_ONLY_NOT_ENTRY_FILTER`
- large loss N: `101`
- counts: `{'MARKET_WIDE_DELEVERAGING': 78, 'BID_WALL_FAILED_TRAP': 10, 'LIQUIDITY_VACUUM_ADVERSE_SELECTION': 7, 'UNCLASSIFIED_NEGATIVE_SKEW': 6}`

| Label | Symbol | Month | Net bps | Route | Key state |
| --- | --- | --- | ---: | --- | --- |
| `LIQUIDITY_VACUUM_ADVERSE_SELECTION` | `ETHUSDT` | `2026-04` | -285.3 | `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | sync=idio, bid=shallow_bid, abs=vacuum_like, imb=-0.7, accel=3782.0 |
| `UNCLASSIFIED_NEGATIVE_SKEW` | `ETHUSDT` | `2026-04` | -256.9 | `ETHUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | sync=idio, bid=deep_bid, abs=absorbed, imb=0.2, accel=3215.8 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-04` | -113.7 | `ETHUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | sync=sync, bid=shallow_bid, abs=mixed, imb=-0.5, accel=294.1 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-04` | -271.1 | `ETHUSDT_SELL_FADE_LONG_T100K_v40_60_H4` | sync=sync, bid=deep_bid, abs=absorbed, imb=0.0, accel=12036.9 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-04` | -271.1 | `ETHUSDT_SELL_FADE_LONG_T150K_v40_60_H4` | sync=sync, bid=deep_bid, abs=absorbed, imb=0.0, accel=12036.9 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-04` | -271.1 | `ETHUSDT_SELL_FADE_LONG_T200K_v40_60_H4` | sync=sync, bid=deep_bid, abs=absorbed, imb=0.0, accel=12036.9 |
| `BID_WALL_FAILED_TRAP` | `ETHUSDT` | `2026-04` | -215.3 | `ETHUSDT_SELL_FADE_LONG_T100K_v20_28_H4` | sync=idio, bid=deep_bid, abs=absorbed, imb=0.6, accel=2321.6 |
| `UNCLASSIFIED_NEGATIVE_SKEW` | `ETHUSDT` | `2026-04` | -221.3 | `ETHUSDT_SELL_FADE_LONG_T150K_v20_28_H4` | sync=idio, bid=deep_bid, abs=absorbed, imb=0.4, accel=5438.9 |
| `BID_WALL_FAILED_TRAP` | `ETHUSDT` | `2026-04` | -224.0 | `ETHUSDT_SELL_FADE_LONG_T200K_v20_28_H4` | sync=idio, bid=deep_bid, abs=absorbed, imb=1.0, accel=7444.3 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-04` | -132.0 | `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | sync=sync, bid=deep_bid, abs=absorbed, imb=0.4, accel=2535.4 |
| `UNCLASSIFIED_NEGATIVE_SKEW` | `ETHUSDT` | `2026-04` | -108.3 | `ETHUSDT_SELL_FADE_LONG_T200K_v20_28_H4` | sync=idio, bid=deep_bid, abs=absorbed, imb=0.5, accel=-1219.4 |
| `MARKET_WIDE_DELEVERAGING` | `ETHUSDT` | `2026-06` | -187.8 | `ETHUSDT_SELL_FADE_LONG_T100K_v28_40_H4` | sync=sync, bid=deep_bid, abs=absorbed, imb=0.7, accel=5009.4 |

## 5. Explicit Kill Criteria

- `KILL_30D_SUM_NEGATIVE`: 30-day forward/live-like closed sum < 0 after >=5 closed fills
- `KILL_60D_SUM_NEGATIVE`: 60-day forward/live-like closed sum < 0 after >=5 closed fills
- `PAUSE_ROLLING_5_NEGATIVE`: rolling last 5 closed fills sum < 0
- `PAUSE_TAIL_BUDGET_BREACH`: any realized or shadow loss exceeds pre-accepted tail budget
- `NO_SCALE_UNTIL_VALIDATED`: no size increase until forward OOS positive across >=2 regimes

## 6. Tick-Level Maker Execution Realism

- status: `EXECUTION_REALISM_ONLY_NOT_ALPHA`
- model: book cross beyond quote plus agg_trade notional through quote >= queue_notional_usd
- event_count: `92`

| Rank | Config | Fill counts | Cal | Hold |
| ---: | --- | --- | --- | --- |
| 1 | `eth_tick_queue_o2_h300_qcross0.5_queue0_fee-0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 2 | `eth_tick_queue_o2_h300_qcross0.5_queue1000_fee-0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 3 | `eth_tick_queue_o2_h300_qcross0.5_queue5000_fee-0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 4 | `eth_tick_queue_o2_h300_qcross0.5_queue10000_fee-0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 5 | `eth_tick_queue_o2_h300_qcross0.5_queue25000_fee-0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 6 | `eth_tick_queue_o2_h300_qcross0.5_queue0_fee0` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 7 | `eth_tick_queue_o2_h300_qcross0.5_queue1000_fee0` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 8 | `eth_tick_queue_o2_h300_qcross0.5_queue5000_fee0` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 9 | `eth_tick_queue_o2_h300_qcross0.5_queue10000_fee0` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 10 | `eth_tick_queue_o2_h300_qcross0.5_queue25000_fee0` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 11 | `eth_tick_queue_o2_h300_qcross0.5_queue0_fee0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=38.0 mean=1.2 med=3.0 T3R=29.0 WR=0.906 maxL=-25.2 | N=60 sum=6.1 mean=0.1 med=3.0 T3R=-2.9 WR=0.933 maxL=-123.3 |
| 12 | `eth_tick_queue_o2_h300_qcross0.5_queue1000_fee0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=38.0 mean=1.2 med=3.0 T3R=29.0 WR=0.906 maxL=-25.2 | N=60 sum=6.1 mean=0.1 med=3.0 T3R=-2.9 WR=0.933 maxL=-123.3 |
| 13 | `eth_tick_queue_o2_h300_qcross0.5_queue5000_fee0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=38.0 mean=1.2 med=3.0 T3R=29.0 WR=0.906 maxL=-25.2 | N=60 sum=6.1 mean=0.1 med=3.0 T3R=-2.9 WR=0.933 maxL=-123.3 |
| 14 | `eth_tick_queue_o2_h300_qcross0.5_queue10000_fee0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=38.0 mean=1.2 med=3.0 T3R=29.0 WR=0.906 maxL=-25.2 | N=60 sum=6.1 mean=0.1 med=3.0 T3R=-2.9 WR=0.933 maxL=-123.3 |
| 15 | `eth_tick_queue_o2_h300_qcross0.5_queue25000_fee0.5` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=38.0 mean=1.2 med=3.0 T3R=29.0 WR=0.906 maxL=-25.2 | N=60 sum=6.1 mean=0.1 med=3.0 T3R=-2.9 WR=0.933 maxL=-123.3 |

## Read

- Management is not an entry filter. Failure modes are descriptive and must not be wired into entry selection.
- The only hard defense against the irreducible tail is sizing and kill/pause criteria.
- Dissipation remains observer-only because v4 did not validate expectancy improvement.
- 600GB data is used here for execution/queue realism, not for new directional alpha mining.
