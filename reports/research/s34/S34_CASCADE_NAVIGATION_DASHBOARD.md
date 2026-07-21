# S34 Cascade Navigation Dashboard

Generated: `2026-07-01T06:41:06.846764+00:00`  |  as_of: `2026-07-01T06:40:59.178000+00:00`

NAVIGATION / PERMISSION layer -- not a trade trigger. Every feature is knowable at `as_of` (asserted via feature-availability). `family_edge` comes from the clean holdout-validated route recheck; market momentum never grants permission on its own.

| Lane | Phase | Cont/Fade Mom | Session | DayTrend | BTC Aligned | Spread | Edge | V-Engine | Permission |
| --- | --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| ETHUSDT BUY | DEAD/distributed | 0.0/0.0 | ASIA | 13.3 | False | 0.1 | UNKNOWN | INACTIVE | OBSERVE_ONLY |
| ETHUSDT SELL | FORMING/one_shot_dominant | 0.0/0.0 | ASIA | 13.3 | True | 0.1 | PROVEN_5TH_WAVE | INACTIVE | OBSERVE_ONLY |
| SOLUSDT BUY | DEAD/distributed | 0.0/0.0 | ASIA | 69.9 | False | 1.3 | UNKNOWN | INACTIVE | OBSERVE_ONLY |
| SOLUSDT SELL | DEAD/distributed | 0.0/0.0 | ASIA | 69.9 | True | 1.3 | UNKNOWN | INACTIVE | OBSERVE_ONLY |
| BTCUSDT BUY | DEAD/distributed | 0.0/0.0 | ASIA | 2.8 | False | 0.0 | UNKNOWN | INACTIVE | OBSERVE_ONLY |
| BTCUSDT SELL | DEAD/distributed | 0.0/0.0 | ASIA | 2.8 | True | 0.0 | UNKNOWN | INACTIVE | OBSERVE_ONLY |

## Execution Management

| Size | Kill | Regime | Env Margin | Tail Budget Margin | Stop Budget Margin | Oversize | Recommendation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED | TRIGGERED_RECOMMENDATION_ONLY | DATA_INSUFFICIENT | 29.8 | 0.3 | 1.0 | 107.8 | OPERATOR_SIZE_REVIEW_OR_DISARM_UNTIL_FORWARD_VALIDATION |

## EXECMGMT Sizing

| Equity | Tail Budget | Current Notional | Tail-Budget Notional | Oversize | Recommended Max Margin | Flag | Mode |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 35.2941 | 0.3529 | 1200.0 | 20.0877 | 59.7 | 0.5022 | URGENT_OVERSIZE_GT_10X | READ_ONLY_RECOMMENDATION_NO_ACTION |

## STOPPROT Stop Protection

| Nominal Stop | Worst Real Fill | Gap-Through | Flag | Atomicity | Unprotected Window | Mode |
| ---: | ---: | ---: | --- | --- | ---: | --- |
| 150.0 | -175.7 | 25.7 | PARTIAL_PROTECTION | ENTRY_THEN_STOP_POLL_LOOP | 2.0 | READ_ONLY_WARNING_NO_ACTION |

## Risk Review v9

| Kill Rule | Tick Atomicity | Worst Gap | Forward Gate |
| --- | --- | ---: | --- |
| FIRST_TAIL_OR_10PCT_DD_PAUSE | NO_TICK_CATASTROPHIC_GAP_FOUND | -22.7 | FROZEN_DECISION_GATE |

## Forward Governance v11

| Forward Integrity | Valid Forward N | Operator Governance | Oversize vs Balanced | Execution Truth | Fee Tier |
| --- | ---: | --- | ---: | --- | --- |
| NO_FORWARD_OOS_YET | 0 | DECISION_REQUIRED | 73.0 | NO_S34_REAL_ORDER_TELEMETRY_FOUND | ACTUAL_FEE_TIER_UNKNOWN |

## V0.2 Shadow Mirror

| Protocol | Permission | Decision | Live Match | Rows | Recent N | Recent Sum | Recent Median | Recent T3R | Kill | Latest Signal |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID | EXPLORATORY_V_FADE_V0_2_SHADOW_MIRROR | OBSERVE_ONLY_NO_ORDER | True | 12 | 8 | 1014.0 | 114.9 | 335.0 | False | 2026-06-30T13:32:16.371000+00:00 |

## V0.2 H4 Shadow Management

| Protocol | Decision | H2 Sum | H4 Sum | H4 T3R | Cross Policy Sum | SL150 Touches | Queue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID | H4_SHADOW_OBSERVATION_ONLY | 1081.6 | 1738.6 | 819.2 | 1790.7 | 0 | PROXY_ONLY_TOP_OF_BOOK |

## Sizing Shadow Paper

| Mode | N | Notional | Margin | Lev | Sum bps | PnL USDT | End Equity | Max DD % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CURRENT_ENV | 12 | 1190.0 | 29.75 | 40.0 | 1161.5 | 138.219 | 173.219 | 0.0 |
| BALANCED | 12 | 16.3 | 0.4 | 40.0 | 1161.5 | 1.893 | 36.893 | 0.0 |
| SURVIVAL | 12 | 11.0 | 0.3 | 40.0 | 1161.5 | 1.28 | 36.278 | 0.0 |

Sizing shadow is observation only; it does not place orders or change live size.
