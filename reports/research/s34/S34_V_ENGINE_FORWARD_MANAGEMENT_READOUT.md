# S34 V Engine Forward Management Readout

Generated: `2026-06-29T07:01:02.104124+00:00`

Mode: `OBSERVATION_RISK_ONLY_NO_LIVE_ORDER_LOGIC_CHANGE`

Frozen forward start: `2026-06-29T06:44:41.536000+00:00`

Live rule: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

## Ledger

- source shadow rows: `11`
- management rows: `11`
- forward rows: `0`
- pre-forward reference rows: `11`

## Live State Snapshot

- active: `None`
- open/order rows in state: `0`
- reconciliation: `{'position_amount': 0.0, 's34ve_open_client_ids': [], 's34ve_open_order_count': 0, 'updated_at_utc': '2026-06-29T07:00:55.903691+00:00'}`

## Tail-Aware Size Monitor

- status: `ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED`
- action: `RECOMMENDATION_ONLY_NO_AUTO_SIZE_CHANGE`
- risk budget: `2.0%` equity
- max tail-budget notional: `$11.0`
- max tail-budget margin: `$0.3`
- max stop-budget notional: `$39.8`
- max stop-budget margin: `$1.0`
- env planned margin/notional: `$29.8` / `$1190.0`
- env stress loss: `$75.4` = `215.6%` equity
- oversize multiple vs budget: `107.8`
- oversize multiple vs stop-budget: `29.9`

## Atomicity Gap

- status: `ALERT_ADVERSE_IN_GAP`
- observed N: `11`
- alert N: `2`
- worst adverse bps: `-18.5`
- recommendation: `DOCUMENT_AND_RECOMMEND_ATOMIC_BRACKET_OPERATOR_SIGNOFF_REQUIRED`

## Regime / Kill

- regime status: `DATA_INSUFFICIENT`
- forward summary: `{'n': 0, 'sum_bps': 0.0, 'mean_bps': None, 'median_bps': None, 'win_rate': None, 'max_loss_bps': None, 't3r_bps': 0.0}`
- reference summary: `{'n': 11, 'sum_bps': 1081.6, 'mean_bps': 98.3, 'median_bps': 46.3, 'win_rate': 1.0, 'max_loss_bps': 13.3, 't3r_bps': 402.6}`
- kill status: `TRIGGERED_RECOMMENDATION_ONLY`
- triggered: `['OPERATOR_SIZE_REVIEW_REQUIRED']`

## Failure-Mode Tracking

- status: `DESCRIPTIVE_ONLY_NOT_ENTRY_FILTER`
- forward large loss N: `0`
- counts: `{}`

## ETH Provision Observation

- status: `FORWARD_OBSERVATION_ONLY_NOT_ALPHA`
- binding question: actual maker fee tier; positive maker fee kills the pocket

| Rank | Scenario | Forward | Reference |
| ---: | --- | --- | --- |
| 1 | `eth_provision_o2_h300_qcross0.5_queue1000_fee-0.5` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=55.0 med=5.0 T3R=40.0 maxL=5.0 |
| 2 | `eth_provision_o2_h300_qcross0.5_queue1000_fee0` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=44.0 med=4.0 T3R=32.0 maxL=4.0 |
| 3 | `eth_provision_o2_h300_qcross0.5_queue1000_fee0.5` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=33.0 med=3.0 T3R=24.0 maxL=3.0 |
| 4 | `eth_provision_o2_h300_qcross1_queue1000_fee-0.5` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=-33.2 med=5.0 T3R=-48.2 maxL=-83.2 |
| 5 | `eth_provision_o2_h300_qcross1_queue1000_fee0` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=-43.7 med=4.0 T3R=-55.7 maxL=-83.7 |
| 6 | `eth_provision_o2_h300_qcross1_queue1000_fee0.5` | N=0 sum=0.0 med=None T3R=0.0 maxL=None | N=11 sum=-54.2 med=3.0 T3R=-63.2 maxL=-84.2 |

## Dashboard Line

`{"env_planned_margin_usdt": 29.8, "forward_closed_n": 0, "forward_sum_bps": 0.0, "kill_status": "TRIGGERED_RECOMMENDATION_ONLY", "max_stop_budget_margin_usdt": 1.0, "max_tail_budget_margin_usdt": 0.3, "oversize_multiple": 107.8, "permission": "UNVALIDATED_OBSERVATION_ONLY", "recommendation": "OPERATOR_SIZE_REVIEW_OR_DISARM_UNTIL_FORWARD_VALIDATION", "regime_status": "DATA_INSUFFICIENT", "rule_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID", "size_status": "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED", "updated_at_utc": "2026-06-29T07:01:02.104147+00:00"}`

## Guardrails

- This readout emits recommendations only.
- No live order logic, size, config, or executor state was changed.
- Failure modes are descriptive and must not become entry filters.
