# S34 V Engine Management Alerts

Generated: `2026-06-29T07:34:27.522854+00:00`

Mode: `NOTIFY_ONLY_NO_ACTION`

Severity: `critical`

Delivery: `EMIT_STATE_CHANGED` state_changed=`True`

| Severity | Code | Message | Recommendation |
| --- | --- | --- | --- |
| critical | `S34_OVERSIZE` | S34 live env planned margin $29.8 vs tail-budget $0.3 (oversize 107.8x). | operator reduce margin to budget or disarm; no automatic action taken |
| warning | `S34_STOP_GAP_THROUGH` | Configured stop 150.0 bps realized worst -175.7 bps. | treat stop as partial protection; size remains primary control |
| warning | `S34_ATOMICITY_GAP` | Adverse move observed inside fill-to-stop gap; worst -18.5 bps. | document atomic bracket requirement; operator sign-off required for live logic change |
| info | `S34_ATOMICITY_SCAN_OK_SMALL_N` | Tick scan found no catastrophic 2s gap; worst -22.7 bps. | continue observation; absence in 23 fills is not proof of zero risk |
| critical | `S34_KILL_CRITERIA` | Kill/pause recommendations triggered: ['OPERATOR_SIZE_REVIEW_REQUIRED'] | recommendation only; executor not changed |
| critical | `S34_OPERATOR_DECISION_REQUIRED` | Oversize vs BALANCED 73.0x with no real decision journal row. | operator must log REDUCE_MARGIN / DISARM / ARMED_KEEP_SIZE rationale |

## Dashboard Line

`{"env_planned_margin_usdt": 29.8, "forward_closed_n": 0, "forward_sum_bps": 0.0, "kill_status": "TRIGGERED_RECOMMENDATION_ONLY", "max_stop_budget_margin_usdt": 1.0, "max_tail_budget_margin_usdt": 0.3, "oversize_multiple": 107.8, "permission": "UNVALIDATED_OBSERVATION_ONLY", "recommendation": "OPERATOR_SIZE_REVIEW_OR_DISARM_UNTIL_FORWARD_VALIDATION", "regime_status": "DATA_INSUFFICIENT", "rule_id": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID", "size_status": "ALERT_OVERSIZE_OPERATOR_ACTION_REQUIRED", "updated_at_utc": "2026-06-29T07:01:02.104147+00:00"}`
