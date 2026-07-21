# Feature Availability Registry

- violation_count: `11`

| feature_name | source_table | definition | class | knowable_at rule | used_at_ts | VIOLATION? |
| --- | --- | --- | --- | --- | --- | --- |
| `liq_total_notional` | `runner_signal` | running cumulative notional up to threshold-cross | `RUNNING_CLUSTER` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `liq_count` | `runner_signal` | running liquidation count up to threshold-cross | `RUNNING_CLUSTER` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `liq_max_notional` | `runner_signal` | running max single liquidation up to threshold-cross | `RUNNING_CLUSTER` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `cluster_max_single_liq_share` | `runner_signal` | running max single share up to threshold-cross | `RUNNING_CLUSTER` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `intensity_per_sec` | `runner_signal` | running notional/sec up to threshold-cross | `RUNNING_CLUSTER` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `day_trend_bps` | `mark_prices` | UTC day-so-far return at decision time | `POINT_IN_TIME` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `day_range_bps` | `mark_prices` | UTC day-so-far range at decision time | `POINT_IN_TIME` | `signal.ts_ms` | `entry_ts_ms` | NO |
| `cluster_notional` | `liq_event_features` | full terminal cluster notional | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `cluster_count` | `liq_event_features` | full terminal cluster count | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `cluster_duration_sec` | `liq_event_features` | full terminal cluster duration | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `cluster_max_notional` | `liq_event_features` | full terminal max single liquidation | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `max_single_liq_share` | `liq_event_features` | full terminal max single liquidation share | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `shape_label` | `liq_event_features` | terminal cluster shape label | `TERMINAL_CLUSTER` | `cluster_end_ts_ms` | `event_ts_ms / entry_ts_ms` | YES |
| `entry_price` | `liq_event_outcome_labels` | route label entry price | `FORWARD_OUTCOME` | `label generation / route simulation` | `entry decision` | YES |
| `exit_price` | `liq_event_outcome_labels` | route label exit price | `FORWARD_OUTCOME` | `after entry path` | `entry decision` | YES |
| `net_bps` | `liq_event_outcome_labels` | route label realized net PnL | `FORWARD_OUTCOME` | `after exit` | `entry decision` | YES |
| `mfe_bps` | `liq_event_outcome_labels` | post-entry max favorable excursion | `FORWARD_OUTCOME` | `after entry path` | `entry decision` | YES |
| `mae_bps` | `liq_event_outcome_labels` | post-entry max adverse excursion | `FORWARD_OUTCOME` | `after entry path` | `entry decision` | YES |