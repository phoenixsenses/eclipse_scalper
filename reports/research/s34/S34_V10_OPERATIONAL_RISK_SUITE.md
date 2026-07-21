# S34 v10 Operational Risk Suite

Generated: `2026-06-29T07:25:05.189981+00:00`

Mode: `OBSERVATION_RISK_ONLY_NO_LIVE_CHANGE`

## Risk Budget Modes

| Mode | Notional | Margin @40x | Loss bps basis | Oversize vs env |
| --- | ---: | ---: | ---: | ---: |
| `SURVIVAL` | $11.0 | $0.3 | 634.0 | 108.2x |
| `BALANCED` | $16.3 | $0.4 | 428.6 | 73.0x |
| `STOP_ASSISTED` | $39.8 | $1.0 | 175.7 | 29.9x |
| `CURRENT_ENV` | $1190.0 | $29.8 | 428.6 | 1.0x |

## Risk Of Ruin

### 30 Trades

| Mode | Ruin% | MinBalance<=15% | P05 final equity | P99 max DD |
| --- | ---: | ---: | ---: | ---: |
| `SURVIVAL` | 0.0 | 0.0 | $29.7 | $6.8 |
| `BALANCED` | 0.0 | 0.0 | $30.3 | $6.5 |
| `STOP_ASSISTED` | 0.0 | 0.0 | $32.5 | $5.3 |
| `CURRENT_ENV` | 73.5 | 77.3 | $-316.9 | $470.8 |

### 60 Trades

| Mode | Ruin% | MinBalance<=15% | P05 final equity | P99 max DD |
| --- | ---: | ---: | ---: | ---: |
| `SURVIVAL` | 0.0 | 0.0 | $26.2 | $11.0 |
| `BALANCED` | 0.0 | 0.0 | $27.4 | $10.0 |
| `STOP_ASSISTED` | 0.0 | 0.0 | $32.3 | $7.1 |
| `CURRENT_ENV` | 86.6 | 88.2 | $-519.1 | $731.1 |

### 100 Trades

| Mode | Ruin% | MinBalance<=15% | P05 final equity | P99 max DD |
| --- | ---: | ---: | ---: | ---: |
| `SURVIVAL` | 0.0 | 0.0 | $21.9 | $15.7 |
| `BALANCED` | 0.0 | 0.0 | $24.0 | $13.9 |
| `STOP_ASSISTED` | 0.0 | 0.0 | $32.7 | $8.5 |
| `CURRENT_ENV` | 93.8 | 94.6 | $-773.1 | $1017.0 |

## Pre-Trade Risk Card

- planned notional/margin: `$1190.0` / `$29.8`
- planned loss at realized 150bps stop: `$20.9`
- kill rule: `FIRST_TAIL_OR_10PCT_DD_PAUSE`

## Executor Readiness

- status: `READY_NO_POSITION`
- issues: `[]`
- checks: `{'pid_file': 25988, 'pid_alive': None, 'pid_read': 'unknown_access_denied_fallback_to_state_freshness', 'mode': 'LIVE', 'rule': 'S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID', 'active': None, 'state_orders_n': 0, 'exchange_position_amount': 0.0, 'exchange_s34_open_order_count': 0, 'kill_switch_file': 'runtime/KILL_SWITCH', 'kill_switch_exists': False, 'book_age_sec': 1.7, 'liq_age_sec': 580.7, 'mark_age_sec': 3.3, 'shadow_mirror_updated_at_utc': '2026-06-29T07:23:53.423897+00:00', 'state_age_sec': 10.5}`

## Fee Tier Reality

- status: `ACTUAL_FEE_TIER_UNKNOWN`
- assumed maker/taker: `2.0` / `3.05` bps
- read: operator confirm actual exchange fee tier; do not promote provision pocket while UNKNOWN

## Stop Slippage Tracker

- status: `FORWARD_TRACKER_SEEDED`
- historical stop N: `1`
- summary: `{'worst_realized_stop_bps': -175.7, 'median_realized_stop_bps': -175.7}`

## Kill Switch Drill

- status: `SIMULATED_ONLY_NO_FILE_CREATED`
- path: `D:\eclipse_scalper\runtime\KILL_SWITCH` exists=`False`
- expected: blocks new entries when file exists; does not auto-close existing active position

## Decision Journal

- status: `READY` path: `D:\eclipse_scalper\reports\research\s34\S34_OPERATOR_DECISION_JOURNAL.jsonl` rows: `1` created: `False`

## Latest Autopsy Cards

- `2026-06-20T14:08:51.159000+00:00` net=152.3 atomic=-1.4 failure=NOT_LARGE_LOSS pnl={'SURVIVAL': 0.2, 'BALANCED': 0.2, 'CURRENT_ENV': 18.1}
- `2026-06-21T23:33:42.690000+00:00` net=227.0 atomic=-2.2 failure=NOT_LARGE_LOSS pnl={'SURVIVAL': 0.2, 'BALANCED': 0.4, 'CURRENT_ENV': 27.0}
- `2026-06-26T02:48:30.475000+00:00` net=149.9 atomic=-1.1 failure=NOT_LARGE_LOSS pnl={'SURVIVAL': 0.2, 'BALANCED': 0.2, 'CURRENT_ENV': 17.8}
- `2026-06-26T10:39:33.530000+00:00` net=17.2 atomic=-18.5 failure=NOT_LARGE_LOSS pnl={'SURVIVAL': 0.0, 'BALANCED': 0.0, 'CURRENT_ENV': 2.0}
- `2026-06-26T13:18:54.877000+00:00` net=299.7 atomic=-13.7 failure=NOT_LARGE_LOSS pnl={'SURVIVAL': 0.3, 'BALANCED': 0.5, 'CURRENT_ENV': 35.7}
