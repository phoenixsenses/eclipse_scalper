# S34 v11 Forward Governance

Generated: `2026-06-29T07:33:35.737941+00:00`

Mode: `OBSERVATION_GOVERNANCE_ONLY_NO_LIVE_CHANGE`

## Forward Sample Integrity

- status: `NO_FORWARD_OOS_YET`
- source shadow N: `11`
- management N: `11`
- forward N / valid forward N: `0` / `0`
- missing in management: `0`
- extra management: `0`
- quality counts all: `{'VALID': 11, 'PARTIAL': 0, 'INVALID': 0}`
- data freshness sec: `{'book': 0.7, 'liquidations': 174.1, 'mark': 3.7}`

## Operator Governance

- status: `DECISION_REQUIRED`
- oversize vs BALANCED: `73.0x`
- real decision rows: `0`
- latest decision age h: `None`
- required action: operator must log REDUCE_MARGIN / DISARM / ARMED_KEEP_SIZE rationale

## Execution Truth Ledger

- status: `NO_S34_REAL_ORDER_TELEMETRY_FOUND`
- scanned telemetry tail rows: `5000`
- S34 order event N: `0`
- missing fields to instrument: `['local_send_ts', 'exchange_ack_ts', 'fill_ts', 'stop_send_ts', 'stop_ack_ts', 'realized_fee_bps', 'realized_stop_slippage_bps']`

## Fee Tier Verification

- status: `ACTUAL_FEE_TIER_UNKNOWN`
- assumed maker/taker bps: `2.0` / `3.05`
- read: No S34 real order fee samples in telemetry; provision pocket cannot be promoted until actual fee tier is verified.

## Kill Switch Drill

- status: `DRY_RUN_CONTRACT_ONLY`
- path: `D:\eclipse_scalper\runtime\KILL_SWITCH` exists=`False`
- would block new entries: `True`
- would auto-close active position: `False`

## Final Read

Forward validation is decision-ready only after valid_forward_N reaches the frozen gate and operator risk decisions are journaled.
