# Live Validate + Protect Audit - 2026-06-29

Mode: operational audit only. No live rule, sizing, route, or executor code was changed.

## Result

- Canonical live executor PID: `25988`
- Command: `python -W ignore -u -m tools.s34_v_engine_live_executor --live --confirm-live-orders`
- Supervisor: `scripts/collector_supervisor.py` manages `tools.s34_v_engine_live_executor`
- State file: `runtime/s34_v_engine_live_state.json`
- State mode: `LIVE`
- Active lifecycle: `null`
- Exchange reconciliation in state:
  - `position_amount`: `0.0`
  - `s34ve_open_order_count`: `0`
  - `s34ve_open_client_ids`: `[]`
- Allowed rule:
  - `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

## Duplicate Process Cleanup

Two old duplicate live executor processes were observed during audit:

- PID `5616` was stopped after PID file showed canonical PID `19984`.
- Later the supervisor launched canonical PID `25988`; old orphan PID `19984` was stopped.

Final state after cleanup: one live executor process remains, PID `25988`.

## Risk Envelope Observed

From `.env` and executor constants:

- `S34_LIVE_TRADING_ENABLED=1`
- `S34_LIVE_DRY_RUN=0`
- `S34_LIVE_MAX_LEVERAGE=40`
- `S34_LIVE_MARGIN_PCT_ETH=85`
- `S34_LIVE_MARGIN_USDT=30` fallback if balance unavailable
- `S34_V_ENGINE_LIVE_STOP_BPS=150`
- Initial maker offset: `20 bps`
- Replace after: `300s`
- Replacement offset: `5 bps`
- Time exit horizon: `2h`
- Max open positions env: `1`
- Kill switch file: `runtime/KILL_SWITCH`

## Read

Live is armed and can submit if the configured V Engine signal appears. This is an operational statement, not an alpha-validation statement. The configured route remains unvalidated under the research discipline; no scaling or new live route was added.
