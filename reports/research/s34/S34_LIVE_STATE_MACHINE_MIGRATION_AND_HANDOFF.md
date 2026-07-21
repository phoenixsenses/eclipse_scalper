# S34 Live State Machine Migration + Handoff

Updated: 2026-06-30 22:56 TRT

## Current Live State

- Active live alpha: `S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3`
- Engine: `STATE_MACHINE_V1`
- Mode: `LIVE`
- Current action: `READY_WAIT`
- Current block reason: `no_fresh_eligible_anchor`
- Open position: `null`
- Open state-machine orders: `0`
- Exchange reconciliation: `LONG=0.0`, `SHORT=0.0`, `state_machine_open_order_count=0`

This means the executor is armed and waiting. If a fresh eligible ETH SELL state-machine signal arrives and exchange/data paths remain healthy, the live executor is ready to place the configured live order.

## What Was Changed

### Live alpha routing

- Old V0.2 live route was removed from direct live startup.
- The only live executor started by the stack is now `tools.s34_state_machine_live_executor`.
- The live state file remains `runtime/s34_v_engine_live_state.json` for compatibility with existing dashboards and tools.
- `s34_v_engine_live_executor.pid` and `s34_state_machine_live_executor.pid` both point to the same state-machine live process for compatibility.

### Shadow / paper buckets

The following observation buckets are running and visible in the S34 live chart payload:

- Legacy S34 shadow paper runner: `s34_shadow_paper_runner`
- State-machine shadow runner: `s34_state_machine_shadow_runner`
- V0.2 mirror bucket: `s34_v_engine_v02_shadow_mirror`
- V0.2 remains observation-only and is not the live alpha.

Profit-lock 100/50 remains observer-only. It was not promoted into live exits because forward exits are still `N=0`.

### Dashboard / chart

`tools/s34_live_chart.py` was updated so the UI follows the state-machine alpha instead of the old V0.2 navigation framing:

- Live execution monitor now shows the alpha decision directly.
- Chart top card now reads state-machine action such as `READY_WAIT`, `HOLD_LONG`, `HOLD_SHORT`, `PENDING_STATE`, or `BLOCKED`.
- Liquidation pressure overlay was switched from BUY-liq framing to SELL-liq framing for the current alpha.
- The navigation indicator label was changed from `V02 navigation indicator` to `State-machine navigation indicator`.
- Shadow / paper buckets remain grouped in the chart payload and show process state.

### Process/runtime startup

- `start_eclipse.ps1` no longer starts the old V0.2 live executor directly.
- `start_eclipse.ps1` starts the state-machine shadow runner separately.
- `stop_eclipse.ps1` knows how to stop state-machine live/shadow PID files.
- `status_eclipse.ps1` now reports:
  - `s34_live_chart_alive`
  - `s34_state_machine_shadow_runner_alive`
  - `s34_state_machine_live_executor_alive`
  - V0.2 mirror status separately.
- PID stale-trust was removed from `start_eclipse.ps1`: a PID file is only reused if the process is actually alive.
- `scripts/collector_supervisor.py` now writes its own PID metadata and supervises the state-machine live executor.
- Microstructure collector is started with REST fallback enabled.
- Supervisor child stdout/stderr are redirected into per-child supervised logs to prevent mixed/noisy supervisor output.

## Final Verification

Command checks passed:

- `python -m py_compile scripts\collector_supervisor.py tools\s34_live_chart.py tools\s34_state_machine_live_executor.py tools\s34_realtime_shadow_runner.py`
- `powershell -NoProfile -ExecutionPolicy Bypass -File .\status_eclipse.ps1`
- `GET http://127.0.0.1:5050/api/data`

Runtime status after fresh external start:

```text
collector_supervisor_alive=True
collector_alive=True
bookticker_collector_alive=True
event_diary_alive=True
heartbeat_watchdog_alive=True
s34_live_chart_alive=True
s34_shadow_paper_runner_alive=True
s34_state_machine_shadow_runner_alive=True
s34_v_engine_v02_shadow_mirror_alive=True
s34_state_machine_live_executor_alive=True
watchdog_overall=GREEN
```

Chart API verification:

```json
{
  "rule": "S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3",
  "action": "READY_WAIT",
  "blocked_by": "no_fresh_eligible_anchor",
  "available": true,
  "orders": 0,
  "active": null,
  "reconciliation": {
    "position_amounts": {
      "LONG": 0.0,
      "SHORT": 0.0
    },
    "state_machine_open_order_count": 0
  },
  "shadow_buckets_available": true,
  "v02_mirror_available": true
}
```

## Important Operational Note

The durable start had to be run outside the sandboxed command job. Inside the managed shell, direct child processes can be killed when the command exits. Running `start_eclipse.ps1` externally/escalated keeps the process tree alive. The current stack was started that way and verified alive after the command returned.

## Known Constraints / Risks

- This is a live 40x executor as configured by the operator. No size, leverage, API key, or `.env` value was changed in this migration.
- Existing stop/bracket risk caveats still apply:
  - bracket placement is not atomic;
  - gap-through can make a nominal stop worse than expected;
  - historical tail risk is still the key risk driver.
- `Get-CimInstance Win32_Process` may return access denied in restricted shells. PID-file status and chart API were used as the reliable runtime checks.
- Collection health is currently GREEN, but Binance websocket/DNS/Windows socket errors have appeared intermittently in prior starts. Watchdog and status panels should be watched after long idle periods.

## Files Touched In This Migration

- `scripts/collector_supervisor.py`
- `start_eclipse.ps1`
- `stop_eclipse.ps1`
- `status_eclipse.ps1`
- `tools/s34_live_chart.py`
- `reports/research/s34/S34_LIVE_STATE_MACHINE_MIGRATION_AND_HANDOFF.md`

## Claude Context

The current production path is:

```text
ETH SELL cascade -> state-machine rule -> live executor
rule = S34_STATE_MACHINE_V1_ETH_SELL_BTC1000_DOW_SCORE3
```

Old V0.2 is not live:

```text
S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID
status = shadow/mirror only
```

Do not treat V0.2 mirror fills as live fills. Use the live state-machine bucket and `runtime/s34_v_engine_live_state.json` for actual live readiness/order state.
