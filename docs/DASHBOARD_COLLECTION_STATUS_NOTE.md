## Dashboard Collection Status Desync After Reboot

Date: 2026-03-06

### Observed

After a machine restart, the dashboard frontend showed data collection as stopped or stale.

At the same time, the actual collection pipeline was still running and writing data:

- collector process active:
  - `python -m data.microstructure_collector --symbols BTCUSDT,ETHUSDT --db-path data/microstructure.db`
- event diary process active:
  - `python -m data.event_diary --db-path data/microstructure.db --csv-path data/event_diary.csv`

### Verified Runtime State

Root runtime paths in use:

- `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\data\microstructure.db`
- `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\data\event_diary.csv`

Observed healthy signals during verification:

- `agg_trades` latest timestamp was only a few seconds behind wall clock
- `mark_prices` latest timestamp was only a few seconds behind wall clock
- `event_diary.csv` was updating

### Likely Cause

The issue appears to be dashboard/backend status reporting rather than actual collection downtime.

Most likely causes:

- dashboard polling reading a stale status source
- backend health/status endpoint checking a different path than the active runtime path
- cache or in-memory status not resetting cleanly after reboot
- watchdog/frontend looking at worktree-local paths instead of root runtime paths

### Priority

Not blocking research work.

Research can continue because the real data pipeline is active and writing.

### Follow-up Task

Recommended runtime/dashboard task:

`dashboard collection status desync after reboot`

### Suggested Debug Order

1. Verify which DB/CSV path the backend health endpoint reads.
2. Verify which path the frontend status panel assumes.
3. Compare root repo runtime paths vs worktree paths.
4. Check whether watchdog status survives reboot with stale state.
5. Confirm polling refresh and cache invalidation after app restart.
