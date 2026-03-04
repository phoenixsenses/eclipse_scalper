# END TO END BUILD

## 1) System Overview
Eclipse Scalper pipeline:
- Collector: Binance streams -> `data/microstructure.db` (`agg_trades`, `mark_prices`, `liquidations`) in WAL mode.
- Features: `core/micro_features.py` computes imbalance, trade_intensity (per-minute equivalent), spread proxy, mark price freshness/readiness.
- Signal: `core/micro_signal.py` evaluates validated micro pockets + regime gate and emits binary confidence (`1.0` on full match, otherwise no signal result).
- Regime: `core/regime.py` rolling 1h log-return + debounce, consumed by entry loops (`execution/entry_loop.py`, `execution/entry_loop_full.py`).
- Risk: `core/regime_risk.py`, kill switch (`risk/kill_switch.py`), guardrails and cooldowns in entry loop.
- Execution: `execution/order_router.py` + `execution/entry.py` (+ passive order handling).
- Logging/Audit: `core/trade_logger.py`, `data/paper_trades.db`, structured logs under `logs/`.
- Monitoring: Telegram bot/status tools, watchdog, health/maintenance scripts.

## 2) Run Commands
### Paper trading boot (recommended)
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_paper_trading.ps1
```

### Direct bootstrap (debug)
```powershell
python -m execution.bootstrap
```

### Supervisor (auto-restart wrapper)
```powershell
python scripts\supervisor.py
```

### DB maintenance (checkpoint/backup/disk)
```powershell
python -m tools.db_maintenance
```

### Telegram command bot
```powershell
python -m tools.telegram_bot
```

### Daily report
```powershell
python -m tools.daily_report
```

### Risk attribution (live papertrade PnL decomposition)
```powershell
python -m tools.risk_attribution --live-root data/live --out-md reports/risk_attribution.md
```

### Execution quality audit (Track 1 KPI)
```powershell
python -m tools.execution_quality_audit --in-parquet data/live/papertrades_live.parquet --out-md reports/execution_quality_audit.md --out-json reports/execution_quality_audit.json
```

### Preflight gate (startup hard checks)
```powershell
python -m tools.preflight_check
```

### Runtime profile lock (freeze/enforce)
```powershell
python -m tools.freeze_runtime_profile --write-lock
python -m tools.freeze_runtime_profile --enforce
```

### Incident bundle (forensics snapshot)
```powershell
python -m tools.incident_bundle
```

### Release tag prep (local safety gate)
```powershell
python -m tools.prepare_release_tag --tag vYYYY.MM.DD-stable
```

### Scratch calibration reports
```powershell
python -m tools.run_scratch_calibration --symbol ETHUSDT --db data/microstructure.db --adverse-sweep 2.0:10.0 --trail-sweep 2.0,3.0,4.0,5.0 --fee-bps 0.5 --exec-model passive_realistic
python -m tools.compare_scratch_live_vs_backtest --trade-db data/paper_trades.db --backtest-sell-json reports/SCRATCH_CALIBRATION_SELL_UP.json --backtest-buy-json reports/SCRATCH_CALIBRATION_BUY_UP.json --out-md reports/SCRATCH_LIVE_VS_BACKTEST.md
```

### Fill timing analysis (5s/10s/30s buckets)
```powershell
python -m tools.analyze_fill_timing --live-parquet data/live/papertrades_live.parquet --trade-db data/paper_trades.db --out-md reports/FILL_TIMING_ANALYSIS.md
```

### Feature distribution analysis (+plots)
```powershell
python -m tools.feature_distribution_analysis --db data/microstructure.db --symbol ETHUSDT --lookback-hours 24 --out reports/FEATURE_STATIONARITY.md --plots-dir reports/plots
```

### Paper/backtest reconciliation
```powershell
python -m tools.reconcile_paper_vs_backtest --paper-db data/paper_trades.db --rank-json reports/PASSIVE_POCKET_RANKING.json --out reports/RECONCILIATION.md
```

### Day-60 sweep + aggregate + go/no-go
```powershell
python -m tools.run_full_sweep --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH.md --workers 2 --output-dir runs/day60_latest
python -m tools.aggregate_sweep_results --run-dir runs/day60_latest --out reports/DAY60_MASTER_RESULTS.md
python -m tools.evaluate_go_nogo --manifest runs/day60_latest/manifest.json --out reports/GO_NOGO_FRAMEWORK.md
```

## 3) Environment Variables and Resolution Rules
### Dotenv auto-load
- Loader order: `.env.paper` -> `.env` -> ambient env.
- Load mode: `override=False`.
- Result: pre-set shell variables remain authoritative; missing variables are filled from dotenv files.

### ACTIVE_SYMBOLS precedence
Resolution order must be treated as:
1. Shell/Process env `ACTIVE_SYMBOLS` (highest)
2. `.env.paper` / `.env` (loaded with `override=False`)
3. `cfg.ACTIVE_SYMBOLS` fallback
4. internal default (`BTCUSDT`)

### Telegram token fallback chain
For runtime token resolution:
1. `TELEGRAM_BOT_TOKEN`
2. `TELEGRAM_TOKEN`
3. `ECLIPSE_TG_BOT_TOKEN`

Chat id fallback chain:
1. `TELEGRAM_CHAT_ID`
2. `ECLIPSE_TG_CHAT_ID`

Order placement (limit-only path):
- `MICRO_SIGNAL_ORDER_PLACEMENT_MODE` = `best | inside_spread | adaptive`
- `MICRO_SIGNAL_QUEUE_DEPTH_THRESHOLD` (adaptive switch threshold)
- `MICRO_SIGNAL_TICK_SIZE` (fallback tick if exchange metadata unavailable)

## 4) Acceptance Checklist
Observable startup checks (first 60 lines):
- `[bootstrap] dotenv loaded from .env.paper` (or `.env`).
- `[bootstrap] ACTIVE_SYMBOLS ... ['ETHUSDT']`.
- `ENTRY_CFG ... min_conf=0.00` when `ENTRY_MIN_CONFIDENCE=0.0` and adaptive guard disabled.
- `ENTRY_LOOP ... micro signal provider enabled symbol=ETHUSDT`.
- `[MICRO_HEARTBEAT] ETHUSDT ready=1 ...` with low feature age.
- `[REGIME] ETHUSDT regime=UP|DOWN ...` every ~60s (not permanently UNKNOWN after warmup).
- No `UnicodeEncodeError` in console on Windows Turkish locale.
- No telegram token missing warning when token exists.

Order-path checks:
- Binary micro signal behavior:
  - pocket full match + regime/gates pass -> `conf=1.00` + order path proceeds.
  - partial mismatch -> no present signal / explicit non-match reason.

## 5) Troubleshooting
### A) `MIN_CONFIDENCE` stuck at `0.72`
- Verify `ENTRY_MIN_CONFIDENCE` exists in shell or `.env.paper`.
- Verify `config/settings.py` maps `MIN_CONFIDENCE` from env.
- Verify adaptive guard not elevating threshold (`ENTRY_ADAPTIVE_GUARD_ENABLED=0` for isolation).

### B) Micro confidence is fractional
- `core/micro_signal.py` must return binary confidence only.
- Ensure no legacy heuristic path is active by default.

### C) `.env.paper` not loading
- Run from repo root.
- Confirm top-of-file dotenv block exists before project imports in bootstrap/tools.
- Confirm `override=False` and that conflicting shell vars are not stale.

### D) Turkish Windows Unicode error (`Φ`)
- `utils/logging.py` stream handlers must bind UTF-8 stream wrapper.
- Optional hardening:
```powershell
$env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'
```

### E) PowerShell quoting bug (`start_paper_trading.ps1` line ~134)
- Avoid nested `GetEnvironmentVariable(...)` in interpolated strings.
- Use two-step assignment then `Write-Host`.

### F) WebSocket disconnect / ping timeouts
- Check watchdog logs and reconnect backoff.
- Verify exchange connectivity and DNS.
- Confirm collector loop keeps updating `mark_prices`.

### G) WAL growth / disk pressure
- Run `python -m tools.db_maintenance`.
- Verify `wal_checkpoint(TRUNCATE)` success.
- Check backup retention and disk free-space alerts.

### H) Startup blocked by preflight
- Open `reports/PREFLIGHT_CHECK.md` and fix `FAIL` rows first.
- Typical issues: stale DB, missing `SCALPER_DRY_RUN=1`, unwritable `logs/`/`reports/`, low free disk.

### I) Runtime drift between sessions
- Freeze and compare runtime profile:
  - `python -m tools.freeze_runtime_profile --write-lock`
  - `python -m tools.freeze_runtime_profile --enforce`
- If mismatch, align env and rerun startup.

## 6) Do Not Change During 60-Day Run
- Do not change validated pocket filters:
  - `MICRO_SIGNAL_POCKETS`
- Do not alter validated horizon/regime research assumptions except runtime hardening.
- Allow only operational improvements (reliability, observability, maintenance, automation).

## End-to-End Smoke Run
```powershell
# 1) Compile critical modules
python -m py_compile config\settings.py execution\bootstrap.py execution\entry_loop.py execution\entry_loop_full.py core\micro_signal.py utils\logging.py

# 2) Run tests
pytest -q --tb=short

# 3) Boot smoke (first lines)
python -m execution.bootstrap 2>&1 | Select-Object -First 60

# 3b) Offline network-safe smoke (no exchange init, no loops)
$env:BOOTSTRAP_SMOKE_ONLY='1'
$env:BOOTSTRAP_SKIP_EXCHANGE_INIT='1'
$env:BOOTSTRAP_SMOKE_SEC='2'
python -m execution.bootstrap 2>&1 | Select-Object -First 60
$env:BOOTSTRAP_SMOKE_ONLY=$null
$env:BOOTSTRAP_SKIP_EXCHANGE_INIT=$null
$env:BOOTSTRAP_SMOKE_SEC=$null

# 4) Push one-shot status
python -m tools.push_status

# 5) Full start script
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_paper_trading.ps1
```
