# Eclipse Scalper — Environment Variable Reference

> **Generated:** 2026-03-01
> **Codebase:** `execution/entry_loop.py`, `execution/exit.py`, `execution/bootstrap.py`,
> `bot/core.py`, `exchanges/binance.py`, `execution/adaptive_guard.py`,
> `execution/anomaly_guard.py`, `execution/order_router.py`, `execution/intent_ledger.py`,
> `execution/telemetry.py`, `notifications/x_twitter.py`, `config/costs.py`,
> `tools/collection_watchdog.py`, `main.py`

## Important: Actual vs Described Names

The Phase 2 regime gate was described in setup docs with three separate variables
(`ENTRY_REGIME_GATE_ENABLED`, `ENTRY_REGIME_GATE_SIDE`, `ENTRY_REGIME_GATE_REGIME`), but the
actual implementation in `execution/entry_loop.py` uses a single combined variable:

| Described name | Actual name | Notes |
|---|---|---|
| `ENTRY_REGIME_GATE_ENABLED` | `ENTRY_REGIME` | Set to `"none"` to disable |
| `ENTRY_REGIME_GATE_SIDE` | (not separate) | Implied by signal direction |
| `ENTRY_REGIME_GATE_REGIME` | `ENTRY_REGIME` | `"up"`, `"down"`, or `"none"` |

This document uses the **actual code names** throughout.

---

## Configuration Priority

All execution variables follow this priority chain (highest wins):
1. `os.environ` / explicit shell env var
2. `bot.cfg.NAME` (config dataclass attribute)
3. Default value in code

The `load_dotenv()` call in `execution/bootstrap.py` loads `.env` at startup using
`override=False` — meaning already-set shell env vars always win over `.env` values.

---

## 1. Exchange / API Credentials

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `BINANCE_API_KEY` | `""` | str | `exchanges/binance.py`, `execution/bootstrap.py` | Binance API key (required for live/paper) |
| `BINANCE_API_SECRET` | `""` | str | `exchanges/binance.py`, `execution/bootstrap.py` | Binance API secret |
| `BINANCE_API_PASSWORD` | `""` | str | `execution/bootstrap.py` | Binance API password (not required for standard keys) |
| `API_KEY` | `""` | str | `execution/bootstrap.py` | Alias for `BINANCE_API_KEY` |
| `API_SECRET` | `""` | str | `execution/bootstrap.py` | Alias for `BINANCE_API_SECRET` |
| `API_PASSWORD` | `""` | str | `execution/bootstrap.py` | Alias for `BINANCE_API_PASSWORD` |
| `EXCHANGE` | `"binance"` | str | `execution/bootstrap.py` | Exchange name (ccxt identifier) |
| `DEFAULT_TYPE` | `"future"` | str | `execution/bootstrap.py` | ccxt market type: `"future"` or `"spot"` |
| `HTTPS_PROXY` | `None` | str | `exchanges/binance.py` | HTTPS proxy URL |
| `HTTP_PROXY` | `None` | str | `exchanges/binance.py` | HTTP proxy URL |

---

## 2. Trading Mode & Runtime

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `SCALPER_DRY_RUN` | `""` (falsy) | str | `main.py`, `execution/order_router.py` | **CRITICAL**: `"1"` = no real orders. `main.py` will REMOVE this unless `--dry-run` is passed. Use `python -m execution.bootstrap` for paper trading to respect `.env` value. |
| `SCALPER_MODE` | `"auto"` | str | `main.py` | Config mode: `"auto"`, `"micro"`, `"production"` |
| `SCALPER_EQUITY` | `None` | str | `main.py` | Override starting equity (e.g. `"45"`) |
| `SCALPER_SIGNAL_PROFILE` | `""` | str | `main.py` | Signal profile: `"micro"` or `""` (default) |
| `SCALPER_SIGNAL_DIAG` | `"0"` | bool | `execution/entry_loop.py` | Enable signal diagnostics logging |
| `MAKER_FEE_BPS` | `"1.0"` | float | `config/costs.py` | Maker fee in basis points (used in research tools) |

---

## 3. Symbols & Sizing

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ACTIVE_SYMBOLS` | `""` | str | `execution/bootstrap.py`, `bot/core.py` | Comma-separated symbol list, e.g. `"ETHUSDT,BTCUSDT"` |
| `MICRO_SYMBOL_WHITELIST` | `None` | str | `bot/core.py` | Whitelist filter for micro mode symbols |
| `MICRO_SYMBOL_LIMIT` | `"12"` | int | `bot/core.py` | Max symbols in micro mode |
| `SYMBOL_LIMIT` | `"60"` | int | `bot/core.py` | Max symbols in default mode |
| `FIXED_NOTIONAL_USDT` | from cfg | float | via `_cfg_env_float` | Fixed notional per trade in USDT |
| `LEVERAGE` | `1.0` | float | `execution/entry.py` | Base leverage level |
| `LEVERAGE_{sym_key}` | `""` | float | `execution/entry.py` | Per-symbol leverage override, e.g. `LEVERAGE_ETHUSDT=3` |
| `LEVERAGE_BY_SYMBOL` | `""` | str | `execution/entry.py` | JSON map of symbol → leverage |
| `LEVERAGE_BY_GROUP` | `""` | str | `execution/entry.py` | JSON map of group → leverage |
| `LEVERAGE_GROUP_DYNAMIC` | `"1"` | bool | `execution/entry.py` | Enable dynamic group-based leverage scaling |
| `LEVERAGE_GROUP_SCALE` | `0.7` | float | `execution/entry.py` | Group leverage scale factor |
| `LEVERAGE_GROUP_SCALE_MIN` | `1.0` | float | `execution/entry.py` | Minimum group leverage scale |
| `LEVERAGE_MIN` | `1.0` | float | `execution/entry.py` | Floor leverage |
| `LEVERAGE_MAX` | `125.0` | float | `execution/entry.py` | Ceiling leverage |
| `MARGIN_MODE` | from cfg | str | via cfg | `"cross"` or `"isolated"` |

---

## 4. Entry Loop Core

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ENTRY_POLL_SEC` | `1.0` | float | `execution/entry_loop.py` | Main loop tick interval (seconds) |
| `ENTRY_WAIT_FOR_DATA_READY_SEC` | `8.0` | float | `execution/entry_loop.py` | Max seconds to wait for data feed at startup |
| `ENTRY_PER_SYMBOL_GAP_SEC` | `2.5` | float | `execution/entry_loop.py` | Gap between symbol evaluations |
| `ENTRY_LOCAL_COOLDOWN_SEC` | `8.0` | float | `execution/entry_loop.py` | Per-symbol cooldown after entry attempt |
| `ENTRY_PENDING_BLOCK_SEC` | `30.0` | float | `execution/entry_loop.py` | Block new entries while an order is pending |
| `ENTRY_MIN_CONFIDENCE` | `0.0` | float | `execution/entry_loop.py`, `execution/bootstrap.py` | Minimum signal confidence to attempt entry |
| `ENTRY_RESPECT_KILL_SWITCH` | `True` | bool | `execution/entry_loop.py` | Respect kill-switch state; set to `"0"` to bypass (danger) |
| `ENTRY_MARGIN_INSUFFICIENT_BACKOFF_SEC` | `900.0` | float | `execution/entry_loop.py` | Backoff after margin-insufficient error (15 min default) |
| `ENTRY_SIZING_WARN_EVERY_SEC` | `30.0` | float | `execution/entry_loop.py` | Throttle sizing warning logs |
| `ENTRY_LOOP_MODE` | `""` | str | `execution/bootstrap.py`, `execution/diagnostics.py` | Entry loop variant selector |
| `HEDGE_MODE` | `False` | bool | `execution/entry_loop.py` | Binance hedge mode (separate long/short positions) |
| `HEDGE_SAFE` | `False` | bool | `execution/entry_loop.py` | Alias for `HEDGE_MODE` |
| `MIN_CONFIDENCE` | `None` | float | `execution/bootstrap.py` | Alias for `ENTRY_MIN_CONFIDENCE` |

---

## 5. Regime Gate (Phase 2)

Controls whether entries are gated on the current market regime (rolling 1h log-return direction).

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ENTRY_REGIME` | `"none"` | str | `execution/entry_loop.py` | Regime gate mode: `"up"` (only enter in UP regime), `"down"` (only DOWN), `"none"` (disabled). This is the primary gate control — replaces the three described variables `ENTRY_REGIME_GATE_ENABLED`, `ENTRY_REGIME_GATE_SIDE`, `ENTRY_REGIME_GATE_REGIME`. |
| `ENTRY_REGIME_LOOKBACK_SEC` | `3600` | int | `execution/entry_loop.py` | Rolling window for regime classification (seconds). Default = 1 hour. |
| `ENTRY_REGIME_DEBOUNCE_SEC` | `60` | int | `execution/entry_loop.py` | Minimum time a regime must persist before triggering a gate change (seconds). |
| `ENTRY_REGIME_BLOCK_TRANSITION` | `True` | bool | `execution/entry_loop.py` | Block entries when regime is `TRANSITION` (ambiguous crossover). |
| `ENTRY_REGIME_BLOCK_UNKNOWN` | `True` | bool | `execution/entry_loop.py` | Block entries when regime is `UNKNOWN` (insufficient data). |

**Validated configurations:**
- `ENTRY_REGIME=up` — sell_UP + buy_UP (both passive short and long entries in UP market condition). Validated NPA > 0 at fee <= 0.5 bps/leg.
- `ENTRY_REGIME=down` — BUY_DOWN (marginal; only min_imb=0.20 pocket viable, fee-sensitive).
- `ENTRY_REGIME=none` — No regime gating (baseline; lower Sharpe, higher drawdown).

---

## 6. Risk Manager (Phase 5)

Feature-flagged risk guard. Enabled via `ENTRY_REGIME_RISK_ENABLED=1`.

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ENTRY_REGIME_RISK_ENABLED` | `False` | bool | `execution/entry_loop.py` | Master feature flag for `RegimeRiskManager`. `"1"` or `"true"` to enable. |
| `RISK_MAX_CONCURRENT_POSITIONS` | `1` | int | `execution/entry_loop.py` | Maximum open positions at any time. |
| `RISK_MAX_DAILY_LOSS_BPS` | `50.0` | float | `execution/entry_loop.py` | Daily loss limit in basis points. Trading stops when exceeded. |
| `RISK_MAX_DAILY_TRADES` | `100` | int | `execution/entry_loop.py` | Maximum entries per calendar day. |
| `RISK_REGIME_CHANGE_POLICY` | `"hold"` | str | `execution/entry_loop.py` | Action on regime flip: `"hold"` (keep positions), `"close"` (exit all), `"reduce"` (scale down). |
| `RISK_REGIME_COOLDOWN_SEC` | `300.0` | float | `execution/entry_loop.py` | Cooldown after a regime change before new entries allowed (seconds). |
| `RISK_MAX_CONSECUTIVE_SCRATCHES` | `3` | int | `execution/entry_loop.py` | Pause trading after this many consecutive scratches. |
| `RISK_SCRATCH_PAUSE_SEC` | `600.0` | float | `execution/entry_loop.py` | Duration of scratch-triggered trading pause (seconds). |
| `RISK_MAX_DRAWDOWN_BPS` | `100.0` | float | `execution/entry_loop.py` | Max peak-to-trough drawdown in basis points before trading halts. |

---

## 7. Trade Logger (Phase 6)

SQLite-based per-trade logging. Enabled separately for entry and exit paths.

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ENTRY_TRADE_LOGGER_ENABLED` | `False` | bool | `execution/entry_loop.py` | Enable trade logger for the entry loop path. Logs risk events and entry decisions. |
| `ENTRY_TRADE_LOG_DB` | `"data/paper_trades.db"` | str | `execution/entry_loop.py` | SQLite database path for entry-side trade log. |
| `EXIT_TRADE_LOGGER_ENABLED` | `False` | bool | `execution/exit.py` | Enable trade logger for the exit loop path. Logs completed trades and P&L. |
| `EXIT_TRADE_LOG_DB` | `"data/paper_trades.db"` | str | `execution/exit.py` | SQLite database path for exit-side trade log. Should match `ENTRY_TRADE_LOG_DB`. |

---

## 8. Exit Loop

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `EXIT_ENABLED` | `True` | bool | `execution/exit.py` | Master exit loop switch. |
| `EXIT_TICK_SEC` | `2.0` | float | `execution/exit.py` | Exit loop tick interval (seconds). |
| `EXIT_MAX_HOLD_SEC` | `0.0` | float | `execution/exit.py` | Hard horizon for all trades (seconds). `0` = disabled. Set to `120` for validated h=120s strategy. |
| `EXIT_TIME_COOLDOWN_SEC` | `10.0` | float | `execution/exit.py` | Cooldown between time-based exit attempts. |
| `EXIT_STAGNATION_SEC` | `0.0` | float | `execution/exit.py` | Exit if price stagnates this long after fill (seconds). `0` = disabled. |
| `EXIT_STAGNATION_ATR` | `0.15` | float | `execution/exit.py` | Stagnation threshold as multiple of ATR. |
| `EXIT_MOM_ENABLED` | `False` | bool | `execution/exit.py` | Enable momentum-based exit signal. |
| `EXIT_MOM_MIN` | `0.0015` | float | `execution/exit.py` | Minimum momentum threshold for exit. |
| `EXIT_MOM_REQUIRE_BOTH` | `True` | bool | `execution/exit.py` | Require momentum signal in both directions. |
| `EXIT_VWAP_ENABLED` | `False` | bool | `execution/exit.py` | Enable VWAP cross exit signal. |
| `EXIT_VWAP_TF` | `"5m"` | str | `execution/exit.py` | VWAP timeframe. |
| `EXIT_VWAP_WINDOW` | `240` | int | `execution/exit.py` | VWAP lookback window (bars). |
| `EXIT_VWAP_REQUIRE_CROSS` | `True` | bool | `execution/exit.py` | Require VWAP cross (not just touch). |
| `EXIT_ATR_SCALE_ENABLED` | `False` | bool | `execution/exit.py` | Scale hold time by ATR regime. |
| `EXIT_ATR_SCALE_REF_PCT` | `0.003` | float | `execution/exit.py` | ATR reference as fraction of price. |
| `EXIT_ATR_SCALE_MIN` | `0.6` | float | `execution/exit.py` | Minimum ATR-based hold scale. |
| `EXIT_ATR_SCALE_MAX` | `1.6` | float | `execution/exit.py` | Maximum ATR-based hold scale. |

---

## 9. Scratch Engine (Phase 3/4)

Post-fill early-exit logic. Feature-flagged behind `EXIT_SCRATCH_ENABLED`.

> **Research result:** Scratch was disabled for the top pocket (min_imb>=0.50, int>=2500, spr<=0.0003)
> because backtest showed no NPA improvement. These settings are DISABLED by default in `.env.paper`.

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `EXIT_SCRATCH_ENABLED` | `False` | bool | `execution/exit.py` | Master scratch feature flag. `"1"` to enable. |
| `EXIT_SCRATCH_ADVERSE_BPS` | `0.0` | float | `execution/exit.py` | Exit if price moves adversely by this many bps after fill. `0` = disabled. |
| `EXIT_SCRATCH_COOLDOWN_SEC` | `10.0` | float | `execution/exit.py` | Minimum seconds between scratch triggers. |
| `EXIT_SCRATCH_TRAILING_BPS` | `0.0` | float | `execution/exit.py` | Trailing stop retracement threshold (bps). `0` = disabled. |
| `EXIT_SCRATCH_TAKE_PROFIT_BPS` | `0.0` | float | `execution/exit.py` | Take-profit target in bps. `0` = disabled. |
| `EXIT_SCRATCH_HARD_HORIZON_SEC` | `EXIT_MAX_HOLD_SEC` | float | `execution/exit.py` | Hard time-based exit override (seconds). Defaults to `EXIT_MAX_HOLD_SEC`. |

---

## 10. Correlation Groups

Advanced risk sizing for multi-symbol portfolios.

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `CORR_GROUPS` | `""` | str | `execution/entry_loop.py`, `execution/entry.py` | JSON definition of correlation groups. |
| `CORR_GROUP_MAX_POSITIONS` | `0` | int | `execution/entry.py` | Default max positions per group. |
| `CORR_GROUP_MAX_NOTIONAL_USDT` | `0.0` | float | `execution/entry.py` | Default max notional per group (USDT). |
| `CORR_GROUP_LIMITS` | `""` | str | `execution/entry_loop.py`, `execution/entry.py` | Per-group position limit overrides (JSON). |
| `CORR_GROUP_NOTIONAL` | `""` | str | `execution/entry_loop.py`, `execution/entry.py` | Per-group notional limit overrides (JSON). |
| `CORR_GROUP_SCALE` | `0.7` | float | `execution/entry_loop.py` | Base group-level sizing scale factor. |
| `CORR_GROUP_SCALE_MIN` | `0.25` | float | `execution/entry_loop.py` | Minimum group scale factor. |
| `CORR_GROUP_SCALE_BY_GROUP` | `""` | str | `execution/entry_loop.py` | Per-group scale overrides (JSON). |
| `CORR_GROUP_EXPOSURE_SCALE` | `0.7` | float | `execution/entry_loop.py` | Exposure-based sizing scale. |
| `CORR_GROUP_EXPOSURE_SCALE_MIN` | `0.25` | float | `execution/entry_loop.py` | Minimum exposure scale. |
| `CORR_GROUP_EXPOSURE_REF_NOTIONAL` | `0.0` | float | `execution/entry_loop.py` | Reference notional for exposure scaling. |
| `CORR_GROUP_EXPOSURE_SCALE_BY_GROUP` | `""` | str | `execution/entry_loop.py` | Per-group exposure scale (JSON). |
| `CORR_GROUP_EXPOSURE_SCALE_MIN_BY_GROUP` | `""` | str | `execution/entry_loop.py` | Per-group exposure scale min (JSON). |
| `CORR_GROUP_EXPOSURE_REF_NOTIONAL_BY_GROUP` | `""` | str | `execution/entry_loop.py` | Per-group reference notional (JSON). |

---

## 11. Data Quality Gate

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ENTRY_DATA_QUALITY_MIN` | `60.0` | float | `execution/entry_loop.py` | Minimum data quality percentage to allow entries. |
| `ENTRY_DATA_QUALITY_TF` | `"1m"` | str | `execution/entry_loop.py` | Timeframe for data quality check. |
| `ENTRY_DATA_QUALITY_WINDOW` | `120` | int | `execution/entry_loop.py` | Lookback window (bars) for quality check. |
| `ENTRY_DATA_QUALITY_EMIT_SEC` | `60.0` | float | `execution/entry_loop.py` | Emit interval for quality telemetry (seconds). |

---

## 12. Telemetry & Log Paths

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `SCALPER_LOG_DIR` | `"logs"` | str | `execution/telemetry.py` | Directory for all log files. |
| `TELEMETRY_PATH` | `"logs/telemetry.jsonl"` | str | `execution/telemetry.py`, `execution/adaptive_guard.py` | Main telemetry log file. |
| `TELEMETRY_DRIFT_PATH` | `"logs/telemetry_drift.jsonl"` | str | `execution/adaptive_guard.py` | Drift telemetry log. |
| `TELEMETRY_ANOMALY_ACTIONS` | `"logs/telemetry_anomaly_actions.json"` | str | `execution/exit.py`, `execution/entry.py` | Anomaly actions state file. |
| `TELEMETRY_ANOMALY_STATE` | `"logs/telemetry_anomaly_state.json"` | str | `execution/anomaly_guard.py` | Anomaly guard state file. |
| `TELEMETRY_GUARD_HISTORY_ACTIONS` | `"logs/telemetry_guard_history_actions.json"` | str | `execution/exit.py` | Guard history actions state. |
| `TELEMETRY_GUARD_HISTORY_EVENTS` | `"logs/telemetry_guard_history_events.jsonl"` | str | `execution/adaptive_guard.py` | Guard history events log. |
| `EXIT_SIGNAL_FEEDBACK_PATH` | `"logs/signal_exit_feedback.json"` | str | `execution/exit.py` | Signal exit feedback file. |
| `TELEMETRY_RECOVERY_STATE` | `"logs/telemetry_recovery_state.json"` | str | `execution/telemetry_recovery.py` | Telemetry recovery state. |
| `ADAPTIVE_GUARD_STATE` | `"logs/telemetry_adaptive_guard.json"` | str | `execution/adaptive_guard.py` | Adaptive guard state file. |
| `INTENT_LEDGER_PATH` | `""` | str | `execution/intent_ledger.py` | Intent ledger file path. |
| `EVENT_JOURNAL_PATH` | `""` | str | `execution/intent_ledger.py` | Event journal file path. |

---

## 13. Adaptive Guard

Fine-grained notional and leverage scaling based on recent error history.

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `ADAPTIVE_GUARD_DURATION_SEC` | `900` | int | `execution/adaptive_guard.py` | Default guard duration (seconds). |
| `ADAPTIVE_GUARD_PARTIAL_DELTA` | `0.1` | float | `execution/adaptive_guard.py` | Scale reduction for partial fill events. |
| `ADAPTIVE_GUARD_PARTIAL_DURATION_SEC` | `600` | int | `execution/adaptive_guard.py` | Duration of partial-fill guard (seconds). |
| `ADAPTIVE_GUARD_PARTIAL_ESCALATE_DELTA` | `0.2` | float | `execution/adaptive_guard.py` | Escalated scale reduction for repeated partials. |
| `ADAPTIVE_GUARD_PARTIAL_ESCALATE_DURATION_SEC` | `900` | int | `execution/adaptive_guard.py` | Duration of escalated partial guard. |
| `ADAPTIVE_GUARD_RETRY_DELTA` | `0.15` | float | `execution/adaptive_guard.py` | Scale reduction for retry events. |
| `ADAPTIVE_GUARD_RETRY_DURATION_SEC` | `600` | int | `execution/adaptive_guard.py` | Duration of retry guard. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_DELTA` | `0.1` | float | `execution/adaptive_guard.py` | Scale reduction from historical guard triggers. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_DURATION_SEC` | `900` | int | `execution/adaptive_guard.py` | Duration of historical guard effect. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_SCALE` | `0.85` | float | `execution/adaptive_guard.py` | Leverage scale from guard history. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_LEVERAGE_DURATION_SEC` | `900` | int | `execution/adaptive_guard.py` | Duration of leverage scaling from guard history. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_SCALE` | `0.8` | float | `execution/adaptive_guard.py` | Notional scale from guard history. |
| `ADAPTIVE_GUARD_GUARD_HISTORY_NOTIONAL_DURATION_SEC` | `900` | int | `execution/adaptive_guard.py` | Duration of notional scaling from guard history. |

---

## 14. Reliability Gate

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `RELIABILITY_GATE_PATH` | `""` | str | `execution/reliability_gate_runtime.py` | Gate state file path. |
| `RELIABILITY_GATE_MAX_REPLAY_MISMATCH` | `0` | int | `execution/reliability_gate_runtime.py` | Max allowed order-state replay mismatches. |
| `RELIABILITY_GATE_MAX_INVALID_TRANSITIONS` | `0` | int | `execution/reliability_gate_runtime.py` | Max invalid state transitions. |
| `RELIABILITY_GATE_MIN_JOURNAL_COVERAGE` | `0.90` | float | `execution/reliability_gate_runtime.py` | Minimum journal coverage fraction (0.0-1.0). |
| `RELIABILITY_GATE_CATEGORY_DEGRADE_SCORE` | `0.80` | float | `execution/reliability_gate_runtime.py` | Category degrade threshold. |
| `RELIABILITY_GATE_MAX_POSITION_MISMATCH` | `1` | int | `execution/reliability_gate_runtime.py` | Max position count mismatch. |
| `RELIABILITY_GATE_MAX_ORPHAN_COUNT` | `0` | int | `execution/reliability_gate_runtime.py` | Max orphaned orders. |
| `RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS` | `0.0` | float | `execution/reliability_gate_runtime.py` | Max coverage gap in journal (seconds). |
| `RELIABILITY_GATE_MAX_REPLACE_RACE_COUNT` | `1` | int | `execution/reliability_gate_runtime.py` | Max order replace race conditions. |
| `RELIABILITY_GATE_MAX_EVIDENCE_CONTRADICTION_COUNT` | `2` | int | `execution/reliability_gate_runtime.py` | Max evidence contradictions. |

---

## 15. Intent Ledger

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `INTENT_LEDGER_ENABLED` | `"1"` | bool | `execution/intent_ledger.py` | Enable intent ledger tracking. |
| `INTENT_LEDGER_REUSE_MAX_AGE_SEC` | `"900"` | int | `execution/intent_ledger.py` | Max age for ledger entry reuse (seconds). |

---

## 16. Bootstrap & Startup

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `BOOT_DATA_READY_TIMEOUT_SEC` | `"8"` | int | `execution/bootstrap.py` | Timeout waiting for data feed at startup (seconds). |
| `BOOT_REQUIRE_POSMGR` | `"1"` | bool | `execution/bootstrap.py` | Require position manager to start. |
| `BOOT_REQUIRE_EXIT` | `"1"` | bool | `execution/bootstrap.py` | Require exit loop to start. |
| `CLOSED_ORDERS_EXTRA_SYMBOLS` | `"8"` | int | `bot/core.py` | Extra symbols tracked for closed orders. |
| `CACHE_SAVE_SEC` | `"180"` | int | `bot/core.py` | State cache save interval (seconds). |
| `PYTHONUTF8` | `"1"` | str | `execution/bootstrap.py` | Set by bootstrap for UTF-8 output (Windows). |
| `PYTHONIOENCODING` | `"utf-8"` | str | `execution/bootstrap.py` | Set by bootstrap for UTF-8 encoding (Windows). |

---

## 17. Exit Telemetry & Signal Feedback Scaling

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `EXIT_TELEMETRY_HIGH_EXPOSURE_USDT` | `0.0` | float | `execution/exit.py` | Alert threshold for high exposure (USDT). |
| `EXIT_TELEMETRY_FORCE_HOLD_SEC` | `0.0` | float | `execution/exit.py` | Force hold duration from telemetry signal (seconds). |
| `EXIT_TELEMETRY_COOLDOWN_MULT` | `0.5` | float | `execution/exit.py` | Hold time multiplier from telemetry cooldown. |
| `EXIT_TELEMETRY_ALERT_INTERVAL_SEC` | `300` | float | `execution/exit.py` | Alert emit interval (seconds). |
| `EXIT_GUARD_HISTORY_HOLD_SCALE` | `0.7` | float | `execution/exit.py` | Hold time scale from guard history. |
| `EXIT_GUARD_HISTORY_STAGNATION_SCALE` | `0.7` | float | `execution/exit.py` | Stagnation scale from guard history. |
| `EXIT_SIGNAL_FEEDBACK_MIN_RATIO` | `0.25` | float | `execution/exit.py` | Minimum signal feedback ratio to apply scaling. |
| `EXIT_SIGNAL_FEEDBACK_MIN_COUNT` | `3` | int | `execution/exit.py` | Minimum count for feedback scaling to activate. |
| `EXIT_SIGNAL_FEEDBACK_HOLD_SCALE` | `0.7` | float | `execution/exit.py` | Hold time scale from signal feedback. |
| `EXIT_SIGNAL_FEEDBACK_STAGNATION_SCALE` | `0.7` | float | `execution/exit.py` | Stagnation scale from signal feedback. |

---

## 18. Notifications

| Variable | Default | Type | File | Description |
|---|---|---|---|---|
| `TELEGRAM_TOKEN` | `None` | str | `bot/core.py`, `tools/collection_watchdog.py` | Telegram bot token for trade alerts. |
| `TELEGRAM_CHAT_ID` | `None` | str | `bot/core.py`, `tools/collection_watchdog.py` | Telegram chat ID for alerts. |
| `ECLIPSE_TG_BOT_TOKEN` | `None` | str | `tools/collection_watchdog.py` | Alternate Telegram token for watchdog (falls back to `TELEGRAM_TOKEN`). |
| `ECLIPSE_TG_CHAT_ID` | `None` | str | `tools/collection_watchdog.py` | Alternate Telegram chat ID for watchdog (falls back to `TELEGRAM_CHAT_ID`). |
| `X_TWITTER_ENABLED` | `"0"` | bool | `notifications/x_twitter.py` | Enable X/Twitter publishing. |
| `X_CONSUMER_KEY` | `None` | str | `notifications/x_twitter.py` | Twitter OAuth consumer key. |
| `X_CONSUMER_SECRET` | `None` | str | `notifications/x_twitter.py` | Twitter OAuth consumer secret. |
| `X_ACCESS_TOKEN` | `None` | str | `notifications/x_twitter.py` | Twitter OAuth access token. |
| `X_ACCESS_TOKEN_SECRET` | `None` | str | `notifications/x_twitter.py` | Twitter OAuth access token secret. |
| `X_TWITTER_COOLDOWN_SEC` | `"30"` | int | `notifications/x_twitter.py` | Minimum seconds between tweets. |

---

## Quick Reference: Paper Trading Variables

Minimum required set for the 60-day paper trading run (sell_UP + buy_UP, ETHUSDT, h=120s):

```bash
# Credentials
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...

# Safety
SCALPER_DRY_RUN=1           # MANDATORY — paper mode

# Strategy
ACTIVE_SYMBOLS=ETHUSDT
ENTRY_REGIME=up             # Regime gate ON (sell_UP + buy_UP both enabled)
EXIT_MAX_HOLD_SEC=120        # Validated h=120s horizon

# Risk
ENTRY_REGIME_RISK_ENABLED=1
RISK_MAX_CONCURRENT_POSITIONS=1
RISK_MAX_DAILY_LOSS_BPS=50
RISK_MAX_DRAWDOWN_BPS=100

# Logging
ENTRY_TRADE_LOGGER_ENABLED=1
EXIT_TRADE_LOGGER_ENABLED=1
```

See `.env.paper` for the full production-ready configuration.
