# Ops Runbook — Eclipse Scalper Runtime

## Quick Reference

| Situation | Action | Details |
|-----------|--------|---------|
| Kill switch fired | Check `logs/kill_switch_state.json` | See [Kill Switch](#kill-switch-fired) |
| Circuit breaker tripped | Check dashboard `/api/risk-overview` | See [Circuit Breaker](#circuit-breaker-tripped) |
| Telegram dead | Check `logs/notifications_fallback.jsonl` | See [Telegram Down](#telegram-down) |
| Exchange degraded | Check `logs/health/heartbeat.json` | See [Exchange Degraded](#exchange-degraded) |
| Bot not responding | Check heartbeat file mtime | See [Bot Hung](#bot-hung-or-crashed) |
| Margin alert | Check margin ratio in heartbeat | See [Margin Alert](#margin-alert) |
| Position stuck | Check guardian logs for STUCK | See [Position Stuck](#position-stuck) |

---

## Kill Switch Fired

**Symptoms:** No new entries, "KILL SWITCH ACTIVE" in logs.

**Diagnosis:**
```bash
cat logs/kill_switch_state.json
# Look for: halted=true, reason, timestamp
```

**Recovery:**
1. Identify root cause from `reason` field
2. Check if positions are safe (dashboard or `logs/health/heartbeat.json`)
3. Clear kill switch via Telegram `/kill` command or dashboard control
4. Kill switch has a cooldown (`KILL_SWITCH_COOLDOWN_SEC=300s`) — if bot crashed within 5 min, it auto-halts for remaining time

**Prevention:**
- Tune `KILL_MAX_DATA_STALENESS_SEC` if data freshness trips it
- Tune `KILL_MAX_API_ERROR_RATE` if exchange flakiness trips it
- `KILL_SWITCH_EMERGENCY_FLAT=False` by default — only enable if you want auto-flatten

---

## Circuit Breaker Tripped

**Symptoms:** "CIRCUIT BREAKER" in logs, entries blocked.

**Diagnosis:**
```bash
grep -i "circuit" logs/*.log | tail -20
```

**Recovery:**
- Circuit breaker auto-resets after cooldown period (configurable in `bot.cfg`)
- If persistent: check if daily loss limit (`MAX_DAILY_LOSS_PCT`) was hit
- Reduce `MAX_RISK_PER_TRADE` or `MAX_PORTFOLIO_HEAT` via hot-reload:
  ```json
  // config/runtime_overrides.json
  {"MAX_RISK_PER_TRADE": 0.05, "MAX_PORTFOLIO_HEAT": 0.30}
  ```

---

## Telegram Down

**Symptoms:** No Telegram messages, circuit breaker icon in dashboard.

**Diagnosis:**
```bash
# Check fallback log for queued messages
tail -20 logs/notifications_fallback.jsonl

# Check circuit breaker state
grep "CIRCUIT BREAKER OPEN" logs/*.log
```

**How it works:**
- 5 consecutive Telegram failures → circuit opens (120s cooldown)
- All messages fall back to `logs/notifications_fallback.jsonl`
- After cooldown expires, half-open retry attempt
- Success → circuit closes, normal operation resumes

**Recovery:**
1. Check Telegram bot token: `echo $TELEGRAM_BOT_TOKEN`
2. Check chat ID: `echo $TELEGRAM_CHAT_ID`
3. Test manually: `curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"`
4. If token expired: update `.env` and restart bot
5. If rate limited: wait 120s for circuit to attempt recovery

---

## Exchange Degraded

**Symptoms:** "ENTERING DEGRADED MODE" in logs, no new entries.

**Diagnosis:**
```bash
cat logs/health/heartbeat.json
# Look for: "exchange_degraded": true
```

**How it works:**
- Guardian probes exchange every 60s via `_health_check()`
- 3 consecutive failures → DEGRADED mode
- Entries blocked, exits still allowed (safety-first)
- Auto-recovers when next probe succeeds

**Recovery:**
1. Check Binance status: https://www.binance.com/en/support/announcement
2. Check API key permissions
3. Check network connectivity
4. Bot auto-recovers — no manual action needed unless exchange is truly down

---

## Bot Hung or Crashed

**Symptoms:** Heartbeat file older than 30s, no log activity.

**Diagnosis:**
```bash
# Check heartbeat freshness
stat logs/health/heartbeat.json
cat logs/health/heartbeat.json

# Check if process exists
cat logs/health/bot.pid 2>/dev/null && ps aux | grep $(cat logs/health/bot.pid)

# Check guardian timeout logs
grep "TIMED OUT" logs/*.log | tail -10
```

**Recovery:**
1. If process exists but hung: send SIGTERM, wait 30s, then SIGKILL
2. Restart bot: `python main.py`
3. Bot auto-loads kill switch state and reconciles positions on startup
4. Check `logs/kill_switch_state.json` for crash cooldown (5 min window)

**Prevention:**
- Guardian has 45s timeout on every step — prevents single step from blocking loop
- All steps are guardian-safe (catch all exceptions, never fatal)
- Each guardian cycle has overall timeout (configurable)

---

## Margin Alert

**Symptoms:** "MARGIN" alert via Telegram or in alert rules log.

**Diagnosis:**
```bash
cat logs/health/heartbeat.json | python -m json.tool
grep "margin" logs/*.log | tail -10
```

**Alert thresholds (configurable in `config/alert_rules.json`):**
- WARNING: margin_ratio < 10%
- CRITICAL: margin_ratio < 5%

**Immediate actions:**
1. If CRITICAL: consider manual position reduction
2. Check Binance position panel for unrealized PnL
3. Verify stop-losses are in place for all positions
4. Consider enabling `KILL_SWITCH_EMERGENCY_FLAT=True` for auto-flatten

**Tune via hot-reload:**
```json
{"MAX_RISK_PER_TRADE": 0.03, "MAX_CONCURRENT_POSITIONS": 3}
```

---

## Position Stuck

**Symptoms:** "POSITION STUCK" alert, position open > TTL without exit activity.

**Diagnosis:**
```bash
grep "STUCK" logs/*.log | tail -10
```

**Recovery:**
1. Check position on Binance — is the stop/TP still active?
2. If stop was cancelled: bot will re-place it next reconcile cycle
3. If position is truly abandoned: manually close on exchange
4. Default TTL: 3600s (1 hour), configurable via `POSITION_STUCK_TTL_SEC`

---

## Config Hot-Reload

**How to change runtime config without restart:**

1. Create/edit `config/runtime_overrides.json`:
```json
{
  "MAX_RISK_PER_TRADE": 0.05,
  "MIN_CONFIDENCE": 0.80,
  "FIXED_NOTIONAL_USDT": 15.0,
  "NOTIFY_ON_ENTRY": false
}
```

2. Bot detects changes within ~10s and applies them
3. Telegram notification confirms which fields changed

**Blocked fields (require restart):**
- `LEVERAGE`, `ACTIVE_SYMBOLS`, `TIMEFRAME*`, `KILL_SWITCH_ENABLED`, `TRADING_HOURS_UTC`

**Full list of reloadable fields:** See `config/hot_reload.py:HOT_RELOAD_ALLOWED`

---

## Monitoring Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/risk-overview` | Consolidated risk metrics |
| `GET /metrics` | Prometheus-format metrics |
| `WS /ws/live` | Real-time WebSocket push |
| `GET /api/status` | Bot status snapshot |
| `GET /api/positions` | Current positions |

---

## Log Files

| File | Purpose | Rotation |
|------|---------|----------|
| `logs/eclipse.log` | Main bot log | Daily, 50MB max |
| `logs/telemetry.jsonl` | Machine telemetry | Weekly compression |
| `logs/notifications_fallback.jsonl` | Telegram fallback queue | Manual review |
| `logs/health/heartbeat.json` | External liveness probe | Overwritten each cycle |
| `logs/health/bot.pid` | Process ID file | Cleaned on exit |
| `logs/kill_switch_state.json` | Kill switch persistence | Overwritten on change |

---

## Emergency Procedures

### Full Flatten (Nuclear Option)
1. Enable via Telegram `/kill` command
2. Or set `KILL_SWITCH_EMERGENCY_FLAT=True` and trip the kill switch
3. Emergency module cancels all open orders and market-closes all positions

### Manual Restart
```bash
# Graceful
kill -SIGTERM $(cat logs/health/bot.pid)
sleep 30
python main.py

# Force (last resort)
kill -9 $(cat logs/health/bot.pid)
python main.py
# Bot will reconcile on startup
```

### API Key Rotation
1. Create new API key on Binance (futures trading + read permissions)
2. Update `.env` with new `BINANCE_API_KEY` and `BINANCE_API_SECRET`
3. Restart bot
