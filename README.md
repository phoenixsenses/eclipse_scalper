



---



```markdown

\# Eclipse Scalper



A modular, safety-first crypto trading bot built in Python.



\*\*Eclipse Scalper\*\* is a research-driven scalping framework designed for

Binance (spot \& futures), with a strong emphasis on:

\- risk control

\- dry-run safety

\- clean architecture

\- debuggability

\- and gradual escalation from simulation â†’ micro â†’ live



This repository contains the \*\*core engine\*\*, not exchange secrets or runtime state.

\## Quick Strategy Map

\- Alpha path: signal components + filters decide entry/exit intents.

\- Reliability path: exchange evidence updates belief/debt, then controller clamps entry risk.

\- Entry controls: confidence gates, notional/leverage scaling, correlation caps, adaptive cooldowns.

\- Exit controls: protective exits stay available even when reliability posture is degraded.

\- Reconcile loop: resolves local-vs-exchange drift and repairs safety gaps.

\- Telemetry loop: dashboards/alerts feed adaptive guards and runtime posture.

See `docs/strategies_summary.md` and `docs/execution_system_summary.md` for the full model.



---



\## âš ï¸ Disclaimer (Read This First)



This project is \*\*experimental software\*\*.



\- It can lose money.

\- It will behave exactly as you tell it to.

\- There are no guarantees of profitability.



Use \*\*dry-run mode\*\* first.  

Use \*\*micro capital\*\* next.  

Only then consider real exposure.



You are responsible for every trade executed.



---



\## Project Structure



```



eclipse\_scalper/

â”‚

â”œâ”€â”€ bot/                # Core runner and orchestration

â”œâ”€â”€ execution/          # Entry, exit, order routing, guardian loops

â”œâ”€â”€ strategies/         # Signal logic (Eclipse Scalper strategy)

â”œâ”€â”€ risk/               # Kill-switches and safety logic

â”œâ”€â”€ exchanges/          # Exchange adapters (Binance)

â”œâ”€â”€ notifications/      # Telegram, alerts

â”œâ”€â”€ config/             # Static configuration helpers

â”œâ”€â”€ utils/              # Logging, helpers

â”œâ”€â”€ tools/              # Smoke tests \& diagnostics

â”‚

â”œâ”€â”€ main.py             # Main entry point

â”œâ”€â”€ guardian.py         # Global safety guardian

â”œâ”€â”€ signal\_check.py     # Signal diagnostics

â”œâ”€â”€ settings.py         # Runtime settings

â””â”€â”€ requirements.txt    # Python dependencies



````



---



\## What Is \*Not\* in This Repo (On Purpose)



The following are \*\*ignored by Git\*\* for safety:



\- `.env` (API keys, secrets)

\- `logs/`

\- `brain/` (state / persistence)

\- `data/`

\- runtime logs and artifacts



You must create these locally.



---



\## Requirements



\- Python \*\*3.10+\*\*

\- Git

\- Binance account (for live trading)

\- Telegram bot token (optional, for alerts)



---



\## Installation



Clone the repository:



```bash

git clone https://github.com/phoenixsenses/eclipse\_scalper.git

cd eclipse\_scalper

````



Create a virtual environment (recommended):



```bash

python -m venv .venv

.venv\\Scripts\\activate   # Windows

```



Install dependencies:



```bash

pip install -r requirements.txt

```



---



\## Environment Variables (`.env`)



Create a `.env` file in the project root.



Example (DO NOT COMMIT THIS):



```env

BINANCE\_API\_KEY=your\_key\_here

BINANCE\_API\_SECRET=your\_secret\_here



SCALPER\_DRY\_RUN=1

SCALPER\_SIGNAL\_PROFILE=micro

TELEGRAM\_TOKEN=optional

TELEGRAM\_CHAT\_ID=optional

EXCHANGE\_ADAPTER=binance  # optional; swap in another registered adapter (defaults to binance)

For quick local testing you can set `EXCHANGE_ADAPTER=mock` (or any registered name) before running the bootstrap, e.g.:
```bash
EXCHANGE_ADAPTER=mock python main.py
```
Weâ€™ve now registered a real `coinbase` adapter as well. Point `EXCHANGE_ADAPTER=coinbase` when you want to run through that venue (ensure `COINBASE_API_KEY`, `COINBASE_API_SECRET`, and `COINBASE_API_PASSPHRASE` are set in `.env`).

```

\*\*Bootstrap profile (choose one):\*\*

Micro (small equity / safer):
```env
SCALPER\_MODE=micro
SCALPER\_DRY\_RUN=1
SCALPER\_SIGNAL\_PROFILE=micro
ENTRY\_LOOP\_MODE=full
ACTIVE\_SYMBOLS=BTCUSDT,ETHUSDT
```

Production (larger equity):
```env
SCALPER\_MODE=production
SCALPER\_DRY\_RUN=0
ENTRY\_LOOP\_MODE=full
ACTIVE\_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
```



\*\*Important flags:\*\*



\* `SCALPER\_DRY\_RUN=1` â†’ no real orders

\* `SCALPER\_DRY\_RUN=0` â†’ live trading (dangerous)
\* `ENTRY\_LOOP\_MODE=full|basic` â†’ choose entry loop when using `execution/bootstrap.py` (default: full if available)



---



\## Strategy Guards (Recommended)

These are optional safety gates inside the strategy:

\- `SCALPER\_DATA\_MAX\_STALE\_SEC` (seconds): block signals if 1m data is too old.
\- `SCALPER\_SESSION\_UTC` (hours): trade only in these UTC ranges.
  \- Format: `"13-17,19-23"` or `"8-12"` (inclusive).
  \- Empty string disables the filter.
\- `SCALPER\_SESSION\_MOM\_UTC` (hours): require 5m + 15m momentum during these UTC ranges.
  \- Empty string disables the guard.
\- `SCALPER\_SESSION\_MOM\_MIN` (float): minimum absolute momentum for both 5m and 15m (long >= +min, short <= -min).
\- `SCALPER\_COOLDOWN\_LOSSES`: number of consecutive losses to trigger cooldown.
\- `SCALPER\_COOLDOWN\_MINUTES`: cooldown duration (minutes).

Example:
```env
SCALPER\_DATA\_MAX\_STALE\_SEC=120
SCALPER\_SESSION\_UTC=12-20
SCALPER\_SESSION\_MOM\_UTC=13-17
SCALPER\_SESSION\_MOM\_MIN=0.0015
SCALPER\_COOLDOWN\_LOSSES=2
SCALPER\_COOLDOWN\_MINUTES=30
```

---

\## Volatility Regime Guard + Adaptive Thresholds (Optional)

Detect low/high volatility regimes using ATR% + Bollinger width and adjust thresholds.

\- `SCALPER\_VOL\_REGIME\_ENABLED` (default `1`): enable regime detection.
\- `SCALPER\_VOL\_REGIME\_GUARD` (default `1`): block signals in LOW regime.
\- `SCALPER\_VOL\_REGIME\_LOW\_ATR\_PCT` / `SCALPER\_VOL\_REGIME\_HIGH\_ATR\_PCT`: ATR% bounds for LOW/HIGH.
\- `SCALPER\_VOL\_REGIME\_LOW\_BB\_PCT` / `SCALPER\_VOL\_REGIME\_HIGH\_BB\_PCT`: BB width bounds for LOW/HIGH.
\- `SCALPER\_VOL\_REGIME\_LOW\_MULT` (default ~`1.15`): tightens `DYN\_MOM\_MULT`, `VOL\_Z\_TH`, `VOL\_Z\_SOFT\_TH`, `VWAP\_BASE\_DIST`.
\- `SCALPER\_VOL\_REGIME\_HIGH\_MULT` (default ~`0.90`): loosens those thresholds in high vol.

Example:
```env
SCALPER\_VOL\_REGIME\_ENABLED=1
SCALPER\_VOL\_REGIME\_GUARD=1
SCALPER\_VOL\_REGIME\_LOW\_ATR\_PCT=0.0010
SCALPER\_VOL\_REGIME\_HIGH\_ATR\_PCT=0.0030
SCALPER\_VOL\_REGIME\_LOW\_BB\_PCT=0.012
SCALPER\_VOL\_REGIME\_HIGH\_BB\_PCT=0.035
SCALPER\_VOL\_REGIME\_LOW\_MULT=1.15
SCALPER\_VOL\_REGIME\_HIGH\_MULT=0.90
```

---

\## Guard Defaults (Reference)

```env
SCALPER\_DATA\_MAX\_STALE\_SEC=180
SCALPER\_SESSION\_UTC=""
SCALPER\_SESSION\_MOM\_UTC=""
SCALPER\_SESSION\_MOM\_MIN=0.0015
SCALPER\_COOLDOWN\_LOSSES=0
SCALPER\_COOLDOWN\_MINUTES=0
SCALPER\_TREND\_CONFIRM=1
SCALPER\_TREND\_CONFIRM\_MODE=gate
SCALPER\_TREND\_CONFIRM\_TF=1h
SCALPER\_TREND\_CONFIRM\_FAST=50
SCALPER\_TREND\_CONFIRM\_SLOW=200
```

---

\## Volume-weighted confidence (Optional)

Boosts confidence when the latest volume is meaningfully above the recent average and pulls
it down when volume collapses, helping the signal gate prefer sustained moves over noise.

\- `SCALPER\_VOL\_RATIO\_WEIGHT` (default `0.4` micro / `0.35` production): multiplier applied to the
  clamped volume ratio delta before it adjusts confidence.
\- `SCALPER\_VOL\_RATIO\_MAX\_BOOST` (default `0.4` micro / `0.5` production): upper bound on the
  positive volume delta (so the multiplier never raises confidence by more than ~40â€“50%).
\- `SCALPER\_VOL\_RATIO\_MAX\_DROP` (default `0.2` micro / `0.25` production): upper bound on how far
  the multiplier can push confidence downward (keeps the factor above ~0.6â€“0.8 even when volumes collapse).

The classifier emits `vol_ratio` plus the resulting multiplier, so you can track the volume pulse in telemetry.

\## Strategy Confidence Tuning (Optional)

These knobs adjust confidence scoring without changing the underlying vote weights:

\- `SCALPER\_CONFIDENCE\_POWER` (default `1.0`): power curve. >1 compresses midrange, <1 boosts midrange.
\- `SCALPER\_CONFIDENCE\_MIN` (default `0.0`): clamp floor for confidence.
\- `SCALPER\_CONFIDENCE\_MAX` (default `1.0`): clamp ceiling for confidence.
\- `SCALPER\_CONFIDENCE\_DIAG` (default off): log raw -> powered -> clamped confidence (throttled).

Example:
```env
SCALPER\_CONFIDENCE\_POWER=1.2
SCALPER\_CONFIDENCE\_MIN=0.05
SCALPER\_CONFIDENCE\_MAX=0.85
SCALPER\_CONFIDENCE\_DIAG=1
```

---

\## Trend Confirmation (Optional)

Higher-TF EMA alignment adds a trend confirmation component.

\- `SCALPER\_TREND\_CONFIRM` (default on): enable the component.
\- `SCALPER\_TREND\_CONFIRM\_MODE` (`gate` or `vote`): hard-block signals or add votes (default `gate`).
\- `SCALPER\_TREND\_CONFIRM\_TF` (default `1h`): timeframe for EMA alignment.
\- `SCALPER\_TREND\_CONFIRM\_FAST` (default `50`): fast EMA length.
\- `SCALPER\_TREND\_CONFIRM\_SLOW` (default `200`): slow EMA length.

Example:
```env
SCALPER\_TREND\_CONFIRM=1
SCALPER\_TREND\_CONFIRM\_MODE=gate
SCALPER\_TREND\_CONFIRM\_TF=1h
SCALPER\_TREND\_CONFIRM\_FAST=50
SCALPER\_TREND\_CONFIRM\_SLOW=200
```

---

\## Exit Momentum Fade (Optional)

Exit when 5m and 15m momentum flips against the position.

\- `EXIT\_MOM\_ENABLED` (default `0`): enable the exit.
\- `EXIT\_MOM\_MIN` (default `0.0015`): momentum threshold (long <= -min, short >= +min).
\- `EXIT\_MOM\_REQUIRE\_BOTH` (default `1`): require both 5m and 15m to flip.

Example:
```env
EXIT\_MOM\_ENABLED=1
EXIT\_MOM\_MIN=0.0015
EXIT\_MOM\_REQUIRE\_BOTH=1
```

---

\## Exit VWAP Cross (Optional)

Exit when price crosses VWAP against the position.

\- `EXIT\_VWAP\_ENABLED` (default `0`): enable the exit.
\- `EXIT\_VWAP\_TF` (default `5m`): timeframe for VWAP.
\- `EXIT\_VWAP\_WINDOW` (default `240`): rolling VWAP window.
\- `EXIT\_VWAP\_REQUIRE\_CROSS` (default `1`): require a cross instead of a simple below/above check.

Example:
```env
EXIT\_VWAP\_ENABLED=1
EXIT\_VWAP\_TF=5m
EXIT\_VWAP\_WINDOW=240
EXIT\_VWAP\_REQUIRE\_CROSS=1
```

---

\## Exit ATR-Scaled Timing (Optional)

Scale time/stagnation exits by entry ATR% so higher-volatility positions get more room and lower-volatility positions are closed faster.

- `EXIT_ATR_SCALE_ENABLED` (default `0`): enable ATR-based scaling in `execution/exit.py`.
- `EXIT_ATR_SCALE_REF_PCT` (default `0.003`): reference ATR% used as scale baseline.
- `EXIT_ATR_SCALE_MIN` (default `0.6`): minimum timing multiplier clamp.
- `EXIT_ATR_SCALE_MAX` (default `1.6`): maximum timing multiplier clamp.
- Per-symbol overrides are supported with `_SYMBOL` suffix, for example:
  - `EXIT_ATR_SCALE_REF_PCT_DOGE`
  - `EXIT_ATR_SCALE_MIN_DOGE`
  - `EXIT_ATR_SCALE_MAX_DOGE`

Telemetry note: when scaling applies, the bot emits `exit.atr_scaled`.

Example:
```env
EXIT_ATR_SCALE_ENABLED=1
EXIT_ATR_SCALE_REF_PCT=0.003
EXIT_ATR_SCALE_MIN=0.6
EXIT_ATR_SCALE_MAX=1.6
```

Per-symbol exit timing overrides are also supported:
- `EXIT_MAX_HOLD_SEC_DOGE`
- `EXIT_STAGNATION_SEC_DOGE`
- `EXIT_STAGNATION_ATR_DOGE`

---

\## Entry Loop Safety (Router Blocks)

Block new entries if the router has recently blocked orders for the same symbol.

\- `ENTRY\_BLOCK\_ON\_ROUTER\_BLOCK` (default `1`)
\- `ENTRY\_BLOCK\_ROUTER\_WINDOW\_SEC` (default `60`)
\- `ENTRY\_BLOCK\_ROUTER\_THRESHOLD` (default `1`)
\- `ENTRY\_BLOCK\_ROUTER\_BACKOFF\_SEC` (default `10`)

Example:
```env
ENTRY\_BLOCK\_ON\_ROUTER\_BLOCK=1
ENTRY\_BLOCK\_ROUTER\_WINDOW\_SEC=60
ENTRY\_BLOCK\_ROUTER\_THRESHOLD=2
ENTRY\_BLOCK\_ROUTER\_BACKOFF\_SEC=15
```

---

\## Entry Loop Safety (Error Throttle)

Throttle entries when too many `entry.blocked` events occur in a short window.

\- `ENTRY\_BLOCK\_ON\_ERRORS` (default `0`)
\- `ENTRY\_BLOCK\_ERRORS\_WINDOW\_SEC` (default `60`)
\- `ENTRY\_BLOCK\_ERRORS\_THRESHOLD` (default `3`)
\- `ENTRY\_BLOCK\_ERRORS\_BACKOFF\_SEC` (default `15`)

Example:
```env
ENTRY\_BLOCK\_ON\_ERRORS=1
ENTRY\_BLOCK\_ERRORS\_WINDOW\_SEC=60
ENTRY\_BLOCK\_ERRORS\_THRESHOLD=3
ENTRY\_BLOCK\_ERRORS\_BACKOFF\_SEC=20
```

---

\## Entry Loop Safety (Partial Fill)

If an entry fills below a minimum ratio, cancel the remainder (limit orders) and back off.

\- `ENTRY\_PARTIAL\_MIN\_FILL\_RATIO` (default `0.5`)
\- `ENTRY\_PARTIAL\_CANCEL` (default `1`)
\- `ENTRY\_PARTIAL\_BACKOFF\_SEC` (default `10`)
\- `ENTRY\_PARTIAL\_ESCALATE\_WINDOW\_SEC` (default `600`)
\- `ENTRY\_PARTIAL\_ESCALATE\_COUNT` (default `3`)
\- `ENTRY\_PARTIAL\_ESCALATE\_BACKOFF\_SEC` (default `120`)
\- `ENTRY\_PARTIAL\_ESCALATE\_TELEM\_CD\_SEC` (default `300`)
\- `ENTRY\_PARTIAL\_CANCEL\_RETRIES` (default `2`)
\- `ENTRY\_PARTIAL\_CANCEL\_DELAY\_SEC` (default `0.25`)
\- `ENTRY\_PARTIAL\_FORCE\_FLATTEN` (default `1`)
\- `ENTRY\_PARTIAL\_FLATTEN\_RETRIES` (default `2`)

Example:
```env
ENTRY\_PARTIAL\_MIN\_FILL\_RATIO=0.6
ENTRY\_PARTIAL\_CANCEL=1
ENTRY\_PARTIAL\_BACKOFF\_SEC=15
ENTRY\_PARTIAL\_FORCE\_FLATTEN=1
```


When `ratio < ENTRY_PARTIAL_MIN_FILL_RATIO` keeps firing, the entry loop records each hit per symbol. After `ENTRY_PARTIAL_ESCALATE_COUNT` hits within `ENTRY_PARTIAL_ESCALATE_WINDOW_SEC` the guard extends the pending block to at least `ENTRY_PARTIAL_ESCALATE_BACKOFF_SEC`, resets the hit counter, and emits `entry.partial_fill_escalation` (throttled via `ENTRY_PARTIAL_ESCALATE_TELEM_CD_SEC`). Include this event in your dashboards/notifications (telemetry dashboard, guard history alerts, etc.) so you can spot when partial fills are straining the execution path before you tweak `ENTRY_PARTIAL_MIN_FILL_RATIO` or leverage.

The partial-fill resolver now emits `entry.partial_fill_state` with explicit outcomes:
- `partial_forced_flatten`: below-threshold fill was detected and a reduce-only market flatten succeeded.
- `partial_stuck`: below-threshold fill remained unresolved (cancel/flatten could not safely resolve it).

The adaptive guard already listens for `entry.partial_fill_escalation` and (by default) raises `ENTRY_MIN_CONFIDENCE` by `ADAPTIVE_GUARD_PARTIAL_ESCALATE_DELTA` for `ADAPTIVE_GUARD_PARTIAL_ESCALATE_DURATION_SEC` seconds. Use those knobs to tune how aggressively the guard pauses entries/leverage when repeated partial fills keep triggering the escalation event.
### Router notional guard (optional)

Set `ROUTER_MAX_NOTIONAL_USDT` (or the legacy alias `ROUTER_NOTIONAL_CAP`) to a per-order cap in USD. The router blocks any new entry whose notional would exceed your configured limit and emits `order.blocked` telemetry (`why=router_notional_cap`), giving you an extra sizing gate without touching the FIRST_LIVE_SAFE guard.

---

\## Data Quality & Staleness (Entry Safety)

Unified staleness checks + data quality scoring for entries.

\- `ENTRY\_DATA\_MAX\_STALE\_SEC` (default `180`): reject entries if data is too old.
\- `ENTRY\_DATA\_QUALITY\_MIN` (default `60`): block entries if quality score drops below this.
\- `ENTRY\_DATA\_QUALITY\_TF` (default `1m`)
\- `ENTRY\_DATA\_QUALITY\_WINDOW` (default `120`)
\- `ENTRY\_DATA\_QUALITY\_EMIT\_SEC` (default `60`): telemetry emit interval.
\- `ENTRY\_DATA\_QUALITY\_HISTORY\_MAX` (default `50`): per-symbol history length.
\- `ENTRY\_DATA\_QUALITY\_ROLL\_SEC` (default `900`): rolling window (seconds) for telemetry roll score.
\- `ENTRY\_DATA\_QUALITY\_ROLL\_MIN` (default same as `ENTRY\_DATA\_QUALITY\_MIN`): rolling score threshold before the guard blocks.
\- `ENTRY\_DATA\_QUALITY\_KILL\_SEC` (default `120`): seconds of sustained low roll score before entries are blocked.
\- Telemetry note: hits emit `data.quality.roll_alert`; use `tools/telemetry_error_classes.py --events data.quality.roll_alert` to monitor.

Example:
```env
ENTRY\_DATA\_MAX\_STALE\_SEC=180
ENTRY\_DATA\_QUALITY\_MIN=70
ENTRY\_DATA\_QUALITY\_TF=1m
ENTRY\_DATA\_QUALITY\_WINDOW=120
ENTRY\_DATA\_QUALITY\_EMIT\_SEC=120
ENTRY\_DATA\_QUALITY\_HISTORY\_MAX=50
ENTRY\_DATA\_QUALITY\_ROLL\_SEC=900
ENTRY\_DATA\_QUALITY\_ROLL\_MIN=70
ENTRY\_DATA\_QUALITY\_KILL\_SEC=120
```

\### Entry sizing scaling (adaptive + confidence)

The entry loop now supports three independent sizing modifiers before order placement:

1. Adaptive guard notional scale (`ADAPTIVE_GUARD_NOTIONAL_SCALE`), emitted as `entry.notional_scaled`.
2. Adaptive guard fixed-qty scale (`ADAPTIVE_GUARD_QTY_SCALE`), emitted as `entry.qty_scaled`.
3. Confidence-based sizing scale (`ENTRY_CONF_SCALE_*`), emitted as `entry.notional_scaled` with `reason=confidence`.

Confidence scaling knobs:
- `ENTRY_CONF_SCALE_ENABLED` (default `0`)
- `ENTRY_CONF_SCALE_MIN_CONF` (default `0.35`)
- `ENTRY_CONF_SCALE_MAX_CONF` (default `0.75`)
- `ENTRY_CONF_SCALE_MIN` (default `0.5`)
- `ENTRY_CONF_SCALE_MAX` (default `1.0`)

Example:
```env
ADAPTIVE_GUARD_QTY_SCALE=1
ENTRY_CONF_SCALE_ENABLED=1
ENTRY_CONF_SCALE_MIN_CONF=0.40
ENTRY_CONF_SCALE_MAX_CONF=0.75
ENTRY_CONF_SCALE_MIN=0.60
ENTRY_CONF_SCALE_MAX=1.00
```

Per-symbol sizing overrides are supported:
- `FIXED_NOTIONAL_USDT_DOGE`
- `FIXED_QTY_DOGE`

### Signal data health report

Read the telemetry JSONL (or `TELEMETRY_PATH`) to summarize per-symbol `data.quality` scores,
`data.stale` counts, and missing-data hits:

```bash
python eclipse_scalper/tools/signal_data_health.py --path logs/telemetry.jsonl --since-min 60
```

This prints the worst average/full-score symbols, staleness counts with max age, and any
`data.ticker_missing`/`data.ohlcv_missing` rows in the window so you can pause the entry loop or
refresh tickers before the next live run.

### Signal -> exit health report

Cross-check the exit.* telemetry with the signal metadata the bot now records (`entry_confidence`, `entry_signal_age_sec`, guard/exposure context) so you can spot low-confidence handoffs or guard-forced closes before the dashboard triggers a pause.

```bash
python eclipse_scalper/tools/signal_exit_health.py --path logs/telemetry.jsonl --since-min 60
```

The helper ranks exit reasons, the worst-hit symbols, flags low-confidence/missing exits, and lists telemetry guard exits (with exposures + timestamps) so tuning can happen before the next live session.

Run the companion feedback helper to append an `exit.signal_issue` record whenever the low-confidence exit ratio crosses your tolerance. Use `--emit-event` plus `--issue-ratio`/`--issue-count` to tune the threshold that informs the telemetry classifier and guards. Add `--recovery-min-confidence`/`--recovery-duration` so the helper also writes `logs/telemetry_recovery_state.json`, temporarily raising the entry minimum confidence until the low-confidence run cools off.

The entry loop now monitors `logs/telemetry_recovery_state.json` (or `TELEMETRY_RECOVERY_STATE`) and raises `MIN_CONFIDENCE` whenever the state is active, so the recovery autopilot stays in force until the file expires.

Additionally, `entry_loop` now scans the telemetry log for new `exit.signal_issue` events before every poll and logs a summary + emits `entry.blocked` when feedback triggers; disable this behavior with `ENTRY_SIGNAL_FEEDBACK_ENABLED=0` if the extra logging isnâ€™t wanted.

You can override CORE_HEALTH_EQUITY (default 100000) so the dashboard reports exposure as a percent of your preferred capital base, and it now surfaces the anomaly pause window/payload directly when the guard is active.

### Core health (risk/sizing/signal) summary

`tools/core_health.py` pulls from the same telemetry JSONL and surfaces:

- Top exposures (order notional) plus the average entry confidence and net order direction per symbol.
- Risk and kill-switch counts so you can spot repeated safety or sizing triggers over the window.
- Blocked-reason counts plus exit event/code summaries to validate how the signal engine is behaving.

```bash
python eclipse_scalper/tools/core_health.py --path logs/telemetry.jsonl --since-min 60 --limit 2000
```

Use the report whenever you want a single snapshot covering risk, sizing, and signal activity before touching the live loop.

---

\## Running the Bot (Dry Run)



Start in dry-run mode:



```bash

python main.py

```



Watch the logs carefully.

No orders should hit the exchange when `SCALPER\_DRY\_RUN=1`.



---



\## Risk Controls Profiles (Recommended)

Use one of the provided PowerShell profiles:

```text
run-bot-ps2.ps1   # safer live / test profile (low leverage, low notional, audit on)
run-bot.ps1       # tuned live profile (higher leverage, tighter guards)
```

Suggested progression:
- Start with `run-bot-ps2.ps1` and `SCALPER_DRY_RUN=1`.
- Move to live only after logs and telemetry look clean.
- Keep `FIRST_LIVE_SAFE=1` + allowlist until you trust the behavior.

Key risk knobs (quick reference):
- `MAX_DAILY_LOSS_PCT`, `MAX_DRAWDOWN_PCT` â†’ hard daily limits.
- `KILL_DAILY_HALT_SEC`, `KILL_DRAWDOWN_HALT_SEC`, `KILL_DAILY_HALT_UNTIL_UTC` â†’ auto-pause window.
- `FIXED_NOTIONAL_USDT`, `LEVERAGE`, `MARGIN_MODE` â†’ exposure sizing.
- `CORR_GROUP_*` â†’ correlation caps (group positions + notional).
- `SCALPER_*` guards â†’ data staleness, session, cooldown.

Leverage overrides:
- `LEVERAGE_BY_SYMBOL` (e.g., `BTCUSDT=10,ETHUSDT=5`) or `LEVERAGE_BTCUSDT=10` for a single symbol.
- `LEVERAGE_BY_GROUP` uses `CORR_GROUPS` names as caps (lowest wins if symbol is in multiple groups).
- `LEVERAGE_MIN` / `LEVERAGE_MAX` clamp final leverage.
- `LEVERAGE_GROUP_DYNAMIC=1` enables dynamic leverage scaling by open positions in the same group.
  - `LEVERAGE_GROUP_SCALE` (default `0.7`) is applied per open symbol in the group.
  - `LEVERAGE_GROUP_SCALE_MIN` (default `1`) is the floor after scaling.
  - `LEVERAGE_GROUP_EXCLUDE_SELF` (default `1`) excludes the current symbol if itâ€™s already open.
  - `LEVERAGE_GROUP_EXPOSURE=1` switches to exposure-weighted scaling (group notional / equity).
    - `LEVERAGE_GROUP_EXPOSURE_REF_PCT` (default `0.10`) is the exposure step size.

---

\## Risk Checklist (Quick)

Before any live session:
- Confirm `SCALPER_DRY_RUN=0` is intentional.
- Verify `FIRST_LIVE_SAFE=1` and `FIRST_LIVE_SYMBOLS` allowlist is set.
- Confirm `FIXED_NOTIONAL_USDT` and `LEVERAGE` are sized for account risk.
- Set `MAX_DAILY_LOSS_PCT` and `MAX_DRAWDOWN_PCT`.
- Set `KILL_DAILY_HALT_SEC` and `KILL_DRAWDOWN_HALT_SEC` (auto-pause).
- Set correlation caps (`CORR_GROUP_*`) for your symbol set.
- Ensure telemetry and audit logs are being written.

Terminal helper:
```bash
python tools/risk_checklist.py
```
With environment snapshot:
```bash
python tools/risk_checklist.py --env
```
With run script values (side-by-side):
```bash
python tools/risk_checklist.py --scripts
```
Combine both:
```bash
python tools/risk_checklist.py --env --scripts
```

---

\## Correlation Groups (Caps)

Use correlation groups to cap simultaneous exposure across related symbols.

Example groups + limits:
```env
# define groups
CORR_GROUPS=MEME:DOGEUSDT,SHIBUSDT,PEPEUSDT;MAJOR:BTCUSDT,ETHUSDT

# global caps (all groups)
CORR_GROUP_MAX_POSITIONS=1
CORR_GROUP_MAX_NOTIONAL_USDT=50

# per-group overrides
CORR_GROUP_LIMITS=MEME=1,MAJOR=2
CORR_GROUP_NOTIONAL=MEME=25,MAJOR=100
```

Validate your settings:
```bash
python tools/corr_group_check.py
```

Dynamic notional scaling by group exposure is also available (in addition to hard caps):

- `CORR_GROUP_EXPOSURE_SCALE_ENABLED` (default `0`)
- `CORR_GROUP_EXPOSURE_SCALE` (default `0.7`)
- `CORR_GROUP_EXPOSURE_SCALE_MIN` (default `0.25`)
- `CORR_GROUP_EXPOSURE_REF_NOTIONAL` (default `0`, falls back to group/global notional cap)
- Optional per-group overrides:
  - `CORR_GROUP_EXPOSURE_SCALE_BY_GROUP`
  - `CORR_GROUP_EXPOSURE_SCALE_MIN_BY_GROUP`
  - `CORR_GROUP_EXPOSURE_REF_NOTIONAL_BY_GROUP`

When active, the entry loop emits `entry.notional_scaled` with `reason=corr_group_exposure ...`.

Example output:
```text
Correlation Groups
- MEME: DOGEUSDT, SHIBUSDT, PEPEUSDT
- MAJOR: BTCUSDT, ETHUSDT

Global caps
- CORR_GROUP_MAX_POSITIONS=1
- CORR_GROUP_MAX_NOTIONAL_USDT=25.0

Per-group overrides
- MAJOR: max_positions=2, max_notional=100.0
- MEME: max_positions=1, max_notional=25.0
```

---

\## Telemetry Report (Entry Blocks)

Quickly summarize why entries are being blocked:

```bash
python tools/telemetry_report.py
```

Uses `TELEMETRY_PATH` if set; otherwise reads `logs/telemetry.jsonl`.

Common options:

```bash
# filter by symbol
python tools/telemetry_report.py --symbol DOGEUSDT

# summarize a different event
python tools/telemetry_report.py --event order.retry

# last 60 minutes, top 10, alphabetical
python tools/telemetry_report.py --since 60 --top 10 --sort alpha

# hide low-frequency reasons and search text
python tools/telemetry_report.py --min 5 --reason-contains cooldown

# summary only + export CSV
python tools/telemetry_report.py --summary-only --csv logs/telemetry_report.csv
```

Quick dashboard (JSONL summary):
```bash
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl --limit 1000
```
Add `--codes-per-symbol` (and optionally `--codes-top`) to show top codes per symbol alongside the dashboard:
```bash
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl --codes-per-symbol --codes-top 4
```
Point the helper at `logs/telemetry.jsonl` (or whichever file `TELEMETRY_PATH` points to) to ensure the combined snapshot (top events, codes, recent entries, and symbol-code breakdown) stays readable, even when the log grows.
When exit.* events exist, the dashboard also prints an exit-events summary (counts + codes) so you can spot new exit reasons right in the same snapshot.

Use `--guard-events` when you want the dashboard to highlight `entry.partial_fill_escalation` + `order.create.retry_alert` counts (plus retry reasons) so you know when the guard is already throttling entries. Add `--guard-history` (plus `--guard-history-path` if you store it elsewhere) to dump the latest rows from `logs/telemetry_guard_history.csv` so you can visually align the guard history spikes with the telemetry alarms.

```bash
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl --guard-events --guard-history
```

## Scheduled telemetry snapshots

Telemetry smoke workflow status: [![Telemetry Smoke Assertions](https://github.com/phoenixsenses/eclipse_scalper/actions/workflows/telemetry-smoke.yml/badge.svg)](https://github.com/phoenixsenses/eclipse_scalper/actions/workflows/telemetry-smoke.yml)

The workflow `.github/workflows/telemetry-dashboard.yml` runs the dashboard helper every six hours (plus whenever you trigger `workflow_dispatch`). It checks out the repo, sets up Python 3.12, installs the notifier runtime dependency (`python-telegram-bot==22.5`), and executes:

For operational procedures (manual trigger/watch, RED_LOCK smoke, reset smoke, and incident triage), see `docs/telemetry_runbook.md`.

```bash
python eclipse_scalper/tools/telemetry_dashboard_notify.py --path logs/telemetry.jsonl --codes-per-symbol --codes-top 4
```

Keep `logs/telemetry.jsonl` (or the file defined by `TELEMETRY_PATH`) up to date so the scheduled job can read real telemetry. You can also edit the workflowâ€™s cron if you want a different cadence.

### Workflow smoke commands (escalation + reset)

Use these manual dispatch commands to validate notifier streak behavior without waiting for live telemetry conditions:

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_red_lock_event=true \
  -f simulate_red_lock_seed_streak=1
```

This injects a synthetic `RED_LOCK` event and, with the default threshold `RECOVERY_RED_LOCK_CRITICAL_STREAK=2`, should transition notifier state to `level=critical`.

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_recovery_stage_override=POST_RED_WARMUP \
  -f simulate_red_lock_seed_streak=2
```

This injects a non-RED recovery stage and should reset `recovery_red_lock_streak` to `0` (de-escalation path). Confirm values in the workflow's **Inspect notifier state** step.

### Notifications

The scheduled job now runs `python eclipse_scalper/tools/telemetry_dashboard_notify.py --path logs/telemetry.jsonl --codes-per-symbol --codes-top 4`. This wrapper prints the dashboard snapshot to the workflow log and posts the same summary to Telegram when `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are configured as [repository secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets). The notifier reuses `eclipse_scalper.notifications.telegram.Notifier`, so you get the same `speak` behavior already used elsewhere.

Leave the secrets empty if you only need GitHub Actions visibility; the job still succeeds and simply skips the Telegram send. If you do set those secrets, keep an eye on Telegram so you know when the latest snapshot hits critical topics like `exit.*` events or repeated `ERR_*` codes.

The workflow also runs `python eclipse_scalper/tools/signal_data_health.py ...` and `python eclipse_scalper/tools/core_health.py ...`, captures their stdout to `logs/signal_data_health.txt` / `logs/core_health.txt`, and uploads those files along with `logs/telemetry_health.html` as the `telemetry-snapshots` artifact for quick download.

The scheduled job now also runs `python eclipse_scalper/tools/telemetry_anomaly.py --path logs/telemetry.jsonl --since-min 60 --output logs/telemetry_anomaly.txt` and uploads the report. When the detector sees an exposure spike (> default 50%), confidence drop (> default 15%), risk event surge (>3), or new exit code, it notifies the Telegram channel using the same secrets so you get a proactive alert alongside the artifact.
The Telegram alert now includes the anomaly pause expiration when one was triggered, so you immediately know not only that trading paused but also when the guard will re-open (and why).
The workflow also runs `telemetry_dashboard_page.py` to render those text reports as a single HTML dashboard and runs `telemetry_alert_summary.py` to summarize all artifacts; the summary script pings Telegram whenever anomalies were found so the artifact download + alert channel stay in sync.

Add the signal/exit telemetry notifier step described above, then immediately generate the recovery dashboard:

```bash
python eclipse_scalper/tools/telemetry_recovery_dashboard.py \
  --path logs/telemetry.jsonl \
  --state logs/telemetry_recovery_state.json \
  --since-min 60
```

This writes `logs/telemetry_recovery_dashboard.html` plus `logs/telemetry_recovery_report.txt`, giving you an HTML snapshot of the recovery override, recent `exit.signal_issue` rows, and telemetry guard events; upload those artifacts with the dashboard snapshot so the team can inspect them without rerunning the scripts.

Replay the guard timeline occasionally (e.g., after the notifier step) with:

```bash
python eclipse_scalper/tools/telemetry_guard_timeline.py \
  --path logs/telemetry.jsonl \
  --state logs/telemetry_recovery_state.json \
  --since-min 120
```

The helper saves `logs/telemetry_guard_timeline.html` + `logs/telemetry_guard_timeline.txt` and highlights the most recent entry-blocked / exit feedback / recovery override hits so you can visually scan when guards have been active. Include these files in the telemetry artifact bundle for quick reference.

### Confidence drift detection

Add drift detection to the telemetry pipeline to catch sudden shifts in entry confidences:

```bash
python eclipse_scalper/tools/telemetry_drift_detection.py \
  --path logs/telemetry.jsonl \
  --since-min 60 \
  --min-count 6 \
  --zscore 2.5 \
  --emit-event \
  --event-path logs/telemetry_drift.jsonl
```

The tool writes `logs/telemetry_drift_summary.txt` and, when a symbolâ€™s current-half mean deviates from the baseline half by more than `zscore Ã— stdev`, it emits `telemetry.confidence_drift` events that you can feed into your dashboards/alerts; drop `--emit-event` if you only need the report.

The scheduled job now runs `python eclipse_scalper/tools/telemetry_signal_exit_notify.py --path logs/telemetry.jsonl --since-min 60 --emit-event --recovery-min-confidence 0.7 --recovery-duration 600 --issue-ratio 0.2 --issue-count 2`. This wrapper calls both `signal_exit_health.py` and `signal_exit_feedback.py`, stores the combined text plus the low-confidence ratio/count/symbol context in `logs/signal_exit_notify.txt`, and posts the same summary to Telegram whenever low-confidence exits or telemetry guards trigger.

Add a workflow step (after the dashboard/alert summary step) such as:

```yaml
    - name: Signal/exit telemetry report
      run: |
        python eclipse_scalper/tools/telemetry_signal_exit_notify.py \
          --path logs/telemetry.jsonl \
          --since-min 60 \
          --emit-event \
          --recovery-min-confidence 0.7 \
          --recovery-duration 600 \
          --issue-ratio 0.2 \
          --issue-count 2
```

The notifier reuses `TELEGRAM_TOKEN` / `TELEGRAM_CHAT_ID` so the same channel receives the health + feedback story. Remove `--emit-event` if you only care about the summary text. `telemetry_alert_summary.py` now reads the same `logs/signal_exit_notify.txt` report (`--signal-exit`) so the Telegram alert highlights the low-confidence ratio, guard hits, and low-confidence symbols that caused the recovery override.

The detector also writes `logs/telemetry_anomaly_state.json`, and `execution/entry_loop.py` consults `execution/anomaly_guard.py` before every poll. When `pause_until` is still in the future, entries are blocked with `anomaly_pause` (plus the reasons show in the report) so the guard automatically mitigates the spike instead of letting the bot continue.

The same anomaly pipeline now feeds `execution/exit.py` via `logs/telemetry_anomaly_actions.json`. When exposures climb above `EXIT_TELEMETRY_HIGH_EXPOSURE_USDT`, the exit loop temporarily lowers its cooldown (`EXIT_TELEMETRY_COOLDOWN_MULT`), logs an `exit.telemetry_guard` event, and once positions surpass `EXIT_TELEMETRY_FORCE_HOLD_SEC` it forces a market exit with telemetry context (signals + exposures) so you can see why the guard engaged in the dashboard artifact.

The adaptive guard now also listens for `entry.blocked` events where `reason=partial_fill` and for `order.create.retry_alert` events emitted when the router exhausts retries. Both raise `ENTRY_MIN_CONFIDENCE` via `execution/adaptive_guard.py` (delta/duration controlled by `ADAPTIVE_GUARD_PARTIAL_*` and `ADAPTIVE_GUARD_RETRY_*`), so partial fills or retry storms throttle new entries until the spike cools. The guard timeline/history artifacts note the ratio/tries so you can trace which event triggered the override.

### Adaptive entry guard

`execution/adaptive_guard.py` now refreshes its state before each scan, reads `telemetry.confidence_drift`, `exit.signal_issue`, and `exit.telemetry_guard`, and raises `ENTRY_MIN_CONFIDENCE` by a small delta whenever one of those telemetry shifts fires. The guard reads both `logs/telemetry.jsonl` and `logs/telemetry_drift.jsonl` (`TELEMETRY_DRIFT_PATH`), stores per-symbol overrides in `logs/telemetry_adaptive_guard.json` (`ADAPTIVE_GUARD_STATE`), and keeps overrides active for the durations configured in `ADAPTIVE_GUARD_DURATION_SEC`, `ADAPTIVE_GUARD_DRIFT_DURATION_SEC`, `ADAPTIVE_GUARD_EXIT_DURATION_SEC`, and `ADAPTIVE_GUARD_TELEMETRY_DURATION_SEC`. With `SCALPER_SIGNAL_DIAG=1` you will see the loop log the guard reason and the temporary `min_conf` increase before skipping new entries.

Use `tools/telemetry_guard_timeline.py` to generate an HTML/text narrative of the most recent guard activity combined with the recovery override (the script already understands the logs + guard state so the summary matches what the guard enforces):

```bash
python eclipse_scalper/tools/telemetry_guard_timeline.py \
  --path logs/telemetry.jsonl \
  --state logs/telemetry_adaptive_guard.json \
  --since-min 180
```

The HTML/text outputs make great additions to the dashboard/job that already publishes `logs/core_health.txt` and `logs/signal_data_health.txt`. `run-bot-ps2.ps1` now mirrors the same telemetry + guard defaults so the scheduled helpers, dashboards, and Telegram alerts all look at the same artifacts.

The telemetry workflow has also been expanded so every scheduled snapshot:

1. Runs `tools/telemetry_drift_detection.py --emit-event --event-path logs/telemetry_drift.jsonl` so the adaptive guard picks up `telemetry.confidence_drift` signals within that run window.
2. Runs `tools/telemetry_signal_exit_notify.py --emit-event --summary logs/signal_exit_notify.txt --issue-ratio 0.25 --issue-count 2 --recovery-min-confidence 0.7 --recovery-duration 600` so low-confidence exits/feedback emit `exit.signal_issue` events, write a summary, and post the combined status to Telegram.
3. Runs `tools/telemetry_guard_timeline.py --state logs/telemetry_adaptive_guard.json --since-min 180 --notify`, which rebuilds the HTML/text timeline, uploads it, and pushes the timeline summary into Telegram so the guard story closes the loop.

Because these steps append events and update `logs/telemetry_anomaly_actions.json`, the guard can automatically throttle new entries or raise `min_confidence` when anomalies or exit-quality issues spike. The workflow now also uploads `logs/telemetry_drift_summary.txt`, `logs/signal_exit_notify.txt`, and the generated timeline artifacts so you can inspect them any time.

The same workflow also runs `tools/telemetry_guard_history.py` to append a row with the drift count, low-confidence ratio, guard hits, override state, and anomaly notes into `logs/telemetry_guard_history.csv` and render the latest window as `logs/telemetry_guard_history.html`. Those artifacts show how the guard and drift signals have evolved across each scheduled snapshot, making long-term comparisons and visual postmortems faster.

Immediately after building the guard history table, the job now executes `tools/telemetry_dashboard.py --path logs/telemetry.jsonl --guard-events --guard-history` so the dashboard log notes the latest guard telemetry counts and history rows alongside the core summary; that output is also uploaded as part of the snapshot artifacts so you can align the partial-fill/retry alerts with the guard history before reviewing the Telegram notifications.

Every scheduled snapshot now runs `tools/telemetry_guard_history_alerts.py --path logs/telemetry_guard_history.csv --window 8 --threshold 3 --hit-rate 0.25 --notify`. The helper checks the partial/retry hits rate plus the raw count within the recent rows, prints the summary, and only posts to Telegram when either the hit count or hit ratio is too highâ€”this keeps you alerted when spikes persist even if the raw count is modest.

The generated `logs/telemetry_guard_history.html` now shows a tiny sparkline above the history table that tracks the `partial_retry_hits` column so you can scan the hit trend visually before diving into the rows or alerts.

The workflow also runs `tools/defense_system.py --history logs/telemetry_guard_history.csv --rows 16 --actions logs/telemetry_anomaly_actions.json --timeline logs/telemetry_guard_timeline.txt --notify`. That helper prints a defense severity score, outlines levered tuning suggestions (confidence, notional, leverage), and notifies Telegram when severity â‰¥ 2 so you can adjust your strategy before the guard has to ratchet `ENTRY_MIN_CONFIDENCE` again.

### Alert classification

A subsequent workflow step runs `python eclipse_scalper/tools/telemetry_classifier.py --path logs/telemetry.jsonl --since-min 60`. This helper keeps a small state file at `logs/telemetry_classifier_state.json`, compares the current window against the previous snapshot, and flags:

- **Exit spikes** when `exit.*` event totals exceed `--exit-threshold` (default `5`) and are higher than the last run.
- **New telemetry codes** by remembering every `data.code` youâ€™ve seen and reporting any novel entries.
- **Confidence drops** by averaging `confidence`/`conf` fields in entry-related telemetry and alerting when the score falls by more than `--confidence-drop-pct` (default `15%`) relative to the prior average.

It reuses the same Telegram secrets so alerts appear in the same channel as the dashboard snapshot, but it only posts when at least one trigger fires. Run the classifier locally with `--no-notify` if you want to inspect the summary without sending Telegram messages, or tweak the thresholds to suit your guardrails.

You can now drive adaptive sensitivity with `--adaptive-events` (default `entry.blocked,data.quality.roll_alert`), `--adaptive-step`, and `--adaptive-min`. The script counts those events in the current window and reduces the exit/confidence thresholds via a multiplier (`max(adaptive-min, 1 - total*adaptive-step)`), making alerts fire sooner while your data quality or entry path is acting up.

### Confidence health panel

Each classifier run appends a row to `logs/telemetry_health.csv` (timestamp, exit totals, confidence average, multiplier, thresholds, etc.). To visualize that history run:

```bash
python eclipse_scalper/tools/telemetry_health.py --health-path logs/telemetry_health.csv --output logs/telemetry_health.html
```

The command produces `logs/telemetry_health.html`, an Altair chart that plots the adaptive multiplier vs. average confidence and stacks exit counts, highlighting runs that triggered alerts. Refresh the classifier data before you regenerate the panel so the HTML reflects the latest adaptive thresholds.

Top codes per symbol:
```bash
python tools/telemetry_codes_by_symbol.py --path logs/telemetry.jsonl --limit 5000 --top 5
```

Error class dashboard (with optional thresholds):
```bash
python tools/telemetry_error_classes.py --path logs/telemetry.jsonl --since-min 60 --thresholds network=5,margin=1
```

Roll alert summary:
```bash
python tools/telemetry_roll_alerts.py --path logs/telemetry.jsonl --since-min 120
```

Latency summary:
```bash
python tools/telemetry_latency_summary.py --path logs/telemetry.jsonl --since-min 60
```

Telemetry threshold watcher (per-event counters):
```bash
python tools/telemetry_threshold_alerts.py --path logs/telemetry.jsonl --thresholds entry.blocked=3,order.create.retry_alert=2
```
Exit-special threshold helper (uses the exit preset which also monitors `exit_loop:error` and can send Telegram alerts when configured):
```bash
python tools/telemetry_exit_thresholds.py --path logs/telemetry.jsonl
```
Add `--since-min` or other `telemetry_threshold_alerts` args to the helper, e.g. `python tools/telemetry_exit_thresholds.py --since-min 60 --thresholds exit.blocked=2`.
Set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to enable auto-notification when exit alerts trigger; the helper wraps the thresholds scan and posts the summary if anything fires.

### Router retry tuning

`ROUTER_RETRY_POLICY` lets you sway the router's retry loop for known transient classes (network blips, exchange busy, timestamp skew). Each entry in the comma-separated list has the form `reason=max:delay:max_delay:extra` and overrides the base `ROUTER_RETRY_BASE_SEC`/`ROUTER_RETRY_MAX_DELAY_SEC` for that reason before stopping. The defaults keep `network=8:0.35:3:2`, `exchange_busy=7:0.45:4:2`, and `timestamp=4:0.3:2:1`, and the router still emits `order.create.retry_alert` (with `tries`, `reason`, and `code`) whenever the throttle limit passes so your dashboards or adaptive guard know when retries are still in flight.

### Reliability model (execution layer)

Execution reliability is defined as bounded wrongness under partial information, not perfect correctness under ideal conditions.

- Invariants: `docs/execution_invariants.md`
- Belief/debt model: `docs/belief_state_model.md`
- Failure-mode analysis: `docs/execution_fmea.md`
- Observability lifecycle contract: `docs/observability_contract.md`
- Router + reconcile hardening summary: `docs/execution_reliability.md`
- Deep system walkthrough: `docs/execution_system_summary.md`
- Strategy architecture summary: `docs/strategies_summary.md`
- 90-day reliability roadmap: `docs/execution_roadmap_90d.md`

Use `docs/execution_system_summary.md` as the primary onboarding reference for execution incidents and hardening work.
It explains the architecture as a closed reliability loop:
exchange evidence -> local belief/debt -> belief controller guard knobs ->
entry-only risk clamping -> reconcile-driven recovery (with exits always safe).

Use `docs/strategies_summary.md` as the strategy-level onboarding companion.
It explains how alpha decisions, sizing/leverage controls, correlation caps, and telemetry-driven reliability controls work together.

### Reliability Workflow Templates

- Milestone planning template:
  - `.github/ISSUE_TEMPLATE/reliability_milestone.md`
- Incident review template:
  - `.github/ISSUE_TEMPLATE/reliability_incident_review.md`

The new chaos scenario test suite (`tools/test_execution_chaos_scenarios.py`) is wired into CI so timeout/duplicate/partial-fill and reconcile contradiction paths are checked on each push/PR.

---

## Unit Tests

Run all unit-style tests in `tools/`:

```bash
python tools/run_unit_tests.py
```

Run the new exit-telemetry helper test directly when you touch `execution.exit`:

```bash
python tools/test_exit_telemetry_helper_unit.py
```

Recent targeted tests added for telemetry/sizing/exit behavior:

```bash
python tools/test_adaptive_guard_unit.py
python tools/test_entry_qty_scale_unit.py
python tools/test_entry_conf_scale_unit.py
python tools/test_entry_symbol_sizing_unit.py
python tools/test_corr_group_exposure_scale_unit.py
python tools/test_exit_atr_scale_unit.py
python tools/test_exit_symbol_overrides_unit.py
python tools/test_exit_quality_dashboard_unit.py
python tools/test_position_closed_unit.py
```
Run a single test:

```bash
python tools/test_diagnostics_unit.py
```

---

## Strategy Audit (Signals & Near-Misses)

Enable CSV audit of signal decisions:

```env
SCALPER_AUDIT=1
SCALPER_AUDIT_COOLDOWN_SEC=5
```

Report summary:

```bash
python tools/strategy_audit_report.py
```

Output file: `logs/strategy_audit.csv` (override with `SCALPER_AUDIT_PATH`).

Common options:

```bash
# custom path
python tools/strategy_audit_report.py --path logs/strategy_audit.csv

# summary only
python tools/strategy_audit_report.py --summary-only
```

---

## Safety Philosophy



Eclipse Scalper is designed around layered defense:



\* Signal confidence thresholds

\* Position sizing limits

\* Guardian loops

\* Kill switches

\* Dry-run first, always



If something looks wrong â€” \*\*it probably is\*\*.

Stop the bot, inspect logs, adjust, retry.



---



\## Development Status



\* âœ… Core execution engine

\* âœ… Signal pipeline

\* âœ… Order router

\* âœ… Risk kill-switches

\* âš ï¸ Strategy tuning ongoing

\* âš ï¸ Live trading requires caution



This project is under \*\*active development\*\*.



---



\## License



Private / personal use for now.

License to be defined later.



---



\## Final Note



Treat this bot like a sharp instrument.



Used carefully, it teaches you a lot.

Used carelessly, it teaches you faster.



Proceed deliberately.


