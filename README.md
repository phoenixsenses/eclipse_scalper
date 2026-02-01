



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

\- and gradual escalation from simulation → micro → live



This repository contains the \*\*core engine\*\*, not exchange secrets or runtime state.



---



\## ⚠️ Disclaimer (Read This First)



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

│

├── bot/                # Core runner and orchestration

├── execution/          # Entry, exit, order routing, guardian loops

├── strategies/         # Signal logic (Eclipse Scalper strategy)

├── risk/               # Kill-switches and safety logic

├── exchanges/          # Exchange adapters (Binance)

├── notifications/      # Telegram, alerts

├── config/             # Static configuration helpers

├── utils/              # Logging, helpers

├── tools/              # Smoke tests \& diagnostics

│

├── main.py             # Main entry point

├── guardian.py         # Global safety guardian

├── signal\_check.py     # Signal diagnostics

├── settings.py         # Runtime settings

└── requirements.txt    # Python dependencies



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
We’ve now registered a real `coinbase` adapter as well. Point `EXCHANGE_ADAPTER=coinbase` when you want to run through that venue (ensure `COINBASE_API_KEY`, `COINBASE_API_SECRET`, and `COINBASE_API_PASSPHRASE` are set in `.env`).

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



\* `SCALPER\_DRY\_RUN=1` → no real orders

\* `SCALPER\_DRY\_RUN=0` → live trading (dangerous)
\* `ENTRY\_LOOP\_MODE=full|basic` → choose entry loop when using `execution/bootstrap.py` (default: full if available)



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
  positive volume delta (so the multiplier never raises confidence by more than ~40–50%).
\- `SCALPER\_VOL\_RATIO\_MAX\_DROP` (default `0.2` micro / `0.25` production): upper bound on how far
  the multiplier can push confidence downward (keeps the factor above ~0.6–0.8 even when volumes collapse).

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

Example:
```env
ENTRY\_PARTIAL\_MIN\_FILL\_RATIO=0.6
ENTRY\_PARTIAL\_CANCEL=1
ENTRY\_PARTIAL\_BACKOFF\_SEC=15
```

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

\### Signal data health report

Read the telemetry JSONL (or `TELEMETRY_PATH`) to summarize per-symbol `data.quality` scores,
`data.stale` counts, and missing-data hits:

```bash
python eclipse_scalper/tools/signal_data_health.py --path logs/telemetry.jsonl --since-min 60
```

This prints the worst average/full-score symbols, staleness counts with max age, and any
`data.ticker_missing`/`data.ohlcv_missing` rows in the window so you can pause the entry loop or
refresh tickers before the next live run.


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
- `MAX_DAILY_LOSS_PCT`, `MAX_DRAWDOWN_PCT` → hard daily limits.
- `KILL_DAILY_HALT_SEC`, `KILL_DRAWDOWN_HALT_SEC`, `KILL_DAILY_HALT_UNTIL_UTC` → auto-pause window.
- `FIXED_NOTIONAL_USDT`, `LEVERAGE`, `MARGIN_MODE` → exposure sizing.
- `CORR_GROUP_*` → correlation caps (group positions + notional).
- `SCALPER_*` guards → data staleness, session, cooldown.

Leverage overrides:
- `LEVERAGE_BY_SYMBOL` (e.g., `BTCUSDT=10,ETHUSDT=5`) or `LEVERAGE_BTCUSDT=10` for a single symbol.
- `LEVERAGE_BY_GROUP` uses `CORR_GROUPS` names as caps (lowest wins if symbol is in multiple groups).
- `LEVERAGE_MIN` / `LEVERAGE_MAX` clamp final leverage.
- `LEVERAGE_GROUP_DYNAMIC=1` enables dynamic leverage scaling by open positions in the same group.
  - `LEVERAGE_GROUP_SCALE` (default `0.7`) is applied per open symbol in the group.
  - `LEVERAGE_GROUP_SCALE_MIN` (default `1`) is the floor after scaling.
  - `LEVERAGE_GROUP_EXCLUDE_SELF` (default `1`) excludes the current symbol if it’s already open.
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

## Scheduled telemetry snapshots

The workflow `.github/workflows/telemetry-dashboard.yml` runs the dashboard helper every six hours (plus whenever you trigger `workflow_dispatch`). It checks out the repo, sets up Python 3.10, installs `eclipse_scalper/requirements.txt`, and executes:

```bash
python eclipse_scalper/tools/telemetry_dashboard_notify.py --path logs/telemetry.jsonl --codes-per-symbol --codes-top 4
```

Keep `logs/telemetry.jsonl` (or the file defined by `TELEMETRY_PATH`) up to date so the scheduled job can read real telemetry. You can also edit the workflow’s cron if you want a different cadence.

### Notifications

The scheduled job now runs `python eclipse_scalper/tools/telemetry_dashboard_notify.py --path logs/telemetry.jsonl --codes-per-symbol --codes-top 4`. This wrapper prints the dashboard snapshot to the workflow log, installs the repo’s dependencies (`python -m pip install -r eclipse_scalper/requirements.txt`), and posts the same summary to Telegram when `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` are configured as [repository secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets). The notifier reuses `eclipse_scalper.notifications.telegram.Notifier`, so you get the same `speak` behavior already used elsewhere.

Leave the secrets empty if you only need GitHub Actions visibility; the job still succeeds and simply skips the Telegram send. If you do set those secrets, keep an eye on Telegram so you know when the latest snapshot hits critical topics like `exit.*` events or repeated `ERR_*` codes.

The workflow also runs `python eclipse_scalper/tools/signal_data_health.py ...` and `python eclipse_scalper/tools/core_health.py ...`, captures their stdout to `logs/signal_data_health.txt` / `logs/core_health.txt`, and uploads those files along with `logs/telemetry_health.html` as the `telemetry-snapshots` artifact for quick download.

### Alert classification

A subsequent workflow step runs `python eclipse_scalper/tools/telemetry_classifier.py --path logs/telemetry.jsonl --since-min 60`. This helper keeps a small state file at `logs/telemetry_classifier_state.json`, compares the current window against the previous snapshot, and flags:

- **Exit spikes** when `exit.*` event totals exceed `--exit-threshold` (default `5`) and are higher than the last run.
- **New telemetry codes** by remembering every `data.code` you’ve seen and reporting any novel entries.
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



If something looks wrong — \*\*it probably is\*\*.

Stop the bot, inspect logs, adjust, retry.



---



\## Development Status



\* ✅ Core execution engine

\* ✅ Signal pipeline

\* ✅ Order router

\* ✅ Risk kill-switches

\* ⚠️ Strategy tuning ongoing

\* ⚠️ Live trading requires caution



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

