# Codex Task: h=120 imb>=0.85 Fill Density Monitor

## Context

`event_block_eth_micro_imb085_v1` blocks `book_proxy_pressure` and
`volatility_burst` lanes before fills. For h=60 imb>=0.85 this improves NPA
consistently (PROMOTE status). For h=120 imb>=0.85 the direction is positive
but fill density after filtering drops below the minimum needed for promotion
(pass=0% due to sparse fills, not bad NPA).

We need a lightweight daily-runnable tool that tracks whether h=120 fill
density is recovering as more data accumulates, without running the full
walk-forward ranker.

## Task

Create `tools/monitor_h120_fill_density.py` — a self-contained CLI tool.

### What it does

1. Loads 1-sec bucket features from SQLite (same schema as other tools)
2. Identifies h=120 imb>=0.85 signal events:
   - `abs(imbalance) >= 0.85`
   - `trade_intensity >= 4000`
   - `spread <= 0.000150`
3. For each signal, simulates whether a passive fill would occur within 120s
   using the simple touch model:
   - LONG signal: fill if any future mid in next 120s drops below `entry * (1 - 0.5 * spread)`
   - SHORT signal: fill if any future mid in next 120s rises above `entry * (1 + 0.5 * spread)`
4. Applies the event block filter: exclude signals where either
   `book_proxy_pressure` or `volatility_burst` was active at signal time
   (use the same quantile-based detection as `tools/book_proxy_pressure_alerts.py`
   and `tools/volatility_burst_alerts.py` — copy the logic, do not import)
5. Reports:
   - Total signals (unfiltered)
   - Signals after event block (filtered_n)
   - Touch rate (filtered): fraction that touched the limit within 120s
   - Fill rate (filtered): `touch_rate * full_cond_touch_proxy` where
     `full_cond_touch_proxy = 0.5` (conservative constant — actual calibration
     requires more data; use this as a lower bound)
   - Estimated fills: `filtered_n * fill_rate`
   - Promotion threshold: `min_fills = 30` (minimum fills needed for reliable ranking)
   - Status: `READY_TO_RANK` if estimated_fills >= 30, else `INSUFFICIENT` with
     how many more fills are needed

### Output format

Print a human-readable summary:

```
h=120 imb>=0.85 Fill Density Monitor
  symbol        : ETHUSDT
  lookback_min  : 20160 (14D)
  signals_total : 142
  signals_filtered: 108  (kept 76.1% after event block)
  touch_rate    : 38.2%
  estimated_fills: 41
  min_fills_needed: 30
  status        : READY_TO_RANK
```

Also support `--json` flag for machine-readable output.

### CLI

```
python -m tools.monitor_h120_fill_density \
    --db ../eclipse_scalper/data/microstructure.db \
    --symbol ETHUSDT \
    --lookback-min 20160
```

Arguments:
- `--db` (required): path to microstructure.db
- `--symbol` (default: ETHUSDT)
- `--lookback-min` (default: 20160 = 14D)
- `--bucket-sec` (default: 1): bucket size for feature computation
- `--min-fills` (default: 30): threshold for READY_TO_RANK
- `--json`: output raw JSON

### Implementation rules

- **Self-contained**: load from SQLite directly. Do not import from `tools/`
  except you MAY import `tools.micro_edge_lib.build_bucket_features` if needed
  for feature computation — but prefer implementing the SQL aggregation inline
  as done in `tools/diagnose_volume_regime.py`.
- **No lookahead**: signal at bucket `i` may only use data from bucket `i` and
  earlier for the imbalance/intensity/spread check. Future mids for fill
  simulation are fine (that's measuring fill rate, not generating the signal).
- **Windows encoding**: add `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`
  at the top.
- **No plots, no external dependencies** beyond stdlib + sqlite3.

### Event block detection (copy from existing tools, do not import)

`book_proxy_pressure` fires on a bucket when:
- HIGH: `abs_imb >= imb_q90 AND intensity >= int_q75 AND spread >= spr_q50`
- MEDIUM: `abs_imb >= imb_q75 AND intensity >= int_q50 AND abs(ret_1) <= ret_q50 AND spread >= spr_q50`

`volatility_burst` fires on a bucket when:
- HIGH: `abs(ret_1) >= ret_q90 AND intensity >= int_q60`
- MEDIUM: `abs(ret_1) >= ret_q75 AND intensity >= int_q40 AND spread <= spr_q75`

All quantiles computed over the full lookback window.

## Deliverables

1. `tools/monitor_h120_fill_density.py`
2. `tests/test_monitor_h120_fill_density.py` with at least 2 tests:
   - One that verifies READY_TO_RANK status when estimated fills >= min_fills
   - One that verifies INSUFFICIENT status when fills are below threshold
   Use synthetic in-memory data (no DB required for tests).

## Branch

Work on branch: `codex/research/h120-density-monitor`
Base off: `main`
Do not touch any files in `execution/`, `risk/`, `brain/`, or `tools/rank_*`.
