# Liquidation Alert Handoff

## Purpose

`high_liq_reversal_regime` is no longer treated as a primary trade pocket candidate.

Current research conclusion:

- useful as an event-intelligence signal
- not useful as a direct extension of `micro_edge_v3_passive_alpha`
- should be consumed by runtime/dashboard as an alert and annotation lane

## Current Research Evidence

- selected `v3` trade surface overlap:
  - 1 day: `tagged_n = 0`
  - 7 day: `tagged_n = 0`
- recent alert feed on real ETHUSDT data, last 240 minutes:
  - `rows_total = 2843`
  - `tagged_count = 112`
  - `tagged_rate = 3.94%`
  - `recent_alert_count = 20`
  - `max_consecutive_tagged = 3`
  - `max_liq_rate_recent = 11.4712`

Interpretation:

- the regime exists
- the regime is recent enough to monitor
- the regime does not currently overlap the selected `v3` trade surface

## Runtime Recommendation

Person 2 should consume this as:

1. dashboard alert feed
2. dashboard state/card payload
2. monitoring annotation
3. optional execution caution context

Do not consume it yet as:

1. direct auto-trade trigger
2. unconditional score boost
3. replacement for existing runtime guards

## Producer

Tool:

- `python -m tools.liquidation_regime_alerts`

Primary real artifact:

- `reports/LIQUIDATION_REGIME_ALERTS_REAL.json`
- `reports/LIQUIDATION_ALERT_STATE_REAL.json`
- `reports/LIQUIDATION_WATCHLIST_REAL.json`

## Minimal Contract

Top-level fields:

- `symbol`
- `rule`
- `lookback_min`
- `bucket_sec`
- `recent_limit`
- `min_liq_rate`
- `summary`
- `alerts`
- `run_summary`

### Summary block

- `rows_total`
- `tagged_count`
- `tagged_rate`
- `recent_alert_count`
- `max_consecutive_tagged`
- `max_liq_rate_recent`
- `side_bias_counts`
- `severity_counts`

### Alert row

- `ts_ms`
- `tag`
- `side_bias`
- `severity`
- `liq_rate_per_sec`
- `liq_imbalance`
- `spread`
- `trade_intensity`
- `ret_1`

## Suggested Runtime Usage

### Dashboard

Show:

- last 20 alerts
- current state card: `quiet / elevated / severe`
- freshness: `fresh / stale`
- side bias distribution
- severity distribution
- max liquidation rate in window
- tagged rate in current window

Suggested card source:

- `python -m tools.liquidation_alert_state --alerts-json reports/LIQUIDATION_REGIME_ALERTS_REAL.json --out-json reports/LIQUIDATION_ALERT_STATE_REAL.json`
- stale payload should be rendered as stale, not active
- runtime can directly use:
  - `dashboard_summary`
  - `notification_text`
  - `recommended_action`

Suggested watchlist source:

- `python -m tools.liquidation_watchlist --symbols ETHUSDT,BTCUSDT --out-json reports/LIQUIDATION_WATCHLIST_REAL.json`
- use this for multi-symbol ranking / watch tables
- `top_summary` can be used for a single top-watch badge without parsing the full rows list
- `banner` can be used for a single dashboard header / top-of-page strip

### Monitoring

Raise a soft alert when:

- `recent_alert_count >= 3` in the chosen window
- or `max_consecutive_tagged >= 2`
- or `max_liq_rate_recent >= 5.0`

### Execution Context

Optional future use:

- if `side_bias` agrees with other runtime stress signals, reduce aggression
- if `side_bias` conflicts with current passive entry assumptions, mark as caution

## Non-Goals

This lane does not currently answer:

- whether to enter a trade
- whether to force exit a position
- whether liquidation reversal is profitable after costs

It only answers:

- whether a meaningful liquidation regime is active now
- which side the regime is biased toward
- how intense the current regime is

## Handoff Note For Person 2

Start with:

1. read `reports/LIQUIDATION_REGIME_ALERTS_REAL.json`
2. add a dashboard card and recent-alert table
3. add monitoring thresholds as soft alerts only
4. do not wire to execution decisions yet
