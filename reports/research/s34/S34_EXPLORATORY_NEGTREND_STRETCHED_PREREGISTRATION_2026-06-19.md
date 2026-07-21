# S34 Exploratory Rule Pre-Registration — 500K Negative-Trend Stretched BUY Liq

Status: exploratory paper only. This does not alter the pre-registered 50K/TP120 S34 validation sample.

Date: 2026-06-19

Rule of record:

`ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30`

## Entry Definition

- Symbol: ETHUSDT
- Liquidation side: BUY
- Direction: LONG
- Cluster notional threshold: >= 500,000 USDT
- Bucket/window: 300 seconds
- Minimum signal gap: 900 seconds
- Day trend gate: `day_trend_bps < 0`
- Cluster shape gate: `stretched_120s`
  - implemented live as `cluster_duration_sec >= 120` and `max_single_liq_share < 80%`
- Entry delay: 0 seconds

All entry features are no-lookahead and known at cluster completion / signal time.

## Exit / Cost Model

- TP: +60 bps
- SL: -40 bps
- BE trigger: +30 bps
- Max horizon: 3600 seconds
- Entry: taker executable ask
- SL/BE/TIME exit: taker executable bid
- TP mode: taker
- Fee: 4 bps per side
- Requires real bookTicker fill. Missing fill data is skipped, not modeled.

## Why This Exists

The 2026-06-19 counter-regime real-fill research showed that the broad `500K + day_trend_bps < 0` condition was not sufficient:

- real-fill N=15
- median=-9.08 bps

Adding cluster geometry isolated a stronger pocket:

- `500K + daytrend negative + stretched_120s`
- real-fill N=7
- median=+53.35 bps
- cum=+381.29 bps
- WR=100%

This is still small-N research and had high historical no-fill coverage loss. Therefore it can only be tested as a separate exploratory paper rule.

## Forward Evaluation

Minimum before any interpretation:

- N >= 30 closed real-fill paper trades
- at least 8 distinct UTC days
- median net bps > 0
- top-3-removed cumulative net bps > 0
- no-fill / quarantine behavior not concentrated during the highest-intensity cascades

Failure before N=30 is not a verdict unless a structural bug or data-integrity issue appears.

## Isolation Rules

- Does not count toward 50K/TP120 pre-reg N.
- Does not change existing 50K/TP120, 200K/TP60, 500K/daytrend, or BTC_PRE15 rules.
- Must remain separately labeled in journal, chart, and analysis outputs.
- Any future tuning creates a new rule id and a new pre-registration note.
