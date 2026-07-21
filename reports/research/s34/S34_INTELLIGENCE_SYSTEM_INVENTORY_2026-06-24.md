# S34 Intelligence System Inventory

Date: 2026-06-24  
Status: implemented inventory, not a live-capital approval document

## Executive Summary

The current S34 intelligence system is an explainability and research layer around the shadow paper runner. It records signals, decisions, predictions, model audits, guardrails, shadow guardrails, trades, and outcomes into a separate SQLite ledger: `data/s34_intelligence.db`.

It is not yet the final prediction engine the user originally described. The system can explain what happened, compare model expectations against outcomes, surface weak buckets, and show risk/guardrail state. It does not yet produce an approved live entry/sizing recommendation.

Current ledger counts:

| Entity | Count |
| --- | ---: |
| Signals | 461 |
| Decisions | 535 |
| Rejected signals | 387 |
| Trades | 74 |
| Outcomes | 74 |
| Predictions | 1844 |
| Model audits | 461 |
| Model guardrails | 461 |
| Shadow guardrails | 922 |
| A/B results | 0 |

## What Exists Now

### 1. Intelligence Ledger

File:

- `tools/s34_intelligence_ledger.py`

Database:

- `data/s34_intelligence.db`

Purpose:

The ledger is a small append/upsert SQLite database separate from `microstructure.db`. It stores the reasoning trace around S34 paper trading without writing to the large market-data DB.

Tables:

| Table | Count | Purpose |
| --- | ---: | --- |
| `s34_signals` | 461 | Every candidate S34 signal seen by the runner |
| `s34_decisions` | 535 | ACCEPT / REJECT / CLOSE decisions |
| `s34_rejected_signals` | 387 | Rejections with reason codes |
| `s34_trades` | 74 | Paper trades opened/closed by the runner |
| `s34_outcomes` | 74 | Closed-trade result and cost decomposition |
| `s34_predictions` | 1844 | Model snapshots per signal |
| `s34_model_audit` | 461 | KNN/base-rate evidence and neighbors |
| `s34_model_guardrails` | 461 | Model-level OK / caution / warning |
| `s34_shadow_guardrails` | 922 | Shadow-only block/observe rules |
| `s34_ab_results` | 0 | Reserved for future model-vs-model evaluation |

### 2. Signal Capture

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/backfill_s34_intelligence_ledger.py`

What is recorded:

| Field group | Meaning |
| --- | --- |
| `signal_id` | Stable id based on rule and bucket timestamp |
| `signal_ts_ms` / `signal_ts_utc` | When the signal cluster occurred |
| `symbol`, `direction`, `liq_side`, `rule_name` | Which strategy fired |
| `cluster_notional` | Total liquidation cluster notional |
| `cluster_liq_count` | Number of liquidation events in cluster |
| `cluster_shape_label` | Geometry label if available |
| `features_json` | Full signal feature payload |

This is the raw memory of what the system saw before accepting or rejecting a trade.

### 3. Decision Ledger

Implemented in:

- `tools/s34_intelligence_ledger.py`
- runner calls inside `tools/s34_shadow_paper_runner.py`

Decision types:

| Decision | Meaning |
| --- | --- |
| `ACCEPT` | A paper trade was accepted/opened |
| `REJECT` | Signal was skipped by a gate |
| `CLOSE` | Trade closed and outcome became available |

Common reject reasons include:

- `REGIME_FILTER`
- `MAX_OPEN_TRADES`
- `MAX_SYMBOL_DIRECTION_OPEN_TRADES`
- `SAME_CLUSTER_LOWER_PRIORITY`
- `DAILY_MAX_SL`
- `COOLDOWN_AFTER_CONSECUTIVE_SL`
- `NO_FILL_DATA`

This layer answers: “Why did the system trade or not trade?”

### 4. Outcome / Cost Decomposition

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/s34_intelligence_ledger.py`

Stored in:

- `s34_outcomes`

Fields:

| Field | Meaning |
| --- | --- |
| `gross_bps` | Directional mark/reference move |
| `entry_adverse_bps` | Entry mark-to-fill adverse selection |
| `exit_adverse_bps` | Exit mark-to-fill adverse selection |
| `spread_cost_bps` | True bid/ask spread cost |
| `fee_cost_bps` | Fee cost |
| `net_bps` | Final net result |

Current identity target:

```text
net = gross - entry_adverse - exit_adverse - spread - fee
```

This was added after the P013/P056 debugging work. It separates signal quality from execution friction.

### 5. Prediction Snapshots

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/backfill_s34_intelligence_ledger.py`

Stored in:

- `s34_predictions`

Current prediction families:

| Model | Purpose |
| --- | --- |
| `base_rate_v1` | Route/symbol base-rate estimate from prior closed outcomes |
| `knn_v0` | Similarity model using early S34 feature set |
| `knn_v1` | KNN with additional geometry/duration/BTC-pre style features |
| `knn_v2` | More selective KNN variant |

Each prediction stores:

- expected net bps
- win rate
- K / neighbor count where relevant
- average similarity where relevant
- confidence note, usually `usable` or `thin`

Important limitation:

These predictions are currently diagnostic. They are not live trade authorization.

### 6. Model Audit

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/backfill_s34_intelligence_ledger.py`

Stored in:

- `s34_model_audit`

Purpose:

The audit stores the evidence behind predictions: base rates, KNN neighbor stats, similar historical trades, distance information, and explanation text.

It answers:

- Which historical events were considered similar?
- What was their median net bps?
- Was K large enough?
- Was similarity meaningful?
- Did the current signal look like a good or bad historical bucket?

### 7. Model Guardrails

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/backfill_s34_intelligence_ledger.py`
- `tools/s34_prediction_guardrail_report.py`

Stored in:

- `s34_model_guardrails`

Guardrail levels:

| Level | Meaning |
| --- | --- |
| `ok` | No negative model consensus |
| `caution` | Models disagree |
| `warning` | Similar signals have negative expectancy |
| `unknown` | Not enough usable prediction data |

Current logic:

- If 3 or more models expect negative net bps, level becomes `warning`.
- If strongly negative warnings exist, level becomes `warning`.
- If models disagree, level becomes `caution`.
- Otherwise, level is `ok`.

This is still a diagnostic layer. It is useful for shadow filters and dashboard warnings.

### 8. Shadow Guardrails

Implemented in:

- `tools/s34_shadow_paper_runner.py`
- `tools/backfill_s34_intelligence_ledger.py`
- `tools/s34_guardrail_v3_audit.py`
- `tools/s34_guardrail_v4_audit.py`

Stored in:

- `s34_shadow_guardrails`

Current shadow guardrails:

| Guardrail | Action | Definition |
| --- | --- | --- |
| `guardrail_v2_warning_100k_200k` | shadow observe / would-block | `model_guardrail=warning AND 100K <= cluster_notional < 200K` |
| `guardrail_v4_50k_warning_lt200k` | shadow observe / would-block | `rule=50K/TP120 AND model_guardrail=warning AND cluster_notional < 200K` |

V4 audit result:

| Metric | Value |
| --- | ---: |
| Baseline closed N | 74 |
| Baseline cum net | +1167.85 bps |
| V4 would-block closed N | 15 |
| V4 would-block cum net | -397.96 bps |
| Kept-after-block closed N | 59 |
| Kept-after-block cum net | +1565.82 bps |
| Delta if blocked | +397.97 bps |

Interpretation:

V4 is a strong in-sample shadow filter candidate, but it is not yet a live hard block. It needs forward confirmation.

### 9. Loss Postmortem

Implemented in:

- `tools/s34_50k_loss_postmortem.py`

Reports:

- `reports/research/s34/S34_50K_LOSS_POSTMORTEM.md`
- `reports/research/s34/S34_50K_LOSS_POSTMORTEM.json`

Key result:

The 50K/TP120 route became fragile because weak clusters dominated losses.

50K route closed performance:

| Metric | Value |
| --- | ---: |
| Closed N | 24 |
| Cum net | -41.93 bps |
| Mean net | -1.75 bps |
| Median net | -27.65 bps |
| Win rate | 29.17% |

Worst bucket:

| Bucket | N | Cum net | Median |
| --- | ---: | ---: | ---: |
| 100K-200K | 8 | -318.37 bps | -47.89 bps |
| <100K | 8 | -88.44 bps | -26.19 bps |

This motivated the V4 shadow guardrail.

### 10. Feature Factory

Implemented research scripts include:

- `tools/research_s34_feature_factory_phase0_coverage.py`
- `tools/research_s34_feature_factory_phase1_eth_buy_200k.py`
- `tools/research_s34_feature_query_phase1.py`
- `tools/research_s34_oos_validation.py`
- `tools/research_s34_real_fill_parity.py`
- `tools/research_s34_cluster_geometry_features.py`
- `tools/research_s34_500k_daytrend_route_sweep.py`

Major reports:

- `S34_FEATURE_FACTORY_PHASE0_COVERAGE_2026-06-16.md`
- `S34_FEATURE_FACTORY_PHASE1_ETH_BUY_200K.md`
- `S34_FEATURE_FACTORY_PHASE1_QUERY_RESULTS.md`
- `S34_FEATURE_FACTORY_PHASE1_OOS_VALIDATION.md`
- `S34_FEATURE_FACTORY_REAL_FILL_PARITY_TOP5.md`
- `S34_500K_DAYTREND_ROUTE_SWEEP.md`
- `S34_CLUSTER_GEOMETRY_FEATURES.md`

What it does:

- Converts liquidation clusters into structured feature rows.
- Separates decision-time features from outcome labels.
- Runs route/filter sweeps.
- Applies OOS validation.
- Applies real-fill parity checks.

Best validated research family so far:

- ETH BUY liquidation continuation.
- Stronger clusters outperform weak clusters.
- 500K/daytrend and 200K/TP60 variants are cleaner than 50K/TP120.

### 11. Live Paper Runner Integration

Implemented in:

- `tools/s34_shadow_paper_runner.py`

The runner now:

- Detects S34 signals.
- Applies regime/risk gates.
- Opens paper trades only.
- Uses real bookTicker fills.
- Quarantines missing fill data.
- Records lifecycle to JSON/CSV and the intelligence ledger.
- Records predictions, audits, model guardrails, and shadow guardrails per signal.

Important:

This is still paper/shadow. It does not place live Binance orders.

### 12. Live Dashboard

Implemented in:

- `tools/s34_live_chart.py`

Dashboard URL:

- `http://127.0.0.1:5050`

Existing dashboard surfaces:

| Surface | What it shows |
| --- | --- |
| Price chart | ETH price, BUY liquidation bars, trade markers, TP/SL/BE |
| Stream health | liquidations, bookTicker, mark_prices, agg_trades freshness |
| Process health | collector, watchdog, runner, chart process status |
| Intelligence panel | latest signal, decisions, base rates, predictions, model guardrail, KNN evidence, calibration |
| Forward cards | per-route trials, closed/open/skipped, guardrail levels, V2/V4 shadow summaries |
| Disk panel | D: free space and DB sizes |
| Constellation tab | 3D route/trade visualization |
| Analysis tab | equity curve, exits, net histogram, UTC hours, cluster scatter, guardrail outcome |
| Risk sandbox block | recently added risk-size diagnostic; useful but not the requested final prediction engine |

### 13. Current Paper Routes

Routes currently represented in dashboard/runner include:

| Route | Role |
| --- | --- |
| `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` | original pre-reg route, now weak/fragile |
| `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | exploratory ETH continuation |
| `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` | exploratory stronger ETH continuation |
| `ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30` | exploratory counter-regime/stretched variant |
| `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60` | exploratory BTC-pre-filter variant |
| `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` | exploratory SOL continuation |
| `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30` | exploratory BTC distributed cluster variant |

### 14. Prediction Risk Sandbox

Implemented in:

- `tools/s34_prediction_risk_sandbox.py`

Reports:

- `S34_PREDICTION_RISK_SANDBOX.md`
- `S34_PREDICTION_RISK_SANDBOX_100EQ_40MARGIN.md`

What it does:

- Takes account equity, trade margin, leverage grid, and risk budget.
- Computes TP/BE/SL dollar scenarios.
- Ranks routes using route stats, model expected median, top3-removed robustness, guardrail state, and shadow blocks.
- Blocks routes when sample/guardrail criteria fail.

What it does not do:

- It does not implement the user's intended final prediction system.
- It does not produce live entry advice.
- It should be treated as a diagnostic risk worksheet.

## What Is Missing

### 1. The Real Prediction Engine

Not yet implemented as the user intends.

Missing behavior:

- User enters account/trade amount.
- Engine looks at the current live signal.
- Engine compares it to the best validated buckets.
- Engine computes route quality, execution risk, current regime, adverse selection, and expected outcome distribution.
- Engine outputs a paper-only decision card:
  - `DO_NOT_TRADE`
  - `PAPER_CANDIDATE`
  - `RESEARCH_ONLY`
  - with exact reasons.

It should not say “go x40 live” until validation is mature. But it can say:

```text
For this paper signal, x10 fits the 2% SL budget; x20+ exceeds it.
Route quality: immature / blocked / candidate.
Reason: N, median, top3-removed, guardrail, adverse, regime.
```

### 2. A/B Model Evaluation

Table exists:

- `s34_ab_results`

Count:

- 0

Missing:

- Compare base_rate vs KNN v0/v1/v2 on later outcomes.
- Record which model was closer.
- Promote/demote models based on actual forecast error.

### 3. Proper Calibration

Dashboard has rough calibration diagnostics, but not a mature calibrated probability model.

Missing:

- Probability calibration buckets.
- Bootstrap confidence intervals.
- Deflated Sharpe / multiple testing correction in live prediction layer.
- Base-rate delta vs unconditional route base rate.

### 4. Temporal Leakage Proof

The architecture requires feature availability contracts. The current ledger stores feature JSON, but it does not yet store a full per-feature availability contract.

Missing:

- `source_min_ts_ms`
- `source_max_ts_ms`
- `available_after_ts_ms`
- `lookahead_safe`
- `null_reason`

### 5. Portfolio Risk Intelligence

Current risk is mostly per-route/per-trade.

Missing:

- Same-cascade correlation scoring across routes.
- Portfolio-level exposure compression.
- “These three trades are actually one event” risk accounting.
- Drawdown-conditional sizing.

### 6. Prediction UI

Live chart has intelligence panels, but not a clean “current prediction decision” card.

Missing ideal card:

| Field | Meaning |
| --- | --- |
| Current signal | latest S34 signal |
| Route candidate | best matching validated route |
| Model distribution | expected/median/worst-case net bps |
| Guardrails | ok/caution/warning plus reasons |
| Shadow filters | would block or observe |
| Risk fit | max leverage/margin under chosen risk budget |
| Verdict | paper candidate / blocked / research only |

## What Should Be Built Next

### Step 1: Replace the risk sandbox with a true prediction card

The sandbox should be demoted to a helper. The next useful system is:

`tools/s34_current_prediction_card.py`

Inputs:

- Latest live signal from `s34_signals`
- Current route stats from `s34_outcomes`
- Latest model predictions from `s34_predictions`
- Model guardrail from `s34_model_guardrails`
- Shadow guardrails from `s34_shadow_guardrails`
- Stream health from live chart API

Output:

- One JSON card for the latest signal.
- One markdown report.
- Dashboard panel.

### Step 2: Add model error tracking

For every closed trade:

- Join prediction snapshot to outcome.
- Compute forecast error per model.
- Store in `s34_ab_results` or a new `s34_prediction_errors` table.

### Step 3: Add feature availability contract

Do this before adding more ML-like models.

### Step 4: Add portfolio/event correlation

Before any leverage/sizing system is trusted, detect when multiple route trades come from the same cascade.

## Bottom Line

The intelligence system currently has a real ledger, live prediction snapshots, KNN/base-rate audits, model guardrails, shadow guardrails, forward-test summaries, live dashboard views, and postmortem reports.

The missing piece is not more visuals. The missing piece is a clean current-signal prediction card that turns those components into one explainable decision:

```text
What did the system see?
What similar events happened before?
What do the models expect?
What do the guardrails say?
What is the execution/risk cost?
Is this only research, paper candidate, or blocked?
```

That should be the next implementation target.
