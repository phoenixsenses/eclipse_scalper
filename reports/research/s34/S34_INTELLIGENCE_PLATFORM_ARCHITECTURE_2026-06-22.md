# S34 Intelligence Platform Architecture

Date: 2026-06-22
Status: architecture locked for implementation planning
Scope: S34 paper/live-intelligence stack, not live capital deployment

## Executive Verdict

The target architecture is directionally correct, but the current system is not yet a full intelligence platform. It is currently a working paper execution and research platform with good data ingestion, real-fill paper trades, and a live dashboard. The missing layer is an append-only intelligence ledger that explains every signal, every rejection, every prediction, every route choice, and every model decision.

The next implementation milestone should be:

**PR #11: S34 Intelligence Ledger + Audit Surface**

No TP/SL tuning, no new alpha rules, no leverage advisor. First make the system explainable and auditable.

## Current State

| Layer | Current State | Verdict |
| --- | --- | --- |
| Data ingestion | `liquidations`, `book_ticker`, `mark_prices`, `agg_trades` are live and fresh | mostly ready |
| Feature extraction | Feature factory exists for research; runner computes inline signal features | partial |
| Paper runner | Real bid/ask fill, chronology fix, no-fill quarantine, same-cluster dedupe | solid |
| Risk gate | Rule-level gates, same-symbol/direction cap, duplicate cluster prevention | partial but improving |
| Dashboard | Price chart, regime, forward cards, constellation, analysis tab | useful |
| Auditability | JSON/CSV journals exist, but no normalized signal/prediction/rejection ledger | not ready |
| SQ/KNN engine | Concept exists, not production-quality | not ready |
| A/B model eval | Not implemented | missing |
| Leverage advisor | Not safe to build yet | defer |

## Design Principles

1. Append-only truth first.
   Do not overwrite or silently mutate signal/trade decisions. Every decision gets a ledger row.

2. Signal, prediction, trade, and outcome are separate entities.
   A signal can be rejected. A prediction can be made without opening a trade. A trade can close later. An outcome label may become available only after exit.

3. Feature availability must be explicit.
   Every feature used by any model must be provably available at decision time.

4. No score without a base rate.
   Any quality score must be compared against unconditional and route/regime base rates.

5. Risk advice is disabled until edge validation is mature.
   No leverage recommendation should influence live trading until calibration, audit, and A/B infrastructure exist.

## Layer 0: Data Ingestion

Existing source tables:

| Source | Table | Required Use |
| --- | --- | --- |
| Binance forceOrder | `liquidations` | S34 cluster signal source |
| Binance bookTicker | `book_ticker` | executable bid/ask fills, spread/adverse selection |
| Binance mark price | `mark_prices` | trigger reference, regime trend/range |
| Binance agg trades | `agg_trades` | regime participation count, future flow features |

Required health contract:

| Check | Threshold | Action |
| --- | ---: | --- |
| `liquidations` freshness | <= 5 min | if stale, halt S34 signal generation |
| `book_ticker` freshness | <= 5 sec for fill timestamp | if missing, `NO_FILL_DATA` quarantine |
| `mark_prices` freshness | <= 30 sec | reject with `STALE_MARK_AT_ENTRY` |
| `agg_trades` freshness | <= 5 min | regime gate unavailable if stale |

Current status: live, but health should be snapshotted into an audit table at decision time.

## Layer 1: Feature Extraction + Validation

### Required Feature Contract

Every feature row must carry:

| Field | Meaning |
| --- | --- |
| `feature_name` | canonical feature id |
| `value` | numeric/string value |
| `source_table` | DB/source used |
| `source_min_ts_ms` | earliest source timestamp read |
| `source_max_ts_ms` | latest source timestamp read |
| `computed_at_ts_ms` | when runner/model computed it |
| `available_after_ts_ms` | earliest time this value is legally usable |
| `lookahead_safe` | boolean |
| `null_reason` | if unavailable |

This is the technical barrier against accidental lookahead. Metadata comments are not enough.

### Feature Table Split

Do not mix decision-time features with future labels.

| Table | Purpose | Lookahead Allowed |
| --- | --- | --- |
| `s34_event_features` | decision-time signal features | no |
| `s34_path_labels` | MFE/MAE/TP/SL/TIME labels after the fact | yes, labels only |

`s34_event_features` can feed models. `s34_path_labels` can train/evaluate models but must never be used as live entry input.

## Layer 2: Signal Quality Engine

The SQ engine should not output a single magic score. It should output a calibrated route distribution.

Required SQ output:

| Component | Example |
| --- | --- |
| `base_rate_net_bps` | unconditional/route/regime median |
| `neighbor_net_bps` | KNN similar-event median |
| `base_rate_delta_bps` | neighbor - base |
| `neighbor_n` | sample size |
| `bootstrap_ci_low` | lower confidence bound |
| `calibration_bucket` | calibrated probability bucket |
| `warnings` | low N, stale data, high no-fill, high adverse |

### KNN Guardrails

Before using KNN as a decision engine:

1. Temporal split must use `label_available_ts_ms`, not only `signal_ts_ms`.
2. Permutation importance must show which features matter.
3. KNN output must beat base rate after confidence bounds.
4. Calibration curve must show predicted probability vs realized probability.
5. Recency weighting must be explicit and logged.

Until those pass, KNN is research-only.

## Layer 3: Execution Intelligence

Current system already has:

- taker entry at executable side;
- taker SL/BE/TIME exit at executable side;
- TP fill mode config;
- fee decomposition;
- entry/exit adverse selection fields;
- no-fill quarantine.

Missing execution analytics:

| Module | Purpose |
| --- | --- |
| `entry_delay_audit` | entry delay vs MAE/net distribution |
| `adverse_selection_tracker` | entry/exit adverse bps by route and regime |
| `mfe_timing_profile` | time-to-MFE and time-to-TP distribution |
| `exit_route_simulator` | route alternatives without changing live runner |
| `fill_depth_simulator` | future larger-size impact model |

Execution layer is allowed to produce research candidates. It should not change live runner parameters without a separate pre-registered exploratory rule.

## Layer 4: Risk Engine

Current implemented risk controls:

- same-rule max open trades;
- same-symbol/same-direction cap;
- same-cluster priority dedupe;
- daily max SL;
- daily max loss;
- cooldown after consecutive SL;
- stale mark rejection;
- no-fill quarantine.

Missing risk controls:

| Module | Status | Priority |
| --- | --- | --- |
| Portfolio correlation | missing | high |
| Cascade correlation by event id | partial via dedupe | high |
| Drawdown-conditional sizing | missing | medium |
| Volatility/funding regime risk gate | missing | medium |
| Leverage advisor | missing | low/defer |

### Risk Engine Rule

No leverage recommendation should be emitted until:

1. N >= 100 clean real-fill trades for the route family;
2. calibration and holdout pass;
3. adverse selection distribution is stable;
4. portfolio correlation is measured;
5. maximum loss and p95 MAE are known.

Until then, leverage advisor output must be `DISABLED_IN_PAPER_VALIDATION`.

## Layer 5: Audit + Validation

This is the missing core.

### Required SQLite Tables

These tables should live in `microstructure.db` or a dedicated attached DB. For write safety, prefer a separate small DB first:

`data/s34_intelligence.db`

The paper runner can append to this DB while keeping `microstructure.db` read-only for market data.

### `s34_signals`

One row per raw candidate signal before regime/risk gates.

| Column | Type | Meaning |
| --- | --- | --- |
| `signal_id` | TEXT PRIMARY KEY | stable id: rule + symbol + side + bucket |
| `created_at_utc` | TEXT | runner insert time |
| `signal_ts_ms` | INTEGER | cluster first ts |
| `entry_ts_ms` | INTEGER | intended entry timestamp |
| `symbol` | TEXT | ETHUSDT/BTCUSDT/SOLUSDT |
| `liq_side` | TEXT | BUY/SELL |
| `direction` | TEXT | LONG/SHORT |
| `rule_id` | TEXT | rule name |
| `cluster_key` | TEXT | symbol/direction/liq_side/bucket |
| `cluster_notional` | REAL | total cluster notional |
| `cluster_liq_count` | INTEGER | liquidation count |
| `cluster_duration_sec` | REAL | cluster duration |
| `cluster_shape_label` | TEXT | geometry label |
| `feature_json` | TEXT | decision-time features |
| `feature_contract_json` | TEXT | availability metadata |
| `data_health_json` | TEXT | stream freshness at decision |

### `s34_predictions`

One row per model/rule prediction. A signal can have many model predictions.

| Column | Type | Meaning |
| --- | --- | --- |
| `prediction_id` | TEXT PRIMARY KEY | stable id |
| `signal_id` | TEXT | FK-ish to signal |
| `model_id` | TEXT | baseline/knn/v2/etc. |
| `model_version` | TEXT | git/config hash |
| `route_id` | TEXT | TP/SL/BE/delay route |
| `predicted_at_utc` | TEXT | timestamp |
| `base_rate_net_bps` | REAL | base median/mean metric |
| `expected_net_bps` | REAL | model EV |
| `p_tp` | REAL | probability TP |
| `p_sl` | REAL | probability SL |
| `p_be` | REAL | probability BE |
| `p_time` | REAL | probability TIME |
| `neighbor_n` | INTEGER | KNN/sample N |
| `quality_score` | REAL | calibrated quality score |
| `confidence` | TEXT | low/medium/high |
| `warnings_json` | TEXT | low N, stale, no-fill, etc. |

### `s34_decisions`

One row per final runner decision.

| Column | Type | Meaning |
| --- | --- | --- |
| `decision_id` | TEXT PRIMARY KEY | stable id |
| `signal_id` | TEXT | source signal |
| `prediction_id` | TEXT | optional prediction |
| `decision` | TEXT | OPEN/SKIP |
| `reason` | TEXT | accepted or skip reason |
| `priority_rank` | INTEGER | rule priority |
| `cluster_owner_signal_id` | TEXT | if skipped by dedupe |
| `risk_gate_json` | TEXT | all risk checks |
| `regime_json` | TEXT | regime snapshot |
| `created_at_utc` | TEXT | timestamp |

### `s34_trades`

One row per opened paper trade.

| Column | Type | Meaning |
| --- | --- | --- |
| `trade_id` | TEXT PRIMARY KEY | Pxxx |
| `signal_id` | TEXT | source signal |
| `decision_id` | TEXT | opening decision |
| `rule_id` | TEXT | rule |
| `status` | TEXT | OPEN/CLOSED |
| `entry_ts_ms` | INTEGER | entry timestamp |
| `entry_price` | REAL | executable fill |
| `entry_reference_price` | REAL | mark/trigger reference |
| `entry_fill_json` | TEXT | bid/ask/mid/source/fee |
| `tp_price` | REAL | TP level |
| `sl_price` | REAL | SL level |
| `be_trigger_price` | REAL | BE trigger |
| `risk_json` | TEXT | position sizing metadata |

### `s34_outcomes`

One row per closed trade/outcome label.

| Column | Type | Meaning |
| --- | --- | --- |
| `trade_id` | TEXT PRIMARY KEY | Pxxx |
| `exit_ts_ms` | INTEGER | exit timestamp |
| `exit_reason` | TEXT | TP/SL/BE/TIME |
| `exit_price` | REAL | executable exit fill |
| `exit_reference_price` | REAL | trigger reference |
| `exit_fill_json` | TEXT | bid/ask/mid/source/fee |
| `gross_bps` | REAL | mark/reference PnL |
| `entry_adverse_bps` | REAL | entry ref-to-mid |
| `exit_adverse_bps` | REAL | exit ref-to-mid |
| `spread_cost_bps` | REAL | true bid/ask spread cost |
| `fee_cost_bps` | REAL | fee cost |
| `net_bps` | REAL | net PnL |
| `mfe_bps` | REAL | max favorable excursion |
| `mae_bps` | REAL | max adverse excursion |
| `label_available_ts_ms` | INTEGER | when label is legally known |

### `s34_rejected_signals`

One row per skipped signal, useful for bias analysis.

| Column | Type | Meaning |
| --- | --- | --- |
| `signal_id` | TEXT PRIMARY KEY | source signal |
| `rule_id` | TEXT | rule |
| `rejected_at_utc` | TEXT | timestamp |
| `reason` | TEXT | REGIME_FILTER/MAX_OPEN/etc. |
| `would_have_entry_ts_ms` | INTEGER | intended entry |
| `cluster_key` | TEXT | event cluster |
| `feature_json` | TEXT | features at rejection |
| `risk_gate_json` | TEXT | risk checks |
| `data_health_json` | TEXT | feed health |

### `s34_model_audit`

One row per model quality evaluation.

| Column | Type | Meaning |
| --- | --- | --- |
| `audit_id` | TEXT PRIMARY KEY | stable id |
| `prediction_id` | TEXT | model prediction |
| `model_id` | TEXT | model |
| `neighbor_ids_json` | TEXT | KNN neighbors |
| `feature_weights_json` | TEXT | weights |
| `permutation_importance_json` | TEXT | feature importance |
| `base_rate_json` | TEXT | base rate comparison |
| `calibration_json` | TEXT | calibration bucket |
| `created_at_utc` | TEXT | timestamp |

### `s34_ab_results`

For parallel model evaluation.

| Column | Type | Meaning |
| --- | --- | --- |
| `ab_id` | TEXT PRIMARY KEY | stable id |
| `signal_id` | TEXT | signal |
| `model_a_prediction_id` | TEXT | model A |
| `model_b_prediction_id` | TEXT | model B |
| `winner` | TEXT | A/B/tie/unknown |
| `outcome_trade_id` | TEXT | realized trade if any |
| `scored_at_utc` | TEXT | after outcome |

## Layer 6: API + Dashboard

Current dashboard is useful but should be split into explicit surfaces:

| Surface | Purpose |
| --- | --- |
| Price Chart | live price/liquidation/regime/open trade |
| Constellation | route/trade structure, time scrub, isolation |
| Analysis | equity, distribution, route stats |
| SQ Panel | signal quality and route probability, future |
| Audit Panel | why trade/why skip, next PR |
| Data Health | feed freshness and no-fill warnings |

### Future API Endpoints

If/when moving to FastAPI:

| Endpoint | Purpose |
| --- | --- |
| `/health` | process + stream health |
| `/signals/latest` | latest signals and rejected signals |
| `/trades/open` | open trade state |
| `/trades/recent` | recent closed trades |
| `/audit/signal/{signal_id}` | full why/why-not trace |
| `/sq/analyze` | research-only SQ output |
| `/risk/state` | portfolio exposure/risk state |

Current `tools/s34_live_chart.py` can stay as the lightweight dashboard until the ledger exists.

## Data Flow

```text
market data
  -> feature extractor
  -> s34_signals
  -> signal quality / baseline models
  -> s34_predictions
  -> regime + risk gates
  -> s34_decisions
  -> s34_trades
  -> s34_outcomes
  -> calibration / A/B / dashboard
```

Rejected path:

```text
s34_signals
  -> risk/regime reject
  -> s34_rejected_signals
  -> bias tracker
```

## Implementation Roadmap

### PR #11: Intelligence Ledger

Goal: normalized append-only audit truth.

Deliverables:

- `data/s34_intelligence.db` schema creation script.
- Append-only writer module.
- Runner writes:
  - `s34_signals`
  - `s34_decisions`
  - `s34_trades`
  - `s34_outcomes`
  - `s34_rejected_signals`
- Backfill script from existing `S34_SHADOW_PAPER_TRADES.json`.
- Dashboard Audit Panel v1:
  - latest signals;
  - latest skipped;
  - reason counts;
  - same-cluster owner trace.

Non-goals:

- no KNN;
- no new strategy;
- no TP/SL tuning;
- no leverage advice.

Acceptance:

- Every new trial in JSON has matching ledger rows.
- Every skipped signal has a visible reason.
- Same-cluster dedupe owner is traceable.
- No market-data writes to `microstructure.db`.

### PR #12: Temporal + Label Quality Audit

Goal: prevent fake edge.

Deliverables:

- strict `label_available_ts_ms` audit;
- mark-vs-bid/ask label comparison;
- no-fill selection bias report;
- feature availability contract checker.

Acceptance:

- any model training query can be checked for lookahead safety.

### PR #13: Base Rate + SQ Panel v1

Goal: quality scores with context.

Deliverables:

- base-rate engine;
- route/regime/symbol base comparison;
- SQ panel in dashboard;
- confidence labels.

Acceptance:

- every displayed quality score shows base-rate delta and sample N.

### PR #14: A/B Prediction Logging

Goal: evaluate new models without changing trading.

Deliverables:

- model predictions logged in parallel;
- no influence on runner decisions;
- realized outcome scorer.

Acceptance:

- model A vs model B can be evaluated on future data without p-hacking.

### PR #15: KNN Similarity Engine

Goal: only after ledger/audit/base-rate exists.

Deliverables:

- KNN feature vector;
- recency weighting;
- permutation importance;
- calibration curve;
- audit neighbor list.

Acceptance:

- KNN output beats base rate after confidence bounds on forward data.

## Readiness Gate

Before any risk/leverage advisor:

| Requirement | Status |
| --- | --- |
| real-fill paper trades | yes |
| same-cluster dedupe | yes |
| normalized signal ledger | no |
| rejected signal bias tracker | no |
| label availability audit | no |
| base-rate comparison | no |
| calibration curve | no |
| A/B prediction logging | no |
| portfolio correlation model | no |

Conclusion: build PR #11 before model sophistication.

## Final Architecture Rule

The platform should not answer:

```text
Should I long now?
```

It should answer:

```text
What signal fired?
What route is statistically favored?
What is the base-rate delta?
What are the execution costs?
What risks block or reduce this trade?
What data freshness and lookahead constraints apply?
What happened to similar prior signals?
Why did the system trade or skip?
```

That is the difference between a dashboard and an intelligence platform.
