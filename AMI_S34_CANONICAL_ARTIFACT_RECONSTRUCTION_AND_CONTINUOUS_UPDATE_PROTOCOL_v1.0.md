# AMI × S34 — Canonical Artifact Reconstruction, Reconciliation and Continuous Update Protocol

**Document ID:** `AMI-S34-ARTIFACT-BUILD-0001`  
**Version:** `1.0.0`  
**Status:** `CANONICAL BUILD SPECIFICATION / EXECUTION RUNBOOK / AGENT HANDOFF`  
**Canonical date:** `2026-07-03`  
**Primary use:** Give this file to an engineering/research agent and instruct it to reconstruct, reconcile, update and publish the complete AMI × S34 knowledge system from the newest available artifacts.  
**Default mode:** `RESEARCH-ONLY / NO LIVE MUTATION`  

---

# 0. Mission

This specification defines how to rebuild and continuously update the complete AMI × S34 research estate from all available artifacts, including:

- SQLite databases and SQL dumps;
- Excel workbooks and CSV exports;
- Word documents, PDFs, Markdown and text reports;
- JSON/JSONL paper, shadow, forward and live ledgers;
- Python scripts, notebooks and test code;
- YAML/TOML/INI/config files and frozen rule definitions;
- runtime logs, watchdog records, order/fill records and position history;
- Git commits, diffs, changelogs, decision records and experiment registries;
- charts, screenshots and dashboard exports;
- rejected, superseded, lookahead-contaminated and deprecated research;
- current canonical whitepapers, session reports and scientific-question registries.

The objective is not merely to copy files. The objective is to construct a governed, reproducible, latest-state representation of:

```text
raw observations
→ events and cycles
→ experiments
→ evidence
→ knowledge objects
→ alpha families
→ paper/shadow/forward/live permissions
→ dashboards and reports
→ future data collection requirements
```

The build must preserve history while ensuring that the newest scientifically valid conclusion is the one shown as current.

---

# 1. Non-Negotiable Requirements

The executing agent must obey all of the following.

## 1.1 Use the newest canonical state

Every final artifact must represent the most current valid state available at execution time.

“Newest” must **not** be determined by filename or filesystem modification time alone. The canonical choice must consider:

1. explicit canonical/supersession declarations;
2. scientific status and evidence level;
3. data coverage end date;
4. experiment creation and completion time;
5. frozen protocol version;
6. code commit and dataset hash;
7. whether the result was later falsified, restricted or rejected;
8. whether a newer file is merely a copy of an older result;
9. whether the artifact belongs to the correct AMI domain;
10. whether it contains raw evidence, a derived summary or a narrative interpretation.

Old versions must remain available in lineage and history, but may not silently override the current conclusion.

## 1.2 Include the complete evidence lifecycle

The build must include, separately and without mixing:

```text
HISTORICAL BACKTEST
REPLAY
PAPER
SHADOW
FORWARD SHADOW
FORWARD PAPER
LIVE OBSERVATION
LIVE EXECUTION
POST-LIVE AUDIT
```

A profitable historical row is not a forward result. A shadow outcome is not a fill. A “LIVE” registry label is not sufficient evidence of live execution. Every metric must state its exact evidence layer.

## 1.3 Never pool unlike policies

Two rows with the same signal name may not be pooled unless they share all of the following:

- exact route version;
- event universe;
- feature availability time;
- entry definition;
- delay;
- side;
- hold/exit policy;
- stop and take-profit rules;
- fee/slippage model;
- ledger implementation;
- data-quality requirements;
- code version.

`LONG_SILENCE` in a realtime ledger and `LONG_SILENCE` in a state-machine ledger must therefore be treated as different policy implementations unless proven identical.

## 1.4 Preserve rejected knowledge

Rejected, null, failed, lookahead-contaminated and superseded results must not be deleted.

They must be retained with explicit status such as:

```text
REJECTED_ECONOMIC
REJECTED_STATISTICAL
FALSIFIED
INSUFFICIENT_SAMPLE
LOOKAHEAD_CONTAMINATED
LATE_INFORMATION
EXECUTION_FAILURE
REGIME_LIMITED
SUPERSEDED
LEGACY
ARCHIVED
```

The latest dashboard may hide these by default, but a dedicated archive and contradiction view must expose them.

## 1.5 No silent live changes

Unless the operator explicitly requests live modification, this protocol must not change:

- live route allowlists;
- `.env` files;
- API credentials;
- leverage;
- position sizing;
- order logic;
- stop/TP logic;
- runtime services;
- exchange permissions;
- watchdog thresholds;
- live configuration.

The default execution mode is reconstruction, analysis and publishing only.

## 1.6 Separate code-test success from alpha success

“119 tests green” means implementation/mutation tests passed. It does **not** mean the hypotheses were validated.

Scientific verdicts such as `FALSIFIES`, `REJECTED[econ]` and `INSUFFICIENT_SAMPLE` must remain visible even when all software tests pass.

## 1.7 No unsupported economic conversion

Basis-point sums may not be presented as account return unless all of the following are explicit:

- fixed or variable notional;
- leverage;
- margin utilization;
- concurrency;
- compounding rule;
- fees and slippage;
- missed fills;
- liquidation risk;
- capital constraints;
- overlap and mutual exclusivity.

`sum_bps` is an arithmetic route statistic, not automatically a portfolio equity return.

---

# 2. Domain and Namespace Safety

## 2.1 Primary domain

In this project, **AMI means Artificial Market Intelligence**.

The canonical project domain is:

```text
crypto and financial market intelligence
market states and transitions
liquidation/cascade research
LONG/SHORT/WAIT decision research
trade lifecycle and execution research
epistemic governance
```

## 2.2 Mandatory acronym-collision gate

Any file that uses AMI to mean something else must be quarantined before ingestion.

A current example is the Word document:

```text
AMI Bağlamında Gözlenen Pazar Durumu Gerçek Bir Süreci Temsil Ediyor mu.docx
```

That document defines AMI as **Advanced Metering Infrastructure** in the electricity-grid/smart-meter domain. It must **not** be merged into Artificial Market Intelligence research, state taxonomies or causal claims.

It may be classified as:

```yaml
artifact_domain: EXTERNAL_ANALOGY
namespace: ADVANCED_METERING_INFRASTRUCTURE
canonical_for_ami_s34: false
allowed_use:
  - analogy about observability
  - analogy about sensor fusion
  - analogy about causal action support
forbidden_use:
  - direct AMI market evidence
  - S34 alpha evidence
  - crypto state definition
  - canonical whitepaper replacement
```

The build must create a `namespace_collision_report` and flag every similarly ambiguous artifact.

## 2.3 Semantic domain classifier

For every file, assign one of:

```text
AMI_ARTIFICIAL_MARKET_INTELLIGENCE
S34_TRADING_RESEARCH
EXECUTION_OPERATIONS
GENERAL_MARKET_RESEARCH
EXTERNAL_ANALOGY
UNRELATED
UNKNOWN_REQUIRES_REVIEW
```

Domain classification should use title, body text, named entities, schema fields and repository location—not filename alone.

---

# 3. Required Input Discovery

The agent must recursively inspect the configured project root and any supplied artifact folders.

## 3.1 Supported file types

At minimum:

```text
.db .sqlite .sqlite3 .sql
.xlsx .xlsm .xls .ods .csv .tsv
.docx .doc .pdf .md .txt .rtf
.json .jsonl .ndjson
.yaml .yml .toml .ini .env
.py .ipynb .r .sql
.log .out .err
.png .jpg .jpeg .webp .svg
.zip .tar .gz .7z
.git metadata and patch/diff files
```

## 3.2 Initial known artifacts

The build should expect and register artifacts such as:

```text
AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE(3).md
S34_SESSION_SONUC_RAPORU_2026-07-03.md
S34_ALL.db
S34_ALL.sql
AMI_S34_COMPLETE_RESULTS_DASHBOARD_v0.2.xlsx
AMI_S34_FULL_QUESTION_TEST_AUDIT_v0.1.md
AMI_S34_QUESTION_TEST_MATRIX_Q396_Q730.csv
AMI_S34_FAMILY_TEST_SUMMARY.csv
AMI_S34_TEST_PACKAGE_v0.1.zip
S34_REPRODUCIBLE_PAPER_ANALYSIS.py
related research scripts and output CSV files
paper/shadow/forward JSON or JSONL ledgers
SYSTEM_STATE.md
Decision Records, PATCH records and experiment reports
```

These filenames are seeds, not an exhaustive list. The agent must search for newer replacements and related artifacts.

## 3.3 Archive handling

Compressed archives must be inventoried and safely extracted into a temporary read-only staging directory.

For every archive:

- hash the archive;
- list members;
- detect duplicate files;
- detect path traversal attempts;
- record archive-to-member lineage;
- avoid overwriting live repository files;
- compare extracted members against already discovered files.

---

# 4. Artifact Registry

The first durable output must be a complete artifact registry.

## 4.1 Required fields

```yaml
artifact_id:
path:
filename:
extension:
size_bytes:
sha256:
created_at_fs:
modified_at_fs:
content_date_start:
content_date_end:
document_version:
code_commit:
dataset_hash:
experiment_ids:
route_ids:
ledger_types:
artifact_domain:
artifact_role:
canonical_status:
supersedes:
superseded_by:
derived_from:
contains_raw_data:
contains_results:
contains_permissions:
parse_status:
quality_flags:
notes:
```

## 4.2 Artifact roles

Assign one or more:

```text
RAW_DATA
SOURCE_DATABASE
SQL_DUMP
DERIVED_DATASET
RESULT_WAREHOUSE
EXPERIMENT_REPORT
SESSION_REPORT
WHITEPAPER
QUESTION_REGISTRY
DASHBOARD
EXECUTION_LEDGER
CONFIGURATION
CODE
TEST
IMAGE_EVIDENCE
CHANGELOG
DECISION_RECORD
ARCHIVE
EXTERNAL_REFERENCE
```

## 4.3 Canonical status

```text
CANONICAL_CURRENT
CANONICAL_COMPONENT
SUPERSEDED
HISTORICAL
DUPLICATE_EXACT
DUPLICATE_SEMANTIC
CONFLICTING
QUARANTINED_NAMESPACE
UNVERIFIED
```

---

# 5. Canonical Selection and Supersession Algorithm

For each topic, route, experiment, report family or output type, select the canonical state through the following ordered procedure.

## 5.1 Group candidates

Group artifacts by semantic identity, not only filename. Examples:

- same experiment ID;
- same route ID;
- same whitepaper family;
- same session date;
- same database schema;
- same question range;
- same dashboard purpose.

## 5.2 Exact duplicate detection

If SHA-256 hashes match, mark duplicates and choose one storage representative.

## 5.3 Semantic duplicate detection

If content is nearly identical but metadata or formatting differs, retain both references but choose one canonical representation.

## 5.4 Supersession rules

A newer result supersedes an older result when it explicitly:

- reruns the same claim on a broader or untouched sample;
- fixes lookahead or data leakage;
- corrects a broken feed;
- replaces hardcoded or invalid statistics;
- adds real-fill or forward evidence;
- uses a frozen preregistered protocol;
- resolves an implementation discrepancy;
- declares the earlier result superseded.

An older result may remain canonical for a narrower scope if the newer result tests a materially different population.

## 5.5 Conflict preservation

If two valid implementations disagree, do not select one by convenience. Create a contradiction object:

```yaml
conflict_id:
claim:
implementation_a:
implementation_b:
shared_scope:
different_assumptions:
current_resolution:
required_test:
permission_effect:
```

Example:

```text
LONG_SILENCE realtime positive
vs
LONG_SILENCE state-machine negative
```

The correct conclusion is implementation dependence, not pooled positivity.

---

# 6. Format-Specific Ingestion Rules

## 6.1 SQLite databases

For every database:

- run integrity checks;
- enumerate tables, views, indexes and triggers;
- capture row counts;
- capture schemas and constraints;
- calculate deterministic table hashes where feasible;
- detect min/max timestamps;
- detect null coverage;
- detect duplicate primary keys or natural keys;
- identify raw vs derived tables;
- identify views with hidden transformations;
- export a schema dictionary.

Never assume a database is raw because it is large. `S34_ALL.db` is primarily a result warehouse.

## 6.2 SQL dumps

For every SQL dump:

- import into an isolated temporary database;
- compare table counts and hashes against the source DB if available;
- identify source tables absent from the dump;
- identify derived DB-only views/tables;
- record engine compatibility and encoding;
- never execute untrusted non-schema shell commands.

Current known state to preserve:

```text
S34_ALL.sql matches the four source tables in S34_ALL.db at row/hash level.
It does not add raw book, OI, path, position or probability data.
The DB-only research_clean object is derived and is not a source-table addition from the SQL dump.
```

## 6.3 Excel workbooks

For every workbook:

- inspect all sheets, tables, named ranges, formulas, pivots and charts;
- distinguish input cells from derived cells;
- detect hardcoded values that appear to be statistics;
- detect formula errors;
- record filters and hidden sheets;
- compare data ranges with source DB/CSV outputs;
- preserve formatting only after validating content.

A beautiful spreadsheet is not evidence unless its numbers trace to a source.

## 6.4 CSV/TSV

- detect delimiter and encoding;
- infer types conservatively;
- preserve original text values;
- identify source/derived status;
- hash normalized rows;
- check duplicate records;
- detect percentage scale ambiguity (`0.73` vs `73`).

## 6.5 Word documents

- extract headings, paragraphs, tables, footnotes, references and embedded images;
- detect document subject and namespace;
- preserve page/table references where useful;
- classify claims and source quality;
- never treat narrative text as a database result without provenance.

## 6.6 PDFs

- extract text and metadata;
- inspect image-based charts and tables;
- retain page references;
- mark OCR-derived content separately;
- classify whether the PDF is a report, source paper, dashboard export or external reference.

## 6.7 Markdown and text

- parse headings and document-control blocks;
- extract experiment IDs, verdicts, route names and file references;
- resolve relative links;
- detect explicit “supersedes,” “rejected,” “live unchanged” and “operator decision” statements.

## 6.8 JSON/JSONL ledgers

- validate every line independently;
- quarantine malformed rows;
- preserve ledger filename and implementation identity;
- infer schema versions;
- normalize timestamps to UTC without discarding original values;
- distinguish anchor time, signal time, decision time, order time, fill time and exit time;
- retain raw payload hashes.

## 6.9 Code and notebooks

- register code commit and file hash;
- identify data inputs and output paths;
- identify random seeds;
- detect lookahead-prone feature construction;
- detect hardcoded p-values, thresholds and dates;
- run tests in an isolated environment where possible;
- compare generated outputs to stored artifacts;
- record reproducibility verdict.

## 6.10 Logs and execution records

- parse service identity, PID, route version and start/stop times;
- distinguish health logs from trading evidence;
- connect signals to orders, orders to fills and fills to positions;
- preserve rejected orders and missing fills;
- calculate end-to-end latency only when all timestamps are present.

---

# 7. Canonical SQL Knowledge Warehouse

The build must create or migrate a canonical SQL warehouse rather than relying on disconnected files.

Recommended database name:

```text
AMI_S34_CANONICAL.db
```

Recommended matching portable dump:

```text
AMI_S34_CANONICAL.sql
```

## 7.1 Core metadata tables

### `artifact_registry`

Stores every discovered artifact and its canonical status.

### `artifact_lineage`

```sql
CREATE TABLE artifact_lineage (
    parent_artifact_id TEXT NOT NULL,
    child_artifact_id  TEXT NOT NULL,
    relation_type      TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    PRIMARY KEY (parent_artifact_id, child_artifact_id, relation_type)
);
```

Relation types:

```text
DERIVED_FROM
EXPORT_OF
SUPERSEDES
DUPLICATE_OF
VALIDATES
CONTRADICTS
GENERATED_BY
```

### `data_quality_events`

Stores feed gaps, stale tables, schema changes, late values and reconstruction flags.

### `namespace_registry`

Stores acronym and domain classifications to prevent AMI-domain contamination.

## 7.2 Scientific governance tables

### `experiment_registry`

Minimum fields:

```text
experiment_id
question_ids
hypothesis_id
preregistered_at
frozen_population
frozen_features
frozen_target
frozen_thresholds
frozen_splits
frozen_economic_gate
frozen_statistical_gate
code_commit
dataset_hash
started_at
completed_at
verdict
mutation_test_count
mutation_test_passed
supersedes_experiment_id
report_artifact_id
```

### `hypothesis_registry`

Tracks null, primary, alternative, confounder and data-quality hypotheses.

### `knowledge_objects`

Must include:

```text
knowledge_id
claim
claim_type
status
evidence_level
scope
applicable_symbols
applicable_regimes
applicable_timeframes
feature_availability_time
statistical_confidence
forward_confidence
execution_confidence
mechanism_confidence
contradictions
assumptions
falsification_conditions
permitted_uses
forbidden_uses
created_at
last_verified_at
review_due_at
supersedes_knowledge_id
```

### `contradiction_registry`

Stores unresolved and resolved conflicts between findings.

### `researcher_exposure_ledger`

Tracks how many splits, thresholds, route variants and reports were seen before a claim was frozen.

## 7.3 Route and alpha tables

### `alpha_families`

Must preserve the original registry but expand it with:

```text
alpha_id
canonical_name
direction
mechanism_family
route_version
universe
entry_rule
feature_availability_time
hold_rule
exit_rule
risk_rule
status
permission_level
current_evidence_level
forward_n
live_n
superseded_by
notes
```

### `route_versions`

Every material policy change creates a new immutable route version.

### `route_permissions`

Separate permissions:

```text
RESEARCH_ONLY
REPLAY_ALLOWED
SHADOW_ALLOWED
FORWARD_SHADOW_ALLOWED
PAPER_ALLOWED
OBSERVER_ALLOWED
LIVE_ALLOWED
SIZING_ALLOWED
PORTFOLIO_ALLOWED
```

## 7.4 Market event and cycle tables

### `ami_events`

```sql
CREATE TABLE ami_events (
    event_id                 TEXT PRIMARY KEY,
    cycle_id                 TEXT,
    symbol                   TEXT NOT NULL,
    venue                    TEXT,
    event_family             TEXT NOT NULL,
    liquidation_side         TEXT,
    anchor_ts_utc            TEXT NOT NULL,
    event_start_ts_utc       TEXT,
    event_end_ts_utc         TEXT,
    feature_available_ts_utc TEXT,
    notional                 REAL,
    event_count              INTEGER,
    duration_seconds         REAL,
    max_single_share         REAL,
    displacement_bps         REAL,
    structural_location      TEXT,
    source_quality           TEXT,
    censor_status            TEXT,
    source_artifact_id       TEXT
);
```

### `ami_cycles`

A cycle is a structural episode, not an arbitrary cooldown bucket.

Fields:

```text
cycle_id
symbol
start_ts
end_ts
cycle_definition_version
entry_state
peak_state
exit_state
event_count
direction_conflict
censored
confidence
```

### `event_cycle_membership`

Allows alternative cycle definitions for sensitivity analysis.

## 7.5 Path and mechanism tables

### `ami_event_paths`

At a minimum:

```text
event_id
ts_utc
seconds_from_anchor
mid_price
mark_price
return_bps
mfe_bps_so_far
mae_bps_so_far
drawdown_from_mfe_bps
new_high_flag
new_low_flag
reclaim_state
progress_state
path_quality
censor_flag
```

### `ami_microstructure_snapshots`

```text
event_id
ts_utc
venue
spread_bps
bid_depth
ask_depth
book_imbalance
pull_speed
refill_speed
queue_survival
OFI
CVD
taker_buy
taker_sell
impact_per_aggressive_dollar
data_quality
```

### `ami_derivatives_state`

```text
event_id
ts_utc
venue
open_interest
funding
basis
liquidation_buy
liquidation_sell
options_context
source_quality
```

### `ami_global_state`

Stores multi-timeframe global market state, session and scheduled boundary context.

## 7.6 Decision, trade and execution tables

### `signal_instances`

A signal instance must identify route version and feature availability time.

### `paper_trades`

Retain original rows and map them to canonical event/cycle IDs.

### `shadow_trades`

Separate decision observation from hypothetical execution.

### `forward_observations`

Stores frozen forward protocol observations, including no-signal periods.

### `live_orders`

Stores order requests, acknowledgements, rejects and cancellations.

### `live_fills`

Stores fill price, quantity, fee and execution venue.

### `positions`

```text
position_id
route_version
side
quantity
average_entry
opened_at
closed_at
position_age
realized_pnl
unrealized_pnl
```

### `position_decisions`

```text
decision_id
event_id
position_id
position_state
available_actions
chosen_action
confidence
abstain_reason
expected_action_values
realized_regret
```

## 7.7 Results tables

### `experiment_results`

One row per exact test cell, not one row per narrative claim.

### `route_metrics`

Must include:

```text
n_rows
n_unique_events
n_unique_cycles
n_days
wins
losses
wr
mean_bps
median_bps
sum_bps
mdd_bps
max_losing_streak
best_day
worst_day
top_3_days_share
cluster_ci_low
cluster_ci_high
real_fill_n
missed_fill_n
forward_n
```

### `question_registry`

Include every numbered question, its family, status, evidence and missing-data requirement.

## 7.8 SQL integrity requirements

The final build must produce:

- `PRAGMA integrity_check = ok`;
- source-to-canonical row reconciliation;
- orphan-key report;
- duplicate-key report;
- timestamp coverage report;
- null-coverage report;
- SQL dump re-import test;
- table hash manifest;
- schema migration log.

---

# 8. Event Identity, Deduplication and Overlap

## 8.1 Do not treat ledger rows as independent events

Current known scale:

```text
748 paper/shadow rows
→ 479 unique event anchors
→ approximately 118 episodes under a 6h gap sensitivity proxy
```

The 6h proxy is not a canonical cycle definition. It is a sensitivity analysis until structural cycle IDs exist.

## 8.2 Required identities

Every observation should have:

```text
raw_record_id
event_id
cycle_id
signal_instance_id
policy_id
route_version
ledger_id
position_id
order_id
fill_id
```

## 8.3 Direction conflicts

Current known overlap:

```text
124 / 479 anchors carry both LONG and SHORT labels.
```

The build must not automatically call this an error. It may represent:

- different horizons;
- different policy versions;
- exit information vs entry information;
- state transition;
- implementation inconsistency;
- true conflict requiring WAIT.

The dashboard must show conflict type and resolution status.

## 8.4 Cooldown views

Produce sensitivity views for at least:

```text
1h
2h
4h
6h
12h
24h
```

But label them as **cooldown sensitivity**, not true independent cycle counts.

---

# 9. Paper, Shadow, Forward and Live Taxonomy

## 9.1 Historical backtest

Uses historical data and may have been designed after seeing part of that history.

## 9.2 Replay

Runs frozen logic over historical market streams with known limitations.

## 9.3 Paper

Generates decisions in real time or replay but does not transmit real orders.

## 9.4 Shadow

Observes live market conditions and records what the strategy would have done. Must record whether the route was frozen before the observation.

## 9.5 Forward shadow

A frozen preregistered route observed only after the freeze timestamp. Must include eligible no-trade periods.

## 9.6 Forward paper

Frozen forward decisions with simulated order mechanics and explicit fill model.

## 9.7 Live observation

A route influences monitoring or operator information but not orders.

## 9.8 Live execution

Real orders and fills. Must be backed by exchange evidence, not a config label.

## 9.9 Required forward record

```yaml
forward_protocol_id:
route_version:
frozen_at:
forward_start:
forward_end:
required_n:
current_n:
eligible_events:
signals:
no_signal_events:
orders_attempted:
fills:
missed_fills:
current_verdict:
claim_permission:
```

Current state that must be preserved until newer evidence exists:

```text
E-HOUR17 + E-CONVCOMP binding VALID
forward n = 0 / 20
there is no forward performance claim yet
```

---

# 10. Current Canonical Scientific State to Seed

The build must ingest newer artifacts if they exist. In their absence, preserve the following 2026-07-03 state.

## 10.1 AMI Phase 6A-R2 Risk/Applicability

```text
Experiment: E-RISKAPP-6AR2-001
Verdict: FALSIFIES / INSUFFICIENT_SAMPLE
```

Key conclusions:

- regime+latent risk reduction was not supported under frequency-normalized controls;
- the candidate overlay did not generate usable selection under regime shift;
- the earlier N=14, MDD −416 cell was not reproduced under honest per-fold refit;
- drift alarm saturated and became applicability-degenerate;
- no live/shadow/route/config changes resulted.

## 10.2 C-BUY-FADE Structural

```text
Experiment: E-BUYFADE-STRUCT-001
Verdict: FALSIFIES, with a silence-information exception
```

Key conclusions:

- broad route ALL was historically negative across train/validation/untouched-like segments;
- small shadow positivity did not generalize;
- silence contains real information but becomes known only around T+30;
- silence is therefore not an entry alpha;
- timing, genesis, management and horizon variants did not rescue the route;
- reclaim frequency alone is not equivalent to a profitable trade.

Special statistical warning:

```text
mc_p = 0.010 was a hard floor from a 100-permutation base.
It must not be presented as precision beyond the permutation design.
```

## 10.3 BUY-FADE Re-entry

```text
Experiment: E-BUYFADE-REENTRY-001
Verdict: FALSIFIES / null confirmed
```

- SHORT→SHORT re-entry was incrementally negative across cooldowns;
- random timing was not beaten;
- interpretation: churn;
- SHORT→LONG validation sample was insufficient;
- BAD_TIMING-stop re-entry is only a new preregistration candidate, not accepted alpha.

## 10.4 Silence-Conditional Exit Timing

```text
Experiment: E-BUYFADE-SILEXIT-001
Verdict: REJECTED[econ] + T45_EXIT_ROBUST
```

- most gain occurred from T0 to T+30;
- T+30 median unrealized gain was positive, but entering at T+30 was negative;
- `bd_first_buy50` passed many criteria but failed the frozen economic threshold;
- it also worked in noisy controls, weakening silence-specific mechanism claims;
- T+45 remained a robust exit region;
- survivor audit must include pre-T30 stopped trades.

## 10.5 Paper/shadow aggregate results

Current result warehouse contains strong-looking paper/shadow statistics, but their interpretation must remain governed.

### `SHORT_NEITHER`

- positive in both realtime and state-machine implementations;
- remains positive under cooldown and clustered views;
- currently the strongest of the three broad paper/shadow labels;
- nevertheless requires exact route-version, overlap and forward-evidence separation.

### `SHORT_NOISY`

- positive in aggregate and 6h cooldown views;
- broader and more regime-dependent;
- raw label is not automatically a deployable frozen policy;
- overlap with `SHORT_NEITHER` must be removed or portfolio-policy controlled.

### `LONG_SILENCE`

- realtime implementation positive;
- state-machine implementation negative;
- silence is late information;
- not a portable standalone entry alpha;
- may be retained as post-event information or management context only where feature timing is explicit.

## 10.6 Question audit

Current whitepaper provides explicit Q396–Q730 text:

```text
335 explicit questions
40 directly re-tested
82 answered mainly by existing result synthesis
162 partially testable by proxy
51 blocked by missing raw/path/decision data
```

The whitepaper references a broader Q001–Q866 system, but missing explicit question text must not be fabricated. Create placeholder registry rows only if clearly marked as unexpanded ranges.

---

# 11. Reproducibility Workflow

For every reproducible experiment:

1. locate the report;
2. locate the generating code;
3. locate the exact dataset or reconstruct its query;
4. identify code commit and dependencies;
5. identify frozen parameters;
6. rerun in isolation;
7. compare row counts and metrics;
8. classify differences;
9. register reproducibility verdict;
10. link output artifacts.

Reproducibility verdicts:

```text
EXACT
NUMERICALLY_EQUIVALENT
PARTIAL
FAILED_MISSING_DATA
FAILED_MISSING_CODE
FAILED_ENVIRONMENT
FAILED_RESULT_MISMATCH
NOT_APPLICABLE_NARRATIVE
```

A stored result that cannot be rerun may still be evidence, but its reproducibility confidence must be lower.

---

# 12. Statistical and Scientific Validation

## 12.1 Mandatory denominators

Every metric table must show where relevant:

```text
row N
unique event N
unique cycle N
active day N
filled N
eligible N
forward N
```

## 12.2 Dependence-aware uncertainty

Use clustered/bootstrap views by:

- event;
- structural cycle;
- day;
- week;
- regime.

Do not rely only on IID trade bootstraps.

## 12.3 Chronological validation

Distinguish:

```text
train
validation
untouched holdout
forward
```

An equal-row chronological half is a lifecycle sensitivity analysis, not automatically a true untouched holdout.

## 12.4 Multiple testing

Track:

- hypotheses tried;
- thresholds viewed;
- route variants;
- target variants;
- reports generated;
- reused splits;
- researcher exposure before freeze.

Results from the same dataset/target/feature family belong to one evidence family unless independence is demonstrated.

## 12.5 Negative controls

Where possible, include:

- opposite direction;
- random time;
- time-shifted event;
- same hour/regime non-event;
- matched-control event;
- label permutation;
- mechanism-specific control.

## 12.6 Economic gates

Statistical significance may not override frozen economic thresholds.

A candidate that passes 8/9 checks but fails the frozen economic requirement remains rejected.

## 12.7 Lookahead and feature availability

For every feature, record:

```text
observation timestamp
computation timestamp
availability timestamp
decision timestamp
```

A feature that is only knowable at T+30 may not be used for a T0 entry claim.

---

# 13. Excel Workbook Construction

Generate a polished workbook named, for example:

```text
AMI_S34_CANONICAL_RESEARCH_DASHBOARD_v<version>.xlsx
```

It must be source-linked, filterable and readable without opening SQL.

## 13.1 Required sheets

### 1. `Executive Dashboard`

Show:

- latest build time;
- canonical data end date;
- source counts;
- current live/paper/shadow/forward counts;
- accepted/rejected/blocked knowledge counts;
- current forward N versus required N;
- data-health status;
- top unresolved conflicts;
- operator decisions waiting.

### 2. `Canonical Alpha Registry`

Columns:

```text
alpha_id
route_version
direction
universe
exact rule
status
permission
evidence level
historical N
forward N
live N
mean/median
cycle-adjusted CI
latest verdict
supersession
```

### 3. `Paper Shadow Forward Live`

Display evidence layers side by side without summing them.

### 4. `Forward Observatory`

Include:

- protocol ID;
- frozen date;
- current N/required N;
- events since freeze;
- signals;
- no-signals;
- fills/missed fills;
- no-claim indicator.

### 5. `Question Matrix`

Every explicit question with status, evidence, answer, missing data and next experiment.

### 6. `Experiment Registry`

All experiments, frozen gates, verdicts and report links.

### 7. `Knowledge Objects`

Claims, confidence components, contradictions, permissions and expiration.

### 8. `Event Cycle Analysis`

Rows vs events vs cycles, overlap, cooldown sensitivities and conflict patterns.

### 9. `Execution and Fill`

Only real or modelled execution metrics, clearly labeled.

### 10. `Sequence and Concentration Risk`

- losing streaks;
- daily concentration;
- top-3-day share;
- autocorrelation;
- month/regime dependence.

### 11. `Data Coverage`

Raw and derived data availability by table/field/date.

### 12. `Data Quality`

Gaps, stale feeds, schema problems, reconstructed fields and current health.

### 13. `Contradictions`

Implementation, regime, version and evidence conflicts.

### 14. `Supersession Graph`

Old finding → new experiment → current verdict.

### 15. `SQL Audit`

Row counts, schemas, hashes and import verification.

### 16. `Artifact Inventory`

Every source file with canonical status.

### 17. `Definitions`

Explain bps, ledger type, evidence layer, cycle, forward, live and permission terms.

## 13.2 Workbook design rules

- freeze headers;
- enable filters;
- use consistent status colors;
- use conditional formatting for rejected, blocked and stale items;
- avoid merged cells inside data tables;
- include chart titles and units;
- expose source identifiers;
- never hide material caveats in comments only;
- include a `READ ME` area.

## 13.3 Formula and data validation

- no formula errors;
- no hidden hardcoded p-values without a note;
- percentages use one scale consistently;
- bps and percent must not be mixed;
- totals must avoid double counting across cooldown alternatives;
- same-event overlapping routes must not be summed as independent portfolio PnL.

---

# 14. Word Report Construction

Generate current Word reports from the canonical database, not by manually copying old prose.

Recommended outputs:

```text
AMI_S34_EXECUTIVE_RESEARCH_REPORT_<date>.docx
AMI_S34_CANONICAL_SCIENTIFIC_REPORT_<date>.docx
AMI_S34_FORWARD_VALIDATION_REPORT_<date>.docx
AMI_S34_DATA_QUALITY_AND_LINEAGE_REPORT_<date>.docx
AMI_S34_CONTRADICTION_AND_SUPERSESSION_REPORT_<date>.docx
```

## 14.1 Executive report structure

1. Executive verdict
2. What changed since prior build
3. Current alpha permissions
4. Paper/shadow/forward/live summary
5. Strongest evidence
6. Rejected and demoted claims
7. Data health
8. Risk and concentration
9. Operator decisions
10. Next data/experiment priorities

## 14.2 Scientific report structure

1. Scope and frozen date
2. Source inventory
3. Data lineage
4. Event/cycle definitions
5. Experiment registry
6. Statistical methods
7. Results by alpha family
8. Execution and fill analysis
9. Forward evidence
10. Negative controls
11. Contradictions
12. Limitations
13. Knowledge object updates
14. Appendices with exact tables

## 14.3 Word update rule

The newest Word report must be regenerated from current structured data. Do not edit an old report in place if that would obscure which results changed.

Use immutable versioned filenames and a stable `LATEST` pointer or manifest entry.

---

# 15. Markdown and Whitepaper Updates

## 15.1 Canonical whitepaper

Update the AMI Artificial Market Intelligence whitepaper only with findings that pass its own epistemic rules.

Do not import the Advanced Metering Infrastructure Word document into the canonical market whitepaper except in a clearly labeled external analogy section.

## 15.2 Required Markdown outputs

```text
AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_<version>.md
AMI_S34_SYSTEM_STATE_LATEST.md
AMI_S34_SESSION_RESULT_<date>.md
AMI_S34_FULL_QUESTION_AUDIT_<version>.md
AMI_S34_BUILD_CHANGELOG.md
AMI_S34_OPERATOR_DECISIONS.md
AMI_S34_RESEARCH_BACKLOG.md
```

## 15.3 Whitepaper update rules

- update document control and date;
- preserve previous version history;
- distinguish constitutional specification from empirical S34 findings;
- link every empirical claim to a report/experiment ID;
- mark evidence level and current permission;
- remove no history; use supersession notes;
- do not promote paper evidence to live language.

---

# 16. Current Question System and Expansion

The current explicit registry covers Q396–Q730.

The build must:

1. preserve all 335 questions verbatim;
2. retain current status and evidence;
3. update answers when newer canonical evidence exists;
4. identify which questions changed and why;
5. track prior answer versions;
6. not fabricate missing Q001–Q395 or Q731–Q866 text;
7. create expansion placeholders only when labeled `QUESTION_TEXT_NOT_YET_CANONICAL`.

Each question row should include:

```text
qid
family
question
current_status
current_answer
confidence
evidence_ids
last_changed_at
change_reason
missing_data
next_experiment
permission_impact
```

---

# 17. Missing Data and Future Collector Construction

The canonical build must generate a machine-readable missing-data roadmap.

Priority collectors:

## 17.1 Event and cycle identity

Collect:

```text
event_id
cycle_id
parent_event_id
episode start/end
overlap flags
censor flags
state age
```

## 17.2 Price paths and lifecycle

Collect all eligible events and controls:

```text
T-60m to T+24h path
MFE/MAE
MFE/MAE timestamps
first reclaim
first higher-low/lower-high
time under water
drawdown from MFE
```

## 17.3 Order book and flow

Collect synchronized:

```text
spread
depth
book imbalance
pull/refill
OFI
CVD
taker flow
impact per dollar
venue
```

## 17.4 OI, funding and basis

Collect paths, not only snapshots.

## 17.5 Position-aware decisions

Collect:

```text
FLAT / ALREADY_LONG / ALREADY_SHORT
position age
cost basis
unrealized PnL
candidate actions
chosen action
confidence
regret
```

## 17.6 Execution telemetry

Collect:

```text
signal timestamp
feature availability timestamp
order timestamp
ack timestamp
fill timestamp
fill quantity
fee
slippage
reject reason
partial fills
```

## 17.7 Probability and abstention

Collect predicted action probabilities, calibration bin, OOD score, novelty and abstain reason.

## 17.8 All-timestamp candidate universe

To test unconditional LONG genesis and selection bias, persist eligible candidates even when no event or trade occurs.

---

# 18. Build Pipeline

The agent should execute the following stages in order.

## Stage 0 — Safety snapshot

- detect repository root;
- record Git status;
- create read-only source manifest;
- copy no secrets;
- confirm no live mutation mode;
- create build ID.

## Stage 1 — Discovery

- recursively enumerate artifacts;
- hash files;
- classify formats and domains;
- identify archives;
- create artifact registry.

## Stage 2 — Parsing

- parse each format using format-specific rules;
- quarantine failures;
- record parser version;
- extract metadata and references.

## Stage 3 — Canonical resolution

- build semantic groups;
- detect duplicates;
- build supersession graph;
- detect contradictions;
- identify latest canonical components.

## Stage 4 — SQL construction

- build/migrate canonical database;
- import source data losslessly;
- generate normalized tables;
- connect lineage;
- run integrity checks.

## Stage 5 — Reproduction

- rerun available analyses;
- compare metrics;
- record reproducibility verdicts;
- do not alter frozen rules.

## Stage 6 — Scientific synthesis

- update experiment and knowledge registries;
- update question answers;
- separate paper/shadow/forward/live evidence;
- recompute overlap, cycles and sequence risk;
- apply supersession and permission rules.

## Stage 7 — Artifact generation

Generate SQL, Excel, Word, Markdown, CSV manifests and optional PDFs.

## Stage 8 — Validation

- source reconciliation;
- formula scans;
- SQL re-import;
- cross-report metric consistency;
- link validation;
- visual inspection;
- namespace contamination scan;
- live-mutation diff check.

## Stage 9 — Publication

- write build manifest;
- publish versioned outputs;
- update `LATEST` pointers;
- preserve prior bundle;
- produce operator summary.

---

# 19. Output Bundle

Recommended structure:

```text
builds/
└── AMI_S34_<BUILD_ID>/
    ├── manifest/
    │   ├── BUILD_MANIFEST.json
    │   ├── ARTIFACT_REGISTRY.csv
    │   ├── FILE_HASHES.sha256
    │   ├── CANONICAL_SELECTION.csv
    │   └── NAMESPACE_COLLISIONS.csv
    ├── database/
    │   ├── AMI_S34_CANONICAL.db
    │   ├── AMI_S34_CANONICAL.sql
    │   ├── SCHEMA_DICTIONARY.md
    │   ├── TABLE_HASHES.csv
    │   └── MIGRATION_LOG.md
    ├── excel/
    │   └── AMI_S34_CANONICAL_RESEARCH_DASHBOARD_vX.xlsx
    ├── word/
    │   ├── AMI_S34_EXECUTIVE_RESEARCH_REPORT_<date>.docx
    │   ├── AMI_S34_CANONICAL_SCIENTIFIC_REPORT_<date>.docx
    │   └── AMI_S34_FORWARD_VALIDATION_REPORT_<date>.docx
    ├── markdown/
    │   ├── AMI_WHITEPAPER_<version>.md
    │   ├── SYSTEM_STATE_LATEST.md
    │   ├── SESSION_RESULT_<date>.md
    │   ├── FULL_QUESTION_AUDIT.md
    │   ├── CHANGELOG.md
    │   └── OPERATOR_DECISIONS.md
    ├── csv/
    │   ├── QUESTION_REGISTRY.csv
    │   ├── ALPHA_REGISTRY.csv
    │   ├── EXPERIMENT_REGISTRY.csv
    │   ├── KNOWLEDGE_OBJECTS.csv
    │   ├── FORWARD_OBSERVATORY.csv
    │   ├── DATA_COVERAGE.csv
    │   └── CONTRADICTIONS.csv
    ├── validation/
    │   ├── REPRODUCIBILITY_REPORT.md
    │   ├── SQL_INTEGRITY_REPORT.md
    │   ├── EXCEL_VALIDATION_REPORT.md
    │   ├── CROSS_ARTIFACT_CONSISTENCY.md
    │   └── LIVE_MUTATION_CHECK.md
    └── archive/
        └── superseded_and_rejected_index.md
```

---

# 20. Cross-Artifact Consistency Rules

The same metric may appear in SQL, Excel, Word and Markdown only if it originates from one canonical metric row or query.

The build must detect mismatches such as:

```text
Excel N differs from report N
Word mean differs from SQL mean
forward count differs between dashboard and ledger
route status differs between alpha registry and system state
same signal name maps to different rule versions
```

Every published metric should carry a stable metric ID where practical.

Example:

```text
METRIC-SHORT-NEITHER-6H-REALTIME-MEAN-BPS-v1
```

---

# 21. Change Detection

Every build must answer:

1. Which files are new?
2. Which files changed?
3. Which data periods extended?
4. Which experiments completed?
5. Which claims changed status?
6. Which alpha permissions changed?
7. Which question answers changed?
8. Which contradictions appeared or resolved?
9. Which forward counts increased?
10. Did any live file change?

Produce both human-readable and machine-readable changelogs.

## 21.1 Question answer change record

```yaml
qid:
old_status:
new_status:
old_answer_hash:
new_answer_hash:
change_reason:
new_evidence:
superseded_evidence:
permission_effect:
```

---

# 22. Operator Decision Queue

Do not silently decide discretionary operational questions.

Maintain a queue with:

```text
decision_id
question
current evidence
recommended options
risk of each option
required approval
status
operator response
implemented_at
```

Current examples to preserve until resolved:

- whether the historically negative BUY_FADE shadow route remains observation-only or is retired/evolved;
- whether `bd_first_buy50` is added only as an observation exit monitor;
- whether to preregister BAD_TIMING re-entry;
- whether to preregister 4h-DOWN + silence continuation;
- when OI-genesis becomes data-ready;
- whether to build adaptive-reference drift monitoring.

---

# 23. Quality Gates

A build is not publishable unless all applicable gates pass.

## 23.1 Data gates

- database integrity passes;
- SQL dump round-trip passes;
- source row reconciliation documented;
- timestamp coverage documented;
- no silent truncation;
- missing data labeled.

## 23.2 Scientific gates

- evidence layers separated;
- feature timing audited;
- superseded claims not shown as current;
- rejected claims retained;
- overlap/effective N shown;
- no unsupported live claims;
- no namespace contamination.

## 23.3 Excel gates

- no formula errors;
- filters and headers work;
- chart units correct;
- key caveats visible;
- totals avoid double count;
- source IDs present.

## 23.4 Word/Markdown gates

- document control correct;
- table values match SQL;
- latest verdicts used;
- citations/lineage available;
- no accidental terminology collision.

## 23.5 Operational gates

- live code/config diff is empty unless explicitly authorized;
- no credentials exported;
- no order actions performed;
- build is reproducible.

---

# 24. Definition of Done

The task is complete only when:

1. all discoverable artifacts are inventoried and hashed;
2. current and superseded artifacts are distinguished;
3. AMI acronym/domain collisions are quarantined;
4. canonical SQL DB and portable SQL dump are built and validated;
5. all paper, shadow, forward and live evidence layers are separated;
6. event/anchor/cycle overlap is represented;
7. experiment, alpha, knowledge and question registries are updated;
8. current 2026-07-03 null/reject results are preserved unless newer evidence supersedes them;
9. Excel dashboard is regenerated from canonical SQL;
10. Word and Markdown reports are regenerated from canonical data;
11. cross-artifact metrics are consistent;
12. a missing-data roadmap is produced;
13. an operator decision queue is produced;
14. no unauthorized live mutation occurred;
15. a complete build manifest and changelog are published.

---

# 25. Agent Execution Instruction

Use the following as the operational instruction when this specification is handed to an engineering agent:

> Read this entire specification first. Discover all artifacts recursively. Treat the repository and supplied folders as evidence sources, not as automatically consistent truth. Determine the newest scientifically canonical state by content, lineage, experiment status and data coverage. Preserve all history, rejected findings and contradictions. Build a canonical SQL warehouse, regenerate Excel and Word outputs, update Markdown/whitepaper state, and publish a versioned bundle with manifests and validation reports. Keep paper, shadow, forward and live evidence separate. Do not modify live routes, configuration, `.env`, order logic or running services. Do not merge documents that use AMI to mean Advanced Metering Infrastructure into the Artificial Market Intelligence knowledge base. If a result cannot be reproduced, retain it with a lower reproducibility status rather than inventing data. Report exactly what changed, what remains blocked, what is forward-only, and which operator decisions remain unresolved.

---

# Appendix A — Status Vocabulary

## Scientific verdicts

```text
SUPPORTED
PROVISIONALLY_SUPPORTED
REPLICATED
FALSIFIED
REJECTED_ECONOMIC
REJECTED_STATISTICAL
INSUFFICIENT_SAMPLE
INCONCLUSIVE
REGIME_LIMITED
EXECUTION_FAILED
DATA_INVALID
```

## Knowledge lifecycle

```text
OBSERVATION
PRELIMINARY
CHRONOLOGICALLY_VALIDATED
UNTOUCHED_HOLDOUT
INDEPENDENTLY_REPLICATED
FORWARD_VALIDATING
OPERATIONAL_CANDIDATE
PROVISIONALLY_ACCEPTED
DEMOTED
EXPIRED
REJECTED
```

## Artifact status

```text
CANONICAL_CURRENT
CANONICAL_COMPONENT
SUPERSEDED
HISTORICAL
CONFLICTING
QUARANTINED_NAMESPACE
DUPLICATE_EXACT
DUPLICATE_SEMANTIC
UNVERIFIED
```

## Evidence layer

```text
HISTORICAL
REPLAY
PAPER
SHADOW
FORWARD_SHADOW
FORWARD_PAPER
LIVE_OBSERVATION
LIVE_EXECUTION
POST_LIVE_AUDIT
```

---

# Appendix B — Build Manifest Template

```yaml
build_id:
built_at_utc:
project_root:
source_artifact_count:
source_total_bytes:
source_hash_manifest:
canonical_data_start:
canonical_data_end:
latest_experiment_completion:
latest_forward_observation:
canonical_whitepaper_version:
canonical_database_version:
canonical_dashboard_version:
canonical_report_version:
live_mutation_authorized: false
live_mutation_detected: false
namespace_collisions:
reproducibility_summary:
question_status_counts:
alpha_permission_counts:
open_contradictions:
operator_decisions_waiting:
outputs:
```

---

# Appendix C — Minimum Alpha Card

```yaml
alpha_id:
name:
route_version:
direction:
mechanism:
universe:
entry_rule:
feature_available_at:
hold_exit_rule:
execution_model:
cost_model:
historical_metrics:
cycle_adjusted_metrics:
forward_metrics:
live_metrics:
known_overlap:
known_contradictions:
latest_verdict:
evidence_level:
permission:
expiration_review:
supersedes:
```

---

# Appendix D — Minimum Session Report

```markdown
# Session Result — YYYY-MM-DD

## Scope
## Source changes
## Experiments completed
## Scientific verdicts
## Software test status
## Alpha permission changes
## Forward observatory status
## Data quality changes
## Contradictions and supersessions
## Live mutation check
## Operator decisions
## Files generated
```

---

# Appendix E — Interpretation Guardrails for Current S34 Results

1. `SHORT_NEITHER` paper/shadow positivity is meaningful evidence, but not proof of universal live profitability.
2. `SHORT_NOISY` is broader and regime-dependent; raw labels must not substitute for exact deployable rules.
3. `LONG_SILENCE` contains late information and is implementation-dependent; it is not a T0 entry alpha.
4. Summing realtime `LONG_SILENCE`, `SHORT_NEITHER` and `SHORT_NOISY` bps creates overlap and policy-mixing risk.
5. Cooldown rows are alternative scenarios and must never be added across 1h/2h/4h/6h/12h.
6. 748 ledger rows are not 748 independent opportunities.
7. Strong means can coexist with negative medians and concentrated winner days.
8. Small forward N supports no performance claim.
9. A live registry status must be checked against actual route version, permission and execution evidence.
10. The newest falsification or untouched result supersedes an older attractive in-sample result within the same claim scope.

---

# Appendix F — Immediate First Run Priorities

On the first execution of this protocol:

1. register and hash every current artifact;
2. quarantine the Advanced Metering Infrastructure Word document from the Artificial Market Intelligence domain;
3. validate `S34_ALL.db` against `S34_ALL.sql`;
4. import the current 335-question matrix;
5. register the 2026-07-03 preregistered experiment verdicts;
6. build exact route/ledger identities for `LONG_SILENCE`, `SHORT_NEITHER` and `SHORT_NOISY`;
7. build a forward observatory with `E-HOUR17` and `E-CONVCOMP` at `n=0/20` unless newer records exist;
8. regenerate Excel from SQL rather than editing the prior workbook;
9. generate new Word and Markdown reports from canonical SQL;
10. produce a final discrepancy report showing every value that differs from previous outputs.

---

**End of canonical build specification.**
