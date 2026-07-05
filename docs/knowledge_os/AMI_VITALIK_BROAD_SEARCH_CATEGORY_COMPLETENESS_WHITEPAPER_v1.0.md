# AMI Broad-Search, Category-Completeness & Knowledge-Artifact Architecture Whitepaper

**Version:** 1.0  
**Date:** 2026-07-05  
**Status:** Proposed repository architecture and implementation roadmap  
**Scope:** Artificial Market Intelligence (AMI), Eclipse Scalper research system, data governance, evidence architecture, operational knowledge, AI-agent search discipline  
**Primary idea:** A serious research system must broaden its search beyond conventional “documents” and include every artifact category capable of changing meaning, validity, safety, or implementation.

---

## 0. Executive Summary

The AMI project already has substantial research and infrastructure foundations:

- canonical SQL state,
- immutable experiments,
- independent-cycle inference,
- corrected path selection,
- known-at-safe feature rules,
- source-quality contracts,
- test-ground-truth discipline,
- paper/shadow/live separation,
- risk, recovery, execution and monitoring components,
- research reports, whitepapers and implementation ledgers.

The next major risk is not merely “missing another indicator.”

The larger risk is **category blindness**:

> searching only the kinds of documents we already expect to be relevant.

A search may include whitepapers and reports but miss:

- code that defines the real behavior,
- tests that encode the actual invariant,
- config that overrides the documentation,
- migrations that changed semantics,
- deleted or stashed collector logic,
- API changelogs,
- incident reports,
- GitHub issues and pull requests,
- runtime journals,
- gap registries,
- operational runbooks,
- screenshots or dashboards that reveal an actual failure mode.

AMI must therefore expand from a research repository into a **category-complete knowledge and evidence operating system**.

The system should know:

- what each entity means,
- where each field comes from,
- when it was knowable,
- which source regime produced it,
- whether source completeness was positively proven,
- which experiments and claims depend on it,
- what would invalidate those claims,
- which ideas failed,
- which ideas are blocked,
- which ideas should be retried later,
- which research results are allowed to influence paper, shadow, or live execution,
- which artifact categories were searched before a conclusion was made.

The central transformation is:

```text
Whitepaper
+ canonical ontology
+ data-source contracts
+ feature/event cards
+ claim-evidence registry
+ experiment dependency graph
+ architecture decisions
+ negative knowledge
+ unresolved-unknown registry
+ data-regime history
+ operational runbooks
+ threat model
+ external evidence index
+ AI search receipts
+ category-completeness audits
= AMI Knowledge and Research Operating System
```

---

# Part I — The Broad-Search Principle

## 1. The Core Interpretation

The broad-search principle means:

> Do not define the search universe too narrowly before relevance is evaluated.

For AMI, “document” must not mean only:

- Markdown,
- PDF,
- whitepaper,
- formal report,
- research paper.

A **knowledge-bearing artifact** is any durable object that can change:

- a definition,
- source interpretation,
- research population,
- known-at status,
- data-quality verdict,
- experiment validity,
- implementation behavior,
- live risk,
- execution behavior,
- recovery behavior,
- promotion permission.

File extension is irrelevant.

---

## 2. The Knowledge-Bearing Artifact Test

An artifact belongs in the AMI knowledge universe when at least one is true:

```text
It can change a definition.
It can change a source interpretation.
It can change a validity period.
It can alter an event identity.
It can alter a research population.
It can alter a split or independence rule.
It can invalidate an experiment.
It can change a claim status.
It can affect risk or execution behavior.
It can prove or disprove recovery.
It can explain an incident.
It can reveal a contradiction.
It can define an unresolved blocker.
It can change paper, shadow, or live permission.
```

---

## 3. Category Blindness

Category blindness occurs when a search, audit, AI agent, or human workflow excludes a relevant artifact class before relevance is assessed.

Examples:

```text
Search only reports
→ miss tests that encode the real invariant

Search only Markdown
→ miss collector semantics in source code

Search only current files
→ miss deleted writer code in Git history or stash

Search only academic research
→ miss exchange API changes and outage reports

Search only table schemas
→ miss restart and runtime behavior

Search only successful experiments
→ miss graveyard collisions and blocked branches

Search only research artifacts
→ miss risk, execution, recovery and live-operation knowledge

Search only filenames
→ miss behavioral equivalents using different terminology
```

Category blindness produces confident but incomplete conclusions.

---

## 4. Category Blindness as a Threat

```yaml
threat_id: THREAT-CATEGORY-BLINDNESS
category: RESEARCH_GOVERNANCE
description: >
  Relevant evidence is excluded because its artifact category is not searched,
  indexed or recognized as knowledge.
possible_effects:
  - false novelty
  - duplicated implementation
  - incorrect source semantics
  - repeated failed research
  - invalid known-at assumptions
  - silent data-regime mixing
  - unsafe live promotion
mitigation:
  - artifact-category registry
  - question-specific category checklist
  - category-completeness audit
  - contradiction registry
  - AI search receipt
status: ACTIVE
```

---

# Part II — AMI Artifact Universe

## 5. Internal Artifact Categories

### 5.1 Canonical state artifacts

- canonical SQLite tables,
- schemas,
- migrations,
- views,
- triggers,
- indexes,
- constraints,
- database fingerprints,
- hashes,
- backup manifests,
- schema-version records.

### 5.2 Research artifacts

- whitepapers,
- preregistrations,
- experiment reports,
- baseline studies,
- feature-frontier reports,
- sufficiency and coverage checks,
- comparison reports,
- failure analyses,
- reconciliation reports.

### 5.3 Source-code artifacts

- collectors,
- event detectors,
- feature builders,
- research modules,
- risk modules,
- execution engines,
- exchange adapters,
- state machines,
- recovery code,
- reconciliation code,
- serializers,
- CLI entry points.

### 5.4 Test artifacts

- unit tests,
- integration tests,
- regression tests,
- mutation tests,
- property tests,
- fixtures,
- synthetic data generators,
- collection commands,
- expected test counts,
- expected failure declarations.

Tests may be more authoritative than prose because they encode executable invariants.

### 5.5 Configuration artifacts

- YAML,
- JSON,
- TOML,
- INI,
- environment templates,
- strategy settings,
- risk settings,
- runtime flags,
- frozen parameter manifests,
- deployment config.

A runtime config may override a whitepaper claim.

### 5.6 Data-quality artifacts

- gap registries,
- collector-health tables,
- heartbeat records,
- source-coverage audits,
- repair manifests,
- archive manifests,
- source-quality contracts,
- data-version registries,
- ingestion-error logs.

### 5.7 Operational artifacts

- startup scripts,
- process-manager configuration,
- watchdogs,
- scheduled tasks,
- health-check code,
- alert routing,
- kill-switch scripts,
- backup and restore commands,
- incident procedures.

### 5.8 Runtime evidence

- JSONL journals,
- intent ledgers,
- event journals,
- order logs,
- fill logs,
- position snapshots,
- reconciliation outputs,
- crash dumps,
- process snapshots,
- paper/shadow/live comparison logs.

### 5.9 Change-history artifacts

- Git commits,
- tags,
- branches,
- deleted code,
- stashes,
- release notes,
- changelogs,
- migration history,
- deprecation records.

Historical code may be necessary to explain historical data.

### 5.10 Collaboration artifacts

- GitHub issues,
- pull requests,
- code-review comments,
- design discussions,
- operator approvals,
- handoff notes,
- scratchpads,
- session summaries.

### 5.11 Visual artifacts

- dashboards,
- charts,
- architecture diagrams,
- screenshots,
- monitoring panels,
- slide decks,
- annotated market examples.

Visual artifacts must be linked to structured definitions and provenance.

### 5.12 Security and safety artifacts

- threat models,
- incident reports,
- vulnerability notes,
- access-control policy,
- secret-handling policy,
- dependency advisories,
- exchange-key permission manifests.

---

## 6. External Artifact Categories

### 6.1 Official exchange documentation

- REST API docs,
- websocket docs,
- field definitions,
- rate-limit rules,
- maintenance notices,
- symbol-filter changes,
- delisting notices,
- incident reports,
- changelogs.

### 6.2 Official and primary code

- exchange SDKs,
- protocol repositories,
- official examples,
- client libraries,
- reference implementations.

### 6.3 GitHub collaboration material

- issue descriptions,
- PR descriptions,
- review threads,
- discussions,
- design proposals,
- release discussions.

### 6.4 Academic and research material

- papers,
- preprints,
- theses,
- conference proceedings,
- replication packages,
- supplementary appendices,
- datasets.

### 6.5 Industry technical material

- market-maker engineering posts,
- execution-quality studies,
- data-vendor methodology,
- broker methodology,
- audit reports,
- microstructure conference slides.

### 6.6 Standards, policy and governance

- RFCs,
- protocol proposals,
- governance proposals,
- committee notes,
- working-group minutes,
- regulator consultation responses,
- market-structure reports.

### 6.7 Failure and incident material

- outage postmortems,
- security advisories,
- API degradation reports,
- liquidation-engine incidents,
- market-event retrospectives.

### 6.8 Weakly indexed or ephemeral material

- HackMD,
- Notion,
- Google Docs,
- GitBook,
- IPFS,
- forum attachments,
- archived project sites,
- mailing lists,
- workshop notes,
- conference decks.

### 6.9 Educational material

- technical tutorials,
- implementation guides,
- lectures,
- transcripts.

Educational material may explain behavior but should not automatically be treated as normative.

---

# Part III — Artifact Category Registry

## 7. Required Registry

AMI should maintain:

```text
docs/knowledge_os/artifacts/ARTIFACT_CATEGORY_REGISTRY.yaml
```

Example:

```yaml
categories:
  - category_id: INTERNAL_TEST
    name: Executable test
    authority_class: EXECUTABLE_CONTRACT
    default_search_priority: CRITICAL
    can_define_invariant: true
    can_support_claim: true
    common_locations:
      - tests/
    common_extensions:
      - .py
    omission_risk:
      - incorrect belief about implemented behavior
      - regression-scope mismatch

  - category_id: EXTERNAL_EXCHANGE_CHANGELOG
    name: Official exchange changelog
    authority_class: PRIMARY_OFFICIAL
    default_search_priority: CRITICAL
    can_define_source_semantics: true
    can_invalidate_data_regime: true
```

---

## 8. Artifact Identity

```yaml
artifact_id: ART-RISK-MANAGER-001
category_id: INTERNAL_SOURCE_CODE
logical_name: risk_manager
physical_location: risk/risk_manager.py
content_sha256: ...
repository_commit: ...
effective_from: ...
effective_to: null
status: ACTIVE
```

File location is not artifact identity. A renamed file may remain the same logical artifact.

---

## 9. Artifact Lifecycle

```text
DISCOVERED
CLASSIFIED
INDEXED
PARSED
LINKED
REVIEWED
AUTHORITATIVE
SUPERSEDED
DEPRECATED
ARCHIVED
UNTRUSTED
```

Indexed does not mean trusted.

---

## 10. Authority Is Question-Specific

Examples:

```text
Current exchange field semantics:
official API docs > third-party blog

Historical collector behavior:
historical code + raw rows > current API docs

Intended experiment design:
preregistration > implementation guess

Enforced invariant:
database constraint + passing tests > prose

Incident chronology:
structured runtime journal > memory

Runtime risk policy:
effective configuration for that run > old documentation
```

No artifact category is universally authoritative.

---

# Part IV — Category-Completeness Audits

## 11. Core Audit Question

Before concluding:

- “this feature is new,”
- “this module is missing,”
- “this source is complete,”
- “this field means X,”
- “this component should be deleted,”
- “this experiment is ready,”

ask:

> Which artifact categories were searched, and which were not?

---

## 12. Audit Record

```yaml
audit_id: CATEGORY-AUDIT-LIQUIDATION-GEOMETRY-001
question: >
  What is the authoritative known-at-safe definition of cascade geometry?
required_categories:
  - research_reports
  - source_code
  - tests
  - schema
  - git_history
  - raw_source_evidence
searched_categories:
  - research_reports
  - source_code
  - tests
  - schema
  - git_history
  - raw_source_evidence
unsearched_critical_categories: []
verdict: CATEGORY_COMPLETE_FOR_SCOPE
```

---

## 13. Completeness Verdicts

```text
CATEGORY_COMPLETE_FOR_SCOPE
CATEGORY_SUFFICIENT_WITH_LIMITATIONS
CATEGORY_INCOMPLETE
CATEGORY_BLOCKED_BY_ACCESS
CATEGORY_BLOCKED_BY_MISSING_HISTORY
CATEGORY_NOT_APPLICABLE
```

Completeness is always scoped to a question.

---

## 14. Category Coverage Metrics

Recommended diagnostics:

- required category count,
- searched category count,
- category count with relevant hits,
- reviewed artifact count,
- authoritative artifact count,
- conflict count,
- inaccessible category count,
- stale artifact count,
- missing metadata count.

```text
category_coverage =
searched_required_categories / required_categories
```

This is a search-process diagnostic, not a truth score.

---

## 15. Category Debt

Category debt means an entire knowledge class is missing.

```yaml
category_debt_id: DEBT-RECLAIM-IDENTITY-001
subject: RECLAIM transition
missing_categories:
  - canonical event definition
  - detector implementation
  - level-interaction contract
  - negative examples
  - tests
effect:
  - event family blocked by identity
priority: HIGH
```

Category debt differs from technical debt:

```text
Technical debt:
existing implementation requires improvement

Category debt:
a necessary class of knowledge does not exist
```

---

# Part V — Search Protocol

## 16. Required-Category Planning

Before search, define the expected evidence categories.

Example question:

```text
“Does the current risk engine support restart-safe daily-loss limits?”
```

Required categories:

- risk source code,
- persistence schema,
- recovery code,
- configuration,
- tests,
- runtime evidence,
- incident history,
- documentation,
- Git history.

Searching only `risk_manager.py` is insufficient.

---

## 17. Internal Search Universe

Important repository questions should consider:

```text
*.md
*.txt
*.py
*.sql
*.yaml / *.yml
*.json
*.toml
*.ini
*.ps1 / *.sh / *.bat
migrations/
tests/
reports/
config/
runtime journals
Git history
deleted files
stashes where relevant
```

---

## 18. Search by Terminology and Behavior

Terminology search alone is insufficient.

Example:

```text
Term:
“kill switch”

Behavioral equivalents:
disable entry
cancel order
close position
set permission false
trip circuit breaker
clamp exposure
```

A component may perform the function without using the expected name.

---

## 19. Semantic Alias Registry

```yaml
concept: independent_cycle
aliases:
  - opportunity group
  - dependency group
  - dedup group
forbidden_equivalences:
  - signal
  - trade
```

Aliases improve retrieval. Forbidden equivalences prevent concept collapse.

---

## 20. Search by Consequence

To locate risk logic, search for:

- order rejection,
- entry veto,
- max quantity,
- drawdown,
- equity,
- leverage cap,
- daily loss,
- exposure,
- position cancellation,
- permission denial.

To locate recovery logic, search for:

- replay,
- reconcile,
- persisted state,
- bootstrap,
- restore,
- orphan order,
- startup ownership,
- crash,
- restart.

---

## 21. External Search Broadening

Do not require the exact project name or expected term in the title.

Relevant material may discuss:

- derivatives state,
- order-flow imbalance,
- market impact,
- execution quality,
- source semantics,
- exchange microstructure,
- state recovery,
- fault tolerance,
- risk budgeting,
- event sourcing.

---

# Part VI — Discovery and Ingestion Pipeline

## 22. Pipeline

```text
Question
→ required-category plan
→ internal and external discovery
→ artifact classification
→ content hashing
→ version identification
→ authority assessment
→ ontology linking
→ claim extraction
→ contradiction detection
→ dependency update
→ unknown-gap registration
```

---

## 23. Duplicate and Version Handling

Do not deduplicate aggressively by filename or title.

Distinguish:

- byte duplicate,
- mirror,
- translated copy,
- revised version,
- fork,
- excerpt,
- partial copy,
- generated summary,
- historical version.

Historical versions may be necessary to reproduce old data.

---

## 24. Section-Level Evidence

A document may contain multiple authority classes.

Example:

```text
API specification:
field definition → normative
tutorial example → explanatory
performance estimate → non-binding
```

Store section-level pointers:

```yaml
artifact_id: ART-BINANCE-MARKPRICE-DOC
section_locator: "Mark Price Stream / field r"
evidence_role: SOURCE_SEMANTICS
```

---

## 25. Preserve Conditions and Exceptions

AI extraction must preserve:

- time range,
- market,
- endpoint,
- stream mode,
- exceptions,
- uncertainty,
- deprecation,
- version.

It must not flatten:

```text
“complete under all-market stream after 2026-06-06”
```

into:

```text
“liquidation data is complete”
```

---

# Part VII — Canonical Ontology

## 26. Why the Ontology Is Essential

AMI uses related but distinct concepts:

```text
event
source event
anchor
bucket
cluster
running cluster
terminal cluster
signal
setup
route
lane
independent cycle
observation
outcome
experiment
claim
order
position
```

These must not be allowed to drift between code, SQL, reports and AI prompts.

---

## 27. Initial Canonical Definitions

### Raw source record

An immutable observation from an external or internal producer.

### Source event

A canonical market event from which signals may derive.

### Anchor

The earliest deterministic timestamp at which an event definition becomes true.

### Bucket

A deterministic computational aggregation interval.

### Running cluster

The portion of a cluster observable up to a specific known-at timestamp.

### Terminal cluster

A completed cluster requiring future information relative to earlier timestamps.

### Signal

A canonical candidate born at `signal_birth_ts`.

### Lane

A strategy or rule variant that may observe the same market event.

### Route

The rule path by which a signal was created.

### Independent cycle

The inferential unit grouping non-independent signals from the same opportunity.

### Observation

A measured state or path attached to a signal and horizon.

### Outcome

A future-dependent label.

### Experiment

An immutable preregistered analysis.

### Claim

A proposition with evidence, scope, status and invalidation rules.

### Order

An exchange instruction.

### Position

A live or simulated inventory state.

A signal is not an order.

---

## 28. Ontology Record Template

```yaml
entity_name: independent_cycle
definition_version: independent-cycle-v1
canonical_definition: >
  A grouping of signals that arise from one underlying market opportunity and
  must not be counted as independent inferential evidence.
primary_key: independent_cycle_id
relationships:
  contains:
    - signal
invariants:
  - no train/test straddling
  - one cycle counts once per comparison
must_not_be_confused_with:
  - signal
  - trade
status: ACTIVE
```

---

# Part VIII — Data-Source Contracts

## 29. Required Questions

Every source contract must answer:

- What is measured?
- Which endpoint or stream produces it?
- What does the timestamp mean?
- What is the update cadence?
- What does silence mean?
- What do duplicates mean?
- What source modes existed historically?
- How is completeness proven?
- What known-at lookup is allowed?
- Which features and experiments depend on it?

---

## 30. Source Contract Template

```yaml
source_contract_id: liquidation-stream-binance-usdtm-v2
source_name: Binance USDT-M liquidation stream
table: liquidations
market: futures
stream_modes:
  current: "!forceOrder@arr"
  historical:
    - per-symbol "@forceOrder"
timestamp_semantics:
  source_ts: exchange event time
  ingestion_ts: collector write time
silence_semantics: ambiguous unless positive completeness proof exists
completeness_contract:
  version: liquidation-source-quality-contract-v2
known_at_contract:
  allowed_lookup: source_ts <= feature_available_ts
dependent_features:
  - birth-truncated cascade geometry
status: ACTIVE
```

---

## 31. Source Quality Taxonomy

```text
SOURCE_COMPLETE
SOURCE_GAPPED
SOURCE_COVERAGE_UNRESOLVED
SOURCE_STALE
SOURCE_SEMANTICS_UNRESOLVED
SOURCE_MODE_DEPRECATED
SOURCE_NOT_APPLICABLE
```

Field-level quality is preferred when fields require different source windows.

---

# Part IX — Feature and Event Cards

## 32. Feature Card

```yaml
feature_id: running_liq_count
feature_definition_version: s34-knowable-anchor-continuation-v1-birth-truncated
description: >
  Number of liquidation source rows observed from reconstructed bucket start
  through signal birth.
entity_scope: signal
source_tables:
  - liquidations
feature_available_ts: signal_birth_ts
known_at_classification: KNOWN_AT_SAFE
source_window:
  start: reconstructed_bucket_start
  end: signal_birth_ts
quality_contract:
  version: liquidation-source-quality-contract-v2
provenance:
  implementation: tools/research_s34_knowable_anchor_continuation.py
forbidden_uses:
  - terminal event_count substitution
status: ACTIVE
```

---

## 33. Event Card

Every event family must define:

- event identity,
- birth timestamp,
- end timestamp,
- membership rule,
- deduplication,
- source data,
- known-at boundary,
- overlap behavior,
- cycle mapping,
- research status,
- live permission.

Required future cards include:

- liquidation anchor,
- RECLAIM transition,
- failed breakout,
- compression release,
- funding-state transition,
- OI anomaly,
- order-flow reversal,
- multi-timeframe entry trigger.

---

# Part X — Claim–Evidence Registry

## 34. Claims Are Not Reports

A report is human-readable narrative.

A claim is a machine-addressable statement with:

- scope,
- status,
- evidence,
- data version,
- feature version,
- support level,
- invalidation rule.

---

## 35. Claim Template

```yaml
claim_id: CLAIM-LONG-TIMING-STABILITY-001
statement: >
  LONG timing distributions are chronologically stable across frozen horizons
  under corrected data and cycle-grouped inference.
scope:
  population: corrected canonical LONG
  data_version: path-v2-candle-repair-r1
evidence:
  experiments:
    - E-W8-LONG-TIMING-STRUCTURE-002-CANDLE-REPAIR-CYCLE-GROUPED
status: SUPPORTED_BASELINE
support_levels:
  descriptive: true
  inferential: true
  independent_replication: false
  forward_validation: false
  economic_validation: false
  live_validation: false
not_alpha: true
invalidation_rules:
  - timing definition changes
  - path definition changes
  - independent-cycle definition changes
```

---

## 36. Claim Status Taxonomy

```text
SUPPORTED
SUPPORTED_BASELINE
SUPPORTED_DESCRIPTIVE
CONTRADICTED
MIXED
INSUFFICIENT_SAMPLE
BLOCKED_BY_DATA
BLOCKED_BY_SOURCE_QUALITY
BLOCKED_BY_KNOWN_AT
BLOCKED_BY_IDENTITY
BLOCKED_BY_SEMANTICS
SUPERSEDED
RETIRED
UNKNOWN
```

---

# Part XI — Experiment Dependency Graph

## 37. Purpose

The graph answers:

> If this source, definition, migration or data version changes, what must be reviewed?

---

## 38. Node Types

```text
ARTIFACT
SOURCE
SOURCE_CONTRACT
DATA_REGIME
DATA_VERSION
MIGRATION
ENTITY_DEFINITION
FEATURE_DEFINITION
EVENT_DEFINITION
POPULATION
SPLIT_MANIFEST
EXPERIMENT
CLAIM
REPORT
TEST_CONTRACT
LIVE_RULE
```

---

## 39. Edge Types

```text
DERIVED_FROM
DEPENDS_ON
SUPERSEDES
CORRECTS
INVALIDATES
REQUIRES_REVIEW
PAIRED_WITH
USES_SPLIT_FROM
USES_FEATURE_VERSION
USES_DATA_VERSION
SUPPORTS
BLOCKED_BY
PROMOTES_TO
```

---

## 40. Automatic Impact Examples

```text
Source timestamp semantics change
→ review dependent features
→ review experiments
→ mark claims REQUIRES_REVIEW

Independent-cycle definition changes
→ invalidate old split manifests
→ block direct comparison

Path definition changes
→ preserve old experiments
→ create corrected-data experiment IDs
→ never overwrite historical results
```

---

# Part XII — Architecture Decision Records

## 41. Required ADR Set

```text
ADR-001 Independent Cycle as Evidence Unit
ADR-002 MIN_BUCKET_N = 20
ADR-003 Effective Path Selection
ADR-004 Immutable Experiments
ADR-005 Liquidation Source Quality Contract V2
ADR-006 Field-Level Known-At and Quality
ADR-007 Canonical Regression Scope
ADR-008 Research–Execution Separation
ADR-009 Broad Artifact Universe
ADR-010 Category-Completeness Audit Requirement
```

Each ADR must include:

- context,
- decision,
- alternatives,
- consequences,
- risks,
- review trigger,
- affected modules and documents.

---

# Part XIII — Negative Knowledge and Unknowns

## 42. Failure Is Not One State

```text
REJECTED_BY_EVIDENCE
INSUFFICIENT_SAMPLE
BLOCKED_BY_DATA
BLOCKED_BY_SOURCE_QUALITY
BLOCKED_BY_KNOWN_AT
BLOCKED_BY_IDENTITY
BLOCKED_BY_SEMANTICS
DEPRECATED_DATA_REGIME
SUPERSEDED
RETIRED
RETRY_WHEN_CONDITION_MET
DESCRIPTIVE_ONLY
```

---

## 43. Retryable Branch Example

```yaml
item_id: FUNDING-STATE-LONG-001
status: RETRY_WHEN_CONDITION_MET
reason: INSUFFICIENT_SAMPLE
definition_frozen: true
outcome_inspected: false
retry_condition:
  all:
    - funding_negative_test_cycles_gte: 20
    - funding_nonnegative_swing24h_train_cycles_gte: 20
```

---

## 44. Unknown Registry

Current-style unknowns:

- historical OI coverage,
- OFI sufficiency,
- CVD trade-level May gaps,
- RECLAIM identity,
- multi-timeframe trigger definitions,
- geometry source-quality readiness,
- funding sample readiness,
- live re-entry policy,
- external evidence archival policy,
- source-semantic change monitoring.

Unknown is a valid canonical state.

---

# Part XIV — Data-Regime Registry

## 45. Purpose

A table name may stay constant while its source regime changes.

Track:

- endpoint changes,
- stream-mode changes,
- collector rewrites,
- cadence changes,
- field-semantic changes,
- exchange API changes,
- repairs,
- backfills,
- gap-monitor changes.

---

## 46. Regime Record

```yaml
regime_id: liquidation-stream-all-market-v1
source: Binance USDT-M forceOrder
start_ts: 2026-06-06T00:00:00Z
mode: "!forceOrder@arr"
previous_mode: per-symbol forceOrder
comparability:
  with_previous: PARTIAL
quality_contract:
  completeness_provable: true
  proof_method: all-market cadence
status: ACTIVE
```

---

# Part XV — Contradiction Management

## 47. Contradiction Registry

```yaml
contradiction_id: CONTRADICTION-RISK-LIMIT-001
subject: daily loss limit
artifacts:
  - config/paper.yaml: 50 bps
  - legacy documentation: 3 percent
conflict_type: CONFIGURATION_VS_DOCUMENTATION
resolution_status: UNRESOLVED
operational_rule: >
  Effective runtime config governs that run; canonical risk policy remains
  unresolved until versioned.
```

---

## 48. Conflict Types

```text
PROSE_VS_CODE
CODE_VS_TEST
CONFIG_VS_CODE
SCHEMA_VS_WRITER
CURRENT_DOC_VS_HISTORICAL_DATA
SOURCE_DOC_VS_RAW_OBSERVATION
REPORT_VS_EXPERIMENT_LEDGER
TWO_CANONICAL_CANDIDATES
TEMPORAL_VERSION_CONFLICT
TERMINOLOGY_COLLISION
```

Never resolve conflicts by automatically choosing the newest file.

---

# Part XVI — Missing Artifacts and Orphans

## 49. Expected Artifact Set for a Data Source

```text
source contract
collector implementation
schema
quality contract
health monitor
tests
regime history
external primary evidence
dependent-feature list
incident history
```

Missing categories must create explicit findings.

---

## 50. Orphan Detection

Detect:

- experiment without preregistration,
- claim without evidence,
- feature without source contract,
- source contract without health test,
- migration without rollback proof,
- config value without ADR,
- report without experiment ID,
- live module without recovery runbook,
- test without documented invariant.

---

# Part XVII — Evidence Bundles

## 51. Bundle Structure

```text
BUNDLE-LIQUIDATION-SOURCE-QUALITY-V2/
  contract.yaml
  official_source_evidence.yaml
  collector_code_pointers.yaml
  raw_measurement_summary.json
  reconciliation_report.md
  tests.yaml
  ADR.md
  dependency_impact.yaml
```

A high-impact conclusion should not depend on one isolated report.

---

## 52. Required Bundle Types

- source-semantics bundle,
- migration bundle,
- experiment bundle,
- data-repair bundle,
- incident bundle,
- deprecation bundle,
- live-promotion bundle.

---

# Part XVIII — AI-Agent Search Contract

## 53. AI Obligations

Before a high-impact recommendation, an AI agent must:

1. identify the question type,
2. load the required category checklist,
3. search required categories,
4. report inaccessible categories,
5. identify temporal versions,
6. detect contradictions,
7. distinguish fact from inference,
8. prefer canonical state over chat memory,
9. stop when a critical category is missing.

---

## 54. Forbidden AI Behaviors

```text
Do not infer implementation from roadmap prose.
Do not infer runtime policy from default config.
Do not infer historical source semantics from current docs.
Do not infer completeness from table density.
Do not infer novelty without failure archives and old branches.
Do not infer live readiness from research support.
Do not infer absence from filename search.
Do not rebuild before auditing behavioral equivalents.
```

---

## 55. Search Receipt

```yaml
search_receipt_id: SR-...
question: ...
required_categories: []
searched_categories: []
unsearched_categories:
  - category: ...
    reason: ...
artifacts_reviewed: []
conflicts_found: []
missing_artifacts: []
temporal_cutoff: ...
limitations: ...
verdict: CATEGORY_SUFFICIENT_WITH_LIMITATIONS
```

---

## 56. Interrupted Session Rule

If methodology existed only in session memory and the session ends:

```text
STOP
→ recover code and tests
→ reconstruct plausible interpretations
→ reconcile differences
→ freeze the contract in repository artifacts
→ only then continue
```

The project already demonstrated why this rule is necessary.

---

# Part XIX — Production Architecture and Rebuild Principle

## 57. Existing Production Components Are Not Assumed Good or Bad

AMI already has partial or substantial components for:

- risk,
- recovery,
- state persistence,
- reconciliation,
- paper execution,
- shadow execution,
- live gating,
- monitoring,
- kill switches,
- exchange adapters.

They should not be blindly preserved.

They should also not be blindly deleted.

Every component must earn a disposition:

```text
KEEP
EXTEND
REFACTOR
REPLACE
DELETE
DUPLICATED
UNSAFE
MISSING
```

---

## 58. Selective Rebuild Principle

The correct future approach is:

```text
Inventory behavior
→ classify existing component
→ preserve proven invariants
→ build replacement beside old system where necessary
→ shadow compare
→ fault test
→ cut over
→ retire old component
```

This is safer than a full rewrite and cleaner than preserving everything.

---

## 59. Production Work Is Deferred, Not Forgotten

Current priority remains:

```text
research integrity
data semantics
known-at safety
event and feature identity
mechanism discovery
```

Later production work includes:

- TCA,
- OMS,
- risk engine hardening,
- portfolio allocation,
- recovery,
- monitoring,
- attribution,
- compounding,
- live promotion.

Production work should begin with an audit, not a rebuild command.

---

# Part XX — Live Operations

## 60. Research Does Not Grant Live Permission

Promotion chain:

```text
research feature
→ descriptive validity
→ inferential support
→ corrected-data consistency
→ forward-shadow validation
→ execution simulation
→ risk review
→ live permission
```

---

## 61. Multi-Lane and Multi-Threshold Behavior

```text
same source event
+ same independent cycle
+ multiple threshold crossings
→ may create multiple research signals
→ must not automatically create multiple live positions
```

Live contracts must define:

- one order right per cycle,
- upgrade vs new entry,
- re-entry,
- cooldown,
- lane priority,
- max position,
- duplicate prevention.

---

## 62. Required Runbooks

- `LIVE_PROMOTION_GATE.md`
- `ORDER_DEDUPLICATION_CONTRACT.md`
- `DATA_STALENESS_RESPONSE.md`
- `LIVE_INCIDENT_RUNBOOK.md`
- `KILL_SWITCH_RUNBOOK.md`
- `DISASTER_RECOVERY.md`
- `HUMAN_OVERRIDE_POLICY.md`

---

# Part XXI — Threat Model

## 63. Threat Categories

### Research

- lookahead,
- leakage,
- threshold fishing,
- multiple testing,
- non-independent samples,
- silent version mixing,
- post hoc cohorts,
- category blindness.

### Data

- silent stream failure,
- dropped messages,
- duplicates,
- wrong timestamps,
- semantic change,
- source-mode change,
- stale rows.

### Infrastructure

- clock drift,
- disk exhaustion,
- DB corruption,
- partial migration,
- crash,
- restart,
- split-brain ownership.

### Market

- spoofing,
- wash trading,
- fake depth,
- engineered liquidations.

### Operational

- duplicate orders,
- stale execution,
- re-entry loop,
- partial-fill inconsistency,
- risk-limit bypass.

---

# Part XXII — External Evidence Governance

## 64. Reliability Classes

```text
PRIMARY_OFFICIAL
PRIMARY_CODE
PRIMARY_RAW_DATA
SECONDARY_TECHNICAL
SECONDARY_ANALYSIS
ANECDOTAL
UNVERIFIED
DEPRECATED
```

---

## 65. External Evidence Record

```yaml
evidence_id: EXT-BINANCE-MARKPRICE-STREAM-001
category: EXCHANGE_DOCUMENTATION
publisher: Binance
retrieved_at: ...
source_version: ...
supports:
  - funding field semantics
reliability: PRIMARY_OFFICIAL
archive_hash: ...
review_due_at: ...
```

---

## 66. Freshness and Invalidation

Record:

- publication date,
- retrieval date,
- version,
- last update,
- archived hash,
- semantic-change risk,
- review due date.

If an exchange changelog changes a field:

```text
source contract → REQUIRES_REVIEW
dependent feature → REQUIRES_REVIEW
live use → may be suspended
```

---

# Part XXIII — Repository Architecture

## 67. Proposed Structure

```text
docs/
  knowledge_os/
    AMI_BROAD_SEARCH_CATEGORY_COMPLETENESS_WHITEPAPER.md

    artifacts/
      ARTIFACT_CATEGORY_REGISTRY.yaml
      ARTIFACT_UNIVERSE_INVENTORY.md
      ARTIFACT_LOCATION_MAP.yaml
      CATEGORY_GAP_REPORT.md
      SEARCH_RECEIPT_SCHEMA.yaml

    ontology/
      AMI_CANONICAL_ONTOLOGY.md
      ENTITY_RELATIONSHIP_MAP.yaml
      TERM_ALIAS_REGISTRY.yaml

    data_contracts/
      DATA_SOURCE_CONTRACT_REGISTRY.yaml
      FUNDING_RATE_CONTRACT.md
      LIQUIDATION_STREAM_CONTRACT.md
      MARK_PRICE_CONTRACT.md
      AGG_TRADES_CONTRACT.md
      BOOK_TICKER_CONTRACT.md
      OPEN_INTEREST_CONTRACT.md
      CANDLE_CONTRACT.md
      PATH_OBSERVATION_CONTRACT.md

    claims/
      CLAIM_EVIDENCE_REGISTRY.yaml
      CLAIM_STATUS_TAXONOMY.md

    dependencies/
      EXPERIMENT_DEPENDENCY_GRAPH.yaml
      DATA_CHANGE_IMPACT_RULES.yaml

    decisions/
      ADR-001-INDEPENDENT-CYCLE-UNIT.md
      ADR-002-MIN-BUCKET-N-20.md
      ADR-003-EFFECTIVE-PATH-SELECTION.md
      ADR-004-EXPERIMENT-IMMUTABILITY.md
      ADR-005-LIQUIDATION-QUALITY-CONTRACT-V2.md
      ADR-006-FIELD-LEVEL-KNOWN-AT.md
      ADR-007-CANONICAL-TEST-SCOPE.md
      ADR-008-RESEARCH-EXECUTION-SEPARATION.md
      ADR-009-BROAD-ARTIFACT-UNIVERSE.md

    negative_knowledge/
      NEGATIVE_KNOWLEDGE_REGISTRY.yaml
      RETRY_CONDITION_REGISTRY.yaml

    unknowns/
      UNKNOWN_UNRESOLVED_REGISTRY.yaml
      BLOCKER_REGISTRY.yaml

    regimes/
      DATA_REGIME_REGISTRY.yaml
      COLLECTOR_MODE_CHANGELOG.md

    conflicts/
      CONTRADICTION_REGISTRY.yaml

    evidence_bundles/
      ...

    live/
      LIVE_PROMOTION_GATE.md
      ORDER_DEDUPLICATION_CONTRACT.md
      DATA_STALENESS_RESPONSE.md
      LIVE_INCIDENT_RUNBOOK.md
      KILL_SWITCH_RUNBOOK.md
      DISASTER_RECOVERY.md
      HUMAN_OVERRIDE_POLICY.md

    threats/
      AMI_MARKET_DATA_RESEARCH_THREAT_MODEL.md

    handoff/
      SESSION_HANDOFF_CONTRACT.md
      CURRENT_CONTEXT_BUNDLE_TEMPLATE.md
```

---

# Part XXIV — Implementation Roadmap

## 68. Phase 0 — Read-Only Artifact Universe Inventory

Deliver:

1. `ARTIFACT_UNIVERSE_INVENTORY.md`
2. `ARTIFACT_LOCATION_MAP.yaml`
3. `CATEGORY_GAP_REPORT.md`
4. semantic collision report
5. list of existing structures that must not be duplicated
6. one recommended next implementation batch

Forbidden:

- canonical DB writes,
- outcome analysis,
- production rebuild,
- duplicate registries.

---

## 69. Phase 1 — Artifact Category Registry and Ontology

Create:

- artifact category registry,
- canonical ontology,
- alias registry,
- validation schemas,
- validation tests.

---

## 70. Phase 2 — Source and Regime Contracts

Create source contracts for:

- funding,
- liquidations,
- mark prices,
- aggregate trades,
- book ticker,
- OI,
- candles,
- paths.

Link source modes and historical data regimes.

---

## 71. Phase 3 — Claim and Dependency Graph

Backfill current accepted claims and dependencies.

Initial claims include:

- corrected LONG raw baseline,
- corrected LONG normalized baseline,
- corrected LONG timing,
- corrected nested path,
- SHORT insufficiency,
- funding sample blocker,
- geometry source-quality blocker.

---

## 72. Phase 4 — Negative Knowledge and Unknowns

Backfill:

- rejected branches,
- insufficient branches,
- blocked branches,
- retry conditions,
- unresolved contradictions,
- missing evidence.

---

## 73. Phase 5 — Production Artifact Audit

Only after research architecture is stable:

```text
KEEP
EXTEND
REFACTOR
REPLACE
DELETE
DUPLICATED
UNSAFE
MISSING
```

Audit:

- risk,
- OMS,
- recovery,
- monitoring,
- deployment,
- paper/shadow/live,
- attribution,
- capital rules.

No automatic rebuild.

---

## 74. Phase 6 — Operational Runbooks and Gates

Create live-promotion, recovery, stale-data, kill-switch and incident procedures.

---

## 75. Phase 7 — Automated Missing-Category Gates

Fail repository checks when:

- a feature lacks a feature card,
- a source lacks a source contract,
- an experiment lacks dependencies,
- a claim lacks evidence,
- a live component lacks recovery ownership,
- a semantic change lacks an impact report.

---

# Part XXV — Initial Priority Gaps

## 76. Highest-Value Gaps

1. canonical ontology,
2. data-source contract registry,
3. artifact category registry,
4. claim-evidence registry,
5. experiment dependency graph,
6. data-regime registry,
7. negative knowledge and retry registry,
8. test-to-invariant map,
9. production artifact inventory,
10. external primary-source index.

---

# Part XXVI — Claude Working Contract

## 77. Claude Must Not Implement Everything at Once

Required sequence:

```text
Batch 1:
read-only artifact-universe inventory

Batch 2:
artifact category registry + canonical ontology

Batch 3:
source contracts + regime registry

Batch 4:
claim/evidence registry + dependency graph

Batch 5:
negative knowledge + unknowns

Batch 6:
production artifact audit

Batch 7:
operational runbooks
```

Each batch stops for operator approval.

---

## 78. Claude First-Batch Acceptance Criteria

Claude must:

- read canonical state files,
- read this whitepaper,
- inventory before creating,
- search broad artifact categories,
- identify existing equivalents,
- avoid duplicates,
- report conflicts,
- modify no canonical database,
- run no outcomes,
- write no production implementation,
- recommend at most one next batch.

---

# Part XXVII — Current Project Decisions Memory Capsule

## 79. Current Decisions to Preserve

```text
1. Current focus remains research integrity, feature/event identity,
   data semantics and mechanism discovery.

2. Production/risk/execution work is deferred, not forgotten.

3. Existing risk, OMS, recovery and monitoring components will not be blindly
   preserved or blindly deleted.

4. Future production work begins with a behavioral audit:
   KEEP / EXTEND / REFACTOR / REPLACE / DELETE / DUPLICATED / UNSAFE / MISSING.

5. Full rewrite is not the default.
   Selective rebuild and shadow parity are preferred.

6. Paper may retain multiple threshold/lane observations for analysis.
   Live must use cycle-level deduplication and explicit order rights.

7. Funding-state design is frozen but inferentially blocked by sample size.

8. Birth-truncated geometry is known-at-safe, but inferential research is
   blocked by liquidation source quality.

9. Baseline corrected-data LONG research is complete and stable.
   It is not alpha, PnL or a live rule.

10. New AI or Claude sessions must recover truth from repository artifacts,
    not conversational memory.

11. Vitalik-inspired broad-search principle is adopted:
    search all required knowledge-bearing artifact categories, not only
    conventional documents.
```

---

# Part XXVIII — Final Position

AMI should not merely collect more reports.

It should know:

- which categories of evidence can change a conclusion,
- which categories were searched,
- which categories are missing,
- which artifacts are authoritative for each question,
- which conflicts remain unresolved,
- which historical versions produced current data,
- which claims depend on which definitions,
- which research findings may or may not affect live execution.

The final question before any major conclusion should be:

> What categories of knowledge could change this answer, and have we searched all categories required for this scope?

That is the category-complete form of Artificial Market Intelligence.

---

# Appendix A — Question-Specific Category Checklist

```yaml
semantics_question:
  required:
    - official_source_documentation
    - historical_collector_code
    - schema
    - raw_data_evidence
    - tests
    - data_regime_history

implementation_question:
  required:
    - source_code
    - configuration
    - tests
    - runtime_logs
    - migrations
    - git_history
    - documentation

novelty_question:
  required:
    - completed_experiments
    - failure_archive
    - graveyard
    - unknown_registry
    - old_branches
    - external_research

live_readiness_question:
  required:
    - research_evidence
    - execution_code
    - risk_policy
    - recovery_tests
    - monitoring
    - incident_runbook
    - source_quality
    - live_permission
```

---

# Appendix B — Search Receipt Template

```yaml
search_receipt_id: SR-...
question: ...
scope: ...
required_categories: []
searched_categories: []
unsearched_categories:
  - category: ...
    reason: ...
artifacts_reviewed: []
conflicts: []
missing_artifacts: []
temporal_cutoff: ...
limitations: ...
verdict: CATEGORY_SUFFICIENT_WITH_LIMITATIONS
```

---

# Appendix C — Category Gap Report Template

```markdown
# Category Gap Report

## Question or subsystem

## Categories currently represented

## Categories missing

## Consequence of each missing category

## Existing equivalents that must not be duplicated

## Highest-risk semantic collisions

## Recommended next controlled batch

## Forbidden actions

## Stop condition
```

---

# Appendix D — ADR Template

```markdown
# ADR-XXX — Title

## Status
Proposed | Accepted | Superseded | Retired

## Context

## Decision

## Alternatives Considered

## Consequences

## Risks

## Review Trigger

## Affected Modules and Documents
```

---

# Appendix E — Non-Goals

This whitepaper does not:

- claim profitability,
- create live orders,
- lower sample requirements,
- invent missing source evidence,
- overwrite immutable experiments,
- select features from outcomes,
- authorize production rebuild,
- authorize live execution,
- treat documentation as a substitute for canonical state,
- allow AI sessions to guess missing methodology.
