# Eclipse News Intelligence — architecture

A research data layer under **Global Market Intelligence**. It turns unstructured
world information into structured, timestamped, auditable events so that a later
research programme can ask whether any of it predicts anything.

It does not answer that question. Nothing in this layer places an order, reads a
sealed arm, modifies E-DER V1 / A2 / V3, or opens a network connection.

```
WORLD
  ↓
GLOBAL MARKET INTELLIGENCE
  └── News Intelligence  ← this layer
        ↓
GLOBAL EVENT BUS   eclipse.news.* · eclipse.social.* · eclipse.macro.* · eclipse.market.*
        ↓
ALPHA / RESEARCH / RISK
        ↓
MASTER CENTER
```

## Where it lives

```
src/eclipse/news_intelligence/
    schemas/        raw · normalized · relevance · reaction · snapshot
    taxonomy/       versioned event vocabulary
    adapters/       source protocol, registry, authority tiers, mock fixtures
    relevance/      entity → asset graph (magnitudes, never directions)
    normalization/  extraction · normalizer · the model-annotation policy
    novelty/        is this new information, or the same information again
    amplification/  how loudly the same information is being repeated
    clustering/     outcome-blind grouping into independent events
    reaction/       measurement contracts + reverse-causality classification
    research/       snapshot builder · read-only API · private panel contract
    publishing/     bus subjects and envelopes, transport-free
    validation/     the audit trail
    integration/    read-only context beside E-DER candidates
    deferred.py     everything that must not start while the machine is busy
    pipeline.py     the order the stages run in

tests/news_intelligence/     68 tests, no database, no network
tools/news_intelligence_demo.py
reports/news_intelligence/
```

## The three invariants

Everything else is detail. These are the reasons the package is shaped the way
it is.

### 1. A feature is what was knowable; an outcome is what happened

They are two objects — `FeatureSnapshot` and `ResearchLabel` — joined only by
`event_id`, by code that had to type the join out. There is no constructor that
takes both, and no `include_future` flag, because a flag like that gets set to
True once in a hurry and nothing downstream can tell afterwards.

Two independent guards, because one is not enough:

- **Timestamp.** Every `Observation` carries its own `as_of`. One later than the
  decision time raises `LookaheadError` *at construction*, not at use.
- **Structure.** A realised return has no clock of its own, so no timestamp
  check can catch it. Field names are checked against `OUTCOME_FIELDS` — on the
  dataclass fields, on observation names, and on the keys of the free-form
  context dict, which is exactly where an outcome arrives wearing a new name.

### 2. The decision time is when *we* could know

Not when the world published. A statement made at 13:44 and received at 13:51 is
actionable at 13:51, and every snapshot and reaction request is anchored to
`first_seen_at`. `published_at` is kept beside it, so the lag is itself
measurable — a source that is consistently late is a source whose "news" the
market has already traded.

### 3. Relevance is not direction

The graph says the Federal Reserve is relevant to the two-year yield. It does not
say which way, and `Edge` refuses a negative weight. A graph encoding "Trump →
BTC bearish" would let every study inherit the conclusion as an assumption and
then rediscover it; the study would look like evidence and be a tautology.

Every edge carries its channel in words, so an auditor can disagree with it.
Where the channel is a narrative rather than a mechanism, the reason says so.

## Novelty ≠ amplification

The distinction the layer turns on.

```
13:44  a statement lands          novelty 1.00   amplification 0.00
13:48  a wire service repeats it  novelty 0.32   amplification 0.44
13:51  a second outlet            novelty 0.38   amplification 0.62
13:54  an aggregator              novelty 0.42   amplification 0.71
```

A market prices information once. The fiftieth article is not fifty times the
news — but it is much more attention, and the two are different features. Fusing
them into one "impact" score makes it impossible to tell "nobody noticed" from
"everybody already knew".

## Raw items are not independent events

Four outlets covering one announcement are one observation. Sample size is the
denominator of every significance test this system will run, and counting
reprints as independent inflates it in exactly the direction that makes noise
look like a finding.

`ResearchStore` therefore reports **both** numbers and publishes the ratio
between them (1.75 on the fixture set), so neither can be quoted as the other.
`one_per_cluster()` selects the **first** item of each cluster, never the
best-scoring one: selecting by any score conditions the sample on a quantity the
study is about to measure.

The clusterer is structurally outcome-blind. It accepts `ClusterInput`, a
projection that physically cannot carry a reaction — stronger than a rule saying
"don't look at outcomes", because if grouping could depend on the result, the
number of independent observations would become a function of the result.

## Reverse causality

The obvious analysis is wrong in a specific way: if a price was already moving
before the news arrived and continued afterwards, a naive post-event return
reports the news as predictive when the news was the consequence. Sentiment
feeds are especially prone to this — commentary follows price all day.

So pre-event windows (−30, −15, −5, −1) are measured as well as post-event ones
(+1, +5, +15, +30, +60, +240), and `classify_causality` returns one of
`NEWS_LEADS_PRICE`, `PRICE_LEADS_NEWS`, `SIMULTANEOUS`, `NO_RELATIONSHIP`,
`UNDETERMINED`. A move already underway is **never** reported as the news
leading, however large the post-event return.

A missing window is `UNDETERMINED`, never zero. An absent measurement and a calm
market are different things, and this repository has already paid once for
confusing them.

## Source authority ≠ impact

Tiers describe proximity to the fact, not reliability of judgement and not market
impact. A central bank publishing about its cafeteria is still TIER 1 and still
irrelevant. Authority and relevance are computed by different objects and never
multiplied together silently. An unregistered source has **no** authority rather
than a default one.

## What a model may write

LLMs may classify, extract entities, name topics, summarise and suggest
relevance. They may not touch timestamps, source identity, payload digests or
ids — `apply_annotation` refuses, so a prompt change cannot quietly turn
provenance into model output. Every annotation carries `model_id`,
`prompt_version`, `confidence` and `produced_at`, and annotating returns a new
object rather than mutating the old one, so both rows survive a model
evaluation.

A model may never emit `buy`, `sell`, `long`, `short`, `position_size` or
`order`. That is enforced, not requested.

## Event bus

Subjects are additive to the namespaces the platform already reserves:

| subject | payload |
|---|---|
| `eclipse.news.raw` | provenance of one received item |
| `eclipse.news.normalized` | the feature row |
| `eclipse.news.high_impact` | events whose relevance clears the routing threshold |
| `eclipse.social.raw` / `.normalized` | as above, social sources |
| `eclipse.macro.scheduled` / `.released` | calendar and release |
| `eclipse.market.cross_asset_context` | synchronised context |
| `eclipse.research.news_event_ready` | a snapshot exists; labels pending |

**Publish candidates, never outcomes** — the platform's existing rule, enforced
here rather than remembered. `assert_no_outcome` walks nested payloads and
refuses anything outcome-shaped; the realistic mistake is not a top-level
`return_bps` but a tidy `{"event": …, "label": …}` envelope built by someone
joining two objects for convenience.

NATS is not required. `InMemoryPublisher` satisfies the same protocol, which is
what makes the pipeline testable today and swappable later.

## E-DER integration

V1, A2 and V3 are frozen. This layer produces `NewsContext`, a separate object
keyed by candidate id, and there is no function that takes an arm's rule set.

The reason is arithmetic, not deference. A frozen arm's forward record is only
evidence because its definition did not move while the sample accumulated. The
moment news state filters which candidates count, the arm under test is a
different arm with a sample of **zero**, and the old sample cannot be carried
over however continuous it looks.

`proposed_arm_name("E-DER-V3", "risk_off")` → `E-DER-V3+NEWS_RISK_OFF`, and it
refuses to hand back a frozen arm's name.

Context is built from events with `first_seen_at ≤ candidate_time` only —
lookahead arriving through a side door is still lookahead.

## Deferred while the machine is busy

Registered in `deferred.py`. Each entry says what it would cost and what has to
be true before it runs. **Calling any of them raises**, because a collector that
returns quietly is indistinguishable from a working collector on a quiet day.

| capability | resource |
|---|---|
| `live_collectors` | network, RAM, a writer process per source |
| `historical_backfill` | network, tens of GB of disk |
| `embedding_index` | RAM, CPU, disk |
| `llm_batch_classification` | CPU or API budget |
| `market_reaction_measurement` | disk I/O on the shared database |
| `cross_market_backtest` | CPU, disk, hours |
| `continuous_research_jobs` | CPU, disk |

## Research families this prepares for

Schemas exist so these can be tested later. None has been run.

**A** macro event alpha · **B** political event alpha · **C** influencer event
alpha · **D** company event alpha · **E** cross-asset lead-lag · **F**
news-reaction alpha (headline plus the first minutes) · **G** whether news state
affects the existing arms.

Family G is the one to be most careful with: it is the one that looks like it can
reuse an existing sample, and it cannot.

## The private panel

`research/dashboard_contract.py`. Private mesh only. Two things are absent from
every payload by construction, and the builder refuses keys rather than trusting
a template: **no realised returns** and **no arm-level aggregates**. An operator
screen is one screenshot away from being a public one.

An asset with no reading renders as `—` and is listed in `incomplete`. Rendering
it as a zero or a stale carry-forward would put a fabricated reading on the
screen — the exact fault a dashboard review caught here once before.

## Running it

```powershell
python -m tools.news_intelligence_demo
python -m tools.news_intelligence_demo --explain 0
python -m tools.news_intelligence_demo --json reports/news_intelligence/demo_payload.json
```

Seven synthetic items through the full path, in memory, in under a second. No
network, no database, no model.

```powershell
python -m pytest tests/news_intelligence/test_leakage_and_schemas.py `
                 tests/news_intelligence/test_pipeline_mock_validation.py `
                 -q -p no:cacheprovider --basetemp=<scratch>
```

Two files per call, per the repository's test convention.

**Note on the local conftest.** `tests/news_intelligence/conftest.py` overrides
the root canonical-database isolation fixture, which copies and hashes a 213 MB
SQLite file at session start. Nothing in this subtree opens a database, so that
cost buys nothing here. The override is fail-closed: a session fixture records
every `sqlite3.connect` call made while these tests run and asserts there were
none, so if someone later adds a database-touching test the suite says so instead
of running unprotected.

## Versioning

Four version numbers, because they move for different reasons and a result has to
say which it was computed under: `SCHEMA_VERSION` (row shape),
`TAXONOMY_VERSION` (what a category means), `GRAPH_VERSION` (which assets an
entity touches), and each judgement's own `model_id`. Collapsing them would make
a result that survives one revision indistinguishable from one that survives all
three.

## Status

Architecture, schemas, engines, contracts, tests and mock validation are
complete and passing. No collector has been started, no historical data has been
downloaded, no backtest has been run, and no existing arm, ledger, collector or
research artifact has been touched.
