# AMI × COMMERCE INTELLIGENCE OS
## Implementation-Ready Whitepaper and Canonical Build Specification

**Version:** 1.1 COMPLETE — MULTI-MODEL HANDOFF EDITION  
**Date:** 5 July 2026  
**Status:** IMPLEMENTATION-READY / PRE-LIVE / EVIDENCE-NOT-YET-PROVEN  
**Origin:** AMI × S34 canonical evidence-governed research operating system  
**Primary launch market:** Türkiye  
**Primary launch model:** Affiliate Product & Content Intelligence  
**Secondary model:** Local supplier / controlled fulfillment / micro-stock  
**Later sourcing model:** Alibaba wholesale, low-MOQ, private label  
**Explicit Phase-1 exclusion:** AliExpress-to-Türkiye direct-per-order dropshipping  
**Canonical repository target:** `D:\commerce_intelligence`  
**Canonical database target:** `data\commerce_canonical.db`  

> This document is sufficient to begin implementation. It is not proof that the business model is profitable. Code readiness, data readiness, channel access, traffic, and commercial evidence are separate states and must never be conflated.

> **Execution rule:** Fable plans and reviews; Sonnet implements; Opus performs preferred independent wave review when available. Every transition is written to `HANDOFF.md` and announced to the operator before work stops.

---

# 0. Operator Summary — Türkçe

Bu sistemin ilk işi mağaza açıp rastgele ürün koymak değildir.

İlk iş:

```text
İnsanların hangi problemleri yaşadığını bul
→ o probleme bağlı ürünleri keşfet
→ ürünün gerçekten yükselip yükselmediğini ölç
→ rekabet ve doygunluğu kontrol et
→ izinli görsellerle faceless içerik üret
→ affiliate linkle kontrollü test et
→ tıklama, satış, iptal ve iadeyi kaydet
→ gerçek kazanan bucket'ları ayır
→ yalnızca kanıtlanan ürünlerde tedarikçi ve stok aşamasına geç
```

Sistem şu soruya cevap verir:

> Hangi problem, hangi ürün, hangi teklif, hangi video yapısı, hangi platform ve hangi zamanda gerçekten para bırakıyor?

İlk sürümde:

- stok alınmaz,
- müşteri ödemesi alınmaz,
- ürün kargolanmaz,
- ücretli reklam zorunlu değildir,
- senin kamera karşısına geçmen gerekmez,
- iPhone gerekmez,
- hot ürünler ve içerik hipotezleri sistem tarafından seçilir,
- gerçek outcome gelmeden “kazanan” ilan edilmez.

Doğru sıra:

```text
Affiliate validation
→ supported product/content buckets
→ local supplier or micro-stock
→ Alibaba samples and low MOQ
→ controlled marketplace selling
→ private label only after repeatable economics
```

---

# 1. Executive Decision

## 1.1 What is being built

The product is not a generic product scraper, a link spammer, or a one-click dropshipping clone.

It is an evidence-governed **Commerce Alpha Factory** composed of:

```text
Hot Problem Discovery
+ Product Discovery
+ Trend Velocity
+ Saturation Detection
+ Offer Intelligence
+ Faceless Creative Generation
+ Controlled Publishing
+ Click / Order / Return Outcomes
+ Evidence and Contradiction Engine
+ Supplier Intelligence
+ Sourcing Permission
= Commerce Intelligence OS
```

## 1.2 First monetization path

```text
We publish tracked affiliate recommendations
→ customer buys from the marketplace/seller
→ marketplace/seller handles payment and fulfillment
→ eligible commission is reported to us
→ cancellations and returns reduce the final result
```

The first commercial question is therefore not “Can we ship products?” It is:

> Can the system repeatedly identify product-content combinations that generate qualified clicks and eligible purchases at a worthwhile return on content effort?

## 1.3 Strategic ladder

| Phase | Business model | Capital risk | Operational burden | Purpose |
|---|---|---:|---:|---|
| 1 | Affiliate intelligence | Very low | Low–medium | Learn demand and content conversion |
| 2 | Local supplier / controlled dropship | Low–medium | Medium | Validate fulfillment and margin |
| 3 | Micro-stock | Medium | Medium | Improve delivery and control |
| 4 | Alibaba low-MOQ sourcing | Medium–high | High | Improve unit economics |
| 5 | Private label / marketplace scale | High | High | Build durable commerce asset |
| 6 | SaaS / intelligence licensing | Variable | High | Monetize the operating system itself |

---

# 2. Non-Negotiable Canonical Principles

The commerce project inherits the epistemic discipline of AMI × S34.

## 2.1 Canonical SQL is the structured source of truth

- Markdown is the human synthesis layer.
- Dashboards are views, not sources of truth.
- AI summaries may interpret evidence but may not silently mutate canonical records.
- Historical evidence is append-only unless a documented correction event exists.

## 2.2 Evidence layers must remain separated

```text
DISCOVERY DATA
SHADOW / ORGANIC OBSERVATION
FORWARD TEST
AFFILIATE LIVE OUTCOME
SELLER / INVENTORY LIVE OUTCOME
```

No historical observation may be represented as a live sale. No simulated profit may be mixed with verified commission.

## 2.3 Frozen hypotheses and bucket versioning

A bucket definition is immutable after its test begins.

Bad:

```text
Change the price band, creative rule, and minimum clicks after seeing results.
```

Correct:

```text
Close bucket v1
→ document result
→ create bucket v2 with the new rule
→ test separately
```

## 2.4 Unknown must remain a valid state

The system must distinguish:

```text
SUPPORTED
REJECTED
UNDER_SAMPLED
CONTRADICTORY
CONTAMINATED
BLOCKED_DATA
BLOCKED_PLATFORM
BLOCKED_COMPLIANCE
UNKNOWN
```

No score may convert missing data into confidence.

## 2.5 Permission is separate from prediction

A product can have a high hotness score while still being blocked from publishing or sourcing because of:

- missing disclosure,
- unclear media rights,
- prohibited category,
- uncertain product safety,
- stale price,
- broken affiliate link,
- weak seller quality,
- unknown landed cost,
- missing sample approval.

---

# 3. Business Models That Must Not Be Confused

## 3.1 Affiliate marketing

```text
We recommend
→ marketplace sells
→ seller fulfills
→ customer pays marketplace/seller
→ we receive eligible commission
```

We do not become seller of record merely by sharing a tracked link.

## 3.2 Dropshipping

```text
Customer buys from our store
→ customer pays us
→ we order from supplier
→ supplier ships
→ we remain customer-facing seller
```

Supplier fulfillment does not remove seller obligations.

## 3.3 Marketplace selling

```text
We list under our seller account
→ customer buys from us through marketplace
→ we or an approved fulfillment partner ships
→ invoice, return, account health, and customer obligations remain ours
```

## 3.4 Why affiliate must come first

Affiliate evidence can reveal:

- product demand,
- click intent,
- price sensitivity,
- category quality,
- content format fit,
- cancellation and return behavior,
- seasonality,
- offer quality.

This information reduces—but never eliminates—the risk of later inventory purchases.

---

# 4. Verified Platform and Regulatory Reality — 5 July 2026

## 4.1 Trendyol Affiliate

Trendyol maintains an influencer affiliate application route for Türkiye and Azerbaijan. Its current affiliate information states that tracked-link attribution is 24 hours unless another influencer link, advertisement, or Trendyol notification intervenes; cancelled and returned purchases are deducted from turnover and earnings; and invoice workflow is managed through the Influencer Panel.[^trendyol_affiliate][^trendyol_application]

**Implementation consequence:**

```text
platform_reported_gross_commission
− cancelled_commission
− returned_commission
= verified_net_commission
```

Never treat click value or gross attributed turnover as final income.

Any daily-link or campaign limits shown inside the live panel must be stored as time-versioned platform configuration rather than hard-coded forever.

## 4.2 Amazon Türkiye Associates

Amazon Türkiye’s review guidance requires at least three qualifying sales in the first 180 days before review and expects public, original content; it states a baseline of at least ten public posts and evaluates submitted sites and social pages.[^amazon_review]

**Implementation consequence:**

- Amazon eligibility is a separate channel state.
- The MVP must not depend on Amazon acceptance.
- Original-content requirements rule out a low-effort copy-only strategy.
- Amazon program content and APIs must be used only within current program permissions.

## 4.3 Trendyol seller integration

Trendyol’s official developer documentation states that Marketplace partners can integrate product transfer, stock and price updates, order handling, invoice submission, and customer-question workflows through its APIs.[^trendyol_api]

**Implementation consequence:**

Seller API modules belong to a later package and must remain disabled in affiliate-only Phase 1.

## 4.4 Türkiye cross-border e-import reality

Türkiye’s Ministry of Trade announced a change to simplified customs declaration treatment for low-value e-imports in January 2026. Current postal and express-cargo guidance must be reviewed before any direct cross-border fulfillment design.[^customs_change][^postal_cargo]

**Implementation consequence:**

```text
AliExpress direct-to-Türkiye customer fulfillment
= BLOCKED_BY_DEFAULT in Phase 1
```

AliExpress may be used for discovery or samples, not assumed to be a stable Turkish last-mile solution.

## 4.5 ETBİS boundary

The Ministry’s ETBİS FAQ states that sellers operating through domestic intermediary service providers do not have an ETBİS registration and notification obligation solely for that channel, while other setups can create registration duties.[^etbis]

**Implementation consequence:**

The system must store channel type:

```text
DOMESTIC_MARKETPLACE
OWN_ECOMMERCE_SITE
FOREIGN_MARKETPLACE
AFFILIATE_ONLY
```

Compliance tasks are derived from channel type, not guessed globally.

## 4.6 Affiliate advertising disclosure

Türkiye’s social-media advertising guide requires commercial advertising to be clear and distinguishable, prohibits covert advertising, and requires the commercial relationship to be disclosed in a visible manner. It also prohibits presenting unexperienced products as personally approved and restricts unsupported health or scientific claims.[^influencer_guide]

**Implementation consequence:**

No asset receives publish permission without:

```text
disclosure_required = true/false
disclosure_present = true
disclosure_position_valid = true
claim_risk = LOW
experience_claim_valid = true
```

## 4.7 Creative research sources

TikTok Creative Center provides Top Ads and trend resources; TikTok’s Symphony tooling advertises AI-assisted video creation from limited inputs. Meta Ad Library exposes currently running ads for search by advertiser or keyword. These sources can inform structure, saturation, and creative research, but they do not grant permission to copy another advertiser’s assets.[^tiktok_top_ads][^tiktok_symphony][^meta_library]

## 4.8 Alibaba sourcing

Alibaba Trade Assurance describes protected payment, support for product or shipping issues, and mediation between buyers and suppliers. Protection does not replace sampling, specification control, compliance checks, or independent inspection.[^trade_assurance]

---

# 5. Project Boundary

## 5.1 Phase-1 in scope

- Manual and CSV product candidate ingestion.
- Platform and source registry.
- Product identity resolution.
- Hot Problem Graph.
- Product hotness and trend velocity.
- Lifecycle state machine.
- Saturation and copycat indicators.
- Review and comment mining.
- Offer intelligence.
- Creative Genome.
- Faceless creative plans and render manifests.
- Affiliate link registry.
- Publication and attribution ledger.
- Click, order, commission, cancellation, and return import.
- Evidence states and bucket versioning.
- Permission engine.
- Health checks.
- Dashboard and Markdown reports.
- Manual approval before publication.

## 5.2 Phase-1 explicitly out of scope

- Automated purchasing.
- Automated seller-account listing.
- Direct customer payment collection.
- Automatic inventory purchase.
- Automatic paid-ad spend.
- Automated public posting without approval.
- Unauthorized scraping.
- Reusing copyrighted seller videos without rights.
- Medical, supplement, baby-safety, counterfeit, or high-regulatory-risk categories.
- AliExpress direct-to-Türkiye consumer dropshipping.

## 5.3 Later phases

- Official seller APIs.
- Local supplier stock synchronization.
- Sample and quality-control workflows.
- Landed-cost engine.
- Micro-stock decisions.
- Customer-service event ingestion.
- Paid creative testing.
- Multi-country opportunity routing.
- Private-label sourcing.
- SaaS tenant architecture.

---

# 6. Repository Strategy and AMI Reuse

## 6.1 Do not clone the live trading system blindly

Claude must first conduct a source audit and classify every reusable component.

```text
DOMAIN_AGNOSTIC_CORE
TRADING_COUPLED
RUNTIME_DANGEROUS
DATA_ONLY
DOCUMENTATION_ONLY
```

## 6.2 Reuse matrix

### Reuse or adapt

- configuration loader,
- structured logging,
- migration runner,
- audit-event framework,
- experiment registry patterns,
- evidence registry patterns,
- state-machine utilities,
- deterministic report generator,
- health-check framework,
- permission-decision framework,
- test utilities,
- archive/retention utilities,
- CLI conventions,
- documentation conventions.

### Do not copy as live dependencies

- liquidation collectors,
- order-book streams,
- exchange credentials,
- order executors,
- leverage and risk controls,
- trading outcomes,
- market symbol assumptions,
- live trading schedulers,
- microstructure database,
- exchange websocket services,
- production trading secrets.

## 6.3 Required isolation

```text
D:\eclipse_scalper              # untouched source project
D:\commerce_intelligence        # independent new repository
```

The commerce repository must have:

- separate Git history,
- separate environment file,
- separate database,
- separate scheduler,
- separate logs,
- separate secrets,
- separate tests,
- no import path into the trading runtime.

## 6.4 Phase-0 audit deliverables

Before implementation:

```text
AMI_REUSE_AUDIT.md
AMI_REUSE_MATRIX.csv
COMMERCE_BOUNDARY.md
DEPENDENCY_GRAPH.md
INITIAL_IMPLEMENTATION_PLAN.md
```

No source file should be copied until it appears in the approved reuse matrix.

---

# 7. Canonical Repository Layout

```text
commerce_intelligence/
├── README.md
├── SYSTEM_STATE.md
├── HANDOFF.md
├── IMPLEMENTATION_ROADMAP.md
├── RECONCILIATION_LOG.md
├── DECISIONS.md
├── CHANGELOG.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── base.yaml
│   ├── development.yaml
│   ├── production.yaml
│   ├── platforms.yaml
│   ├── scoring.yaml
│   ├── permissions.yaml
│   └── retention.yaml
├── migrations/
│   ├── 0001_core_identity.sql
│   ├── 0002_catalog_offers.sql
│   ├── 0003_problems_and_features.sql
│   ├── 0004_creatives_and_publications.sql
│   ├── 0005_experiments_and_outcomes.sql
│   ├── 0006_evidence_and_permissions.sql
│   ├── 0007_suppliers_and_sourcing.sql
│   └── 0008_health_and_audit.sql
├── src/commerce_intelligence/
│   ├── __init__.py
│   ├── cli.py
│   ├── settings.py
│   ├── db/
│   │   ├── connection.py
│   │   ├── migrations.py
│   │   ├── repositories/
│   │   └── models.py
│   ├── identity/
│   │   ├── product_resolution.py
│   │   ├── offer_resolution.py
│   │   └── source_registry.py
│   ├── ingestion/
│   │   ├── manual.py
│   │   ├── csv_import.py
│   │   ├── affiliate_reports.py
│   │   └── validators.py
│   ├── problems/
│   │   ├── graph.py
│   │   ├── clustering.py
│   │   └── scoring.py
│   ├── products/
│   │   ├── discovery.py
│   │   ├── hotness.py
│   │   ├── velocity.py
│   │   ├── lifecycle.py
│   │   ├── saturation.py
│   │   └── fake_hype.py
│   ├── reviews/
│   │   ├── miner.py
│   │   ├── taxonomy.py
│   │   └── risk_extraction.py
│   ├── offers/
│   │   ├── router.py
│   │   ├── economics.py
│   │   └── eligibility.py
│   ├── creatives/
│   │   ├── genome.py
│   │   ├── hypotheses.py
│   │   ├── rights.py
│   │   ├── disclosure.py
│   │   ├── render_manifest.py
│   │   └── no_shoot_factory.py
│   ├── experiments/
│   │   ├── registry.py
│   │   ├── preregistration.py
│   │   ├── contamination.py
│   │   └── counterfactuals.py
│   ├── outcomes/
│   │   ├── attribution.py
│   │   ├── affiliate.py
│   │   ├── returns.py
│   │   └── profit_truth.py
│   ├── evidence/
│   │   ├── states.py
│   │   ├── evaluator.py
│   │   ├── contradictions.py
│   │   └── bucket_versioning.py
│   ├── permissions/
│   │   ├── engine.py
│   │   ├── publish.py
│   │   ├── sourcing.py
│   │   └── scale.py
│   ├── portfolio/
│   │   ├── allocator.py
│   │   └── daily_selection.py
│   ├── suppliers/
│   │   ├── digital_twin.py
│   │   ├── samples.py
│   │   ├── landed_cost.py
│   │   └── quality.py
│   ├── research/
│   │   ├── question_tree.py
│   │   └── next_best_experiment.py
│   ├── health/
│   │   ├── checks.py
│   │   └── snapshot.py
│   ├── reports/
│   │   ├── daily.py
│   │   ├── weekly.py
│   │   ├── evidence_register.py
│   │   └── templates/
│   └── dashboard/
│       ├── app.py
│       ├── views/
│       └── queries/
├── tools/
│   ├── import_products.py
│   ├── import_affiliate_outcomes.py
│   ├── generate_daily_portfolio.py
│   ├── generate_creative_manifest.py
│   └── reconcile_outcomes.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── contract/
│   ├── migration/
│   ├── property/
│   └── fixtures/
├── data/
│   ├── commerce_canonical.db
│   ├── imports/
│   ├── exports/
│   ├── archives/
│   └── quarantine/
├── reports/
│   ├── daily/
│   ├── weekly/
│   ├── research/
│   └── audits/
└── docs/
    ├── architecture/
    ├── runbooks/
    ├── platform_rules/
    ├── data_dictionary/
    └── decisions/
```

---

# 8. Canonical Domain Model

## 8.1 Central evidence unit

```text
Problem
× Product family
× Exact offer
× Affiliate platform
× Traffic platform
× Audience context
× Creative genome
× Price/discount regime
× Time window
= Versioned commerce bucket
```

## 8.2 Identity hierarchy

```text
canonical_problem
    └── canonical_product_family
          └── canonical_product
                └── product_variant
                      └── platform_listing
                            └── offer_snapshot
```

A listing is not a product. A price snapshot is not an offer identity. An offer is not a creative.

## 8.3 Required identifiers

Every canonical entity receives:

```text
internal UUID
source system
source-native ID
created_at UTC
updated_at UTC
valid_from UTC
valid_to UTC nullable
record_version
source confidence
```

---

# 9. Canonical SQL Schema

The exact SQL syntax may be adapted to SQLite, but semantics are frozen.

## 9.1 Core registry

### `sources`

```text
source_id PK
source_name
source_type
base_url
access_method
terms_reviewed_at
allowed_use_notes
is_active
created_at
```

### `platforms`

```text
platform_id PK
platform_name
platform_role              # affiliate, traffic, marketplace, supplier
country_code
channel_type
is_enabled
access_state
rules_version
rules_checked_at
```

### `canonical_products`

```text
product_id PK
title_normalized
brand_normalized
category_id
product_family_id
default_currency
risk_category
regulatory_risk
created_at
retired_at
```

### `platform_listings`

```text
listing_id PK
product_id FK
platform_id FK
source_native_id
canonical_url_hash
seller_id
listing_title
first_seen_at
last_seen_at
listing_state
```

### `offer_snapshots`

```text
offer_snapshot_id PK
listing_id FK
observed_at
price
reference_price
discount_rate
currency
stock_state
shipping_promise_days
seller_rating
review_count
average_rating
commission_rate
commission_eligible
snapshot_quality
```

## 9.2 Hot Problem Graph

### `problems`

```text
problem_id PK
problem_name
problem_description
problem_cluster
risk_level
created_at
```

### `problem_observations`

```text
observation_id PK
problem_id FK
source_id FK
observed_at
text_excerpt_hash
signal_type
signal_value
language
confidence
```

### `product_problem_edges`

```text
edge_id PK
problem_id FK
product_id FK
relationship_type          # solves, partly_solves, creates, associated
strength
confidence
valid_from
valid_to
```

## 9.3 Reviews and comments

### `review_observations`

```text
review_id PK
listing_id FK
source_native_review_hash
observed_at
rating
review_text_hash
review_language
verified_purchase nullable
media_present
```

### `review_features`

```text
review_feature_id PK
review_id FK
purchase_reason
problem_statement
positive_feature
objection
complaint
return_reason
size_expectation_issue
quality_issue
shipping_issue
claim_risk
sentiment
model_version
human_verified
```

Do not store unnecessary personal data from reviewers.

## 9.4 Creative registry

### `media_assets`

```text
asset_id PK
asset_type                  # image, video, audio, generated
source_id FK
source_url_hash nullable
local_path nullable
rights_basis
rights_evidence_path
usage_scope
expires_at nullable
content_hash
created_at
```

### `creative_assets`

```text
creative_id PK
product_id FK
creative_version
creative_status
rendered_asset_id nullable
script_text
voiceover_text
caption_text
cta_text
disclosure_text
created_at
approved_at nullable
```

### `creative_genome`

```text
creative_id PK/FK
hook_type
hook_text
first_frame_type
problem_visible_second
product_visible_second
human_presence
face_presence
ugc_style
demonstration
before_after
comparison
social_proof
voiceover_type
subtitle_density
video_duration_seconds
scene_count
cut_rate
music_type
offer_type
urgency_type
cta_type
claim_risk
```

### `render_manifests`

```text
manifest_id PK
creative_id FK
tool_name
input_asset_ids_json
scene_plan_json
aspect_ratio
resolution
voice_model
subtitle_style
render_state
output_asset_id nullable
error_message nullable
```

The OS may produce a render manifest for Pippit, CapCut, Canva, TikTok Symphony, or another approved tool. Phase 1 does not require direct API automation.

## 9.5 Publication and attribution

### `affiliate_links`

```text
affiliate_link_id PK
platform_id FK
listing_id FK
tracking_url_encrypted
tracking_url_hash
campaign_code
created_at
expires_at nullable
link_state
last_verified_at
```

### `publications`

```text
publication_id PK
creative_id FK
traffic_platform_id FK
affiliate_link_id FK
published_at
post_native_id
post_url_hash
organic_or_paid
status
disclosure_present
disclosure_verified
```

### `publication_snapshots`

```text
snapshot_id PK
publication_id FK
observed_at
impressions
qualified_views
watch_time_seconds
three_second_views
completion_rate
likes
comments
shares
profile_visits
link_clicks
unique_clicks
```

## 9.6 Experiments

### `experiments`

```text
experiment_id PK
experiment_name
hypothesis
primary_metric
secondary_metrics_json
start_at
end_at
status
preregistered_at
frozen_definition_hash
```

### `experiment_arms`

```text
arm_id PK
experiment_id FK
arm_name
product_id FK
listing_id FK
creative_id FK
audience_context_json
offer_context_json
```

### `contamination_events`

```text
contamination_id PK
experiment_id FK
observed_at
contamination_type
severity
description
affects_primary_metric
```

## 9.7 Affiliate outcomes

### `affiliate_outcomes`

```text
outcome_id PK
platform_id FK
affiliate_link_id FK
publication_id nullable
report_date
gross_clicks
eligible_orders
gross_order_value
gross_commission
cancelled_orders
cancelled_commission
returned_orders
returned_commission
verified_net_commission
currency
report_source
import_batch_id
```

### `attribution_quality`

```text
attribution_id PK
outcome_id FK
attribution_method
confidence
known_overlap
known_missingness
notes
```

## 9.8 Evidence and buckets

### `bucket_definitions`

```text
bucket_id PK
bucket_name
bucket_type
version
frozen_definition_json
frozen_definition_hash
created_at
retired_at nullable
```

### `bucket_membership`

```text
membership_id PK
bucket_id FK
entity_type
entity_id
valid_from
valid_to nullable
```

### `evidence_evaluations`

```text
evaluation_id PK
bucket_id FK
evaluated_at
evidence_state
sample_size
qualified_exposure
primary_metric_value
confidence_interval_json
replication_count
contradiction_count
contamination_count
applicable_regime_json
reason_code
```

### `contradictions`

```text
contradiction_id PK
bucket_id FK
observed_at
regime_a_json
regime_b_json
metric_a
metric_b
severity
status
```

## 9.9 Permissions

### `permission_decisions`

```text
permission_id PK
entity_type
entity_id
permission_type
decision                 # ALLOW, DENY, REVIEW
reason_codes_json
rule_version
input_snapshot_hash
created_at
expires_at nullable
human_override
human_override_reason nullable
```

## 9.10 Supplier and sourcing

### `suppliers`

```text
supplier_id PK
supplier_name
platform_id FK
country_code
verified_status
trade_assurance_status
created_at
```

### `supplier_quotes`

```text
quote_id PK
supplier_id FK
product_id FK
quoted_at
moq
unit_price
currency
sample_price
production_lead_days
shipping_term
shipping_cost
customization_available
quote_valid_until
```

### `samples`

```text
sample_id PK
quote_id FK
ordered_at
received_at
sample_cost_total
quality_score
packaging_score
spec_match_score
compliance_document_score
approval_state
```

### `landed_cost_models`

```text
landed_cost_id PK
product_id FK
quote_id FK
calculated_at
unit_product_cost
unit_shipping_cost
customs_estimate
brokerage_estimate
marketplace_fee_estimate
payment_fee_estimate
return_reserve
quality_reserve
tax_allocation
expected_unit_contribution
stress_unit_contribution
assumption_hash
```

## 9.11 Health and audit

```text
import_batches
data_quality_events
health_snapshots
audit_events
research_questions
research_answers
model_runs
manual_reviews
```

---

# 10. Data Source Policy

## 10.1 Source tiers

### Tier A — canonical outcome sources

- official affiliate panel exports,
- official seller reports,
- owned social analytics,
- approved platform APIs,
- payment and accounting reports when seller phase begins.

### Tier B — research sources

- official trend tools,
- official ad libraries,
- manually reviewed marketplace pages,
- supplier-provided catalogs,
- written supplier quotations.

### Tier C — weak signals

- third-party trend lists,
- social commentary,
- unverified sales estimates,
- influencer claims,
- scraped counts of uncertain provenance.

Tier C may generate a candidate but may not establish support.

## 10.2 Forbidden acquisition behavior

- bypassing authentication or access controls,
- evading rate limits,
- scraping where terms prohibit it,
- using private customer data without authorization,
- fake clicks, fake orders, or self-purchases,
- automated engagement manipulation,
- copying another creator’s media without rights,
- storing credentials in source code.

## 10.3 Ingestion contract

Every ingestion batch must record:

```text
source
retrieved_at
coverage window
record count
schema version
file hash
validation status
quarantine count
```

Invalid rows go to quarantine and may not silently disappear.

---

# 11. Hot Problem Engine

## 11.1 Purpose

Products change. Problems persist.

The engine discovers repeated consumer problems from:

- reviews,
- comments,
- search phrases,
- product questions,
- return reasons,
- creative hooks,
- manual research notes.

## 11.2 Problem states

```text
UNSEEN
DISCOVERED
REPEATED
GROWING
HIGH_INTENT
SOLUTION_CROWDED
UNRESOLVED
DECLINING
```

## 11.3 Problem score

```text
Problem Opportunity Score =
  frequency_z
× growth_multiplier
× purchase_intent_weight
× solution_gap_weight
× content_demonstrability
× compliance_safety_factor
```

The first implementation may use normalized weighted components. It must not pretend to be causal.

## 11.4 Example

```text
Problem: pet hair embedded in fabric sofa
Frequency: high
Growth: medium
Existing solutions: crowded
Demonstrability: very high
Return risk: low–medium
Compliance risk: low
State: REPEATED / TESTABLE
```

---

# 12. Product Discovery and Hotness

## 12.1 Candidate generation

Candidate sources may include:

- marketplace category browsing,
- official trend tools,
- creative libraries,
- review growth,
- seller catalog changes,
- manual operator ideas,
- problem-to-solution graph expansion.

## 12.2 Provisional hotness components

```text
Demand momentum
Review velocity
Search/social momentum
Offer attractiveness
Seller quality
Content demonstrability
Competitive whitespace
Commission attractiveness
Return-risk inverse
Compliance-risk inverse
```

## 12.3 Hotness score

```text
Hotness =
  0.18 demand_momentum
+ 0.14 review_velocity
+ 0.12 social_search_momentum
+ 0.10 offer_attractiveness
+ 0.10 seller_quality
+ 0.12 content_demonstrability
+ 0.08 competitive_whitespace
+ 0.08 commission_attractiveness
+ 0.04 return_safety
+ 0.04 compliance_safety
```

These weights are a starting hypothesis. Freeze them before observing outcomes and version all changes.

## 12.4 Confidence modifier

```text
Adjusted Hotness = Raw Hotness × Data Confidence
```

A product with excellent weak-signal data cannot outrank a moderately strong product with verified outcome data solely through missingness.

---

# 13. Trend Velocity Engine

## 13.1 Goal

Identify whether the product is early, accelerating, mature, saturated, or decaying.

## 13.2 Features

```text
review_count_delta_7d
review_count_delta_30d
price_change_velocity
seller_count_velocity
stockout_frequency
ad_presence_velocity
search_interest_slope
our_click_velocity
our_order_velocity
```

## 13.3 Lifecycle states

```text
DISCOVERED
EARLY_SIGNAL
EMERGING
ACCELERATING
PROVEN
MAINSTREAM
SATURATING
SATURATED
DECAYING
DEAD
REVIVING
```

## 13.4 Transition rules

- A state transition requires minimum data confidence.
- One viral observation cannot jump directly from DISCOVERED to PROVEN.
- PROVEN requires verified outcomes.
- SATURATED is not equivalent to unprofitable; it requires a margin and competition assessment.

---

# 14. Saturation and Copycat Detector

## 14.1 Signals

- seller count,
- listing similarity,
- repeated identical titles,
- repeated image hashes,
- creative-hook repetition,
- ad-count growth,
- price dispersion collapse,
- discount escalation,
- declining commission or margin,
- rising customer complaint density.

## 14.2 States

```text
LOW_COMPETITION
COMPETITION_RISING
CREATIVE_CROWDING
PRICE_WAR
SATURATED
COMMODITIZED
```

## 14.3 Rule

High saturation does not automatically reject affiliate use. It does block inventory sourcing unless margin, differentiation, and supplier control remain acceptable.

---

# 15. Fake-Hype Detector

## 15.1 Core patterns

```text
High views + low clicks           = entertainment without intent
High clicks + low orders          = offer/product mismatch
Orders + high cancellations       = weak purchase quality
Orders + high returns             = false commercial winner
Low views + high conversion       = hidden commercial alpha
High gross commission + low net   = accounting illusion
```

## 15.2 Required metrics

```text
view_to_click
click_to_order
gross_to_net_commission
cancel_rate
return_rate
net_commission_per_100_clicks
net_commission_per_content_hour
```

## 15.3 Evidence rule

The system must never label a product profitable from views, likes, CTR, or gross turnover alone.

---

# 16. Review & Comment Miner

## 16.1 Extracted taxonomy

```text
purchase_reason
primary_problem
positive_feature
unexpected_benefit
objection
quality_complaint
size_or_fit_problem
expectation_gap
shipping_problem
return_reason
repeat_use_case
audience_identity_without_sensitive profiling
```

## 16.2 Dual use

Review data feeds:

1. creative hooks,
2. product risk,
3. offer selection,
4. return prediction,
5. problem graph,
6. supplier requirements.

## 16.3 Claim safety

A review-derived statement cannot automatically become an objective product claim. The system must distinguish:

```text
consumer opinion
observed recurring complaint
verified specification
unverified marketing claim
```

---

# 17. Offer Intelligence and Offer Router

## 17.1 Same product, different economics

The router compares:

- price,
- commission,
- seller score,
- shipping promise,
- stock,
- review quality,
- return risk,
- tracking reliability,
- affiliate eligibility.

## 17.2 Expected affiliate value

```text
Expected Net Commission Per Click =
  predicted_purchase_conversion
× average_order_value
× commission_rate
× (1 − expected_cancel_rate)
× (1 − expected_return_rate)
```

## 17.3 Utility score

```text
Offer Utility =
  consumer_value_score
× seller_reliability
× attribution_reliability
× expected_net_commission_per_click
```

The router must not select an inferior consumer offer only because it pays more commission. Minimum consumer-value and seller-quality gates apply first.

---

# 18. Creative Genome

## 18.1 Principle

A video is not one indivisible object. It is a feature vector.

## 18.2 Genome dimensions

```text
Hook type
Hook text
First frame
Problem visibility time
Product visibility time
Human/face presence
Demonstration
Before/after
Comparison
Social proof
Scene count
Cut rate
Duration
Voice type
Subtitle density
Music type
Offer framing
Urgency
CTA
Disclosure
Claim risk
```

## 18.3 Example bucket

```text
Category: home organization
Hook: visible problem
Problem shown: < 1.5 sec
Product shown: < 4 sec
Duration: 14–22 sec
Before/after: yes
Large captions: yes
CTA: soft recommendation
```

Support requires replication across multiple products or time windows; one winning video is not a genome rule.

---

# 19. No-Shoot Faceless Creative Factory

## 19.1 Objective

Create controlled creative variants without requiring the operator to film or appear on camera.

## 19.2 Approved input assets

- supplier media with written permission,
- official affiliate creative kits,
- licensed stock media,
- operator-owned media,
- commissioned UGC,
- AI-generated backgrounds and transitions,
- product images whose usage rights are documented.

## 19.3 Prohibited input assets

- downloaded competitor TikTok videos,
- copied marketplace seller media without permission,
- watermark removal,
- deceptive synthetic demonstrations,
- fabricated before/after results,
- product performance not supported by actual evidence.

## 19.4 Default creative variants

```text
A. Problem → solution
B. Three benefits
C. Why it is trending
D. Know before buying
E. Comparison
F. Review-derived objection answer
G. Price/offer alert
H. Use-case tutorial
```

## 19.5 Render-manifest example

```yaml
creative_id: CR-000124-V2
format: vertical_9_16
duration_target_seconds: 18
scenes:
  - start: 0
    end: 2
    asset: problem_image_01
    text: "Kablo karmaşası masanı böyle gösteriyorsa..."
  - start: 2
    end: 7
    asset: product_demo_allowed_02
    text: "Bu düzenleyici kabloları tek noktada topluyor"
  - start: 7
    end: 13
    asset: detail_image_03
    text: "Masa altına veya kenara sabitleniyor"
  - start: 13
    end: 18
    asset: offer_card_generated
    text: "Güncel ürün linki açıklamada"
voiceover: turkish_neutral
captions: large
commercial_disclosure: "#Reklam #Ortaklık"
```

## 19.6 Tool boundary

The OS generates:

- script,
- storyboard,
- scene plan,
- captions,
- disclosure,
- render manifest.

A third-party editor may perform rendering. Direct rendering automation is optional and must not block MVP completion.

---

# 20. Synthetic Creative Quality Gate

Before publication, deterministic and AI-assisted checks evaluate:

```text
Hook present in first 2 seconds
Product appears early enough
Text fits safe area
Duration within planned range
No unsupported claim
No prohibited category
Disclosure visible
Affiliate link valid
Rights evidence present
Price snapshot fresh
Seller score above minimum
```

Possible results:

```text
PASS
PASS_WITH_REVIEW
FAIL_RIGHTS
FAIL_DISCLOSURE
FAIL_CLAIM
FAIL_STALE_OFFER
FAIL_LINK
```

A quality score is not a sales prediction.

---

# 21. Daily Link Portfolio Optimizer

## 21.1 Explore versus exploit

The daily portfolio must reserve capacity for both proven and new ideas.

Default allocation hypothesis:

```text
40% supported / strongest
30% promising
20% emerging
10% novel exploration
```

The exact number of permitted links is read from current platform configuration or operator input. Never hard-code a permanent platform limit.

## 21.2 Constraints

- no duplicate offer unless deliberate retest,
- category concentration cap,
- seller concentration cap,
- compliance gate,
- link validity,
- minimum offer quality,
- no recently rejected bucket,
- exposure balance across experiment arms.

## 21.3 Objective

```text
maximize expected information gain
+ expected net commission
− concentration risk
− compliance risk
− creative fatigue
```

---

# 22. Counterfactual and Causal Research Layer

## 22.1 Questions

- Was the product strong or the hook strong?
- Did a discount cause conversion?
- Was the result payday- or campaign-specific?
- Did a platform algorithm boost contaminate the test?
- Does the same creative structure transfer to another product?
- Does the same product convert under another creative?

## 22.2 Factorial path

```text
Product A × Hook 1
Product A × Hook 2
Product B × Hook 1
Product B × Hook 2
```

Low traffic may make full factorial tests impossible. The system should then label causal confidence as low instead of fabricating certainty.

## 22.3 Next Best Experiment

The research engine ranks tests by:

```text
expected information gain
× commercial relevance
× feasibility
÷ cost
```

---

# 23. Profit Truth Engine

## 23.1 Affiliate truth

```text
Verified Net Affiliate Profit =
  gross commission
− cancelled commission
− returned commission
− paid distribution cost
− outsourced creative cost
− attributable software/tool cost
```

## 23.2 Content efficiency

```text
Net commission per content hour
Net commission per 1,000 qualified views
Net commission per 100 unique clicks
```

## 23.3 Seller truth — later phase

```text
Unit Contribution =
  sale price
− product cost
− supplier shipping
− customs/brokerage
− marketplace/payment fees
− outbound fulfillment
− return reserve
− defect reserve
− customer-service allocation
− tax allocation
− advertising cost
```

## 23.4 Stress case

Every physical product must be evaluated under:

- currency +10%,
- return rate +50% relative,
- shipping cost +20%,
- sale price −10%,
- defect reserve,
- delayed inventory turnover.

No sourcing permission if stress contribution is materially negative unless explicitly approved as a capped learning experiment.

---

# 24. Evidence State Machine

## 24.1 Full states

```text
UNREGISTERED
CANDIDATE
RESEARCHED
READY_FOR_SHADOW
SHADOW_ACTIVE
UNDER_SAMPLED
PROMISING
CONTRADICTORY
CONTAMINATED
FORWARD_VALIDATION
SUPPORTED
REGIME_SPECIFIC
REJECTED
BLOCKED_DATA
BLOCKED_PLATFORM
BLOCKED_COMPLIANCE
BLOCKED_RIGHTS
READY_FOR_SUPPLIER_RESEARCH
SAMPLE_VALIDATION
READY_FOR_MICRO_STOCK
MICRO_STOCK_ACTIVE
SCALE_ALLOWED
SATURATED
DECAYING
RETIRED
```

## 24.2 State meanings

### `CANDIDATE`
Interesting enough to research; no publish permission.

### `UNDER_SAMPLED`
Valid test exists, but exposure or outcome count is insufficient.

### `PROMISING`
Positive signal exists, but replication or net outcome is insufficient.

### `FORWARD_VALIDATION`
Bucket is frozen and being tested in a later time window, creative, platform, or audience.

### `SUPPORTED`
Meets frozen evidence thresholds with acceptable data quality and replication.

### `REGIME_SPECIFIC`
Works only under identified contexts.

### `REJECTED`
Adequate test exposure failed frozen commercial or quality criteria.

### `CONTAMINATED`
Test interpretation is invalid because important variables changed or attribution failed.

### `READY_FOR_MICRO_STOCK`
Affiliate support, sample quality, supplier, compliance, and stress economics all passed.

## 24.3 Default transition example

```text
CANDIDATE
→ RESEARCHED
→ READY_FOR_SHADOW
→ SHADOW_ACTIVE
→ UNDER_SAMPLED
→ PROMISING
→ FORWARD_VALIDATION
→ SUPPORTED
→ READY_FOR_SUPPLIER_RESEARCH
→ SAMPLE_VALIDATION
→ READY_FOR_MICRO_STOCK
```

Direct transitions to `SUPPORTED` or `SCALE_ALLOWED` are prohibited.

---

# 25. Provisional Evidence Thresholds

These are implementation defaults, not proven optimal rules. They must remain configurable and versioned.

## 25.1 Affiliate bucket

```yaml
minimum_unique_clicks_for_rejection: 300
minimum_eligible_orders_for_promising: 3
minimum_eligible_orders_for_supported: 10
minimum_replication_windows_for_supported: 2
maximum_cancel_return_rate_for_supported: 0.20
minimum_net_commission_per_100_clicks_try: 30
maximum_contamination_events: 0
minimum_attribution_confidence: 0.70
```

## 25.2 Creative genome bucket

```yaml
minimum_creatives: 5
minimum_distinct_products: 3
minimum_total_unique_clicks: 500
minimum_replication_windows: 2
```

## 25.3 Sourcing gate

```yaml
affiliate_state_required: SUPPORTED
minimum_verified_orders: 20
sample_approval_required: true
landed_cost_complete: true
stress_contribution_positive: true
compliance_review: PASS
supplier_risk: LOW_OR_MEDIUM
```

Operators may change thresholds only through a versioned configuration and decision record.

---

# 26. Permission Engine

## 26.1 Permission types

```text
ALLOW_RESEARCH
ALLOW_CREATE_ASSET
ALLOW_PUBLISH_ORGANIC
ALLOW_RETEST
ALLOW_PAID_TEST
ALLOW_SUPPLIER_CONTACT
ALLOW_SAMPLE_ORDER
ALLOW_MICRO_STOCK
ALLOW_LIST_MARKETPLACE
ALLOW_SCALE
DENY
REQUIRE_HUMAN_REVIEW
```

## 26.2 Publish permission

Required:

- asset rights documented,
- disclosure present,
- link valid,
- price/stock snapshot fresh,
- seller quality above minimum,
- no prohibited claim,
- no category block,
- experiment arm registered,
- daily portfolio capacity available.

## 26.3 Supplier-contact permission

Required:

- product at least PROMISING,
- enough commercial signal to justify research,
- category compliance not blocked.

## 26.4 Sample-order permission

Required:

- supplier identity and quote recorded,
- specification frozen,
- total sample budget approved,
- no prohibited product class.

## 26.5 Micro-stock permission

Required:

- supported affiliate evidence,
- sample passed,
- landed cost complete,
- stress economics acceptable,
- return route defined,
- invoice and fulfillment flow documented,
- human approval.

---

# 27. Supplier Digital Twin

## 27.1 Supplier dimensions

```text
identity confidence
verified status
Trade Assurance state
response time
quotation completeness
MOQ
sample availability
sample quality
specification match
production lead time
shipping lead time
tracking quality
packaging control
compliance documents
defect policy
refund/dispute behavior
```

## 27.2 Supplier state

```text
UNRESEARCHED
CONTACTED
QUOTED
SAMPLE_ORDERED
SAMPLE_RECEIVED
SAMPLE_PASSED
SAMPLE_FAILED
APPROVED_LIMITED
APPROVED
SUSPENDED
REJECTED
```

## 27.3 Alibaba sequence

```text
Supported product family
→ 3–5 quotations
→ written specification
→ protected order where eligible
→ samples from at least two candidates
→ quality and document review
→ landed cost
→ low MOQ negotiation
→ micro-stock only
```

---

# 28. Multi-Country Opportunity Router — Deferred

The future router evaluates:

```text
product × country × marketplace × traffic platform × fulfillment route
```

Features:

- local demand,
- competition,
- commission/margin,
- language fit,
- customs,
- delivery,
- local regulations,
- payment behavior,
- return cost.

This module must remain disabled until the Turkish MVP has reliable attribution and outcome accounting.

---

# 29. Research Question Tree

The system may generate research questions, but may not answer them without evidence.

Example tree:

```text
Why are clicks rising but orders flat?
├── price increased?
├── seller rating declined?
├── shipping promise worsened?
├── wrong audience attracted?
├── link attribution broken?
├── product page quality poor?
└── hype without purchase intent?
```

Every question records:

```text
question_id
parent_question_id
triggering_evidence
priority
required_data
proposed_test
status
answer_state
```

---

# 30. Human, Deterministic Code, and AI Responsibilities

## 30.1 Deterministic code owns

- data import,
- hashes and identity,
- calculations,
- threshold checks,
- state transitions,
- permissions,
- attribution reconciliation,
- reporting,
- audit history.

## 30.2 AI may assist with

- review classification,
- problem clustering,
- hook classification,
- creative scripts,
- contradiction summaries,
- research-question generation,
- supplier-message drafting.

## 30.3 AI may not

- silently edit historical outcomes,
- change frozen experiments,
- invent sales,
- invent product specifications,
- invent usage rights,
- claim personal experience that did not occur,
- approve sourcing,
- spend money,
- publish without configured approval,
- override compliance blocks.

## 30.4 Human approval remains required for

- publishing until explicitly relaxed,
- external account actions,
- supplier contact,
- sample purchase,
- inventory purchase,
- legal/claim review,
- paid advertising,
- scaling.

---

# 31. Daily Operating Workflow

```text
01. Import affiliate and social reports
02. Validate and quarantine malformed rows
03. Reconcile prior outcomes and returns
04. Run data-health checks
05. Update offer snapshots
06. Update problem, product, and lifecycle features
07. Evaluate buckets
08. Generate contradictions
09. Generate daily portfolio candidates
10. Run permission engine
11. Human reviews selected products and scripts
12. Render faceless assets using approved inputs
13. Validate rights, disclosure, claims, and links
14. Publish manually
15. Snapshot publication metrics
16. Generate daily Markdown report
```

---

# 32. Dashboard Specification

## 32.1 Home

- health status,
- current evidence counts,
- verified net commission,
- unresolved blocks,
- daily portfolio,
- stale links,
- pending human approvals.

## 32.2 Product Radar

Columns:

```text
product
problem
hotness
confidence
velocity state
saturation state
offer quality
creative potential
evidence state
permission
```

## 32.3 Creative Lab

- creative previews,
- genome comparison,
- hook performance,
- product visibility timing,
- publication outcomes,
- fatigue indicators.

## 32.4 Outcome Truth

- gross vs. net commission,
- cancellations,
- returns,
- attribution confidence,
- net commission per click,
- net commission per content hour.

## 32.5 Evidence Registry

- bucket definition,
- version,
- evidence state,
- sample,
- replication,
- contradiction,
- applicable regime,
- decision history.

## 32.6 Supplier Lab — later

- quotes,
- sample status,
- supplier digital twin,
- landed cost,
- sourcing permission.

---

# 33. CLI Specification

```bash
commerce init
commerce db migrate
commerce db status

commerce products import data/imports/products.csv
commerce products register --interactive
commerce products score --as-of 2026-07-05
commerce products lifecycle

commerce reviews import data/imports/reviews.csv
commerce reviews mine

commerce creatives plan --product-id <id> --variants 3
commerce creatives validate --creative-id <id>
commerce creatives manifest --creative-id <id>

commerce links verify
commerce outcomes import data/imports/affiliate_report.csv
commerce outcomes reconcile

commerce evidence evaluate
commerce permissions run
commerce portfolio generate --date 2026-07-06

commerce health check
commerce report daily --date 2026-07-05
commerce report weekly
```

All commands must support `--dry-run` where external or destructive behavior could occur.

---

# 34. Configuration Specification

## 34.1 `scoring.yaml`

```yaml
hotness:
  version: 1
  demand_momentum: 0.18
  review_velocity: 0.14
  social_search_momentum: 0.12
  offer_attractiveness: 0.10
  seller_quality: 0.10
  content_demonstrability: 0.12
  competitive_whitespace: 0.08
  commission_attractiveness: 0.08
  return_safety: 0.04
  compliance_safety: 0.04
```

## 34.2 `permissions.yaml`

```yaml
publish:
  require_rights_evidence: true
  require_disclosure: true
  require_link_fresh_hours: 24
  minimum_seller_rating: 4.0
  prohibited_risk_categories:
    - supplements
    - medical_claims
    - baby_safety
    - counterfeit
    - uncertain_electrical

micro_stock:
  require_human_approval: true
  require_sample_pass: true
  require_stress_margin_positive: true
```

## 34.3 `platforms.yaml`

Platform rules must include `checked_at` and `source_url`. Rules expire and trigger review.

---

# 35. Health Engine

## 35.1 Checks

```text
Database writable
Migrations current
Latest import successful
Affiliate report freshness
Broken link rate
Stale price rate
Unknown commission rate
Missing disclosure count
Missing rights evidence count
Outcome reconciliation gap
Duplicate listing rate
Unresolved quarantine rows
Permission engine freshness
```

## 35.2 Health states

```text
GREEN
YELLOW
RED
BLOCKED
```

Publishing is blocked when:

- database integrity fails,
- link verification is stale,
- outcome reconciliation is materially broken,
- disclosure validation is unavailable,
- permission engine has not run on current inputs.

---

# 36. Reconciliation

## 36.1 Required reconciliations

- publication link ↔ affiliate link,
- affiliate link ↔ platform offer,
- platform report ↔ imported outcome,
- gross commission ↔ cancellations/returns ↔ net commission,
- product snapshot ↔ publication time,
- creative version ↔ publication,
- experiment arm ↔ publication.

## 36.2 Reconciliation report

```text
matched rows
unmatched rows
duplicate rows
late adjustments
negative corrections
currency issues
manual overrides
```

No unmatched commission should be silently assigned to the highest-performing content.

---

# 37. Testing Strategy

## 37.1 Unit tests

- score normalization,
- missing-value handling,
- lifecycle transitions,
- evidence transitions,
- commission math,
- cancellation and return deductions,
- link expiration,
- disclosure rules,
- permission gates,
- stress-margin calculations.

## 37.2 Migration tests

- empty database to latest,
- sequential migration integrity,
- rollback where supported,
- constraints and indexes,
- idempotent migration command.

## 37.3 Integration tests

- product import → scoring → permission,
- creative registration → publication → outcome,
- return adjustment → evidence reevaluation,
- stale offer → publish denial,
- contaminated experiment → blocked support.

## 37.4 Contract tests

Every platform/report adapter gets frozen fixture files and schema-drift detection.

## 37.5 Property tests

Examples:

- net commission can never exceed gross commission after only deductions,
- a blocked rights state can never yield publish ALLOW,
- a bucket version hash cannot change after preregistration,
- an entity cannot be both RETIRED and SCALE_ALLOWED,
- missing outcomes cannot increase confidence.

## 37.6 Golden report tests

Daily and weekly Markdown outputs are compared against reviewed fixtures.

## 37.7 Failure injection

- malformed CSV,
- duplicate orders,
- late return corrections,
- negative commission adjustment,
- stale link,
- missing price,
- database locked,
- interrupted migration,
- AI classification unavailable.

The deterministic core must continue safely when AI modules fail.

---

# 38. Minimum Acceptance Criteria for MVP

## 38.1 Technical

- clean repository created,
- migrations run from zero,
- 100 products import successfully,
- invalid rows quarantined,
- hotness and confidence calculated,
- 3 creative manifests per selected product,
- affiliate links registered and verified,
- outcome report imported,
- cancellations and returns reconcile,
- evidence states update deterministically,
- permission engine blocks invalid publications,
- dashboard and daily report render,
- test suite passes.

## 38.2 Operational

- operator can add a product without editing SQL,
- operator can see why a product was selected,
- operator can see why a product was blocked,
- every score links to input features,
- every permission links to rule version,
- every supported state links to outcome evidence,
- no external post is made automatically.

## 38.3 Not required for MVP

- proven profit,
- automatic video rendering,
- direct Trendyol affiliate API,
- Amazon access,
- marketplace seller integration,
- supplier automation,
- paid advertising.

---

# 39. Implementation Roadmap

## Phase 0 — Audit and isolation, 1–2 days

Deliver:

```text
AMI_REUSE_AUDIT.md
AMI_REUSE_MATRIX.csv
COMMERCE_BOUNDARY.md
repo skeleton
initial migration plan
```

Exit gate:

- no dependency on live trading runtime,
- no copied secrets,
- approved reuse list.

## Phase 1 — Canonical MVP, days 3–7

Build:

- schema,
- manual/CSV ingestion,
- product registry,
- offer snapshots,
- hotness scoring,
- lifecycle states,
- experiments,
- evidence registry,
- permissions,
- CLI.

## Phase 2 — Creative and outcome MVP, days 8–14

Build:

- Creative Genome,
- rights registry,
- faceless render manifests,
- affiliate links,
- publication ledger,
- outcome import,
- reconciliation,
- dashboard,
- reports.

## Phase 3 — 30-day live research

- choose one narrow niche,
- register 50–100 products,
- run controlled organic content,
- collect qualified clicks and sales,
- diagnose traffic versus product failure.

## Phase 4 — 60-day intelligence expansion

- Hot Problem Graph,
- review miner,
- saturation detector,
- Fake-Hype Detector,
- offer router,
- Next Best Experiment.

## Phase 5 — 90-day sourcing decision

Only after supported evidence:

- supplier digital twins,
- quotations,
- samples,
- landed cost,
- micro-stock permission.

---

# 40. First 14 Days — Detailed Build Order

## Day 1

- initialize repository,
- freeze scope,
- audit AMI utilities,
- create decisions log.

## Day 2

- core identity migrations,
- database connection,
- migration CLI,
- audit events.

## Day 3

- products, listings, offers,
- CSV importer,
- validation and quarantine.

## Day 4

- features and hotness,
- score explainability,
- lifecycle engine.

## Day 5

- experiment registry,
- frozen definitions,
- contamination events.

## Day 6

- evidence states,
- bucket versions,
- permission engine.

## Day 7

- first end-to-end dry run,
- technical reconciliation report.

## Day 8

- media rights registry,
- Creative Genome,
- script templates.

## Day 9

- render manifests,
- disclosure validation,
- claim-risk checks.

## Day 10

- affiliate links,
- publication ledger,
- link verification.

## Day 11

- outcome importer,
- cancellation and return adjustment.

## Day 12

- daily portfolio optimizer,
- dashboard core.

## Day 13

- first 50–100 candidate dataset,
- generate first controlled portfolio,
- create assets.

## Day 14

- canonical review,
- test suite,
- operator runbook,
- first live publication only after approval.

---

# 41. 30/60/90-Day Decision Gates

## Day 30

Useful evidence requires:

- reliable attribution,
- 60–100 controlled contents or a justified lower count,
- at least 1,000 qualified clicks **or** a clear conclusion that traffic generation failed,
- no critical compliance breach,
- first product/content hypotheses.

## Day 60

Expected:

- at least one promising bucket,
- measured gross-to-net commission gap,
- evidence on strongest traffic platform,
- evidence on strongest creative structure,
- clear rejection library.

## Day 90

Choose one:

```text
SCALE_AFFILIATE
CONTINUE_RESEARCH
PIVOT_NICHE
START_SUPPLIER_RESEARCH
REJECT_MODEL
```

No inventory purchase is mandatory.

---

# 42. Kill and Pivot Criteria

Pause or pivot when:

- platform access remains blocked,
- attribution cannot be trusted,
- content production is operationally unsustainable,
- adequate qualified traffic produces no viable orders,
- cancellation/return rates erase economics,
- commissions are too low relative to labor,
- rights-compliant assets cannot be sourced,
- compliance risk is unacceptable,
- the system cannot distinguish traffic failure from product failure.

A rejected model is a valid, valuable research result.

---

# 43. Security and Privacy

- secrets in environment variables or secret manager,
- tracking URLs encrypted at rest where appropriate,
- no unnecessary customer PII,
- reviewer identities not stored,
- logs redact tokens and full tracked URLs,
- write actions require explicit command and approval,
- database backups encrypted,
- supplier documents access-controlled,
- model prompts exclude secrets.

---

# 44. Retention and Storage

Commerce data is smaller than microstructure data but still requires discipline.

```text
Active SQLite: current operational records
Monthly Parquet archive: snapshots and historical observations
Permanent: outcomes, evidence, bucket definitions, permissions, audit events
Configurable retention: raw media derivatives and intermediate render files
```

Do not delete evidence required to reproduce a decision.

---

# 45. Reports

Required canonical reports and governance files:

```text
SYSTEM_STATE.md
HANDOFF.md
IMPLEMENTATION_ROADMAP.md
RECONCILIATION_LOG.md
DECISIONS.md
CHANGELOG.md
DAILY_PORTFOLIO_YYYY-MM-DD.md
DAILY_OUTCOME_YYYY-MM-DD.md
WEEKLY_EVIDENCE_REVIEW_YYYY-WW.md
REJECTED_PRODUCTS_REGISTER.md
CONTRADICTIONS_REGISTER.md
DATA_HEALTH.md
PLATFORM_RULES_STATUS.md
SUPPLIER_VALIDATION.md
```

Each report states:

- data cutoff,
- source coverage,
- missing data,
- assumptions,
- evidence states,
- permissions,
- unresolved questions.

---

# 46. Recommended First Niche Universe

Do not freeze a permanent niche before research. Initial low-regulatory-risk candidates:

```text
home organization
desk and workspace organization
pet-owner convenience
travel organization
car organization excluding safety-critical parts
non-medical fitness accessories
kitchen organization excluding uncertain electrical items
```

Avoid Phase 1:

```text
supplements
ingestibles
medical claims
baby safety
protective equipment
cosmetics with uncertain compliance
electrical products with uncertain conformity
counterfeit/branded copies
weapons or hazardous goods
```

---

# 47. Example Canonical Bucket

```yaml
bucket_id: AFF_TR_HOMEORG_PROBLEMDEMO_V1
business_model: affiliate
affiliate_platform: trendyol
traffic_platform: tiktok
problem_cluster: small_space_clutter
category: home_organization
price_band_try: [400, 900]
creative:
  hook_type: visible_problem
  problem_visible_before_sec: 1.5
  product_visible_before_sec: 4
  before_after: true
  duration_sec: [14, 22]
  disclosure_required: true
qualification:
  unique_clicks_min: 300
  eligible_orders_min: 10
  replication_windows_min: 2
  attribution_confidence_min: 0.70
  cancel_return_rate_max: 0.20
  net_commission_per_100_clicks_min_try: 30
state: UNDER_SAMPLED
permission: ALLOW_CONTROLLED_ORGANIC_TEST
```

---

# 48. Example Failure Diagnosis

```text
Product: cable organizer
Views: 120,000
Unique clicks: 240
Eligible orders: 1
Net commission: 18 TL
```

Diagnosis:

```text
view_to_click = weak
click_to_order = weak
commercial result = REJECTED or UNDER_SAMPLED depending threshold
possible causes:
- entertainment-heavy creative
- weak offer
- low purchase urgency
- wrong audience
```

The system may not call this a hot winner merely because views are high.

---

# 49. Example Hidden Alpha

```text
Product: compact drawer divider
Views: 8,000
Unique clicks: 420
Eligible orders: 18
Net commission: 640 TL
Return/cancel: low
```

Possible state:

```text
PROMISING
→ forward validation with a new creative and week
```

Low reach with strong purchase conversion can be more valuable than viral reach.

---

# 50. Claude Implementation Protocol

Paste this whitepaper into Claude together with read-only access to the AMI repository and use the following instruction.

## Canonical implementation prompt

```text
You are operating inside the AMI × Commerce Intelligence OS v1.1
multi-model execution protocol. The new isolated repository is named
commerce_intelligence.

Before doing any work, read SYSTEM_STATE.md, HANDOFF.md,
IMPLEMENTATION_ROADMAP.md, RECONCILIATION_LOG.md, and this whitepaper.
Obey the CURRENT_MODEL_ROLE in HANDOFF.md. Never perform work assigned to a
different model role.

First, audit the existing AMI repository. Do not modify it. Classify reusable
code as DOMAIN_AGNOSTIC_CORE, TRADING_COUPLED, RUNTIME_DANGEROUS, DATA_ONLY,
or DOCUMENTATION_ONLY. Produce AMI_REUSE_AUDIT.md and AMI_REUSE_MATRIX.csv.

Do not copy exchange credentials, live trading services, collectors, trading
databases, schedulers, order executors, or trading-specific assumptions.

Then create D:\commerce_intelligence as an independent repository. Implement
only Phase 0, Phase 1, and Phase 2. The architecture must remain compatible
with later phases, but do not implement seller execution, paid-ad automation,
automatic public posting, supplier purchasing, inventory purchasing, or
marketplace order execution.

Use canonical SQL as the structured source of truth. Preserve append-only
evidence and audit history. Bucket definitions must be immutable after
preregistration. All score and permission rules must be configurable and
versioned. Missing data must not increase confidence.

For every implementation batch:
1. Confirm that HANDOFF.md assigns the current batch to your model role.
2. State scope and files to be changed.
3. Implement migrations first.
4. Add or update tests.
5. Run the full relevant test suite.
6. Run a clean-database migration test.
7. Produce or append the reconciliation report.
8. Update SYSTEM_STATE.md and RECONCILIATION_LOG.md.
9. Write the next required model role and action into HANDOFF.md.
10. Stop at the required handoff boundary and tell the user exactly which
    model to select next.

No model may silently continue into another model's role. No external write action may occur. Use manual/CSV fixtures for platform data.
The MVP is complete only when all acceptance criteria in the whitepaper pass.
```

## Batch order

```text
BATCH 0 — audit and boundary
BATCH 1 — repository, settings, DB, migrations
BATCH 2 — products, listings, offers, ingestion
BATCH 3 — features, hotness, lifecycle
BATCH 4 — experiments, buckets, evidence
BATCH 5 — permissions and health
BATCH 6 — creative genome, rights, render manifests
BATCH 7 — affiliate links, publications, outcomes
BATCH 8 — reconciliation, portfolio, reports, dashboard
BATCH 9 — integration tests, documentation, operator runbook
```

---


# 51. Mandatory Multi-Model Execution and Handoff Protocol

This protocol is mandatory for the new Commerce Intelligence repository. Model memory, prior chat context, or informal understanding must never be treated as the canonical handoff mechanism.

The canonical transition truth is:

```text
HANDOFF.md
+ SYSTEM_STATE.md
+ IMPLEMENTATION_ROADMAP.md
+ RECONCILIATION_LOG.md
```

A chat message is required for the human operator, but the chat message alone is not canonical.

## 51.1 Model roles

### FABLE — Architecture, audit, planning, and reconciliation

Fable is responsible for:

- reading the complete whitepaper,
- auditing `D:\eclipse_scalper` in read-only mode,
- classifying reusable and forbidden components,
- identifying important canonical files,
- producing the dependency graph,
- producing and freezing the implementation roadmap,
- defining batch boundaries and acceptance criteria,
- reviewing implementation waves,
- identifying architectural drift, contamination, missing tests, and false completion claims,
- accepting a wave or requiring a corrective batch.

Fable must not:

- implement production batches,
- modify `D:\eclipse_scalper`,
- silently begin Sonnet work,
- mark code complete without test and reconciliation evidence,
- continue after the planning handoff boundary.

### SONNET — Controlled implementation

Sonnet is responsible for:

- implementing only the roadmap and batch assigned in `HANDOFF.md`,
- creating migrations before dependent code,
- writing and running tests,
- updating documentation,
- producing reconciliation evidence,
- updating `SYSTEM_STATE.md`,
- updating `RECONCILIATION_LOG.md`,
- preparing the next review handoff.

Sonnet must not:

- redesign the frozen architecture without a recorded Fable decision,
- change frozen bucket definitions without versioning,
- implement later batches early,
- modify the trading repository,
- skip tests or clean-database migration checks,
- declare a wave accepted,
- silently continue into the next wave before review when review is required.

### OPUS — Independent critical review

Opus is the preferred independent reviewer for major implementation waves when available.

Opus is responsible for:

- reviewing the frozen roadmap against the implementation,
- checking schema, migration, test, state, and evidence consistency,
- searching for hidden coupling to the trading runtime,
- identifying silent data corruption or false confidence,
- issuing `ACCEPTED`, `CORRECTIVE_BATCH_REQUIRED`, or `BLOCKED`.

When Opus is unavailable, Fable performs the independent review role. The same review contract applies.

## 51.2 Required phase state machine

```text
UNINITIALIZED
    ↓
FABLE_AUDIT_ACTIVE
    ↓
FABLE_PLANNING_COMPLETE
    ↓
WAITING_FOR_SONNET
    ↓
SONNET_BATCH_ACTIVE
    ↓
SONNET_BATCH_COMPLETE
    ↓
WAITING_FOR_REVIEW
    ↓
REVIEW_ACTIVE
    ├── ACCEPTED → WAITING_FOR_SONNET_NEXT_BATCH
    ├── CORRECTIVE_BATCH_REQUIRED → WAITING_FOR_SONNET_CORRECTION
    └── BLOCKED → OPERATOR_DECISION_REQUIRED
    ↓
FINAL_RECONCILIATION
    ↓
MVP_ACCEPTED
```

No state may be skipped merely because the model believes the next action is obvious.

## 51.3 Canonical `HANDOFF.md` schema

`D:\commerce_intelligence\HANDOFF.md` must always exist after Phase 0 initialization and contain the following fields:

```yaml
handoff_version: 1
project: commerce_intelligence

current_model_role: FABLE
current_phase: ARCHITECTURE_AUDIT
status: ACTIVE

current_batch: BATCH_0
current_wave: PLANNING

source_commit: null
target_commit: null
roadmap_version: null

completed_actions: []
open_blockers: []
required_files_read: []
required_files_written: []

next_model_role: FABLE
next_action: COMPLETE_REUSE_AUDIT
next_batch: BATCH_0

review_required: false
review_model_preference: OPUS
review_fallback: FABLE

stop_boundary:
  required: true
  condition: PLANNING_COMPLETE

operator_message_required: true
updated_at: 2026-07-05T00:00:00+03:00
updated_by_role: FABLE
```

The exact timestamp changes on every update. A model may not begin work when `current_model_role` does not match its assigned role.

## 51.4 Fable planning completion contract

Fable planning is complete only when all of the following exist and agree:

```text
AMI_REUSE_AUDIT.md
AMI_REUSE_MATRIX.csv
COMMERCE_BOUNDARY.md
DEPENDENCY_GRAPH.md
IMPLEMENTATION_ROADMAP.md
SYSTEM_STATE.md
HANDOFF.md
```

Fable then writes:

```yaml
current_model_role: FABLE
current_phase: PLANNING_COMPLETE
status: COMPLETE
next_model_role: SONNET
next_action: IMPLEMENT_BATCH_1
next_batch: BATCH_1
review_required: false
```

Fable must stop implementation work and display this operator message:

```text
Fable audit and implementation planning are complete.

Canonical files have been written and the roadmap is frozen.
Do not continue implementation in Fable.

Next model: Sonnet
Next action: Read HANDOFF.md, SYSTEM_STATE.md, and
IMPLEMENTATION_ROADMAP.md, then implement the batch identified in HANDOFF.md.
```

Equivalent Turkish wording is acceptable, but the model name and exact next action must be explicit.

## 51.5 Sonnet batch completion contract

A Sonnet batch is complete only when:

- assigned scope is implemented,
- migrations pass on a clean database,
- relevant tests pass,
- no unexplained test exclusion exists,
- reconciliation is written,
- state and documentation agree with the code,
- blockers are honestly recorded,
- the Git diff stays within the approved boundary.

Sonnet then writes:

```yaml
current_model_role: SONNET
current_phase: IMPLEMENTATION_BATCH_COMPLETE
status: COMPLETE
next_model_role: OPUS
next_action: REVIEW_CURRENT_WAVE
review_required: true
```

When Opus is unavailable:

```yaml
next_model_role: FABLE
next_action: REVIEW_CURRENT_WAVE
review_required: true
```

Sonnet must stop and display:

```text
The assigned Sonnet implementation batch is complete.

Tests, clean-database migration, reconciliation, SYSTEM_STATE.md, and
HANDOFF.md have been updated.

Next model: Opus for independent review.
Fallback: Fable if Opus is unavailable.
Do not begin the next implementation batch before review.
```

## 51.6 Review output contract

The reviewer must choose exactly one result.

### ACCEPTED

```yaml
review_result: ACCEPTED
next_model_role: SONNET
next_action: IMPLEMENT_NEXT_APPROVED_BATCH
review_required: false
```

Operator message:

```text
Review result: ACCEPTED.

Next model: Sonnet
Next action: Implement the next batch identified in HANDOFF.md.
```

### CORRECTIVE_BATCH_REQUIRED

```yaml
review_result: CORRECTIVE_BATCH_REQUIRED
next_model_role: SONNET
next_action: IMPLEMENT_CORRECTIVE_BATCH
review_required: false
```

The reviewer must list:

- defect,
- severity,
- affected files,
- required test,
- acceptance condition,
- whether roadmap revision is required.

Operator message:

```text
Review result: CORRECTIVE_BATCH_REQUIRED.

Next model: Sonnet
Next action: Apply only the corrective scope recorded in HANDOFF.md and the
review report. Do not start the next planned batch.
```

### BLOCKED

```yaml
review_result: BLOCKED
next_model_role: HUMAN_OPERATOR
next_action: RESOLVE_RECORDED_BLOCKER
review_required: false
```

The reviewer must state why neither implementation nor safe correction can continue.

## 51.7 Roadmap change control

After Fable freezes `IMPLEMENTATION_ROADMAP.md`, any architectural change requires:

```text
DECISIONS.md entry
+ roadmap version increment
+ affected-batch analysis
+ migration impact analysis
+ test impact analysis
+ Fable or Opus approval
```

Sonnet may suggest a change but cannot silently adopt it.

## 51.8 `SYSTEM_STATE.md` mandatory model fields

The top of `SYSTEM_STATE.md` must contain:

```yaml
project: commerce_intelligence
system_state_version: 1
current_phase: FABLE_AUDIT_ACTIVE
current_batch: BATCH_0
current_model_role: FABLE
next_model_role: FABLE
handoff_status: ACTIVE
roadmap_version: null
last_review_result: null
last_verified_commit: null
critical_blockers: []
```

The human operator must be able to open one file and immediately see:

- where the project is,
- which model should be active,
- what was completed,
- what remains,
- whether review is required,
- what model to select next.

## 51.9 `RECONCILIATION_LOG.md` entries

Every completed batch or review appends an immutable entry:

```text
## 2026-07-05 — BATCH_1 — SONNET

Scope:
Files changed:
Migrations:
Tests run:
Tests passed:
Tests failed:
Clean DB result:
Known limitations:
Reconciliation result:
Commit:
Next role:
Next action:
```

Corrections are appended; previous entries are not rewritten to hide mistakes.

## 51.10 Important file discovery during audit

Fable must not assume it already knows the important AMI files. It must inspect the source repository and classify discovered artifacts.

At minimum, it should search for:

```text
SYSTEM_STATE.md
canonical whitepapers
reconciliation reports
architecture protocols
migration maps
test manifests
evidence registries
permission specifications
dashboard specifications
data dictionaries
operator runbooks
retention and archive policies
```

Only files actually found and verified should be listed as canonical dependencies.

## 51.11 Session-resume protocol

At the start of every new Claude session or model change:

```text
1. Read HANDOFF.md.
2. Read the top and latest relevant section of SYSTEM_STATE.md.
3. Read IMPLEMENTATION_ROADMAP.md.
4. Read the latest RECONCILIATION_LOG.md entry.
5. Confirm assigned role and scope.
6. Refuse to continue if files conflict.
7. Resolve conflicts through Fable review or operator decision.
```

No model may rely on chat memory instead of these files.

## 51.12 Handoff integrity tests

The repository test suite must include governance checks that fail when:

- `HANDOFF.md` is missing,
- required fields are absent,
- `current_model_role` and `SYSTEM_STATE.md` disagree,
- a completed batch lacks reconciliation,
- a review-required state points directly to another Sonnet batch,
- an unaccepted batch is marked complete in `SYSTEM_STATE.md`,
- roadmap version and state version conflict,
- a model changes files outside the assigned batch boundary.

A lightweight parser and CI test should validate these conditions.

## 51.13 Final MVP acceptance handoff

After the last implementation batch:

```text
SONNET_FINAL_BATCH_COMPLETE
→ OPUS_OR_FABLE_FINAL_REVIEW
→ FINAL_RECONCILIATION
→ MVP_ACCEPTED
```

The final reviewer must verify the complete Definition of Done. File creation alone is never sufficient.

The final operator message must state either:

```text
MVP ACCEPTED
```

or:

```text
MVP NOT ACCEPTED
```

with exact failed criteria.

---

# 52. Required Claude Deliverables

```text
AMI_REUSE_AUDIT.md
AMI_REUSE_MATRIX.csv
COMMERCE_BOUNDARY.md
DEPENDENCY_GRAPH.md
INITIAL_IMPLEMENTATION_PLAN.md
IMPLEMENTATION_ROADMAP.md
SYSTEM_STATE.md
HANDOFF.md
RECONCILIATION_LOG.md
DECISIONS.md
CHANGELOG.md
DATA_DICTIONARY.md
MIGRATION_MAP.md
PERMISSION_RULES.md
OPERATOR_RUNBOOK.md
TEST_REPORT.md
RECONCILIATION_REPORT.md
```

Claude must not mark a phase complete based only on file creation. Completion requires passing acceptance criteria and tests.

---

# 53. Definition of Done

The implementation-ready MVP is done when:

```text
[ ] Independent repository exists
[ ] Existing AMI repo unchanged
[ ] HANDOFF.md exists and passes schema validation
[ ] SYSTEM_STATE.md and HANDOFF.md model roles agree
[ ] Fable planning stop boundary is enforced
[ ] Sonnet cannot advance past required review
[ ] Review result determines the next model role
[ ] Every completed batch has a reconciliation entry
[ ] Clean migrations pass
[ ] 100 candidate products can be imported
[ ] Scores are explainable and versioned
[ ] Lifecycle states update deterministically
[ ] Creative manifests can be generated
[ ] Rights and disclosure blocks work
[ ] Affiliate links and publications are registered
[ ] Outcome files reconcile gross, returns, and net
[ ] Evidence buckets are immutable after freeze
[ ] Permission engine produces reason codes
[ ] Dashboard and Markdown reports work
[ ] Health engine blocks unsafe operation
[ ] Full relevant tests pass
[ ] No external write automation exists
[ ] Operator runbook is complete
```

Commercial profitability is not part of software Definition of Done. It is a later evidence outcome.

---

# 54. Future Extension Backlog

## High priority after live data

- Hot Problem Graph automation,
- product identity resolution improvements,
- review miner evaluation set,
- saturation and copycat detector,
- Fake-Hype Detector,
- creative fatigue,
- Bayesian or sequential evidence updates,
- better attribution uncertainty,
- Next Best Experiment ranking.

## After supported affiliate buckets

- supplier digital twin,
- sample workflow,
- landed-cost engine,
- local fulfillment research,
- micro-stock experiments,
- seller API adapters.

## Much later

- paid creative testing,
- multi-country routing,
- private label,
- autonomous inventory recommendations,
- multi-tenant SaaS,
- client-facing intelligence reports.

---

# 55. Final Strategic Position

The strongest version of this project is not:

```text
Find viral product → post link → hope
```

It is:

```text
Discover persistent problems
→ identify early product solutions
→ measure trend and saturation
→ select the best consumer offer
→ generate rights-safe faceless creative hypotheses
→ run controlled affiliate tests
→ reconcile sales, cancellations, and returns
→ separate hype from repeatable net value
→ validate suppliers only for supported products
→ buy minimal inventory only after permission
→ scale with evidence
```

The system’s durable advantage is not a secret product list. It is the accumulating canonical knowledge of:

- what sells,
- what does not,
- why it appears to sell,
- where it stops working,
- which creative structures transfer,
- which suppliers can be trusted,
- and which unknowns still block action.

---

# 56. Official Source Register

Verified on **5 July 2026**. Platform rules can change; `platforms.rules_checked_at` must trigger periodic review.

[^trendyol_affiliate]: Trendyol, “Affiliate Programı.” Current page states 24-hour tracked-link attribution conditions, deduction of cancelled/returned products, and invoice workflow through the Influencer Panel. https://www.trendyol.com/s/trendyol-affiliate-programi

[^trendyol_application]: Trendyol Influencer Affiliate Program application portal for Türkiye and Azerbaijan. https://influencer.trendyol.com/

[^amazon_review]: Amazon Türkiye Associates, “Başvuru İnceleme Süreci.” Current guidance states at least three qualifying sales within the first 180 days, public original content, and a baseline of at least ten public posts. https://gelirortakligi.amazon.com.tr/help/node/topic/G8TW5AE9XL2VX9VM

[^trendyol_api]: Trendyol Developers, official Marketplace integration documentation. https://developers.trendyol.com/

[^customs_change]: T.C. Ticaret Bakanlığı, “E-İthalatta Basitleştirilmiş Gümrük Beyannamesi Kapsamının Değiştirilmesine İlişkin Yeni Düzenleme Hakkında Basın Açıklaması,” 7 January 2026. https://ticaret.gov.tr/haberler/e-ithalatta-basitlestirilmis-gumruk-beyannamesi-kapsaminin-degistirilmesine-iliskin-yeni-duzenleme-hakkinda-basin-aciklamasi

[^postal_cargo]: T.C. Ticaret Bakanlığı, “Posta ve Hızlı Kargo Muafiyeti,” updated 5 March 2026. https://ticaret.gov.tr/gumruk-islemleri/sikca-sorulan-sorular/bireysel/posta-ve-hizli-kargo-muafiyeti

[^etbis]: T.C. Ticaret Bakanlığı, ETBİS Sıkça Sorulan Sorular. https://etbis.ticaret.gov.tr/tr/SSS

[^influencer_guide]: T.C. Ticaret Bakanlığı, “Sosyal Medya Etkileyicileri Tarafından Yapılan Ticari Reklam ve Haksız Ticari Uygulamalar Hakkında Kılavuz,” contained in the Ministry’s consumer-law compilation, pp. 254–258. https://tuketici.ticaret.gov.tr/data/5e81982d13b876a1b04c7a42/2023-6502%20Say%C4%B1l%C4%B1%20T%C3%BCketicinin%20Korunmas%C4%B1%20Hakk%C4%B1nda%20Kanun.pdf

[^tiktok_top_ads]: TikTok Creative Center, Top Ads. https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/en

[^tiktok_symphony]: TikTok for Business, Creative Tools / Symphony Creative Studio. https://ads.tiktok.com/business/creativecenter/tools/pc/en

[^meta_library]: Meta Ad Library. https://www.facebook.com/ads/library

[^trade_assurance]: Alibaba.com, Trade Assurance. https://tradeassurance.alibaba.com/

---

# Closing Canon

```text
Do not confuse virality with intent.
Do not confuse clicks with orders.
Do not confuse gross commission with net profit.
Do not confuse affiliate demand with sourcing permission.
Do not confuse code completion with commercial validation.
Do not scale what the system cannot explain.
Do not erase rejected evidence.
Do not fabricate certainty where data is missing.
```
