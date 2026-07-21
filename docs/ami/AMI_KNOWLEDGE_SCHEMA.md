# AMI Knowledge Schema (canonical, v1)

Kaynak kod: `ami/knowledge/objects.py` (tek doğruluk noktası). Bu doküman insan-okur özetidir.

## KnowledgeObject

| Alan | Tip | Not |
|---|---|---|
| knowledge_id | str | `K-<aile>-<konu>-NNN` |
| claim | str | boş olamaz |
| claim_type | enum | DESCRIPTIVE/PREDICTIVE/MECHANISTIC/CAUSAL/OPERATIONAL/META_RESEARCH |
| status | enum | Appendix C statüleri; ladder `PROMOTION_LADDER` |
| provenance | Provenance | source_tables + data_time_range + code_ref ZORUNLU |
| mechanism, direction | str | |
| effect_size | dict | ör. `{delta_bps: 70}` |
| evidence_level / required_evidence_level | int enum | Level 0-9 (Part XII §5) |
| evidence_families | list | bağımlılık modeli (§6) — benzer testler tek aile |
| replications / holdouts / forward_events | int | promotion gate girdileri |
| contradictions | list[str] | LIVE/SIZING iznini otomatik bloklar |
| scope | dict | symbols/sessions/regimes/timeframes/venues |
| assumptions | list[str] | governor.invalidate_assumption kaskadı |
| confidence | dict | decomposed (§9): statistical/mechanism/forward/execution/generalization |
| falsification | list[str] | MECHANISTIC/CAUSAL için zorunlu |
| permitted / forbidden | list[Permission] | çakışma = ConstitutionViolation |
| decay_half_life_days | float | is_fresh = yaş ≤ 2×half-life |
| version / history / frozen / freeze_hash | | touch_version → forward_events=0 (§74) |

## Executable contract (§71)
`is_applicable(context)`, `is_fresh()`, `is_permitted(permission)`,
`required_evidence_gap()`, `freeze()`, `touch_version(reason)`

## Promotion gates (governor, §23)
| Hedef statü | Koşul |
|---|---|
| PRELIMINARY | evidence ≥ IN_SAMPLE |
| REPLICATED | evidence ≥ CHRONOLOGICAL, replications ≥ 1 |
| HOLDOUT_VALIDATED | evidence ≥ UNTOUCHED_HOLDOUT, holdouts ≥ 1 |
| FORWARD_VALIDATING | + frozen=True |
| OPERATIONAL_CANDIDATE | evidence ≥ FORWARD_SHADOW, forward_events ≥ 20, çelişki yok |
| PROVISIONALLY_ACCEPTED | evidence ≥ CONTROLLED_PAPER, forward_events ≥ 40, çelişki yok |

Atlama girişimi = `ConstitutionViolation`. Demotion izinleri otomatik söker.

## Graph ilişkileri (§21)
SUPPORTS, CONTRADICTS, DEPENDS_ON, REQUIRES, INVALIDATES, RESTRICTS,
SUPERSEDES, CAN_COEXIST_WITH, CANNOT_COEXIST_WITH, DERIVED_FROM
— CONTRADICTS edge'i her iki objeye `contradictions` olarak yansır.

## Diğer nesneler
- **ResearchQuestion / Hypothesis / ExperimentSpec / EvidenceBundle:** `ami/research/registry.py`
  — spec.freeze() olmadan kayıt yok; hash değişirse kanıt eklenemez.
- **StateObject / StateBundle:** `ami/states/objects.py` — her state timeframe + data_quality taşır.
- **PermissionDecision / DecisionTrace:** `ami/governance/governor.py`, `ami/decision/trace.py`
  — her karar `data/ami/decisions.jsonl`'a immutable paket olarak yazılır.
