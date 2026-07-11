# SCHEMA_AND_DATA_MIGRATION_MAP

**Tarih:** 2026-07-03. İlke: mevcut store'lar TAŞINMAZ, referanslanır; tüm yeni şema `data/ami/canonical.sqlite`'ta; her migration MIGRATION_LOG.md'ye; irreversible/destructive migration = operatör onayı.

## Mevcut store'lar (kaynak, değişmez)
- `data/microstructure.db` — RO kaynak (event/path ingest)
- `data/ami/knowledge.sqlite`, `research.sqlite`, `decisions.jsonl` — AMI çekirdek; warehouse REFERANSLAR (kopyalamaz)
- `reports/shadow/*.jsonl` — shadow kaynak; RO ingest
- `reports/research/s34/S34_ALL.db`, `mechanism_store.sqlite` — meta kaynaklar; artifact registry'ye kayıt

## Yeni: data/ami/canonical.sqlite (Protocol §7 ailesi, fazlara bölünmüş)
| Faz | Tablolar |
|---|---|
| P1 (M-0001..3) | artifact_registry, artifact_lineage, question_registry, question_families, contradiction_registry, operator_decision_queue, namespace_registry, schema_versions |
| P2 (M-0004..5) | evidence_contamination, researcher_exposure, mt_family_registry, data_quality_events, market_structure_versions |
| P3 (M-0006..8) | ami_events, ami_cycles, event_cycle_membership, ami_event_paths (+event_family, source_quality, definition_version zorunlu) |
| P4-5 (M-0009..) | chart_candles/swings/levels/pushes (+known_at_ts), feature_dictionary |
| P8 | forward_events, forward_snapshots, forward_positions, forward_path_ledger, observer tabloları (Obs §7/12/13/30; activation_ts frozen) |

Kurallar: her tabloda PK + schema_version + provenance + created_at; WAL + busy_timeout; idempotent init; SQLite (mevcut pratikle uyumlu); dump round-trip testi P1 DoD'unda.
