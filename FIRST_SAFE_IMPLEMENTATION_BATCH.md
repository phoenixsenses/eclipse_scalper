# FIRST_SAFE_IMPLEMENTATION_BATCH

**Batch ID:** BATCH-P1-001 · **Phase:** 1 (Canonical reconciliation) · **Model:** Sonnet 5
**Hazırlayan:** Fable 5, 2026-07-03

## Scope
Canonical warehouse iskeletini kur — YENİ dosya `data/ami/canonical.sqlite` + `ami/warehouse/` modülü. Yalnız şema + idempotent init + testler. Hiçbir mevcut store'a yazılmaz, hiçbir proses etkilenmez.

## Sözleşme

| Alan | Değer |
|---|---|
| dependencies | Phase 0 artifact'leri (tamam) |
| preconditions | git temiz olmayabilir (taban kabul); protected diff boş |
| new_files | `ami/warehouse/__init__.py`, `ami/warehouse/schema.py`, `ami/warehouse/init_db.py`, `tests/test_ami_warehouse_schema.py` |
| changed_files | YOK (mevcut dosyalara dokunulmaz) |
| protected_untouched | UNTOUCHED_LIVE_SHADOW_COMPONENTS.md listesi aynen |
| schema_effect | Yeni SQLite dosyası; tablolar: artifact_registry, artifact_lineage, question_registry, question_families, contradiction_registry, operator_decision_queue, namespace_registry, schema_versions (Protocol §7.1 + §16 alanları; her tabloda schema_version + created_at + provenance) |
| runtime_effect | SIFIR (hiçbir çalışan proses bu dosyayı bilmez) |
| scientific_effect | SIFIR (veri yok, yalnız şema) |
| migration | MIGRATION_LOG.md'ye M-0001 kaydı; WAL + busy_timeout (mevcut AMI store pratiğiyle aynı) |
| tests | 1 test dosyası: şema oluşturma idempotent, round-trip insert/select, version kaydı, RO-açılım. pytest --basetemp scratchpad -p no:cacheprovider |
| rollback | `data/ami/canonical.sqlite` + `ami/warehouse/` + test dosyasını sil; başka hiçbir iz yok |
| documentation | SYSTEM_STATE §40+1, IMPLEMENTATION_PROGRESS_LEDGER, TEST_STATUS_LATEST, SCHEMA_DICTIONARY, MIGRATION_LOG |

## Definition of Done
1. `python -m ami.warehouse.init_db` idempotent çalışır (2. koşum no-op).
2. Test dosyası yeşil (tek pytest çağrısı, ≤2 dosya kuralı).
3. `git diff` yalnız new_files + dokümantasyon gösterir; protected diff boş.
4. SCHEMA_DICTIONARY.md tablolarla senkron.
5. Hiçbir izin/verdict değişmedi.

## Sonraki batch önizlemesi
BATCH-P1-002: read-only artifact discovery → artifact_registry ingest (hash+rol+statü; AMI-acronym collision gate). BATCH-P1-003: QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv + 14 slug-soru → question_registry seed.
