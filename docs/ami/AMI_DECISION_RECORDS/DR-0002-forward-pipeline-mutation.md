# DR-0002 — Forward Evidence Pipeline + Mutation Suite Kararları (2026-07-02)

**Durum:** KABUL — operatör paketi (Paket 1-3 + faz kapısı).

## Kararlar

1. **Binding modeli:** Forward kanıt akışı `forward_bindings` tablosuyla deney↔bilgi↔sinyal
   üçlüsüne bağlanır. Binding, freeze anındaki `spec_hash`, `candidate_version`,
   `dataset_hash`, `execution_model`'i DONDURUR; her `run_once` başında dördü de yeniden
   doğrulanır — biri değiştiyse `BINDING_INVALID` (evidence taşınamaz, R2).
2. **Trade-başına EvidenceBundle:** Agregat yerine trade başına kanıt. Gerekçe: dedupe
   (PK), audit izlenebilirliği ve ileride per-trade counterfactual analiz. `processed_trades`
   kalıcı — restart duplicate'i yapısal olarak imkânsız (m12 kanıtı).
3. **Pipeline izin VERMEZ:** Yalnız governor gate'lerine başvurur (promote çağrısı bile
   `ConstitutionViolation` ile geri dönebilir — m19). Falsifikasyonda otomatik
   DEMOTE + failure archive (m15).
4. **E-MECHCOMP-FWD-001 bilinçli BAĞLANMADI:** Shadow runner mech_score loglamıyor;
   loglanmayan feature'a bağlanan deney sahte kanıt üretirdi. Conviction-composite için
   ayrı deney (E-CONVCOMP-FWD-001, conviction>=4) açıldı — dürüst eşleme.
5. **Mutation suite tek kaynak:** Senaryolar `ami/mutation_suite.py`'de fonksiyon;
   pytest sarmalayıcı + MD rapor üretici aynı fonksiyonları çağırır (çift bakım yok).
6. **KnowledgeStore WAL + busy_timeout=30s:** m13 eşzamanlı yazma tutarlılığı için.
7. **MFE50 deneyi Research OS üzerinden:** Spec, HESAPLAMADAN ÖNCE gerçek registry'de
   donduruldu; TRAIN kural seçim protokolü spec metninde frozen; sonuç ne olursa olsun
   knowledge/failure sistemine yazılır. Bu, "Research OS'i kendi araştırmamızda kullan"
   ilkesinin ilk gerçek uygulaması.

## Reddedilenler
- Ledger'ı DB'ye migrate etmek (JSONL yeterli, salt-okuma).
- Pipeline'ı shadow runner içine gömmek (governance düzlemi ayrı kalmalı; runner'a dokunmadık).
- Mutation testlerini yalnız pytest'te tutmak (operatör raporu MD istiyor → tek kaynak modül).
