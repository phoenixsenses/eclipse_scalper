# DEFINITION_OF_DONE_BY_PHASE

**Tarih:** 2026-07-03. Genel şart (her faz): implementation + tests + migration verification + reproducibility + rollback + documentation + (kapı fazlarında) Opus acceptance; protected diff boş; çift verdict raporu; SYSTEM_STATE + progress ledger güncel.

| Phase | Ek DoD |
|---|---|
| 0 ✅ | 10 audit artifact + question matrix + protokol v1.1 + SYSTEM_STATE §40 + handoff checkpoint |
| 1 | Warehouse round-trip + dump testi; artifact registry tüm canonical dosyaları hash'li kapsar; question registry 1058 ID + 14 slug map; hiçbir mevcut store değişmedi |
| 2 | Contamination/exposure/MT registry yazılabilir + mutation testli; known-at kontratı tek modül; veri-kapsama raporu (funding/OI dahil) yayınlandı |
| 3 ✅(OD-003 hariç) | **Tamamlanan (OD-003'ten bağımsız):** immutable event_id + real-vs-proxy source_quality (ami_events, 251 gerçek event); non-canonical cooldown-sensitivity view'ları (§8.4, 6 pencere); cycle-grouped purge/embargo split yardımcısı. **BLOCKED_PENDING_OPERATOR_DECISION(OD-003):** ami_cycles canonical seed (cycle_definition_version + reset/censoring + direction-conflict policy seçimi olmadan 0 satır kalır, bilinçli). Opus REVIEW A bu ayrımla checkpoint'e taşınır — OD-003 kapıyı bloklamaz, yalnız ami_cycles'ı bloklar. |
| 4 | Closed-candle-only + known_at'li swing/level/push; definition versioning; observer YOK |
| 5 | Feature dictionary tam; duplicate engine kalmadı (envanter kanıtlı); known-at testleri her feature'da; Opus REVIEW B ACCEPTED |
| 6 | Her dalga: prereg-freeze hash + §9 kontrol seti + verdict + failure/KO kaydı; historical≠forward etiketi denetimli |
| 7 | Timing labels/metrics canonical; transition/hold/stop/re-entry dalgaları verdict'li; Opus REVIEW C ACCEPTED |
| 8 | Activation-ts frozen; N=0 başlangıç; orderless mutation-testli; zero-N honesty görünür; Opus REVIEW D (aktivasyonlar ayrıca OD'li) |
| 9 | Dashboard-to-SQL testleri; downstream-only kanıtı; Excel/Word rejenerasyonu canonical'dan; Opus REVIEW D ACCEPTED |
| 10 | Yalnız diagnostics; Opus REVIEW E ACCEPTED + operatör onayı olmadan model eğitimi YOK |
