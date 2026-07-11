# CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER

**Tarih:** 2026-07-03 · Fable 5 · Hiçbir çatışma sessizce harmonize edilmedi.

## Precedence sırası (kayıtlı karar)

1. Frozen experiment spec'leri + immutable Decision Records + reproducible evidence (en yüksek)
2. Reconstruction Protocol — canonical state/warehouse/lineage konularında
3. Whitepaper — bilimsel anayasa/evidence/governance konularında
4. Forward Observatory — forward kayıt/observer/timing konularında
5. Chart-Native — yalnız proposed extension alanında
6. Reconciliation Initial Report — rehber (mutable narrative)
7. SYSTEM_STATE.md — runtime gerçeği raporlar, spec'leri override etmez

## Conflict kayıtları

### CONFLICT-001 — "Explicit Q396–Q730 registry" repoda yok · **SEVERITY: HIGH** · STATUS: OPEN (operatör kararı OD-001)
Reconstruction Protocol §16: "current explicit registry covers Q396–Q730 … preserve all 335 questions verbatim". Repo genelinde grep: numeric Q-ID'ler yalnız iki belgede aralık olarak var; `data/ami/research.sqlite` 14 slug-ID soru içeriyor; 335 verbatim metin HİÇBİR artifact'te yok.
**Çözüm önerisi:** Q001–Q866 family-level canonical + ID-level `MISSING_CANONICAL_TEXT` (QUESTION_COVERAGE_MATRIX böyle üretildi). Metinler operatörde ayrı bir kaynakta varsa teslim edilir ve matrix yeniden üretilir; yoksa Protocol §16 iddiası SUPERSEDED_BY_AUDIT işaretlenir. Metin UYDURULMAZ.

### CONFLICT-002 — Whitepaper v0.2 vs v0.3_COMPLETE ikiliği · **SEVERITY: HIGH** · STATUS: OPEN (OD-002)
CLAUDE.md + çalışma sözleşmesi: canonical spec = v0.2, yaşayan-doküman PATCH zinciri (0.2.7, PATCH-0007) v0.2 Appendix H'de. Master protokol v0.3_COMPLETE'i inceleme kaynağı yapar. v0.3'te Appendix H = "Canonical Database Extensions" (PATCH registry DEĞİL) → Appendix-H referansları belgeye göre farklı anlama geliyor.
**Çözüm önerisi:** v0.3_COMPLETE = canonical CONTENT (yeni Volume VIII + Parts/Appendix genişlemeleri); PATCH zinciri v0.2'de devam eder VE v0.3 doküman-kontrol bölümüne senkron kayıt düşülür; ileride tek dosyada birleştirme ayrı operatör kararı. Geçici kural: içerik çelişirse v0.3 metni + en yüksek PATCH kaydı birlikte okunur, çelişki bu register'a işlenir.

### CONFLICT-003 — Çoklu faz sistemleri · SEVERITY: MEDIUM · STATUS: RESOLVED_BY_MAPPING
Master protokol Phase 0–10; whitepaper Part XVII Phase 0–9 (mimari olgunluk) + §77 Phase A–F (v0.3 araştırma programı); observatory §36 Phase A–G (deployment); chart-native §29 Phase 0–7. **Karar:** Master protokol 0–10 çatıdır; diğerleri MASTER_ROADMAP.md §2 tablosuyla map edilir. Whitepaper'ın eski 0–9'u "mimari olgunluk sırası", A–F "operatif araştırma sırası" (Reconciliation Report önerisiyle uyumlu).

### CONFLICT-004 — `cycle_id` vs `structural_cycle_id` · SEVERITY: HIGH · STATUS: RESOLVED_BY_PRECEDENCE
İkisi aynı independent-sample problemini çözüyor. **Karar:** TEK canonical identity: Protocol §8 event-identity alanları + Observatory §5.2 structural-cycle çözücüsü aynı warehouse'da; 6h-gap kuralı yalnız PROXY etiketiyle yaşar (real vs proxy population ayrımı korunur). Cycle-definition versiyonu + reset/censoring policy operatör onayına tabi (OD-003).

### CONFLICT-005 — Warehouse vs Observatory tablo ontolojileri · SEVERITY: HIGH · STATUS: RESOLVED_BY_PRECEDENCE
İkisi ayrı inşa edilirse çifte truth doğar. **Karar:** Protocol §7 canonical warehouse system-of-record; Observatory §7/§12/§13/§30 tabloları AYNI DB'de canonical extension olarak, tek schema dictionary ile. Yeni paralel DB açılmaz (mevcut `research.sqlite`/`knowledge.sqlite` korunur, warehouse bunları REFERANSLAR).

### CONFLICT-006 — Dashboard view mi source mu · SEVERITY: MEDIUM · STATUS: RESOLVED
Tüm belgeler hemfikir ama observatory dashboard-ağır. **Sert kural:** dashboard/Excel/Word downstream-only; her sayfa canonical SQL + test edilmiş research view'dan; hiçbir dashboard sorgusu collector'ı bloklayamaz.

### CONFLICT-007 — Protocol'ün Excel/Word rejenerasyon zorunluluğu vs mevcut ortam · SEVERITY: LOW · STATUS: DEFERRED
Protocol §13-14 Excel workbook + Word raporu ister. Mevcut ortamda öncelik SQL+Markdown; Excel/Word üretimi Phase 9 (dashboard) kapsamına ertelendi. Veri kaybı yok (hepsi canonical SQL'den üretilebilir). Operatör isterse öne çekilir.

### CONFLICT-008 — Real liquidation vs proxy cascade populasyonu · SEVERITY: HIGH · STATUS: ENFORCED
Protocol §8 + master protokol §12: birleştirme yasak. Mevcut research scriptleri cascade proxy kullanıyor; warehouse'da `event_family` + `source_quality` + `definition_version` alanları zorunlu olacak (Phase 3 DoD).

### CONFLICT-009 — Shadow runner'ın "untouched" statüsü vs mevcut uncommitted diff · SEVERITY: LOW · STATUS: RECORDED
`tools/s34_realtime_shadow_runner.py` Phase 0 başında zaten modified (önceki oturumlar, observation-only mech_score loglaması, operatör onaylıydı). Taban çizgisi bu haliyle kabul edildi; YENİ diff eklenmez (UNTOUCHED_LIVE_SHADOW_COMPONENTS.md).

### CONFLICT-010 — Reconciliation Report'un rolü · SEVERITY: LOW · STATUS: RESOLVED
Docx mutable narrative; wave sıralaması ve risk listesi benimsenmiştir ama bağlayıcılık sırası 6. sıradadır. Spec'lerle çelişirse spec kazanır ve buraya kayıt düşülür.
