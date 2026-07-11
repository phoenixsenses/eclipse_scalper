# DOCUMENT_RECONCILIATION_MATRIX

**Tarih:** 2026-07-03 · Fable 5 Phase 0 · Beş canonical belge uzlaştırması
Conflict detayları: `CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md`

## 1. Roller ve otorite

| Belge | Rol | Canonical otorite alanı | Yapamayacağı |
|---|---|---|---|
| Whitepaper v0.3_COMPLETE (≈9.1K satır) | CONSTITUTIONAL | Bilimsel anayasa, evidence hierarchy, KO modeli, question-family sistemi (Appendix O), cycle intelligence (Vol VIII), promotion/demotion, validation ladder, Phase A–F programı | "Hepsini hemen inşa et" diye okunamaz; hiçbir faz otomatik live izni vermez |
| Reconstruction Protocol v1.0 | CANONICAL DATA | Artifact registry/lineage, canon-selection + supersession, canonical SQL warehouse (§7), evidence-layer taksonomisi (§9), question sistemi durumu (§16), collector roadmap (§17), quality gates (§23), operator decision queue (§22) | Live/shadow'u sessizce değiştiremez; unlike-policy havuzlayamaz; route-bps'ten ekonomik iddia şişiremez |
| Forward Observatory v1.0 | FORWARD LAYER | Forward event/cycle/path kaydı, structural-cycle N, known-at kontratı (§6), orderless observer'lar (§11,18,19), timing ledger/metrikler (§13-15), readiness (§24), backend servisler + API + 16 dashboard sayfası (§25-27), deployment Phase A–G | Route/stop/sizing/permission değiştiremez; "daha çok gözlem ≠ daha çok izin"; replay forward sayılamaz |
| Chart-Native Extension v1.0 | PROPOSED EXTENSION | Candle/Swing/Level/Pattern/Setup objeleri (§4), setup lifecycle (§5), morphology/push/grammar/sweep/breakout/compression/channel/RS/session (§6-14), Q867–Q1058 verbatim (§23), chart observation registry (§17), implementation Phase 0–7 | Order üretemez; pattern'i otomatik alpha sayamaz; hindsight'ı forward gösteremez; whitepaper'ı override edemez |
| Reconciliation Initial Report (docx) | GUIDANCE | Precedence stack, conflict listesi, wave 1-3 sıralaması, 15+1 risk, paralel/bekleyecek iş ayrımı, operator-approval konuları | Dört spec'in yerine geçemez; mutable narrative — frozen spec/DR/reproducible evidence ondan üstün |

## 2. Dependency zinciri (belgelerin kendi ifadeleriyle)

```
Reconstruction Protocol (canonical truth + warehouse + identity)
        ↓
Whitepaper Phase A (evidence safety) + Phase B (cycle integrity)
        ↓
Forward Observatory (event master, cycle resolver, path ledger, observers)
        ↓
Chart-Native Extension (chart objects → descriptive → controls → prereg adayları)
```
Chart-native kendi §32'sinde "önce whitepaper + protocol + observatory oku" der; observatory kendi Phase A'sında repo/safety audit ister; protocol Stage 0'da safety snapshot ister. Üçü de master protokol Phase 0-sırasını doğrular.

## 3. Overlap → tek canonical'a indirgeme

| Kavram | Geçtiği yerler | Canonical karar |
|---|---|---|
| Event/cycle identity | Protocol §8 (`cycle_id`), Observatory §5 (`structural_cycle_id`), Whitepaper §65 | TEK identity modeli; Protocol warehouse system-of-record, Observatory resolver onun üstünde (CONFLICT-004) |
| Event/cycle/path tabloları | Protocol §7.4-7.5, Observatory §7/§12/§13/§30, Whitepaper App H (v0.3) | Protocol §7 şema ailesi esas; Observatory tabloları canonical extension olarak AYNI warehouse'a (CONFLICT-005) |
| Question sistemi | Whitepaper §69 + App O (aileler), Protocol §16 (Q396–730 iddiası), Chart-native §23 (Q867–1058 verbatim) | Family-registry yaklaşımı; 1058 atomik task YOK; eksik metin = MISSING_CANONICAL_TEXT (CONFLICT-001) |
| Evidence layer taksonomi | Protocol §9 (8 katman), Observatory §3, Whitepaper §78 | Protocol §9 vocabulary esas; Observatory readiness/permission-ceiling alanları eklenir |
| Dashboard | Observatory §27 (16 sayfa), Chart-native §27 (10 sayfa), Protocol §13 (Excel) | Hepsi downstream-only; canonical SQL + test edilmiş view'lardan (CONFLICT-006) |
| Mutation/integration test | Whitepaper App I, Observatory §34, Chart-native §28, Protocol §23 | Birleşik strateji: TEST_AND_MUTATION_STRATEGY.md; mevcut 119 AMI testi taban |
| Faz sistemleri | Master protokol 0–10, Whitepaper 0–9 + A–F, Observatory A–G, Chart-native 0–7 | Master protokol 0–10 çatı; mapping MASTER_ROADMAP.md §2 (CONFLICT-003) |

## 4. Mevcut implementasyonla kesişim (yeniden yazma YASAK)

| Belge gereksinimi | Repo'daki mevcut karşılık | Karar |
|---|---|---|
| Forward evidence pipeline | `ami/research/forward_pipeline.py` (VALID, 2 binding) | KORUNUR; observatory bunu genelleştirir, paralel ikinci hat açılmaz |
| KO + governor + failure archive | `ami/knowledge/`, `ami/governance/governor.py` (119 test) | KORUNUR; warehouse KO tablosu bunun üstüne map edilir |
| Research registry (prereg-freeze) | `ami/research/registry.py` | KORUNUR; numeric question registry BUNA eklenir |
| Multi-TF state engine | `ami/states/` | KORUNUR; Observatory §9 state store bunun kaydına dayanır |
| Latent/regime/drift | `ami/latent/` (REJECTED/PASS-dar sonuçlarıyla) | Verdict'ler immutable; kod research-only kalır |
| Shadow ledger'lar | `reports/shadow/*.jsonl` | Kaynak; warehouse'a READ-ONLY ingest edilir |
