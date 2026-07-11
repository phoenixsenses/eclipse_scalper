# MASTER_ROADMAP — AMI × S34

**Tarih:** 2026-07-03 · Fable 5 · Çatı: master protokol Phase 0–10 · Sıfırdan yeniden yazma YOK — mevcut doğrulanmış çekirdek (ami/ 119 test) üzerine eklemeli.

## 1. Fazlar

### Phase 0 — Repository audit and safety ✅ TAMAM (bu oturum)
Çıktılar: bu dosya + 9 kardeş artifact + QUESTION_COVERAGE_MATRIX. DoD: karşılandı (DEFINITION_OF_DONE_BY_PHASE.md).

### Phase 1 — Canonical reconciliation (Sonnet, ~3-4 batch)
- B1: Canonical warehouse iskeleti `data/ami/canonical.sqlite` (yeni dosya; rollback=sil): artifact_registry, artifact_lineage, question_registry, contradiction_registry, operator_decision_queue, namespace_registry + SCHEMA_DICTIONARY kayıtları.
- B2: Artifact discovery/ingest (read-only tarama → registry; hash + rol + canonical statü; AMI-acronym collision gate).
- B3: Question registry seed: matrix CSV → warehouse; 14 slug-soru map; family kayıtları; Q867–Q1058 verbatim.
- B4: Experiment/evidence/KO registry'lerinin warehouse görünümleri (mevcut research.sqlite/knowledge.sqlite REFERANSLANIR, taşınmaz).
- DoD: warehouse round-trip testli; hiçbir mevcut store değişmedi; conflict register güncel.

### Phase 2 — Evidence, timestamp, contamination integrity (Sonnet, ~3 batch)
- Contamination ledger + researcher-exposure ledger + multiple-testing family registry (warehouse tabloları + ami/ API).
- Merkezi known-at/available-at kontrat modülü (Obs §6; mevcut scriptlerdeki ad-hoc disiplinin tek kaynağa çekilmesi — davranış değişikliği YOK, yalnız kayıt).
- Veri-kapsama denetimi: funding/OI süreklilik raporu (OD-006 girdisi); missing-data roadmap (Protocol §17).
- → **OPUS REVIEW A** (Phase 1–3 birlikte de olabilir; erken review tercih edilirse Phase 2 sonunda ara paket).

### Phase 3 — Event/cycle/path foundation (Sonnet, ~3-4 batch)
- Canonical event identity (Protocol §8) + structural cycle resolver (Obs §5.4) — real vs PROXY etiketli; cycle-definition versiyonu operatör onayı (OD-003).
- ami_events / ami_cycles / event_cycle_membership / ami_event_paths şemaları + tarihsel replay ingest (read-only kaynaklardan).
- Cycle-grouped split + purge/embargo yardımcıları (research engine'lerin ortak kullanımı için).
- → **OPUS REVIEW A** (Phase 1–3 kapanışı).

### Phase 4 — Chart-native object foundation (Sonnet, ~3 batch)
- Candle normalizasyonu (closed-candle-only), confirmed-swing extraction (known_at_ts'li), level registry, push geometry — CN §4-7; definition versioning zorunlu.
- Hiçbir observer aktive edilmez; yalnız tarihsel/descriptive hesap.
- → **OPUS REVIEW B** (Phase 4–5 kapanışı).

### Phase 5 — Shared feature engines (Sonnet, ~2-3 batch)
- Feature dictionary (CN §19 + mevcut script feature'larının envanteri); duplicate-engine tespiti ve tek kaynağa konsolidasyon (davranış değişikliği yok).
- feature_known_at enforcement testleri.
- → **OPUS REVIEW B**.

### Phase 6 — Historical research waves (Sonnet, dalga başına 1-2 batch)
- HISTORICAL_RESEARCH_WAVES.md sırasıyla; her dalga: prereg-freeze → §9 zorunlu kontroller → verdict → failure archive/KO.
- Mezarlık yeniden test edilmez (AMI failure archive + docs/ami protokol §5).
- → **OPUS REVIEW C** (Phase 6–7 önemli dalgaları sonunda).

### Phase 7 — Lifecycle/entry/hold/exit/transition research (Sonnet)
- Timing labels/metrics canonical'a; competing-risk hold, stop taxonomy, re-entry, transitions — hepsi Phase 3 identity üstünde.
- → **OPUS REVIEW C**.

### Phase 8 — Forward observatory (Sonnet; her yeni observer aktivasyonu OPERATÖR ONAYI)
- Forward event master + snapshot scheduler + observer engine + timing aggregates (Obs §7-21); mevcut forward_pipeline korunur ve genelleştirilir.
- Aktivasyon timestamp'leri frozen; forward N=0'dan; replay ayrı etiket.
- → **OPUS REVIEW D** (Phase 8–9 kapanışı).

### Phase 9 — Canonical dashboard (Sonnet)
- Obs §26 API + §27 sayfaları downstream-only; Excel/Word rejenerasyonu (CONFLICT-007) burada.
- → **OPUS REVIEW D**.

### Phase 10 — Advanced research prerequisites (yalnız hazırlık; World Model'e GEÇİŞ YOK)
- Readiness diagnostics (Obs §24), OOD/kalibrasyon önkoşul denetimi.
- → **OPUS REVIEW E** zorunlu kapı; operatör onayı olmadan hiçbir model eğitimi yok.

## 2. Faz sistemleri eşlemesi

| Master 0–10 | Whitepaper A–F | Observatory A–G | Chart-native 0–7 |
|---|---|---|---|
| 0 | — | A (audit) | 0 (audit) |
| 1 | — (öncül) | — | — |
| 2 | A evidence safety | — | — |
| 3 | B cycle integrity | B data foundation | — |
| 4 | — | — | 1-2 (feature/pattern foundation) |
| 5 | — (C/D öncülü) | — | 1-2 devam |
| 6 | C route separation | — | 5 kısmi (historical outcome) |
| 7 | D lifecycle | — | 3 (setup lifecycle) |
| 8 | E observatory | C-D (observer+aggregate) | 4-5 (observation bridge/observer) |
| 9 | E devam | E-F-G (API/dashboard/report) | 6 (dashboard) |
| 10 | F advanced decision | — (§24 readiness) | 7 (prereg adayları) |

## 3. Model orkestrasyonu

Fable 5 (bu oturum, Phase 0) → Sonnet 5 (Phase 1'den itibaren batch'ler) → Opus 4.8 kapıları: REVIEW A (P1-3), B (P4-5), C (P6-7), D (P8-9), E (P10 öncesi). Fable'a dönüş yalnız protokol §3'teki reset koşullarında.

## 4. Paralel workstream sınırı

Yalnız aynı warehouse + aynı identity + aynı timestamp kontratı + aynı vocabulary + aynı safety boundary şartıyla; Phase 1-2 içinde artifact-ingest ile question-seed paralel gidebilir. RAM kuralı: prosesler SIRAYLA.
