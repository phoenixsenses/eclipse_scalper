# AMI Gap Analysis — repo vs whitepaper

> Appendix F gereği teslimat. **Aktif sürüm: v0.3** (BÖLÜM I).
> v0.2 analizi (2026-07-02) tarihsel kayıt olarak BÖLÜM II'de değiştirilmeden korunur.

---

# BÖLÜM I — v0.3 GAP ANALİZİ (2026-07-17, Opus 4.8)

**Kapsam:** `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md`
§57-78 (Volume VIII), §77 (Phase A-F), §78 (DoD), Appendix H (17 tablo),
Appendix I (35 mutasyon), Appendix J (öncelikler).

**Yöntem:** 3 bağımsız salt-okunur repo taraması + doğrudan SQLite şema/satır
sayımı (`mode=ro`). Hiçbir dosya değiştirilmedi, hiçbir proses başlatılmadı.

**Statü:** `V03_GAP_ANALYSIS_IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW`

## 0. Yönetici özeti — TEK kök neden

v0.3'ün eksikleri bağımsız değil; **tek bir yapı taşının yokluğundan** zincirleme
türüyor:

> **Phase E / Forward Observatory (§68 + Appendix H `ami_forward_*`) hiç inşa edilmedi.**
> Appendix H'nin **17 tablosundan 0'ı** mevcut. Kodda `ami_forward_` yalnız bir
> docstring notunda geçiyor (`ami/research/w10a_multi_tf_structural_conflict.py`, OD-017).

Doğrudan bloke ettikleri: C1 (unconditional LONG genesis), C2 (failed-fade LONG),
D1 (signal aging), D5 (entry mechanics), W7-tam (OD-016), W10-transitions (OD-017),
W12 (position-aware action-value).

**Bu boşluk DÜRÜST.** `SYSTEM_STATE.md:3247` `first_known_ts`/`first_executable_ts`
alanlarını `FORWARD_ONLY` ilan ediyor — tarihsel olarak uydurulmaları YASAK.
Eksiklik bir ihmal değil, bilinçli bir "veri yokken üretme" kararı.

**İkinci desen:** v0.3 boşluklarının neredeyse tamamı **beyan edilmiş ve backlog'a
alınmış** (OD-012…OD-017), gözden kaçmış değil. `ami/research/w*` modülleri
disiplinli: docstring-as-preregistration, `freeze_and_record()`, cycle-grouped
split, Holm düzeltmesi, bloke dalgayı zorlamayı açıkça reddetme.

**İzleyici uyarısı:** `HISTORICAL_RESEARCH_WAVES.md` (W1-W12) v0.3 Volume VIII'in
**canlı izleyicisi**. `docs/ami/AMI_ROADMAP.md` bayat (v0.2 §90-99 hizalı) —
gap analizi için KULLANILMAMALI.

## 1. Definition of Done (§78) — 24 madde

| # | DoD maddesi | Durum |
|---|---|---|
| 1 | v0.2 nesneleri/verdict'leri değişmedi | ✅ |
| 2 | Question Registry tüm yeni aileleri destekliyor | ✅ 1072 soru / 29 aile |
| 3 | Event- ve cycle-seviyesi örneklem ayrıldı | ✅ 252 event → 167 cycle |
| 4 | Position-aware state'ler kaydediliyor | ❌ |
| 5 | Unconditional LONG genesis ölçülebilir | ⚠️ evren var, deney ertelendi (OD-012) |
| 6 | Failed-fade LONG observer var | ❌ |
| 7 | Signal ve state age ayrı kaydediliyor | ⚠️ state age var, signal age yok (OD-016) |
| 8 | Market-clock alanları mevcut | ⚠️ 6'dan 3'ü |
| 9 | Scalp/intraday/swing route'ları ayrı | ❌ yalnız horizon adlandırması |
| 10 | Path taxonomy horizon-aware + versiyonlu | ⚠️ horizon-aware ama **hard-label** |
| 11 | Structural location + normalized geometry saklanıyor | ⚠️ DEGENERATE (OD-013) |
| 12 | Overlap/reset/censoring kuralları zorlanıyor | ⚠️ RIGHT_CENSORED var; reset ❌; 3/5 censor state ❌ |
| 13 | `feature_known_at` ile latency açık | ❌ kodda **sıfır** geçiyor |
| 14 | ACTUAL_SHADOW / OBSERVER / HISTORICAL_REPLAY ayrı | ⚠️ **vakum** (aşağı bkz.) |
| 15 | Evidence contamination + researcher exposure görünür | ⚠️ exposure ✅ (1180); contamination ❌ (0) |
| 16 | OOD ve uncertainty tek skora çökmüyor | ⚠️ **vakum** |
| 17 | Action-value çıktısı emir üretemez | ⚠️ **vakum** |
| 18 | Policy evaluation sequential + versiyonlu | ❌ |
| 19 | Benchmark ve regret raporları var | ❌ |
| 20 | Tüm yazımlar idempotent | ✅ |
| 21 | Mutation + integration suite geçiyor | ⚠️ 20/20 geçiyor ama **v0.2 kapsamı** |
| 22 | Live/shadow operasyonel bileşenler değişmedi | ✅ |
| 23 | Rollback belgeli | ✅ |
| 24 | Her çıktı promosyona kadar operasyonel yasaklı | ✅ |

**Skor: 7 ✅ / 10 ⚠️ / 7 ❌ → v0.3 DoD KARŞILANMADI.**

### ⚠️ "Vakum uyumu" — bu analizin en önemli uyarısı

DoD 14, 16, 17 şu anda **yalnızca üretici taraf hiç inşa edilmediği için**
sağlanıyor. Bu bir *invariant* değil, bir *implementasyon kazası*:

- **DoD 14:** `ObservationMode` (`ami/lifecycle/canonical_schema.py:92`) yalnız
  `HISTORICAL_REPLAY` + (hiç yazılmayan) `FORWARD_OBSERVATION` taşıyor.
  `ACTUAL_SHADOW` / `OBSERVER_HYPOTHETICAL` = **sıfır hit**. Hiçbir şey karışmıyor
  çünkü forward hiç yok.
- **DoD 16:** OOD ve uncertainty tek skora çökmüyor çünkü **ikisi de modellenmiyor**
  (`OOD`/`novelty`/`epistemic`/`aleatoric` = sıfır hit).
- **DoD 17:** `ami_forward_action_values` yok, üreticisi yok — ve Appendix I-17'yi
  karşılayan **hiçbir assertion/test de yok**. Tablo inşa edilirse aşılacak bariyer
  mevcut değil. (`dashboard/backend/server.py:5` GET/HEAD-only, ama bu dashboard
  özelliği; action-value guard'ı değil.)

**Sonuç:** Phase E inşa edildiği AN bu üç madde sessizce ihlale döner.
**Guard'lar üreticiden ÖNCE yazılmalı.**

## 2. Appendix H — 17 tablodan 0'ı mevcut

`data/ami/canonical.sqlite` 45 tablo taşıyor; hiçbiri `ami_forward_*` değil.

**İsim uzlaştırması (önemli):** 3 tablo farklı adla ama mantıksal olarak mevcut —
spec adına göre değil, özüne göre puanlandı:

| Appendix H adı | Repo'daki gerçek ad | Satır |
|---|---|---|
| `ami_market_cycles` | `ami_cycles` | 167 |
| `ami_evidence_contamination` | `evidence_contamination` | **0** |
| `ami_researcher_exposure` | `researcher_exposure_ledger` | 1180 |
| 14 × `ami_forward_*` | — | **YOK** |

`ami_events` (252) / `ami_cycles` (167) **tarihsel** store'lar; §68.1'in forward
alanlarını (`data_health_at_event`, `feature_coverage`, `missing_features`,
`session`, `regime`, `timeframe_alignment`, `code_commit`) taşımıyorlar.

**Tuzak:** `ami/run_forward_pipeline.py` + `ami/research/forward_pipeline.py`
adı "forward" ama §68 observatuvarı **DEĞİL** — shadow-ledger→evidence-binding
pipeline'ı (`research.sqlite`'a `forward_bindings`/`evidence` yazıyor).
§68 karşılığı olarak sayılmamalı.

## 3. Phase A-F durumu

| Faz | Kapsam | Durum |
|---|---|---|
| **A** | Evidence safety | ⚠️ **1/5** — şema iskeleti, tek canlı uzuv |
| **B** | Cycle integrity | ⚠️ **2/5** — motor gerçek, kenar durumlar yok |
| **C** | Route separation | ⚠️ **0.5/5** — bilinçli ertelendi (OD-012/013/014) |
| **D** | Dynamic lifecycle | ⚠️ **1.5/5** — W7A gerçek, gerisi Phase 8 bloklu |
| **E** | Forward observatory | ❌ **~15%** — yalnız Question Registry |
| **F** | Advanced decision | ❌ **0/5** |

### Phase A — "şema iskeleti, tek canlı uzuv"

4 tablo tek batch'te (`BATCH-P2-001`) yaratıldı ve açıkça ertelendi —
`SYSTEM_STATE.md:2388`: *"Hepsi boş (Phase 6+'da beslenecek) — bu batch yalnız
iskelet."* Yalnız `researcher_exposure_ledger` yazıcı kazandı, o da tesadüfen
(`ami/research/feature_gateway.py:90` her fetch'te satır ekliyor).

| Bileşen | Durum |
|---|---|
| A2 Researcher exposure | ✅ 1180 satır, `feature_gateway` yazıyor, testli |
| A1 Evidence contamination | ⚠️ şema tam (`ami/warehouse/schema.py:137-152`), **0 satır, yazıcı yok** |
| A3 Multiple-testing family registry | ⚠️ **0 satır**; ~18 `FAM_*` string'i docstring'lerde yüzüyor |
| A4 Market-structure ledger | ⚠️ **0 satır**; kardeşi `data_quality_events` (7 satır) çalışıyor |
| A5 Causal-assumption registry | ⚠️ **0 satır, yazıcı yok** |

> **A1 nüansı (şişirmeye karşı düzeltme — bağımsız review F-1 ile KAPSAMLANDI):**
> `evidence_contamination`=0 olması tek başına işlevsel boşluk DEĞİL: §70'in
> contamination yasası `knowledge.sqlite`'ta **fail-closed zorlanıyor** —
> `ami/governance/epistemic_gates.py` (test-evidence nullifier + graveyard
> slash-set), M-0033 ile `experiment_ledger`'a wire'landı.
> `evidence_contamination` tablosu bir **orphan şema**.
>
> **KAPSAM SINIRI (F-1, bu iddia genel DEĞİL):** zorlama yalnız
> `register_experiment_with_gates()` üzerinden geçen çağıranlar için geçerlidir.
> `record_experiment_registry`/`record_experiment_results`
> (`ami/warehouse/experiment_ledger.py:104,142`) **doğrudan çağrılabilir durumda**
> ve **10 legacy research modülü** (`w1`/`w3`/`w4`/`w5a`/`w6`/`w6rs_confirmation`/
> `w6rs_confound_resolution`/`w7a`/`w10a`) hem immutability guard'ını hem epistemic
> gate'leri **yapısal olarak baypas ediyor** — `experiment_ledger.py`'nin kendi
> docstring'i (satır 46-69) bunu "KNOWN, UNCLOSED bypass surface" olarak adlandırıyor.
> Yani §70 **kısmen** zorlanıyor, evrensel olarak değil.
>
> Gerçek A1 boşlukları: (i) 4 `evidence_status` değeri
> (`INDEPENDENT_EVIDENCE`/`REUSED_EVIDENCE`/`CONTAMINATED_FOR_CONFIRMATION`/
> `FORWARD_ONLY_CONFIRMATION_REQUIRED`) hiçbir yerde atanmıyor; (ii) yukarıdaki
> 10-modül baypas yüzeyi açık.

> **A2 nüansı:** spec'in 4 kategorisinden yalnız 1'i emit ediliyor
> (`BLINDLY_PREREGISTERED` 1179, `CROSS_FAMILY_TEST_CYCLE_REUSE_DISCLOSURE` 1);
> `RESULT_INFORMED_HYPOTHESIS`/`POST_HOC_EXPLORATION`/`INDEPENDENT_REPLICATION`
> hiç kullanılmıyor, `manual_override_log` boş → "her manuel override loglanmalı"
> mekanizmayla değil **teamülle** sağlanıyor.

### Phase B — motor gerçek, kenar durumlar yok

| Bileşen | Durum |
|---|---|
| B1 Cycle engine + dedup | ✅ `ami/identity/cycle_resolver.py`, `canonical-v1` donduruldu, 167 cycle |
| B3 Censoring | ⚠️ `COMPLETED`/`RIGHT_CENSORED` canlı (247/5); `DATA_CENSORED`/`SHOCK_CENSORED`/`NEW_CYCLE_CENSORED` **sıfır hit**; survival analizi yok |
| B4 State dwell time | ⚠️ W7A'da hesaplanıyor, **hiç persist edilmiyor** |
| B5 Soft path taxonomy | ⚠️ horizon-aware ✅ ama **CHECK constraint soft label'ı imkânsız kılıyor** |
| B2 Event overlap + reset | ❌ **YOK** |

> **B4 nüansı (şişirmeye karşı düzeltme):** `StateObject.age_ms`
> (`ami/states/objects.py:28`) `time.time()` kullanıyor → point-in-time güvenli
> DEĞİL. Ama **hiçbir araştırma kodu onu çağırmıyor** (tek tüketici `to_dict()`'in
> `age_min` alanı; W7A kendi point-in-time yaşını hesaplıyor). Yani **aktif
> kirlenme değil, tüketicisi olmayan gizli tuzak**. Bir lookahead bulgusu olarak
> raporlanmamalı; Phase E onu tüketirse ihlale döner.

> **B5 kök nedeni:** `ami/warehouse/schema.py:682` —
> `CHECK (horizon_outcome_class IS NULL OR horizon_outcome_class IN
> ('CONTINUATION','REVERSAL','CHOP'))`. §58.3'ün tüm amacı "hard label
> belirsizliği silmemeli" iken şema ikinci bir aday etiketi veya olasılığı
> **temsil edemiyor**. `secondary_path`/`path_probability`/`label_confidence` =
> sıfır hit. Ayrıca §58'in 10-etiketli sözlüğü (`FAILED_FADE_LONG`,
> `LONG_GENESIS`, `POST_EVENT_UNRESOLVED`…) tamamen yok.

### Phase C/D — bilinçli ertelemeler

- **C1/C2 (W2):** ERTELENDİ (OD-012) — mezarlık #8/#17 çakışması; #17 retry-koşulu
  (OI verisi) kontrol edildi, **zayıf** (38/252 anchor, 2 kopuk pencere).
  Evren altyapısı hazır: `ami/research/candidate_universe.py` (event varlığından
  bağımsız, her mum bir aday satırı).
- **C2 uyarısı:** `tools/research_s34_buyfade_structural.py:840`'ta `FAILED_FADE`
  etiketi var ama bu **gerçekleşmiş MAE'den hesaplanan bir outcome etiketi** —
  §60.3'ün tam da yasakladığı hindsight kurgusu. Failed-fade LONG observer'ı
  olarak sayılmamalı.
- **C4:** 6 canonical route ailesi (`LONG_SCALP`…`SHORT_SWING`) sıfır hit.
  Mevcut olan **horizon adlandırması** (`scalp_30m`/`swing_4h`) — §62'nin
  açıkça reddettiği kurgu ("aynı girişi farklı süre tutmak").
- **C5 §64.3:** `tools/s34_mechanism_*.py` — **disiplinin karşılanmadığı tek yer**:
  `ami/` dışında, `feature_gateway`'i baypas ediyor, test/prereg/freeze yok.
- **D5:** `feature_known_at` kodda **sıfır**. Hafifletici: `known_at_ts` `ami/`
  içinde 154 kez, `NOT NULL` → point-in-time güvenliğin *ruhu* var; §67.2'nin
  7-alanlı latency zinciri (`exchange_ts`/`receive_ts`/`processed_ts`/
  `staleness_ms`/`coverage_state`) yok.

### Phase F — 0/5

`OOD`/`novelty`/`epistemic`/`aleatoric`/`hierarchical`/`shrinkage`/`regret`/
`portfolio_overlap`/`incrementality`/`ergodic` — **hepsi sıfır hit**.
`ami/latent/drift_monitor.py` (`STABLE/WARNING/SHIFTED/UNUSABLE`) farklı bir
kavram (artifact drift), kısmi kredi sayılmaz.

## 4. Appendix I — 35 mutasyondan ~6'sı kısmen karşılanıyor

`ami/mutation_suite.py` 20 senaryo (m01-m20) taşıyor, 20/20 geçiyor — ama bunlar
**v0.2 dönemi jenerik ihlaller**. v0.3'ün 35 spesifik maddesine eşleme:

| Appendix I | Mevcut karşılık | Durum |
|---|---|---|
| I-1 pre-event LONG'da gelecek bilgisi | m01 `future_lookahead` (jenerik) | ⚠️ |
| I-4 cycle'ı train/val arasında bölme | `tests/test_ami_identity_split_utils.py` | ⚠️ **helper testi, kapı değil** |
| I-26 origin-split'i bağımsız teyit sayma | `epistemic_gates` nullifier + m18 | ⚠️ |
| I-27 bağımlı iddiaları bağımsız sayma | m09 `duplicate_trade_evidence` | ⚠️ |
| I-28 result-informed değişikliği loglamama | m08 `prereg_metric_changed` | ⚠️ |
| I-31 staleness/receive-time yoksayma | m03 + m20 | ⚠️ receive-time yok |
| **I-2, 3, 5-25, 29, 30, 32-35** | — | ❌ **YOK** |

> **En kritik yapısal bulgu (I-3/I-4/I-12):** cycle bağımsızlığı için **yardımcı
> fonksiyon ve ölçüm var, ama ZORLAYICI KAPI yok.** `ami/identity/split_utils.py`
> doğru cycle-grouped split'i yapıyor ve testli — ama bir araştırmacı onu
> **çağırmazsa hiçbir şey reddetmiyor**.
>
> Bu, repo'nun **zaten bir kez çözdüğü** desenin aynısı — `epistemic_gates.py:20`:
> *"`researcher_exposure_ledger` RECORDED exposure but nothing BLOCKED a family
> from re-reading the same TEST cycles…"* → M-0033 wiring ile kapatıldı.
> **Çözüm şablonu repoda mevcut ve emsalli.**

## 5. Governance boşluğu

- `docs/ami/AMI_ROADMAP.md` + `AMI_CHANGELOG.md` **v0.3'ten hiç söz etmiyor**
  (roadmap hâlâ "whitepaper §90-99 hizalı" = v0.2). Bayat.
- Bu dosya v0.3 öncesi **v0.2 tarihliydi** — bu revizyonla düzeltildi.
- **Question Registry triyaj edilmemiş:** 1072 sorudan 1058'i `FUTURE_PHASE`,
  866'sı `FAMILY_TRIAGE_PENDING`; `evidence_layer` **1072/1072 boş**;
  `READY_FOR_PREREG`/`research_ready` = **sıfır hit** → §69.4 research-ready
  kapısı hiç işletilmemiş.
- **Weekly scientific report yok** (`tools/daily_research_report.py` var,
  haftalık muadili yok) — DoD dışı ama §77 Phase E-6 gereği.

## 6. Önerilen sıra (Appendix J P0 hizalı)

Appendix J P0 = *holdout contamination; evidence dependency; researcher exposure;
multiple-testing governance; data/market-structure versioning; cycle deduplication;
overlap and censoring; data availability latency.*

1. **[BAŞLANDI] Appendix I cycle-independence enforcement mutasyonları**
   (I-3/I-4/I-12/I-21/I-32) — mevcut motorlara karşı test edilebilir, Phase E'yi
   BEKLEMİYOR, P0'ın tam merkezi, emsal şablonu (M-0033) mevcut.
2. **A3 multiple-testing family registry'ye yazıcı** (P0; `FAM_*` string'lerini
   satıra bağla). *(F-3 ile 3.→2. sıraya alındı: tablo ve `FAM_*` string'leri
   ZATEN var, yalnız yazıcı eksik → spekülatif tasarım gerektirmiyor, hemen
   uygulanabilir ve Appendix J'de açıkça P0.)*
3. **A1-(i): 4 `evidence_status` değerinin atanması** + **A1-(ii): 10-modül
   baypas yüzeyinin kapatılması** (P0 "holdout contamination" + "evidence
   dependency"). *(F-3: v1'de bu madde listeden tamamen DÜŞMÜŞTÜ — §3/A1'de
   "gerçek A1 boşluğu" diye tespit edilip önceliklendirilmemişti.)*
4. **Phase E guard'ları — üreticiden ÖNCE** (DoD 14/16/17 vakum uyumunu
   invariant'a çevir; özellikle I-17 action-value→order bariyeri).
   *(F-3 ile 2.→4.: gerekçesi hâlâ geçerli — inşa edilmemiş koda guard yazmanın
   emsali bu dosyanın kendi I-3/I-4/I-12 yaklaşımı — ama Phase E üretici arayüzü
   henüz yetkilendirilmedi, dolayısıyla A3/A1'den daha spekülatif ve daha pahalı.)*
5. B5 soft-label şeması (CHECK gevşetme + `path_probability`) — yeni
   `path_definition_version`, eski dondurulmuş sürüm korunur.
6. Phase E forward observatory (büyük; ayrı yetkilendirme + prereg ister).

**Dokunulmayan dosyalar (guardrail):** `tools/s34_state_machine_live_executor.py`,
`.env`, `execution/`, `risk/`, `brain/`, leverage/sizing — **hiçbiri
okunmadı/değiştirilmedi.** Hiçbir KO/verdict değiştirilmedi, hiçbir store'a
yazılmadı.

## 7. Rollback

Bu revizyon salt-dokümantasyon + additive test. Geri alma = bu dosyanın v0.2
sürümüne dönmek + yeni test dosyasını silmek. Hiçbir şema/veri/proses değişmedi.

---

# BÖLÜM II — v0.2 GAP ANALİZİ (2026-07-02, tarihsel kayıt — değiştirilmedi)

> Appendix F gereği ilk teslimat. Durum: Faz 0-5 temeli BU OTURUMDA inşa edildi.

## 1. AMI gereksinimini ZATEN karşılayan mevcut bileşenler

| Whitepaper kavramı | Mevcut karşılık |
|---|---|
| Reality Interface (kısmi) | `data/microstructure_collector.py`, `bookticker_collector`, `oi_spot_poller` (2026-07-02'de eklendi) |
| Event/Mechanism Store | `reports/research/s34/mechanism_store.sqlite` (418 event + 418 kontrol, pencere hizalı) |
| Forward shadow observers | `tools/s34_realtime_shadow_runner.py` (25 bucket, profit-lock + scale-in gözlemcileri) |
| Research raporları/kanıt | `reports/research/s34/*` + `S34_ALL.db` meta-DB (44K satır, trust skorlu) |
| Failure graveyard (ham) | `S34_ALL.db` mezarlık analizi (613 red) |
| Validation standartları (pratik) | TRAIN/TEST, MC permütasyon, no-overlap, max-stat — research scriptlerinde yerleşik |
| Live guardrails | `start/stop_eclipse.ps1`, executor sign-off kuralları, SYSTEM_STATE §7 |

## 2. Eksik olan ve BU OTURUMDA inşa edilen temeller

| Bileşen | Yeni modül |
|---|---|
| Scientific Constitution (kod) | `ami/constitution.py` |
| Canonical enums/statüler/izinler | `ami/enums.py` |
| KnowledgeObject + executable contract (§71) | `ami/knowledge/objects.py` |
| Knowledge Graph + Audit + Failure Archive | `ami/knowledge/store.py` (`data/ami/knowledge.sqlite`) |
| Epistemic Governor (promotion/demotion/breaker/revision) | `ami/governance/governor.py` |
| Multi-TF StateObject/Bundle + veri sağlığı | `ami/states/objects.py`, `ami/states/engine.py` |
| Structure transition matrix (Faz 2) | `ami/states/structure.py` |
| Trade Lifecycle Engine + MFE classifier verisi (Faz 3) | `ami/lifecycle/engine.py` |
| Research OS: soru/hipotez/prereg-freeze/kanıt (Faz 4) | `ami/research/registry.py`, `marketplace.py` |
| DecisionTrace + authorize akışı | `ami/decision/trace.py` |
| S34 bilgi + mezarlık + backlog tohumlama | `ami/seed_s34.py` |
| Uçtan uca doğrulama | `ami/run_phase_checks.py` → `AMI_PHASE_VALIDATION.md` |
| Testler (Appendix F minimum seti + fazlası) | `tests/test_ami_knowledge_governance.py`, `tests/test_ami_states_research.py` (17 test ✓) |

## 3. Hâlâ eksik (sonraki fazlar)

- **Faz 6 ML/latent states:** clustering/HMM yok (kasıtlı — Faz 6).
- **Faz 7 World Model/Digital Twin:** senaryo simülasyonu + kalibrasyon döngüsü yok.
- **Faz 8 Autonomous Scientist:** anomali→soru üretimi manuel; agent reliability skorları yok.
- **Faz 9 Cross-market:** tek borsa (Binance), 3 sembol.
- Cross-exchange feed, options, on-chain, macro takvim — Reality Interface genişletmesi.
- Event bus / servis ayrımı — şimdilik tek-repo modüler (whitepaper §76'ya uygun).
- Kalibrasyon motoru (Brier vb.) — iskelet yok, Faz 6 ile.

## 4. Veri kalitesi riskleri (aktif)

1. `vol_state` STALE (producer silindi; tüketiciler mark-proxy'ye taşındı) — state engine doğru şekilde STALE raporluyor.
2. OI/spot geçmişi kısa (2026-07-02'den itibaren) — OI soruları forward birikim istiyor.
3. mark_prices gap'leri (7 günde 11×>120s) — sensör staleness kontratı state engine'de var.
4. Tek venue (Binance) — cross-exchange doğrulaması imkânsız şimdilik.

## 5. Bilgi-yönetişim riskleri

- Eski research raporları (S34_ALL.db) Knowledge Object formatında değil — 9 çekirdek iddia elle taşındı (`seed_s34`), kalan geçmiş toplu migrasyon adayı.
- Trust skoru ≠ lookahead kontrolü (bilinen) — governor'daki evidence-level alanı bunu ayrıştırıyor.

## 6. Dokunulmayan dosyalar (Appendix F guardrail)

`tools/s34_state_machine_live_executor.py`, `.env`, `execution/`, `risk/`, `brain/`,
leverage/sizing sabitleri, canlı emir mantığı — **hiçbiri değiştirilmedi.**

## 7. Rollback

AMI tamamen eklemeli: `ami/`, `data/ami/`, `docs/ami/`, 2 test dosyası.
Geri alma = bu dizinleri silmek; mevcut sisteme hiçbir bağımlılık eklenmedi.
