# SCHEMA_DICTIONARY

**Güncelleme:** 2026-07-03 (Phase 0 tabanı). Yeni tablo/kolon eklendikçe güncellenir (yalnız değişiklikte).

## Mevcut store'lar
| Store | Tablolar | Sahip |
|---|---|---|
| data/ami/knowledge.sqlite | knowledge(11), edges(4), audit_log(40), failure_archive(21) | ami/knowledge/store.py |
| data/ami/research.sqlite | questions(14, slug-ID), hypotheses(2), experiments(10), evidence(8), forward_bindings(2), processed_trades | ami/research/registry.py + forward_pipeline.py |
| data/ami/decisions.jsonl | immutable DecisionTrace paketleri | ami/decision/trace.py |
| data/microstructure.db | likidasyon + mark price (684GB, RO) | collector'lar |
| data/s34_intelligence.db | S34 canlı istihbarat | shadow/dashboard katmanı |
| reports/research/s34/S34_ALL.db | geçmiş test meta-DB | research scriptleri |
| reports/research/s34/mechanism_store.sqlite | mekanizma feature store | s34_mechanism_* |

## data/ami/canonical.sqlite (Phase 1-6/W1 TAMAM: P1-001..004 + P2-001..003 + P3-001..005 + P4-001..004 + P6-000..002)
Kaynak: `ami/warehouse/schema.py` (CANONICAL_SCHEMA_VERSION=6, `PRAGMA foreign_keys=ON`). Her tabloda `schema_version` + `provenance` + `created_ms`(/`updated_ms`) zorunlu.

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| schema_versions | component→version izleme | component | 1 |
| artifact_registry | dosya/rol/canonical-statü/hash envanteri | artifact_id | 64 (62 CANONICAL + 2 UNDER_RECONCILIATION — whitepaper v0.2/v0.3, CONFLICT-002) |
| artifact_lineage | supersession/predecessor ilişkileri | lineage_id (autoinc) | 1 (whitepaper v0.2↔v0.3, relation=UNDER_RECONCILIATION — fabrikasyon yok) |
| question_families | Appendix O + chart-native aileleri | family_id | 29 |
| question_registry | Q001–Q1058 + 14 legacy slug soru | question_id | 1072 (866 MISSING_CANONICAL_TEXT, 192 CANONICAL_TEXT_PRESENT_PROPOSED, 14 LEGACY_SLUG). **BATCH-P3-004 düzeltmesi:** FK enforcement `family_id`'nin `question_families.family_id` ile eşleşmediğini ortaya çıkardı (prefix farkı, 1058/1058 satır); kod + gerçek veri onarıldı, 0 mismatch |
| contradiction_registry | CONFLICT-001..010 (doc-level) + CT-001..004 (evidence-level) | contradiction_id | 14 |
| operator_decision_queue | OD-001..010 (OD-006 P2-003 bulgularıyla güncellendi, hâlâ OPEN) | decision_id | 10 (hepsi OPEN) |
| namespace_registry | AMI-acronym collision gate | namespace | 1 (ami_s34) |
| evidence_contamination | whitepaper §56.1 — hipotez kontaminasyon durumu | contamination_id | 0 (Phase 6 research dalgalarında beslenecek) |
| researcher_exposure_ledger | whitepaper §56.4 — split/threshold/route exposure | exposure_id | 0 (Phase 6+) |
| mt_family_registry | whitepaper §56.3 — family-level multiple-testing | family_id | 0 (Phase 6+) |
| causal_assumption_registry | whitepaper §56.5 — assumed-arrow + confounder takibi | assumption_id | 0 (Phase 6+) |
| data_quality_events | Protocol §7.1 — feed gap/stale/schema/late-value kayıtları | event_id | **7** (BATCH-P2-003 funding/OI denetimi: 4 stale/orphaned + 3 per-symbol OI coverage) |
| market_structure_versions | whitepaper Appendix N.11 — fee/contract/tick versiyonlama | version_id | 0 (henüz beslenmedi) |
| ami_events | Protocol §7.4/Obs §5.1 — immutable event identity, real-vs-proxy | event_id | **252** (gerçek shadow ledger'dan dedup edilmiş anchor, canlı ledger'dan büyüyebilir; hepsi source_quality=REAL_LIQUIDATION) |
| ami_cycles | Protocol §7.4/Obs §5.2 — canonical cycle (OD-003 onaylı A2+B2+C2) | cycle_id | **167** (cycle_definition_version=canonical-v1; 162 COMPLETED/5 RIGHT_CENSORED; 15 direction_conflict=1 [WAIT, otomatik çözülmedi]; confidence ort=1.0) |
| event_cycle_membership | Protocol §7.4 — sensitivity (is_canonical=0) + canonical (is_canonical=1) | membership_id (autoinc) | **1512+252** (252 event × 6 sensitivity pencere, is_canonical=0, DOKUNULMADI + 252 canonical-v1 satırı, is_canonical=1, append-only) |

Kaynak kod: `ami/warehouse/{schema,init_db,artifact_ingest,question_seed,registry_seed,funding_oi_audit}.py` + `ami/timing/contract.py` (known-at kontratı) + `ami/identity/{event_identity,shadow_ledger_ingest,cooldown_sensitivity,split_utils,cycle_resolver}.py` (Phase 3). Hepsi idempotent (2× koşum doğrulandı her batch'te).

**Phase 2 kritik bulgu (data_quality_events, OD-006):** OI zaten CANLI (oi_spot_poller→microstructure.db); funding_rates'in HİÇBİR canlı üreticisi yok (iki kaynak da donmuş: data/funding_history.db son 2026-05-12, microstructure.db:funding_rates son 2026-04-13 + yalnız ETHUSDT). Yeni collector başlatılmadı — OD-006 operatör kararı bekliyor.

**Phase 3 — OD-003 ANSWERED/IMPLEMENTED (2026-07-03):** Operatör A2+B2+C2 önerisini onayladı. Canonical resolver (`ami/identity/cycle_resolver.py`): symbol+family grouping + 4h continuity gate (mevcut LONG_HORIZON_MS'e dayalı) + `ami/states/engine.py`'den point-in-time dominant-structural-state sinyali (1h TF); cascade-continuity/shared-parent-event/observer-horizon sinyalleri v1'de NOT_IMPLEMENTED (belgeli sınırlılık, fabrikasyon değil — Obs §5.4'ün geri kalanı gelecek versiyona). Reset/censoring: 2-durum alt-kümesi (COMPLETED/RIGHT_CENSORED, cycle-seviyesinde tüm-üyeler-kapandı mantığı). Direction-conflict: flag-only, WAIT-eşdeğeri, otomatik çözülmedi.

## Phase 4 — Chart-native object foundation (BATCH-P4-001..004, Chart-Native Extension §4/§6/§7)

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| ami_candles | §4.1 Candle Object — closed-candle-only, `gaps`-çapraz-işaretli (F-B3) | candle_id | **3518** (ETHUSDT 1m+5m, canlı lookback; 0 OHLC ihlali; data_quality: AVAILABLE — mevcut pencerede 0 GAPPED, kod+testlerle doğrulandı) |
| ami_candle_morphology | §6.1 atomik özellikler + §6.3 3-durum close-quality (provisional eşik) | candle_id | 1:1 candle ile; REJECTION/ACCEPTANCE/FOLLOW_THROUGH NOT_IMPLEMENTED |
| ami_swings | §4.2 confirmed Swing (N=3 simetrik fraktal, known_at≠pivot_ts) | swing_id | **556** (294 HIGH/262 LOW; 0 known_at≤pivot ihlali) |
| ami_levels | §4.3 Level — SESSION/PREVIOUS_DAY/SWING; **level_definition_version=level-v2-boundary-safe** (F-B4/F-B1 sonrası) | level_id | **572** (eski `level-v1` — 4 satırı truncated/boundary-hatalı içeriyordu — kontrollü silindi, 0 kaldı); `touch_stats_point_in_time` kolonu her satırda **0** (F-B2: point-in-time-safe DEĞİL, bkz. aşağı) |
| ami_pushes | §7.1 Push — ardışık alternating swing çiftleri | push_id | **454** (228 UP/226 DOWN; 0 efficiency_ratio>1 ihlali — 2 gerçek bug bulunup düzeltildi, bkz. MIGRATION_LOG M-0006); liquidation_notional NOT_IMPLEMENTED |

Kaynak kod: `ami/chart/{candle_builder,candle_morphology,swing_extractor,level_registry,push_geometry}.py`. Hepsi idempotent, gerçek veriyle doğrulandı (2× koşum + bağımsız SQL çapraz-kontrol).

## Phase 5 — Feature dictionary + duplicate-engine denetimi (BATCH-P5-001)

**Feature dictionary (CN §19 karşılığı):** yukarıdaki 5 tablo + morfoloji/prominence/efficiency alanları AMI'nin TEK canonical chart-feature kaynağıdır. Her feature `known_at_ts` taşır ve test edilmiştir (candle: closed-only enforcement; swing: known_at>pivot_ts testli; level: known_at≥dönem-sonu-sınırı testli; push: known_at≥end_ts + pullback ile ileri taşıma testli).

**Duplicate-engine denetimi:** repo `tools/*.py` grep taraması → `tools/s34_v_engine_failure_anatomy.py:candle_features()` gerçek bir ÖNCEKİ (Phase 0'dan kalma) mum-morfolojisi hesaplayıcısı buldu — ANCAK farklı normalizasyon kullanıyor (ref_price bps, CN'nin range-oranı formülü değil) ve belirli bir tarihsel araştırmaya (FADE_DIRECTION) bağlı, dondurulmuş sonuçlarla. **Karar: DOKUNULMADI** (geçmiş araştırma scriptleri yeniden yazılmaz — master protokol §12/reproducibility).

## Phase 6-öncesi remediation (BATCH-P6-000 + P6-001, FABLE REVIEW B bulgularının düzeltmesi)

**F-B4 (pencere-kenarı truncation, MANIFEST bug — düzeltildi):** `level_registry` artık bir dönemin İLK mumunun gerçek sınırda olup olmadığını kontrol ediyor; değilse (lookback penceresi dönemin ortasında başlıyorsa) dönem tamamen atlanıyor. Gerçek veride 1 hatalı truncated SESSION_HIGH/LOW çifti tespit edilip kaldırıldı.

**F-B1 (known_at boundary, latent lookahead — düzeltildi):** `known_at_ts = max(son mum close, gerçek dönem-sonu sınırı)` — artık veri erken kesilse bile dönemin gerçekten bittiği zamandan önce "biliniyor" sayılmıyor.

**F-B3 (gaps çapraz-işaretleme — düzeltildi):** `candle_builder`, `microstructure.db:gaps` (stream=agg_trades) ile örtüşen pencereleri artık `data_quality=GAPPED` işaretliyor (mevcut 48h pencerede örtüşme yok, testlerle doğrulandı).

**F-B2 (touch-stats point-in-time güvenliği — bloklandı):** `ami_levels.touch_stats_point_in_time` her satırda 0. `ami/research/feature_gateway.py` (BATCH-P6-001) bu kolonları (`touch_count`/`rejection_count`/`acceptance_count`/`last_touch_ts`/`strength_score`) flag=1 olmadan **reddediyor** — point-in-time-safe bir yeniden-hesaplama motoru üretilene kadar Phase 6 bunlara erişemez.

**F-B5/F3 (zorunlu engine gateway — uygulandı):** `ami/research/feature_gateway.py` Phase 6'nın TEK sanksiyonlu erişim noktası: `fetch_events`/`fetch_cycles` REAL+PROXY karışımını reddeder (`assert_not_pooled`), `fetch_level_features`/`fetch_chart_feature` yalnız bilinen tablo/kolonlara (+ allowlist'li `equals` filtresi) izin verir, her başarılı erişim `researcher_exposure_ledger`'a (Phase 2 tablosu) kaydedilir. Kural: Phase 6+ research dalgaları YENİ chart-native feature ihtiyacı için bu gateway'i kullanır, script-içi ad-hoc formül icat etmez veya tabloyu doğrudan sorgulamaz.

## Phase 6 — W1: Cycle Integrity & Deduplication (BATCH-P6-002)

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| experiment_registry | Protocol §7.2 — dondurulmuş deney tanımı + verdict | experiment_id | **1** (E-W1-CYCLE-INTEGRITY-001, software_verdict=PASSED, scientific_verdict=ANSWERED_SUPPORTED) |
| experiment_results | Protocol §7.7 — deney metrikleri (canonical SQL, yalnız Markdown değil) | result_id (autoinc) | **12** (yalnız son snapshot — re-run önceki satırları siler, biriktirmez) |

**W1 bulgusu (gerçek veri, `feature_gateway` üzerinden):** raw_ledger_trades_n=270 → anchor_n=252 → **cycle_n=167** (anchor_to_cycle_ratio=0.66). 16 multi-route anchor (event_count dağılımı: 236×1/15×2/1×4). 15/167 cycle direction_conflict=1 (WAIT, çözülmedi). Cooldown sensitivity: 1h=152/2h=111/4h=88/6h=82/12h=55/24h=38 — canonical-v1(167) 1h ile 2h arası, çünkü 4h-gap'e ek olarak yapısal-durum-değişimi de zorla bölüyor (yalnız gap-tabanlı bir 4h pencereden FARKLI, daha fazla bölünme üretiyor).

**Bulunan+düzeltilen 2 bug:** (1) `feature_gateway._record_exposure` içerik-hash+ms-timestamp'ten exposure_id üretiyordu — aynı milisaniyede birden çok çağrı UNIQUE ihlaline yol açtı, `uuid.uuid4()` ile düzeltildi. (2) `experiment_results` INSERT'i idempotent değildi (re-run'da 12→24 satır), `DELETE ... WHERE experiment_id=?` eklendi.

**OD-011 (yeni):** Mevcut 11 K-S34-*/K-BUYFADE-* Knowledge Object ve SYSTEM_STATE route-N tabloları cycle-adjusted N kullanmadan üretildi — W1'in anchor_to_cycle_ratio=0.66 bulgusu bu N'lerin ortalama ~1.5x şişirilmiş olabileceğini gösteriyor. **Hiçbir bucket/KO değiştirilmedi** — yeniden-değerlendirme operatör kararına bırakıldı.

## Phase 6 — All-Timestamp Candidate Universe (BATCH-P6-003)

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| ami_candidate_universe | Protocol §17.8 — event varlığına koşullanmamış, her kapanmış mum slotu için candidate satırı (known-at-safe, gap-aware, deterministik) | candidate_id | **2932** (ETHUSDT/1m, 22 event-aligned + 2910 no-event) |

**Kapsam uyarısı (kod yazılırken bulundu, gizlenmedi):** `ami_candles` Phase 4'ten kalma bilinçli `lookback_hours=48` sınırlı pencereyle inşa ediliyor; candidate universe bu pencereyi miras alıyor. `event_n`/`anchor_n`/`independent_cycle_n` (W1) ise TÜM tarihi (2026-02-17→bugün) kapsıyor. Bu iki ölçek KARIŞTIRILMADAN, `ami/research/candidate_universe.py:freeze_and_record()` her ikisini de AYRI metrik olarak yazar (`*_all_history` vs `*_in_candidate_window`); oran metrikleri (`candidate_to_anchor_ratio`/`candidate_to_cycle_ratio`) yalnız pencere-kapsamlı sayıyı kullanır.

**4 denominatör (E-CANDIDATE-UNIVERSE-001, canonical.sqlite — W1'i süpersede etmez, ayrı deney):**
| Seviye | N | Kapsam |
|---|---|---|
| raw_candidate_n | 2932 | ~48h candle penceresi |
| anchor_n_in_candidate_window | 22 | aynı pencere |
| cycle_n_in_candidate_window | 17 | aynı pencere |
| event_n_all_history | 270 | tüm tarih (W1) |
| anchor_n_all_history | 252 | tüm tarih (W1) |
| independent_cycle_n_all_history | 167 | tüm tarih (W1) |

**OD-011 zenginleştirme + CT-005 (yeni, `CONTRADICTION_REGISTER.md`):** 11 KO'nun per-KO sınıflaması — **RECOMPUTE_REQUIRED (8):** K-S34-HOUR17-001 (FORWARD_VALIDATING, deploy adayı, en yüksek öncelik), K-S34-BOOK-PULL-001, K-S34-FUNDING-LEVEL-001, K-S34-MECH-COMPOSITE-001, K-S34-MGMT-6H-001, K-S34-MONDAY-VETO-001, K-S34-PRECASCADE-001, K-S34-SCALEIN-100-001. **AFFECTED (3, descriptive):** K-BUYFADE-SILENCE-INFO-001, K-LATENT-REGIME-001, K-S34-REFILL-CTX-001. `knowledge.sqlite` DEĞİŞTİRİLMEDİ — yalnız flag; canonical cycle-adjusted recomputation ayrı, operatör onaylı bir Phase 6 dalgası olarak planlanacak (bu batch YÜRÜTMEDİ).

## Phase 7A — Canonical Lifecycle Schema (SUPERSEDED BY v9 BELOW — historical M-0023 snapshot, CANONICAL_SCHEMA_VERSION=8, BATCH-P7A-CANON, M-0023)

**[2026-07-04] SÜPERSEDE EDİLDİ:** bu bölümdeki 536-transition/setup_version-NOT-NULL/lifecycle_status-CLOSED=266 sayıları artık gerçek DB'nin GÜNCEL durumunu YANSITMIYOR — `CANONICAL_SCHEMA_VERSION=9` (BATCH-P7A-P-CANON, M-0028) bu durumu 802-transition/setup_version-NULL/lifecycle_status-OPEN=270'e düzeltti. Bu bölüm yalnız TARİHSEL kayıt olarak korunuyor (M-0023'ün o anki gerçeği); güncel/canonical durum için aşağıdaki "Phase 7A-P — Canonical Provenance Migration (APPLIED)" bölümüne bakın.

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| ami_signal_lifecycle | Canonical signal identity (setup_id/version, source_event_id, independent_cycle_id, timestamp'ler, lifecycle_status) | signal_id | **270** (252 event, 18'inin 2. route'u var — hem LONG_\* hem SHORT_\* aynı anchor'a bağlı olabiliyor) |
| ami_lifecycle_transitions | Append-only state-transition ledger (previous/new status, known_at_ts, reason_code, transition_version) | transition_id | **536** (270 genesis + 266 terminal; 4 signal hâlâ OPEN — terminal_ts yok) |

Kaynak kod: `ami/lifecycle/{canonical_schema,canonical_backfill,migration_rehearsal}.py` (identity/validator/rebuild/backfill mantığı — Phase 7A.1'de disposable kopyada doğrulandıktan sonra hash-değişmeden gerçek DB'ye uygulandı) + `ami/warehouse/schema.py:_SCHEMA_PHASE7A` (aynı DDL'nin birebir kopyası, `init_schema()`'ya kayıtlı — `CANONICAL_SCHEMA_VERSION=8`). Field classification: `source_event_id`/`independent_cycle_id`/`signal_birth_ts`/`terminal_ts`/`symbol`/`setup_id`/`route_version`/`evidence_layer`/`is_proxy`=DETERMINISTIC_HISTORICAL_SAFE; `direction`=HISTORICAL_PROXY (route-adı önek-ayrıştırması, canonical route-registry yok); `setup_version`/`timeframe`/`first_known_ts`/`last_confirmation_ts`/`invalidation_ts`=NOT_IMPLEMENTED; `first_executable_ts`=FORWARD_ONLY (candle/mark proxy'sinden geriye dönük üretilmedi). Backup: `data/ami/backups/canonical_pre_phase7a_migration_20260704_130548.sqlite`.

**[2026-07-04 EK] PHASE_7A_PROVENANCE_GAP bulgusu:** yukarıdaki field-classification satırı yalnız Python kaynak kodunda (`FIELD_CLASSIFICATION` dict) vardı — canonical.sqlite'ta HİÇBİR tabloya/kolona yazılmamıştı. `ami_signal_lifecycle.provenance` kolonu yalnız düz bir batch-etiketi (`"batch-p7a-canonical-migration"`, 270 satırda AYNI) — per-field değil. Kapatma girişimi aşağıda (Phase 7A-P1, henüz disposable-only).

## Phase 7A-P1 — Field-Level Provenance Closure (APPLIED TO CANONICAL DB — BATCH-P7A-P-CANON, M-0028; disposable validation was BATCH-P7A-P1, M-0024)

**Kritik ayrım (operatör talebiyle açıkça tanımlandı):** `ami_signal_lifecycle`'da İKİ AYRI provenance ekseni vardır, birbirine KARIŞTIRILMAZ:
- **Row-level (`evidence_layer`/`is_proxy`)** — bu SATIRIN temsil ettiği anchor EVENT'in `ami_events.source_quality`'den gelen gerçek/proxy durumu (REAL_LIQUIDATION vs PROXY_\*). "is_proxy=0 (REAL)" **bu satırın HER ALANININ gerçek gözlem olduğu anlamına GELMEZ.**
- **Field-level (`ami_lifecycle_field_provenance`, yeni tablo)** — HER (signal_id, field_name) çifti için o ALANIN DEĞERİNİN nasıl türetildiği (DETERMINISTIC_HISTORICAL_SAFE/HISTORICAL_PROXY/FORWARD_ONLY/NOT_IMPLEMENTED/BLOCKED_BY_DATA). Örnek: bir satırın `is_proxy=0` (REAL anchor) olması ile aynı satırın `direction` alanının field-level `is_proxy=1` (HISTORICAL_PROXY, route-adı heuristiği) olması **eş zamanlı ve tutarlı** — iki farklı soruya cevap veriyorlar.

| Tablo/View | Amaç | PK | Doluluk (disposable) |
|---|---|---|---|
| ami_lifecycle_field_provenance | Her (signal_id,field_name) için field_classification/is_proxy/derivation_method/source_reference/limitations | provenance_id | 4320 (270 signal × 16 alan) |
| ami_lifecycle_direction_view | **Canonical query/export sözleşmesi:** `direction` hiçbir zaman classification'sız sunulmaz — bu view her zaman `direction_classification`/`direction_is_proxy` eşlik ettirir. Ham `ami_signal_lifecycle.direction` sorgusu teknik olarak mümkün ama yalnız iç-bookkeeping amaçlı, DIŞA sunum için KULLANILMAZ | (view) | — |

Kaynak kod: `ami/lifecycle/{canonical_field_provenance,provenance_rehearsal}.py`. Reconciliation: mevcut `data_quality_events`/`causal_assumption_registry`/`evidence_contamination`/`mt_family_registry`/`market_structure_versions` incelendi — hiçbiri per-field provenance karşılamıyor, paralel truth-layer kurulmadı (yalnız `signal_id`+`field_name` referans eder, identity/timestamp alanlarını KOPYALAMAZ).

## Phase 7A-P1 — Scope/Version Correction (v9, 16-field) + Semantic Closure (APPLIED TO CANONICAL DB — CANONICAL_SCHEMA_VERSION=9, BATCH-P7A-P-CANON, M-0028; disposable validation was BATCH-P7A-P1-V9, M-0025)

**Version contract düzeltmesi:** Phase 7A-P1'in ilk hâli (yukarıdaki bölüm) zımnen "CANONICAL_SCHEMA_VERSION 8→8" varsayıyordu — bu, repo'nun "her şema-şekli değişikliği bir version bump'a karşılık gelir" (v1→v8 boyunca kurulan hassasiyet) ilkesini ihlal ederdi. Operatör düzeltmesiyle: `ami/lifecycle/canonical_schema.py:LIFECYCLE_SCHEMA_VERSION` **1→2** (proposed canonical **v9**). `ami/warehouse/schema.py:CANONICAL_SCHEMA_VERSION` gerçek DB'de HÂLÂ **8** — v9 bump'ı yalnız ayrı bir canonical-migration onayıyla uygulanır.

**setup_version semantik düzeltmesi:** `field_classification` **NOT_IMPLEMENTED**, `derivation_method=not_computed`. Kolon artık `TEXT` (nullable; eskiden `TEXT NOT NULL` idi — `migrate_setup_version_nullable()` table-rebuild migration'ıyla gevşetildi, idempotent). Canonical kolon değeri **NULL** — `SETUP_VERSION_DEFAULT`="setup-v1" donmuş sabiti artık YALNIZ `generate_signal_id()`'in hash-girdisi (identity_version ile ASLA karıştırılmaz, `signal_id` bu değişiklikle sabit kaldı).

**terminal_ts semantik düzeltmesi:** `field_classification` **NOT_IMPLEMENTED** (`derivation_method` eskiden `verbatim_copy` idi). Canonical kolon değeri **NULL** — `ami_events.event_end_ts_ms`'in gerçek lifecycle TERMINAL geçiş zamanı olduğu doğrulanmadı (yalnız source-event'in kendi bitiş zamanı). Yeni internal-only `_terminal_transition_ts` alanı (`canonical_backfill.py`, tabloya YAZILMAZ) `ami_lifecycle_transitions`'ın TERMINAL_CLOSE zaman-damgasını AYNEN eskisi gibi sürdürüyor — 536 transition satırı bu düzeltmeden ETKİLENMEDİ.

**Actual value null/non-null matrisi (270 signal, provenance-row varlığı DEĞİL gerçek kolon değeri):**
| Alan | Durum |
|---|---|
| source_event_id, independent_cycle_id, signal_birth_ts, symbol, setup_id, route_version, evidence_layer, is_proxy, direction | 270/270 non-NULL |
| setup_version, terminal_ts, timeframe, first_known_ts, first_executable_ts, last_confirmation_ts, invalidation_ts | 270/270 NULL |

270 signal / 536 transition / direction=HISTORICAL_PROXY(270/270) / FORWARD_N=0 bu düzeltmeyle DEĞİŞMEDİ (doğrulandı).

## Phase 7A-P1 — Semantic Consistency Round 2: TERMINAL_CLOSE / LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER (APPLIED TO CANONICAL DB — BATCH-P7A-P-CANON, M-0028; disposable validation was BATCH-P7A-P1-V9R2, M-0026)

**LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER bulgusu:** yukarıdaki `terminal_ts=NULL/NOT_IMPLEMENTED` düzeltmesiyle, ledger'ın (`ami_lifecycle_transitions`) hâlâ AYNI unvalidated `ami_events.event_end_ts_ms`'i kullanarak bir `TERMINAL_CLOSE` transition'ı (OPEN→CLOSED) yazması ve `rebuild_current_state()`'in bunu CURRENT status olarak CLOSED'a çevirmesi arasında bir çelişki bulundu — terminal_ts alan-seviyesinde "bilmiyoruz" derken, ledger/status seviyesinde AYNI veriyle "biliyoruz, kapandı" iddia ediliyordu. Sınıflandırma: **LIFECYCLE_TERMINAL_SEMANTIC_BLOCKER** (consistent-but-wrong: denormalize kolon ile ledger-rebuild birbiriyle tutarlıydı, ikisi de aynı şekilde yanlıştı — bu bir cache/ledger diverjansı değil, semantik bir aşırı-iddiaydı).

**Taxonomy reconciliation:** `LifecycleReasonCode`/`TradeLifecycleState` incelendi — mevcut taksonomide "non-terminal, source-event-ended" için ayrı bir karşılık YOK (bu fazda yalnız OPEN/CLOSED kullanılabilir, path-label motoru yasak). Yeni bir isim (`SOURCE_EVENT_ENDED` vb.) icat ETMEK yerine: `backfill_lifecycle()` artık hiçbir lifecycle_status/transition iddiasını `event_end_ts_ms`'ten türetmiyor — her taze sinyal yalnız OPEN/SIGNAL_BIRTH. `event_end_ts_ms` kaybolmadı, `ami_events` tablosunda (source-event semantiğiyle) erişilebilir kalıyor.

**Düzeltme (`ami/lifecycle/canonical_backfill.py:correct_unvalidated_terminal_close()`):** gerçek DB'nin (M-0023, disposable-kopyada tekrarlanan) 266 eski TERMINAL_CLOSE→CLOSED satırını append-only bir **CORRECTION** transition'ıyla (`CLOSED→OPEN`, `transition_version=2`, `correction_of=<orijinal transition_id>`, `validate=False`) tersine çeviriyor. Orijinal satır SİLİNMEDİ/DEĞİŞTİRİLMEDİ — append-only ledger disiplini korundu.

| | ÖNCE | SONRA |
|---|---|---|
| lifecycle_status OPEN | 4 | **270** |
| lifecycle_status CLOSED | 266 | **0** |
| ami_lifecycle_transitions toplam | 536 (270 SIGNAL_BIRTH+266 TERMINAL_CLOSE) | **802** (+266 CORRECTION, hiçbiri silinmedi) |
| current_state_rebuild_consistency | consistent=True (yanlış) | consistent=True (doğru) |

270 signal identity / source_event_id / independent_cycle_id DEĞİŞMEDİ. `ami.lifecycle.canonical_schema.UNKNOWN_SETUP_VERSION_TOKEN` (=`SETUP_VERSION_DEFAULT`) eklendi — supersession contract: gerçek bir setup_version kaynağı bulunursa mevcut signal_id'ler sessizce mutate edilmeyecek (yeni IDENTITY_VERSION veya açık supersession-kaydı, karar geleceğe bırakıldı).

## Phase 7A-P1 — Effective-Ledger Semantics Round 3: Two-Layer Ledger Contract (APPLIED TO CANONICAL DB — BATCH-P7A-P-CANON, M-0028, LIFECYCLE_SCHEMA_VERSION=3; disposable validation was BATCH-P7A-P1-V9R3, M-0027)

## Phase 7A-P — Canonical Provenance Migration (APPLIED, BATCH-P7A-P-CANON, M-0028, CANONICAL_SCHEMA_VERSION=9)

Gerçek `data/ami/canonical.sqlite`'ın GÜNCEL, CANONICAL durumu (yukarıdaki 4 "Phase 7A-P1" bölümünün tamamı bu tek batch'te gerçek DB'ye uygulandı):

| | Değer |
|---|---|
| schema_versions.canonical_warehouse | **9** |
| ami_signal_lifecycle | **270** satır |
| ami_lifecycle_transitions (raw, immutable) | **802** satır (270 SIGNAL_BIRTH + 266 TERMINAL_CLOSE[artık geçersiz kılınmış, append-only korunan] + 266 CORRECTION) |
| ami_lifecycle_effective_transitions (effective view) | **270** satır (1/sinyal, yalnız genesis) |
| ami_lifecycle_field_provenance | **4320** satır (270×16, 0 eksik, 0 duplicate) |
| lifecycle_status dağılımı | **OPEN=270, CLOSED=0** |
| setup_version | 270/270 NULL, NOT_IMPLEMENTED |
| terminal_ts | 270/270 NULL, NOT_IMPLEMENTED |
| first_executable_ts | 270/270 NULL, FORWARD_ONLY |
| direction | 270/270 HISTORICAL_PROXY |
| effective CLOSED interval count | **0** |

Backup'lar: `data/ami/backups/canonical_pre_phase7a_p_provenance_migration_20260704_153645.sqlite` (temiz v8, restore noktası), `canonical_ACCIDENTAL_V9_SCHEMA_DRIFT_snapshot_20260704_161554.sqlite` (forensic, SCHEMA_DRIFT_BLOCKER anı), `canonical_post_phase7a_p_provenance_migration_v9_20260704_162055.sqlite` (başarılı migrasyon sonrası).

**Test-isolation safety closure (kalıcı altyapı):** `ami/warehouse/schema.py:REAL_CANONICAL_PATH_IMMUTABLE`/`_TEST_ISOLATION_ACTIVE`/`connect()`'in fail-closed guard'ı + `tests/conftest.py`'nin session-scoped autouse fixture'ı — artık hiçbir test suite çalıştırması gerçek canonical.sqlite'a yazamaz (bu, SCHEMA_DRIFT_BLOCKER olayının kök nedenini kalıcı olarak kapatıyor).

**Bulgu:** Round 2'nin append-only CORRECTION düzeltmesi (OPEN→CLOSED/TERMINAL_CLOSE→OPEN/CORRECTION) `rebuild_current_state()` (latest-row-wins) için STATUS'u doğru veriyordu, ama raw `ami_lifecycle_transitions`'ı DOĞRUDAN okuyan (rebuild_current_state'i çağırmayan) bir downstream INTERVAL/DURATION sorgusu için hiçbir gate yoktu — 266 düzeltilmiş sinyalin her biri için ham TERMINAL_CLOSE satırından **sahte bir CLOSED interval** hesaplanabilirdi. Bu üç satır (genesis+TERMINAL_CLOSE+CORRECTION) gerçek bir close-reopen hareketi DEĞİL — ilk CLOSED iddiası bilimsel olarak geçersizdi, correction yalnız onu supersede etmek için var.

**Çözüm — canonical iki-katmanlı sorgu ayrımı** (mevcut repository'de eşdeğer yapı yok, reconciliation yapıldı, paralel truth-layer KURULMADI):
- **immutable raw audit ledger** = `ami_lifecycle_transitions` (değişmedi, her satır sonsuza dek korunur, artık-geçersiz-kılınmışlar dahil).
- **effective lifecycle ledger** = yeni `ami_lifecycle_effective_transitions` VIEW — (a) `correction_of` ile başka bir satır tarafından hedeflenen (superseded) her satırı, VE (b) hedefinin TAM TERSİ (pure reversal: `correction.previous_status==original.new_status AND correction.new_status==original.previous_status`) olan bir CORRECTION satırını dışlar. Metadata-only bir correction (AYNI yöndeki bir transition'ı yeniden onaylayan) pure-reversal SAYILMAZ, dışlanmaz — gerçek bir transition olduğu için effective view'da kalır.

**Yeni fonksiyonlar (`ami/lifecycle/canonical_schema.py`):** `effective_lifecycle_status()` (effective-view eşdeğeri rebuild), `count_effective_closed_signals()` (gerçek/geçerli CLOSED interval'i olan sinyal sayısı — canonical existence-check, terminal/hold-duration araştırması bunu ÖNCE kontrol etmeli).

**`validate=False` fail-closed safety contract (`insert_transition()`):** yalnız `reason_code=CORRECTION` + var olan bir `correction_of` BİRLİKTE verildiğinde izinli; hedef yoksa veya aynı hedef ikinci kez farklı bir correction ile hedeflenirse `LifecycleIntegrityViolation` (fail-closed). Normal transition writer'lar (`backfill_lifecycle`) bu bypass'a yapısal olarak erişemez.

**Canonical query contract (OPEN != live):** her sinyal artık lifecycle_status=OPEN, ama bu "aktif/live" anlamına GELMEZ — yalnız "terminal durum hiç doğrulanmadı" anlamına gelir. UNKNOWN/CENSORED/UNRESOLVED taksonomiye icat EDİLMEDİ — mevcut `terminal_ts=NOT_IMPLEMENTED` field-provenance kaydı (yukarıdaki bölüm) canonical gate olarak kullanılır.

| | Sayı |
|---|---|
| raw ledger rows (immutable) | **802** (değişmedi) |
| effective ledger rows | **270** (1/sinyal — yalnız genesis) |
| superseded/dışlanan satır | 266 (TERMINAL_CLOSE) + 266 (pure-reversal CORRECTION) = 532 |
| effective CLOSED interval count | **0** |
| current-state (raw + effective) | OPEN=270 / CLOSED=0, ikisi birebir uyumlu |

## Phase 7B — Path/MFE/MAE Observation Engine (APPLIED, BATCH-P7B-CANON, CANONICAL_SCHEMA_VERSION=10)

Disposable doğrulama: BATCH-P7B-0 (operatör onayı: "APPROVE PHASE 7B-0 DISPOSABLE PATH ENGINE FOUNDATION") + BATCH-P7B-0.1 (operatör onayı: "APPROVE PHASE 7B-0.1 DISPOSABLE SEMANTIC CLOSURE"). Canonical migration: BATCH-P7B-CANON (operatör onayı: "APPROVE PHASE 7B CANONICAL PATH METRICS MIGRATION") — schema v9→v10.

| Tablo | Amaç | PK | Doluluk |
|---|---|---|---|
| ami_lifecycle_path_observations | Sabit-ufuk (scalp_30m/1h, swing_4h/24h — `ami/research/w4_post_event_path_taxonomy.py:PATH_HORIZONS_MS`'ten birebir reuse) MFE/MAE/endpoint-return/timing/vol-normalizasyon ölçüm katmanı, `path_definition_version=path-v2` | observation_id | **1080** (270 sinyal×4 ufuk); UNIQUE(signal_id,horizon_name,path_definition_version); FK→ami_signal_lifecycle |

Kaynak kod: `ami/lifecycle/{path_schema,path_metrics,path_field_provenance,path_canonical_migration}.py` (`ami/warehouse/schema.py:_SCHEMA_PHASE7B` — disposable-doğrulanmış DDL'yle byte-for-byte aynı, fingerprint `3a26ffa86ecec9d8b63eff9455e3cfbbd594cc59eb36896feeba4d3bf232f1e7`). 31 kolon (30 operatör-spesifikasyonu + `volatility_status`), 12 CHECK constraint.

**İKİ BAĞIMSIZ STATÜ EKSENİ (operatör kilidi, 7B-0.1):**
- `observation_status` — yalnız PATH hesaplanabilirliği (5 değer: OK/MISSING_REFERENCE_PRICE/MISSING_INTERNAL_GAP/EXCLUDED_NO_HORIZON_DATA/NOT_COMPUTABLE_DIRECTION).
- `volatility_status` — yalnız vol-normalizasyon baseline'ının kullanılabilirliği (OK/INVALID_VOLATILITY_BASELINE/NOT_APPLICABLE), DB-level CHECK ile eşleştirilmiş (`NOT_APPLICABLE` ⟺ `observation_status≠OK`). Path geçerli ama vol-baseline geçersizse (`observation_status=OK`/`volatility_status=INVALID_VOLATILITY_BASELINE`) endpoint/mfe/mae/timing/horizon_outcome_class KORUNUR, yalnız 3 `*_anchor_vol_units` alanı NULL kalır.

**Gerçek veri (2026-07-04):** observation_status: OK=914, MISSING_INTERNAL_GAP=153, EXCLUDED_NO_HORIZON_DATA=13, MISSING_REFERENCE_PRICE=0, NOT_COMPUTABLE_DIRECTION=0. volatility_status: OK=912, NOT_APPLICABLE=166, INVALID_VOLATILITY_BASELINE=2. independent_cycle_id-deduplicated denominatör: 167 (W1 ile birebir); source_event_id: 252 (18 multi-route event, N şişirilmedi).

**Candle-boundary semantiği:** `reference_price`=signal_birth_ts'den ÖNCE tam kapanmış son 1m candle close'u; `effective_path_start_ts`=signal_birth_ts'den SONRAKİ ilk candle open'ı. Birth'i saran kısmi candle (open<birth<close) HER İKİSİNDEN de hariç — high/low'u asla MFE/MAE'ye girmez. `expected_candle_count` bu nedenle `effective_path_start_ts`'e (signal_birth_ts'e DEĞİL) göre hesaplanır — aksi halde her non-aligned sinyal sahte bir gap_count≥1 alırdı (yapısal sınır-etkisi, veri kalitesi sorunu değil).

**Zero-ekstremum timing (7B-0.1):** t=0 referans noktası path'in bir parçası — gerçek fiyat hiç favorable/adverse'e ulaşmazsa `time_to_mfe_ms`/`time_to_mae_ms`=0 (rastgele bir gerçek candle'ın ts'i DEĞİL). Eşitlik durumunda EN ERKEN ts kazanır (insertion-order değil).

**MFE/MAE formülleri (LONG/SHORT sign-symmetric, mirror-path testiyle kilitli):**
```
LONG:  mfe_bps=max(0,max((high-ref)/ref*1e4));  mae_bps=min(0,min((low-ref)/ref*1e4))
SHORT: mfe_bps=max(0,max((ref-low)/ref*1e4));   mae_bps=min(0,min((ref-high)/ref*1e4))
```
`endpoint_return_bps`/`horizon_outcome_class` yön-BAĞIMSIZ (aynı anchor'daki LONG/SHORT sinyal için birebir aynı) — `horizon_outcome_class`, `ami.research.w4_post_event_path_taxonomy.classify_path()`'in CONTINUATION/REVERSAL/CHOP taksonomisi birebir reuse edilerek üretilir (paralel taksonomi icat edilmedi).

**Vol-normalizasyon adlandırması (7B-0.1, Seçenek A):** `endpoint_return_anchor_vol_units`/`mfe_anchor_vol_units`/`mae_anchor_vol_units` — "anchor_vol" infiksi, paydanın (`realized_vol_at_anchor`, 60-candle log-return stdev, ufuktan BAĞIMSIZ — aynı sinyalin 4 ufuk satırında birebir aynı) ufuklar-arası sigma-karşılaştırılabilir OLMADIĞINI isimde netleştirir.

**Field-level provenance (`ami_lifecycle_field_provenance`, MEVCUT tablo, paralel tablo YOK — `path_observations.*` namespace):** 23 substantive alan × 270 sinyal = **6210** yeni satır (mevcut 4320 ile toplam **10530**, 0 eksik/duplicate). `realized_vol_at_anchor`=DETERMINISTIC_HISTORICAL_SAFE (tek proxy-olmayan vol-bitişik alan — reference_price/direction'a dokunmuyor); `reference_price`/`endpoint_return_bps`/`horizon_outcome_class`=HISTORICAL_PROXY (yön-bağımsız neden: kısmi-candle-dışlama); `mfe_bps`/`mae_bps`/timing/`intrabar_order_status`/`*_anchor_vol_units` (mfe/mae)=HISTORICAL_PROXY (iki üst-üste neden: kısmi-candle-dışlama + `direction` alanının kendisi HISTORICAL_PROXY). Proxy→safe downgrade guard'ı kodlu (`PathFieldProvenanceDowngradeViolation`).

**Gateway erişimi (`ami/research/feature_gateway.py`):** `fetch_lifecycle_signals()` (curated kolon + REAL/PROXY pooling guard, `ami_signal_lifecycle.evidence_layer`'ın kendi REAL/PROXY vocabulary'si — `ami_events.source_quality`'nin REAL_LIQUIDATION/PROXY_* ile KARIŞTIRILMAZ), `fetch_lifecycle_effective_transitions()` (raw `ami_lifecycle_transitions` DEĞİL, effective view — superseded/pure-reversal TERMINAL_CLOSE asla terminal-evidence olarak sunulmaz), `fetch_path_observations()` (equals allowlist).

**Yan bulgu+düzeltme:** `ami/lifecycle/provenance_rehearsal.py`'nin `provenance_row_counts()`/`provenance_content_hash()`'i `ami_lifecycle_field_provenance` üzerinde scope'suz sorgu yapıyordu (tablo artık 2. bir yazıcı tarafından paylaşılınca 4320 varsayımı 10530 ile çakıştı) — `provenance_version=?` filtresiyle düzeltildi.

**270 OPEN ≠ aktif pozisyon (tekrar vurgu):** `count_effective_closed_signals()`=0 sabit kaldı; path-observation backfill hiçbir terminal_ts/invalidation_ts/last_confirmation_ts türetmedi, yalnız FROZEN, sabit ufuklara karşı ölçüm yaptı.
