# AMI × S34 MASTER RECONCILIATION, ROADMAP, IMPLEMENTATION AND MULTI-MODEL EXECUTION PROTOCOL

**Version:** v1.1 (v1.0 + canonical documentation update policy amendment — token efficiency)
**Status:** CANONICAL — bu dosya her oturumda geçerli çalışma sözleşmesidir.
**Kaydeden:** Fable 5, 2026-07-03, operatör talimatıyla.

Bu görev mevcut AMI × S34 repository'sini sıfırdan yeniden yazma görevi DEĞİLDİR.

## Amaç

1. Beş canonical belgeyi tamamen incelemek ve uzlaştırmak.
2. Repository'de gerçekten en son nerede kalındığını bulmak.
3. Mevcut doğrulanmış implementation'ları korumak.
4. Eksik, çelişkili, bozuk ve bloke bileşenleri belirlemek.
5. Ayrıntılı Master Roadmap ve Implementation Dependency Map oluşturmak.
6. Çalışmayı doğru model, doğru phase ve doğru bilimsel kapılarla yürütmek.
7. Research-only, güvenli ve geri alınabilir implementation'ı kontrollü batch'ler hâlinde tamamlamak.
8. Her aşamada canonical state, test, verdict ve progress dokümanlarını güncel tutmak.

Model kullanımı: **Fable 5** = planlama/reconciliation/repo-audit; **Sonnet 5** = kontrollü implementation batch'leri; **Opus 4.8** = kritik wave review/acceptance. Model kendiliğinden değiştirilemez: değişim noktasında dur, state/progress dosyalarını güncelle, operatöre hangi modele geçileceğini açıkça bildir. Operatör doğrulamadan sonraki aşamaya geçme.

---

## 1. Canonical belgeler

1. `reports/research/s34/AMI Reconciliation Initial Report.docx`
2. `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md`
3. `AMI_S34_CANONICAL_ARTIFACT_RECONSTRUCTION_AND_CONTINUOUS_UPDATE_PROTOCOL_v1.0.md`
4. `AMI_FORWARD_INTELLIGENCE_OBSERVATORY_TIMING_DASHBOARD_v1.0_COMPLETE.md`
5. `AMI_CHART_NATIVE_PRICE_STRUCTURE_INTELLIGENCE_EXTENSION_v1.0_COMPLETE.md`

İncelenecekler: canonical araştırma soruları, soru/test aileleri, research engine'leri, engineering/collector gereksinimleri, event/cycle/path yapıları, observer'lar, dashboard gereksinimleri, chart-native yapılar, governance, safety sınırları, evidence/permission katmanları, Definition of Done. Belgeler arası dependency / overlap / duplicate / terminology-schema-roadmap conflict / precedence / supersession / safety / missing-prerequisite ilişkileri çıkarılır. Yüzeysel özet yasak.

## 2. Belge rolleri ve precedence

- **Whitepaper** — constitutional: bilimsel anayasa, evidence hierarchy, question-family sistemi, Knowledge Object, LONG/SHORT cycle intelligence, promotion/demotion, validation, epistemic governance, engineering felsefesi.
- **Artifact Reconstruction Protocol** — canonical: artifact inventory, canonical state selection, duplicates, supersession, contradiction preservation, evidence-layer separation, canonical SQL warehouse, lineage, experiment/alpha/question registry, report regeneration, operator decision queue.
- **Forward Observatory Spec** — canonical: forward observation, event/cycle/path recording, structural-cycle N, timing ledger, position path, orderless observers, forward readiness, forward-facing dashboard/API.
- **Chart-Native Extension** — proposed extension: candle morphology, swing grammar, push geometry, liquidity sweep, breakout/retest, compression, channel, relative strength, session structure, chart observation registry, Q867–Q1058, setup lifecycle.
- **Reconciliation Initial Report** — rehber (spec'lerin yerine geçmez): precedence, conflict identification, dependency order, implementation waves, risk register, paralel workstream sınırları.

Frozen experiment spec, immutable Decision Record ve doğrudan reproducible evidence > mutable narrative summary. Repo audit'te daha doğru precedence/explicit supersession bulunursa `CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md` içinde gerekçelendir. Hiçbir çatışmayı sessizce harmonize etme.

## 3. Model orchestration

### STAGE A — FABLE 5
Görev: 5 belgeyi reconcile et; repo+runtime incele; son doğrulanmış checkpoint'i bul; current/target gap çıkar; risk+dependency belirle; Question Coverage Matrix tasarla; Master Roadmap ve implementation batch planı oluştur.

Yasak (Stage A): production kod değişikliği, schema migration, collector başlatma, observer aktivasyonu, dashboard implementasyonu, research test batch çalıştırma, runtime process durdurma/başlatma, live/shadow dosyalarına dokunma. Sadece read-only audit + audit/roadmap Markdown/CSV artifact'leri.

Bitiş koşulu: 5-belge reconciliation, repo/runtime audit, last verified checkpoint, gap analysis, protected-components manifest, precedence register, question coverage tasarımı, Master Roadmap, dependency graph, first safe implementation batch, model handoff package. Sonra şu blokla DUR:

```
==================================================
MODEL SWITCH REQUIRED
Completed model: FABLE 5
Next required model: SONNET 5
Reason: / Completed artifacts: / Last verified checkpoint: /
Next exact implementation batch: / Protected components: /
Open operator decisions: / Resume instruction for Sonnet 5:
==================================================
```

### STAGE B — SONNET 5
Görev: canonical roadmap'i takip et; kontrollü batch'ler uygula (migration, registry, event/cycle/path, API, observer, dashboard); testleri çalıştır; her batch sonunda state+documentation güncelle; kritik wave checkpoint'lerinde dur. Tek seferde bütün roadmap'i uygulamaya çalışma. Her batch: küçük, geri alınabilir, dependency-safe, test edilebilir, documentation-complete, protected live/shadow'dan izole.

Opus review kapıları:
- **REVIEW A** (Phase 1–3): canonical reconciliation; evidence/timestamp integrity; event/cycle/path foundation.
- **REVIEW B** (Phase 4–5): chart-native object foundation; shared feature engines; known-at safety; duplicate feature prevention.
- **REVIEW C** (Phase 6–7): research protocol; contamination; multiple testing; scientific verdicts; LONG/SHORT transition; hold/exit/re-entry.
- **REVIEW D** (Phase 8–9): forward observatory; dashboard; canonical DB/API consistency; no-permission boundary.
- **REVIEW E** (Phase 10 / World Model öncesi): prerequisites; sample readiness; OOD; calibration; causal/evidence safety.

Kapı geldiğinde Sonnet: aktif batch'i tamamla, testleri çalıştır, rollback doğrula, SYSTEM_STATE.md + progress ledger güncelle, git diff kaydet, live/shadow diff doğrula, Opus Review Package hazırla; sonra şu blokla DUR:

```
==================================================
CRITICAL REVIEW CHECKPOINT
Completed model: SONNET 5
Next required model: OPUS 4.8
Review checkpoint: / Completed phases/batches: / Changed files: /
Tests passed: / Tests failed: / Scientific verdict changes: /
Blocked questions: / Live/shadow diff: / Exact review scope for Opus 4.8:
==================================================
```

### STAGE C — OPUS 4.8
Geniş implementation YAPMAZ. Denetler: architectural consistency, document compliance, hidden dependency inversion, duplicate canonical truth, schema conflicts, event/cycle identity, timestamp/lookahead safety, split contamination, researcher exposure, multiple testing, statistical independence, scientific protocol, incorrect evidence promotion, observer→order leakage, route/version mismatch, paper/shadow/forward/live mixing, rollback validity, migration safety, dashboard/source-of-truth boundaries, storage/compute risks, missing tests, incomplete DoD.

Verdict sınıfları: `ACCEPTED | ACCEPTED_WITH_MINOR_FIXES | REQUIRES_REMEDIATION | ARCHITECTURAL_BLOCKER | SCIENTIFIC_BLOCKER | SAFETY_BLOCKER`

Her bulgu: `finding_id, severity, affected_phase, affected_files, description, evidence, required_fix, required_test, permission_effect, blocking_status`. Review sonunda şu blokla DUR:

```
==================================================
REVIEW COMPLETE
Completed model: OPUS 4.8
Next required model: SONNET 5
Review verdict: / Accepted components: / Blocking findings: /
Required remediation: / Required new tests: / Operator decisions: /
Exact resume instruction for Sonnet 5:
==================================================
```

Remediation sonrası gerekirse aynı checkpoint tekrar Opus'a gider.

### Fable'a yeniden dönme şartı
Sadece: roadmap temel varsayımları geçersizleşirse; büyük yeni artifact/canonical spec eklenirse; canonical warehouse/topology tamamen değişirse; phase sırası ciddi yeniden tasarım gerektirirse; Opus system-level architectural reset isterse. Bu durumda `FABLE 5 REPLANNING REQUIRED` yazıp dur.

## 4. Resume-from-current-state zorunluluğu

Sıfırdan implementasyon YOK. Önce son doğrulanmış çalışma noktası: SYSTEM_STATE.md, session/result raporları, git log/status/diff, Decision Records, experiment registry, Knowledge Objects, progress kayıtları, çalışan process/PID'ler, collector/watchdog state, DB son timestamp'leri, forward/shadow ledger, son test raporları, schema/migration sürümleri, dashboards, APIs, chart/candle/swing kodu, archived/superseded outputs.

Her hedef bileşen statüsü: `EXISTS_VALIDATED | EXISTS_UNVALIDATED | PARTIALLY_IMPLEMENTED | CONFLICTING_IMPLEMENTATION | SUPERSEDED | BROKEN | MISSING | BLOCKED | NOT_APPLICABLE`

Doğrulanmış sistemi yeniden yazma; paralel ikinci implementation oluşturma; aynı isimli farklı-logic bileşenleri otomatik birleştirme. Implementation yalnızca en erken eksik dependency'den devam eder. Tamamlanmış testleri sebepsiz yeniden tasarlama; reproducibility şüpheliyse yeniden doğrula ve sebebi kaydet. Çalışan collector/paper/shadow/forward/live process'lerini inceleme amaçlı durdurma; process değişikliği = operator approval.

## 5. Question coverage zorunluluğu

Q001–Q1058 aralığındaki bütün MEVCUT soru metinlerini ve ailelerini çıkar → `QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv`. Zorunlu kolonlar:

`question_id, canonical_parent, question_family, source_document, question_text, text_status, current_status, existing_evidence, evidence_layer, required_data, required_features, experiment_engine, minimum_event_n, minimum_cycle_n, minimum_day_n, required_regimes, required_sessions, required_forward_duration, historical_testable, forward_required, blocked_by, implementation_phase, retry_condition, permission_ceiling, final_verdict, last_updated`

Hiçbir mevcut soruyu sessizce atlama. Canonical metni olmayan ID'leri UYDURMA → `MISSING_CANONICAL_TEXT`. Soruları 1058 ayrı script/issue/task'a dönüştürme; duplicate/child/varyasyonları canonical parent questions / experiment families / shared feature engines / shared research engines / shared control generators altında grupla. Her family için: kapsanan sorular, data dependency, ortak engine, negative control'lar, test phase'i.

## 6. Tamamlanma ve verdict tanımı

Tamamlanma ≠ bütün hipotezler pozitif. Dürüst statüler:
`ANSWERED_SUPPORTED | ANSWERED_FALSIFIED | REJECTED_ECONOMIC | REJECTED_STATISTICAL | INSUFFICIENT_SAMPLE | BLOCKED_BY_DATA | BLOCKED_BY_FEATURE | BLOCKED_BY_REGIME | FORWARD_ACCUMULATING | FUTURE_PHASE | DUPLICATE | RETIRED | SUPERSEDED | INCONCLUSIVE`

Timestamp-safe data ile çalışabilen historical/engineering testlerini tamamla. Eksik data / yeni forward örneklem gerektiren soruları zorla çalıştırma; bunlar için gerekli collector, feature, min event N, min independent-cycle N, min gün, rejimler, session coverage, retry condition, permission ceiling tanımla. Engineering component'ın çalışması ≠ hipotez supported.

## 7. Phase sırası

```
Phase 0 — Repository audit and safety
Phase 1 — Canonical reconciliation
Phase 2 — Evidence, timestamp and contamination integrity
Phase 3 — Event/cycle/path foundation
Phase 4 — Chart-native object foundation
Phase 5 — Shared feature engines
Phase 6 — Historical research waves
Phase 7 — Lifecycle, entry, hold, exit and transition research
Phase 8 — Forward observatory
Phase 9 — Canonical dashboard
Phase 10 — Advanced research prerequisites
```

DoD geçmeden bağımlı phase'e geçme. Paralel workstream yalnız: aynı canonical warehouse, aynı event/cycle identity, aynı timestamp contract, aynı version vocabulary, aynı evidence-layer taxonomy, aynı safety boundary kullanılıyorsa.

## 8. Phase 0 — Repository audit and safety

İncele: repo/artifact inventory, file hash, duplicate/semantic-duplicate files, superseded reports, DB/schema/table inventory + integrity + timestamp coverage, collector inventory, shadow/forward/live process inventory + PID/health, experiment/report inventory, Question Registry, Knowledge Object state, Failure Archive, event/cycle/path implementation, chart/candle/swing implementation, duplicate feature engines, dashboards, APIs, migration history, storage/DB growth/compute, archive requirements, protected files/processes, current-vs-target gaps.

Phase 0 çıktıları: `LAST_VERIFIED_CHECKPOINT.md, REPOSITORY_RUNTIME_AUDIT.md, DOCUMENT_RECONCILIATION_MATRIX.md, CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md, CURRENT_STATE_VS_TARGET_GAP_ANALYSIS.md, PROTECTED_COMPONENTS_MANIFEST.md, UNTOUCHED_LIVE_SHADOW_COMPONENTS.md, MASTER_ROADMAP.md, IMPLEMENTATION_DEPENDENCY_GRAPH.md, FIRST_SAFE_IMPLEMENTATION_BATCH.md`

Phase 0 sonunda implementation'a GEÇME; Fable→Sonnet handoff checkpoint'inde dur.

## 9. Historical research zorunlulukları

Öncelikli family'ler: cycle integrity/dedup; unconditional LONG/SHORT genesis; failed-fade LONG; early LONG/SHORT entries; T0/delayed/late entries; candle morphology; swing grammar; push/momentum geometry; liquidity sweep; breakout/retest; compression; channel; relative strength; session structure; structural location; scalp/intraday/swing separation; post-event path taxonomy; position-aware action comparison; signal aging; state aging; market clock; competing-risk hold; stop taxonomy; re-entry taxonomy; setup cancellation; LONG↔SHORT transitions; multi-timeframe conflict; multi-sensor liquidation/cascade proxy reconstruction; real-liquidation vs proxy.

Her family için zorunlu: all eligible events; matched non-event controls; event-never-arrived controls; cycle-grouped chronological split; purge+embargo; feature_known_at enforcement; event N; independent-cycle N; active day N; regime+session coverage; realistic fees+fill model; missed fills; top-k day/cycle removal; threshold-neighbour stability; family-level multiple-testing correction; contamination ledger; researcher-exposure ledger; simple baselines; WAIT + NO_TRADE benchmark; opposite-direction control; matched random-time control.

Historical sonuçlar forward evidence olarak kaydedilemez.

## 10. Software vs scientific testler

Mümkün olanları çalıştır: unit, integration, mutation, data-integrity, schema, migration, reconciliation, timestamp, lookahead, idempotency, restart, duplicate, rollback, API consistency, dashboard-to-SQL consistency, scientific-protocol, negative-control, reproducibility, performance/capacity.

`software_verdict` ve `scientific_verdict` AYRI alanlar (ör. PASSED + FALSIFIED, PASSED + INSUFFICIENT_SAMPLE). "Bütün testler yeşil" ≠ alpha supported.

## 11. Forward test protokolü

candidate shortlist → exact frozen definition → immutable candidate version → activation timestamp → forward N=0 → orderless observer → min independent-cycle N → day/regime/session/volatility coverage → operator review.

Historical replay ≠ forward evidence. Observer aktivasyonu öncesi kayıtlar `HISTORICAL` veya `REPLAY` kalır. Candidate tanımı material değişirse: yeni version + yeni activation timestamp + forward N sıfırla + eski geçmişi koru. Min N'ye ulaşmak otomatik promotion DEĞİLDİR.

## 12. Güvenlik sınırları (değiştirilemez)

Live executor'a dokunma. Shadow policy davranışını değiştirme. `.env` / API key'e dokunma. Leverage / position sizing / order logic / stop-TP logic değiştirme. Observer'dan order üretme. Dashboard'u canonical source yapma. Min N'de otomatik promotion yok. Historical'ı forward gibi gösterme. Real liquidation ile proxy cascade aynı population'da birleşmez. Missing data ≠ zero. Partial candle ≠ closed candle. Swing pivot `known_at_ts` öncesi kullanılamaz. Yeni chart pattern ≠ otomatik alpha. Phase 10 / World Model prerequisites'siz açılmaz. Research output ≠ operational permission. Eski rejected/falsified sonuç silinmez. Dashboard her zaman canonical SQL + test edilmiş research view'lardan çalışır; asla canonical source değildir.

### 12.1 Bucket freeze politikası (operatör ek talimatı, 2026-07-03)

- Mevcut bucket'lar research scope içindedir; ilgili soru/experiment aileleri kapsamında **Phase 6–7'de yeniden değerlendirilir**.
- **Phase 0–5'te bucket'lara yalnız** inventory, definition verification, freeze ve dependency audit yapılır; **bucket koşulları optimize edilmez**.
- Material değişiklik gerekiyorsa mevcut frozen bucket değiştirilmez: **yeni version oluşturulur ve forward N=0'dan başlatılır**; eski bucket geçmişi korunur.

## 13. Operator approval gerektirenler

live/shadow behavior değişikliği; operational permission değişikliği; canonical cycle definition değişikliği; reset/censoring policy; irreversible/destructive schema migration; veri silme; archive taşıma; yüksek storage maliyetli collector aktivasyonu; yeni forward observer aktivasyonu; frozen experiment değişikliği; route definition; risk/sizing/leverage; running process restart; insan kararı gerektiren precedence conflict; chart-native family'nin forward observer'a dönüştürülmesi.

Approval GEREKTİRMEYEN (phase sırasına göre serbest): read-only audit; reversible research-only schema hazırlığı; tests; reports; registry construction; historical analysis; negative controls; documentation; isolated API/dashboard development; historical/replay feature computation.

## 14. Canonical documentation update policy (v1.1 AMENDMENT — token verimliliği)

Her batch sonunda bütün Markdown dosyalarını yeniden yazma.

**Her implementation batch sonunda zorunlu olarak yalnızca:**
- `SYSTEM_STATE.md`
- `IMPLEMENTATION_PROGRESS_LEDGER.md`

**Test çalıştırıldıysa ayrıca:**
- `TEST_STATUS_LATEST.md`

**Yalnızca ilgili değişiklik olduğunda:**
- `MIGRATION_LOG.md` — yalnız schema/migration değiştiğinde
- `SCHEMA_DICTIONARY.md` — yalnız tablo/kolon/index/data contract değiştiğinde
- `FAILURE_ARCHIVE.md` — yalnız yeni failed/falsified/rejected sonuç olduğunda
- `CONTRADICTION_REGISTER.md` — yalnız yeni contradiction bulunduğunda/çözüldüğünde
- `AMI_S34_BUILD_CHANGELOG.md` — yalnız material implementation/canonical-state değişikliğinde
- `OPERATOR_DECISION_QUEUE.md` — yalnız yeni approval gerektiğinde veya karar çözüldüğünde
- `REPRODUCIBILITY_REPORT.md` — yalnız experiment/artifact yeniden üretildiğinde

Detaylı phase report, reconciliation report ve model handoff package yalnız: phase completion; kritik Opus checkpoint; model switch; blocker; operator approval boundary oluştuğunda üretilir.

Değişmeyen dokümanları yeniden yazma. Değişiklik yoksa `NO_UPDATE_REQUIRED` kaydı yeterlidir.

**Ek kurallar (operatör, 2026-07-03):**
- **Yeni MD dosyası oluşturmak İSTİSNADIR** — Phase 0 artifact seti bir kerelikti; normal batch'lerde yeni dosya yalnız phase completion / model switch / blocker / operator decision gerektirdiğinde açılır.
- **Aynı bilgi birden fazla MD'ye kopyalanmaz** — tek canonical yer + gerekirse diğerlerinden kısa pointer.
- Batch dokümantasyonu "keep it simple but solid": az dosya, tam içerik.

SYSTEM_STATE güncellemesi minimum içeriği: update timestamp UTC, build ID, current model, active phase/wave, last verified checkpoint, completed/partial work, open blockers, running processes/PIDs, DB+ledger timestamps, active forward observers, forward raw N + independent-cycle N, active experiments, scientific verdicts, software test status, data-health, storage, live/shadow diff status, next safe dependency, operator decisions waiting, next required model. Historical/replay/paper/shadow/forward/live AYRI bölümlerde. Geçmiş kararlar sessizce yeniden yazılmaz; material değişiklik = yeni version + supersession + Decision Record + previous verdict + new evidence + permission effect.

## 15. Implementation batch disiplini

Her batch sözleşmesi: `batch_id, phase, scope, dependencies, preconditions, changed_files, new_files, protected_untouched_files, schema_effect, runtime_effect, scientific_effect, migration, tests, rollback, documentation_updates, Definition of Done`

Her batch sonunda: (1) ilgili testler, (2) git diff incele, (3) protected live/shadow diff doğrula, (4) DB/schema doğrula, (5) rollback doğrula, (6) SYSTEM_STATE.md, (7) changelog (yalnız material ise, §14), (8) progress ledger, (9) failure/contradiction kayıtları (yalnız ilgiliyse), (10) sonraki dependency.

Kodun varlığı ≠ phase tamamlanması. Phase tamamlanması: implementation + tests + migration verification + reproducibility + rollback + documentation + DoD + (gerekiyorsa) Opus acceptance. Büyük monolitik değişiklik yok; phase/epic bazlı checkpoint'ler.

## 16. Zorunlu çıktılar

`DOCUMENT_RECONCILIATION_MATRIX.md, CANONICAL_PRECEDENCE_AND_CONFLICT_REGISTER.md, LAST_VERIFIED_CHECKPOINT.md, REPOSITORY_RUNTIME_AUDIT.md, CURRENT_STATE_VS_TARGET_GAP_ANALYSIS.md, MASTER_ROADMAP.md, IMPLEMENTATION_DEPENDENCY_GRAPH.md, QUESTION_COVERAGE_MATRIX_Q001_Q1058.csv, QUESTION_FAMILY_TO_ENGINE_MAP.md, WORKSTREAM_AND_EPIC_MAP.md, SCHEMA_AND_DATA_MIGRATION_MAP.md, HISTORICAL_RESEARCH_WAVES.md, FORWARD_OBSERVER_ROADMAP.md, TEST_AND_MUTATION_STRATEGY.md, STORAGE_COMPUTE_CAPACITY_PLAN.md, RISK_REGISTER.md, OPERATOR_DECISION_QUEUE.md, DEFINITION_OF_DONE_BY_PHASE.md, PROTECTED_COMPONENTS_MANIFEST.md, UNTOUCHED_LIVE_SHADOW_COMPONENTS.md, FIRST_SAFE_IMPLEMENTATION_BATCH.md, IMPLEMENTATION_PROGRESS_LEDGER.md, AMI_S34_BUILD_CHANGELOG.md, AMI_S34_RESEARCH_BACKLOG.md, FAILURE_ARCHIVE.md, CONTRADICTION_REGISTER.md, SCHEMA_DICTIONARY.md, MIGRATION_LOG.md, REPRODUCIBILITY_REPORT.md, TEST_STATUS_LATEST.md, SYSTEM_STATE.md`

Aynı bilgi farklı dosyalarda çelişemez. Output'lar mümkünse canonical structured data / canonical SQL'den üretilir.

## 17. Phase-sonu zorunlu rapor alanları

Phase, Status, Completed/Partial components, Changed files, New schemas, Migrations, Software tests, Scientific tests, Supported/Falsified/Rejected findings, Blocked questions, Forward-accumulating questions, Raw event N, Independent-cycle N, Data quality, Storage impact, Live/shadow diff, Rollback status, Open operator decisions, Next dependency, Next required model.

## 18. Son temel kurallar

```
Do not restart from zero.
Do not duplicate validated implementations.
Do not silently resolve contradictions.
Do not confuse row N with independent-cycle N.
Do not confuse replay with forward.
Do not confuse software success with scientific success.
Do not confuse a chart pattern with an edge.
Do not confuse an edge with operational permission.
Do not use future-known information.
Do not let dashboards become truth sources.
Do not change live or shadow behavior without approval.
Do not continue under the wrong model.
```

Model değişim zamanında yalnızca öneride bulunma: state'i kaydet, checkpoint üret, hangi modele geçileceğini açıkça yaz ve DUR.
