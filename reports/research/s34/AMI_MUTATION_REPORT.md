# AMI Mutation / Adversarial Test Report

> 2026-07-02 20:11 UTC — **20/20 ihlal yakalandi**

| # | Injected violation | Expected rejection | Actual | Blocking component | Pass |
|---|---|---|---|---|---|
| 01_future_lookahead | freeze ONCESI acilmis trade forward evidence olarak sunuldu | PRE_FREEZE reddi, evidence yazilmaz | accepted=0 rejected=1 reason=PRE_FREEZE | ForwardEvidencePipeline.run_once (R1) | ✅ |
| 02_train_test_leakage | train/test kesisen event (id=3) | ConstitutionViolation | raised: Train/test leakage: 1 duplicate events (e.g. [3]) | registry.assert_no_overlap | ✅ |
| 03_stale_sensor_healthy | 9999dk yasli feed HEALTHY diye sunuldu | STALE etiketi + circuit breaker | STALE, breaker=['data_health'] | EpistemicGovernor.check_data_health | ✅ |
| 04_research_only_live_auth | HOLDOUT_VALIDATED bilgiyle OPEN_LONG istendi | GRANTED degil | SHADOW_ONLY | EpistemicGovernor.authorize (PERMISSION_MIN_STATUS) | ✅ |
| 05_candidate_version_change | bind sonrasi candidate versiyonu degisti, eski binding kullanildi | BINDING_INVALID, eski forward evidence yeni versiyona tasinamaz | valid=False why=candidate_version_changed(1->2) | ForwardEvidencePipeline._validate_binding (R2) | ✅ |
| 06_dataset_hash_mismatch | KO dataset hash'i binding sonrasi degistirildi | BINDING_INVALID(dataset_hash_changed) | valid=False why=dataset_hash_changed | ForwardEvidencePipeline._validate_binding | ✅ |
| 07_execution_model_mismatch | spec=ask_fill_fee10bps, KO=mark_fill_fee5bps | ConstitutionViolation | raised: Execution model mismatch: KO=mark_fill_fee5bps spec=ask_fill_fee10bps | ForwardEvidencePipeline.bind | ✅ |
| 08_prereg_metric_changed | decision_criteria post-hoc degistirildi | ConstitutionViolation (hash mismatch) | raised: Preregistration violated: spec changed after freeze (§74). | ResearchRegistry.attach_evidence (§74) | ✅ |
| 09_duplicate_trade_evidence | ayni trade ikinci kez evidence olarak islendi | duplicate sayilir, evidence 1 kalir | run1_acc=1 run2_dup=1 evidence=1 | processed_trades PK + evidence PK (R3) | ✅ |
| 10_assumption_permission_open | assumption gecersizlesti ama LIVE izni acik birakilmak istendi | izin otomatik sokulur, authorize GRANTED olmaz | authorize=SHADOW_ONLY status=WEAKENED live_forbidden=True | governor.invalidate_assumption + demote | ✅ |
| 11_contradiction_confidence | celiskiye ragmen LIVE izni surdurulmek istendi | celiski LIVE/SIZING iznini otomatik dusurur | pre=True post=False why=unresolved_contradiction | KnowledgeStore.link + KnowledgeObject.is_permitted | ✅ |
| 12_restart_duplicate_processing | proses restart sonrasi ayni ledger yeniden islendi | kalici processed_trades sayesinde 0 yeni evidence | restart_accepted=0 evidence=1 | processed_trades (persistent PK) | ✅ |
| 13_concurrent_sqlite | iki baglanti ayni store'a esz. yazdi | 20 kayit + integrity ok | n=20 integrity=ok | KnowledgeStore (WAL + busy_timeout) | ✅ |
| 14_decision_replay | ayni bundle+context iki kez karar | deterministik ayni cikti | a1=OPEN_LONG/SHADOW_ONLY a2=OPEN_LONG/SHADOW_ONLY eq=True | decide() (durum disi rastgelelik yok) | ✅ |
| 15_failed_experiment_archive | falsifiye deney arsivlenmeden birakilmak istendi | otomatik DEMOTE + failure archive kaydi | archived=True status=WEAKENED | pipeline._governor_review | ✅ |
| 16_missing_provenance | provenance.code_ref bos KO olusturuldu | ConstitutionViolation | raised: No claim without provenance. | KnowledgeObject.validate | ✅ |
| 17_permission_escalation | PRELIMINARY KO'ya elle LIVE_ALLOWED izni eklendi | statu min. kosulu izni gecersiz kilar | is_permitted=False why=status_too_low:PRELIMINARY<OPERATIONAL_CANDIDATE authorize=SHADOW_ONLY | PERMISSION_MIN_STATUS + governor | ✅ |
| 18_exploration_in_holdout | exploration donemi trade'i holdout/forward diye sunuldu | sinir ihlali reddi + audit kaydi | audit_rejected=True evidence=0 | pipeline freeze-boundary (R1) + audit | ✅ |
| 19_top_winner_report_missing | top-3-removed olmadan promotion istendi | promotion yapilmaz | action=None status=FORWARD_VALIDATING | pipeline promotion preconditions | ✅ |
| 20_dq_propagation | bayat mark feed'iyle state uretildi | tum structure state'ler non-HEALTHY + guven dusurulmus | non_healthy=True low_conf=True | StateEngine.build_bundle (dq propagation) | ✅ |

Audit kanallari: `data/ami/knowledge.sqlite:audit_log` (PUT/LINK/AUTHORIZE/BREAKER_TRIP/BINDING_INVALID/EVIDENCE_REJECTED/ARCHIVE_FAILURE), `research.sqlite:processed_trades(reject_reason)`, `decisions.jsonl`.

*Kaynak: `ami/mutation_suite.py` — pytest esdegeri: `tests/test_ami_mutation_suite.py`*