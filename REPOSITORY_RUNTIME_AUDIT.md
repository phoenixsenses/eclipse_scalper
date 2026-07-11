# REPOSITORY_RUNTIME_AUDIT

**Tarih:** 2026-07-03 · **Model:** Fable 5 · **Kapsam:** read-only Phase 0 audit

## 1. Proses envanteri

12 canlı Python prosesi (detay: `LAST_VERIFIED_CHECKPOINT.md`). Sınıflandırma:
- **Collector:** microstructure_collector, bookticker_collector, oi_spot_poller, event_diary, collector_supervisor, heartbeat_watchdog
- **Shadow/paper (observation-only):** s34_shadow_paper_runner, s34_v_engine_v02_shadow_mirror, s34_realtime_shadow_runner
- **Dashboard/replay (read-only):** s34_live_chart (:5050), orderflow_chart (:5051), s34_replay (:5052)
- **Live executor: YOK** (default kapalı)

## 2. Veri deposu envanteri

| Store | Boyut | Son yazma | Not |
|---|---|---|---|
| data/microstructure.db | 684.7 GB | 2026-07-03 20:23 | Ana DB, SALT-OKUNUR açılır |
| data/s34_intelligence.db | 23.7 MB | 2026-07-03 18:57 | Aktif |
| data/funding_history.db | 45.8 MB | 2026-05-12 | Bayat (funding collector durmuş?) — Phase 2 veri-kapsama denetimi |
| data/oi_history.db | 8.7 MB | 2026-05-14 | Bayat; OI artık oi_spot_poller ile ayrı akışta — doğrulanacak |
| data/s34_feature_factory.db | 1.4 MB | 2026-06-26 | |
| data/paper_trades.db, risk_state.db | küçük | eski | |
| data/ami/knowledge.sqlite | 0.1 MB | 2026-07-03 | 11 KO, 21 failure, 40 audit |
| data/ami/research.sqlite | 0.1 MB | 2026-07-03 | 14 soru (slug-ID), 10 deney, 8 evidence, 2 forward binding |
| reports/research/s34/mechanism_store.sqlite | 1 MB | 2026-07-02 | |
| reports/research/s34/S34_ALL.db | 12.6 MB | 2026-07-02 | Meta-DB (geçmiş test sonuçları) |

**Çöp:** `data/test_s34_*` yüzlerce 0-boyutlu test DB (Nisan 2026) — silme operatör onayı ister (OPERATOR_DECISION_QUEUE OD-007).

## 3. Kod envanteri (AMI-ilgili)

- `ami/` 32 modül: constitution, enums, mutation_suite, forward_pipeline, governor, knowledge (objects/store), latent (dataset/discovery/drift_monitor/models/regime/risk_applicability), lifecycle, research (registry/marketplace/forward_pipeline), states (engine/objects/structure), decision/trace, seed_s34, koşum girişleri.
- Testler: 7 AMI test dosyası, toplam 119 test (bkz. TEST_STATUS_LATEST.md).
- Research scriptleri: `tools/research_s34_*.py`, `tools/s34_mechanism_*.py`; 364 rapor md `reports/research/s34/`.
- Dashboard: `tools/s34_live_chart.py`, `tools/s34_cascade_navigation_dashboard.py` (read-only, JSON+md output).

## 4. Kritik bulgular

1. **Numeric question registry YOK** — Reconstruction Protocol §16 "explicit Q396–Q730 registry (335 soru verbatim)" der; repo genelinde grep sonucu: numeric Q-ID'ler yalnız iki canonical belgede aralık olarak geçiyor. `research.sqlite`'ta 14 slug-ID soru var. → CONFLICT-001.
2. **Whitepaper v0.2 VE v0.3_COMPLETE birlikte ve untracked** — CLAUDE.md canonical=v0.2 der (PATCH zinciri 0.2.7 v0.2'de); master protokol v0.3_COMPLETE'i inceleme kaynağı sayar. Appendix H anlam çakışması (v0.2: PATCH registry; v0.3: DB extensions). → CONFLICT-002.
3. **Canonical warehouse yok** — Reconstruction Protocol §7'nin tablolarının (artifact_registry, experiment_registry, ami_events, ami_cycles, …) hiçbiri mevcut değil. Event/cycle identity implementasyonu yok (6h-gap proxy yalnız research scriptlerinde geçici).
4. **Contamination/exposure ledger yok** (whitepaper §70 zorunlu kılar; bugüne kadar disiplin manuel prereg-freeze ile sağlanmış).
5. **Tüm canonical belgeler + AMI kodu untracked** — commit yok; checkpoint güvenilirliği git'e değil dosya sistemine dayanıyor (R-16).
6. `.pytest_cache/` ve `tmp_pytest_*` dizinlerinde izin hatası (git bile okuyamıyor) — pytest `--basetemp` scratchpad kuralının gerekçesi teyit edildi.
7. funding/oi history DB'leri Mayıs ortasından beri bayat — Phase 2 veri-kapsama denetiminde ele alınacak (yeni collector aktivasyonu operatör onayı ister).
8. Chart-native altyapı (candle/swing/pattern engine) repo'da YOK — Phase 4 tamamen MISSING.
9. Forward observatory tabloları/servisleri YOK; ancak `ami/research/forward_pipeline.py` dar-kapsamlı forward evidence hattı VAR ve VALID (2 binding) — observatory inşasında bu hat korunacak, ikinci paralel truth yaratılmayacak.

## 5. Depolama / kapasite

D: 881 GB dolu, 982 GB boş. Dominant: microstructure.db (684 GB, büyüme sürüyor). Snapshot/path ledger eklemeleri öncesi STORAGE_COMPUTE_CAPACITY_PLAN.md sınırları geçerli.
