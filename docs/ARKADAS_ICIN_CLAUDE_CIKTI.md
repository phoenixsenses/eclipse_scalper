# ARKADAS İÇİN CLAUDE ÇIKTISI

> Kisi 1 (Research/Data) Claude'u tarafından üretildi.
> Tarih: 2026-03-10 | Hedef: Kisi 2'nin Claude'u okuyup analiz edecek.

---

## 1. GENEL DURUM

- **Branch:** `main`
- **Son commit:** `c07427f Add microstructure research fitness baseline`
- **Toplam .py dosyası (testler hariç, __init__ hariç):** ~150
- **Toplam test fonksiyonu:** 746 (`def test_` grep sonucu)
- **Toplam satır (yaklaşık):** 42,000+

**Senkronizasyon durumu:**
- Local main, origin/main'den **4 commit ileride** (henüz push edilmedi)
- origin/main, local'den **34 commit ileride** (PR #28, #29, #30, #31 dahil — pull lazım)

```
Local ahead:
  c07427f Add microstructure research fitness baseline
  0f0c287 Kisi1 sync Phase 1+2: execution hardening modules + dashboard polish
  b6897ca Merge branch 'main' of ...
  a07f7c6 Build research market chart and pocket overlay

Origin ahead (son 10):
  fc42199 Merge PR #31 docs/arkadas-codex
  f9c704e docs: add ARKADAS_CODEX.md
  c5bf47b Merge PR #30 docs/todo-status-update
  1a28e09 docs: update status files to TODO-focused format
  fec60bd Merge PR #29 codex/runtime/ops-phase3
  7322b81 feat(ops): invariant tests, infra modules, status docs
  ...
```

---

## 2. KİSİ 1 ALANLARI — DETAYLI TARAMA

### strategies/ (3 dosya)

| Dosya | Satır | Ne yapıyor |
|-------|-------|------------|
| eclipse_scalper.py | 1,535 | Ana üretim stratejisi: micro_edge_v3_passive_alpha sinyal tespiti, cosmic signal v3.0, debug/force-entry modları |
| risk.py | 38 | Risk konfigürasyon katmanı (named constants) |
| __init__.py | 0 | Package init |

### data/ (12 dosya, ~3,600 satır)

| Dosya | Satır | Ne yapıyor |
|-------|-------|------------|
| microstructure_collector.py | 1,045 | Binance USDT-M veri toplama pipeline'ı (agg_trades, mark_prices, liquidations → SQLite) |
| cache.py | 715 | Cosmic data truth oracle v4.5: monotonic cache age, stale reporting, market bootstrap |
| event_diary.py | 542 | SQLite tarayıp ilginç anları otomatik günlüğe kaydeder |
| microstructure_analysis.py | 459 | 1-saniyelik bucket'lardan OHLCV, imbalance, spread, intensity hesapları |
| features/micro_features.py | 254 | Mikroyapı sinyalleri için feature engineering |
| features/snapshot.py | 202 | Piyasa durumunun anlık snapshot'ları |
| quality.py | 150 | Veri doğrulama ve kalite kontrolleri |
| features/registry.py | 138 | Feature kayıt defteri ve enumerasyon |
| microstructure_signals.py | 34 | Sinyal üretimi — **PLACEHOLDER** (TODO: 2+ hafta veri gerekli) |
| labels/forward_return.py | 57 | Supervised learning için ileri getiri etiketleme |
| labels/__init__.py | 2 | Package init |
| __init__.py | 0 | Package init |

### tools/ (200+ dosya, en önemli 15'i)

| Dosya | Satır | Ne yapıyor |
|-------|-------|------------|
| report_schema_validator.py | 3,084 | Kapsamlı rapor schema doğrulama ve metrik puanlama |
| build_presentation.py | 1,517 | Rapor sunum üretimi ve formatlama |
| micro_edge_backtest.py | 1,462 | Çekirdek micro-edge paper simülasyonu (sadece research) |
| rank_passive_pockets_forward.py | 1,344 | Pocket sıralama CLI; `--regime none\|up\|down` destekler |
| npa_decomposition_and_regime.py | 936 | NPA ayrıştırma analizi ve rejim stratifikasyonu |
| audit_fill_calibration.py | 909 | Fill kalibrasyon denetimi |
| validate_passive_pocket_forward.py | 885 | Pocket ileri validasyon; `_ROWS_CACHE` ile (db, symbol, lookback, bucket_sec, profile) başına önbellek |
| diagnose_sell_asymmetry.py | 829 | Satış/alış asimetrisi kök neden analizi |
| run_alpha_pipeline.py | 677 | Uçtan uca alpha keşif pipeline'ı |
| micro_edge_lib.py | 656 | Paylaşılan yardımcılar: build_bucket_features, sinyal zenginleştirme, fill simülasyonu |
| validate_micro_edge_forward.py | 519 | Micro-edge ileri doğrulama koşum takımı |
| fit_adverse_model.py | 516 | Olumsuz fill model fitting |
| walkforward_sweep.py | 471 | Walkforward sweep orkestrasyonu |
| walkforward_eval.py | 445 | Walkforward değerlendirme koşum takımı |
| validate_env.py | 420 | Ortam doğrulama aracı |

**Diğer önemli tools:** `check_event_lanes.py`, `watch_regime_recovery.py`, `run_rank_sweep.py`, `daily_research_report.py`, `build_regimes.py`

### tests/ (327 dosya, 746 test fonksiyonu)

**Dizin yapısı:**
- `tests/` — Genel entegrasyon testleri
- `tests/contracts/` — Veri sözleşme testleri
- `tests/execution/` — Execution katmanı testleri
- `tests/fixtures/` — Test fixture'ları
- `tests/fixtures/execution/` — Execution-specific fixture'lar
- `tests/legacy_tools/` — Eski tool testleri (bazıları kırık — IndentationError mevcut)
- `tests/parity/` — Parity/uyumluluk testleri
- `tests/replay/` — Replay testleri
- `tests/runtime/` — Runtime güvenlik testleri

**Durum:** 746 testin büyük çoğunluğu geçiyor. `tests/legacy_tools/test_entry_qty_scale_unit.py` — pre-existing IndentationError (bloke değil). `tests/test_status_snapshot.py::test_collect_pnl_and_render` — pre-existing string mismatch ("Paper Run" vs "Health Check").

---

## 3. KİSİ 2 ALANLARI — ÖZET TARAMA

### execution/ (45 dosya, 19,200+ satır)

| Dosya | Satır |
|-------|-------|
| entry_loop.py | 3,030 |
| order_router.py | 2,912 |
| exit.py | 2,018 |
| entry.py | 2,013 |
| reconcile.py | 2,011 |
| bootstrap.py | 1,432 |
| guardian.py | 1,054 |
| emergency.py | 655 |
| health_monitor.py | 637 |
| position_manager.py | 887 |
| telemetry.py | 515 |
| entry_watch.py | 505 |
| protection_manager.py | 503 |
| state_machine.py | 488 |
| intent_ledger_persistence.py | 473 |
| flatten_intent.py | 422 |
| alpha_gate.py | 429 |
| order_verifier.py | 394 |
| system_status.py | 392 |
| rebuild.py | 542 |
| metrics_collector.py | 400 |
| event_journal.py | 375 |
| reliability_gate_runtime.py | 369 |
| rate_limiter.py | 358 |
| circuit_breaker.py | 317 |
| data_quality.py | 316 |
| adaptive_guard.py | 306 |
| entry_loop_full.py | 755 |
| data_loop.py | 615 |
| belief_controller.py | 592 |
| intent_ledger.py | 389 |
| position_lock.py | 441 |
| health_gate.py | 265 |
| replace_manager.py | 206 |
| belief_evidence.py | 176 |
| shutdown_control.py | 155 |
| runtime_helpers.py | 140 |
| bot_factory.py | 134 |
| diagnostics.py | 112 |
| error_codes.py | 49 |
| guard_knobs.py | 66 |
| telemetry_recovery.py | 69 |
| anomaly_guard.py | 32 |
| __init__.py | 0 |
| sim/ (3 alt dosya) | — |

### bot/ (3 dosya)

| Dosya | Satır |
|-------|-------|
| runner.py | 782 |
| core.py | 747 |
| __init__.py | 0 |

### exchanges/ (6 dosya)

| Dosya | Satır |
|-------|-------|
| binance.py | 474 |
| validator.py | 170 |
| coinbase.py | 143 |
| base.py | 75 |
| mock.py | 73 |
| __init__.py | 31 |

### notifications/ (9 dosya)

| Dosya | Satır |
|-------|-------|
| manager.py | 275 |
| x_twitter.py | 114 |
| daily_summary.py | 87 |
| trade_alerts.py | 87 |
| health_alerts.py | 75 |
| risk_alerts.py | 62 |
| telegram.py | 42 |
| events.py | 30 |
| __init__.py | 0 |

### dashboard/ (6 dosya)

| Dosya | Satır |
|-------|-------|
| backend/data_sources.py | 2,097 |
| backend/app.py | 998 |
| backend/control_actions.py | 876 |
| backend/models.py | 475 |
| backend/tailer.py | 57 |
| backend/__init__.py | 0 |

### monitoring/ (3 dosya)

| Dosya | Satır |
|-------|-------|
| status_snapshot.py | 255 |
| log_rotation.py | 179 |
| __init__.py | 2 |

### risk/ (1 dosya)

| Dosya | Satır |
|-------|-------|
| kill_switch.py | 850 |

---

## 4. PAYLAŞILAN DOSYALAR

### config/ (4 dosya)

| Dosya | Satır | İçerik |
|-------|-------|--------|
| settings.py | 227 | Ana konfigürasyon parametreleri |
| costs.py | 25 | Ücret/maliyet modeli sabitleri |
| symbols.py | 17 | Borsa sembol listesi |
| __init__.py | 0 | Package init |

### utils/ (3 dosya)

| Dosya | Satır | İçerik |
|-------|-------|--------|
| logging.py | 261 | Yapılandırılmış loglama: log_core, log_entry, log_debug |
| symbols.py | 19 | Sembol yardımcıları |
| __init__.py | 0 | Package init |

### core/ (12 dosya)

| Dosya | Satır | İçerik |
|-------|-------|--------|
| micro_features.py | 463 | Mikro-sinyal feature engineering (imbalance, intensity, spread) |
| order_placement.py | 300 | Kayma modellemeli order placement |
| micro_signal.py | 288 | Mikro-sinyal üretimi ve v2/v3 zenginleştirme |
| scratch.py | 129 | Scratch/test sinyal implementasyonu |
| trade_logger.py | 235 | Ticaret yürütme loglama |
| performance_monitor.py | 223 | Performans takibi ve metrikleri |
| regime.py | 193 | Rejim tespiti (UP/DOWN, 1h rolling log-return) |
| regime_risk.py | 168 | Rejim tabanlı risk ayarlaması |
| latency_profiler.py | 63 | Gecikme ölçümü |
| chart_generator.py | 57 | Grafik oluşturma |
| fee_model.py | 51 | Binance ücret hesaplama |
| __init__.py | 2 | Package init |

### brain/ (4 dosya)

| Dosya | Satır | İçerik |
|-------|-------|--------|
| persistence.py | 845 | Beyin durumu kalıcılığı (disk/SQLite) |
| state.py | 602 | Beyin durum konteyneri (pozisyonlar, inançlar, metrikler) |
| performance_memory.py | 207 | Tarihsel performans hafızası ve öğrenme |
| __init__.py | 0 | Package init |

---

## 5. EXECUTION DOSYALARI KARŞILAŞTIRMASI — 12/12 MEVCUT

Tüm 12 dosya `execution/` altında mevcut. Bu dosyalar **bu oturumda** (2026-03-10, commit `0f0c287`) merge edildi.

| Dosya | Satır | Durum | Ana görev |
|-------|-------|-------|-----------|
| circuit_breaker.py | 317 | **MEVCUT** — yeni eklendi | CLOSED→OPEN→HALF_OPEN devre kesici |
| event_journal.py | 375 | **MEVCUT** — yeni eklendi | JSONL denetim izi |
| flatten_intent.py | 422 | **MEVCUT** — yeni eklendi | WAL crash-safe intent düzleştirici |
| intent_ledger_persistence.py | 473 | **MEVCUT** — yeni eklendi | Kalıcı intent depolama |
| order_verifier.py | 394 | **MEVCUT** — yeni eklendi | Post-fill order doğrulama döngüsü |
| position_lock.py | 441 | **MEVCUT** — yeni eklendi | TTL'li sembol başına pozisyon kilidi |
| rate_limiter.py | 358 | **MEVCUT** — yeni eklendi | Token-bucket hız sınırlayıcı |
| health_monitor.py | 637 | **MEVCUT** — yeni eklendi | HealthStatus enum + bileşen kontrolleri |
| metrics_collector.py | 400 | **MEVCUT** — yeni eklendi | Counter/Gauge/Histogram windowed toplayıcı |
| system_status.py | 392 | **MEVCUT** — yeni eklendi | Periyodik durum raporlama döngüsü |
| protection_manager.py | 503 | **MEVCUT** — yeni eklendi | SL/TP kapsam değerlendirmesi |
| state_machine.py | 488 | **MEVCUT** — üst küme ile değiştirildi | PositionState + MachineKind (her ikisi birden) |

**guardian.py değişikliği:** 4 opsiyonel tick hook eklendi — `verification_tick`, `health_check_tick`, `collect_bot_metrics`, `status_tick`. Tümü `callable()` ile korunuyor, asla fatal değil.

---

## 6. TODO / FIXME / HACK / XXX

Tüm projede yalnızca **1 adet** bulundu:

```
data/microstructure_signals.py:21 — TODO: Implement after collecting 2+ weeks of microstructure data.
```

**Değerlendirme:** Kod tabanı temiz. Teknik borç işaretçileri minimumdur.

---

## 7. IMPORT ÇAKIŞMALARI

### health_monitor.py vs guardian.py
- **Çakışma yok.** Ortogonal.
  - `health_monitor.py`: `HealthStatus` enum, `HealthMonitor` class, `get_health()`, `add_health_check()` — bileşen sağlık izleme
  - `guardian.py`: async tick döngüsü, exchange bağlantı probe, margin refresh, emergency handler — orkestrasyon
  - guardian artık `health_check_tick` üzerinden health_monitor'u **isteğe bağlı** çağırıyor

### metrics_collector.py vs telemetry.py
- **Çakışma yok. Tamamlayıcı.**
  - `telemetry.py`: Olay tabanlı JSONL loglama (emit, emit_throttled, emit_fill, emit_order_create) — ne oldu kaydı
  - `metrics_collector.py`: Windowed aggregation (Counter/Gauge/Histogram, export_json, export_prometheus) — ne kadar sıklıkla sayacı
  - Öneri: telemetry olayları → metrics_collector'a beslenebilir (ileride)

### system_status.py vs status_snapshot.py
- **Küçük örtüşme, çakışma değil.**
  - `system_status.py`: Toplayıcı (health_monitor + metrics_collector + state_machine'i toplar), `periodic_status_loop()` async (5dk'da bir), `get_system_status()` döndürür
  - `status_snapshot.py` (monitoring/): Yalnızca sorgu: `_latest_micro_ts()`, `collect_pnl()`, JSON export
  - system_status büyük olasılıkla status_snapshot yardımcılarını tüketiyor

### protection_manager.py vs kill_switch.py
- **Çakışma yok. Net ayrım.**
  - `protection_manager.py`: Pozisyon düzeyinde — tek bir pozisyon için SL/TP kapsamı var mı? (`CoverageStatus.FULL/PARTIAL/NONE`)
  - `kill_switch.py`: Sistem düzeyinde — tüm bot için halt mı? (`is_halted()`, `halt()`, `escalate_flat()`, `escalate_shutdown()`)

### state_machine.py — iki versiyonu birleşti
- Eski local (54 satır): Yalnızca `MachineKind.ORDER_INTENT` + `MachineKind.POSITION_BELIEF` + `is_valid_transition()` + `transition()` — order_router, rebuild, reconcile, replace_manager bunları kullanıyor
- Yeni versiyon (488 satır): Üst küme — eski API'yi **koruyarak** `PositionState` lifecycle, `PositionStateMachine`, `journal_transition()` ekledi
- `event_journal.py` artık `from execution.state_machine import journal_transition` diyebilir ✅
- Mevcut kullananlar (`order_router`, `rebuild`, vb.) kırılmadı ✅

---

## 8. GİT DURUMU

```
Branch: main
Local 4 commit ahead of origin/main (push edilmedi)
Origin 34 commit ahead of local (pull lazım)

Untracked:
  data/event_diary.csv.bak   ← backup dosyası, commit etme
```

**Önerilen eylem:** `git pull --rebase origin main` → conflict çöz → push

---

## 9. ÇIKARIM VE ÖNERİ

### Kisi 2 ile senkron için yapılacaklar

1. **Pull gerekli (öncelikli):** origin/main 34 commit ileride. PR #28 (ops-phase2), #29 (ops-phase3), #30, #31 (arkadas-codex) var. `git pull --rebase origin main`.

2. **12 dosya merge tamamlandı:** Kisi 1 STATUS.md'deki #1 görevi bitti. Kisi 2'ye bildirmeye gerek var — local commit `0f0c287` ile yapıldı.

3. **Çakışma riski yok:** 12 dosyanın tamamı local'de yoktu (state_machine hariç, o da üst küme ile değiştirildi). guardian.py'a eklenenler tamamen opsiyonel.

4. **Mimari kararlar çözümlendi:**
   - health_monitor ≠ guardian → ortogonal ✅
   - metrics_collector ≠ telemetry → tamamlayıcı ✅
   - system_status ≠ status_snapshot → toplayıcı vs sorgu ✅
   - protection_manager ≠ kill_switch → pozisyon vs sistem ✅
   - state_machine → üst küme, geriye dönük uyumlu ✅

5. **Kisi 1'e kalan görevler** (`data/` ve `tools/` bölgesi, execution'a dokunmadan):
   - `data/microstructure_signals.py` — implement (veri yetersiz şu an, bloke)
   - `tools/validate_canonical.py` — integrity gate
   - `docs/MICROSTRUCTURE_DATA_CONTRACT.md`
   - `canonical_symbol()` vs `symkey()` tekrar temizliği

### Merge stratejisi önerisi

```
Kisi 1:
  git pull --rebase origin main
  # Conflict varsa: execution/ dosyaları Kisi 2 lehine bırak
  # data/, tools/, strategies/ → Kisi 1 lehine tut
  git push origin main (veya PR aç)

Kisi 2:
  git pull → local'de 0f0c287 commit'i görecek (12 dosya + dashboard)
  Doğrulama: python -m pytest tests/ -x -q (699 pass beklenir)
```

---

*Rapor guardian-safe prensibiyle üretildi: hiçbir dosya değiştirilmedi, sadece okunup raporlandı.*
