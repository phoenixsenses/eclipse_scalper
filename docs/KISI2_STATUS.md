# Kisi 2 (Runtime/Ops) — Yapilacaklar

Tarih: 2026-03-09 | Branch: `codex/runtime/ops-foundation`

## Durum: TUM AKTIF ISLER TAMAMLANDI

3 phase, 26 commit, 63 dosya audit, 143+ test fonksiyonu.
Kalan isler dusuk oncelikli iyilestirmeler.

---

## P2 — Dusuk Oncelik (Opsiyonel)

| # | Gorev | Detay | Durum |
|---|-------|-------|-------|
| 1 | exchanges/binance.py unit testleri | Order submission, symbol resolution, API call edge case'leri. Simdi sadece mock.py test ediliyor | Baslanmadi |
| 2 | bot/core.py unit testleri | `_trade_allowed()` ve state machine logic icin dedicated test. Simdi sadece integration test var | Baslanmadi |
| 3 | notifications/ genisletilmis test | Manager, telegram, events, health_alerts, daily_summary icin unit test. Simdi sadece basic integration test | Baslanmadi |
| 4 | monitoring/prometheus.py ayristir | Prometheus metrikleri su an `dashboard/backend/app.py` icinde inline. Standalone module olarak cikar | Baslanmadi |
| 5 | scripts/deploy_checklist.sh | Pre-deploy dogrulama scripti: preflight + env_sanity check'lerini sar | Baslanmadi |

---

## Tamamlanan Isler (26 Commit)

### Phase 1 — Safety Hardening (18 commit)
- **P0:** Kill switch fail-closed, timeout guard'lar (order_router 10s, reconcile 8s)
- **P0:** Heartbeat, Telegram circuit breaker
- **P1:** Size validation, dashboard rate limit, OOM fix (intent_ledger, dashboard, tailer)
- **P1:** Atomic disk write (5 dosya), startup validation gate (symbols, hedge, API)
- **P1:** Margin/liquidation alert, guardian step timeout (45s), memory cap
- **P1:** HTML injection fix (Telegram), /kill komutu
- **Tech:** `_symkey()` consolidation (15 dosya), dead code silme (-482 satir)
- **Test:** 31 runtime safety test

### Phase 2 — Ops Enhancements (5 commit)
- Alert escalation (tekrarlayan alert -> otomatik seviye artisi)
- Graceful degradation (exchange down -> entry bloke, exit acik)
- Config hot-reload (JSON override, 40+ guvenli alan)
- Structured alert rules engine (6 default kural)
- Ops runbook (incident response playbook)

### Phase 3 — Profiling & Gaps (3 commit)
- Guardian step profiling (avg/max/last ms)
- Default alert rules JSON
- Degraded mode -> entry gate'e baglandi
- Alert rules API endpoint (`/api/alert-rules`)
- Adaptive guard memory cap (500 symbol siniri)
- 3 invariant test dosyasi (28 test): EXE-01, EXE-02, SAF-02
- 3 infra modulu: preflight.py, env_sanity.py, shared_locks.py
- Kisi 1/Kisi 2 status dokumanlari

### PR Gecmisi
- **#27** Merged: Phase 1 safety hardening
- **#28** Merged: Phase 2 ops enhancements
- **#29** Merged: Phase 3 profiling & gaps

### Audit Tablosu
| Dizin | Dosya | Fixed | Clean | Deleted |
|-------|-------|-------|-------|---------|
| execution/ | 37 | 5 | 29 | 2 |
| bot/ | 2 | 1 | 1 | — |
| brain/ | 3 | 0 | 3 | — |
| risk/ | 1 | 0 | 1 | — |
| exchanges/ | 4 | 0 | 4 | — |
| notifications/ | 6 | 2 | 4 | — |
| dashboard/ | 5 | 1 | 4 | — |
| monitoring/ | 1 | 1 | 0 | — |
| **Toplam** | **63** | **10** | **48** | **2** |
