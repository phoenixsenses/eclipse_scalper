# Eclipse Scalper — Master Plan Status (2026-03-10)

> Tüm P0/P1/P2 aktif işler tamamlandı (26 commit, 63 dosya audit, 143+ test).
> Bu plan projeyi bir sonraki seviyeye taşımak için 4 alanda kapsamlı görev listesi içerir.

---

## Kişi 2 (Runtime/Ops + Dashboard)

### Hafta 1

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| A2 | Arkadaşın dashboard kodunu review | P0 | ✅ | data_sources.py (99KB), models.py (17KB), status_snapshot.py — temiz, resilient |
| C4 | entry_watch.py dead code temizliği | P2 | ✅ | `we_dont_have_this()` silindi |
| D1 | Kritik unit testler (~90 hedef) | P0 | ✅ | **178 test** yazıldı (hedefin 2x) — 7 dosya + conftest.py |

**D1 Test Detayları:**
- `test_circuit_breaker_unit.py` — 15 test (state transitions, fail-open, singleton)
- `test_health_gate_unit.py` — 16 test (degradation, escalation, ingestion probe)
- `test_shutdown_control_unit.py` — 19 test (traced events, shutdown request, paper mode)
- `test_event_journal_unit.py` — 29 test (write/read, rotation, hooks, queries)
- `test_kill_switch_unit.py` — 16 test (persist/restore, daily loss, drawdown, API errors)
- `test_emergency_unit.py` — 12 test (flatten, forced escalation, state fallback)
- `test_reconcile_unit.py` — 20 test (symkey, helpers, phantom store, optional imports)
- `tests/runtime/conftest.py` — shared fixtures (make_state, make_bot)

---

### Hafta 2

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| A1 | Docker Deployment Pipeline | P0 | ✅ | Dockerfile (multi-stage), docker-compose.yml (bot+dashboard+prometheus+grafana), docker-entrypoint.sh, .dockerignore, prometheus.yml |
| C2 | diagnostics.py print→logger | P1 | ✅ | v0.4 — tüm print() → log_core.info() |
| D2 | Integration test suite (~40 test) | P1 | ⏳ | Henüz başlanmadı |

---

### Hafta 3

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| C1 | entry_loop.py split (145KB→<80KB) | P1 | ⏳ | Gate check'leri → entry_gates.py, signal eval → ayrı fn |
| C3 | Dashboard app.py router refactor (43KB) | P1 | ⏳ | routes/overview.py, routes/risk.py, routes/debug.py, routes/live.py |

---

### Hafta 4

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| A3 | Per-symbol Prometheus metrics | P1 | ⏳ | per-symbol gauge: pnl, fill_rate, latency |
| D3 | pytest-asyncio geçişi + fixtures | P1 | ⏳ | `_run(coro)` → `@pytest.mark.asyncio` |
| D4 | Dashboard API test genişletme | P2 | ⏳ | /api/risk-overview, /api/alert-rules, /api/live/metrics |

---

### Hafta 5

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| A4 | Mobile app backend (JWT, push, WS) | P1 | ⏳ | notifications/push.py, auth/jwt.py |
| B1 | DB backup & recovery | P1 | ✅ | scripts/backup.sh + scripts/restore.sh |
| B4 | Slack/Discord webhook | P2 | ✅ | notifications/webhook.py (circuit breaker'lı) |
| D5 | Monitoring testleri | P2 | ⏳ | prometheus.py, log_rotation.py |

---

### Kişi 2 Toplam İlerleme: **8/15 görev tamamlandı**

---

## Kişi 1 (Research/Data)

### Hafta 1

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| 1 | Microstructure signal'leri merge (12 dosya) | P0 | ⏳ | data/ + features/ içindeki yeni signal dosyaları |
| 2 | strategies/ yeni signal pipeline | P0 | ⏳ | scalper_signal + micro_signal entegrasyonu |

---

### Hafta 2

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| 3 | tools/ validation test'leri | P1 | ⏳ | Validator ve analiz araçları için test |
| 4 | data/ collection health iyileştirme | P1 | ⏳ | OHLCV, ticker, funding toplama güvenilirliği |

---

### Hafta 3-4

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| 5 | Mobile app frontend (React Native Phase 1-2) | P1 | ⏳ | docs/MOBILE_APP_PLAN.md'ye göre |

---

### Hafta 5+

| # | Görev | Öncelik | Durum | Detay |
|---|-------|---------|-------|-------|
| 6 | C5 — Encoding audit (kendi dosyaları) | P2 | ⏳ | BOM/smart quote taraması tüm .py'lerde |
| 7 | B2 — Distributed tracing (research tarafı) | P2 | ⏳ | OpenTelemetry spans research |
| 8 | B3 — A/B test framework (strateji tarafı) | P2 | ⏳ | config/experiments.py, symbol-level override |

---

### Kişi 1 Toplam İlerleme: **0/8 görev tamamlandı**

---

## Ortak Görevler

| # | Görev | Öncelik | Durum | Notlar |
|---|-------|---------|-------|--------|
| 1 | docs/ senkronizasyonu | P1 | ⏳ | Her merge sonrası |
| 2 | config/settings.py güncellemeleri | P1 | ⏳ | Yeni parametreler eklenince |
| 3 | CI workflow bakımı | P2 | ⏳ | Yeni testler eklendikçe |

---

## Verification Checklist

Her adım sonrası:
1. `pytest tests/ -x` — tüm testler geçmeli
2. CI pipeline (4 workflow) yeşil olmalı
3. `python -c "import execution; import bot; import risk; import exchanges"` — import check
4. Dashboard: `uvicorn dashboard.backend.app:app` → `/api/health` → 200 OK
5. Docker: `docker build . && docker run --rm eclipse-scalper python -c "print('ok')"`
