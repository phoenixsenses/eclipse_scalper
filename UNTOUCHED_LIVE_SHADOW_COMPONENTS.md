# UNTOUCHED_LIVE_SHADOW_COMPONENTS

**Tarih:** 2026-07-03 · Phase 0 taban çizgisi. Her batch sonunda bu liste diff'lenir; boş diff = geçer.

## Live (davranış + dosya değişmez)

- `tools/s34_state_machine_live_executor.py` — git diff'te görünmemeli
- `.env` — yazılmaz
- `execution/`, `risk/`, `brain/` — hiçbir dosya değişmez
- `runtime/s34_v_engine_live_state.json` — yalnız executor yazar
- Leverage/sizing/order/stop-TP sabitleri — değişmez

## Shadow (policy davranışı değişmez)

- `tools/s34_realtime_shadow_runner.py` — Phase 0 başlangıcında zaten modified (önceki oturumlar, observation-only); yeni değişiklik eklenmez
- `tools/s34_shadow_paper_runner.py`, `tools/s34_v_engine_v02_shadow_mirror.py`
- `reports/shadow/*.jsonl` ledger'ları — yalnız runner'lar yazar; araştırma READ-ONLY tüketir

## Collector'lar (durdurulmaz, config değişmez)

- `scripts/collector_supervisor.py`, `data/microstructure_collector` (module), `data/bookticker_collector.py`, `data/oi_spot_poller.py`, `data/event_diary` (module), `tools/heartbeat_watchdog.py`

## Batch-sonu doğrulama komutu

```
git diff --name-only -- tools/s34_state_machine_live_executor.py .env execution/ risk/ brain/ tools/s34_realtime_shadow_runner.py tools/s34_shadow_paper_runner.py tools/s34_v_engine_v02_shadow_mirror.py
```
Beklenen çıktı: boş (shadow_runner'ın Phase-0-öncesi mevcut diff'i hariç — o taban çizgisidir).
