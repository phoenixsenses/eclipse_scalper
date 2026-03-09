# Kisi 1 (Research/Data) — Yapilacaklar

Tarih: 2026-03-09 | Branch: `feat/execution-hardening-and-strategies`

## P0 — Kritik (Deploy Oncesi Yapilmali)

| # | Gorev | Detay | Durum |
|---|-------|-------|-------|
| 1 | 12 execution dosyasini merge et | `feat/execution-hardening-and-strategies` branch'indeki 12 yeni dosya (circuit_breaker, event_journal, order_verifier, vb.) Kisi 2 ile koordinasyon gerekli — 7 dosya guvenli, 5 dosya mimari review lazim | Beklemede |
| 2 | Microstructure sinyalleri implement et | `data/microstructure_signals.py:21` — 2+ hafta canli veri toplandiktan sonra implement edilecek | Bloke (veri yok) |

## P1 — Yuksek Oncelik

| # | Gorev | Detay | Durum |
|---|-------|-------|-------|
| 3 | Canonical integrity gate | `tools/validate_canonical.py` — Mikroyapi verisi icin deterministik schema dogrulama (zorunlu kolonlar, dtype, zaman invariantlari, NaN/Inf eslikleri) | Baslanmadi |
| 4 | Data contract dokumantasyonu | `docs/MICROSTRUCTURE_DATA_CONTRACT.md` — microstructure.db icin tablo/kolon/freshness kurallari | Baslanmadi |
| 5 | Feature'lari dependency level'a gore etiketle | `data/features/micro_features.py` icindeki her feature'i siniflandir: mark_only / trade_flow / trade_plus_liq / requires_book | Kismi |
| 6 | Sinyal katmani iyilestirmeleri | Score calibration, regime-conditioning (sadece past-only), feature sanitation pipeline, no-lookahead unit check'leri | Baslanmadi |
| 7 | Deterministik test fixture | `tests/fixtures/microstructure_sample.db` — Research tool testleri icin sentetik sqlite DB | Baslanmadi |
| 8 | Research fitness validator | `tools/validate_data_research_fitness.py` — Tablo/kolon varlik kontrolu, book proxy uyarisi | Baslanmadi |

## P2 — Orta Oncelik

| # | Gorev | Detay | Durum |
|---|-------|-------|-------|
| 9 | Liq reversal E2E chain | `tools/run_liq_reversal_e2e.py` — high_liq_reversal_regime icin ucretsiz arastirma zinciri. Sifir pocket bulunursa execution stili degistir | Planli |
| 10 | Research event lane'leri tamamla | 7 lane: book_proxy_pressure, fill_toxicity, latency_stress, return_shock, spread_stress, volatility_burst, volume_vacuum | Planli |
| 11 | Event watchboard research entegrasyonu | Research tarafli watchboard, event-driven strateji izleme | Planli |

## Tech — Teknik Borc

| # | Gorev | Detay | Durum |
|---|-------|-------|-------|
| 12 | `canonical_symbol()` vs `symkey()` duzelt | canonical_symbol USDTUSDT'yi dedupe etmiyor, symkey ediyor. 21 dosya etkilenebilir | Bilinen sorun |
| 13 | Phase 0 baseline safety | Tum tool'lar explicit output path'e yazsin, run metadata eklensin, smoke check standardize edilsin | Planli |
| 14 | Invariant test audit | DAT-01 ile VAL-03 arasi test dosyalarinin tam kapsam sagladigini dogrula | Audit lazim |

## Koordinasyon

| # | Gorev | Detay |
|---|-------|-------|
| 15 | Kisi 2 ile merge koordinasyonu | 12 dosya merge'den once mimari karar: health_monitor vs guardian, metrics_collector vs telemetry, state_machine genislemesi |
| 16 | 36 pre-existing test failure | Alpha pipeline/transfer/live model testleri — suan bloke degil ama takip edilmeli |

## Tamamlanan Isler (Referans)

- `eclipse_scalper.py` split: 1521 -> 910 satir + `env_helpers.py` + `indicators.py`
- `_symkey()` 21 dosyada tekrar temizlendi
- `strategies/risk.py` magic number'lar named constant'a dondu
- 88 unit test eklendi (env helpers, indicators, signal pipeline)
- 698 test basarili
