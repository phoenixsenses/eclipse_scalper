# Eclipse Scalper — Codex Skill (Kişi 1: Research & Data)

> Bu dosya Kişi 1'in (Research/Data) Codex'i için yazılmıştır.
> Repo: `phoenixsenses/eclipse_scalper` | Python 3.13 | Binance USDT-M Futures

---

## 0. Sen Kimsin

Sen **Kişi 1'in Codex'isin** — bu repo'nun **araştırma ve veri** katmanının sahibisin.

| Alan | Sahip |
|------|-------|
| `data/` — veri toplama, cache, feature engineering | **Kişi 1 (sen)** |
| `tools/` — research pipeline, backtest, validation | **Kişi 1 (sen)** |
| `strategies/` — sinyal mantığı, eclipse_scalper.py | **Kişi 1 (sen)** |
| `core/` — micro_signal, regime, features | **Kişi 1 (sen)** |
| `reports/` — araştırma çıktıları | **Kişi 1 (sen)** |
| `execution/`, `bot/`, `exchanges/`, `risk/`, `brain/` | Kişi 2 — **dokunma** |
| `config/`, `utils/`, `tests/`, `dashboard/` | Ortak |

**Altın kural:** `execution/`, `bot/`, `exchanges/`, `risk/`, `brain/` klasörlerine **dokunma**. Değişiklik gerekiyorsa Kişi 2'ye bildir.

---

## 1. Proje Bağlamı

**Ne yapıyor:** Binance USDT-M futures'ta microstructure sinyallerine dayalı algo trading botu.

**Birincil sembol:** ETHUSDT (BTCUSDT de izlenir)

**Veritabanı:** `data/microstructure.db` (SQLite)
- Tablolar: `agg_trades` (ts_ms, symbol, price, quantity, is_buyer_maker), `mark_prices` (ts_ms, symbol, mark_price), `liquidations`
- `is_buyer_maker=0` → taker BUY; `is_buyer_maker=1` → taker SELL
- Veri aralığı: ~20+ gün, ~1 saniye çözünürlük

**Temel sinyal:** `micro_edge_v3_passive_alpha`
- Koşullar: `abs(imbalance) >= 0.5`, `trade_intensity >= 3500`, `spread <= 0.0003`
- SELL: imbalance <= -0.5 (SHORT), BUY: imbalance >= +0.5 (LONG)
- Ufuk (horizon): 120 saniye

**Test:** `python -m pytest` — 746 test fonksiyonu
**Modül çalıştırma:** `python -m tools.rank_passive_pockets_forward` (her zaman `-m` ile)

---

## 2. Research Pipeline (Sahip Olduğun Alan)

```
data/microstructure.db
  └─ tools/micro_edge_lib.py          ← build_bucket_features, sinyal zenginleştirme
      └─ tools/micro_edge_signal_v2.py ← enrich_rows_with_v2 (v2/v3 sinyaller)
          └─ tools/micro_edge_backtest.py ← simulate_rule_trades (çekirdek simülasyon)
              └─ execution/passive_execution_simulator.py ← fill simulation (Kişi 2 dosyası — oku, değiştirme)
                  └─ tools/validate_passive_pocket_forward.py ← forward validation, _ROWS_CACHE
                      └─ tools/rank_passive_pockets_forward.py ← CLI ranker, --regime flag
                          └─ reports/*.md, reports/*.json
```

### Kritik Önbellekler
- `_ROWS_CACHE` in `validate_passive_pocket_forward.py` — (db, symbol, lookback, bucket_sec, profile) başına satır önbelleği
- Mark price array: dense numpy, `(ts_sec - min_sec)` index ile O(1) lookup
- SQL: trade aggregate ~19s, mark prices ayrı yükle ~1.2s

---

## 3. Temel Kavramlar

### 1-Saniye Bucket Feature'ları
```python
imbalance = (buy_qty - sell_qty) / (buy_qty + sell_qty)
trade_intensity = count * 60   # trades/min normalize
spread = abs(VWAP - mark_price) / mark_price
```

### Passive Fill Mekaniği
```
SHORT: limit = ep * (1 + 0.5*spread)  → fiyat YUKARI gidince dolacak
LONG:  limit = ep * (1 - 0.5*spread)  → fiyat AŞAĞI gidince dolacak
depth = (px - limit) / (ep*spread)     # SHORT için
full_proxy = touched AND depth >= 0.5
adverse_bps (SHORT) = max(0, (p_touch - p_next) / p_touch * 10000)
```

### Top Pocket (Referans)
- `abs(imbalance) >= 0.5`, `trade_intensity >= 3500`, `spread <= 0.0003`, ETHUSDT, h=120s
- SELL hit rate: 55.5%, BUY hit rate: 56.4%
- NPA gap SELL/BUY = 9:1 (fill rate veya path excursion kaynaklı değil)
- SELL_UP ve BUY_UP regimes → GO (break-even fee ~0.8 bps/leg)

### Regime Analizi (Track A — uygulandı)
```bash
python -m tools.rank_passive_pockets_forward --regime up    # sadece UP
python -m tools.rank_passive_pockets_forward --regime down  # sadece DOWN
python -m tools.rank_passive_pockets_forward --regime none  # tümü (default)
```
`_add_regime_labels(rows)` → 1h rolling log-return'den 'UP'/'DOWN'/'' atar

---

## 4. Dosya Haritası — Kişi 1 Alanı

### data/
| Dosya | Ne yapıyor |
|-------|-----------|
| `microstructure_collector.py` | Binance WebSocket → SQLite pipeline |
| `cache.py` | Cosmic data truth oracle v4.5, monotonic cache age |
| `microstructure_analysis.py` | 1-sn bucket OHLCV, imbalance, spread hesapları |
| `features/micro_features.py` | Feature engineering |
| `features/snapshot.py` | Anlık piyasa snapshot'ları |
| `quality.py` | Veri doğrulama ve kalite kontrolleri |
| `event_diary.py` | SQLite'ı tarayıp ilginç anları kaydeder |
| `microstructure_signals.py` | **PLACEHOLDER** — 2+ hafta veri gerekli |

### tools/ (en kritik)
| Dosya | Ne yapıyor |
|-------|-----------|
| `micro_edge_lib.py` | `build_bucket_features`, sinyal zenginleştirme |
| `micro_edge_signal_v2.py` | `enrich_rows_with_v2` — v2/v3 sinyal ekler |
| `micro_edge_backtest.py` | `simulate_rule_trades` — çekirdek simülasyon döngüsü |
| `validate_passive_pocket_forward.py` | Forward validation, _ROWS_CACHE |
| `rank_passive_pockets_forward.py` | CLI ranker, `--regime`, `--candidates-md` |
| `diagnose_sell_asymmetry.py` | SELL/BUY asimetri root cause analizi |
| `npa_decomposition_and_regime.py` | NPA ayrıştırma + rejim stratifikasyonu |
| `audit_fill_calibration.py` | Fill kalibrasyon denetimi |
| `run_alpha_pipeline.py` | Uçtan uca alpha discovery |
| `validate_env.py` | Ortam doğrulama |
| `check_event_lanes.py` | Event lane gate kontrolü |
| `watch_regime_recovery.py` | Watchdog — gate durumunu izler |
| `daily_research_report.py` | Günlük araştırma raporu üretimi |

### strategies/
| Dosya | Ne yapıyor |
|-------|-----------|
| `eclipse_scalper.py` | Ana strateji: micro_edge_v3_passive_alpha tespiti, cosmic signal v3.0 |
| `risk.py` | Risk konfig sabitleri |

### core/
| Dosya | Ne yapıyor |
|-------|-----------|
| `micro_signal.py` | Mikro-sinyal üretimi ve v2/v3 zenginleştirme |
| `micro_features.py` | İmbalance, intensity, spread feature engineering |
| `regime.py` | UP/DOWN tespiti (1h rolling log-return) |
| `trade_logger.py` | Trade execution loglama |

---

## 5. Determinizm Zorunluluğu (EN KRİTİK KURAL)

**Aynı girdi + aynı seed = her zaman aynı çıktı.**

Bu kural araştırma sonuçlarının güvenilirliğinin temelidir.

### Yapman Gerekenler
```python
# DOĞRU: seed explicit geçirilmeli
import random, numpy as np
random.seed(seed)
np.random.seed(seed)
```

### Asla Yapma
```python
import random, uuid, time, datetime
random.random()        # ← seetsiz — YASAK
uuid.uuid4()           # ← non-deterministic — YASAK (event_id için)
time.time()            # ← wall-clock — YASAK (araştırma logic'inde)
datetime.now()         # ← wall-clock — YASAK (araştırma logic'inde)
np.random.rand()       # ← seetsiz — YASAK
df.rolling(N, center=True)  # ← lookahead bias — YASAK
```

### Doğru Rolling Kullanımı
```python
# DOĞRU: lag-1 ile sinyal t anında sadece t-1'e kadar veri kullanır
df['feature'] = df['col'].rolling(N, min_periods=1).mean().shift(1)

# YANLIŞ: center=True gelecek veri kullanır
df['feature'] = df['col'].rolling(N, center=True).mean()
```

### Rapor Çıktılarında Seed Echo
Her rapor başlığında seed ve config yansıtılmalı:
```markdown
## Config
seed=42 | lookback=20160 | splits=3 | horizon=120 | fee=0.5
```

---

## 6. Lookahead Bias Önleme (DAT-01)

Sinyal t anında sadece ≤ t verisini kullanabilir.

**Yasak pattern'lar:**
- `df.rolling(N, center=True)` — centered window
- `.shift(-k)` — gelecek veriyi şimdiye kaydırır
- Forward-fill sonrası feature hesaplama
- t anında t+horizon fiyatını label olarak kullanıp sonra feature olarak okumak

**Kontrol:** Her yeni feature için `signal_idx < entry_idx < exit_idx` zincirini doğrula.

---

## 7. Maliyet Birimi Kuralları (DAT-04)

**Karışıklık yaratma:**
- CLI parametreler: **bps** (örn. `--maker-fee-bps-grid 0.5,1.0`)
- İç hesaplamalar: **ratio** (bps × 0.0001)
- Dönüşümü **iki kez uygulama** — 10x hata oluşur

```python
fee_bps = 0.5
fee_ratio = fee_bps * 0.0001   # 0.00005 — sadece bir kez dönüştür
```

**Test:** `tests/test_exec_cost_models.py`, `tests/test_micro_edge_backtest_metrics.py`

---

## 8. Araştırma Tool'larını Doğru Çalıştırma

```bash
# DOĞRU: -m ile (sys.path doğru ayarlanır)
python -m tools.rank_passive_pockets_forward --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md

# YANLIŞ:
python tools/rank_passive_pockets_forward.py  # sys.path sorunu

# Candidates dosyası 'pass' kolonuna sahip olmalı
# FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md → 7 YES satırı var, kullan
# PASSIVE_POCKET_RANKING_*.md → 'pass' kolonu YOK, input olarak kullanma
```

### Regime Sweep Örneği
```bash
python -m tools.rank_passive_pockets_forward \
  --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md \
  --regime up \
  --lookback-min 20160 \
  --splits 3
```

---

## 9. JSONL Debug Schema — Additive Only

Research tool'ların emit ettiği JSONL'ler downstream analyzer'lar tarafından okunur.

**Kural:** Field ekleyebilirsin, **rename/remove yasak**.

**Stable core fields:**
```
symbol, rule_name, seed, split, intent_id, event_id,
signal_idx, entry_idx, exit_idx, side, pnl
```

**event_id üretimi — deterministik olmalı:**
```python
import hashlib
event_id = hashlib.md5(f"{symbol}:{ts_ms}:{row_idx}".encode()).hexdigest()[:16]
# uuid4() kullanma — non-deterministic
```

---

## 10. Test Kuralları

```bash
# Tüm testler
python -m pytest -q

# Sadece araştırma testleri
python -m pytest tests/ -k "not chaos and not execution" -q

# Specific test
python -m pytest tests/test_micro_edge_backtest_metrics.py -q
```

**Pre-existing hatalar (senin sorumluluğun değil):**
- `tests/legacy_tools/test_entry_qty_scale_unit.py` — IndentationError (eski kod)
- `tests/test_status_snapshot.py::test_collect_pnl_and_render` — string mismatch

**Her patch için:**
1. `python -m py_compile <değiştirilen_dosya.py>`
2. `python -m pytest tests/ -q` (tümü geçmeli)
3. Eğer yeni araştırma tool'u ise: `python -m tools.<tool_adi> --help` smoke test

---

## 11. Kişi 2 ile Arayüz Noktaları

Bu dosyalar her iki kişiyi bağlar — değiştirirsen Kişi 2'ye bildir:

| Dosya | Neden Kritik |
|-------|-------------|
| `core/micro_signal.py` | Execution'ın sinyal kaynağı |
| `execution/alpha_gate.py` | Research → execution köprüsü |
| `execution/passive_execution_simulator.py` | Fill sim — Kişi 2'ye ait, sadece oku |
| `config/settings.py` | Her ikisi de okur |
| `utils/logging.py` | Ortak loglama |

**Canonical symbol:**
```python
from execution.entry_primitives import symkey
# data/ ve tools/ da bu fonksiyonu kullanır — tutarlılık şart
```

---

## 12. Raporlama Standardı

Her araştırma raporu şunları içermeli:

```markdown
## Config
seed=42 | symbol=ETHUSDT | lookback_min=20160 | splits=3
horizon=120 | fee_bps=0.5 | regime=up

## Summary
- Toplam işlenen satır: N
- Atlanan satır: M (sebep: ...)
- Pass: X / Total: Y
- NPA (fee=0): ...
- NPA (fee=0.5): ...

## Verdict
GO / NO-GO / MARGINAL
```

---

## 13. Güncel Araştırma Durumu (2026-03-10)

### Tamamlanan (GO)
- SELL_UP h=120, fee ≤ 0.5 bps → GO
- BUY_UP h=120, fee ≤ 0.5 bps → GO
- Break-even fee: ~0.8 bps/leg

### NO-GO
- SELL_DOWN h=120 → tüm NPA negatif
- BUY_DOWN h=120, fee=0.5 → pass=%22, yetersiz
- h=240/300 → veri yetersiz (splits=2 veya 30+ gün gerekli)

### Açık Görevler
- `data/microstructure_signals.py` implement (bloke: veri yetersiz)
- `tools/validate_canonical.py` integrity gate
- `docs/MICROSTRUCTURE_DATA_CONTRACT.md`
- `canonical_symbol()` vs `symkey()` tutarlılık temizliği

---

*Bu skill dosyası Kişi 1'in Claude'u tarafından üretilmiştir.*
*Tam proje doktirini için: `docs/CLAUDE.md`*
