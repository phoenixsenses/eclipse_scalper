# Eclipse Scalper — Kapsamlı Teknik Referans Dokümani (Türkçe)

> **Hedef Kitle:** Projeyi ilk kez inceleyen, yazılım geliştirme ve sistem mimarisi bilen geliştiriciler.
> **Kapsam:** Sistemin amacı, mimarisi, her bileşenin teknik detayı, Codex kullanımı ve geliştirici workflow'u.
> **Son güncelleme:** 2026-02-20

---

## İçindekiler

1. [Eclipse Scalper Nedir](#1-eclipse-scalper-nedir)
2. [Sistem Mimarisi Genel Bakış](#2-sistem-mimarisi-genel-bakış)
3. [Execution Pipeline Detaylı Açıklama](#3-execution-pipeline-detaylı-açıklama)
4. [Signal Generation Sistemi](#4-signal-generation-sistemi)
5. [Risk Management Sistemi](#5-risk-management-sistemi)
6. [Brain ve State Sistemi](#6-brain-ve-state-sistemi)
7. [Order Execution Sistemi](#7-order-execution-sistemi)
8. [Codex Nedir ve Bu Projeye Nasıl Yardımcı Olur](#8-codex-nedir-ve-bu-projeye-nasıl-yardımcı-olur)
9. [Codex CLI Kullanım Rehberi](#9-codex-cli-kullanım-rehberi)
10. [Geliştirici Workflow](#10-geliştirici-workflow)
11. [Gerçek Execution Flow — Derin Teknik Analiz](#11-gerçek-execution-flow--derin-teknik-analiz)
12. [Gelecek Geliştirme Alanları](#12-gelecek-geliştirme-alanları)

---

## 1. Eclipse Scalper Nedir

### 1.1 Projenin Genel Amacı

Eclipse Scalper, Binance USD-margined futures piyasalarında otomatik kripto para alım-satımı yapan, **tamamen asenkron Python** ile yazılmış gelişmiş bir trading botudur. Amaç; kısa vadeli fiyat hareketlerinden (scalp) tutarlı, küçük ama tekrarlanabilir kar elde etmektir.

Sistem şu soruyu yanıtlamak için inşa edilmiştir:

> *"Gerçek piyasa koşullarında, execution kalitesi bozulmadan ve risk kontrol mekanizmaları asla devre dışı bırakılmadan sürdürülebilir bir scalping edge var mı?"*

### 1.2 Hangi Tür Bir Trading Bot?

| Özellik | Detay |
|---|---|
| **Piyasa** | Binance USDM Futures (hedge mode) |
| **Enstrüman** | BTC/USDT, ETH/USDT ve diğer likit çiftler |
| **Strateji tipi** | Multi-timeframe teknik analiz scalping |
| **Zaman dilimi** | 1dk (primary), 5dk + 15dk (confirmation) |
| **Kaldıraç** | 20x (production) / 35x (micro hesap) |
| **Execution** | ccxt async — CCXT kütüphanesi üzerinden |
| **Mod** | Paper trading (varsayılan) veya live trading |

### 1.3 Çözdüğü Teknik Problemler

#### Problem 1 — Durum Tutarsızlığı (State Inconsistency)

Bir trading botunun en kritik sorunu: botun kendi kafasındaki "pozisyon var" ile gerçek exchange'deki durum arasındaki fark. Eclipse Scalper bunu **reconciliation-first mimari** ile çözer — exchange her zaman gerçeğin tek kaynağıdır (single source of truth), bot'un local state'i sadece bir "inanç" (belief) olarak kabul edilir.

#### Problem 2 — Yeniden Başlatma Güvenliği (Restart Safety)

Bot kapanıp yeniden açıldığında, açık pozisyonlar kaybolmamalı, stop/TP emirleri tekrar gönderilmemeli, aynı emir iki kez submit edilmemelidir. Bu; WAL (Write-Ahead Log) intent sistemi, idempotent order ID'ler ve bootstrap'ta exchange'den state rebuild mekanizması ile çözülür.

#### Problem 3 — Bounded Retry ve Kill-Switch

Ağ hatası, API rate limit veya stale data durumunda bot sonsuz döngüye girmemeli, paniğe kapılmamalıdır. Kill-switch sistemi, belirli eşikleri aştığında trading'i durdurur; router retry mekanizması hataları sınıflandırır (retryable, fatal, idempotent_safe) ve bounded şekilde yeniden dener.

#### Problem 4 — Windows Async Uyumluluğu

Binance WebSocket + ccxt async kombinasyonu Windows'ta `ProactorEventLoop` ile çalışmaz. Sistem, `SelectorEventLoop` politikasını zorlar ve tüm pipe/stdout UTF-8 uyumsuzluklarını bootstrap'ta çözer.

### 1.4 Neden Bu Mimari?

Sistem bir **distributed execution system** olarak tasarlanmıştır — yani her modül bağımsız çalışabilir, birinin çökmesi diğerini durdurmamalıdır. Bu yüzden:

- Her async fonksiyon "guardian-safe" — içeride hata yakalar, dışarı asla raise etmez
- Tüm kritik operasyonlar telemetri üretir (JSONL format)
- Modüller arası iletişim shared `bot` objesi üzerinden, lock ile serialize edilmiş
- Config ENVironmen Variable'larla override edilebilir — deploy zamanı yeniden derleme gerekmez

---

## 2. Sistem Mimarisi Genel Bakış

### 2.1 Klasör Yapısı

```
eclipse_scalper/
├── bot/                    # Bot çekirdeği — EclipseEternal sınıfı
├── brain/                  # State persistence — PsycheState, Position
├── config/                 # Config şemaları — Config, MicroConfig
├── data/                   # Market data layer — DataCache
├── exchanges/              # Exchange adaptörleri — Binance, Paper Trading
├── execution/              # Tüm order execution mantığı
│   ├── bootstrap.py        # TEK başlangıç noktası — singleton orchestrator
│   ├── entry_loop.py       # Ana entry signal döngüsü — 20+ gate check
│   ├── entry_primitives.py # Ortak primitifler — symkey() SOT
│   ├── order_router.py     # Order submission + retry + idempotency
│   ├── position_manager.py # Stop/TP yönetimi
│   ├── reconcile.py        # Exchange ↔ local state senkronizasyonu
│   ├── shared_locks.py     # Per-symbol asyncio lock'ları
│   └── telemetry.py        # JSONL event logging
├── features/               # Feature engineering (teknik indikatörler)
├── integrations/           # Telegram notifier + control
├── policies/               # Trading policy katmanı
├── risk/                   # Risk yönetimi — RiskManager, KillSwitch
├── strategies/             # Sinyal üretimi — EclipseScalper stratejisi
├── utils/                  # Loglama, yardımcı fonksiyonlar
├── state/                  # Runtime state dosyaları (lock, pid, json)
├── logs/                   # Telemetri log'ları (telemetry.jsonl)
├── reports/                # Analiz raporları (backtest, ranking, vs.)
├── tools/                  # Test dosyaları + diagnostic araçlar
├── docs/                   # Bu doküman dahil tüm dokümantasyon
├── main.py                 # CLI giriş noktası (--dry-run, --equity, --mode)
└── pytest.ini              # Test konfigürasyonu
```

### 2.2 Klasör Detayları

#### `execution/` — Sistemin Kalbi

En kritik klasördür. Tüm order yaşam döngüsü burada yönetilir.

| Dosya | Satır | Rolü |
|---|---|---|
| `bootstrap.py` | ~1846 | Singleton sistem başlatıcısı |
| `entry_loop.py` | ~200+ | Signal'dan order'a ana döngü |
| `entry_primitives.py` | ~114 | Paylaşılan primitifler (symkey SOT) |
| `order_router.py` | ~150+ | Order gönderme, retry, idempotency |
| `position_manager.py` | ~150+ | Stop/TP koruma yönetimi |
| `reconcile.py` | ~150+ | Exchange reality sync |
| `shared_locks.py` | ~35 | Per-symbol lock'lar |
| `telemetry.py` | — | JSONL event sistemi |
| `error_codes.py` | — | Standart hata sınıfları |

#### `strategies/` — Signal Üretimi

```
strategies/
└── eclipse_scalper.py     # Ana strateji — scalper_signal() fonksiyonu
```

Multi-timeframe teknik analiz ile long/short sinyali ve confidence değeri üretir. ATR, Bollinger Bands, Stochastic Oscillator, ADX kullanır.

#### `risk/` — Güvenlik Katmanı

```
risk/
├── risk_manager.py        # Pozisyon açma veto'su (advisory only)
└── kill_switch.py         # Acil durdurma (data quality / API errors)
```

Risk Manager sadece "hayır" diyebilir — asla emir göndermez. Kill Switch, veri kalitesi bozulduğunda tüm entry'leri durdurur.

#### `brain/` — Hafıza Sistemi

```
brain/
└── state.py              # PsycheState + Position dataclass'ları
```

Tüm bot state'i `PsycheState` dataclass'ında tutulur ve LZ4-compressed binary olarak `~/.blade_eternal.brain.lz4` dosyasına yazılır. Restart sonrası kaldığı yerden devam eder.

#### `bot/` — Orchestration Core

```
bot/
├── core.py               # EclipseEternal — tüm fonksiyonlara geçilen bot objesi
└── runner.py             # run_bot() — asyncio event loop'u başlatır
```

`EclipseEternal` bot objesi şunları taşır: exchange adapter, state, data cache, config, active symbols, semaphore, shutdown event.

#### `config/` — Konfigürasyon

```
config/
└── settings.py           # Config (production) + MicroConfig (küçük hesap)
```

Tüm parametreler burada tanımlıdır. ENV variable'lar `main.py` tarafından override edilir.

#### `data/` — Piyasa Verisi

```
data/
├── cache.py              # DataCache / GodEmperorDataOracle
└── event_diary.py        # Signal kalitesi için olay günlüğü
```

OHLCV bar'ları ve anlık fiyatlar cache'de tutulur. `bot.data_ready` event'i, ilk veri gelene kadar entry_loop'u bekletir.

#### `exchanges/` — Exchange Adaptörleri

```
exchanges/
├── binance.py            # CCXT async Binance Futures wrapper
└── paper_trading.py      # Simülatif trading adapter (dry-run)
```

Her iki adapter aynı interface'i implemente eder: `create_order`, `cancel_order`, `fetch_open_orders`, `fetch_open_positions`, `fetch_balance`.

#### `integrations/` — Dış Servisler

```
integrations/
├── telegram_notifier.py  # Bildirimler (startup, entry, exit, error)
└── telegram_control.py   # Uzaktan komut kontrolü
```

#### `tools/` — Test ve Diagnostic

100+ test ve araç dosyası içerir. Unit testler `test_*_unit.py` pattern'ini takip eder, pytest ile çalışır.

#### `state/` — Runtime State Dosyaları

```
state/
├── locks/execution_bootstrap.lock   # Singleton bootstrap lock
├── locks/execution_bootstrap.pid    # Bootstrap PID
├── micro_edge_gates.json            # Mikrostrüktür gate konfigürasyonu
├── passive_realistic_profiles.json  # Pasif execution profilleri
└── paper_scoreboard.json            # Paper trading skor tablosu
```

#### `logs/` — Telemetri

```
logs/
└── telemetry.jsonl       # Tüm order/entry/exit/error olayları
```

---

## 3. Execution Pipeline Detaylı Açıklama

### 3.1 Sistem Nasıl Başlatılır?

#### Adım 1 — CLI Giriş Noktası (`main.py`)

```bash
# Dry-run (güvenli simülasyon)
python main.py --dry-run

# Gerçek hesap — live arm zorunlu
SCALPER_LIVE_TRADING=1 python main.py

# Micro hesap modu
python main.py --mode micro --equity 50
```

`main.py` v4.2 HARDENED şu işlemleri yapar:

```python
# Windows için asyncio politikası
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Komut satırı argümanları
parser.add_argument("--dry-run")
parser.add_argument("--equity")
parser.add_argument("--mode", choices=["auto", "micro", "production"])

# Ana bot runner'a devreder
bot.runner.run_bot(dry_run=..., equity=..., mode=...)
```

#### Adım 2 — Bootstrap Singleton (`execution/bootstrap.py`)

`bootstrap.py`, v1.8 ONE TRUE ENTRYPOINT — tüm sistemin tek ve otoriteli başlangıç noktasıdır. Birden fazla instance çalışmasını `_BootstrapSingleton` ile engeller (singleton lock + PID file).

Bootstrap sırası şu şekildedir:

```
1. UTF-8 hardening (Windows pipe uyumluluğu)
2. .env yükle (BINANCE_API_KEY, BINANCE_API_SECRET)
3. Singleton lock al → state/locks/execution_bootstrap.lock
4. Config yükle (settings.py → ENV override)
5. Exchange init (CCXT async — Binance Futures)
6. Brain state yükle ve doğrula (LZ4 decompress)
7. Binance health gate (ağ + auth kontrolü)
8. Paper trading setup (paper mode aktifse)
9. Risk manager init (optional — yoksa devam)
10. Telegram notifier başlat (optional)
11. Data cache init + market bootstrap
12. Exchange'den state rebuild (BOOT_REBUILD_ON_START=1 ise)
13. Async task loop başlat:
    ├─ guardian_loop          (health check)
    ├─ data_loop              (piyasa verisi toplama)
    ├─ position_manager_loop  (stop/TP yönetimi)
    ├─ exit_loop              (opsiyonel çıkış döngüsü)
    └─ entry_loop             (data_ready event'i bekler)
14. Paper smoke test (PAPER_FORCE_ONE_FILL=1 ise)
15. Graceful shutdown → state persist
```

**Kritik ENV Variable'lar:**

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `SCALPER_DRY_RUN` | `1` | Simülasyon modu |
| `SCALPER_PAPER_TRADING` | `1` | Paper trading |
| `SCALPER_LIVE_TRADING` | — | Live arm (açık silah güvenliği) |
| `ACTIVE_SYMBOLS` | `BTCUSDT` | Hangi semboller işlem görecek |
| `BOOT_REBUILD_ON_START` | — | Exchange'den pozisyon rebuild |
| `BOOT_MAINTENANCE_ONESHOT` | — | Tek reconcile tick, çık |
| `PAPER_INITIAL_EQUITY` | `1000` | Paper başlangıç bakiyesi |

### 3.2 Modüller Nasıl Yüklenir?

Tüm modüller best-effort (hata toleranslı) import ile yüklenir:

```python
# Opsiyonel modüllerin güvenli import'u
try:
    from execution.telemetry import emit
except Exception:
    emit = None  # Telemetri yoksa sessizce devam

try:
    from execution.reliability_gate_runtime import gate_check
except Exception:
    gate_check = None
```

Bu pattern, eksik veya bozuk bir modülün tüm sistemi çökertmesini önler.

### 3.3 Execution Flow Özeti

```
Veri Alma → Signal Üretme → Risk Kontrol → Order Gönderme → State Update
```

Her adım detaylı olarak sonraki bölümlerde açıklanmaktadır.

---

## 4. Signal Generation Sistemi

### 4.1 Strateji Dosyası

**Dosya:** `strategies/eclipse_scalper.py` (v3.0 — COSMIC SIGNAL ASCENDANT)

### 4.2 Ana Fonksiyon

```python
def scalper_signal(
    symbol: str,
    data=None,
    cfg=None,
    bot=None,
    **kw
) -> Tuple[bool, bool, float]:
    """
    Returns:
        is_long  (bool): True ise long sinyal
        is_short (bool): True ise short sinyal
        confidence (float): 0.0 – 1.0 arası güven skoru
    """
```

Sadece ikisinden biri True olabilir — aynı anda hem long hem short sinyali mümkün değildir.

### 4.3 Teknik İndikatörler

| İndikatör | Amaç |
|---|---|
| **ATR** (Average True Range) | Volatilite ölçümü — stop mesafesi hesabı |
| **Bollinger Bands** | Fiyat aralığı tanımı — squeeze + breakout tespiti |
| **Stochastic Oscillator** | Momentum ölçümü — aşırı alım/satım bölgeleri |
| **ADX** (Average Directional Index) | Trend gücü — sadece güçlü trend dönemlerinde işlem |
| **SciPy find_peaks** | Lokal zirve/dip tespiti (opsiyonel, import ile) |

### 4.4 Multi-Timeframe Mantığı

Sistem üç zaman dilimini birlikte değerlendirir:

```
1m (Primary)   → İşlem zamanlaması ve anlık momentum
5m (Confirm)   → Kısa vadeli trend onayı
15m (Context)  → Orta vadeli piyasa yapısı
```

15dk trend'e karşı 1dk'da işlem açılmaz. Her üç timeframe uyumlu olmalıdır.

### 4.5 Confidence Hesabı

Confidence değeri, kaç tane teknik koşulun sağlandığına göre 0.0–1.0 arası üretilir:

```python
MIN_CONFIDENCE = 0.72           # Production: minimum 72%
MIN_CONFIDENCE_HIGH_VOL = 0.65  # Yüksek volatilite döneminde: 65%
MIN_CONFIDENCE_MICRO = 0.35     # Micro mod (küçük hesap): 35%
```

### 4.6 Signal Lifecycle

```
1. Data Quality Check
   └─ staleness_check() → veri tazeliği doğrulanır
   └─ min bar count → yeterli OHLCV verisi var mı?

2. Multi-Timeframe Indicator Computation
   └─ 1m / 5m / 15m DataFrame'leri hesaplanır

3. Gate Conditions Evaluation
   └─ ATR > min_atr? (pozisyon büyüklüğü kontrollü)
   └─ ADX > threshold? (trend gücü yeterli?)
   └─ Bollinger + Stochastic confluence?

4. Confidence Score Aggregation
   └─ Kaç gate geçti? → confidence hesabı

5. Signal Decision
   └─ confidence >= MIN_CONFIDENCE?
   └─ is_long veya is_short → entry_loop'a ilet

6. Debug Output (opsiyonel)
   └─ SCALPER_SIGNAL_DIAG=1 → throttled logging
   └─ SCALPER_DEBUG_LOOSE=1 → gevşek gate'ler (test amaçlı)
   └─ SCALPER_FORCE_ENTRY_TEST=1 → zorla entry (plumbing validation)
```

### 4.7 Veri Kalitesi Koruması

```python
# Veri eskimişse işlem yok
if data_age_sec > KILL_MAX_DATA_STALENESS_SEC:  # 150 saniye
    return False, False, 0.0

# Yetersiz bar sayısı ise işlem yok
if len(df_1m) < MIN_BARS_REQUIRED:
    return False, False, 0.0
```

---

## 5. Risk Management Sistemi

### 5.1 İki Katmanlı Risk Kontrolü

Eclipse Scalper'da risk yönetimi iki ayrı katmandan oluşur ve ikisi birbirinden bağımsız çalışır:

| Katman | Dosya | Sorumluluk |
|---|---|---|
| **RiskManager** | `risk/risk_manager.py` | Pozisyon büyüklüğü, kayıp limitleri, concurrent limit |
| **KillSwitch** | `risk/kill_switch.py` | Data quality / API error bazlı acil durdurma |

### 5.2 RiskManager — Veto Sistemi

**Önemli:** RiskManager sadece danışmanlık (advisory) rolündedir. Asla emir göndermez — sadece "hayır" diyebilir.

```python
class RiskManager:
    def can_open_position(
        self,
        symbol: str,
        side: str,         # 'long' veya 'short'
        size_usd: float,   # Pozisyon büyüklüğü (USD)
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Returns:
            (True, "") → Pozisyon açılabilir
            (False, "Reason") → Reddedildi, sebep açıklanır
        """
```

**Kontrol Edilen Limitler:**

| Limit | Production | Micro |
|---|---|---|
| Günlük kayıp limiti | %3 | %3 |
| Haftalık kayıp limiti | %7 | %7 |
| Maksimum drawdown | %15 | %15 |
| Pozisyon başına max | %5 equity | %5 equity |
| Eşzamanlı pozisyon | 6 | 1 |
| İşlemler arası cooldown | 300 saniye | 300 saniye |

**State Persistence:** RiskManager SQLite veritabanı kullanır (`data/risk_state.db`). Restart sonrası günlük/haftalık kayıp sayacı sıfırlanmaz.

### 5.3 Kill Switch — Acil Durdurma

`risk/kill_switch.py` v1.0; veri kalitesi veya API hata oranı eşiği aşıldığında tüm entry'leri durdurur.

**Tetikleyici Koşullar:**

```python
# Veri eskimesi
if data_age_sec > KILL_MAX_DATA_STALENESS_SEC:  # 150s
    request_halt(bot, "stale_data")

# API hata oranı
if api_error_rate > KILL_MAX_API_ERROR_RATE:    # %35
    request_halt(bot, "api_error_rate")

# Ardışık API hatası
if consecutive_errors > KILL_MAX_API_ERROR_BURST:  # 12
    request_halt(bot, "api_error_burst")
```

**Escalation Seviyeleri:**

```
1. Normal Halt → Entry durdurulur, exitler devam eder
2. Trip Sayacı → Her halt bir sayaç artırır
3. Emergency Flatten → KILL_ESCALATE_FLAT_AFTER_TRIPS aşılırsa pozisyonlar kapatılır
4. Shutdown → KILL_ESCALATE_SHUTDOWN_AFTER_TRIPS aşılırsa bot kapanır

Cooldown: KILL_SWITCH_COOLDOWN_SEC = 300 saniye
(Halt sonrası 5 dakika geçmeden yeniden entry açılmaz)
```

**Kritik Güvenlik Kuralı:** Kill switch yalnızca entry'leri etkiler. Exit emirleri (stop, take-profit, trailing) her zaman geçer. Açık pozisyonlar koruma altında kalır.

### 5.4 Position Sizing

Position büyüklüğü şu şekilde belirlenir:

```python
# Sinyal size öneriyorsa → onu kullan
notional_usd = signal.get("notional_usd", None)

# Sinyal yoksa → config'den varsayılan
if notional_usd is None:
    notional_usd = cfg.FIXED_NOTIONAL_USDT  # 25 USDT production, 8 USDT micro

# Kaldıraç ile nominal konum büyüklüğü
position_size = notional_usd * leverage / current_price
```

### 5.5 Stop / Take-Profit Yapısı

```python
# Stop Loss — ATR bazlı
stop_distance = atr * STOP_ATR_MULT          # 1.10x ATR
stop_distance = min(stop_distance,
                    entry_price * MAX_STOP_PCT)  # max %3

# Trailing Stop
# Aktivasyon: Entry'den itibaren +1.30x ATR kazanınca
TRAILING_ACTIVATION_RR = 1.30
# Callback rate: %0.45 geri çekilmede trailing tetiklenir
TRAILING_CALLBACK_RATE = 45  # bps
```

---

## 6. Brain ve State Sistemi

### 6.1 State Architecture

Bot'un tüm hafızası tek bir `PsycheState` dataclass'ında tutulur. Bu obje LZ4-compressed binary olarak diske yazılır:

```
Kayıt yeri: ~/.blade_eternal.brain.lz4
Format: LZ4-compressed pickle/binary
```

### 6.2 PsycheState Veri Yapısı

**`brain/state.py` v4.6 — GOD-EMPEROR**

```python
@dataclass
class PsycheState:
    # === Pozisyon Yönetimi ===
    positions: Dict[str, Position]        # canonical_symbol → Position

    # === Equity Takibi ===
    current_equity: float
    peak_equity: float
    current_drawdown_pct: float

    # === P&L Takibi ===
    daily_pnl: float
    start_of_day_equity: float
    current_day: str                      # ISO date

    # === Performans ===
    total_trades: int
    total_wins: int
    win_streak: int
    win_rate: float

    # === Sembol Kısıtlamaları ===
    blacklist: Dict[str, float]           # symbol → expiry_ts
    blacklist_reason: Dict[str, str]      # symbol → reason
    consecutive_losses: Dict[str, int]   # symbol → kayıp sayısı

    # === Timing ===
    last_exit_time: Dict[str, float]      # symbol → son çıkış zamanı

    # === Reconcile Idempotency ===
    known_exit_order_ids: Set[str]        # Duplicate exit önleme (max 50k)

    # === Per-Symbol Performans ===
    symbol_performance: Dict[str, Dict]  # pnl, wins, losses, trailing_ids

    # === Signal Kalitesi ===
    entry_confidence_history: Dict[str, List[float]]  # max 200 per symbol

    # === Entry Watch Persistence ===
    entry_watches: Dict[str, Dict]        # max 500 entry

    # === Ephemeral Runtime (Restart'a Dayanır) ===
    run_context: Dict[str, Any]           # WAL intents, pending blocks, etc.

    # === Kill Switch Metrikleri ===
    kill_metrics: Dict

    # === Adaptif Knob'lar ===
    guard_knobs: Dict
```

### 6.3 Position Dataclass

```python
@dataclass
class Position:
    symbol: str           # Canonical symbol (örn. "BTCUSDT")
    side: str             # 'long' veya 'short'
    size: float           # Absolute miktar (BTC, ETH, vb.)
    entry_price: float
    entry_ts: float       # Unix timestamp
    leverage: int
    atr: float            # Entry zamanındaki ATR değeri

    # Stop/TP Durumu
    hard_stop_order_id: Optional[str]
    trailing_active: bool
    breakeven_moved: bool

    # Entry Kalitesi
    confidence: float     # Sinyal güven skoru
```

### 6.4 Symbol Canonicalization (Sembol Normalizasyonu)

Tüm semboller canonical formata dönüştürülür. Bu dönüşümün tek kaynağı (SOT) `execution/entry_primitives.py:symkey()` fonksiyonudur:

```python
# Tüm bu formatlar → "BTCUSDT" olur
symkey("BTC/USDT:USDT")  → "BTCUSDT"
symkey("BTC/USDT")       → "BTCUSDT"
symkey("btcusdt")        → "BTCUSDT"
symkey("BTC-USDT")       → "BTCUSDT"
symkey(None)             → ""

# Diğer modüllerde import
from execution.entry_primitives import symkey as _shared_symkey
```

### 6.5 State Persistence — Nasıl Çalışır?

```
1. Bootstrap → LZ4 dosyası varsa yükle
   ├─ from_loaded(obj) → Migration + validation
   ├─ validate() → Key normalizasyonu, bozuk veri onarımı, koleksiyon limitleri
   └─ recompute_derived() → win_rate, drawdown yeniden hesap

2. Her önemli olay sonrası → save (async)
   └─ to_dict() → Set'leri list'e, date'leri ISO'ya çevir
   └─ LZ4 compress → diske yaz

3. Graceful shutdown → Final save
   └─ Signal handler (SIGINT, SIGTERM) → state.save()
```

### 6.6 Restart Recovery

```
Bot kapanır →
  state.run_context['wal_intents'] içinde açık intent'ler var

Bot açılır →
  bootstrap.py: BOOT_REBUILD_ON_START=1 ise exchange'den pozisyonlar çekilir
  → Orphan pozisyonlar adopt edilir (reconcile.py)
  → WAL intent'ler kontrol edilir, duplicate submit engellenir
  → known_exit_order_ids ile duplicate exit engellenir
```

### 6.7 Collection Kapasitesi

| Koleksiyon | Max Boyut |
|---|---|
| `known_exit_order_ids` | 50.000 |
| `entry_confidence_history` (per symbol) | 200 |
| `trailing_order_ids` (per symbol) | 20 |
| `entry_watches` | 500 |

---

## 7. Order Execution Sistemi

### 7.1 Order Router (`execution/order_router.py`)

v2.4 — BINANCE COID<36 + IDEMPOTENT CANCEL

Order Router, tüm emir gönderme ve iptal operasyonlarını yönetir. Temel güvenlik garantileri:

- **Binance clientOrderId < 36 karakter:** Otomatik hash/truncate
- **Idempotent cancel:** `-2011 "order not found"` başarı sayılır
- **Duplicate handler:** `-4116` hatası bounded+deduped (max 12 variant)
- **Bounded retry:** Sonsuz döngü yok; hata tipi retry stratejisini belirler

### 7.2 create_order() Fonksiyonu

```python
async def create_order(
    bot,
    symbol: str,
    type: str,          # 'limit', 'market', 'stop_market', vb.
    side: str,          # 'buy' veya 'sell'
    amount: float,
    price: Optional[float],
    params: dict,
    retries: int,
    intent_component: str,
    intent_kind: str    # 'entry', 'stop', 'tp', 'trailing'
) -> dict:              # ccxt order dict
```

**Dahili İşlem Sırası:**

```
1. clientOrderId oluştur
   └─ intent + symbol + timestamp hash → max 35 karakter

2. Hedge mode inject
   └─ params['positionSide'] = 'LONG' veya 'SHORT'

3. reduceOnly kontrolü
   └─ closePosition=True ise amount=0.0, reduceOnly kaldırılır

4. FIRST_LIVE_SAFE kontrolü
   └─ İlk live işlem → küçük notional cap

5. Exchange'e gönder (ccxt)

6. Hata sınıflandırması
   ├─ Retryable → bounded retry (max retries)
   ├─ Fatal → anında fail
   ├─ Idempotent safe → başarı say
   └─ Duplicate (-4116) → variant ID ile yeniden dene (max 12)

7. Telemetri emit
   └─ emit_order_create(event_type, payload)

8. WAL intent ledger güncelle
   └─ state.run_context['wal_intents']
```

### 7.3 Exit Emirleri Ayrıcalıklı

**Kritik kural:** `intent_reduce_only=True` olan emirlere hiçbir entry kısıtlaması uygulanmaz:

```python
# Kill switch, circuit breaker, reliability gate...
# SADECE entry emirleri için geçerlidir.
# Stop, TP, trailing emirleri her zaman geçer.
if intent_reduce_only:
    skip_all_entry_gates()
    proceed_to_submit()
```

Bu kural, açık pozisyonların korumasız kalmasını önler.

### 7.4 cancel_order() — Idempotent İptal

```python
async def cancel_order(
    bot,
    symbol: str,
    order_id: str,
    ...
) -> dict:
```

Binance `-2011` hatası (order not found) başarı olarak kabul edilir. Bu, `position_manager_loop` veya `reconcile`'ın "zaten iptal edilmiş" emirleri tekrar tekrar iptal etmeye çalışmamasını sağlar.

### 7.5 Hedge Mode Desteği

Binance Futures hedge mode'da her pozisyon `positionSide` gerektirir:

```python
# Long pozisyon için buy emri
params['positionSide'] = 'LONG'

# Short pozisyon için sell emri
params['positionSide'] = 'SHORT'

# Long pozisyon kapatmak için sell emri
params['positionSide'] = 'LONG'
params['reduceOnly'] = True
```

### 7.6 Entry Loop — 20+ Gate Kontrolü

**`execution/entry_loop.py` v1.6 — AUTHORITATIVE ENTRY ORCHESTRATOR**

Entry Loop, signal'dan order submission'a giden yolda 20+ kontrol noktası uygular:

```
Gate 1:  entries_allowed()           → Genel giriş izni
Gate 2:  trade_allowed() (kill sw.)  → Kill switch kontrolü
Gate 3:  anomaly_should_pause()      → Anomali tespiti
Gate 4:  staleness_check()           → Veri tazeliği
Gate 5:  reliability_gate_runtime    → Runtime güvenilirlik kapısı
Gate 6:  compute_entry_decision()    → Sinyal + belief state değerlendirme
Gate 7:  RiskManager.can_open()      → Risk veto kontrolü
Gate 8:  Per-symbol pending lock     → Aynı sembol için concurrent entry önleme
Gate 9:  ENTRY_LOCAL_COOLDOWN_SEC    → Submit sonrası cooldown (8s)
Gate 10: ENTRY_PER_SYMBOL_GAP_SEC   → Semboller arası throttle (2.5s)
         ... (circuit breaker, confidence threshold, vs.)
Gate N:  create_order()              → Emir gönder
```

**Throttle Mekanizması:**

```python
# Sembol başına throttle
ENTRY_PER_SYMBOL_GAP_SEC = 2.5  # saniye

# Her submit sonrası cooldown (başarılı veya başarısız)
ENTRY_LOCAL_COOLDOWN_SEC = 8.0  # saniye

# Döngü frekansı
ENTRY_POLL_SEC = 1.0  # saniye başına kontrol
```

### 7.7 Fill Detection ve Protection Staging

Emir gönderildi, peki doldu mu?

```python
# Dolum kontrolü
filled_qty = order_filled(order)    # ccxt order dict'ten

# Koruma planı — fill oranına göre hangi aşamalar aktif?
plan = build_staged_protection_plan(
    requested_qty=amount,
    filled_qty=filled_qty,
    min_fill_ratio=cfg.MIN_FILL_RATIO,    # 0.85
    trailing_enabled=cfg.TRAILING_ENABLED
)

# Aşama 1: Acil stop
if plan['stage1_active']:
    await _maybe_place_stage1_emergency_stop(bot, symbol, ...)

# Entry watch kaydı
await register_entry_watch(bot, symbol, order_id, ...)
```

---

## 8. Codex Nedir ve Bu Projeye Nasıl Yardımcı Olur

### 8.1 Codex Nedir?

OpenAI Codex (ve benzer AI kod asistanları), kaynak kodu anlayan ve onunla etkileşime giren büyük dil modelidir. Bir projeyi analiz ederek:

- Kod kalitesi sorunlarını tespit eder
- Bug'ları açıklar ve fix önerir
- Dokümantasyon üretir
- Test yazar
- Refactoring önerir
- Yeni feature tasarlar

Eclipse Scalper gibi büyük, çok modüllü bir projede Codex'in değeri şunlardır:

### 8.2 Eclipse Scalper'da Codex'in Değeri

#### Modül Bağımlılıklarını Anlama

Eclipse Scalper'ın en karmaşık tarafı modüller arası implicit bağımlılıklardır. Örneğin:

- `symkey()` her yerde kullanılır ama SOT `entry_primitives.py`'dadır
- `bot.state.run_context` birçok modül tarafından yazılır/okunur
- Kill switch sadece entry'leri durdurmalı, exit'leri durdurmamalı

Codex bu pattern'leri tüm codebase'i okuyarak öğrenebilir ve yeni kod yazarken aynı kuralları uygular.

#### Bug Tespiti — Gerçek Örnek

```python
# Hatalı kod — kill switch exit'i de blokluyor
if not trade_allowed(bot):
    return  # BUG: Exit emirleri de burada bloklaniyor!

# Doğru kod — intent kontrolü
if not trade_allowed(bot) and not intent_reduce_only:
    return  # Sadece entry emirleri bloklanır
```

Codex bu pattern'i öğrenip yeni dosyalarda aynı bug'ı işaretleyebilir.

#### Test Yazımı

```python
# Eclipse Scalper test pattern'ini öğrenen Codex şunu yazabilir:
class TestKillSwitchExitSafety(unittest.TestCase):
    def test_exit_bypasses_kill_switch(self):
        bot = SimpleNamespace(
            cfg=SimpleNamespace(KILL_SWITCH_ENABLED=True),
            state=PsycheState(...),
        )
        # Kill switch aktif
        bot.state.kill_metrics['halted'] = True

        # Exit emri → geçmeli
        result = await create_order(
            bot, "BTCUSDT", "market", "sell", 0.001,
            intent_reduce_only=True
        )
        self.assertIsNotNone(result)  # Exit geçti
```

#### Diagnostic ve Audit

Codex şu soruları yanıtlayabilir:

- "Bu fonksiyonda lookahead bias var mı?"
- "Bu retry loop neden sonsuz dönebilir?"
- "Bu state field nerede yazılıyor, nerede okunuyor?"
- "Bu modülü unit test etmek için ne gerekiyor?"

### 8.3 Codex'in Sınırlamaları

| Sınırlama | Açıklama |
|---|---|
| **Runtime bilgisi yok** | Codex kodu okur ama çalıştırmaz |
| **Exchange API davranışı** | Binance'in edge case'lerini test etmez |
| **Async timing** | Race condition'ları statik analizle bulmak zordur |
| **State corruption** | LZ4 dosyasındaki bozulmayı tespit edemez |

### 8.4 Gerçek Kullanım Senaryoları

**Senaryo 1 — Yeni Strateji Ekleme:**

```
Codex'e söyle:
"strategies/eclipse_scalper.py'e bakarak yeni bir RSI-based strateji yaz.
Aynı interface'i kullan: scalper_signal() → (is_long, is_short, confidence)"
```

Codex mevcut strateji pattern'ini okur ve uyumlu yeni strateji üretir.

**Senaryo 2 — Bug Fix:**

```
Codex'e söyle:
"execution/order_router.py'de -4116 hatası 12 denemeden sonra da devam ediyor.
_MAX_VARIANTS limitini bul ve neden aşıldığını açıkla."
```

**Senaryo 3 — Reliability Audit:**

```
Codex'e söyle:
"execution/ klasöründeki tüm async fonksiyonları incele.
Guardian-safe contract'ı ihlal eden — yani try/except olmadan raise edebilecek —
fonksiyonları listele."
```

---

## 9. Codex CLI Kullanım Rehberi

### 9.1 Codex'i Bu Repo İçin Nasıl Başlatırsınız?

Codex'i kullanmak için önce repo'yu context olarak vermelisiniz. Claude Code (bu asistan) veya OpenAI Codex CLI ile şu şekilde çalışılır:

### 9.2 Temel Komutlar ve Örnekler

#### Repo Analizi

```bash
# Tüm repo'yu analiz et, mimarini açıkla
codex "Bu repo'nun mimarisini açıkla. execution/ klasörüne odaklan."

# Belirli bir modülü incele
codex "execution/bootstrap.py'i oku ve başlangıç sırasını adım adım açıkla."

# Bağımlılık haritası
codex "bot objesi hangi modüllerde kullanılıyor? Her kullanım noktasını listele."
```

#### Bug Bulma

```bash
# Execution güvenilirliği analizi
codex "execution/ klasöründe guardian-safe contract'ı ihlal eden fonksiyonları bul."

# State corruption riski
codex "PsycheState'in run_context'ini aynı anda hangi modüller yazıyor?
Race condition riski var mı?"

# Kill switch bypass riski
codex "Kill switch kontrolü yapılmadan order gönderen kod var mı?
intent_reduce_only kontrolünü doğrula."
```

#### Fix Üretme

```bash
# Belirli hatayı düzelt
codex "order_router.py'deki retry loop'u incele.
-4116 hatasının 13. denemede de dönebildiği bir path var mı? Düzelt."

# Pattern standardizasyonu
codex "execution/ klasöründeki tüm fonksiyonlarda try/except kullanımını
standartlaştır. Guardian-safe contract'a uymayan her fonksiyonu düzelt."
```

#### Geliştirme

```bash
# Signal geliştirme
codex "strategies/eclipse_scalper.py'e VWAP-based entry filter ekle.
Mevcut gate'lerle aynı pattern'i kullan."

# Yeni metric
codex "entry_loop.py'e entry latency metrici ekle.
signal_time → order_submit_time arasını telemetry'ye logla."

# Yeni test
codex "risk/kill_switch.py için unit test yaz.
bot stub olarak SimpleNamespace kullan, test pattern'i için
tools/test_entry_loop_unit.py'e bak."
```

#### Audit ve Doğrulama

```bash
# Execution invariant kontrolü
codex "execution/ klasöründeki tüm modülleri incele.
EXE-01 (idempotency), EXE-02 (lifecycle), EXE-03 (kill-switch precedence)
invariantlarını doğrula."

# Data hazırlık
codex "entry_loop'un data_ready event'i gelmeden çalışmayacağını doğrula.
Potansiyel race condition var mı?"

# Config coverage
codex "config/settings.py'deki tüm config key'lerini listele.
execution/ klasöründe kullanılmayan key varsa işaretle."
```

### 9.3 Claude Code ile Entegrasyon

Bu repo için Claude Code (bu asistan) kullanıyorsanız:

```bash
# Claude Code CLI
claude "execution/entry_loop.py'i incele ve tüm gate'leri sırala"
claude "strategies/ klasöründeki sinyal üretim mantığını türkçe açıkla"
claude "risk manager ve kill switch arasındaki farkı açıkla"

# Dosya bazlı soru
claude read execution/bootstrap.py -- "singleton pattern nasıl uygulanmış?"
```

### 9.4 İyi Prompt Yazma Rehberi

| Kötü Prompt | İyi Prompt |
|---|---|
| "kodu düzelt" | "execution/order_router.py'de -4116 retry limitini bul ve 12 denemede durduğunu doğrula" |
| "test yaz" | "risk/kill_switch.py için test yaz, tools/test_entry_loop_unit.py pattern'ini kullan" |
| "hata var mı" | "entry_loop.py'de kill switch bypass riski var mı? intent_reduce_only kontrolünü izle" |
| "açıkla" | "brain/state.py'deki validate() metodunun ne zaman çağrıldığını açıkla" |

---

## 10. Geliştirici Workflow

### 10.1 Repo'yu İlk Kez İnceleme

```bash
# 1. Repo'yu klonla / dizine gir
cd eclipse_scalper

# 2. Bağımlılıkları kur
pip install -r requirements.txt

# 3. .env dosyasını hazırla
cp .env.example .env
# BINANCE_API_KEY=...
# BINANCE_API_SECRET=...
# TELEGRAM_TOKEN=...   (opsiyonel)
# TELEGRAM_CHAT_ID=... (opsiyonel)

# 4. Test suite'ini çalıştır (repo'nun sağlıklı olduğunu doğrula)
python -m pytest tools/ -v

# 5. Paper trading ile başlat
python main.py --dry-run
```

### 10.2 Geliştirme Döngüsü

```
1. Değişiklik planla
   └─ Hangi modül etkileniyor?
   └─ Guardian-safe contract korunuyor mu?
   └─ symkey() SOT ihlal ediliyor mu?

2. Test yaz (önce)
   └─ tools/test_<module>_unit.py dosyasına ekle
   └─ SimpleNamespace bot stub kullan
   └─ sys.stdout.reconfigure(encoding="utf-8")

3. Değişikliği yap
   └─ Mevcut pattern'leri takip et
   └─ Best-effort import kullan (try/except)
   └─ Telemetry emit ekle

4. Testi çalıştır
   python -m pytest tools/test_<module>_unit.py -v

5. Tüm suite'i çalıştır
   python -m pytest tools/ -v

6. Paper trading ile manual test
   python main.py --dry-run

7. Reliability gate kontrol
   python tools/reliability_gate.py
```

### 10.3 Test Yazma Standardı

```python
# tools/test_yeni_ozellik_unit.py

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

# Windows codepage fix
sys.stdout.reconfigure(encoding="utf-8")

# Root import
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brain.state import PsycheState, Position
from execution.entry_primitives import symkey

class YeniOzellikTests(unittest.TestCase):

    def _make_bot(self, **cfg_overrides):
        """Standart bot stub oluştur."""
        cfg = SimpleNamespace(
            KILL_SWITCH_ENABLED=False,
            CIRCUIT_BREAKER_ENABLED=False,
            MIN_CONFIDENCE=0.72,
            LEVERAGE=20,
            FIXED_NOTIONAL_USDT=25.0,
            **cfg_overrides
        )
        state = PsycheState()
        return SimpleNamespace(
            cfg=cfg,
            state=state,
            ex=None,
            exchange=None,
            data=SimpleNamespace(price={}, ohlcv={}),
            active_symbols={"BTCUSDT"},
        )

    def test_temel_senaryo(self):
        bot = self._make_bot()
        # Test mantığı
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
```

### 10.4 Yeni Strateji Ekleme

```python
# strategies/yeni_strateji.py

from execution.entry_primitives import symkey

def yeni_strateji_signal(
    symbol: str,
    data=None,
    cfg=None,
    bot=None,
    **kw
):
    """
    Yeni strateji implementasyonu.

    Returns:
        (is_long, is_short, confidence)
    """
    try:
        # 1. Veri kalitesi kontrolü
        if data is None:
            return False, False, 0.0

        # 2. İndikatör hesapla
        # ...

        # 3. Gate koşullarını değerlendir
        # ...

        # 4. Sonuç
        return is_long, is_short, confidence

    except Exception as e:
        # Guardian-safe: asla raise etme
        return False, False, 0.0
```

### 10.5 Yeni Risk Kuralı Ekleme

```python
# risk/risk_manager.py — can_open_position() içine yeni kural:

def can_open_position(self, symbol, side, size_usd, current_price):
    # ... mevcut kurallar ...

    # YENİ KURAL: Gece saat 02:00-04:00 arası işlem yapma
    current_hour = datetime.utcnow().hour
    if 2 <= current_hour < 4:
        return False, "low_liquidity_hours"

    return True, ""
```

### 10.6 Codex ile Pair Programming

```bash
# 1. Görevi tanımla
claude "entry_loop.py'e yeni bir gate eklemem gerekiyor:
Son 1 saatte 3'ten fazla ardışık kayıp varsa entry'i durdur.
Mevcut gate pattern'ini takip et ve telemetry emit ekle."

# 2. Üretilen kodu incele
# Claude kodu önerir → sen gözden geçirirsin

# 3. Test yaz
claude "önerilen gate için unit test yaz.
test_entry_loop_unit.py pattern'ini takip et."

# 4. Birlikte debug
claude "test başarısız: [hata mesajı]
entry_loop.py satır 145'teki gate kontrolünü incele."
```

---

## 11. Gerçek Execution Flow — Derin Teknik Analiz

### 11.1 Bot Başlatıldığında Ne Olur?

Şimdi bir `python main.py --dry-run` komutundan itibaren sistemi microsaniye seviyesinde takip edelim:

#### Faz 1: Sistem Başlatma (t=0ms → t=~3000ms)

```
main.py:
├─ asyncio.WindowsSelectorEventLoopPolicy() ← Windows fix
├─ .env parse → sys.environ populate
├─ CLI args → dry_run=True, equity=None, mode="auto"
└─ asyncio.run(bot.runner.run_bot(...))

bot/runner.py:
└─ run_bot() → execution.bootstrap.main()

execution/bootstrap.py (_BootstrapSingleton):
├─ [t=10ms]  stdout/stderr UTF-8 reconfigure
├─ [t=15ms]  .env load (python-dotenv)
├─ [t=20ms]  Singleton lock acquire
│            → state/locks/execution_bootstrap.lock
├─ [t=25ms]  Config instantiate (settings.Config or MicroConfig)
├─ [t=30ms]  ENV bridge: ACTIVE_SYMBOLS → bot.active_symbols
│            SCALPER_EQUITY → cfg.INITIAL_EQUITY
├─ [t=50ms]  CCXT exchange init (async)
│            → exchanges/binance.py (dry-run: exchanges/paper_trading.py)
├─ [t=100ms] brain state load
│            → ~/.blade_eternal.brain.lz4
│            → PsycheState.from_loaded() + validate() + recompute_derived()
├─ [t=500ms] Binance health gate (network + auth ping)
│            → Dry-run'da skip
├─ [t=550ms] Risk manager init
│            → risk/risk_manager.py (SQLite: data/risk_state.db)
├─ [t=560ms] Telegram notifier (async, best-effort)
├─ [t=600ms] EclipseEternal (bot) instance create
│            → bot.state, bot.cfg, bot.ex, bot.data, bot.active_symbols
├─ [t=700ms] DataCache bootstrap
│            → data/cache.py: OHLCV cache initialize
│            → Exchange'den sembol metadata çek
└─ [t=800ms] Async task loop başlat
```

#### Faz 2: Async Task Loop (t=800ms → sürekli)

```
asyncio.gather(
    guardian_loop(bot),        ← Health check, anomaly detection
    data_loop(bot),            ← Market data collection
    position_manager_loop(bot),← Stop/TP protection
    entry_loop(bot),           ← Signal → Order (data_ready bekler)
    reconcile_task(bot),       ← Exchange ↔ state sync
)
```

Her task kendi döngüsünde çalışır, bağımsızdır.

### 11.2 Piyasa Verisi Akışı (data_loop)

```
data_loop(bot):
├─ Her sembol için (BTCUSDT, ETHUSDT, ...):
│  ├─ exchange.fetch_ohlcv(symbol, "1m", limit=200)
│  ├─ exchange.fetch_ohlcv(symbol, "5m", limit=100)
│  ├─ exchange.fetch_ohlcv(symbol, "15m", limit=50)
│  ├─ exchange.fetch_ticker(symbol) → anlık fiyat
│  └─ bot.data.ohlcv[symbol] = {...}  ← cache güncelle
│     bot.data.price[symbol] = last_price
│
└─ İlk veri başarıyla geldi → bot.data_ready.set()
   → entry_loop artık çalışabilir
```

**Veri tazeliği:**
```python
# Veri 150 saniyeden eski → kill switch tetiklenir
if time.time() - last_data_time > KILL_MAX_DATA_STALENESS_SEC:
    kill_switch.request_halt(bot, "stale_data")
```

### 11.3 Signal Üretimi (entry_loop → scalper_signal)

```
entry_loop(bot):
├─ await bot.data_ready.wait()  ← Veri hazır mı?
│
└─ DÖNGÜ (ENTRY_POLL_SEC=1.0s):
   ├─ [Gate 1] entries_allowed() → False ise skip
   ├─ [Gate 2] trade_allowed() → Kill switch kontrol
   ├─ [Gate 3] anomaly_should_pause() → Anomali var mı?
   ├─ [Gate 4] data staleness kontrolü
   │
   └─ Her sembol için (ENTRY_PER_SYMBOL_GAP_SEC=2.5s throttle):
      ├─ scalper_signal(symbol, data=bot.data, cfg=bot.cfg, bot=bot)
      │  → (is_long, is_short, confidence)
      │
      ├─ confidence >= ENTRY_MIN_CONFIDENCE? (0.72)
      │  Hayır → skip
      │
      ├─ RiskManager.can_open_position() → Hayır → skip
      │
      ├─ Per-symbol lock al → concurrent entry engelle
      │
      └─ [Sinyal geçti] → create_order()
```

### 11.4 Order Gönderme (order_router)

```
create_order(bot, "BTCUSDT", "market", "buy", 0.001, ...):

├─ clientOrderId oluştur
│  "ent_BTCUSD_1708434523_a3f2" → 26 karakter (<36 limit)

├─ params hazırla
│  {'positionSide': 'LONG', 'newOrderRespType': 'RESULT'}

├─ exchange.create_order() → ccxt
│  Paper mode: PaperTradingAdapter.create_order()
│  → Anlık fiyatta simüle edilmiş dolum
│  → Slippage uygula (%0.05)
│  → Fee çıkar (taker: %0.04)

├─ Telemetri
│  emit(bot, "order_create", {symbol, type, side, amount, ...})
│  → logs/telemetry.jsonl

└─ WAL intent kaydet
   state.run_context['wal_intents'][order_id] = {...}
```

### 11.5 Fill Detection ve Koruma

```
order = await create_order(...)

filled_qty = order_filled(order)  # ccxt order dict'ten

if filled_qty > 0:
    # Position state güncelle
    bot.state.positions["BTCUSDT"] = Position(
        symbol="BTCUSDT",
        side="long",
        size=filled_qty,
        entry_price=order_avg_price(order),
        entry_ts=time.time(),
        leverage=bot.cfg.LEVERAGE,
        atr=current_atr,
        confidence=confidence,
    )

    # Koruma planı hesapla
    plan = build_staged_protection_plan(
        requested_qty=0.001,
        filled_qty=filled_qty,
        min_fill_ratio=0.85,
        trailing_enabled=True
    )

    # Acil stop — hemen gönder
    await _maybe_place_stage1_emergency_stop(bot, "BTCUSDT", plan)

    # State persist
    await bot.state.save()
```

### 11.6 Position Manager — Stop/TP Yönetimi

```
position_manager_loop(bot):

└─ DÖNGÜ:
   └─ Her açık pozisyon için:
      ├─ exchange.fetch_open_positions(symbol)
      │  → Gerçek pozisyon boyutunu doğrula
      │
      ├─ Per-symbol lock al
      │  (reconcile ile paylaşılır — shared_locks.py)
      │
      ├─ compute_exit_profile(bot, symbol, position, price)
      │  → stop_price, tp_price, trailing_params hesapla
      │
      ├─ assess_stop_coverage(bot, symbol)
      │  → Exchange'de aktif stop emri var mı?
      │
      ├─ should_refresh_protection()
      │  → Fiyat değişti, stop güncellemesi gerekiyor mu?
      │
      └─ [Refresh gerekli ise]:
         ├─ place_stop_ladder_router(bot, symbol, ...)
         │  → create_order(intent_reduce_only=True) ← Exit! Her zaman geçer
         │
         └─ place_trailing_router(bot, symbol, ...)
            → create_order(intent_reduce_only=True)
```

### 11.7 Reconcile — Reality Sync

```
reconcile_tick(bot):

├─ exchange.fetch_open_orders(symbol) → Gerçek emirler
├─ exchange.fetch_open_positions(symbol) → Gerçek pozisyonlar

├─ Orphan detection:
│  Exchange'de var ama brain'de yok → adopt et
│  Brain'de var ama exchange'de yok → temizle

├─ Equity refresh:
│  exchange.fetch_balance() → USDT balance
│  bot.state.update_equity(balance)

├─ Stop coverage kontrolü:
│  Her pozisyon için stop emri yoksa → place_stop_ladder_router()

└─ Belief controller tick (opsiyonel):
   Ağ kalitesi + veri kalitesi → guard_knobs güncelle
```

### 11.8 Graceful Shutdown

```
Ctrl+C basıldı (SIGINT):
├─ bot._shutdown.set()  ← Cooperative shutdown event

Tüm async task'lar:
├─ while not bot._shutdown.is_set(): ...
└─ Döngüden çık

bootstrap.py:
├─ bot._cancel_all_tasks()
├─ await bot.state.save()  ← Final state persist
│  → ~/.blade_eternal.brain.lz4
├─ Telegram: notify_shutdown()
└─ exchange.close()  ← CCXT websocket kapat

Exit code: 130 (SIGINT standard)
```

---

## 12. Gelecek Geliştirme Alanları

### 12.1 Reliability (Güvenilirlik)

#### 12.1.1 WAL Intent Temizleme

Şu an `run_context['wal_intents']` sınırsız büyüyebilir. Eski, tamamlanmış intent'lerin TTL-tabanlı silinmesi gerekiyor:

```python
# Önerilen: 24 saat üzerinden purge
WAL_INTENT_TTL_SEC = 86400

def cleanup_wal_intents(state):
    now = time.time()
    state.run_context['wal_intents'] = {
        k: v for k, v in state.run_context.get('wal_intents', {}).items()
        if now - v.get('ts', 0) < WAL_INTENT_TTL_SEC
    }
```

#### 12.1.2 Multi-Instance Koruması

Singleton lock şu an `state/locks/execution_bootstrap.lock` dosya bazlı. Güçlendirme için:
- Docker container ortamı için process namespace lock
- Cloud deployment için distributed lock (Redis SETNX)

#### 12.1.3 Heartbeat Monitoring

`guardian_loop` şu an internal. Dış monitoring için:
- HTTP health endpoint (`/health`, `/metrics`)
- Prometheus metrics export
- PagerDuty / OpsGenie integration

### 12.2 Signal Quality (Sinyal Kalitesi)

#### 12.2.1 Feature Engineering Genişletme

`features/` klasörü mevcut ama sınırlı. Eklenebilecekler:

```python
# Piyasa mikrostrüktürü özellikleri
- Order book imbalance (bid/ask ratio)
- Trade intensity (son N saniyedeki işlem hacmi)
- Spread genişliği
- VWAP sapması

# Zaman bazlı özellikler
- Gün içi saat etkisi (Asian/European/US session)
- Haber saati buffer (yüksek volatilite öncesi çekilme)

# Cross-asset
- BTC/ETH korelasyonu
- Crypto fear & greed index
```

#### 12.2.2 ML-Based Signal Filtering

```python
# Mevcut: kural bazlı sinyal
# Gelecek: hafif ML filtre
from sklearn.ensemble import GradientBoostingClassifier

class SignalFilter:
    def should_trade(self, features: dict) -> float:
        """Confidence'ı ML ile kalibre et."""
        return self.model.predict_proba([feature_vector])[0][1]
```

#### 12.2.3 Walk-Forward Validation

`tools/` klasöründeki backtest araçlarını canlı pipeline'a bağlamak:

```bash
# Haftalık otomatik backtest + validation
python tools/rank_passive_pockets_forward.py \
    --symbol BTCUSDT \
    --lookback-days 21 \
    --fee-mult 1.0 \
    --adv-mult 1.2 \
    --min-attempt-fill-rate 0.40
```

### 12.3 Execution Safety (Execution Güvenliği)

#### 12.3.1 Partial Fill Handling

Şu an kısmi dolum (`filled_qty < requested_qty`) acil stop ile kapatılıyor. Gelişmiş yaklaşım:

```python
# Kısmi dolum durumunda:
# 1. Dolum oranını hesapla
fill_ratio = filled_qty / requested_qty

# 2. Threshold'a göre karar
if fill_ratio >= MIN_FILL_RATIO:  # 0.85
    # Pozisyon kabul et, koruma ayarla
    place_stop_for_partial(filled_qty)
elif fill_ratio >= PARTIAL_ACCEPT_FLOOR:  # 0.50 (yeni)
    # Kısmi pozisyon tut, geri kalanı iptal et
    cancel_remaining_and_protect(filled_qty)
else:
    # Çok küçük dolum, çık
    market_close_partial(filled_qty)
```

#### 12.3.2 Slippage Monitoring

Execution kalitesini ölçmek için:

```python
# Her dolumda slippage'ı kaydet
expected_price = signal_price
actual_price = order_avg_price(order)
slippage_bps = (actual_price - expected_price) / expected_price * 10000

emit(bot, "execution_quality", {
    "symbol": symbol,
    "expected_price": expected_price,
    "actual_price": actual_price,
    "slippage_bps": slippage_bps,
})
```

#### 12.3.3 Adaptive Order Type Selection

Piyasa koşuluna göre limit/market seçimi:

```python
def _choose_order_type(spread_bps, urgency):
    if spread_bps < 2.0 and urgency == "normal":
        return "limit"  # Dar spread → maker fee
    elif urgency == "exit":
        return "market"  # Çıkışta her zaman market
    else:
        return "market"  # Güvende kal
```

### 12.4 Performance (Performans)

#### 12.4.1 WebSocket Stream

Şu an REST polling (`fetch_ohlcv`). WebSocket stream ile:

```python
# ccxt pro / WebSocket
async def ws_ohlcv_stream(bot, symbol):
    async for ohlcv in bot.ex.watch_ohlcv(symbol, "1m"):
        bot.data.ohlcv[symbol]["1m"] = ohlcv
        # Anlık güncelleme → daha hızlı sinyal
```

Latency kazancı: ~300ms → ~10ms (REST polling → WebSocket push)

#### 12.4.2 State Persistence Optimizasyonu

Şu an her kayıtta tam state serialize/compress ediliyor. Incremental diff ile:

```python
# Sadece değişen field'ları kaydet
def save_incremental(self, changed_fields: set):
    partial = {k: getattr(self, k) for k in changed_fields}
    append_to_wal(partial)  # LZ4 compressed append
```

#### 12.4.3 Data Cache Parallel Fetch

```python
# Şu an: Semboller sırayla fetch ediliyor
# Gelecek: Parallel
await asyncio.gather(*[
    fetch_symbol_data(bot, symbol)
    for symbol in bot.active_symbols
])
```

### 12.5 Observability (Gözlemlenebilirlik)

#### 12.5.1 Dashboard

```bash
# Mevcut: JSONL tabanlı dashboard
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl

# Gelecek: Grafana + InfluxDB
# telemetry.jsonl → InfluxDB writer → Grafana dashboard
```

#### 12.5.2 Alerting

```python
# P&L % kayıp alert
if daily_pnl_pct < -0.02:  # -2% günlük
    await telegram.send("UYARI: Günlük kayıp -%2.1f%%" % abs(daily_pnl_pct))

# Execution kalitesi bozulma alert
if recent_slippage_avg > SLIPPAGE_ALERT_BPS:
    await telegram.send("UYARI: Ortalama slippage yüksek: %.1f bps" % avg)
```

#### 12.5.3 Post-Trade Analysis

Otomatik haftalık rapor:

```bash
# Mevcut araçlarla
python tools/rank_passive_pockets_forward.py --output reports/weekly.md
python tools/micro_edge_report.py --last-days 7
```

---

## Appendix A — Kritik ENV Variable Referansı

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `SCALPER_DRY_RUN` | `1` | Simülasyon modu (güvenli) |
| `SCALPER_PAPER_TRADING` | `1` | Paper trading (varsayılan) |
| `SCALPER_LIVE_TRADING` | — | Live arm — açık silah güvenliği |
| `SCALPER_MODE` | `auto` | `auto`, `micro`, `production` |
| `SCALPER_EQUITY` | — | Başlangıç equity override |
| `ACTIVE_SYMBOLS` | `BTCUSDT` | Virgülle ayrılmış semboller |
| `BINANCE_API_KEY` | — | Binance API key |
| `BINANCE_API_SECRET` | — | Binance API secret |
| `TELEGRAM_TOKEN` | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | — | Bildirim chat ID |
| `BOOT_REBUILD_ON_START` | — | Exchange'den pozisyon rebuild |
| `BOOT_MAINTENANCE_ONESHOT` | — | Tek reconcile tick, çık |
| `PAPER_INITIAL_EQUITY` | `1000` | Paper başlangıç bakiyesi (USDT) |
| `PAPER_MIN_CONFIDENCE` | `0.35` | Paper modda düşürülmüş confidence |
| `PAPER_FORCE_ONE_FILL` | — | Smoke test: bir emir aç/kapat |
| `SCALPER_SIGNAL_DIAG` | — | Signal diagnostic logging |
| `SCALPER_DEBUG_LOOSE` | — | Gevşek gate'ler (sadece test) |
| `SCALPER_FORCE_ENTRY_TEST` | — | Zorla entry (plumbing validation) |

---

## Appendix B — Test Komutları

```bash
# Tüm testler
python -m pytest tools/ -v

# Tek test dosyası
python -m pytest tools/test_entry_loop_unit.py -v

# Tek test
python -m pytest tools/test_entry_loop_unit.py::EntryLoopTelemetryTests::test_recent_router_blocks_counts -v

# Test kategorisi
python -m pytest tools/ -k "kill_switch" -v

# Hızlı kontrol (quiet)
python -m pytest tools/ -q

# Reliability gate
python tools/reliability_gate.py
```

---

## Appendix C — Hızlı Referans: Kritik Kurallar

```
1. GUARDIAN-SAFE: Tüm async execution fonksiyonları try/except ile sarılır,
   dışarı asla raise etmez.

2. EXIT ALWAYS PASSES: intent_reduce_only=True olan emirlere hiçbir
   entry kısıtlaması uygulanmaz. Kill switch bile geçer.

3. EXCHANGE IS TRUTH: bot.state pozisyon bilgisi "inanç"tır.
   Exchange'deki durum her zaman gerçektir.

4. SYMKEY SOT: Symbol normalizasyonu için tek kaynak
   execution/entry_primitives.py:symkey()

5. BOUNDED RETRY: Sonsuz retry loop yasaktır.
   Her retry bounded olmalı ve hata sınıflandırmalı çalışmalıdır.

6. HEDGE MODE: Binance Futures'da positionSide (LONG/SHORT)
   her order'da gereklidir.

7. CLIENT ORDER ID: Binance'e gönderilen clientOrderId < 36 karakter
   olmalıdır. router.py otomatik hash/truncate yapar.

8. LOCKS: shared_locks.py üzerinden per-symbol lock kullan.
   Aynı anda 1'den fazla lock tutma (deadlock riski).

9. CONFIG PRIORITY: ENV variable → bot.cfg → hardcoded default
   (bu sırayla override edilir, main.py otoritedir).

10. TELEMETRY: Her önemli olay emit() ile loglanır.
    Telemetri sistemi yoksa trading sessizce devam eder.
```

---

*Bu doküman `docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` olarak kaydedilmiştir.*
*Repo analiz tarihi: 2026-02-20*
*Referans branch: feat/reliability-gate-automation*
