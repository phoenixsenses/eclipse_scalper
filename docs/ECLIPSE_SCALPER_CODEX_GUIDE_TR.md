# Eclipse Scalper â€” KapsamlÄ± Teknik Referans DokÃ¼mani (TÃ¼rkÃ§e)

> **Hedef Kitle:** Projeyi ilk kez inceleyen, yazÄ±lÄ±m geliÅŸtirme ve sistem mimarisi bilen geliÅŸtiriciler.
> **Kapsam:** Sistemin amacÄ±, mimarisi, her bileÅŸenin teknik detayÄ±, Codex kullanÄ±mÄ± ve geliÅŸtirici workflow'u.
> **Son gÃ¼ncelleme:** 2026-02-20

---

## Ä°Ã§indekiler

1. [Eclipse Scalper Nedir](#1-eclipse-scalper-nedir)
2. [Sistem Mimarisi Genel BakÄ±ÅŸ](#2-sistem-mimarisi-genel-bakÄ±ÅŸ)
3. [Execution Pipeline DetaylÄ± AÃ§Ä±klama](#3-execution-pipeline-detaylÄ±-aÃ§Ä±klama)
4. [Signal Generation Sistemi](#4-signal-generation-sistemi)
5. [Risk Management Sistemi](#5-risk-management-sistemi)
6. [Brain ve State Sistemi](#6-brain-ve-state-sistemi)
7. [Order Execution Sistemi](#7-order-execution-sistemi)
8. [Codex Nedir ve Bu Projeye NasÄ±l YardÄ±mcÄ± Olur](#8-codex-nedir-ve-bu-projeye-nasÄ±l-yardÄ±mcÄ±-olur)
9. [Codex CLI KullanÄ±m Rehberi](#9-codex-cli-kullanÄ±m-rehberi)
10. [GeliÅŸtirici Workflow](#10-geliÅŸtirici-workflow)
11. [GerÃ§ek Execution Flow â€” Derin Teknik Analiz](#11-gerÃ§ek-execution-flow--derin-teknik-analiz)
12. [Gelecek GeliÅŸtirme AlanlarÄ±](#12-gelecek-geliÅŸtirme-alanlarÄ±)

---

## 1. Eclipse Scalper Nedir

### 1.1 Projenin Genel AmacÄ±

Eclipse Scalper, Binance USD-margined futures piyasalarÄ±nda otomatik kripto para alÄ±m-satÄ±mÄ± yapan, **tamamen asenkron Python** ile yazÄ±lmÄ±ÅŸ geliÅŸmiÅŸ bir trading botudur. AmaÃ§; kÄ±sa vadeli fiyat hareketlerinden (scalp) tutarlÄ±, kÃ¼Ã§Ã¼k ama tekrarlanabilir kar elde etmektir.

Sistem ÅŸu soruyu yanÄ±tlamak iÃ§in inÅŸa edilmiÅŸtir:

> *"GerÃ§ek piyasa koÅŸullarÄ±nda, execution kalitesi bozulmadan ve risk kontrol mekanizmalarÄ± asla devre dÄ±ÅŸÄ± bÄ±rakÄ±lmadan sÃ¼rdÃ¼rÃ¼lebilir bir scalping edge var mÄ±?"*

### 1.2 Hangi TÃ¼r Bir Trading Bot?

| Ã–zellik | Detay |
|---|---|
| **Piyasa** | Binance USDM Futures (hedge mode) |
| **EnstrÃ¼man** | BTC/USDT, ETH/USDT ve diÄŸer likit Ã§iftler |
| **Strateji tipi** | Multi-timeframe teknik analiz scalping |
| **Zaman dilimi** | 1dk (primary), 5dk + 15dk (confirmation) |
| **KaldÄ±raÃ§** | 20x (production) / 35x (micro hesap) |
| **Execution** | ccxt async â€” CCXT kÃ¼tÃ¼phanesi Ã¼zerinden |
| **Mod** | Paper trading (varsayÄ±lan) veya live trading |

### 1.3 Ã‡Ã¶zdÃ¼ÄŸÃ¼ Teknik Problemler

#### Problem 1 â€” Durum TutarsÄ±zlÄ±ÄŸÄ± (State Inconsistency)

Bir trading botunun en kritik sorunu: botun kendi kafasÄ±ndaki "pozisyon var" ile gerÃ§ek exchange'deki durum arasÄ±ndaki fark. Eclipse Scalper bunu **reconciliation-first mimari** ile Ã§Ã¶zer â€” exchange her zaman gerÃ§eÄŸin tek kaynaÄŸÄ±dÄ±r (single source of truth), bot'un local state'i sadece bir "inanÃ§" (belief) olarak kabul edilir.

#### Problem 2 â€” Yeniden BaÅŸlatma GÃ¼venliÄŸi (Restart Safety)

Bot kapanÄ±p yeniden aÃ§Ä±ldÄ±ÄŸÄ±nda, aÃ§Ä±k pozisyonlar kaybolmamalÄ±, stop/TP emirleri tekrar gÃ¶nderilmemeli, aynÄ± emir iki kez submit edilmemelidir. Bu; WAL (Write-Ahead Log) intent sistemi, idempotent order ID'ler ve bootstrap'ta exchange'den state rebuild mekanizmasÄ± ile Ã§Ã¶zÃ¼lÃ¼r.

#### Problem 3 â€” Bounded Retry ve Kill-Switch

AÄŸ hatasÄ±, API rate limit veya stale data durumunda bot sonsuz dÃ¶ngÃ¼ye girmemeli, paniÄŸe kapÄ±lmamalÄ±dÄ±r. Kill-switch sistemi, belirli eÅŸikleri aÅŸtÄ±ÄŸÄ±nda trading'i durdurur; router retry mekanizmasÄ± hatalarÄ± sÄ±nÄ±flandÄ±rÄ±r (retryable, fatal, idempotent_safe) ve bounded ÅŸekilde yeniden dener.

#### Problem 4 â€” Windows Async UyumluluÄŸu

Binance WebSocket + ccxt async kombinasyonu Windows'ta `ProactorEventLoop` ile Ã§alÄ±ÅŸmaz. Sistem, `SelectorEventLoop` politikasÄ±nÄ± zorlar ve tÃ¼m pipe/stdout UTF-8 uyumsuzluklarÄ±nÄ± bootstrap'ta Ã§Ã¶zer.

### 1.4 Neden Bu Mimari?

Sistem bir **distributed execution system** olarak tasarlanmÄ±ÅŸtÄ±r â€” yani her modÃ¼l baÄŸÄ±msÄ±z Ã§alÄ±ÅŸabilir, birinin Ã§Ã¶kmesi diÄŸerini durdurmamalÄ±dÄ±r. Bu yÃ¼zden:

- Her async fonksiyon "guardian-safe" â€” iÃ§eride hata yakalar, dÄ±ÅŸarÄ± asla raise etmez
- TÃ¼m kritik operasyonlar telemetri Ã¼retir (JSONL format)
- ModÃ¼ller arasÄ± iletiÅŸim shared `bot` objesi Ã¼zerinden, lock ile serialize edilmiÅŸ
- Config ENVironmen Variable'larla override edilebilir â€” deploy zamanÄ± yeniden derleme gerekmez

---

## 2. Sistem Mimarisi Genel BakÄ±ÅŸ

### 2.1 KlasÃ¶r YapÄ±sÄ±

```
eclipse_scalper/
â”œâ”€â”€ bot/                    # Bot Ã§ekirdeÄŸi â€” EclipseEternal sÄ±nÄ±fÄ±
â”œâ”€â”€ brain/                  # State persistence â€” PsycheState, Position
â”œâ”€â”€ config/                 # Config ÅŸemalarÄ± â€” Config, MicroConfig
â”œâ”€â”€ data/                   # Market data layer â€” DataCache
â”œâ”€â”€ exchanges/              # Exchange adaptÃ¶rleri â€” Binance, Paper Trading
â”œâ”€â”€ execution/              # TÃ¼m order execution mantÄ±ÄŸÄ±
â”‚   â”œâ”€â”€ bootstrap.py        # TEK baÅŸlangÄ±Ã§ noktasÄ± â€” singleton orchestrator
â”‚   â”œâ”€â”€ entry_loop.py       # Ana entry signal dÃ¶ngÃ¼sÃ¼ â€” 20+ gate check
â”‚   â”œâ”€â”€ entry_primitives.py # Ortak primitifler â€” symkey() SOT
â”‚   â”œâ”€â”€ order_router.py     # Order submission + retry + idempotency
â”‚   â”œâ”€â”€ position_manager.py # Stop/TP yÃ¶netimi
â”‚   â”œâ”€â”€ reconcile.py        # Exchange â†” local state senkronizasyonu
â”‚   â”œâ”€â”€ shared_locks.py     # Per-symbol asyncio lock'larÄ±
â”‚   â””â”€â”€ telemetry.py        # JSONL event logging
â”œâ”€â”€ features/               # Feature engineering (teknik indikatÃ¶rler)
â”œâ”€â”€ integrations/           # Telegram notifier + control
â”œâ”€â”€ policies/               # Trading policy katmanÄ±
â”œâ”€â”€ risk/                   # Risk yÃ¶netimi â€” RiskManager, KillSwitch
â”œâ”€â”€ strategies/             # Sinyal Ã¼retimi â€” EclipseScalper stratejisi
â”œâ”€â”€ utils/                  # Loglama, yardÄ±mcÄ± fonksiyonlar
â”œâ”€â”€ state/                  # Runtime state dosyalarÄ± (lock, pid, json)
â”œâ”€â”€ logs/                   # Telemetri log'larÄ± (telemetry.jsonl)
â”œâ”€â”€ reports/                # Analiz raporlarÄ± (backtest, ranking, vs.)
â”œâ”€â”€ tools/                  # Test dosyalarÄ± + diagnostic araÃ§lar
â”œâ”€â”€ docs/                   # Bu dokÃ¼man dahil tÃ¼m dokÃ¼mantasyon
â”œâ”€â”€ main.py                 # CLI giriÅŸ noktasÄ± (--dry-run, --equity, --mode)
â””â”€â”€ pytest.ini              # Test konfigÃ¼rasyonu
```

### 2.2 KlasÃ¶r DetaylarÄ±

#### `execution/` â€” Sistemin Kalbi

En kritik klasÃ¶rdÃ¼r. TÃ¼m order yaÅŸam dÃ¶ngÃ¼sÃ¼ burada yÃ¶netilir.

| Dosya | SatÄ±r | RolÃ¼ |
|---|---|---|
| `bootstrap.py` | ~1846 | Singleton sistem baÅŸlatÄ±cÄ±sÄ± |
| `entry_loop.py` | ~200+ | Signal'dan order'a ana dÃ¶ngÃ¼ |
| `entry_primitives.py` | ~114 | PaylaÅŸÄ±lan primitifler (symkey SOT) |
| `order_router.py` | ~150+ | Order gÃ¶nderme, retry, idempotency |
| `position_manager.py` | ~150+ | Stop/TP koruma yÃ¶netimi |
| `reconcile.py` | ~150+ | Exchange reality sync |
| `shared_locks.py` | ~35 | Per-symbol lock'lar |
| `telemetry.py` | â€” | JSONL event sistemi |
| `error_codes.py` | â€” | Standart hata sÄ±nÄ±flarÄ± |

#### `strategies/` â€” Signal Ãœretimi

```
strategies/
â””â”€â”€ eclipse_scalper.py     # Ana strateji â€” scalper_signal() fonksiyonu
```

Multi-timeframe teknik analiz ile long/short sinyali ve confidence deÄŸeri Ã¼retir. ATR, Bollinger Bands, Stochastic Oscillator, ADX kullanÄ±r.

#### `risk/` â€” GÃ¼venlik KatmanÄ±

```
risk/
â”œâ”€â”€ risk_manager.py        # Pozisyon aÃ§ma veto'su (advisory only)
â””â”€â”€ kill_switch.py         # Acil durdurma (data quality / API errors)
```

Risk Manager sadece "hayÄ±r" diyebilir â€” asla emir gÃ¶ndermez. Kill Switch, veri kalitesi bozulduÄŸunda tÃ¼m entry'leri durdurur.

#### `brain/` â€” HafÄ±za Sistemi

```
brain/
â””â”€â”€ state.py              # PsycheState + Position dataclass'larÄ±
```

TÃ¼m bot state'i `PsycheState` dataclass'Ä±nda tutulur ve LZ4-compressed binary olarak `~/.blade_eternal.brain.lz4` dosyasÄ±na yazÄ±lÄ±r. Restart sonrasÄ± kaldÄ±ÄŸÄ± yerden devam eder.

#### `bot/` â€” Orchestration Core

```
bot/
â”œâ”€â”€ core.py               # EclipseEternal â€” tÃ¼m fonksiyonlara geÃ§ilen bot objesi
â””â”€â”€ runner.py             # run_bot() â€” asyncio event loop'u baÅŸlatÄ±r
```

`EclipseEternal` bot objesi ÅŸunlarÄ± taÅŸÄ±r: exchange adapter, state, data cache, config, active symbols, semaphore, shutdown event.

#### `config/` â€” KonfigÃ¼rasyon

```
config/
â””â”€â”€ settings.py           # Config (production) + MicroConfig (kÃ¼Ã§Ã¼k hesap)
```

TÃ¼m parametreler burada tanÄ±mlÄ±dÄ±r. ENV variable'lar `main.py` tarafÄ±ndan override edilir.

#### `data/` â€” Piyasa Verisi

```
data/
â”œâ”€â”€ cache.py              # DataCache / GodEmperorDataOracle
â””â”€â”€ event_diary.py        # Signal kalitesi iÃ§in olay gÃ¼nlÃ¼ÄŸÃ¼
```

OHLCV bar'larÄ± ve anlÄ±k fiyatlar cache'de tutulur. `bot.data_ready` event'i, ilk veri gelene kadar entry_loop'u bekletir.

#### `exchanges/` â€” Exchange AdaptÃ¶rleri

```
exchanges/
â”œâ”€â”€ binance.py            # CCXT async Binance Futures wrapper
â””â”€â”€ paper_trading.py      # SimÃ¼latif trading adapter (dry-run)
```

Her iki adapter aynÄ± interface'i implemente eder: `create_order`, `cancel_order`, `fetch_open_orders`, `fetch_open_positions`, `fetch_balance`.

#### `integrations/` â€” DÄ±ÅŸ Servisler

```
integrations/
â”œâ”€â”€ telegram_notifier.py  # Bildirimler (startup, entry, exit, error)
â””â”€â”€ telegram_control.py   # Uzaktan komut kontrolÃ¼
```

#### `tools/` â€” Test ve Diagnostic

100+ test ve araÃ§ dosyasÄ± iÃ§erir. Unit testler `test_*_unit.py` pattern'ini takip eder, pytest ile Ã§alÄ±ÅŸÄ±r.

#### `state/` â€” Runtime State DosyalarÄ±

```
state/
â”œâ”€â”€ locks/execution_bootstrap.lock   # Singleton bootstrap lock
â”œâ”€â”€ locks/execution_bootstrap.pid    # Bootstrap PID
â”œâ”€â”€ micro_edge_gates.json            # MikrostrÃ¼ktÃ¼r gate konfigÃ¼rasyonu
â”œâ”€â”€ passive_realistic_profiles.json  # Pasif execution profilleri
â””â”€â”€ paper_scoreboard.json            # Paper trading skor tablosu
```

#### `logs/` â€” Telemetri

```
logs/
â””â”€â”€ telemetry.jsonl       # TÃ¼m order/entry/exit/error olaylarÄ±
```

---

## 3. Execution Pipeline DetaylÄ± AÃ§Ä±klama

### 3.1 Sistem NasÄ±l BaÅŸlatÄ±lÄ±r?

#### AdÄ±m 1 â€” CLI GiriÅŸ NoktasÄ± (`main.py`)

```bash
# Dry-run (gÃ¼venli simÃ¼lasyon)
python main.py --dry-run

# GerÃ§ek hesap â€” live arm zorunlu
SCALPER_LIVE_TRADING=1 python main.py

# Micro hesap modu
python main.py --mode micro --equity 50
```

`main.py` v4.2 HARDENED ÅŸu iÅŸlemleri yapar:

```python
# Windows iÃ§in asyncio politikasÄ±
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Komut satÄ±rÄ± argÃ¼manlarÄ±
parser.add_argument("--dry-run")
parser.add_argument("--equity")
parser.add_argument("--mode", choices=["auto", "micro", "production"])

# Ana bot runner'a devreder
bot.runner.run_bot(dry_run=..., equity=..., mode=...)
```

#### AdÄ±m 2 â€” Bootstrap Singleton (`execution/bootstrap.py`)

`bootstrap.py`, v1.8 ONE TRUE ENTRYPOINT â€” tÃ¼m sistemin tek ve otoriteli baÅŸlangÄ±Ã§ noktasÄ±dÄ±r. Birden fazla instance Ã§alÄ±ÅŸmasÄ±nÄ± `_BootstrapSingleton` ile engeller (singleton lock + PID file).

Bootstrap sÄ±rasÄ± ÅŸu ÅŸekildedir:

```
1. UTF-8 hardening (Windows pipe uyumluluÄŸu)
2. .env yÃ¼kle (BINANCE_API_KEY, BINANCE_API_SECRET)
3. Singleton lock al â†’ state/locks/execution_bootstrap.lock
4. Config yÃ¼kle (settings.py â†’ ENV override)
5. Exchange init (CCXT async â€” Binance Futures)
6. Brain state yÃ¼kle ve doÄŸrula (LZ4 decompress)
7. Binance health gate (aÄŸ + auth kontrolÃ¼)
8. Paper trading setup (paper mode aktifse)
9. Risk manager init (optional â€” yoksa devam)
10. Telegram notifier baÅŸlat (optional)
11. Data cache init + market bootstrap
12. Exchange'den state rebuild (BOOT_REBUILD_ON_START=1 ise)
13. Async task loop baÅŸlat:
    â”œâ”€ guardian_loop          (health check)
    â”œâ”€ data_loop              (piyasa verisi toplama)
    â”œâ”€ position_manager_loop  (stop/TP yÃ¶netimi)
    â”œâ”€ exit_loop              (opsiyonel Ã§Ä±kÄ±ÅŸ dÃ¶ngÃ¼sÃ¼)
    â””â”€ entry_loop             (data_ready event'i bekler)
14. Paper smoke test (PAPER_FORCE_ONE_FILL=1 ise)
15. Graceful shutdown â†’ state persist
```

**Kritik ENV Variable'lar:**

| DeÄŸiÅŸken | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `SCALPER_DRY_RUN` | `1` | SimÃ¼lasyon modu |
| `SCALPER_PAPER_TRADING` | `1` | Paper trading |
| `SCALPER_LIVE_TRADING` | â€” | Live arm (aÃ§Ä±k silah gÃ¼venliÄŸi) |
| `ACTIVE_SYMBOLS` | `BTCUSDT` | Hangi semboller iÅŸlem gÃ¶recek |
| `BOOT_REBUILD_ON_START` | â€” | Exchange'den pozisyon rebuild |
| `BOOT_MAINTENANCE_ONESHOT` | â€” | Tek reconcile tick, Ã§Ä±k |
| `PAPER_INITIAL_EQUITY` | `1000` | Paper baÅŸlangÄ±Ã§ bakiyesi |

### 3.2 ModÃ¼ller NasÄ±l YÃ¼klenir?

TÃ¼m modÃ¼ller best-effort (hata toleranslÄ±) import ile yÃ¼klenir:

```python
# Opsiyonel modÃ¼llerin gÃ¼venli import'u
try:
    from execution.telemetry import emit
except Exception:
    emit = None  # Telemetri yoksa sessizce devam

try:
    from execution.reliability_gate_runtime import gate_check
except Exception:
    gate_check = None
```

Bu pattern, eksik veya bozuk bir modÃ¼lÃ¼n tÃ¼m sistemi Ã§Ã¶kertmesini Ã¶nler.

### 3.3 Execution Flow Ã–zeti

```
Veri Alma â†’ Signal Ãœretme â†’ Risk Kontrol â†’ Order GÃ¶nderme â†’ State Update
```

Her adÄ±m detaylÄ± olarak sonraki bÃ¶lÃ¼mlerde aÃ§Ä±klanmaktadÄ±r.

---

## 4. Signal Generation Sistemi

### 4.1 Strateji DosyasÄ±

**Dosya:** `strategies/eclipse_scalper.py` (v3.0 â€” COSMIC SIGNAL ASCENDANT)

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
        confidence (float): 0.0 â€“ 1.0 arasÄ± gÃ¼ven skoru
    """
```

Sadece ikisinden biri True olabilir â€” aynÄ± anda hem long hem short sinyali mÃ¼mkÃ¼n deÄŸildir.

### 4.3 Teknik Ä°ndikatÃ¶rler

| Ä°ndikatÃ¶r | AmaÃ§ |
|---|---|
| **ATR** (Average True Range) | Volatilite Ã¶lÃ§Ã¼mÃ¼ â€” stop mesafesi hesabÄ± |
| **Bollinger Bands** | Fiyat aralÄ±ÄŸÄ± tanÄ±mÄ± â€” squeeze + breakout tespiti |
| **Stochastic Oscillator** | Momentum Ã¶lÃ§Ã¼mÃ¼ â€” aÅŸÄ±rÄ± alÄ±m/satÄ±m bÃ¶lgeleri |
| **ADX** (Average Directional Index) | Trend gÃ¼cÃ¼ â€” sadece gÃ¼Ã§lÃ¼ trend dÃ¶nemlerinde iÅŸlem |
| **SciPy find_peaks** | Lokal zirve/dip tespiti (opsiyonel, import ile) |

### 4.4 Multi-Timeframe MantÄ±ÄŸÄ±

Sistem Ã¼Ã§ zaman dilimini birlikte deÄŸerlendirir:

```
1m (Primary)   â†’ Ä°ÅŸlem zamanlamasÄ± ve anlÄ±k momentum
5m (Confirm)   â†’ KÄ±sa vadeli trend onayÄ±
15m (Context)  â†’ Orta vadeli piyasa yapÄ±sÄ±
```

15dk trend'e karÅŸÄ± 1dk'da iÅŸlem aÃ§Ä±lmaz. Her Ã¼Ã§ timeframe uyumlu olmalÄ±dÄ±r.

### 4.5 Confidence HesabÄ±

Confidence deÄŸeri, kaÃ§ tane teknik koÅŸulun saÄŸlandÄ±ÄŸÄ±na gÃ¶re 0.0â€“1.0 arasÄ± Ã¼retilir:

```python
MIN_CONFIDENCE = 0.72           # Production: minimum 72%
MIN_CONFIDENCE_HIGH_VOL = 0.65  # YÃ¼ksek volatilite dÃ¶neminde: 65%
MIN_CONFIDENCE_MICRO = 0.35     # Micro mod (kÃ¼Ã§Ã¼k hesap): 35%
```

### 4.6 Signal Lifecycle

```
1. Data Quality Check
   â””â”€ staleness_check() â†’ veri tazeliÄŸi doÄŸrulanÄ±r
   â””â”€ min bar count â†’ yeterli OHLCV verisi var mÄ±?

2. Multi-Timeframe Indicator Computation
   â””â”€ 1m / 5m / 15m DataFrame'leri hesaplanÄ±r

3. Gate Conditions Evaluation
   â””â”€ ATR > min_atr? (pozisyon bÃ¼yÃ¼klÃ¼ÄŸÃ¼ kontrollÃ¼)
   â””â”€ ADX > threshold? (trend gÃ¼cÃ¼ yeterli?)
   â””â”€ Bollinger + Stochastic confluence?

4. Confidence Score Aggregation
   â””â”€ KaÃ§ gate geÃ§ti? â†’ confidence hesabÄ±

5. Signal Decision
   â””â”€ confidence >= MIN_CONFIDENCE?
   â””â”€ is_long veya is_short â†’ entry_loop'a ilet

6. Debug Output (opsiyonel)
   â””â”€ SCALPER_SIGNAL_DIAG=1 â†’ throttled logging
   â””â”€ SCALPER_DEBUG_LOOSE=1 â†’ gevÅŸek gate'ler (test amaÃ§lÄ±)
   â””â”€ SCALPER_FORCE_ENTRY_TEST=1 â†’ zorla entry (plumbing validation)
```

### 4.7 Veri Kalitesi KorumasÄ±

```python
# Veri eskimiÅŸse iÅŸlem yok
if data_age_sec > KILL_MAX_DATA_STALENESS_SEC:  # 150 saniye
    return False, False, 0.0

# Yetersiz bar sayÄ±sÄ± ise iÅŸlem yok
if len(df_1m) < MIN_BARS_REQUIRED:
    return False, False, 0.0
```

---

## 5. Risk Management Sistemi

### 5.1 Ä°ki KatmanlÄ± Risk KontrolÃ¼

Eclipse Scalper'da risk yÃ¶netimi iki ayrÄ± katmandan oluÅŸur ve ikisi birbirinden baÄŸÄ±msÄ±z Ã§alÄ±ÅŸÄ±r:

| Katman | Dosya | Sorumluluk |
|---|---|---|
| **RiskManager** | `risk/risk_manager.py` | Pozisyon bÃ¼yÃ¼klÃ¼ÄŸÃ¼, kayÄ±p limitleri, concurrent limit |
| **KillSwitch** | `risk/kill_switch.py` | Data quality / API error bazlÄ± acil durdurma |

### 5.2 RiskManager â€” Veto Sistemi

**Ã–nemli:** RiskManager sadece danÄ±ÅŸmanlÄ±k (advisory) rolÃ¼ndedir. Asla emir gÃ¶ndermez â€” sadece "hayÄ±r" diyebilir.

```python
class RiskManager:
    def can_open_position(
        self,
        symbol: str,
        side: str,         # 'long' veya 'short'
        size_usd: float,   # Pozisyon bÃ¼yÃ¼klÃ¼ÄŸÃ¼ (USD)
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Returns:
            (True, "") â†’ Pozisyon aÃ§Ä±labilir
            (False, "Reason") â†’ Reddedildi, sebep aÃ§Ä±klanÄ±r
        """
```

**Kontrol Edilen Limitler:**

| Limit | Production | Micro |
|---|---|---|
| GÃ¼nlÃ¼k kayÄ±p limiti | %3 | %3 |
| HaftalÄ±k kayÄ±p limiti | %7 | %7 |
| Maksimum drawdown | %15 | %15 |
| Pozisyon baÅŸÄ±na max | %5 equity | %5 equity |
| EÅŸzamanlÄ± pozisyon | 6 | 1 |
| Ä°ÅŸlemler arasÄ± cooldown | 300 saniye | 300 saniye |

**State Persistence:** RiskManager SQLite veritabanÄ± kullanÄ±r (`data/risk_state.db`). Restart sonrasÄ± gÃ¼nlÃ¼k/haftalÄ±k kayÄ±p sayacÄ± sÄ±fÄ±rlanmaz.

### 5.3 Kill Switch â€” Acil Durdurma

`risk/kill_switch.py` v1.0; veri kalitesi veya API hata oranÄ± eÅŸiÄŸi aÅŸÄ±ldÄ±ÄŸÄ±nda tÃ¼m entry'leri durdurur.

**Tetikleyici KoÅŸullar:**

```python
# Veri eskimesi
if data_age_sec > KILL_MAX_DATA_STALENESS_SEC:  # 150s
    request_halt(bot, "stale_data")

# API hata oranÄ±
if api_error_rate > KILL_MAX_API_ERROR_RATE:    # %35
    request_halt(bot, "api_error_rate")

# ArdÄ±ÅŸÄ±k API hatasÄ±
if consecutive_errors > KILL_MAX_API_ERROR_BURST:  # 12
    request_halt(bot, "api_error_burst")
```

**Escalation Seviyeleri:**

```
1. Normal Halt â†’ Entry durdurulur, exitler devam eder
2. Trip SayacÄ± â†’ Her halt bir sayaÃ§ artÄ±rÄ±r
3. Emergency Flatten â†’ KILL_ESCALATE_FLAT_AFTER_TRIPS aÅŸÄ±lÄ±rsa pozisyonlar kapatÄ±lÄ±r
4. Shutdown â†’ KILL_ESCALATE_SHUTDOWN_AFTER_TRIPS aÅŸÄ±lÄ±rsa bot kapanÄ±r

Cooldown: KILL_SWITCH_COOLDOWN_SEC = 300 saniye
(Halt sonrasÄ± 5 dakika geÃ§meden yeniden entry aÃ§Ä±lmaz)
```

**Kritik GÃ¼venlik KuralÄ±:** Kill switch yalnÄ±zca entry'leri etkiler. Exit emirleri (stop, take-profit, trailing) her zaman geÃ§er. AÃ§Ä±k pozisyonlar koruma altÄ±nda kalÄ±r.

### 5.4 Position Sizing

Position bÃ¼yÃ¼klÃ¼ÄŸÃ¼ ÅŸu ÅŸekilde belirlenir:

```python
# Sinyal size Ã¶neriyorsa â†’ onu kullan
notional_usd = signal.get("notional_usd", None)

# Sinyal yoksa â†’ config'den varsayÄ±lan
if notional_usd is None:
    notional_usd = cfg.FIXED_NOTIONAL_USDT  # 25 USDT production, 8 USDT micro

# KaldÄ±raÃ§ ile nominal konum bÃ¼yÃ¼klÃ¼ÄŸÃ¼
position_size = notional_usd * leverage / current_price
```

### 5.5 Stop / Take-Profit YapÄ±sÄ±

```python
# Stop Loss â€” ATR bazlÄ±
stop_distance = atr * STOP_ATR_MULT          # 1.10x ATR
stop_distance = min(stop_distance,
                    entry_price * MAX_STOP_PCT)  # max %3

# Trailing Stop
# Aktivasyon: Entry'den itibaren +1.30x ATR kazanÄ±nca
TRAILING_ACTIVATION_RR = 1.30
# Callback rate: %0.45 geri Ã§ekilmede trailing tetiklenir
TRAILING_CALLBACK_RATE = 45  # bps
```

---

## 6. Brain ve State Sistemi

### 6.1 State Architecture

Bot'un tÃ¼m hafÄ±zasÄ± tek bir `PsycheState` dataclass'Ä±nda tutulur. Bu obje LZ4-compressed binary olarak diske yazÄ±lÄ±r:

```
KayÄ±t yeri: ~/.blade_eternal.brain.lz4
Format: LZ4-compressed pickle/binary
```

### 6.2 PsycheState Veri YapÄ±sÄ±

**`brain/state.py` v4.6 â€” GOD-EMPEROR**

```python
@dataclass
class PsycheState:
    # === Pozisyon YÃ¶netimi ===
    positions: Dict[str, Position]        # canonical_symbol â†’ Position

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

    # === Sembol KÄ±sÄ±tlamalarÄ± ===
    blacklist: Dict[str, float]           # symbol â†’ expiry_ts
    blacklist_reason: Dict[str, str]      # symbol â†’ reason
    consecutive_losses: Dict[str, int]   # symbol â†’ kayÄ±p sayÄ±sÄ±

    # === Timing ===
    last_exit_time: Dict[str, float]      # symbol â†’ son Ã§Ä±kÄ±ÅŸ zamanÄ±

    # === Reconcile Idempotency ===
    known_exit_order_ids: Set[str]        # Duplicate exit Ã¶nleme (max 50k)

    # === Per-Symbol Performans ===
    symbol_performance: Dict[str, Dict]  # pnl, wins, losses, trailing_ids

    # === Signal Kalitesi ===
    entry_confidence_history: Dict[str, List[float]]  # max 200 per symbol

    # === Entry Watch Persistence ===
    entry_watches: Dict[str, Dict]        # max 500 entry

    # === Ephemeral Runtime (Restart'a DayanÄ±r) ===
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
    symbol: str           # Canonical symbol (Ã¶rn. "BTCUSDT")
    side: str             # 'long' veya 'short'
    size: float           # Absolute miktar (BTC, ETH, vb.)
    entry_price: float
    entry_ts: float       # Unix timestamp
    leverage: int
    atr: float            # Entry zamanÄ±ndaki ATR deÄŸeri

    # Stop/TP Durumu
    hard_stop_order_id: Optional[str]
    trailing_active: bool
    breakeven_moved: bool

    # Entry Kalitesi
    confidence: float     # Sinyal gÃ¼ven skoru
```

### 6.4 Symbol Canonicalization (Sembol Normalizasyonu)

TÃ¼m semboller canonical formata dÃ¶nÃ¼ÅŸtÃ¼rÃ¼lÃ¼r. Bu dÃ¶nÃ¼ÅŸÃ¼mÃ¼n tek kaynaÄŸÄ± (SOT) `execution/entry_primitives.py:symkey()` fonksiyonudur:

```python
# TÃ¼m bu formatlar â†’ "BTCUSDT" olur
symkey("BTC/USDT:USDT")  â†’ "BTCUSDT"
symkey("BTC/USDT")       â†’ "BTCUSDT"
symkey("btcusdt")        â†’ "BTCUSDT"
symkey("BTC-USDT")       â†’ "BTCUSDT"
symkey(None)             â†’ ""

# DiÄŸer modÃ¼llerde import
from execution.entry_primitives import symkey as _shared_symkey
```

### 6.5 State Persistence â€” NasÄ±l Ã‡alÄ±ÅŸÄ±r?

```
1. Bootstrap â†’ LZ4 dosyasÄ± varsa yÃ¼kle
   â”œâ”€ from_loaded(obj) â†’ Migration + validation
   â”œâ”€ validate() â†’ Key normalizasyonu, bozuk veri onarÄ±mÄ±, koleksiyon limitleri
   â””â”€ recompute_derived() â†’ win_rate, drawdown yeniden hesap

2. Her Ã¶nemli olay sonrasÄ± â†’ save (async)
   â””â”€ to_dict() â†’ Set'leri list'e, date'leri ISO'ya Ã§evir
   â””â”€ LZ4 compress â†’ diske yaz

3. Graceful shutdown â†’ Final save
   â””â”€ Signal handler (SIGINT, SIGTERM) â†’ state.save()
```

### 6.6 Restart Recovery

```
Bot kapanÄ±r â†’
  state.run_context['wal_intents'] iÃ§inde aÃ§Ä±k intent'ler var

Bot aÃ§Ä±lÄ±r â†’
  bootstrap.py: BOOT_REBUILD_ON_START=1 ise exchange'den pozisyonlar Ã§ekilir
  â†’ Orphan pozisyonlar adopt edilir (reconcile.py)
  â†’ WAL intent'ler kontrol edilir, duplicate submit engellenir
  â†’ known_exit_order_ids ile duplicate exit engellenir
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

v2.4 â€” BINANCE COID<36 + IDEMPOTENT CANCEL

Order Router, tÃ¼m emir gÃ¶nderme ve iptal operasyonlarÄ±nÄ± yÃ¶netir. Temel gÃ¼venlik garantileri:

- **Binance clientOrderId < 36 karakter:** Otomatik hash/truncate
- **Idempotent cancel:** `-2011 "order not found"` baÅŸarÄ± sayÄ±lÄ±r
- **Duplicate handler:** `-4116` hatasÄ± bounded+deduped (max 12 variant)
- **Bounded retry:** Sonsuz dÃ¶ngÃ¼ yok; hata tipi retry stratejisini belirler

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

**Dahili Ä°ÅŸlem SÄ±rasÄ±:**

```
1. clientOrderId oluÅŸtur
   â””â”€ intent + symbol + timestamp hash â†’ max 35 karakter

2. Hedge mode inject
   â””â”€ params['positionSide'] = 'LONG' veya 'SHORT'

3. reduceOnly kontrolÃ¼
   â””â”€ closePosition=True ise amount=0.0, reduceOnly kaldÄ±rÄ±lÄ±r

4. FIRST_LIVE_SAFE kontrolÃ¼
   â””â”€ Ä°lk live iÅŸlem â†’ kÃ¼Ã§Ã¼k notional cap

5. Exchange'e gÃ¶nder (ccxt)

6. Hata sÄ±nÄ±flandÄ±rmasÄ±
   â”œâ”€ Retryable â†’ bounded retry (max retries)
   â”œâ”€ Fatal â†’ anÄ±nda fail
   â”œâ”€ Idempotent safe â†’ baÅŸarÄ± say
   â””â”€ Duplicate (-4116) â†’ variant ID ile yeniden dene (max 12)

7. Telemetri emit
   â””â”€ emit_order_create(event_type, payload)

8. WAL intent ledger gÃ¼ncelle
   â””â”€ state.run_context['wal_intents']
```

### 7.3 Exit Emirleri AyrÄ±calÄ±klÄ±

**Kritik kural:** `intent_reduce_only=True` olan emirlere hiÃ§bir entry kÄ±sÄ±tlamasÄ± uygulanmaz:

```python
# Kill switch, circuit breaker, reliability gate...
# SADECE entry emirleri iÃ§in geÃ§erlidir.
# Stop, TP, trailing emirleri her zaman geÃ§er.
if intent_reduce_only:
    skip_all_entry_gates()
    proceed_to_submit()
```

Bu kural, aÃ§Ä±k pozisyonlarÄ±n korumasÄ±z kalmasÄ±nÄ± Ã¶nler.

### 7.4 cancel_order() â€” Idempotent Ä°ptal

```python
async def cancel_order(
    bot,
    symbol: str,
    order_id: str,
    ...
) -> dict:
```

Binance `-2011` hatasÄ± (order not found) baÅŸarÄ± olarak kabul edilir. Bu, `position_manager_loop` veya `reconcile`'Ä±n "zaten iptal edilmiÅŸ" emirleri tekrar tekrar iptal etmeye Ã§alÄ±ÅŸmamasÄ±nÄ± saÄŸlar.

### 7.5 Hedge Mode DesteÄŸi

Binance Futures hedge mode'da her pozisyon `positionSide` gerektirir:

```python
# Long pozisyon iÃ§in buy emri
params['positionSide'] = 'LONG'

# Short pozisyon iÃ§in sell emri
params['positionSide'] = 'SHORT'

# Long pozisyon kapatmak iÃ§in sell emri
params['positionSide'] = 'LONG'
params['reduceOnly'] = True
```

### 7.6 Entry Loop â€” 20+ Gate KontrolÃ¼

**`execution/entry_loop.py` v1.6 â€” AUTHORITATIVE ENTRY ORCHESTRATOR**

Entry Loop, signal'dan order submission'a giden yolda 20+ kontrol noktasÄ± uygular:

```
Gate 1:  entries_allowed()           â†’ Genel giriÅŸ izni
Gate 2:  trade_allowed() (kill sw.)  â†’ Kill switch kontrolÃ¼
Gate 3:  anomaly_should_pause()      â†’ Anomali tespiti
Gate 4:  staleness_check()           â†’ Veri tazeliÄŸi
Gate 5:  reliability_gate_runtime    â†’ Runtime gÃ¼venilirlik kapÄ±sÄ±
Gate 6:  compute_entry_decision()    â†’ Sinyal + belief state deÄŸerlendirme
Gate 7:  RiskManager.can_open()      â†’ Risk veto kontrolÃ¼
Gate 8:  Per-symbol pending lock     â†’ AynÄ± sembol iÃ§in concurrent entry Ã¶nleme
Gate 9:  ENTRY_LOCAL_COOLDOWN_SEC    â†’ Submit sonrasÄ± cooldown (8s)
Gate 10: ENTRY_PER_SYMBOL_GAP_SEC   â†’ Semboller arasÄ± throttle (2.5s)
         ... (circuit breaker, confidence threshold, vs.)
Gate N:  create_order()              â†’ Emir gÃ¶nder
```

**Throttle MekanizmasÄ±:**

```python
# Sembol baÅŸÄ±na throttle
ENTRY_PER_SYMBOL_GAP_SEC = 2.5  # saniye

# Her submit sonrasÄ± cooldown (baÅŸarÄ±lÄ± veya baÅŸarÄ±sÄ±z)
ENTRY_LOCAL_COOLDOWN_SEC = 8.0  # saniye

# DÃ¶ngÃ¼ frekansÄ±
ENTRY_POLL_SEC = 1.0  # saniye baÅŸÄ±na kontrol
```

### 7.7 Fill Detection ve Protection Staging

Emir gÃ¶nderildi, peki doldu mu?

```python
# Dolum kontrolÃ¼
filled_qty = order_filled(order)    # ccxt order dict'ten

# Koruma planÄ± â€” fill oranÄ±na gÃ¶re hangi aÅŸamalar aktif?
plan = build_staged_protection_plan(
    requested_qty=amount,
    filled_qty=filled_qty,
    min_fill_ratio=cfg.MIN_FILL_RATIO,    # 0.85
    trailing_enabled=cfg.TRAILING_ENABLED
)

# AÅŸama 1: Acil stop
if plan['stage1_active']:
    await _maybe_place_stage1_emergency_stop(bot, symbol, ...)

# Entry watch kaydÄ±
await register_entry_watch(bot, symbol, order_id, ...)
```

---

## 8. Codex Nedir ve Bu Projeye NasÄ±l YardÄ±mcÄ± Olur

### 8.1 Codex Nedir?

OpenAI Codex (ve benzer AI kod asistanlarÄ±), kaynak kodu anlayan ve onunla etkileÅŸime giren bÃ¼yÃ¼k dil modelidir. Bir projeyi analiz ederek:

- Kod kalitesi sorunlarÄ±nÄ± tespit eder
- Bug'larÄ± aÃ§Ä±klar ve fix Ã¶nerir
- DokÃ¼mantasyon Ã¼retir
- Test yazar
- Refactoring Ã¶nerir
- Yeni feature tasarlar

Eclipse Scalper gibi bÃ¼yÃ¼k, Ã§ok modÃ¼llÃ¼ bir projede Codex'in deÄŸeri ÅŸunlardÄ±r:

### 8.2 Eclipse Scalper'da Codex'in DeÄŸeri

#### ModÃ¼l BaÄŸÄ±mlÄ±lÄ±klarÄ±nÄ± Anlama

Eclipse Scalper'Ä±n en karmaÅŸÄ±k tarafÄ± modÃ¼ller arasÄ± implicit baÄŸÄ±mlÄ±lÄ±klardÄ±r. Ã–rneÄŸin:

- `symkey()` her yerde kullanÄ±lÄ±r ama SOT `entry_primitives.py`'dadÄ±r
- `bot.state.run_context` birÃ§ok modÃ¼l tarafÄ±ndan yazÄ±lÄ±r/okunur
- Kill switch sadece entry'leri durdurmalÄ±, exit'leri durdurmamalÄ±

Codex bu pattern'leri tÃ¼m codebase'i okuyarak Ã¶ÄŸrenebilir ve yeni kod yazarken aynÄ± kurallarÄ± uygular.

#### Bug Tespiti â€” GerÃ§ek Ã–rnek

```python
# HatalÄ± kod â€” kill switch exit'i de blokluyor
if not trade_allowed(bot):
    return  # BUG: Exit emirleri de burada bloklaniyor!

# DoÄŸru kod â€” intent kontrolÃ¼
if not trade_allowed(bot) and not intent_reduce_only:
    return  # Sadece entry emirleri bloklanÄ±r
```

Codex bu pattern'i Ã¶ÄŸrenip yeni dosyalarda aynÄ± bug'Ä± iÅŸaretleyebilir.

#### Test YazÄ±mÄ±

```python
# Eclipse Scalper test pattern'ini Ã¶ÄŸrenen Codex ÅŸunu yazabilir:
class TestKillSwitchExitSafety(unittest.TestCase):
    def test_exit_bypasses_kill_switch(self):
        bot = SimpleNamespace(
            cfg=SimpleNamespace(KILL_SWITCH_ENABLED=True),
            state=PsycheState(...),
        )
        # Kill switch aktif
        bot.state.kill_metrics['halted'] = True

        # Exit emri â†’ geÃ§meli
        result = await create_order(
            bot, "BTCUSDT", "market", "sell", 0.001,
            intent_reduce_only=True
        )
        self.assertIsNotNone(result)  # Exit geÃ§ti
```

#### Diagnostic ve Audit

Codex ÅŸu sorularÄ± yanÄ±tlayabilir:

- "Bu fonksiyonda lookahead bias var mÄ±?"
- "Bu retry loop neden sonsuz dÃ¶nebilir?"
- "Bu state field nerede yazÄ±lÄ±yor, nerede okunuyor?"
- "Bu modÃ¼lÃ¼ unit test etmek iÃ§in ne gerekiyor?"

### 8.3 Codex'in SÄ±nÄ±rlamalarÄ±

| SÄ±nÄ±rlama | AÃ§Ä±klama |
|---|---|
| **Runtime bilgisi yok** | Codex kodu okur ama Ã§alÄ±ÅŸtÄ±rmaz |
| **Exchange API davranÄ±ÅŸÄ±** | Binance'in edge case'lerini test etmez |
| **Async timing** | Race condition'larÄ± statik analizle bulmak zordur |
| **State corruption** | LZ4 dosyasÄ±ndaki bozulmayÄ± tespit edemez |

### 8.4 GerÃ§ek KullanÄ±m SenaryolarÄ±

**Senaryo 1 â€” Yeni Strateji Ekleme:**

```
Codex'e sÃ¶yle:
"strategies/eclipse_scalper.py'e bakarak yeni bir RSI-based strateji yaz.
AynÄ± interface'i kullan: scalper_signal() â†’ (is_long, is_short, confidence)"
```

Codex mevcut strateji pattern'ini okur ve uyumlu yeni strateji Ã¼retir.

**Senaryo 2 â€” Bug Fix:**

```
Codex'e sÃ¶yle:
"execution/order_router.py'de -4116 hatasÄ± 12 denemeden sonra da devam ediyor.
_MAX_VARIANTS limitini bul ve neden aÅŸÄ±ldÄ±ÄŸÄ±nÄ± aÃ§Ä±kla."
```

**Senaryo 3 â€” Reliability Audit:**

```
Codex'e sÃ¶yle:
"execution/ klasÃ¶rÃ¼ndeki tÃ¼m async fonksiyonlarÄ± incele.
Guardian-safe contract'Ä± ihlal eden â€” yani try/except olmadan raise edebilecek â€”
fonksiyonlarÄ± listele."
```

---

## 9. Codex CLI KullanÄ±m Rehberi

### 9.1 Codex'i Bu Repo Ä°Ã§in NasÄ±l BaÅŸlatÄ±rsÄ±nÄ±z?

Codex'i kullanmak iÃ§in Ã¶nce repo'yu context olarak vermelisiniz. Claude Code (bu asistan) veya OpenAI Codex CLI ile ÅŸu ÅŸekilde Ã§alÄ±ÅŸÄ±lÄ±r:

### 9.2 Temel Komutlar ve Ã–rnekler

#### Repo Analizi

```bash
# TÃ¼m repo'yu analiz et, mimarini aÃ§Ä±kla
codex "Bu repo'nun mimarisini aÃ§Ä±kla. execution/ klasÃ¶rÃ¼ne odaklan."

# Belirli bir modÃ¼lÃ¼ incele
codex "execution/bootstrap.py'i oku ve baÅŸlangÄ±Ã§ sÄ±rasÄ±nÄ± adÄ±m adÄ±m aÃ§Ä±kla."

# BaÄŸÄ±mlÄ±lÄ±k haritasÄ±
codex "bot objesi hangi modÃ¼llerde kullanÄ±lÄ±yor? Her kullanÄ±m noktasÄ±nÄ± listele."
```

#### Bug Bulma

```bash
# Execution gÃ¼venilirliÄŸi analizi
codex "execution/ klasÃ¶rÃ¼nde guardian-safe contract'Ä± ihlal eden fonksiyonlarÄ± bul."

# State corruption riski
codex "PsycheState'in run_context'ini aynÄ± anda hangi modÃ¼ller yazÄ±yor?
Race condition riski var mÄ±?"

# Kill switch bypass riski
codex "Kill switch kontrolÃ¼ yapÄ±lmadan order gÃ¶nderen kod var mÄ±?
intent_reduce_only kontrolÃ¼nÃ¼ doÄŸrula."
```

#### Fix Ãœretme

```bash
# Belirli hatayÄ± dÃ¼zelt
codex "order_router.py'deki retry loop'u incele.
-4116 hatasÄ±nÄ±n 13. denemede de dÃ¶nebildiÄŸi bir path var mÄ±? DÃ¼zelt."

# Pattern standardizasyonu
codex "execution/ klasÃ¶rÃ¼ndeki tÃ¼m fonksiyonlarda try/except kullanÄ±mÄ±nÄ±
standartlaÅŸtÄ±r. Guardian-safe contract'a uymayan her fonksiyonu dÃ¼zelt."
```

#### GeliÅŸtirme

```bash
# Signal geliÅŸtirme
codex "strategies/eclipse_scalper.py'e VWAP-based entry filter ekle.
Mevcut gate'lerle aynÄ± pattern'i kullan."

# Yeni metric
codex "entry_loop.py'e entry latency metrici ekle.
signal_time â†’ order_submit_time arasÄ±nÄ± telemetry'ye logla."

# Yeni test
codex "risk/kill_switch.py iÃ§in unit test yaz.
bot stub olarak SimpleNamespace kullan, test pattern'i iÃ§in
tests/legacy_tools/test_entry_loop_unit.py'e bak."
```

#### Audit ve DoÄŸrulama

```bash
# Execution invariant kontrolÃ¼
codex "execution/ klasÃ¶rÃ¼ndeki tÃ¼m modÃ¼lleri incele.
EXE-01 (idempotency), EXE-02 (lifecycle), EXE-03 (kill-switch precedence)
invariantlarÄ±nÄ± doÄŸrula."

# Data hazÄ±rlÄ±k
codex "entry_loop'un data_ready event'i gelmeden Ã§alÄ±ÅŸmayacaÄŸÄ±nÄ± doÄŸrula.
Potansiyel race condition var mÄ±?"

# Config coverage
codex "config/settings.py'deki tÃ¼m config key'lerini listele.
execution/ klasÃ¶rÃ¼nde kullanÄ±lmayan key varsa iÅŸaretle."
```

### 9.3 Claude Code ile Entegrasyon

Bu repo iÃ§in Claude Code (bu asistan) kullanÄ±yorsanÄ±z:

```bash
# Claude Code CLI
claude "execution/entry_loop.py'i incele ve tÃ¼m gate'leri sÄ±rala"
claude "strategies/ klasÃ¶rÃ¼ndeki sinyal Ã¼retim mantÄ±ÄŸÄ±nÄ± tÃ¼rkÃ§e aÃ§Ä±kla"
claude "risk manager ve kill switch arasÄ±ndaki farkÄ± aÃ§Ä±kla"

# Dosya bazlÄ± soru
claude read execution/bootstrap.py -- "singleton pattern nasÄ±l uygulanmÄ±ÅŸ?"
```

### 9.4 Ä°yi Prompt Yazma Rehberi

| KÃ¶tÃ¼ Prompt | Ä°yi Prompt |
|---|---|
| "kodu dÃ¼zelt" | "execution/order_router.py'de -4116 retry limitini bul ve 12 denemede durduÄŸunu doÄŸrula" |
| "test yaz" | "risk/kill_switch.py iÃ§in test yaz, tests/legacy_tools/test_entry_loop_unit.py pattern'ini kullan" |
| "hata var mÄ±" | "entry_loop.py'de kill switch bypass riski var mÄ±? intent_reduce_only kontrolÃ¼nÃ¼ izle" |
| "aÃ§Ä±kla" | "brain/state.py'deki validate() metodunun ne zaman Ã§aÄŸrÄ±ldÄ±ÄŸÄ±nÄ± aÃ§Ä±kla" |

---

## 10. GeliÅŸtirici Workflow

### 10.1 Repo'yu Ä°lk Kez Ä°nceleme

```bash
# 1. Repo'yu klonla / dizine gir
cd eclipse_scalper

# 2. BaÄŸÄ±mlÄ±lÄ±klarÄ± kur
pip install -r requirements.txt

# 3. .env dosyasÄ±nÄ± hazÄ±rla
cp .env.example .env
# BINANCE_API_KEY=...
# BINANCE_API_SECRET=...
# TELEGRAM_TOKEN=...   (opsiyonel)
# TELEGRAM_CHAT_ID=... (opsiyonel)

# 4. Test suite'ini Ã§alÄ±ÅŸtÄ±r (repo'nun saÄŸlÄ±klÄ± olduÄŸunu doÄŸrula)
python -m pytest tools/ -v

# 5. Paper trading ile baÅŸlat
python main.py --dry-run
```

### 10.2 GeliÅŸtirme DÃ¶ngÃ¼sÃ¼

```
1. DeÄŸiÅŸiklik planla
   â””â”€ Hangi modÃ¼l etkileniyor?
   â””â”€ Guardian-safe contract korunuyor mu?
   â””â”€ symkey() SOT ihlal ediliyor mu?

2. Test yaz (Ã¶nce)
   â””â”€ tests/legacy_tools/test_<module>_unit.py dosyasÄ±na ekle
   â””â”€ SimpleNamespace bot stub kullan
   â””â”€ sys.stdout.reconfigure(encoding="utf-8")

3. DeÄŸiÅŸikliÄŸi yap
   â””â”€ Mevcut pattern'leri takip et
   â””â”€ Best-effort import kullan (try/except)
   â””â”€ Telemetry emit ekle

4. Testi Ã§alÄ±ÅŸtÄ±r
   python -m pytest tests/legacy_tools/test_<module>_unit.py -v

5. TÃ¼m suite'i Ã§alÄ±ÅŸtÄ±r
   python -m pytest tools/ -v

6. Paper trading ile manual test
   python main.py --dry-run

7. Reliability gate kontrol
   python tools/reliability_gate.py
```

### 10.3 Test Yazma StandardÄ±

```python
# tests/legacy_tools/test_yeni_ozellik_unit.py

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
        """Standart bot stub oluÅŸtur."""
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
        # Test mantÄ±ÄŸÄ±
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
        # 1. Veri kalitesi kontrolÃ¼
        if data is None:
            return False, False, 0.0

        # 2. Ä°ndikatÃ¶r hesapla
        # ...

        # 3. Gate koÅŸullarÄ±nÄ± deÄŸerlendir
        # ...

        # 4. SonuÃ§
        return is_long, is_short, confidence

    except Exception as e:
        # Guardian-safe: asla raise etme
        return False, False, 0.0
```

### 10.5 Yeni Risk KuralÄ± Ekleme

```python
# risk/risk_manager.py â€” can_open_position() iÃ§ine yeni kural:

def can_open_position(self, symbol, side, size_usd, current_price):
    # ... mevcut kurallar ...

    # YENÄ° KURAL: Gece saat 02:00-04:00 arasÄ± iÅŸlem yapma
    current_hour = datetime.utcnow().hour
    if 2 <= current_hour < 4:
        return False, "low_liquidity_hours"

    return True, ""
```

### 10.6 Codex ile Pair Programming

```bash
# 1. GÃ¶revi tanÄ±mla
claude "entry_loop.py'e yeni bir gate eklemem gerekiyor:
Son 1 saatte 3'ten fazla ardÄ±ÅŸÄ±k kayÄ±p varsa entry'i durdur.
Mevcut gate pattern'ini takip et ve telemetry emit ekle."

# 2. Ãœretilen kodu incele
# Claude kodu Ã¶nerir â†’ sen gÃ¶zden geÃ§irirsin

# 3. Test yaz
claude "Ã¶nerilen gate iÃ§in unit test yaz.
test_entry_loop_unit.py pattern'ini takip et."

# 4. Birlikte debug
claude "test baÅŸarÄ±sÄ±z: [hata mesajÄ±]
entry_loop.py satÄ±r 145'teki gate kontrolÃ¼nÃ¼ incele."
```

---

## 11. GerÃ§ek Execution Flow â€” Derin Teknik Analiz

### 11.1 Bot BaÅŸlatÄ±ldÄ±ÄŸÄ±nda Ne Olur?

Åimdi bir `python main.py --dry-run` komutundan itibaren sistemi microsaniye seviyesinde takip edelim:

#### Faz 1: Sistem BaÅŸlatma (t=0ms â†’ t=~3000ms)

```
main.py:
â”œâ”€ asyncio.WindowsSelectorEventLoopPolicy() â† Windows fix
â”œâ”€ .env parse â†’ sys.environ populate
â”œâ”€ CLI args â†’ dry_run=True, equity=None, mode="auto"
â””â”€ asyncio.run(bot.runner.run_bot(...))

bot/runner.py:
â””â”€ run_bot() â†’ execution.bootstrap.main()

execution/bootstrap.py (_BootstrapSingleton):
â”œâ”€ [t=10ms]  stdout/stderr UTF-8 reconfigure
â”œâ”€ [t=15ms]  .env load (python-dotenv)
â”œâ”€ [t=20ms]  Singleton lock acquire
â”‚            â†’ state/locks/execution_bootstrap.lock
â”œâ”€ [t=25ms]  Config instantiate (settings.Config or MicroConfig)
â”œâ”€ [t=30ms]  ENV bridge: ACTIVE_SYMBOLS â†’ bot.active_symbols
â”‚            SCALPER_EQUITY â†’ cfg.INITIAL_EQUITY
â”œâ”€ [t=50ms]  CCXT exchange init (async)
â”‚            â†’ exchanges/binance.py (dry-run: exchanges/paper_trading.py)
â”œâ”€ [t=100ms] brain state load
â”‚            â†’ ~/.blade_eternal.brain.lz4
â”‚            â†’ PsycheState.from_loaded() + validate() + recompute_derived()
â”œâ”€ [t=500ms] Binance health gate (network + auth ping)
â”‚            â†’ Dry-run'da skip
â”œâ”€ [t=550ms] Risk manager init
â”‚            â†’ risk/risk_manager.py (SQLite: data/risk_state.db)
â”œâ”€ [t=560ms] Telegram notifier (async, best-effort)
â”œâ”€ [t=600ms] EclipseEternal (bot) instance create
â”‚            â†’ bot.state, bot.cfg, bot.ex, bot.data, bot.active_symbols
â”œâ”€ [t=700ms] DataCache bootstrap
â”‚            â†’ data/cache.py: OHLCV cache initialize
â”‚            â†’ Exchange'den sembol metadata Ã§ek
â””â”€ [t=800ms] Async task loop baÅŸlat
```

#### Faz 2: Async Task Loop (t=800ms â†’ sÃ¼rekli)

```
asyncio.gather(
    guardian_loop(bot),        â† Health check, anomaly detection
    data_loop(bot),            â† Market data collection
    position_manager_loop(bot),â† Stop/TP protection
    entry_loop(bot),           â† Signal â†’ Order (data_ready bekler)
    reconcile_task(bot),       â† Exchange â†” state sync
)
```

Her task kendi dÃ¶ngÃ¼sÃ¼nde Ã§alÄ±ÅŸÄ±r, baÄŸÄ±msÄ±zdÄ±r.

### 11.2 Piyasa Verisi AkÄ±ÅŸÄ± (data_loop)

```
data_loop(bot):
â”œâ”€ Her sembol iÃ§in (BTCUSDT, ETHUSDT, ...):
â”‚  â”œâ”€ exchange.fetch_ohlcv(symbol, "1m", limit=200)
â”‚  â”œâ”€ exchange.fetch_ohlcv(symbol, "5m", limit=100)
â”‚  â”œâ”€ exchange.fetch_ohlcv(symbol, "15m", limit=50)
â”‚  â”œâ”€ exchange.fetch_ticker(symbol) â†’ anlÄ±k fiyat
â”‚  â””â”€ bot.data.ohlcv[symbol] = {...}  â† cache gÃ¼ncelle
â”‚     bot.data.price[symbol] = last_price
â”‚
â””â”€ Ä°lk veri baÅŸarÄ±yla geldi â†’ bot.data_ready.set()
   â†’ entry_loop artÄ±k Ã§alÄ±ÅŸabilir
```

**Veri tazeliÄŸi:**
```python
# Veri 150 saniyeden eski â†’ kill switch tetiklenir
if time.time() - last_data_time > KILL_MAX_DATA_STALENESS_SEC:
    kill_switch.request_halt(bot, "stale_data")
```

### 11.3 Signal Ãœretimi (entry_loop â†’ scalper_signal)

```
entry_loop(bot):
â”œâ”€ await bot.data_ready.wait()  â† Veri hazÄ±r mÄ±?
â”‚
â””â”€ DÃ–NGÃœ (ENTRY_POLL_SEC=1.0s):
   â”œâ”€ [Gate 1] entries_allowed() â†’ False ise skip
   â”œâ”€ [Gate 2] trade_allowed() â†’ Kill switch kontrol
   â”œâ”€ [Gate 3] anomaly_should_pause() â†’ Anomali var mÄ±?
   â”œâ”€ [Gate 4] data staleness kontrolÃ¼
   â”‚
   â””â”€ Her sembol iÃ§in (ENTRY_PER_SYMBOL_GAP_SEC=2.5s throttle):
      â”œâ”€ scalper_signal(symbol, data=bot.data, cfg=bot.cfg, bot=bot)
      â”‚  â†’ (is_long, is_short, confidence)
      â”‚
      â”œâ”€ confidence >= ENTRY_MIN_CONFIDENCE? (0.72)
      â”‚  HayÄ±r â†’ skip
      â”‚
      â”œâ”€ RiskManager.can_open_position() â†’ HayÄ±r â†’ skip
      â”‚
      â”œâ”€ Per-symbol lock al â†’ concurrent entry engelle
      â”‚
      â””â”€ [Sinyal geÃ§ti] â†’ create_order()
```

### 11.4 Order GÃ¶nderme (order_router)

```
create_order(bot, "BTCUSDT", "market", "buy", 0.001, ...):

â”œâ”€ clientOrderId oluÅŸtur
â”‚  "ent_BTCUSD_1708434523_a3f2" â†’ 26 karakter (<36 limit)

â”œâ”€ params hazÄ±rla
â”‚  {'positionSide': 'LONG', 'newOrderRespType': 'RESULT'}

â”œâ”€ exchange.create_order() â†’ ccxt
â”‚  Paper mode: PaperTradingAdapter.create_order()
â”‚  â†’ AnlÄ±k fiyatta simÃ¼le edilmiÅŸ dolum
â”‚  â†’ Slippage uygula (%0.05)
â”‚  â†’ Fee Ã§Ä±kar (taker: %0.04)

â”œâ”€ Telemetri
â”‚  emit(bot, "order_create", {symbol, type, side, amount, ...})
â”‚  â†’ logs/telemetry.jsonl

â””â”€ WAL intent kaydet
   state.run_context['wal_intents'][order_id] = {...}
```

### 11.5 Fill Detection ve Koruma

```
order = await create_order(...)

filled_qty = order_filled(order)  # ccxt order dict'ten

if filled_qty > 0:
    # Position state gÃ¼ncelle
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

    # Koruma planÄ± hesapla
    plan = build_staged_protection_plan(
        requested_qty=0.001,
        filled_qty=filled_qty,
        min_fill_ratio=0.85,
        trailing_enabled=True
    )

    # Acil stop â€” hemen gÃ¶nder
    await _maybe_place_stage1_emergency_stop(bot, "BTCUSDT", plan)

    # State persist
    await bot.state.save()
```

### 11.6 Position Manager â€” Stop/TP YÃ¶netimi

```
position_manager_loop(bot):

â””â”€ DÃ–NGÃœ:
   â””â”€ Her aÃ§Ä±k pozisyon iÃ§in:
      â”œâ”€ exchange.fetch_open_positions(symbol)
      â”‚  â†’ GerÃ§ek pozisyon boyutunu doÄŸrula
      â”‚
      â”œâ”€ Per-symbol lock al
      â”‚  (reconcile ile paylaÅŸÄ±lÄ±r â€” shared_locks.py)
      â”‚
      â”œâ”€ compute_exit_profile(bot, symbol, position, price)
      â”‚  â†’ stop_price, tp_price, trailing_params hesapla
      â”‚
      â”œâ”€ assess_stop_coverage(bot, symbol)
      â”‚  â†’ Exchange'de aktif stop emri var mÄ±?
      â”‚
      â”œâ”€ should_refresh_protection()
      â”‚  â†’ Fiyat deÄŸiÅŸti, stop gÃ¼ncellemesi gerekiyor mu?
      â”‚
      â””â”€ [Refresh gerekli ise]:
         â”œâ”€ place_stop_ladder_router(bot, symbol, ...)
         â”‚  â†’ create_order(intent_reduce_only=True) â† Exit! Her zaman geÃ§er
         â”‚
         â””â”€ place_trailing_router(bot, symbol, ...)
            â†’ create_order(intent_reduce_only=True)
```

### 11.7 Reconcile â€” Reality Sync

```
reconcile_tick(bot):

â”œâ”€ exchange.fetch_open_orders(symbol) â†’ GerÃ§ek emirler
â”œâ”€ exchange.fetch_open_positions(symbol) â†’ GerÃ§ek pozisyonlar

â”œâ”€ Orphan detection:
â”‚  Exchange'de var ama brain'de yok â†’ adopt et
â”‚  Brain'de var ama exchange'de yok â†’ temizle

â”œâ”€ Equity refresh:
â”‚  exchange.fetch_balance() â†’ USDT balance
â”‚  bot.state.update_equity(balance)

â”œâ”€ Stop coverage kontrolÃ¼:
â”‚  Her pozisyon iÃ§in stop emri yoksa â†’ place_stop_ladder_router()

â””â”€ Belief controller tick (opsiyonel):
   AÄŸ kalitesi + veri kalitesi â†’ guard_knobs gÃ¼ncelle
```

### 11.8 Graceful Shutdown

```
Ctrl+C basÄ±ldÄ± (SIGINT):
â”œâ”€ bot._shutdown.set()  â† Cooperative shutdown event

TÃ¼m async task'lar:
â”œâ”€ while not bot._shutdown.is_set(): ...
â””â”€ DÃ¶ngÃ¼den Ã§Ä±k

bootstrap.py:
â”œâ”€ bot._cancel_all_tasks()
â”œâ”€ await bot.state.save()  â† Final state persist
â”‚  â†’ ~/.blade_eternal.brain.lz4
â”œâ”€ Telegram: notify_shutdown()
â””â”€ exchange.close()  â† CCXT websocket kapat

Exit code: 130 (SIGINT standard)
```

---

## 12. Gelecek GeliÅŸtirme AlanlarÄ±

### 12.1 Reliability (GÃ¼venilirlik)

#### 12.1.1 WAL Intent Temizleme

Åu an `run_context['wal_intents']` sÄ±nÄ±rsÄ±z bÃ¼yÃ¼yebilir. Eski, tamamlanmÄ±ÅŸ intent'lerin TTL-tabanlÄ± silinmesi gerekiyor:

```python
# Ã–nerilen: 24 saat Ã¼zerinden purge
WAL_INTENT_TTL_SEC = 86400

def cleanup_wal_intents(state):
    now = time.time()
    state.run_context['wal_intents'] = {
        k: v for k, v in state.run_context.get('wal_intents', {}).items()
        if now - v.get('ts', 0) < WAL_INTENT_TTL_SEC
    }
```

#### 12.1.2 Multi-Instance KorumasÄ±

Singleton lock ÅŸu an `state/locks/execution_bootstrap.lock` dosya bazlÄ±. GÃ¼Ã§lendirme iÃ§in:
- Docker container ortamÄ± iÃ§in process namespace lock
- Cloud deployment iÃ§in distributed lock (Redis SETNX)

#### 12.1.3 Heartbeat Monitoring

`guardian_loop` ÅŸu an internal. DÄ±ÅŸ monitoring iÃ§in:
- HTTP health endpoint (`/health`, `/metrics`)
- Prometheus metrics export
- PagerDuty / OpsGenie integration

### 12.2 Signal Quality (Sinyal Kalitesi)

#### 12.2.1 Feature Engineering GeniÅŸletme

`features/` klasÃ¶rÃ¼ mevcut ama sÄ±nÄ±rlÄ±. Eklenebilecekler:

```python
# Piyasa mikrostrÃ¼ktÃ¼rÃ¼ Ã¶zellikleri
- Order book imbalance (bid/ask ratio)
- Trade intensity (son N saniyedeki iÅŸlem hacmi)
- Spread geniÅŸliÄŸi
- VWAP sapmasÄ±

# Zaman bazlÄ± Ã¶zellikler
- GÃ¼n iÃ§i saat etkisi (Asian/European/US session)
- Haber saati buffer (yÃ¼ksek volatilite Ã¶ncesi Ã§ekilme)

# Cross-asset
- BTC/ETH korelasyonu
- Crypto fear & greed index
```

#### 12.2.2 ML-Based Signal Filtering

```python
# Mevcut: kural bazlÄ± sinyal
# Gelecek: hafif ML filtre
from sklearn.ensemble import GradientBoostingClassifier

class SignalFilter:
    def should_trade(self, features: dict) -> float:
        """Confidence'Ä± ML ile kalibre et."""
        return self.model.predict_proba([feature_vector])[0][1]
```

#### 12.2.3 Walk-Forward Validation

`tools/` klasÃ¶rÃ¼ndeki backtest araÃ§larÄ±nÄ± canlÄ± pipeline'a baÄŸlamak:

```bash
# HaftalÄ±k otomatik backtest + validation
python tools/rank_passive_pockets_forward.py \
    --symbol BTCUSDT \
    --lookback-days 21 \
    --fee-mult 1.0 \
    --adv-mult 1.2 \
    --min-attempt-fill-rate 0.40
```

### 12.3 Execution Safety (Execution GÃ¼venliÄŸi)

#### 12.3.1 Partial Fill Handling

Åu an kÄ±smi dolum (`filled_qty < requested_qty`) acil stop ile kapatÄ±lÄ±yor. GeliÅŸmiÅŸ yaklaÅŸÄ±m:

```python
# KÄ±smi dolum durumunda:
# 1. Dolum oranÄ±nÄ± hesapla
fill_ratio = filled_qty / requested_qty

# 2. Threshold'a gÃ¶re karar
if fill_ratio >= MIN_FILL_RATIO:  # 0.85
    # Pozisyon kabul et, koruma ayarla
    place_stop_for_partial(filled_qty)
elif fill_ratio >= PARTIAL_ACCEPT_FLOOR:  # 0.50 (yeni)
    # KÄ±smi pozisyon tut, geri kalanÄ± iptal et
    cancel_remaining_and_protect(filled_qty)
else:
    # Ã‡ok kÃ¼Ã§Ã¼k dolum, Ã§Ä±k
    market_close_partial(filled_qty)
```

#### 12.3.2 Slippage Monitoring

Execution kalitesini Ã¶lÃ§mek iÃ§in:

```python
# Her dolumda slippage'Ä± kaydet
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

Piyasa koÅŸuluna gÃ¶re limit/market seÃ§imi:

```python
def _choose_order_type(spread_bps, urgency):
    if spread_bps < 2.0 and urgency == "normal":
        return "limit"  # Dar spread â†’ maker fee
    elif urgency == "exit":
        return "market"  # Ã‡Ä±kÄ±ÅŸta her zaman market
    else:
        return "market"  # GÃ¼vende kal
```

### 12.4 Performance (Performans)

#### 12.4.1 WebSocket Stream

Åu an REST polling (`fetch_ohlcv`). WebSocket stream ile:

```python
# ccxt pro / WebSocket
async def ws_ohlcv_stream(bot, symbol):
    async for ohlcv in bot.ex.watch_ohlcv(symbol, "1m"):
        bot.data.ohlcv[symbol]["1m"] = ohlcv
        # AnlÄ±k gÃ¼ncelleme â†’ daha hÄ±zlÄ± sinyal
```

Latency kazancÄ±: ~300ms â†’ ~10ms (REST polling â†’ WebSocket push)

#### 12.4.2 State Persistence Optimizasyonu

Åu an her kayÄ±tta tam state serialize/compress ediliyor. Incremental diff ile:

```python
# Sadece deÄŸiÅŸen field'larÄ± kaydet
def save_incremental(self, changed_fields: set):
    partial = {k: getattr(self, k) for k in changed_fields}
    append_to_wal(partial)  # LZ4 compressed append
```

#### 12.4.3 Data Cache Parallel Fetch

```python
# Åu an: Semboller sÄ±rayla fetch ediliyor
# Gelecek: Parallel
await asyncio.gather(*[
    fetch_symbol_data(bot, symbol)
    for symbol in bot.active_symbols
])
```

### 12.5 Observability (GÃ¶zlemlenebilirlik)

#### 12.5.1 Dashboard

```bash
# Mevcut: JSONL tabanlÄ± dashboard
python tools/telemetry_dashboard.py --path logs/telemetry.jsonl

# Gelecek: Grafana + InfluxDB
# telemetry.jsonl â†’ InfluxDB writer â†’ Grafana dashboard
```

#### 12.5.2 Alerting

```python
# P&L % kayÄ±p alert
if daily_pnl_pct < -0.02:  # -2% gÃ¼nlÃ¼k
    await telegram.send("UYARI: GÃ¼nlÃ¼k kayÄ±p -%2.1f%%" % abs(daily_pnl_pct))

# Execution kalitesi bozulma alert
if recent_slippage_avg > SLIPPAGE_ALERT_BPS:
    await telegram.send("UYARI: Ortalama slippage yÃ¼ksek: %.1f bps" % avg)
```

#### 12.5.3 Post-Trade Analysis

Otomatik haftalÄ±k rapor:

```bash
# Mevcut araÃ§larla
python tools/rank_passive_pockets_forward.py --output reports/weekly.md
python tools/micro_edge_report.py --last-days 7
```

---

## Appendix A â€” Kritik ENV Variable ReferansÄ±

| DeÄŸiÅŸken | VarsayÄ±lan | AÃ§Ä±klama |
|---|---|---|
| `SCALPER_DRY_RUN` | `1` | SimÃ¼lasyon modu (gÃ¼venli) |
| `SCALPER_PAPER_TRADING` | `1` | Paper trading (varsayÄ±lan) |
| `SCALPER_LIVE_TRADING` | â€” | Live arm â€” aÃ§Ä±k silah gÃ¼venliÄŸi |
| `SCALPER_MODE` | `auto` | `auto`, `micro`, `production` |
| `SCALPER_EQUITY` | â€” | BaÅŸlangÄ±Ã§ equity override |
| `ACTIVE_SYMBOLS` | `BTCUSDT` | VirgÃ¼lle ayrÄ±lmÄ±ÅŸ semboller |
| `BINANCE_API_KEY` | â€” | Binance API key |
| `BINANCE_API_SECRET` | â€” | Binance API secret |
| `TELEGRAM_TOKEN` | â€” | Telegram bot token |
| `TELEGRAM_CHAT_ID` | â€” | Bildirim chat ID |
| `BOOT_REBUILD_ON_START` | â€” | Exchange'den pozisyon rebuild |
| `BOOT_MAINTENANCE_ONESHOT` | â€” | Tek reconcile tick, Ã§Ä±k |
| `PAPER_INITIAL_EQUITY` | `1000` | Paper baÅŸlangÄ±Ã§ bakiyesi (USDT) |
| `PAPER_MIN_CONFIDENCE` | `0.35` | Paper modda dÃ¼ÅŸÃ¼rÃ¼lmÃ¼ÅŸ confidence |
| `PAPER_FORCE_ONE_FILL` | â€” | Smoke test: bir emir aÃ§/kapat |
| `SCALPER_SIGNAL_DIAG` | â€” | Signal diagnostic logging |
| `SCALPER_DEBUG_LOOSE` | â€” | GevÅŸek gate'ler (sadece test) |
| `SCALPER_FORCE_ENTRY_TEST` | â€” | Zorla entry (plumbing validation) |

---

## Appendix B â€” Test KomutlarÄ±

```bash
# TÃ¼m testler
python -m pytest tools/ -v

# Tek test dosyasÄ±
python -m pytest tests/legacy_tools/test_entry_loop_unit.py -v

# Tek test
python -m pytest tests/legacy_tools/test_entry_loop_unit.py::EntryLoopTelemetryTests::test_recent_router_blocks_counts -v

# Test kategorisi
python -m pytest tools/ -k "kill_switch" -v

# HÄ±zlÄ± kontrol (quiet)
python -m pytest tools/ -q

# Reliability gate
python tools/reliability_gate.py
```

---

## Appendix C â€” HÄ±zlÄ± Referans: Kritik Kurallar

```
1. GUARDIAN-SAFE: TÃ¼m async execution fonksiyonlarÄ± try/except ile sarÄ±lÄ±r,
   dÄ±ÅŸarÄ± asla raise etmez.

2. EXIT ALWAYS PASSES: intent_reduce_only=True olan emirlere hiÃ§bir
   entry kÄ±sÄ±tlamasÄ± uygulanmaz. Kill switch bile geÃ§er.

3. EXCHANGE IS TRUTH: bot.state pozisyon bilgisi "inanÃ§"tÄ±r.
   Exchange'deki durum her zaman gerÃ§ektir.

4. SYMKEY SOT: Symbol normalizasyonu iÃ§in tek kaynak
   execution/entry_primitives.py:symkey()

5. BOUNDED RETRY: Sonsuz retry loop yasaktÄ±r.
   Her retry bounded olmalÄ± ve hata sÄ±nÄ±flandÄ±rmalÄ± Ã§alÄ±ÅŸmalÄ±dÄ±r.

6. HEDGE MODE: Binance Futures'da positionSide (LONG/SHORT)
   her order'da gereklidir.

7. CLIENT ORDER ID: Binance'e gÃ¶nderilen clientOrderId < 36 karakter
   olmalÄ±dÄ±r. router.py otomatik hash/truncate yapar.

8. LOCKS: shared_locks.py Ã¼zerinden per-symbol lock kullan.
   AynÄ± anda 1'den fazla lock tutma (deadlock riski).

9. CONFIG PRIORITY: ENV variable â†’ bot.cfg â†’ hardcoded default
   (bu sÄ±rayla override edilir, main.py otoritedir).

10. TELEMETRY: Her Ã¶nemli olay emit() ile loglanÄ±r.
    Telemetri sistemi yoksa trading sessizce devam eder.
```

---

*Bu dokÃ¼man `docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` olarak kaydedilmiÅŸtir.*
*Repo analiz tarihi: 2026-02-20*
*Referans branch: feat/reliability-gate-automation*

