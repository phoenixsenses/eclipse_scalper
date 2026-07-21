# AMI State Taxonomy v0 (el-tanımlı; Faz 6 latent keşfi öncesi)

Kaynak kod: `ami/enums.py` + `ami/states/engine.py`.

## Timeframe'ler ve rolleri
1m (mikroyapı/execution) · 5m (scalp basıncı) · 15m (geçiş) · 1h (intraday yapı)
· 4h (ana swing) · 1D (günlük rejim) · 1W (arka plan döngü)

## STRUCTURE_STATE (kural v0 — `StateEngine._structure`)
| Etiket | Kural özü |
|---|---|
| COMPRESSION | rv oranı < 0.6 ve |ret3| < 30bps |
| BREAKDOWN | vol genişleme + ret1 < −50 |
| EARLY_EXPANSION | vol genişleme + ret1 > +50 |
| RECOVERY | ret12 < −150 ama ret3 > 0 |
| DISTRIBUTION | ret12 > +150 ama ret3 < 0 |
| MATURE_TREND | ret12 > +150 ve ret3 > 0 |
| EXPANSION (bearish) | ret12 < −150 ve ret3 < 0 |
| RANGE | diğer |
+ `meta.direction` ∈ UP/DOWN/FLAT (conflict raporu için)

## CASCADE_STATE (`_cascade`)
5dk SELL ≥200K → EARLY_CASCADE/ACCELERATION; 30dk ≥200K → EXHAUSTION;
2h ≥200K → RECOVERY; yoksa NONE.

## LEVERAGE_STATE (`_leverage`)
funding<0 & vel≤0 → SHORT_CROWDED_BUILDING; funding<0 & vel>0 → SHORT_CROWDED_EASING;
funding>1e-4 → LONG_CROWDED; OI tazeliği meta'da.

## BOOK_STATE (`_book`)
bid_qty / 10dk-ortalama < 0.5 → BOOK_PULLS; > 1.6 → BOOK_THICKENS; arası BOOK_STABLE.

## TradeLifecycleState (`ami/lifecycle/engine.classify_lifecycle_path`)
OPEN → {HEALTHY, ACCELERATING, STALLING, WEAKENING, EXHAUSTED, RECOVERING,
REVERSING, INVALIDATED} → CLOSED. Kurallar 1m pnl path + 15m eğim + peak-giveback.

## Veri sağlığı eşiği (`FEED_LIMITS`, dk)
mark 5 · book 5 · agg 5 · OI 10 · spot 10 · vol_state 10 · liq 120
HEALTHY ≤ limit < DEGRADED ≤ 10×limit < STALE; tablo yok/boş = UNAVAILABLE.

> Not: Bu taksonomi bilinçli olarak basit. Amaç Faz 6'da latent state'lerin
> karşılaştırılacağı şeffaf bir baseline kurmak (whitepaper §35: model keşfeder, insan adlandırır).
