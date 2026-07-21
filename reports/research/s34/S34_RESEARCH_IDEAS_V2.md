# S34 Research Ideas V2 — Kapsamlı Test Planı
**Tarih:** 2026-07-01  
**Kapsam:** 3 onaylanmış sinyal genişlemesi + 10 yeni araştırma alanı  
**Evren:** 596 ETH SELL anchor, 4.5 ay (Şubat–Haziran 2026)

---

## BÖLÜM 1 — ONAYLANMIŞ SİNYALLER (Test Planları)

### 1A. LONG Gate Relax: btc4h<0 OR btc7d<0
**Mevcut durum:** N=68, /ay=15, WR=73.5%, avg=+55bps, MC p=0.0  
**Mevcut live:** sadece `btc7d<0` gate — N=31, /ay=6.9  

#### Açık Sorular (test)
| # | Soru | Hipotez | Metod |
|---|---|---|---|
| 1A-T1 | btc4h<0 tek başına vs btc7d<0 tek başına — hangisi birincil driver? | btc4h<0 daha kısa vadeli dolayısıyla daha güçlü sinyal olabilir | İki grubu split et, WR/avg/MC ayrı hesapla |
| 1A-T2 | btc4h<0 ONLY eklenen eventler (btc7d>=0 ama btc4h<0): kalite nedir? | 37 yeni event — WR'ı kötüleştiriyor mu? | "Added-only" subgroup analizi |
| 1A-T3 | Session x gate interaksiyon: btc4h<0 US session + ASIA session farklı mı? | US session btc4h korelasyonu daha güçlü olabilir | 2x2 grid: (btc4h/7d) x (US/ASIA) |
| 1A-T4 | Month stability: OR gate her ay tutarlı mı? | OR gate daha stabil olmalı (daha büyük N) | Aylık WR/avg trend tablosu |
| 1A-T5 | DOW x gate: Mon block hâlâ geçerli mi OR gate'de? | Daha geniş universe'de Mon block kaldırılabilir | Mon içi vs Mon-block performance |

#### Uygulama Planı (live'a gitmeden önce gerekli)
1. 70/30 chronological holdout: holdout (son 1.35 ay) WR >= 60%
2. MC permutation: mc_p < 0.05
3. Walk-forward 5-fold: 4+/5 pozitif
4. Live executor gate değişikliği: operator sign-off gerekli (SAF-02)
5. Shadow runner'da önce test: 2-3 hafta ileriye dönük izleme

---

### 1B. SHORT_NOISY: BTC>=1M delay5
**Mevcut durum:** N=25, /ay=5.5, WR=76%, avg=+129.2bps, MC p=0.0  
**Shadow runner:** `if False` ile kapalı — HENÜZ AKTİVE EDİLMEDİ  

#### Açık Sorular (test)
| # | Soru | Hipotez | Metod |
|---|---|---|---|
| 1B-T1 | Optimal hold time: 2h vs 3h vs 4h? | 2h yeterli (BTC rebound hızlı) | Hold window sweep: 60/90/120/150/180m |
| 1B-T2 | BTC threshold stability: 500K vs 1M vs 2M | 1M optimum ama stability testi gerekli | Threshold x holdout matris |
| 1B-T3 | Delay sensitivity: 5m vs 10m vs 15m | 10m daha güçlü ama daha az N | Delay sweep: 5/10/15m, WR + N |
| 1B-T4 | Score gate ekle: score>=3 zorunlu mu? | BTC>=1M delay5 sc>=3: N=20, WR=80% (freq expan.) | Score-conditioned subgroup |
| 1B-T5 | Vol regime filter: yüksek vol'de SHORT daha iyi mi? | Cascade sonrası vol yüksekse SHORT daha uzun sürer | vol_decile quartile x SHORT WR |
| 1B-T6 | ETH noisy sequence: kaç follow-on cascade var 0-30m? | Daha fazla noisy cascade = daha güçlü SHORT | count_noisy_0_30m quartile analizi |

#### Aktivasyon Planı
1. Shadow runner `if False` → koşullu aktif et: `if btc_conf_notional >= 500_000`
2. 500K eşikle başla (N fazla → daha hızlı veri birikiyor)
3. 30 live event sonra holdout istatistiğine bak
4. Live'a yükseltmek için operator sign-off

---

### 1C. Echo Cascade 30-90m Silence
**Mevcut durum:** N=76, /ay=16.8, WR=68.4%, avg=+63.1bps, MC p=0.0  
**Not:** Echo = anchor sonrası 30-90m içinde ikinci ETH SELL ≥200K cascade  

#### Açık Sorular (test)
| # | Soru | Hipotez | Metod |
|---|---|---|---|
| 1C-T1 | Window sweep: 20-60 vs 30-90 vs 45-120 vs 60-180 | 30-90 optimum mu yoksa daha geniş daha iyi mi? | 6 window konfigürasyonu karşılaştır |
| 1C-T2 | Echo büyüklüğü: ilk cascade'e göre oran önemli mi? | echo/anchor > 0.5 (küçük echo) → daha iyi | echo_notional/anchor_notional quartile |
| 1C-T3 | Echo sonrası silence: echo'nun kendisi de silence mi olmalı? | Echo + silence gate → daha temiz sinyal | echo + (no follow-on 0-30m after echo) |
| 1C-T4 | Echo = aynı yön (SELL) zorunlu mu? | SELL echo daha iyi (BUY echo = farklı signal) | SELL echo vs BUY echo ayrı analiz |
| 1C-T5 | Echo entry timing: anında vs +5m delay? | Anında girilse slippage fazla, +5m delay optimize eder | Entry delay sweep: 0/1/5/10m |
| 1C-T6 | Hold time: 4h mı yoksa 2h optimum mu? | İlk cascade LONG = 4h, echo = daha kısa rebound? | 2h vs 3h vs 4h hold sweep |
| 1C-T7 | Third cascade (echo of echo)? Degradasyon var mı? | 3rd cascade = exhaustion, edge biter | 3. cascade detect et, subgroup WR |

#### Uygulama Planı (shadow runner)
1. `_detect_echo(anchor_ts_ms, conn, window_lo_ms=30*60_000, window_hi_ms=90*60_000)` fonksiyonu
2. Echo anchor'u yeni `signal="ECHO_SILENCE"` olarak logla
3. Ayrı shadow bucket'ı izle: `C_echo_cascade`
4. Live'a geçiş için 30 forward event + holdout + operator sign-off

---

## BÖLÜM 2 — YENİ ARAŞTIRMA ALANLARI

### 2A. ⭐⭐⭐⭐⭐ ETH SELL BTC1000 DOW SCORE3 → Simetri Analizi

**Hipotez:** Mevcut state machine (ETH SELL → LONG countertrend) simetrik bir SHORT state machine ile tamamlanabilir mi?  
**Not:** ETH BUY → SHORT zaten KAPANDI (WR=46%, MC p=0.988). Simetri farklı açıdan test edilecek.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2A-T1 | Condition reversal SHORT | ETH SELL anchor + ALL long conditions MET ama BTC 1M+ ile CONFIRM → SHORT (tersine). Şu an SHORT_NOISY budur ama bu daha formalize | BTK ile confirm = fiyat düşmeye devam = SHORT |
| 2A-T2 | DOW asymmetry SHORT | SHORT için hangi DOW günleri daha iyi? LONG için Mon/Wed kötü — SHORT için de aynı mı? | SHORT için Cuma/Perş daha iyi olabilir (US kapanış yakın) |
| 2A-T3 | Score asymmetry SHORT | LONG score3 optimal. SHORT için score4 mi score3 mi daha iyi cut? | Yüksek score = piyasa zayıf = SHORT daha güçlü olabilir |
| 2A-T4 | Full symmetric backtest | ETH BUY ≥200K + btc4h>0 + score≥3 + US session → SHORT 2h | Muhtemelen ölü ama sistematik test |
| 2A-T5 | BTC SELL ≥1M + DOW + score → LONG (BTC state machine) | ETH'in yerinde BTC kullan — standalone BTC bouncer | BTC-led sinyal dead olduğu biliniyor ama DOW/score filtresiyle? |

**Veri kaynağı:** `liquidations` BTC SELL, `mark_prices` BTC  
**Öncelik:** MEDIUM — ⭐⭐⭐⭐⭐ ama 2A-T4 büyük ihtimalle ölü

---

### 2B. ⭐⭐⭐⭐⭐ State Transition Analizi (SILENCE→NOISY→BTC Confirm)

**Hipotez:** Cascade sonrası state transitions dizisi (ne ne zaman geliyor) alpha'nın birincil gürültü kaynağı. Doğru sequence mapping = daha temiz entry / exit sinyali.

#### Transition Grafiği
```
ETH SELL ≥200K anchor
    │
    ├─ [0-5m] ETH SELL follow-on ≥50K?
    │      ├─ YES → NOISY_FAST (danger)
    │      └─ NO  → SILENCE_START
    │
    ├─ [5-30m] daha fazla ETH SELL?
    │      ├─ YES after silence → SILENCE_BROKEN (?)
    │      └─ NO → SILENCE_HOLD
    │
    └─ [0-∞] BTC SELL ≥1M?
           ├─ YES + noisy → SHORT_CONFIRMED_DANGER
           └─ YES + silence → SHORT_CONFIRMED_SILENCE (?)
```

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2B-T1 | 4 state partition | anchor'ları 4'e böl: (1) silence-no-btc, (2) silence-btc, (3) noisy-no-btc, (4) noisy-btc. WR/avg/tail her grup | silence-no-btc LONG best, noisy-btc SHORT best |
| 2B-T2 | Transition timing | İlk noisy event ne kadar geç geldi? <5m vs 5-15m vs >15m. Gecikme = daha güçlü silece sonrası? | Geç gelen noisy = silence daha solid = better LONG |
| 2B-T3 | Re-silence sonrası tekrar LONG | Noisy burst oldu, sonra tekrar suskunlaştı (10-30m silence sonra noisy sonra tekrar silence) → LONG'a gir | Re-silence = exhaustion = bounce daha güçlü |
| 2B-T4 | BTC confirm içinde silence: BTC sürerken ETH susuyor mu? | BTC SELL süresi içinde ETH SELL yoksa → ETH decoupled from BTC = bounce güçlü | ETH-BTC decoupling sırasında LONG daha iyi |
| 2B-T5 | Full state sequence histogram | Tüm anchor'lar için transition string yaz (SELL/SILENCE zaman serisi, hash it), top-10 pattern'i bul | Hangi pattern en sık ve en karlı? |

**Veri:** `liquidations` tüm ETH + BTC timestamps, `mark_prices`  
**Öncelik:** YÜKSEK — ⭐⭐⭐⭐⭐ ve mevcut sistemle doğrudan ilgili

---

### 2C. ⭐⭐⭐⭐⭐ Second Cascade / Second Wave

**Hipotez:** İlk büyük cascade'in ardından gelen ikinci büyük cascade (>30m sonra, yani Echo değil), piyasanın gerçekten "ikinci dalga satışı" gösterdiği ve bu noktada LONG entry'nin daha güçlü olduğu anlar.

#### Echo vs Second Wave Farkı
- **Echo (30-90m):** İlk cascade'in "yankısı" — görece küçük, hızlı, aynı momentum
- **Second Wave (>90m, <6h):** Yeni bağımsız dalga — daha uzun süre geçti, ilk bounce başladı ama sonra tekrar sattı

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2C-T1 | Second wave window tarama | 90-180m, 180-360m, 360-720m sonraki cascade'i bul; her window için WR | 90-180m pencere en güçlü |
| 2C-T2 | Price between waves | İki cascade arası fiyat hareketi: yükseldi sonra düştü mü (gerçek 2. dalga) vs hiç yükselmedi (1. dalga devam) | Aradaki yükseliş >20bps = gerçek 2. dalga = daha iyi LONG |
| 2C-T3 | Second wave büyüklüğü vs first | 2nd/1st notional oranı: azalan baskı mı (2nd < 1st) yoksa artan mı? | Azalan baskı = capitulation = daha güçlü bounce |
| 2C-T4 | N-th wave analizi | Kaç cascade grubu var 24h'ta? 1 büyük cascade vs 2 vs 3+ | 2 cascade optimal (exhaustion olmadan yeterli sell) |
| 2C-T5 | Second wave state machine | Second wave + silence (30m) → LONG as new route | Ayrı standalone signal olarak test et |

**Veri:** `liquidations` ETH SELL, `mark_prices`  
**Öncelik:** YÜKSEK — mevcut Echo araştırmasının doğal devamı

---

### 2D. ⭐⭐⭐⭐⭐ BTC Lead-Lag Analizi

**Hipotez:** BTC ve ETH cascade zamanlaması arasındaki lag, ETH LONG sinyalinin kalitesini etkiler.  
**Bilinen:** BTC-led sinyal ÖLÜDÜR (WR=50%, MC p=0.77). Ama LAG'ın LONG kalitesi üzerindeki etkisi test edilmedi.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2D-T1 | BTC-ETH zamansal ilişki | Her ETH anchor için en yakın BTC SELL ≥500K'yı bul. delta_ms = BTC_ts - ETH_ts. <0 = BTC önce, >0 = ETH önce | ETH önce (BTC gecikiyor) → bounce daha güçlü |
| 2D-T2 | BTC isolation quartile | btc5m_bps (BTC 5m return anında): quartile split. ETH cascade'iyle birlikte BTC de çöküyorsa WR düşer mi? | BTC negatif ama az → ETH isolated = daha iyi LONG |
| 2D-T3 | BTC return during cascade | btc_return_during_cascade (detector_signals field, 73 row) — bu field daha geniş evren için hesapla | BTC küçük düşüş sırasında ETH büyük düşüş = ETH oversold |
| 2D-T4 | BTC divergence indicator | ETH anchor'dan 5m sonra: ETH fiyatı düşüyor, BTC yükseliyor mu? BTC-ETH divergence | Divergence = ETH oversold relative to BTC = strong bounce |
| 2D-T5 | btc_lead_offset_sec kullan | detector_signals'daki mevcut field: btc_lead_before_cascade ve btc_lead_offset_sec var (73 row) — correlation with outcomes | Field'ların predictive değeri ne? |

**Veri:** `liquidations` BTC + ETH, `mark_prices` BTC + ETH, `detector_signals` (73 row, mevcut fields)  
**Öncelik:** YÜKSEK — yeni açı, mevcut veriden çıkarılabilir

---

### 2E. ⭐⭐⭐⭐ Failed Bounce → SHORT

**Hipotez:** ETH SELL cascade sonrası fiyat yükseldi (failed_cascade=True, anchor+5m price UP) → bu "başarısız bounce" artık resistance'a dönüştü → SHORT pozisyonu açılabilir.

#### Failed Bounce Tanımı
```
failed_cascade = True: anchor fiyatı < anchor+5m fiyatı (yükseliş)
```
Şu an LONG bucket'ı (WR=78.3%, N=23) — ama SHORT hipotezi tersine.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2E-T1 | Failed cascade → SHORT | failed_cascade=True ve NOISY follow-on (50K+ in 0-15m after bounce) → SHORT 2h | Bounce + yeni satış = double top = SHORT |
| 2E-T2 | Bounce magnitude gate | Bounce büyüklüğü: price(+5m)-price(anchor) bps olarak. >30bps bounce vs <10bps bounce | Büyük bounce = daha güçlü resistance = SHORT |
| 2E-T3 | Time-to-reverse | Bounce oldu, kaç dakika sonra fiyat geri döndü anchor fiyatına? Hızlı reverse = daha güçlü SHORT | <30m reverse → SHORT güçlü |
| 2E-T4 | Failed bounce + BTC confirm | failed_cascade=True + BTC SELL ≥1M → SHORT (mevcut SHORT_NOISY ile overlap kontrolü) | Potansiyel overlap: ayrıştır |
| 2E-T5 | Failed cascade + session | Session bazında failed cascade WR: US'de bounce tutmuyor mu? | US session failed bounce → daha sık SHORT döner |

**Veri:** `mark_prices` ETH (5m marks), shadow runner'daki failed_cascade field  
**Öncelik:** ORTA — ilginç ama N az şimdilik (23 failed cascade backfill'de)

---

### 2F. ⭐⭐⭐⭐ Cascade Density

**Hipotez:** Son 24 saatteki cascade sayısı (density_24h), piyasanın "kalabalık" ya da "temiz" olduğunu gösterir.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2F-T1 | Density quartile analizi | density_24h = 0, 1, 2, 3, 4+ gruplar. Her grup LONG WR | density=0 (temiz) en iyi LONG; density>=3 kötü |
| 2F-T2 | Density + short interaction | Yüksek density → kalabalık piyasa → SHORT için iyi mi? | density>=3 + BTC confirm → SHORT daha güçlü |
| 2F-T3 | Density velocity | Son 6h'taki cascade sayısı vs son 24h: hızlanıyor mu? | Hızlanan density = crowded = LONG'u avoid et |
| 2F-T4 | Session-adjusted density | US session density yüksek her zaman → normalize | Session'a göre normalize density'nin edge'i |
| 2F-T5 | Density ceiling gate | density_24h >= 3 → LONG girme (veto filter) | WR çok düşükse → veto ekle live'a |

**Veri:** `liquidations` ETH SELL (density_24h hesapla = son 24h'ta anchor öncesi ≥200K eventler)  
**Öncelik:** ORTA — density_24h shadow runner'da loglanıyor, 3 ay sonra daha iyi veri

---

### 2G. ⭐⭐⭐⭐ Liquidity Vacuum

**Hipotez:** Cascade sonrası "tam" sessizlik (hiç follow-on likidasyon yok, hatta küçük olanlar bile) vs partial silence (küçük eventler var ama büyük yok) farklı edge veriyor.

#### Vacuum vs Silence Farkı
- **Silence:** ≥200K follow-on yok (current gate)
- **Vacuum:** ≥50K follow-on bile yok — derin sessizlik

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2G-T1 | Vacuum definition test | 0-30m içinde toplam likid. notional < X bps thresholds (10K, 25K, 50K, 100K). Her threshold için LONG WR | Daha derin vacuum = daha iyi LONG |
| 2G-T2 | Vacuum duration | Sessizlik kaç dakika sürdü? 5m, 10m, 15m, 30m cut | Daha uzun vacuum = daha güçlü signal |
| 2G-T3 | Partial silence pattern | Küçük BUY cascade geldi 0-30m içinde: market test etti ama satış yok. Bu "dip test" = daha iyi? | BUY micro-cascade in silence = buyers testing = LONG güçlü |
| 2G-T4 | Book spread recovery | book_ticker spread_pct: anchor'dan sonra spread normalleşiyor mu? Hızlı normalleşme = vacuum tamam | spread_recovery_time < 5m → vacuum kaliteli |
| 2G-T5 | agg_trades OFI vacuum | agg_trades: anchor sonrası 0-10m buyer notional / seller notional. OFI > 0 (net buyers) = vacuum dolduruluyor | OFI > 0 in vacuum → güçlü LONG |

**Veri:** `liquidations` ETH (tüm boyutlar), `book_ticker` spread_pct, `agg_trades` OFI  
**Öncelik:** YÜKSEK — `agg_trades` index'li, sorgular hızlı, yeni açı

---

### 2H. ⭐⭐⭐⭐ Cascade Exhaustion

**Hipotez:** Anchor öncesi birden fazla cascade geldi, son cascade anchor büyük ama öncekilerden KÜÇÜK (azalan baskı = exhaustion). Bu pattern → güçlü bounce.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2H-T1 | prebuildup level analizi | prebuildup = 0 (tek cascade), 1, 2, 3+ (önceki cascade var). Her level için WR/avg | prebuildup=2 optimum (yeterli exhaustion, fazla crowded değil) |
| 2H-T2 | Anchor büyüklüğü vs prebuildup ortalaması | anchor_notional / mean(prebuildup_notionals): > 1 = artan (climax), < 1 = azalan | Azalan seri (climax geride) daha iyi bounce |
| 2H-T3 | Interburst gap | Prebuildup cascades arası süre: < 5m (hızlı) vs > 15m (yavaş). Yavaş = more patient exhaustion | Yavaş prebuildup = daha sağlıklı setup |
| 2H-T4 | Volume exhaustion metric | Toplam prebuildup notional vs anchor notional. Total pressure discharged? | anchor / (total_prebuildup + anchor) < 0.3 = last gasp |
| 2H-T5 | prebuildup + density composite | density_24h mantığına prebuildup'ı ekle: composite exhaustion score | Composite score > threshold = strong LONG |

**Veri:** `liquidations` ETH SELL (prebuildup = count(ts_anchor-30m, ts_anchor, ≥200K events))  
**Öncelik:** ORTA — prebuildup shadow runner'da loglanıyor; backfill'de mevcut ama threshold 30m değil

---

### 2I. ⭐⭐⭐ Cluster Shape

**Hipotez:** Cascade'in şekli (ani spike vs kademeli tırmanma) bounce kalitesini etkiler.

#### Mevcut Veri
`detector_signals` (73 satır) zaten bazı şekil field'larına sahip:
- `cascade_rise_time_sec`: cascade oluşma süresi
- `fingerprint_class`: cascade parmak izi sınıfı
- `liq_velocity_profile`: likidasyon hız profili
- `liq_composition`: SELL/BUY karışımı
- `confirmation_delay_sec`: cascade teyit gecikmesi

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2I-T1 | Rise time analizi (73 row) | cascade_rise_time_sec: hızlı (<30s) vs yavaş (>120s). Her grubun LONG WR | Hızlı spike = more panic = better bounce |
| 2I-T2 | Fingerprint class cluster | fingerprint_class categories. Hangi sınıf en çok LONG WR? | Bazı class'lar sistematik olarak kötü olabilir |
| 2I-T3 | Velocity profile | liq_velocity_profile: ilk vs son ms notional yoğunluğu. Front-heavy vs tail-heavy | Front-heavy spike = capitulation bottom |
| 2I-T4 | Composition bias | liq_composition: cascade SELL-only mu, yoksa BUY liq karışık mı? | Pure SELL cascade = cleaner signal |
| 2I-T5 | Shape reconstruction (596 anchor) | 596 anchor için agg_trades kullanarak 5m cascade shape hesapla: (peak ms - start ms) / duration. Kendi velocity metric'imiz | Yeni cascade shape metriği; detector_signals'dan bağımsız |

**Veri:** `detector_signals` (73 row, mevcut fields) + `agg_trades` (custom shape compute)  
**Öncelik:** DÜŞÜK — detector_signals sadece 73 satır; 596 için custom hesap gerekli

---

### 2J. ⭐⭐⭐ Volatility Compression

**Hipotez:** Cascade öncesi volatilite düşükse (compression), büyük likidasyon daha anlamlı → daha güçlü bounce.

**Not:** `vol_state` ile "vol regime filtresi" Mega testinde geçmedi (MC p>0.07). Ama o test vol_decile AT cascade zamanı'ydı. Bu test vol ÖNCESI compression'ı ölçecek.

#### Test Planı
| # | Test | Tanım | Beklenti |
|---|---|---|---|
| 2J-T1 | Pre-cascade vol compression | Anchor'dan 30m önce vol_decile ortalaması: < 3 (low) vs > 7 (high). Düşük pre-vol = compression = daha büyük sürpriz | Pre-cascade düşük vol → daha iyi LONG |
| 2J-T2 | Vol expansion at cascade | anchor zamanı vol_decile vs 30m önceki vol_decile. Artış büyüklüğü = expansion ratio | Büyük expansion (2→8) = panic kaynak = bounce güçlü |
| 2J-T3 | high_vol_alert timing | anchor sırasında high_vol_alert=True vs False. Yüksek vol sırasında gelen cascade: crowded? | high_vol_alert=True = kötü giriş; False = daha temiz |
| 2J-T4 | rv_5m at cascade | rv_5m (5m realized vol) anchor sırasında. Quartile analizi | rv_5m düşük = compression sonrası cascade = daha güçlü |
| 2J-T5 | Vol compression + silence combo | Pre-vol<3 + vol expansion>5 + silence 30m → LONG | Full compression-expansion-silence sequence en temiz |

**Veri:** `vol_state` (53,648 rows, 4.5 ay kapsam, ts_ms align ile anchor'a bağla)  
**Öncelik:** ORTA — vol_state iyi kapsam var; join işlemi basit (nearest ts_ms)

---

## BÖLÜM 3 — TEST ÖNCELİK MATRİSİ

| Öncelik | Test | Alan | N (backfill) | Yeni veri? | Beklenen bulgu |
|---|---|---|---:|---|---|
| P1 | 2B-T1: 4 state partition | State Transition | ~600 | Hayır | WR >80% silence-no-btc, <30% noisy-btc |
| P2 | 2D-T1: BTC-ETH timing delta | BTC Lead-Lag | ~500 | Hayır | ETH-önce > BTC-önce WR |
| P3 | 2G-T5: OFI vacuum (agg_trades) | Liquidity Vacuum | ~596 | `agg_trades` | OFI>0 in silence → WR artış |
| P4 | 2J-T2: vol expansion ratio | Vol Compression | ~596 | `vol_state` | Expansion >5 → WR artış |
| P5 | 1B-T1: SHORT_NOISY hold sweep | SHORT_NOISY | 25 | Hayır | 90-120m optimal |
| P6 | 1C-T1: Echo window sweep | Echo Cascade | 76 | Hayır | 30-90m optimal teyid |
| P7 | 2C-T1: Second wave window | Second Wave | ~100 | Hayır | 90-180m pencere test |
| P8 | 2A-T2: DOW asymmetry SHORT | Simetri | ~200 | Hayır | SHORT için farklı DOW |
| P9 | 2F-T1: Density quartile | Cascade Density | ~596 | Hayır | density=0 best WR |
| P10 | 2H-T2: Exhaustion ratio | Cascade Exhaustion | ~596 | Hayır | Azalan seri → WR artış |

---

## BÖLÜM 4 — KAPALI HİPOTEZLER (Tekrar Test Edilmeyecek)

| Hipotez | Kapatma Nedeni | Tarih |
|---|---|---|
| ETH BUY → SHORT | WR=46%, MC p=0.988 (Mega-F) | 2026-07-01 |
| BTC SELL → ETH LONG standalone | WR=50%, MC p=0.77 (Freq Exp) | 2026-07-01 |
| SOL → ETH LONG | MC p=0.57 (Freq Exp) | 2026-07-01 |
| Vol regime filtresi (vol_decile AT cascade) | Tüm MC p>0.07 (Mega-C) | 2026-07-01 |
| ETH eşik düşürme 100-150K | +10-15 bps net — marginal (Freq Exp-A) | 2026-07-01 |
| basis_reversion strategy | WR=44.5%, avg=-9.5bps | 2026-07-01 |
| Partial exit 2h+4h | Tek 4h daha iyi (Mega-partial) | 2026-07-01 |
| Cumartesi sabahı LONG | WR=14.3%, avg=-90.8bps (Mega) | 2026-07-01 |

---

## BÖLÜM 5 — VERİ ENVANTERİ (bu araştırmalar için)

| Tablo | Satır | Kapsam | Hangi testler |
|---|---:|---|---|
| `liquidations` ETH SELL | 1,151,415 | ~4.5 ay | 2B, 2C, 2D, 2F, 2G, 2H |
| `mark_prices` ETH | 19,714,793 | ~4.5 ay | Tümü |
| `agg_trades` (indexed) | 376,616,237 | ~4.5 ay | 2G-T5 (OFI) |
| `vol_state` ETH | 53,648 | ~4.5 ay | 2J |
| `book_ticker` | 4,301,353,514 | ~4.5 ay | 2G-T4 (spread) |
| `detector_signals` | 73 | Son 2 ay | 2I-T1..T4 |
| `detector_heartbeat` | 551,629,265 | ~4.5 ay | 2B (state trace) |

**UYARI:** `book_ticker` 4.3B satır — timestamp bound ile sorgulanmalı (index=ts_ms). `agg_trades` 376M satır — `idx_trade_symbol_ts (symbol, ts_ms)` indexed, time-bound sorgu hızlı.

---

*Script'ler: `tools/research_s34_*.py`  
Çıktılar: `reports/research/s34/`  
Tüm testler: no-lookahead (DAT-01), 70/30 holdout, MC permutation, real cost (FEE_BPS=5.0)*
