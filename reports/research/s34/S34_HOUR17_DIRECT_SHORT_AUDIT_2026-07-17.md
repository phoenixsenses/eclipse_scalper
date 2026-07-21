# S34 HOUR17 — Forward Ayrıştırma ve DIRECT SHORT Audit

**Tarih:** 2026-07-17
**Kapsam:** `LONG_HOUR17_HOLD6H` forward shadow kanıtının yeniden ölçümü + DIRECT SHORT (P2) hipotezinin istatistiksel audit'i
**Statü:** `AUDIT_5_AND_7_COMPLETE — ITEMS_6_(same-day placebo dahil)_9_10_PARTIAL` (AUDIT 5, 7 tamamlandı 2026-07-17; 9, 10 hâlâ NOT_EXECUTED)
**Erişim:** Tüm analiz SALT-OKUNUR. `data/microstructure.db` `mode=ro` açıldı. Hiçbir parametre, strateji davranışı veya runtime dosyası değiştirilmedi.

> ⚠️ **Bu rapor bir deploy önerisi DEĞİLDİR.** Live'da HOUR17 SHORT route'u yoktur; tüm SHORT risk kuralları araştırma varsayımıdır. Yön değişikliği CLAUDE.md §110 gereği kendi preregistration'ı + tam bağımsız inceleme zinciri gerektirir.

---

## 0. Yönetici özeti

| Bulgu | Sonuç |
|---|---|
| Shadow trade sayısı | **11** (hepsi `CLOSED_TIME_EXIT`, 0 gerçek emir) |
| Loglanan toplam | +2477.9 bps — **%115'i kesinti artefaktı** |
| Kurala uyan (6h) 8 trade | **−376.4 bps, WR %25** |
| 6h alpha (eşleşmiş kontrol) | **−41 bps**, CI95 [−96, +15] |
| İlk LONG bacağının katkısı | **negatif her hücrede** (kimlik: P5−P3 = P4) |
| DIRECT SHORT MICRO ailesi (1m/2m/5m) | FWER p = **0.0000** ✅ |
| DIRECT SHORT MEDIUM ailesi (6h–18h) | FWER p = **0.0523** ❌ |
| 10h hücresi exact randomization | p = **0.0781** ❌ |
| **BİRİNCİL HÜKÜM (§4 sonrası)** | `DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT` |
| **AUDIT 5 (time control)** | **`MEDIUM_SHORT_FAILS_EXACT_TIME_CONTROL`** — bkz. §10 |
| **AUDIT 7 (execution)** | **`MICRO_SHORT_FAILS_EXECUTION_MODEL`** — bkz. §11 |
| **BİRLEŞİK NİHAİ** | **`BOTH_FAIL`** — bkz. §12 |

**Önceki `DIRECT_SHORT_ALPHA_SUPPORTED` hükmü GERİ ÇEKİLDİ** (§5). AUDIT 5 ve 7 sonrası, kalan iki dayanak (MICRO ve MEDIUM) da bağımsız olarak düştü — bkz. §10–§12.

---

## 1. Forward shadow kanıtının durumu

`reports/shadow/s34_state_machine_shadow.jsonl`, 2026-07-04 → 2026-07-17 (13 gün).

### 1.1 Trade sayımı

- `pnl` bloğu: n=11, avg_net +225.3, WR %45.5, total +2477.9
- Ledger CLOSE sayımı: **11 distinct** — aggregate ile birebir uyumlu
- OPEN satırları 22 görünüyor ama **11 anchor id'nin her biri 2× loglanmış** → duplicate-logging artefaktı (§127 BUY_FADE duplicate-close ile aynı sınıf). PnL CLOSE'dan sürüldüğü için muhasebe şişmemiş; OPEN sayan her tüketici şişer.
- `runtime/s34_live_executor_state.json` içinde **HOUR17 geçmiyor** → hiç gerçek emir yok.

### 1.2 Kesinti artefaktı (BLOCKING for evidence integrity)

Ledger'da **07-13 20:50 → 07-15 19:13 arası 46.4 saatlik aktivite boşluğu**. Üç pozisyon (07-13 17:36/17:57/18:19) açıkken kalmış; runner dönünce **üçü de aynı milisaniyede (07-15 19:13:39) aynı fiyattan (1925.19740577)** kapatılmış. `exit_due_ms` üçü için 07-13 23:36 idi → ~44 saat geç.

| | n | WR | total | avg |
|---|---|---|---|---|
| Loglandığı hâliyle | 11 | 45.5% | **+2477.9** | +225.3 |
| **Kuralına uyan (6h)** | **8** | **25.0%** | **−376.4** | **−47.0** |
| Kesinti artefaktları | 3 | 100% | +2854.3 | +951.4 |

Ayrıca `mark_prices`'ta **07-13 21:07 → 07-14 12:21 arası 15.2h boşluk** — aynı kesinti. Bu boşluk 2-gün ufkunda **en kötü iki kaybedeni de siliyor** (07-11 22:20 → −286, 07-11 23:21 → −224 civarındaydı). NA'lar sonuçla korele; "2 gün %100 WR" tablosu bunun artefaktı:

| 2-gün ufku | n | WR | toplam |
|---|---|---|---|
| tabloda görünen (NA atılmış) | 6 | **100%** | +3659 |
| + kesinti boşluğundaki 2 trade | 8 | 75% | +3149 |
| + sansürlü 3 trade (alt sınır) | 11 | **54.5%** | +2011 |

### 1.3 Stop davranışı — CONTRACT FINDING

- `tools/s34_realtime_shadow_runner.py:680` → `"observer_note": "hold_predictor_hour17_no_early_exit_no_stop"` — **shadow STOPSUZ**
- `tools/s34_state_machine_live_executor.py:70` → `HOUR17_STOP_BPS = 300.0`, satır 678 `stop_bps_override` — **live 300bps stoplu**
- Bu örneklemde **fark yok**: 11/11 `TIME_EXIT`, en derin MAE **−200.9 bps** (07-11 22:20, T+2.2h) → stop'a 99.1 bps marj
- **Gelecekte divergence**: MAE −300'ü aşan ilk trade'de shadow tutmaya devam eder, live kapatır → `E-HOUR17-FWD-001` forward kanıtı canlı davranışı temsil etmeyi **sessizce** bırakır. Ledger'da 11/11 `TIME_EXIT` görmek bu riski gizliyor.

---

## 2. Metodoloji

### 2.1 Event cluster'lar (single-linkage, ≤2h)

| cluster | n | üyeler (UTC) |
|---|---|---|
| C1 | 1 | 07-04 23:14 |
| C2 | 1 | 07-10 18:54 |
| C3 | 2 | 07-11 22:20, 23:21 |
| C4 | 1 | 07-12 22:00 |
| C5 | 3 | 07-13 17:36, 17:57, 18:19 |
| C6 | 1 | 07-15 23:42 |
| C7 | 2 | 07-16 23:03, 23:20 |

**11 raw trade → 7 bağımsız olay.** Çıkarım birimi cluster.

### 2.2 Log-return muhasebesi

```
l(a,b) = ln(P_b / P_a)
R_short_log = −l(entry, exit)
net_log = R_short_log + ln(1 − k/1e4) + ln(1 + Σfunding_signed)   [SHORT funding ALIR: +r]
net_bps_simple = (exp(net_log) − 1)·1e4
```
**Reconciliation:** `|l(a,c) − (l(a,b)+l(b,c))|` max = **1.34e−16** → tam sıfır. Basit-getiri muhasebesindeki 11.61 bps I3/I4 sapması ortadan kalktı.

### 2.3 Politika kimlikleri (makine hassasiyetinde doğrulandı)

| kimlik | anlamı | max hata |
|---|---|---|
| **I1: P5 = P4 + P3** | "ilk LONG bacağının katkısı" P5−P3 **= P4** (tam) | 1.14e−13 |
| **I2: P1 + P2 = −2k** | aynı pencerede LONG+SHORT = iki RT maliyeti | 7.46e−14 |

### 2.4 Matched control

±7 gün penceresi, `hour≥17`, aynı BTC rejimi (`btc4h<0 OR btc7d<0`), sinyal ankorlarının ±6h komşuluğu dışlanmış, 10dk grid. **Kontrol satırları bağımsız gözlem sayılmadı** — olay başına tek baseline üretir.

### 2.5 Maliyet modeli

| kalem | değer | kaynak |
|---|---|---|
| fee | 5.0 bps / RT | repo standardı (mark-fill) |
| spread @T0 (cascade) | **0.0760 bps** | `book_ticker.spread_pct`, ölçüldü |
| spread @kontrol | **0.0617 bps** | ölçüldü, n=40 |
| **T0 execution disadvantage** | **+0.0142 bps** | alpha'dan düşüldü |
| funding | gerçek signed rate | LONG öder / SHORT alır |
| **boyut-bağımlı market impact** | **NOT_MODELED** | `ORDER_NOTIONAL` guardrail'li, okunmadı |

### 2.6 NA / censoring kuralları

- Giriş/çıkış fiyatı ±2dk toleransta yoksa → **NA** (doldurma / sentetik fiyat / en-yakın-tick ile boşluk kapatma **YOK**)
- Çıkış > DB sonu (07-17 13:26) → **CENSORED**
- 07-13 21:07 → 07-14 12:21 (15.2h) kesinti boşluğuna düşen hücre → **GAP_NA**
- complete_n / censored_n / gap_NA_n her uzun ufukta ayrı raporlanır

---

## 3. LONG tarafı bulguları

### 3.1 Ufuk taraması (T0 LONG, 11 anchor)

| ufuk | n | WR | ort | alpha (vs kontrol) | alpha CI95 |
|---|---|---|---|---|---|
| 1m | 11 | 18% | −9 | −10 | [−18, −3] |
| 5m | 11 | 36% | −6 | −11 | [−22, −1] |
| 1h | 11 | 45% | −4 | −11 | [−39, +18] |
| 4h | 8 | 38% | −20 | −38 | [−80, +6] |
| **6h (canlı kural)** | 8 | **25%** | **−47** | **−41** | [−96, +15] |
| 24h | 8 | 62% | +231 | +64 | [−240, +391] |
| 48h | 6 | 100% | +610 | +325 | [−18, +731] |
| 1w | 1 | — | +182 | — | — |

**Hiçbir ufukta alpha CI95'i sıfırı pozitif yönde dışlamıyor.** 24h/48h'teki artı, 07-13 rali üçlüsünden geliyor: rali çıkarılınca 24h **+1844 → −49**.

### 3.2 Negatif kontrol

Aynı pencerede gözü kapalı giriş (10dk grid, n=1476–1633):

| ufuk | sinyal | kontrol | fark |
|---|---|---|---|
| 1m–1h | −9…−4 | −5 | **−7…+3** |
| 24h | +231 | +16 | +215 |
| **1w** | **+182** | **+486** | **−303** |

1 haftada gözü kapalı giriş **sinyali yeniyor**. Kısa ufuklarda fark fee mertebesinde gürültü.

### 3.3 Pre-entry analizi

`pre_signal_ret` ortalaması: T−1w **+354**, T−24h −93, T−6h **−64**, T−1m **−13**. Sinyal, uzun vadeli yükseliş içindeki kısa vadeli bir düşüşün dibinde ateşleniyor.

T0'a göre katkı (6h ufku): T−1m **−15** [−21,−9], T−5m **−23** [−32,−10], T−15m **−28** [−40,−15] → **erken girmek CI-anlamlı biçimde kötü.** `SIGNAL_APPEARS_LATE_TO_MOVE` reddedildi; T0 uygulanabilirlerin en iyisi. Ama "timing adds value" ≠ "alpha var" — T0 alpha'sı 6h'te hâlâ −41.

### 3.4 LONG→FLAT→SHORT geçiş geometrisi

- **P5 − P3 = P4** (kimlik I1): ilk LONG bacağı **uygulanabilir gridin tamamında negatif** (−4…−47). Brüt bacak ≈ **+1 bps** (düz) → zarar bilgi eksikliğinden değil, fazladan bir round-trip'ten.
- **P5 − P2** ≈ 2·P4 + k: her hücrede negatif (−4…−89)
- **P3 − P2**: ağırlıkla ≤0 → beklemek de yardımcı olmuyor
- Sıralama: **P2 ≥ P3 ≥ P5**
- LONG başlangıcı hüküm tablosu (154 hücre): **GEREKLİ 0** · NÖTR 21 · KARARSIZ 131 · ZARARLI 2
- LOCO: merkez bölge (30m/45m/1h × 6h/7h/8h) **7/7'de işaret koruyor** → tek-event artefaktı değil, ama CI_alt ort **−40**, %3 hücre CI-pozitif

---

## 4. DIRECT SHORT (P2) audit

### 4.1 AUDIT 1 — Sample count reconciliation (BLOCKING, DÜZELTİLDİ)

**Hata:** `eff_n` **tüm 7 cluster** üzerinden hesaplanıyordu, `cluster_n` ise yalnız veri olanları sayıyordu → `eff_n > cluster_n` ihlali.

**Düzeltilmiş tanım:**
```
S_h = { cluster i : en az bir üyesinde horizon h'de VERİ VAR }
W_i = [ min_j(entry_ij) , max_j(entry_ij) + h ]      ∀ i ∈ S_h
M_h = merge_overlapping({ W_i })
effective_independent_n(h) = |M_h|
```
Bu tanımla `|M_h| ≤ |S_h| = cluster_n` **zorunlu** olarak sağlanır.

| hold | raw_n | cluster_n | eff_n ESKİ | eff_n DÜZELT | ihlal |
|---|---|---|---|---|---|
| 1m–3h | 10–11 | 7 | 7 | 7 | — |
| **4h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **5h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **6h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **7h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **8h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **10h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **12h** | 8 | 6 | **7** | **6** | ESKİ İHLAL |
| **18h** | 7 | 6 | **7** | **6** | ESKİ İHLAL |
| 24h | 8 | 5 | 4 | 5 | — |
| 36h | 8 | 5 | 3 | 4 | — |
| 48h | 6 | 4 | 3 | 3 | — |
| 72h | 8 | 5 | 2 | 2 | — |
| 1w | 1 | 1 | 1 | 1 | — |

**8 ufukta ihlal. Düzeltilmiş tabloda ihlal: 0.**

**Etki:** eski aile filtresi `cluster_n≥6 AND eff_n≥7` idi. Bozuk `eff_n` kısa ufuklarda hep 7 döndüğü için **4h–18h aileye YANLIŞ girdi**. Düzeltilmiş `eff_n=6` ile aynı filtre onları dışlardı → **max-T ailesi değişti, FWER yeniden hesaplandı.**

### 4.2 AUDIT 2 — MICRO / MEDIUM aile ayrımı

**MICRO ailesi (1m, 2m, 5m):**

| hold | raw | clu | eff | alpha | T | cluster_p | CI95 | sign_cons |
|---|---|---|---|---|---|---|---|---|
| 1m | 11 | 7 | 7 | +10 | **6.67** | **0.0000** | [+3, +18] | 86% |
| 2m | 11 | 7 | 7 | +11 | **5.32** | **0.0000** | [+2, +20] | 86% |
| 5m | 11 | 7 | 7 | +11 | 2.54 | 0.0039 | [+1, +22] | 71% |

max-T null: medyan 0.54, %95 **1.94**, max 4.90 → **FWER p = 0.0000 → ANLAMLI ✅**

**MEDIUM ailesi (6h, 7h, 8h, 10h, 12h, 18h):**

| hold | raw | clu | eff | alpha | T | cluster_p | CI95 | sign_cons |
|---|---|---|---|---|---|---|---|---|
| 6h | 8 | 6 | 6 | +41 | 1.01 | 0.1601 | [−16, +95] | 67% |
| 7h | 8 | 6 | 6 | +70 | 1.71 | 0.0406 | [−3, +142] | 83% |
| 8h | 8 | 6 | 6 | +77 | 1.88 | 0.0266 | [−2, +149] | 83% |
| **10h** | 8 | 6 | 6 | **+107** | **2.01** | 0.0192 | [+7, +208] | 83% |
| 12h | 8 | 6 | 6 | +90 | 1.39 | 0.0780 | [−18, +190] | 83% |
| 18h | 7 | 6 | 6 | +6 | 0.07 | 0.4740 | [−165, +207] | 50% |

max-T null: medyan 0.59, %95 **2.03**, max 3.51 → **FWER p = 0.0523 → ANLAMLI DEĞİL ❌**

> ### 🔑 TEK CEVAP
> **SORU:** 6h–18h ailesi, 1m/2m/5m tamamen çıkarıldığında family-wise anlamlı mı?
> **CEVAP: HAYIR.** FWER p = **0.0523**. En iyi ufkun T'si (10h, T=2.01) aile max-T eşiğinin (%95 = 2.03) **altında kalıyor.**

### 4.3 AUDIT 3 — 10h hücresinin tek başına exact testi

**Event-level paired alpha:** 07-04 **+71** · 07-10 **−85** · 07-11 22:20 **+47** · 07-11 23:21 **+12** · 07-12 **+85** · 07-13 ×3 **NA** · 07-15 **+274** · 07-16 23:03 **+293** · 07-16 23:20 **+249**

**Cluster-level:** C1 +71 · C2 −85 · C3 +29 · C4 +85 · **C5 TAMAMI NA** · C6 +274 · C7 +271

| test | sonuç |
|---|---|
| observed_cluster_n / missing_cluster_n | **6 / 1** |
| **exact randomization (sign-flip, 2⁶=64)** | **p = 0.0781** ❌ |
| sign test (5/6 pozitif) | **p = 0.1094** ❌ |
| Wilcoxon signed-rank exact (W=17) | **p = 0.1094** ❌ |
| bootstrap CI95 | [+7, +208] (mean +107, medyan +78) |

**Bootstrap CI yanıltıcı:** n=6'da bootstrap anti-konservatif; exact randomization p=0.0781 gerçek belirsizliği gösteriyor.

**LOCO (10h):**

| çıkarılan | n | mean | medyan | exact_p | sign_p | CI95 |
|---|---|---|---|---|---|---|
| C2 | 5 | +114 | +85 | 0.1250 | 0.1875 | [−5, +235] |
| **C3** | 5 | +146 | +85 | **0.0312** | **0.0312** | [+57, +235] |
| C4 | 5 | +123 | +85 | 0.1250 | 0.1875 | [+11, +235] |
| C5 | 5 | +111 | +71 | 0.1250 | 0.1875 | [−8, +232] |
| C6 | 5 | +74 | +71 | 0.1562 | 0.1875 | [−20, +185] |
| C7 | 5 | +74 | +71 | 0.1562 | 0.1875 | [−20, +187] |

Yalnız C3 çıkarıldığında p<0.05. **Diğer 5 çıkarımda p ≥ 0.125.** 10h hücresi tek başına ayakta durmuyor.

### 4.4 AUDIT 4 — C5 missing-cluster sensitivity bounds

**C5 durumu (10h):** 3 raw trade, çıkışları 07-14 03:36 / 03:57 / 04:19 → hepsi **kesinti boşluğunun içinde**.
`observed_cluster_n=6` · `missing_cluster_n=1` · `censored_cluster_n=0` · **`gap_NA_cluster_n=1`**

> ⚠️ Aşağıdaki tablo **yalnız missing-cluster sensitivity**'dir. İmputasyon **değildir**, gerçek gözlem **değildir**. "C5 çıkarılınca değişmiyor" ifadesi robustness kanıtı **olarak kullanılamaz** — C5 zaten gözlenmemiştir.

| C5 varsayımı | n | mean | medyan | sign_cons | exact_p | CI95 |
|---|---|---|---|---|---|---|
| **−500** | 7 | **+18** | +71 | 71% | **0.4297** | [−177, +182] |
| −300 | 7 | +48 | +71 | 71% | 0.2891 | [−101, +182] |
| −200 | 7 | +62 | +71 | 71% | 0.1797 | [−57, +183] |
| −100 | 7 | +77 | +71 | 71% | 0.1406 | [−22, +183] |
| **0** | 7 | +92 | +71 | 71% | **0.0781** | [+2, +183] |
| +100 | 7 | +106 | +85 | 86% | 0.0391 | [+21, +194] |
| +200 | 7 | +120 | +85 | 86% | 0.0391 | [+27, +210] |
| +500 | 7 | +162 | +85 | 86% | 0.0391 | [+37, +307] |

**10h sonucu, gözlenmemiş tek bir cluster'ın değerine göre p=0.0391 ile p=0.4297 arasında salınıyor.** Sonuç bu cluster'a karşı belirlenmemiş durumda.

### 4.5 Event-aligned impulse response (T0 SHORT)

| t (dk) | A(t) | ΔA | Δ²A | poz% |
|---|---|---|---|---|
| 1 | **+10** | +10 | — | 55% |
| 5 | +11 | +0 | −1 | 73% |
| 30 | +3 | −10 | −13 | 55% |
| 120 | +7 | −8 | −14 | 45% |
| 240 | +38 | +14 | −2 | 50% |
| 420 | +70 | +29 | +18 | 75% |
| **600** | **+107** | +30 | +23 | **88%** |
| 720 | +90 | −17 | −47 | 88% |
| 1080 | +6 | −87 | −91 | 43% |
| 1440 | −62 | — | — | 25% |

- ilk alpha>0: **t=1 dk** · maksimum **t=600 dk (+107)** · %90 maksimuma ilk ulaşma **t=600 dk**
- pozitif alan integrali **+626**, negatif alan **−0**
- A(t) ≥ %90·max olan nokta: **1/19 → TEK TEPE**, geniş plato **değil**
- İki ayrı yapı: t=1–5 dk keskin küçük sıçrama (+10/+11) ve t=420–720 dk geniş tümsek (+70…+107); arası düz

### 4.6 Stop/TP duyarlılığı (⚠ tümü araştırma varsayımı)

150 kombinasyon (5 stop × 6 TP × 5 maxhold). Pozitif: **98/150 = %65**.

| stop | tp | maxh | net | WR | stop% | CI95 |
|---|---|---|---|---|---|---|
| 150 | 100 | 10h | +35 | 71% | 11% | [−48, +95] |
| **300** | **100** | **10h** | **+71** | 86% | 0% | **[+21, +95]** |
| **300** | **200** | **10h** | **+98** | 83% | 0% | **[+21, +163]** |
| 300 | 400 | 10h | +80 | 83% | 0% | [+6, +145] |

Dar stop (150) tutarlı kötüleştiriyor; 300bps'te stop hiç tetiklenmiyor. Plato TP'ye karşı düz. Ex-post en iyi (⚠strateji değil): stop=400/tp=200/maxh=12h → +99.

### 4.7 Rejim diagnostic (⚠ KANIT DEĞİL; medyan split, n=7 cluster)

16/16 alt-rejimde 10h SHORT alpha işareti **pozitif** (eth1h, eth4h, btc4h, btc7d, vol, liq, spread, fund). Tek rejime bağımlılık yok — ama örneklem 12 günlük tek bir ETH yükseliş rejiminden geliyor, düşüş rejimi hiç görülmedi.

---

## 5. Geri çekilen hüküm

**`DIRECT_SHORT_ALPHA_SUPPORTED` (önceki tur) GERİ ÇEKİLDİ.**

Gerekçe:
1. **Count reconciliation hatası** — aile `cluster_n≥6 AND eff_n≥7` ile tanımlanmıştı; bozuk `eff_n` 4h–18h ufuklarını aileye yanlış soktu. Aile bileşimi hatalıydı.
2. **Aile ayrıştırılmamıştı** — FWER p=0.0000, MICRO (1m, T=6.67) tarafından taşınıyordu. MEDIUM tek başına p=**0.0523**.
3. **10h hücresi exact testlerde düşüyor** — randomization p=0.0781, sign p=0.1094, Wilcoxon p=0.1094. Bootstrap CI [+7,+208] n=6'da anti-konservatifti.

---

## 6. HÜKÜMLER

### Birincil: `DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT`

Tüm istatistiksel destek **1m/2m/5m** ailesinden geliyor (FWER p=0.0000, T=6.67). Bu aile çıkarıldığında MEDIUM ailesi family-wise anlamlı **değil** (p=0.0523). MICRO etkisi ise tam olarak execution'a en kırılgan olan: cascade anında 1 dakikalık scalp, **boyut-bağımlı market impact NOT_MODELED**, ve kitabın karşı tarafında cascade'in kendisi var.

### Alt hükümler

| alan | hüküm |
|---|---|
| **micro 1m–5m** | `SUPPORTED_STATISTICALLY_PENDING_EXECUTION_AUDIT` — FWER p=0.0000, 7/7 LOCO pozitif; ama AUDIT 7 (book_ticker executable fill, notional $1k–$100k) **YAPILMADI** |
| **medium 6h–18h** | `NOT_FAMILY_WISE_SUPPORTED` — FWER p=0.0523 |
| **10h fixed candidate** | `FAILS_EXACT_TEST` — randomization p=0.0781, sign/Wilcoxon p=0.1094; C5 sensitivity p∈[0.0391, 0.4297] |
| **execution realism** | `NOT_ASSESSED` — AUDIT 7/8 yapılmadı; boyut-bağımlı impact NOT_MODELED |
| **direction semantics** | `NOT_AUDITED` — AUDIT 9 yapılmadı; liquidation BUY/SELL semantiği ve historical/live yön eşlemesi **doğrulanmadı** |
| **historical artifact readiness** | `NOT_LOCATED` — AUDIT 10 yapılmadı; ~4MB HOUR17/history dosyası aranmadı |
| **1 gün** | `PATH_UNSTABLE` — alpha −62, CI [−377,+242], cluster_p=0.7524, eff_n=4; C5 çıkınca işaret dönüyor |
| **1 hafta** | `INSUFFICIENT_INDEPENDENT_EVENTS` — complete_n=1, censored_n=10, **eff_n=1**; CI hesaplanamıyor; max-T ailesinden dışlandı (dahil edilseydi tek pencere FWER'i domine ederdi, sahte p=0.0001 üretirdi) |

---

## 7. Yapılmayan audit maddeleri (açık)

| # | madde | statü |
|---|---|---|
| 5 | Time-of-day / session control (aynı UTC saat ±15/±30/±60 dk, weekday, funding-cycle, kombine) | **NOT_EXECUTED** |
| 6 | Same-day placebo (T0−6h…T0−1h, T0+1h…T0+6h) | **NOT_EXECUTED** |
| 7 | Micro execution audit (book_ticker bid/ask, $1k–$100k) | **NOT_EXECUTED** |
| 8 | 10h execution audit (notional-dependent) | **NOT_EXECUTED** |
| 9 | Direction semantics audit (7 katman) | **NOT_EXECUTED** |
| 10 | Historical ~4MB artifact discovery | **NOT_EXECUTED** |

**Kritik açık soru (AUDIT 5):** 10h/600dk tümseği gece→sabah UTC seans geçişine denk geliyor (girişler 17:36–23:42 UTC, çıkışlar 03:36–09:42 UTC). Mevcut kontrol yalnız `hour≥17` ile eşleşiyor — bu 7 saatlik bir pencere ve seans etkisini **absorbe etmemiş olabilir**. Bu audit yapılmadan MEDIUM ailesinin p=0.0523'ü bile iyimser sayılmalıdır.

> **GÜNCELLEME (2026-07-17):** AUDIT 5 ve 7 tamamlandı — §10, §11. Yukarıdaki kritik açık soru **çözüldü ve MEDIUM aleyhine sonuçlandı.**

---

## 10. AUDIT 5 — TIME-OF-DAY / SESSION CONTROL

**Amaç:** 10h SHORT alpha'sı HOUR17 sinyalinden mi, yoksa girişlerin gece / çıkışların sabah UTC olmasından mı geliyor?

### 10.1 Event saatleri — sistematik seans örüntüsü

| T0 (UTC) | wd | fc_pos | entry_ses | **exit_ses (10h)** | btc4h | eth4h |
|---|---|---|---|---|---|---|
| 07-04 23:14 | Sat | 7 | LATE | EU | −18 | −62 |
| 07-10 18:54 | Fri | 2 | US | ASIA | −37 | −20 |
| 07-11 22:20 | Sat | 6 | LATE | EU | −6 | −47 |
| 07-11 23:21 | Sat | 7 | LATE | EU | −41 | −112 |
| 07-12 22:00 | Sun | 6 | LATE | EU | −40 | −68 |
| 07-13 17:36 | Mon | 1 | US | ASIA | −94 | −35 |
| 07-13 17:57 | Mon | 1 | US | ASIA | −47 | −62 |
| 07-13 18:19 | Mon | 2 | US | ASIA | −92 | −108 |
| 07-15 23:42 | Wed | 7 | LATE | EU | −36 | −52 |
| 07-16 23:03 | Thu | 7 | LATE | EU | −51 | −46 |
| 07-16 23:20 | Thu | 7 | LATE | EU | −56 | −60 |

**Tüm girişler 17:36–23:42 UTC (US/LATE), tüm 10h çıkışları 03:36–09:42 UTC (ASIA/EU).** 10h politikası sistematik olarak gece→sabah seans geçişini örneklüyor.

### 10.2 Kontrol aileleri (±7g, gerçek event ±6h dışlandı, event başına tek baseline)

| aile | tanım | havuz/event (min-med-max) | **10h alpha** | exact_p | T |
|---|---|---|---|---|---|
| A | aynı UTC saat ±15dk | 14-40-84 | +99 | 0.1250 | 2.00 |
| B | aynı UTC saat ±30dk | 26-73-156 | +97 | 0.1250 | 1.95 |
| C | aynı UTC saat ±60dk | 50-145-300 | +96 | 0.1250 | 1.87 |
| **D** | aynı weekday + saat ±30dk | **0-13-13** ⚠ | +158 | 0.0312 | 12.93 |
| E | aynı funding-cycle + saat ±30dk | 14-50-108 | +101 | 0.1250 | 1.97 |
| **F** | saat ±30dk + weekday + BTC4h | **0-0-13** ⚠ | +204 | 0.0625 | 30.06 |
| **G** | saat ±30dk + ETH4h rejimi | 7-14-93 | +170 | 0.0625 | 3.81 |
| H | aynı entry+exit session | 40-136-264 | +96 | 0.1250 | 1.90 |

**Sıkı zaman eşleşmesi (A/B/C/E/H)** — hepsinde exact_p = 0.1250, sign_p = 0.3438, Wilcoxon = 0.1562, **anlamlı DEĞİL.** D/F/G'nin düşük p'leri **spurious**: havuzları dejenere (D bazı event'lerde 0, F bazı event'lerde **0 kontrol** → null SD çöküyor, T yapay şişiyor 13–30). Bunlar gerçek sinyal değil, dar-havuz artefaktı.

### 10.3 MEDIUM ailesi FWER (studentize max-T), kontrol ailesine göre

| kontrol ailesi | en iyi ufuk | T | max-T null %95 | **FWER p** | |
|---|---|---|---|---|---|
| A (saat ±15) | 7h | 2.07 | 2.43 | **0.1207** | ❌ |
| B (saat ±30) | 7h | 2.04 | 2.43 | **0.1247** | ❌ |
| C (saat ±60) | 7h | 1.99 | 2.42 | **0.1450** | ❌ |
| E (funding-cycle) | 7h | 2.11 | 2.33 | **0.0860** | ❌ |
| H (session-matched) | 7h | 1.98 | 2.36 | **0.1327** | ❌ |
| D ⚠dejenere | 10h | 13.30 | 2.64 | 0.0000 | spurious |
| F ⚠dejenere | 7h | 31.43 | 2.14 | 0.0000 | spurious |
| G ⚠dar | 8h | 3.93 | 2.24 | 0.0000 | şüpheli |

**Geçerli havuzlu tüm ailelerde (A/B/C/E/H) MEDIUM FWER anlamlı DEĞİL** (p=0.086–0.145). §4'teki hour≥17 kontrolüyle p=0.0523 idi; sıkılaştırınca daha da zayıflıyor.

### 10.4 Same-day placebo — belirleyici test

Her event için aynı gün T0−6h…T0−1h ve T0+1h…T0+6h pencerelerinden placebo (event çevresi hariç), 10.000 cluster-level set:

| | değer |
|---|---|
| gerçek T0 alpha | +97 bps |
| same-day placebo null | medyan **+107**, %95 +155, max +204 |
| **gerçek T0 percentile** | **37.2** |
| **p-value** | **0.6276** |

**Aynı gün rastgele bir saatte SHORT açmak, HOUR17 sinyal saatinden DAHA İYİ.** Nearest-time placebo (aynı gün ±15/30/60dk): gerçek percentile **0.2 / 1.1 / 3.3**, p = 0.998 / 0.989 / 0.967 — gerçek sinyal, kendi saatinin hemen komşusundaki girişlerin neredeyse tamamının **altında**.

### 10.5 AUDIT 5 hükmü: `MEDIUM_SHORT_FAILS_EXACT_TIME_CONTROL`

Sıkı-zaman kontrollerinde MEDIUM FWER anlamlı değil (p=0.086–0.145); same-day placebo'da gerçek sinyal 37. persentilde (p=0.63) ve nearest-time placebo'da komşu girişlerin %97–99.8'inin altında. 10h etkisi bir **seans/gün-içi-zaman etkisidir, HOUR17 sinyaline atfedilemez.** D/F/G'nin "anlamlı" görünmesi dejenere kontrol havuzu artefaktıdır.

---

## 11. AUDIT 7 — MICRO EXECUTION AUDIT

**Amaç:** 1m/2m/5m alpha'sı `book_ticker` executable bid/ask ve notional etkisi altında kalıyor mu?

**Model:** SHORT giriş = SELL @ executable **bid**(entry+latency); SHORT çıkış = BUY @ executable **ask**(exit+latency). Mark price kullanılmadı. `book_ticker` yalnız **top-of-book** → çok seviyeli derinlik YOK → notional top-of-book'u aşarsa **UNKNOWN/UPPER_BOUND** (optimistic fill üretilmedi).

### 11.1 Event snapshot (latency=0)

11 event'in tamamında spread **0.05–0.06 bps**, hiçbiri crossed/locked değil. Top-of-book notional (bidq_usd) $8.5k–$343k arası — bazı event'lerde ($5k üstü) derinlik dar.

### 11.2 Executable net SHORT (hold=1m) — mark-price +10 bps'in çöküşü

| latency (ms) | $1k | $10k | $100k | feasible% |
|---|---|---|---|---|
| 0 | **−2.3** | −2.3 | −2.3 | 80% |
| 500 | −2.0 | −2.0 | −2.0 | 74% |
| 2000 | −1.0 | −1.0 | −1.0 | 71% |
| 5000 | −0.2 | −0.2 | −0.2 | 71% |

Mark-price 1m getirisi ~+10 bps idi; **executable bid/ask + fee ile −2.3 bps.** Spread'i gerçekten geçmek (mid-to-mid değil) +10'u siliyor. (Not: gecikme arttıkça net "iyileşiyor" çünkü fiyat o pencerede SHORT lehine düşmeye devam ediyor — bu bir edge değil, aynı yön sürüklenmesi.)

### 11.3 Executable paired alpha (SHORT, bid/ask, vs executable kontrol)

| latency | 1m alpha | exact_p | sign_p | sign_cons |
|---|---|---|---|---|
| 0 | **+3.4** | 0.1719 | 0.5000 | 57% |
| 500 | +3.5 | 0.1797 | 0.5000 | 57% |
| 2000 | +4.9 | 0.1250 | 0.5000 | 57% |

**Mark-price'ta 1m alpha +10 (studentize T=6.67, cluster_p=0.0000) idi; executable bid/ask ile +3.4 bps, exact_p=0.17, sign consistency %57.** İstatistiksel anlamlılık kayboluyor. §4.2'deki FWER p=0.0000, mark-to-mid ölçümünün artefaktıymış.

### 11.4 Break-even adverse impact

| adverse (bps/leg) | net alpha | işaret |
|---|---|---|
| 0 | +3.4 | + |
| 1 | +1.4 | + |
| **2** | **−0.6** | **−** |
| 5 | −6.6 | − |

**Break-even = 1.72 bps/leg.** Her bacakta >1.7 bps ek maliyet alpha'yı sıfırlar.

### 11.5 Depth feasibility (top-of-book)

| notional | feasible | not |
|---|---|---|
| $1k | 11/11 | top-of-book yeter |
| $5k | 11/11 | top-of-book yeter |
| $10k | **9/11** | DERİNLİK-BİLİNMEZ (UNKNOWN) |
| $25k | 8/11 | UNKNOWN |
| $50k | 8/11 | UNKNOWN |
| $100k | **6/11** | UNKNOWN |

$10k üstünde event'lerin en az 2'si top-of-book'ta dolmuyor; çok seviyeli derinlik yok, o event'lerin gerçek fill fiyatı bilinmiyor.

### 11.6 AUDIT 7 hükmü: `MICRO_SHORT_FAILS_EXECUTION_MODEL`

Executable bid/ask ile 1m net getiri **negatif** (−2.3 bps); executable paired alpha yalnız +3.4 bps (exact_p=0.17, sign consistency %57 — anlamsız); break-even adverse impact 1.72 bps/leg gibi kırılgan bir eşik; ve $10k üstünde derinlik bilinmiyor. Cascade anında bir dakikalık scalp'te 1.7 bps'i aşan slippage/impact gerçekçi. Mark-price'taki +10 bps (FWER p=0.0000) execution altında hayatta kalmıyor.

---

## 12. BİRLEŞİK NİHAİ HÜKÜM: `BOTH_FAIL`

İki bağımsız falsification audit'i, DIRECT SHORT desteğinin kalan iki dayanağını da ayrı ayrı düşürdü:

- **MEDIUM (6h–18h):** `MEDIUM_SHORT_FAILS_EXACT_TIME_CONTROL` — seans/zaman etkisi, sinyal değil. Same-day placebo 37. persentil (p=0.63).
- **MICRO (1m–5m):** `MICRO_SHORT_FAILS_EXECUTION_MODEL` — executable fiyatlarda net −2.3 bps, alpha +3.4 (p=0.17), break-even 1.72 bps/leg.

Bu iki başarısızlık **bağımsız ve farklı sebeplerden**: MEDIUM istatistiksel (zaman confound), MICRO execution (spread + derinlik). Hiçbiri diğerini kurtarmıyor. `DIRECT_SHORT_SUPPORT_DEPENDS_ON_MICRO_EFFECT` (§4) hükmü, o mikro etkinin execution altında yok olmasıyla **boşa düşüyor.**

**DIRECT SHORT bir preregistration adayı değildir.** Bu, "SHORT çalışmaz" demek değil — 7 bağımsız olay, tek varlık, tek rejimde **yeni bir yön için kanıt yok** demektir. HOUR17 LONG tarafı da desteklenmiyordu (§3: 6h alpha −41). Net durum: **bu forward pencerede hiçbir yönde harvestable edge kanıtlanamadı.**

Hâlâ yapılmamış: AUDIT 9 (direction semantics — liquidation BUY/SELL → pressure → position eşlemesi) ve AUDIT 10 (~4MB historical artifact). Bunlar `BOTH_FAIL` hükmünü değiştirmez ama HOUR17'nin tarihsel 93-cycle kanıtının yön bütünlüğünü ayrıca doğrular.

---

## 8. Kalıcı sınırlamalar

- **7 bağımsız olay** (uzun ufuklarda efektif 6/5/4/3/2/1)
- **12 günlük tek rejim** (ETH 1751→1826 yükselişi); düşüş rejimi hiç görülmedi
- **Tek varlık** (ETHUSDT)
- **C5 (3 raw trade)** 10h civarındaki hücrelerde yapısal olarak gözlenmemiş
- **SHORT 300bps stop** araştırma varsayımı; live'da hour17 SHORT route'u yok
- **Boyut-bağımlı market impact** modellenmedi
- Forward binding `min_sample=20` istiyor; gerçek bağımsız olay sayısı **7**

Bu rapor "HOUR17 LONG ölü" veya "DIRECT SHORT çalışıyor" demiyor. **Mevcut forward verinin tarihsel iddiayı desteklemediğini**, ve DIRECT SHORT alternatifinin desteğinin execution-kırılgan bir mikro etkiye bağlı olduğunu söylüyor.

---

## 9. Kaynaklar

| ne | nerede |
|---|---|
| Shadow ledger | `reports/shadow/s34_state_machine_shadow.jsonl` |
| Shadow state | `reports/shadow/s34_state_machine_shadow_state.json` |
| Fiyat verisi | `data/microstructure.db` (`mode=ro`): `mark_prices`, `book_ticker`, `liquidations`, `agg_trades` |
| Shadow route | `tools/s34_realtime_shadow_runner.py:612-680` |
| Live route | `tools/s34_state_machine_live_executor.py:70,678` |
| Live executor state | `runtime/s34_live_executor_state.json` |
| Route tanımı | `SYSTEM_STATE.md` §28 |
| Cycle-adjusted recompute | `SYSTEM_STATE.md` §104, `S34_HOUR17_CYCLE_ADJUSTED_RECOMPUTE_AND_MAY_GAP_FORENSIC_2026-07-11.md` |
| Forward binding | `E-HOUR17-FWD-001` (`ami/run_forward_pipeline.py`) |
| Sonuç tabloları | `S34_HOUR17_DIRECT_SHORT_AUDIT_2026-07-17.sql` |

**Reproducibility:** tüm testlerde `random.seed(20260717)`; bootstrap 5000–20000 resample; placebo 10000 set; exact testler tam enumerasyon (2⁶=64, 2⁵=32).
