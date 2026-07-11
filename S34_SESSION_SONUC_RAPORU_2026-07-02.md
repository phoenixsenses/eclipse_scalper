# S34 Oturum Sonuç Raporu — Çeşitlendirme + Sinyal Keşfi + Veri Düzeltmesi (2026-07-02)

> Kapsam: (1) conviction-weighted sizing / feature-interaction / premium-sleeve kombinasyonları
> SQL meta-analiz + taze backtest, (2) yeni T0 sinyal keşfi + eski adayların holdout doğrulaması,
> (3) açık soruların testi + tespit edilen problemlerin çözümü.
> Mezarlığa girilmedi: buy-side fade / reversal / cross-asset transfer TEKRAR TEST EDİLMEDİ.
> Tüm testler research-only; live executor / `.env` / sizing / leverage DOKUNULMADI.
> Tek kod değişikliği: shadow runner rv kaynağı düzeltmesi (aşağıda, §5).

---

## 0. TL;DR — En Önemli 5 Bulgu

1. **🔴 PAZARTESİ VETOSU (en büyük yeni kazanç):** hour17 evreninde Pazartesi
   N=25, WR %32, avg −47.4; 100K evreninde N=27, WR %37, avg −42.4; TEST split'lerde
   daha da kötü (WR %18-25). İki bağımsız evren + iki split = tutarlı zehir.
   `base + notMonday` tek başına TEST'te WR %90, RA 17.97. **Live gate'e eklenmesi önerilir
   (operatör sign-off).**
2. **🔴 VERİ PROBLEMİ BULUNDU + ÇÖZÜLDÜ:** `vol_state` tablosu 2026-06-05 19:00'dan beri ÖLÜ
   (producer silinmiş; repo'da yazan kod yok). Son ~27 günün TÜM event'leri bayat rv=0.253 ile
   rv-hit aldı (55/55; gerçekte 29/55 olmalıydı — %47 flip). Edge bayat rv'nin eseri DEĞİL:
   düzeltilmiş rv ile sonuçlar daha da iyi. Shadow runner'a staleness-aware mark-price proxy
   eklendi (restart gerekli).
3. **Yeni T0 sinyalleri (TEST-OOS geçti):** `funding<0` (WR 100%, Δ+116), `eth1h<−80bps derin
   düşüş` (WR 100%, Δ+107), `tsl<115dk clustering` (WR 81.8%, Δ+84, N=22), `two_sided` BUY-liq
   varlığı (WR 77.8%, Δ+66 — YENİ), `basis>0` (WR 86.7%, Δ+62 — YENİ).
4. **Risk-ayarlı şampiyonlar:** interaction `rv+shelf+whale_lo` (üçlü) RA=59.6, WR 91.7%,
   worst −35.4; sizing'de `double-trigger 3u` flat'in mdd'sini BÜYÜTMEDEN toplamı 2×'liyor
   (TEST RA 11.4 vs flat 2.4).
5. **Çürüyenler:** cascade gentleness (holdout'ta tutarsız — KAPANDI), taker1h (yön tutmadı),
   spread@T0 (dejenere), profit-target +200 (hafif iyi ama non-monotonik — kırılgan, düşük öncelik).

---

## 1. Görev 1 — Çeşitlendirme (SQL meta + taze backtest)

### 1a. SQL meta-analiz (S34_ALL.db, trust≥2, n≥8)
Rapor: `reports/research/s34/S34_DIVERS_META_SQL.md` — Script: `tools/research_s34_divers_meta_sql.py`

| Kategori | En iyi (risk_adj = TOT / max(|mdd|,50)) | RA |
|---|---|--:|
| Interaction | `I_rv+shelf` (N=45, WR 77.8, avg +87.9) | 39.6 |
| Premium sleeve | `M1_deep7d_s>=4` (N=28, WR 82.1, avg +111) | 48.6 |
| Portfolio | `funding_lo & sync_ratio_hi` combo (N=17, WR 100) | 37.1 |
| Sizing | SQL'de neredeyse yok → taze backtest gerekti | — |

### 1b. Taze backtest (hour17 200K composite, 127 event, 4.54 ay)
Rapor: `reports/research/s34/S34_DIVERSIFICATION.md` — Script: `tools/research_s34_diversification_gauntlet.py`

**Interaction (holdout + mdd):** en iyi risk-ayarlı üçlüler —
| Combo | N | WR | avg | worst | mdd | RA | TEST |
|---|--:|--:|--:|--:|--:|--:|---|
| **rv+shelf+whale_lo** | 24 | 91.7% | +124.1 | −35.4 | −35.4 | **59.6** | WR 92.9, avg +106.8 |
| sync+rv+shelf | 22 | 90.9% | +124.8 | −44.8 | −50.5 | 54.4 | WR 87.5, avg +104.9 |
| rv+shelf+be | 16 | 93.8% | +110.1 | −44.8 | −44.8 | 35.2 | WR 90.0, avg +80.2 |

**Sizing politikaları (score≥2 admit, no-overlap, N=45):**
| Politika | wTOT | perU | wMDD | RA | 15x hesap-sim | TEST RA |
|---|--:|--:|--:|--:|--:|--:|
| flat 1u | 2586 | 57.5 | −237 | 10.9 | 1.46x / −3.5% | 2.4 |
| unit=score | 9235 | 58.8 | −815 | 11.3 | 3.55x / −12.0% | 6.1 |
| sleeve 1/2/3 | 4628 | 60.9 | −416 | 11.1 | 1.93x / −6.2% | 9.9 |
| premium-only s4 2u | 2535 | 60.3 | −539 | 4.7 | 1.44x / −7.9% | **12.9** |
| **double-trigger 3u** (rv+shelf→2u, +whale→3u) | **5220** | **74.6** | **−237** | **22.0** | 2.11x / −3.5% | 11.4 |

> **Sonuç:** En yüksek risk-ayarlı sizing = **interaction-tetikli** (`double-trigger`): iki/üç sinyal
> birden geldiğinde birim artır. Flat ile AYNI drawdown'da (−237 / −3.5%) toplamı 2× yapıyor.
> `unit=score` en çok parayı getiriyor ama mdd 3.4×. Premium-only TEST'te en iyi RA ama N düşük.

---

## 2. Görev 2 — Sinyal Keşfi
Rapor: `reports/research/s34/S34_SIGNAL_DISCOVERY_V2.md` — Script: `tools/research_s34_signal_discovery_v2.py`

**Yeni sinyaller (TRAIN yön seçti → TEST raporlandı):**
| Sinyal | Yön | TEST N | TEST WR | TEST avg | Δ (anti'ye fark) | full mc |
|---|---|--:|--:|--:|--:|--:|
| **funding_rate** | <0 | 10 | 100% | +155.7 | **+116.1** | 0.0 |
| **eth1h** (1h getiri) | <−80bps | 8 | 100% | +154.8 | +107.4 | 0.0 |
| **tsl** (son anchor'dan süre) | <115dk | 22 | 81.8% | +106.0 | +84.0 | 0.0 |
| **two_sided** (pre-1h BUY liq) | ≥68K | 18 | 77.8% | +104.9 | +65.9 | 0.0 |
| **basis** (mark−trade) | >+0.9bp | 15 | 86.7% | +107.3 | +61.6 | 0.002 |
| predrop (sharp) | <−50bps | 13 | 69.2% | +79.6 | +15.3 | 0.0 |
| taker1h | — | — | — | — | −20.6 ❌ | — |
| spread@T0 | — | — | dejenere ❌ | — | — | — |

**Composite genişletme (S5):** tsl + two_sided eklenmiş **score9**: `s9≥6` TEST N=14, WR 92.9%,
avg +134.2 — `s7≥5`ten (WR 90.9, avg +97.2, N=11) hem N hem kalite olarak iyi.

**Doğrulama sonuçları:**
- ❌ **Gentleness KAPANDI:** full'de sharp daha iyi (mc 0.0 vs 0.042), TEST'te gentle — tutarsız = robust değil (28d "şekil robust değil" ile tutarlı).
- 🟡 Momentum-after-win: zayıf teyit (after_win mc 0.002 vs after_loss mc 0.104) — sadece navigation etiketi.
- 🟡 Profit-target +200: avg +63.8 vs fixed +58.9 — yön doğru ama pt150/pt250 kötü (non-monotonik) → kırılgan, düşük öncelik.
- **dow full taraması:** Pazar (dow=6) en iyi gün (WR 84.6, avg +104.3, mc 0.0); Pazartesi felaket (aşağıda).

---

## 3. Görev 3 — Rafine Tarif + Açık Sorular
Rapor: `reports/research/s34/S34_REFINED_RECIPE.md` — Script: `tools/research_s34_refined_recipe_gauntlet.py`

**Pazartesi (R1):** max-stat MC p=0.068 (7 gün düzeltmesiyle marjinal) AMA TEST bağımsız teyit:
Mon TEST N=8, WR 25%, avg −46.9, mc 0.982. + 100K evreninde bağımsız teyit
(`S34_100K_NOTMON.md`): Mon-only N=27, WR 37%, avg −42.4, TEST WR 18.2%, noov mc 0.954.
**İki evren + iki split tutarlı → gerçek etki, dow-fishing değil.**

**Tarif merdiveni (no-overlap, kümülatif):**
| Adım | N | /ay | WR | avg | TOT | RA | TEST |
|---|--:|--:|--:|--:|--:|--:|---|
| L0 base s7≥2 | 45 | 9.9 | 71.1% | +58.9 | 2648 | 11.2 | WR 75, RA 5.6 |
| **L1 +notMonday** | 38 | 8.4 | 78.9% | +79.0 | **3003** | **17.7** | **WR 90, RA 18.0, mc 0.002** |
| L2 score9≥4 | 26 | 5.7 | 73.1% | +68.7 | 1785 | 10.5 | WR 100, N=6 |
| L3 +funding<0 | 17 | 3.7 | 88.2% | +103.2 | 1754 | 11.4 | N=3 |
| L6 L2+double-trigger sizing | 26 | — | — | — | w4071 | 24.0 | — |

> **Ders:** Pazartesi bloğu tek başına en iyi TOTAL+RA artışı; skor sertleştirmek (L2/L3/L5)
> WR'yi artırıyor ama parayı düşürüyor (bilinen over-filter paterni). Doğru kombinasyon:
> **geniş admit (s≥2) + notMon + conviction-SIZING** (filtre değil).

**Portföy (R3):** rafine LONG + SHORT 13-17 confirm tek-slot: N=30, WR 73.3%, +70.2 avg, mc 0.0.
SHORT 13-17 kendi başına: N=11, WR 81.8%, avg +111.1, mc 0.006 (28i ile tutarlı).

**100K frekans yolu (`S34_100K_NOTMON.md`):** `100K + notMon + s≥3`: full N=100, /ay 22,
WR 74%, avg +81.6, TOT 8159, TEST WR 86.1% avg +95.9, noov 9.9/ay WR 71.1% mc 0.0 —
**100K composite + notMon şu an eldeki en iyi frekans+kalite dengesi.**

---

## 4. Veri Sağlığı (R4)

| Feed | Son yaş | Durum |
|---|--:|---|
| mark_prices | 0.1 dk | ✅ |
| liquidations | 1.0 dk | ✅ |
| book_ticker | 0.0 dk | ✅ |
| agg_trades | 0.0 dk | ✅ |
| **vol_state** | **38516 dk (~27 gün)** | 🔴 ÖLÜ — §5 |
| mark gap (>120s, son 7g) | 11 adet, max 786s | 🟡 orta — collector hıçkırıkları izlenmeli |

---

## 5. Problem Çözümü: vol_state / rv_5m Bayatlığı
Rapor: `reports/research/s34/S34_RV_STALE_FIX.md` — Script: `tools/research_s34_rv_stale_fix_validation.py`

- **Kök neden:** `vol_state` tablosuna yazan producer repo'da YOK (Mayıs'ta silinen detector
  zinciriyle gitti); tablo 2026-06-05 19:00'da üç sembolde birden durdu.
- **Etki:** `SELECT ... ts_ms<=? ORDER BY ts_ms DESC LIMIT 1` deseni sessizce son satırı döndürdü →
  5 Haziran sonrası 55 event'in 55'i rv-hit aldı (bayat 0.253 ≥ 0.0304). Doğrusu 29/55 (26 flip).
  TEST split tam bu döneme denk → rv'li TEST sonuçları şüpheliydi.
- **Yeniden doğrulama (düzeltilmiş rv = mark-price 1m log-return RMS(5m), TRAIN medyan eşik
  0.0026337):** Edge HAYATTA ve daha temiz —
  `s7≥3 proxy TEST`: WR 86.4%, avg +106.2 (bayat: 80%/+85.6);
  `rv+shelf proxy TEST`: WR 87.5%, avg +109.8, mc 0.004;
  `rv+shelf+whale proxy TEST`: WR 100%, avg +121.9. Bayat rv sinyali şişirmemiş, SULANDIRMIŞ.
- **Kod düzeltmesi (`tools/s34_realtime_shadow_runner.py`):** `rv5m_robust()` eklendi —
  vol_state satırı ≤10dk taze ise onu (eşik 0.0304), değilse mark-proxy (eşik 0.0026337) kullanır.
  `py_compile` ✓, `--once` dry-run EXIT=0 ✓. **Çalışan proses eski kodla dönüyor — restart
  operatör aksiyonu** (sandbox içi restart network bozuyor).

---

## 6. Operatör Aksiyonları (önem sırasıyla)

1. **[URGENT — süregelen]** Per-trade margin tail-budget'a indirilmeli (60x oversize hâlâ açık).
2. **[YENİ — yüksek etki]** `dow != Monday` vetosu hour17 LONG live gate'ine eklensin
   (iki evren + iki split kanıtlı; mevcut Mon/Wed bloğu sadece T15 route'unda var, hour17'de YOK).
   Operatör sign-off + live executor değişikliği gerektirir.
3. **[YENİ]** Shadow runner restart (rv düzeltmesinin devreye girmesi için):
   sandbox dışı `powershell -File start_eclipse.ps1`.
4. **[İZLE]** mark_prices gap'leri (7 günde 11×>120s) — collector stabilitesi.
5. **[ADAY — forward sonrası]** double-trigger conviction sizing (2u/3u) — 15x'te güvenli
   görünüyor (−3.5% mdd sim); önce shadow'da forward doğrulama.

## 7. Kapanan Hipotezler (bir daha test etme)

- Cascade gentleness/grind sinyali (holdout tutarsız)
- taker1h yönü, spread@T0 (dejenere)
- Bayat-rv şüphesi ("son dönem güçlü" bulgusu rv artefaktı DEĞİL — düzeltilmişle de güçlü)
- (Önceden kapalı, girilmedi: buy-side fade, reversal, cross-asset transfer, 00-13 UTC LONG)

## 8. Bu Oturumda Üretilen Dosyalar

| Tür | Dosya |
|---|---|
| Script | `tools/research_s34_divers_meta_sql.py` |
| Script | `tools/research_s34_diversification_gauntlet.py` |
| Script | `tools/research_s34_signal_discovery_v2.py` |
| Script | `tools/research_s34_refined_recipe_gauntlet.py` |
| Script | `tools/research_s34_rv_stale_fix_validation.py` |
| Script | `tools/research_s34_100k_notmon_check.py` |
| Kod fix | `tools/s34_realtime_shadow_runner.py` (rv5m_robust) |
| Rapor | `reports/research/s34/S34_DIVERS_META_SQL.{md,json}` |
| Rapor | `reports/research/s34/S34_DIVERSIFICATION.{md,json}` |
| Rapor | `reports/research/s34/S34_SIGNAL_DISCOVERY_V2.{md,json}` |
| Rapor | `reports/research/s34/S34_REFINED_RECIPE.{md,json}` |
| Rapor | `reports/research/s34/S34_RV_STALE_FIX.{md,json}` |
| Rapor | `reports/research/s34/S34_100K_NOTMON.{md,json}` |

---
*Tüm sonuçlar FEE=5bps net, MC=500 permütasyon, TRAIN/TEST=70/30 kronolojik. Guardrail ihlali yok.*
