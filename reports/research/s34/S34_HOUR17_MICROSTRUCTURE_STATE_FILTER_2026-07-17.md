# S34 HOUR17 — Microstructure State Filter (OOS Falsification)

**Tarih:** 2026-07-17
**Soru:** T0'a kadar görülebilen mikroyapı verisi, sonraki 6h'te continuation_down olacak ve LONG'un zarar edeceği event'leri önceden veto edebilir mi?
**Politika:** REVERSAL olasılığı yüksekse LONG 6h, aksi halde NO TRADE. SHORT yalnız diagnostic.
**Statü:** `COMPLETE — FROZEN OOS STUDY`
**Erişim:** SALT-OKUNUR. `data/microstructure.db` `mode=ro`, yalnız event-çevresi indeksli pencereler. Hiçbir DB/repo/runtime/parametre/strateji değişmedi. Reproducibility: `seed=20260717`.

> ⚠️ Bu rapor deploy önerisi DEĞİLDİR. Hiçbir feature deploy edilmedi, hiçbir eşik sonuçlara bakılarak optimize edilmedi.

---

## 0. Sample-count reconciliation (67 vs 74 çözüldü)

Önceki mikrodiag'daki 74, **hatalı** temsilci-dedup'tı (60–120s zincirlerde fazla sayıyordu). Kanonik sayım chain/transitive 60s merge = **67**.

| tanım | değer | kaynak |
|---|---|---|
| raw config-trade rows | 4144 | artefaktın 64 config × trade satırları |
| unique signal_ts | 141 | `signal_ts_ms` distinct |
| unique cascade (60s chain) | **67** | ardışık ≤60s signal_ts birleştir → canonical |
| unique feature event | 67 | her cascade-start = 1 event |
| cluster (≤2h) | 20 | cascade-start'lar ≤2h |
| non-overlap 6h pencere | 6 | |
| non-overlap 10h pencere | **4** | |
| distinct gün | **4** | 06-07, 06-11, 06-14, 06-15 |

**Zorunlu kimlik doğrulandı:** `eff10h(4) ≤ cluster(20) ≤ unique_cascade(67) ≤ unique_signal(141) ≤ raw(4144)`.

**Dondurulmuş canonical event_id** = `symbol + cascade_start + cascade_end (chain-60s)`. Aynı cascade'in farklı threshold/TP/delay config'leri ayrı sayılmadı.

---

## 1. Kohortlar (ayrı tutuldu)

| kohort | pencere | canonical event | gün | binary target (CONT/REV) |
|---|---|---|---|---|
| **A_JUNE** (historical) | 2026-06-07→06-15 | 67 | **4** | 31 (16 CONT / 15 REV) |
| **B_JULY** (forward, external holdout) | 2026-07-04→07-17 | 11 | 7 | **6 (5 CONT / 1 REV)** |

June üzerinde keşfedilen hiçbir eşik/model July'ya yeniden ayarlanmadı. July = untouched external holdout. June içinde calendar-day leave-one-day-out (aynı günün event'leri train/val'e bölünmedi).

**Çözünürlük uyarısı:** June yalnız **4 distinct gün** → leave-one-day-out = 4 fold, effective_n=4. Bu, herhangi bir OOS model doğrulaması için tabandan yetersiz. July'da **yalnız 1 REVERSAL** event var — sınıflandırıcının yakalayacağı pozitif sınıf holdout'ta neredeyse yok.

---

## 2. Dondurulmuş label'lar (feature'lardan sonra, yalnız fiyat yolundan)

Primary ekonomik eşik: `|LONG6h net| ≥ 20 bps`.

- **CONTINUATION** = LONG6h net < −20 ∧ yol ∈ {continuation_down, oscillatory}
- **REVERSAL** = LONG6h net > +20 ∧ yol ∈ {reversal_up, bounce_up}
- **UNCERTAIN** = |net|<20 (SMALL) ∨ gap ∨ işaret/yol uyuşmazlığı (MISMATCH)

| kohort | REVERSAL | CONTINUATION | UNCERTAIN (small/mismatch/gap) |
|---|---|---|---|
| JUNE | 24 | 17 | 26 (11/15/0) |
| JULY | 1 | 5 | 5 (1/1/3-gap) |

Binary hedef yalnız CONTINUATION vs REVERSAL. Path label T0-sonrası veri kullandı (izinli); feature'lar **kesinlikle** T0-sonrası veri kullanmadı.

---

## 3–4. Frozen feature set + leakage audit

12 model feature'ı (hepsi T-penceresi..T0, `latest_source ≤ T0`):

| feature | SQL kaynak | pencere | offset | leakage | JUNE miss% | JULY miss% |
|---|---|---|---|---|---|---|
| absorption | mark_prices | T−5m..T0 | ≤0 | PASS | 0% | 0% |
| accel | liquidations | T−10m..T0 | ≤0 | PASS | 0% | 0% |
| sell_persistence | agg_trades | T−15m..T0 | ≤0 | PASS | 0% | 0% |
| bid_refill | book_ticker | T−5m..T0 | ≤0 | PASS | 0% | 0% |
| spread_z | book_ticker | T−30m..T0 (past baseline) | ≤0 | PASS | 28% | 0% |
| imbalance | liquidations | T−60m..T0 | ≤0 | PASS | 0% | 0% |
| vel_ratio_1_5 | liquidations | T−5m..T0 | ≤0 | PASS | 0% | 0% |
| rv_30m | mark_prices | T−30m..T0 | ≤0 | PASS | 0% | 0% |
| dist_from_low_bps | mark_prices | T−60m..T0 | ≤0 | PASS | 0% | 0% |
| eth_btc_rel_15m | mark_prices(ETH,BTC) | T−15m..T0 | ≤0 | PASS | 0% | 0% |
| btc_15m | mark_prices(BTC) | T−15m..T0 | ≤0 | PASS | 0% | 0% |
| sell_flow_share | agg_trades | T−15m..T0 | ≤0 | PASS | 0% | 0% |

**`latest_source > T0` olan feature: YOK — 0 BLOCKED.** Duplicate-event leakage canonical event_id (67 chain-merge) ile önlendi. Frozen listenin kalan aileleri (ask refill, refill asymmetry, quote update intensity, jerk, new-low frequency) hesaplandı ama modele alınmadı veya book kapsam sınırı nedeniyle kısmen `NOT_COMPUTED` — model yalnız yukarıdaki 12'yi kullandı.

---

## 5–7. Modeller + June leave-one-day-out (thr=0.70 primary)

M0=always-LONG, M1=no-trade, M2=single-feature veto, M3=frozen linear score, **M4=L2 logistic (primary)**, M5=depth-2 tree (diagnostic). Scaler/model yalnız train fold'unda fit.

**M4 June leave-one-day-out (4 fold):**

| metrik | değer |
|---|---|
| confusion | TP=2, TN=11, FP=5, FN=13 |
| continuation_recall (primary sınıflandırma) | **0.69** |
| reversal_precision | 0.29 |
| balanced_accuracy | **0.41** (< 0.5) |
| **MCC** | **−0.21** (rastgeleden kötü) |
| Brier | 0.358 |
| coverage (trade rate) | 7/31 |
| **always-LONG net** | **+685 bps** |
| **filtered net** | **−454 bps** |
| **improvement (primary ekonomik)** | **−1139 bps** |

June'da filtre değer **yok ediyor**: kazanan reversalları veto edip kaybedenleri işliyor (MCC negatif). continuation_recall 0.69 tek başına yanıltıcı — model çoğunu veto ediyor ama yanlış olanları.

---

## 8. July external holdout (June-frozen, apply-once, thr=0.70)

| event (UTC) | p_rev | pred | gerçek | LONG6h | aksiyon |
|---|---|---|---|---|---|
| 07-04 23:14 | 0.75 | REV | CONT | −60 | **LONG** (yanlış) |
| 07-11 22:20 | 0.39 | CONT | CONT | −42 | VETO ✓ |
| 07-12 22:00 | 0.67 | CONT | CONT | −151 | VETO ✓ |
| **07-15 23:42** | 0.37 | CONT | **REV** | **+42** | **VETO (kaçan kazanç)** |
| 07-16 23:03 | 0.30 | CONT | CONT | −124 | VETO ✓ |
| 07-16 23:20 | 0.49 | CONT | CONT | −80 | VETO ✓ |

Model, işlediği **tek** event'te (07-04) −60 bps kaybetti ve holdout'taki **tek REVERSAL'ı (07-15, +42) veto etti**. 07-13 üçlüsü DATA_QUALITY_VETO (gap) — model başarısı sayılmadı.

---

## 9. P0/P1/P2 (July, 8 non-gap event)

| politika | n | net (bps) | median |
|---|---|---|---|
| **P0 NO-TRADE** | — | **0** | 0 |
| P1 ALWAYS-LONG | 8 | −375 | −51 |
| P2 FILTERED-LONG | 1 | **−60** | −60 |

improvement P2−P1 = +315 bps, ama **P2 (−60) < P0 (0)**. Filtre always-LONG'u yeniyor **ama NO-TRADE'i yenmiyor** — sadece işlem sayısını sıfıra yaklaştırarak zararı azaltıyor, pozitif edge üretmiyor. MODEL_VETO=9, LONG=1.

→ **`FILTER_NO_BETTER_THAN_NO_TRADE`**

---

## 10. Feature stability June↔July

| feature | Jun dir | Jul dir | verdict |
|---|---|---|---|
| absorption | + | − | **SIGN_FLIP** |
| accel | − | + | **SIGN_FLIP** |
| spread_z | + | − | **SIGN_FLIP** |
| rv_30m | + | − | **SIGN_FLIP** |
| eth_btc_rel_15m | − | + | **SIGN_FLIP** |
| sell_persistence | − | − | SAME |
| bid_refill | − | − | SAME |
| imbalance | + | + | SAME |
| vel_ratio_1_5 | − | − | SAME |
| btc_15m | − | − | SAME |
| sell_flow_share | − | − | SAME |
| dist_from_low_bps | + | ? | INSUFFICIENT |

**5 SIGN_FLIP / 6 SAME / 1 yetersiz.** Kritik: M3 frozen skorunun çekirdek üç feature'ı (**absorption, accel, spread_z**) hepsi SIGN_FLIP — a priori tanımlanan `z(absorption) − z(accel) − ... − z(spread_z)` skoru OOS geçersiz.

---

## 11. Negatif kontrol (day-level label permütasyonu)

| ölçüm | değer |
|---|---|
| observed June LOOCV improvement | −1139 bps |
| **day-permutation p (≥ observed)** | **0.7616** (10.000 perm) |

Gözlenen improvement, rastgele-etiketlenmiş modellerin %76'sından **kötü**. Filtre, gerçek sinyalden değil, gürültüden ayırt edilemez (aslında ondan kötü).

---

## 12. Missingness / data-gap ayrımı

July: MODEL_VETO=9, DATA_QUALITY_VETO=0 (görünür — 07-13 üçlüsü binary target'a girmedi çünkü gap→UNCERTAIN_GAP). 07-13 gap event'leri model başarısı olarak sayılmadı. spread_z June'da %28 eksik (past-baseline snapshot yetersizliği) — o event'ler NaN-guard ile model matrisinden düştü, imputasyon yapılmadı.

---

## 13. Ekonomik muhasebe — shadow vs live stop

LONG 6h: fee 5bps, spread ölçüldü (~0.076bps), funding gerçek signed. Boyut-bağımlı market impact **NOT_MODELED**. İki ayna ayrı hesaplandı:
- **shadow-faithful** (no-stop): yukarıdaki tüm rakamlar
- **live-faithful** (300bps stop): July'da hiçbir LONG6h event −300 MAE'yi aşmadı (en derin −200.9) → live=shadow bu örneklemde. June'da da 6h penceresinde stop nadir. İki ayna bu holdout'ta **aynı** sonucu verdi (karıştırılmadı, ayrı doğrulandı).

---

## 15. HÜKÜMLER

### Birincil: `MICROSTRUCTURE_FEATURES_NOT_OOS_STABLE`

Çekirdek feature'ların yön ilişkisi kohortlar arası çöküyor (5/12 SIGN_FLIP, M3'ün üç çekirdeği dahil); June-fit model July'da tek reversalı veto edip kaybedeni işliyor; day-permutation p=0.76. Feature'lar bir sonraki rejimde işaret koruyamıyor.

Bu hükümle birlikte **`INSUFFICIENT_INDEPENDENT_EVENTS` de geçerli**: June eff_n=4 gün, July'da 1 reversal — hiçbir sınıflandırıcı bu tabanla OOS doğrulanamaz. İkisi birlikte kesin sonuç veriyor.

### Alt hükümler

| alan | hüküm |
|---|---|
| June internal validation | **FAILS** — MCC −0.21, balanced_acc 0.41, improvement −1139 bps, perm p=0.76 |
| July external holdout | **FAILS** — tek reversal veto edildi; işlenen tek event kaybetti; P2−P1 yalnız no-trade'e yaklaşmadan |
| economic value | **`FILTER_NO_BETTER_THAN_NO_TRADE`** — P2(−60) < P0(0) |
| feature stability | **`SIGN_FLIP` dominant** — 5/12, çekirdek M3 feature'ları dahil |
| shadow/live consistency | **CONSISTENT** bu örneklemde (300bps stop hiç tetiklenmedi); contract-divergence riski hâlâ açık (gelecekte −300 aşımında) |
| data quality | 07-13 gap DATA_QUALITY_VETO olarak izole edildi, model başarısı sayılmadı; spread_z %28 June-missing |

**Ana soruya cevap: HAYIR.** T0-öncesi mikroyapı, continuation_down event'lerini OOS-güvenilir biçimde veto edemiyor. Filtre ne pozitif edge yaratıyor ne de NO-TRADE'i yeniyor; feature'lar rejim-stabil değil; örneklem (4 gün / 1 reversal) taban-yetersiz.

Yeni eşik optimize edilmedi, hiçbir feature deploy edilmedi. Önceki `BOTH_FAIL` (DIRECT SHORT) + LONG 6h alpha −41 + mekanizma-karışımı bulgularıyla tutarlı: HOUR17 tek bir mikroyapı-koşullu kuralla kurtarılamıyor.

---

## Kaynaklar

| ne | nerede |
|---|---|
| June event kaynağı | `reports/research/s34/S34_SELL_LIQ_REVERSAL_LONG_2026-06-07_15.json` (salt-okunur) |
| July event kaynağı | `reports/shadow/s34_state_machine_shadow.jsonl` |
| Fiyat/likidite | `data/microstructure.db` (`mode=ro`): mark_prices, liquidations, agg_trades, book_ticker |
| İlişkili audit | `S34_HOUR17_DIRECT_SHORT_AUDIT_2026-07-17.md` (BOTH_FAIL), `.sql` |
| Sonuç tabloları | `S34_HOUR17_MICROSTRUCTURE_STATE_FILTER_2026-07-17.sql` |
