# OD-011 — Cycle-Adjusted Recompute: Frozen Population Manifest

**v1** 2026-07-17 (Opus 4.8) → **bağımsız review** (`..._CORRECTIVE_REQUIRED`, F-1..F-4)
→ **v2 corrective** 2026-07-17 (Opus 4.8)

**Statü:** `OD011_FROZEN_POPULATION_MANIFEST_V2_CORRECTED_AWAITING_REREVIEW`
**Tip:** salt-okunur manifest. Hiçbir recompute YÜRÜTÜLMEDİ, hiçbir KO değiştirilmedi,
`knowledge.sqlite` yazım için AÇILMADI.

> **Amaç.** OD-011'in talep ettiği "frozen popülasyon manifesti": her KO'nun **hangi
> evrende** iddia ettiğini ve yeniden-kurulabilirliğini kayda geçirmek.
>
> **Neden zorunlu.** 2026-07-10'daki ilk geçiş `INVALID_WRONG_POPULATION` ile çöpe
> gitti — HOUR17 rakamı HOUR17-**etiketsiz** `ami_events` popülasyonundan üretilmişti.
> Doğru rakam (127→93, deflasyon 0.732) ancak gerçek `LONG_HOUR17_HOLD6H` evreni
> eşleştirilince çıktı.

---

## 0. v1 → v2 DÜZELTME KAYDI (bağımsız review bulguları)

v1'in merkezi tespiti **YANLIŞTI** ve bu bölüm onu açıkça geri alır.

| # | Sev | v1'in iddiası | Gerçek |
|---|---|---|---|
| **F-1** | HIGH | *"MGMT-6H'nin sayıları hiç kaydedilmemiş → KO denetlenemez/çürütülemez"* | **YANLIŞ.** Sayılar `reports/research/s34/S34_TRADE_MGMT.md`+`.json`'da duruyor. Kusur "kanıt yok" değil, **payload'a taşınmamış** — daha hafif ve farklı sınıf. |
| **F-2** | HIGH | *"M-2/M-3'te N YOK → deflasyon uygulanamaz"* | **YANLIŞ.** İkisinin de N'i kurtarılabilir (M-2: n=59; M-3: N=217). §3'ün öncelik gerekçesi bu yanlış öncüle dayanıyordu. |
| **F-3** | MED | §1.2'de OD-011'in muafiyeti **sorgulanmadan** kabul edildi | `K-BUYFADE-SILENCE-INFO-001` açıkça N-bağımlı istatistik taşıyor (p<5e-5, N=26) → muafiyet **geçersiz**, kapsama geri alındı. |
| **F-4** | LOW | — | `K-S34-MONDAY-VETO-001` `DESCRIPTIVE_ONLY_LOW_SAMPLE` + N=15 olmasına rağmen `confidence.statistical="HIGH"` taşımaya devam ediyor. |
| **S-1** | MED | *"tek PAPER_ALLOWED → en yüksek operasyonel bahis"* | **ÖZ-DÜZELTME (review dışı, bu oturumda bulundu).** `PAPER_ALLOWED` **atıl bir etiket**: `knowledge.sqlite` paper/shadow/executor'a **hiç bağlı değil** (tüm repoda `ami/`'ye referans veren 2 dosya var, ikisi de araştırma aracı). Gerekçe düştü. |

**Kök neden (F-1/F-2):** v1, `reports/research/s34/` altına — CLAUDE.md'nin araştırma
raporları için gösterdiği **canonical konum** — bakmadan "sayı yok" ilan etti.

---

## 1. Kapsam

### 1.1 Tamamlanmış (v2 payload, cycle-adjusted) — 5 KO, İŞ YOK ✅ *(review: DOĞRULANDI)*

Her birinin `history[]` alanında 2026-07-11 tarihli gerçek recompute kaydı var.

| KO | Deflasyon | Sınıflandırma | Net etki |
|---|---|---|---|
| `K-S34-BOOK-PULL-001` | 0.598 | `CONFIRMED_DIRECTION_HISTORICAL` | **GÜÇLENDİ** (+70.2 → +80.3) |
| `K-S34-HOUR17-001` | 0.732 | `..._POSITIVE_BUT_FRAGILE` | zayıfladı |
| `K-S34-FUNDING-LEVEL-001` | 0.598 | `CONFIRMED_DIRECTION_MAGNITUDE_REDUCED` | 114.8 → 69.2 |
| `K-S34-MECH-COMPOSITE-001` | 0.598 | `WEAKENED` | 88.0/82.6% → 39.3/68.4% |
| `K-S34-MONDAY-VETO-001` | 0.732 | `DESCRIPTIVE_ONLY_LOW_SAMPLE` | N=15, fiilen düştü |

> **OD-011 kaydı ve oturum-öncesi bilgi "7 KO bekliyor" diyordu — BAYAT.** Gerçekte 8
> RECOMPUTE_REQUIRED'ın 5'i yapılmış.
>
> **Track record:** 5 recompute → 3 zayıflattı/düşürdü, 1 güçlendirdi, 1 nötr.
> Egzersiz sonucu gerçekten değiştiriyor.
>
> **F-4 (taşınan iş):** MONDAY-VETO'nun `status` ve `confidence.statistical="HIGH"`
> alanları recompute'a rağmen **hiç değiştirilmedi** — "fiilen düştü" veri değil,
> yorum. Governance düzeltme adayı.

### 1.2 Muafiyet — YENİDEN TÜRETİLDİ *(v1 devralmıştı; F-3)*

| KO | v1 | v2 |
|---|---|---|
| `K-LATENT-REGIME-001` | muaf | ✅ **muafiyet geçerli** — sayısal/N-taşıyan iddia yok (review doğruladı) |
| `K-S34-REFILL-CTX-001` | muaf | ✅ **muafiyet geçerli** — aynı |
| `K-BUYFADE-SILENCE-INFO-001` | muaf | ❌ **MUAFİYET DÜŞTÜ** → **M-4** |

`K-BUYFADE-SILENCE-INFO-001` claim metni: *"45m fade PnL silence'ta +20/+30/+20 bps …
noisy'de −36/−26/−39; **matched-control diff +54bps p<5e-5** … **shadow N=26** +2.8
genellemiyor."* → p-değeri + N + split-bazlı delta = **açıkça N-bağımlı**.
OD-011'in "N-bağımlı iddia taşımıyor" gerekçesi bu KO için **gerçek dışı**.

### 1.3 Kalan kapsam — **4 KO** (v1'de 3'tü)

`K-S34-MGMT-6H-001`, `K-S34-SCALEIN-100-001`, `K-S34-PRECASCADE-001`,
`K-BUYFADE-SILENCE-INFO-001`.

---

## 2. Manifest

**Ortak provenance (M-1..M-3):** `data_time_range = 2026-02-15..2026-07-02`,
`dataset_hash = s34-2026H1`.
**Kapsam uyarısı (hepsi):** bu aralık 2026-04-27T14:27Z..2026-06-06T17:47Z arasındaki
**~40 günlük tüm-sembol likidasyon boşluğunu** içeriyor (`TRANSPORT_SPECIFIC_OUTAGE`,
kök-neden kapatılmadı) → tarihsel kanıt **süreksiz**.

### M-1 · `K-S34-MGMT-6H-001` — ÖNCELİK 1

| Alan | Değer |
|---|---|
| Statü / versiyon | `HOLDOUT_VALIDATED` / v1 / evidence_level 3 |
| İzin | `RESEARCH_ONLY`, `SHADOW_ALLOWED`, `OBSERVER_ALLOWED`, `PAPER_ALLOWED` **(atıl — S-1)** |
| İddia | *"Uniform 6h hold with no stop and no early exit remains the strongest tested management baseline; sub-1h bar trailing destroys the edge."* |
| Payload `effect_size` | `{}` — **boş (kusur)** |
| **Frozen popülasyon** | ✅ **`reports/research/s34/S34_TRADE_MGMT.md` + `.json`** (mtime Jul 2 18:29, `code_ref` ile aynı gün/saat) |
| `code_ref` | `tools/research_s34_trade_mgmt_gauntlet.py` (400 satır, mevcut) |
| `falsification` | `["a management variant beats baseline on TEST without higher mdd"]` |

**Kurtarılan sayılar (F-1):**
```
baseline      : n=59, wr=69.5, avg=65.1, mdd=-362.5, mc_p=0.0
baseline_TEST : n=18, wr=72.2, avg=58.3, mdd=-165.9, mc_p=0.028
M10.trail_1m  : avg=-1.5,  mc_p=0.556      M10.trail_1h : avg=+61.7, mc_p=0.0
M10.trail_5m  : avg=-0.1,  mc_p=0.476      M10.trail_4h : avg=+65.1  (== baseline)
```
İddianın iki yarısı da rapordan **doğrulanıyor**: baseline +65.1 (mc_p=0.0) ve
sub-1h trailing sıfıra çöküyor (−1.5 / −0.1, mc_p anlamsız).

**Düzeltilmiş tespit (v1'in yerine):** KO **denetlenemez değil**. Kusur, kanıtın
payload'a taşınmamış olması — `effect_size`/`scope`/`assumptions` boş, `history` boş,
buna karşılık `confidence.statistical="HIGH"`. Yani KO **kanıtsız değil, izlenemez**.

**Asıl bulgu — recompute'un gerçek bahsi:** TEST kolu **N=18, mc_p=0.028**. §1.1'deki
gözlenen deflasyon aralığı (0.598–0.732) uygulanırsa TEST ≈ **11–13 bağımsız cycle**
→ aynı batch'te kullanılan `MIN_BUCKET_N=20` eşiğinin **altında**, ve mc_p=0.028'in
anlamlılığını koruması olası değil. TRAIN N=59 ≈ 35–43 cycle.
**Öngörü (test edilecek, iddia değil): MGMT-6H'nin TEST kanıtı `INSUFFICIENT_SAMPLE`'a
düşebilir.** v1'in "denetlenemez" çerçevesi tam da bu ölçülebilir bahsi gizliyordu.

### M-2 · `K-S34-SCALEIN-100-001` — ÖNCELİK 2

| Alan | Değer |
|---|---|
| Statü | `PRELIMINARY` / v1 / **evidence_level 1** |
| İzin | `RESEARCH_ONLY`, `OBSERVER_ALLOWED` · yasak: `PAPER_ALLOWED`, `SIZING_ALLOWED` |
| İddia | *"…improved per-unit expectancy and reduced mdd **in-sample**; requires forward observation before any sizing use."* |
| Payload | `{"per_unit_bps": 86.3, "baseline_bps": 65.1}` |
| **Frozen popülasyon** | ✅ `S34_TRADE_MGMT.json` **§M7**: `scalein_100 = {n:59, per_unit:86.3, w_mdd:-329.3}` — **N=59 kurtarıldı (F-2)** |

M-1 ile **aynı rapor + aynı script** → popülasyon işi paylaşılır. İddia kendini
`in-sample` + `PRELIMINARY` + evid 1 olarak çitliyor ve sizing'i yasaklıyor → düşük bahis.

### M-3 · `K-S34-PRECASCADE-001` — ÖNCELİK 4

| Alan | Değer |
|---|---|
| Statü | `HOLDOUT_VALIDATED` / v1 / evidence_level 3 · izin: **`RESEARCH_ONLY`** |
| İddia | *"~2x lift … but honest all-alert trading is **negative** both directions; early entry is navigation, not a trade rule."* |
| Payload | `{"lift10": 2.1, "long_at_trigger_bps": -30.4}` |
| **Frozen popülasyon** | ✅ `reports/research/s34/S34_PRECASCADE.md`: `K>=4: **217 tetik** P(casc10)=15.2% lift=2.1x | LONG6h avg=-30.4 mc=0.992` — **N=217 kurtarıldı (F-2)** |

İddianın operasyonel içeriği **zaten negatif** (mc=0.992), izin `RESEARCH_ONLY` →
**en düşük bahis**. Mezarlık #1 ile tutarlı.

### M-4 · `K-BUYFADE-SILENCE-INFO-001` — ÖNCELİK 3 *(YENİ, F-3)*

| Alan | Değer |
|---|---|
| Statü | `HOLDOUT_VALIDATED` / v1 · payload `effect_size` `{}` |
| `code_ref` | `tools/research_s34_buyfade_structural.py` |
| İddiadaki istatistik | matched-control diff **+54bps, p<5e-5**; **shadow N=26** (+2.8, genellemiyor) |

OD-011'in muafiyeti **geçersiz**. Not: bu KO **ayrı bir route/popülasyon** (BUY-fade),
M-1..M-3'ün ETH SELL cascade evreninden farklı → kendi popülasyon eşleştirmesini ister;
tam cycle-adjust gerekli olmayabilir, ama "N-bağımlı iddia yok" öncülü düzeltilmeli.

---

## 3. Öncelik — YENİDEN TÜRETİLDİ (F-2 + S-1 sonrası)

v1'in **iki gerekçesi de düştü**: (a) "N yok" yanlıştı; (b) "PAPER_ALLOWED → operasyonel
bahis" yanlıştı (etiket atıl). Yeni gerekçe **yalnız ölçülebilir kırılganlık + statü
ağırlığı**:

| Sıra | KO | Gerekçe |
|---|---|---|
| 1 | `K-S34-MGMT-6H-001` | `HOLDOUT_VALIDATED` + **TEST N=18 → deflasyonda ~11–13 cycle**, eşik altı; statüsünü kaybetme olasılığı en yüksek olan KO |
| 2 | `K-S34-SCALEIN-100-001` | M-1 ile aynı rapor/script (marjinal maliyet ~0); N=59 |
| 3 | `K-BUYFADE-SILENCE-INFO-001` | `HOLDOUT_VALIDATED` + gerçek p-değeri taşıyor; ayrı popülasyon → ayrı iş |
| 4 | `K-S34-PRECASCADE-001` | N=217 sağlam; iddia zaten negatif + RESEARCH_ONLY → karar değişmez |

**İş hacmi düzeltmesi (F-2):** v1 "400 satırlık script'i oku" diyordu — **abartılıydı**.
M-1/M-2/M-3 için gereken, sıfırdan rekonstrüksiyon değil: **rapor lookup + trade→cycle
eşleştirme + cycle-adjust**. Script yalnız trade evreninin tanımı belirsiz kalırsa okunur.

---

## 4. Bu manifestin YAPMADIKLARI

- Hiçbir recompute yürütülmedi; §M-1'deki `INSUFFICIENT_SAMPLE` **bir öngörü**, sonuç değil.
- `knowledge.sqlite` yalnız `mode=ro`; hiçbir KO statüsü/izni değiştirilmedi.
- `code_ref` script'lerinin içeriği okunmadı (raporlar okundu).
- F-4 (MONDAY-VETO `statistical=HIGH`) **düzeltilmedi**, yalnız kaydedildi.
- Guardrail dosyaları (`execution/`, `risk/`, `brain/`, `.env`,
  `s34_state_machine_live_executor.py`) okunmadı/değiştirilmedi.

## 5. Sonraki kapı

**BAĞIMSIZ RE-REVIEW** (v2 corrective'i inceleyecek yeni geçiş), sonra operatör sign-off.
Re-review odağı: (a) F-1..F-4 gerçekten kapandı mı yoksa yeniden çerçevelendi mi;
(b) §M-1'deki "TEST N=18 → 11–13 cycle → eşik altı" aritmetiği ve deflasyon aralığının
bu popülasyona taşınması meşru mu (0.598/0.732 **başka** popülasyonlardan geldi —
M-1'in kendi deflasyonu ölçülmedi); (c) M-4'ün önceliklendirmesi doğru mu;
(d) S-1 öz-düzeltmesi yeterli mi.

**Rollback:** salt-dokümantasyon. Geri alma = bu dosyayı silmek.
