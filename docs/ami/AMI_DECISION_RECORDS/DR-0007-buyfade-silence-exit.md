# DR-0007 — BUY-FADE Silence-Conditional Exit Timing Kararları (2026-07-03)

**Durum:** KABUL — `E-BUYFADE-SILEXIT-001` (hash bd7d1f63…) sonuç: **REJECTED / EXIT_TIMING_NON_INCREMENTAL (fails=[econ])** + **T45_EXIT_ROBUST**. Önceki iki deney
(E-BUYFADE-STRUCT-001, E-BUYFADE-REENTRY-001) DEĞİŞTİRİLMEDİ; verdict'leri açılmadı.

## Kararlar
1. **Bağımsız yeni prereg:** aynı route evreni; T0 entry / SL75 / fee / silence_v1 SABİT.
   Struct raporundaki silence sonuçları yalnız hipotez kaynağı olarak beyan edildi.
2. **Survivor/lookahead kuralları YAPISAL:** T+30 öncesi SL'ler silence evreninde KALIR
   (`g_survivor_universe`); silence bilgisi <T+30m hiçbir kararda kullanılamaz
   (`g_no_pre_t30_silence_use`); breakdown zamanı ileriye taranır (`g_breakdown_causal`);
   kapanmış trade'e yönetim uygulanamaz (`g_no_manage_closed`); unrealized-realized ayrımı
   (`g_realized_only`); fee uzatmada da uygulanır (`g_fee_on_extension`); route dosyaları
   yazılamaz (`g_no_route_mutation`).
3. **Senaryo ayrımı:** A (ANA) = T0 route + T30-survivor yönetimi; B (T+30 observer entry) =
   AYRI kontrol, yeni-entry deneyi olarak etiketli, A ile karıştırılmadı.
4. **Aday sayıları prereg'de sınırlı:** 13 fixed + 3 breakdown tanımı × (3 grace + pconf) +
   4 structural + 6 partial. Untouched'ın baseline için kontamine olduğu (struct deneyi
   gördü) prereg'de beyan edildi → tavan CHRONOLOGICALLY_SUPPORTED_PENDING_FORWARD idi.
5. **Lock/BE semantiği dürüst:** T+30'da floor zaten aşılmışsa ANINDA T+30 fiyatıyla çıkış
   (zarardaki trade'e profit-lock uygulanmış gibi gösterilmez).

## Sonuç yorumu (silence-open evreni: train 99 / val 39 / untouched 22; pre-T30 SL 7/0/1)
- **Kazancın yeri:** T0→T30 taşıyor (+36.7/+31.7/+22.9 brüt) — T30→45 küçük pozitif
  (+0.6/+3.0/+5.4). Frozen kurala göre etiket POST_T30_CONTINUATION (post>0) ama büyüklük
  oranı ~10:1 pre-T30 lehine. **T+30'da medyan unrealized +22bps** — silence doğrulandığında
  hareketin çoğu bitmiş.
- **T+45 çıkışı SAĞLAM:** hiçbir uzun fixed (60m..24h) train+val'de 45m'i tutarlı geçmedi;
  eğri 45-120m platosundan sonra düşüyor (val 180m +7.6, 720m −13.5). Untouched'taki
  720/1440m sıçramaları (+62/+81) val'de karşılıksız → gürültü.
- **En iyi aday `bd_first_buy50_g0m`** (T30 sonrası ilk yeni BUY≥50K'da çık, yoksa 24h cap):
  9 kontrolden 8'i GEÇTİ (train +8.4, val +1.4, untouched +10.1 incremental; random-p95
  üstü; top3 OK; retention 0.74; tail OK) — **yalnız econ başarısız (val +1.37 < 3bps)**
  → frozen kritere göre REJECTED. Kriter gevşetilmedi.
- **Silence'a özgülük ZAYIF:** aynı breakdown-çıkışı noisy-T30-açık kontrolünde de val
  +31.3 verdi → mekanizma "yeni BUY liq gelince çık" genel bir yönetim etkisi olabilir,
  silence-koşullu değil. (Noisy evreninin T30-açık alt kümesi survivorship içerir —
  kontrol tanımı gereği; pre-T30 SL'ler noisy kaybının çoğunu taşıyor.)
- **Senaryo B (T+30 observer entry) NET NEGATİF** (train −15.9, val −13.3) → silence
  doğrulandığında girmek için ÇOK GEÇ; struct deneyinin "delayed-entry yakalayamıyor"
  bulgusunun üçüncü bağımsız teyidi. Silence = yönetim bilgisi, giriş sinyali değil.
- **Silence maturity (T+30'da hesap, sadece post-T30 kullanım):** immediate_noise_then_silent
  sınıfında 240m uzatma çöküyor (val +3 vs f45 +30); early_continuous/late_silence 240m'de
  tutuyor — "daha temiz silence → daha uzun continuation" yönünde ama hücreler 10-18,
  iddia değil.
- **Dar hipotez 4h-DOWN+silence:** val hücresi <12 → **INSUFFICIENT_SAMPLE** (gevşetilmedi).

## Reddedilenler
- econ eşiğini val sonucuna göre 3→1 bps'e çekmek
- bd-exit'i "silence-koşullu edge" diye etiketlemek (noisy kontrolü aksini gösteriyor)
- untouched 720/1440m sıçramalarına dayalı "T45 too early" ilanı (val karşılıksız)
- Senaryo B'yi Senaryo A sonuçlarına karıştırmak

## Mutation suite
`tests/test_buyfade_silexit_mutations.py` — **16/16**: survivor-exclusion, pre-T30-SL silme,
<T30 silence kullanımı, future-breakdown, full-dataset seçim, horizon-overlap, cycle-purge,
unrealized-as-realized, fee-on-extension, structural-timestamp, tiny-4hDOWN, top-winner,
noisy-control zorunlu, closed-trade yönetimi, route-mutation yasağı, post-hoc freeze.
Toplam suite: 103 + 16 = **119**.

## Rollback
`tools/research_s34_buyfade_silence_exit.py` + `tests/test_buyfade_silexit_mutations.py` +
`reports/research/s34/BUYFADE_SILENCE_EXIT.*` sil; registry/failure kayıtları tarihsel kalır.

## Dokunulmayan canlı bileşenler
Live executor, order logic, risk, leverage, sizing, `.env`, shadow runner — SIFIR değişiklik.
Otomatik route değişikliği YOK. Forward öneri (operatör kararı): shadow'a observation-only
`bd_first_buy50` çıkış gözlemcisi eklenmesi (sipariş yok, sadece delta loglama).
