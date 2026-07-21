# DR-0004 — Faz 6A-R Regime-Conditioned + Drift Kararları (2026-07-03)

**Durum:** KABUL — sonuç **PASS** (rejim-koşullu, dar kapsamlı).

## Kararlar
1. **Yeni dış veri YOK:** latent_dataset.npz yeniden kullanıldı (BTC-sync rejim boyutu bu
   yüzden atlandı — dataset'te BTC feature'ı yok; sınırlılık olarak kayıtlı).
2. **Rejim tanımları deterministik + exploration-fit:** trend(±100bps/24h sabit),
   vol/stress/leverage exploration percentile'ları; `RegimeDefiner.fit_range` guard'ı
   validation-leakage'i yapısal engeller (mutation #2).
3. **Küçük rejim = UNKNOWN:** min_regime_sample=1500 (mutation #7).
4. **Transition leakage guard'ı:** fold/rejim sınırını aşan geçişler sayılmaz
   (`transition_matrix_within`, mutation #10).
5. **Walk-forward persistence tanımı:** occ-band [0.3,3.0] ≥3/4 fold VE ardışık-fold
   merkez-cos ≥0.8 ≥2 geçiş. Tek fold'da görünen state persistent DEĞİL.
6. **Drift monitor yetkisi:** yalnız öneri; SHIFTED/UNUSABLE'da öneri listesi boş
   OLAMAZ (alarm-susturma mutation #11); artifact-kullanımı UNUSABLE'da bloklanır (#12).
7. **Alpha eval'de outcome'suz state seçimi:** "calm" state merkez-normu en küçük olan
   (outcome'a bakarak state seçme yasağı korunur).

## Sonuç yorumu
- Drift attribution: 6A kırılması **MARKET_SHIFT** (rv/stress/buyliq/spread/trades PSI 0.16-7.4,
  missingness delta ~0) — DATA_ISSUE elendi; trend dağılımı değişmedi → kayma
  likidasyon/vol YAPISINDA, yön karışımında değil.
- Rejim-koşullu kronolojik stabilite: trend=UP, vol=LOW, vol=NORMAL, stress=NORMAL (70/30);
  **walk-forward'da yalnız trend=UP persistent (3/4)**. K-LATENT-REGIME-001 bu dar kapsamla
  yazıldı (max SHADOW, LIVE/SIZING/PORTFOLIO yasak).
- Alpha: **non-incremental** — regime+latent en iyi hücre (PF 1.41, mdd −416 vs baseline −1363)
  ama top3-removed NEGATİF (−458) → top-winner bağımlı. Değer sınıfı: market-description +
  RİSK (drawdown daralması), alpha değil.
- Canlı drift monitor şu an **UNUSABLE** diyor (güncel dönem referans dağılımdan kopuk) —
  monitor önerileri audit'e yazıldı; nihai karar governor'da (otomatik uygulanmadı).

## Reddedilenler
- Latent/deep regime modeli (önce deterministik — paket şartı)
- Kriter gevşetme, tek-alt-rejim pozitifiyle promotion
