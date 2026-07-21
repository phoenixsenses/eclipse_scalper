# DR-0003 — Faz 6A Latent State Discovery Kararları (2026-07-02)

**Durum:** KABUL — operatör Faz 6A paketi. Sonuç: dürüst REJECTED/NO_STABLE_STATE.

## Kararlar
1. **Veri temsili:** 5dk çözünürlüklü sürekli grid (23,635 örnek, 9 backward-looking feature),
   `latent_dataset.npz` + meta (şema/feature versiyonu, missingness maskı, session, dq).
   Event-pencereleri yerine grid: transition dinamiği için doğal, outcome-leakage riski düşük.
2. **Outcome ayrımı MİMARİ:** dataset dosyasında outcome kolonu YOK (`assert_no_outcome` +
   identity-leakage guard'ı); değerlendirme freeze SONRASI mark index'ten ayrı katmanda.
3. **Model sırası uygulandı:** CUSUM changepoint (betimleyici) → seeded k-means (birincil,
   temporal smoothing) → saf-numpy diagonal-Gaussian HMM (çapraz doğrulama). Deep model
   KULLANILMADI (gerek kalmadı — basit yöntemler yeterince stabil sonuç verdi).
4. **k-seçim kuralı frozen:** k∈[2..6]; min occupancy ≥5% + seed-ARI ≥0.6 sağlayanlar
   içinden en yüksek seed-ARI. k=5,6 occupancy'den elendi; k=4 seçildi (ARI 0.851).
5. **Missing-data politikası:** exploration>%30 VEYA herhangi bir erada >%90 eksik feature
   model DIŞI (`era_missing_drop`) — missingness'in sahte state olmasını engeller.
   (basis_spot bu yüzden dataset'e hiç alınmadı; 9 feature'ın hepsi kaldı.)
6. **Normalizasyon:** Standardizer yalnız exploration'da fit (fit_range kayıtlı, test ediliyor).
7. **İsimlendirme nötr:** LS-001..LS-004; mekanizma adı için ayrı deney şartı korundu.

## Sonuç yorumu (kayıt için)
Exploration'da k=4 state seed-ARI 0.851 / perturbasyon-ARI 0.991 ile ÇOK stabil; ama
untouched validasyonda occupancy oranları frozen bandı [0.3,3.0] aştı (LS-003 selloff-state
0.14× — validasyon dönemi rally; LS-004 stres-state 4.99×). trans_corr 0.690 geçti.
Bu model instabilitesi değil **rejim kayması** göstergesi — yine de frozen kriter gereği
REJECTED; failure archive kaydı retry koşuluyla yazıldı (daha uzun veri / rejim-koşullu
yeniden test, YENİ prereg ile). Kriterleri sonuca göre gevşetmedik (m14/m15 mutation'larının
koruduğu davranış).

## Reddedilenler
- hmmlearn/sklearn bağımlılığı (saf numpy yeterli, determinizm tam kontrol)
- Outcome'la yarı-süpervize kümeleme (leakage yasağı)
- Kriter gevşetip "ACCEPT" üretmek (bilimsel dürüstlük)
