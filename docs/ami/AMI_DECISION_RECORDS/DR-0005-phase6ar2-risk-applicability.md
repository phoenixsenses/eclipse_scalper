# DR-0005 — Faz 6A-R2 Risk and Applicability Validation Kararları (2026-07-03)

**Durum:** KABUL — sonuç **FALSIFIES / INSUFFICIENT_SAMPLE** (dürüst null; frozen kural gereği).

## Kararlar
1. **Yeni giriş alpha'sı aranmadı:** deney sınıfı yalnız risk/applicability
   (applicability restriction, shadow suspension, risk warning, research prioritization).
2. **Veto yorumu:** filtreler AYNI trade popülasyonunun (328 no-overlap 6h LONG grid trade)
   altkümesini seçer; yeniden simülasyon yok — matched-count/exposure karşılaştırmaları
   bu yüzden temiz.
3. **Frequency-normalization yapısal:** ham MDD/cum karşılaştırması yalnız eşit-N
   dağılımları içinde (matched-count moving-block bootstrap N=2000 blok=5 + random-veto
   N=2000); farklı-N ham karşılaştırma `guard_frequency_normalized` ile bloklanır (m01).
4. **Untouched veri YOK ve prereg'de beyan edildi:** 85-100% penceresi 6A-R alpha_eval'in
   hipotez kaynağı → fold-4 CONTAMINATED etiketiyle raporlanır, kanıt sayılmaz. Bu deney
   yapısal olarak tam PASS üretemez; tavan CHRONO_SUPPORTED_PENDING_FORWARD idi.
5. **Per-fold dürüst refit:** standardizer + rejim eşikleri + k-means merkezleri her fold'da
   yalnız [0, val_lo) üzerinde; label'lama prefix verisiyle (impute lookahead yok);
   k=4/seed=11 fold'da yeniden SEÇİLMEZ (araştırmacı serbestlik derecesi kapalı).
6. **Alarm forward-only:** pencere yalnız kendi sonuna kadarki veriyle değerlendirilir
   (`guard_no_retroactive_alarm`, m06); deterioration eşiği train-era'dan frozen.
7. **Sınıflandırma önceliği frozen:** INSUFFICIENT_SAMPLE > FREQUENCY_ARTIFACT >
   RISK_NON_INCREMENTAL > LATE_DRIFT_DETECTION > REJECTED; kriter gevşetme yasak.
8. **İzin tavanı:** RESEARCH_ONLY / BACKTEST_ALLOWED / SHADOW_ALLOWED +
   SHADOW_SUSPEND_SUGGESTION; LIVE/SIZING/PORTFOLIO `guard_permissions` ile yapısal yasak (m11).

## Sonuç yorumu
- **Aday overlay validation erasında devreye giremiyor:** fold 1-3'te n_cand=1/2/1
  (calm-state kimliği refit'te değişiyor — 6A rejim kaymasının doğrudan sonucu).
  Applicability overlay'i seçim üretemiyorsa risk avantajı iddiası test edilemez.
- **Fold0 (tek değerlendirilebilir):** aday = regime-only birebir aynı set → latent
  incremental katkı SIFIR; cvar5 matched-pct 0.644 / random-veto-pct 0.650 (<0.75);
  retention_ratio 0.47 (<0.90) — kazanan feda oranı yüksek. Random veto'dan
  ayırt edilemez.
- **6A-R N=14 (mdd −416) YENİDEN ÜRETİLEMEDİ** per-fold dürüst artifact'larla —
  o sonuç ALL-era-fit + hipotez-penceresi artefaktıydı. Hipotez kaynağı olarak kalır.
- **Drift alarmı SATÜRE:** 13/13 pencere UNUSABLE, fp-suspension 0.69. "Leading" frozen
  kriteri teknik sağlandı ama sürekli-açık alarm ayırt edici değildir →
  applicability-**degenerate/saturated** (koruyucu applicability alpha İDDİA EDİLEMEZ).
- Failure archive kaydı: `Faz6A-R2 regime+latent risk/applicability overlay` /
  INSUFFICIENT_SAMPLE; retry = forward shadow ≥6 ay birikince YENİ prereg.

## Reddedilenler
- Kriter gevşetme (PCT_BEAT, RETENTION_MIN, MIN_TOTAL_CAND sonuca göre değiştirilmedi)
- N=14 vs N=50 ham MDD karşılaştırmasını "başarı" sayma (m01 guard'ı)
- Fold cherry-picking (m08: verdict tüm fold'lardan yeniden hesaplanır)
- Satüre alarmı "leading applicability alpha" olarak pazarlama

## Mutation suite
`tests/test_ami_risk_mutations.py` — **13/13**: m01 ham-MDD-cross-N, m02 random-veto
atlama, m03 exposure-norm eksik, m04 top-winner gizleme, m05 post-hoc kriter (freeze),
m06 retroaktif alarm, m07 winner-sacrifice raporlamama, m08 fold cherry-pick,
m09 regime-only atlama, m10 UNUSABLE-drift'te selection + stale artifact,
m11 LIVE/SIZING izin isteme, m12 small-N bootstrap abartısı, m13 pozitif kontrol.
Toplam AMI testi: 66 + 13 = **79**.

## Rollback
`ami/latent/risk_applicability.py` + `tests/test_ami_risk_mutations.py` +
`reports/research/s34/AMI_PHASE6AR2_RISK.*` sil; research.sqlite'ta E-RISKAPP-6AR2-001
kaydı ve knowledge.sqlite failure-archive satırı kalır (tarihsel kayıt — silinmez).

## Dokunulmayan canlı bileşenler
`tools/s34_state_machine_live_executor.py`, `.env`, `execution/`, `risk/`, `brain/`,
shadow runner, dashboard, oi_spot_poller — SIFIR değişiklik. Deney research-only,
DB salt-okunur; hiçbir operasyonel izin verilmedi/istenmedi.
