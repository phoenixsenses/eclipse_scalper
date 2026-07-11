# S34 Oturum Sonuç Raporu — 2026-07-03

> Dört preregistered paket tamamlandı; hepsi dürüst null/reject. Live/shadow/route/`.env`
> SIFIR değişiklik. Detaylar: SYSTEM_STATE §37-39, DR-0005..0007, PATCH-0005..0007.

## 1. AMI Faz 6A-R2 — Risk/Applicability (E-RISKAPP-6AR2-001) → FALSIFIES / INSUFFICIENT_SAMPLE
- "regime+latent risk azaltır" iddiası frekans-normalize kontrollerde desteklenmedi;
  aday overlay rejim kayması altında SEÇİM üretemiyor (fold 1-3: n=1/2/1).
- 6A-R'nin N=14 mdd −416 hücresi per-fold dürüst refit ile YENİDEN ÜRETİLEMEDİ.
- Drift alarmı satüre (13/13 UNUSABLE, fp 0.69) → applicability-degenerate.
- Mutation 13/13. Rapor: `AMI_PHASE6AR2_RISK.md`.

## 2. C-BUY-FADE Yapısal (E-BUYFADE-STRUCT-001) → FALSIFIES (+ silence-info istisnası)
- 391 event: route ALL varyantı tarihsel NEGATİF (−9.5/−1.1/−10.7 bps) — shadow N=26 +2.8
  genellemedi. Dashboard mc_p=0.010 hardcoded (100-perm tabanı).
- SILENCE tek gerçek bilgi: +20/+30/+20 üç split; matched-control diff +54bps p<5e-5
  (proxy değil) — ama T+30m-bilinir → giriş alfası DEĞİL. KO: K-BUYFADE-SILENCE-INFO-001.
- Timing/genesis/management/horizon NON; tüm ufuklar negatif; reclaim %62-93.
- Mutation 24/24 (re-entry ile ortak dosya). Rapor: `BUYFADE_STRUCTURAL.md`.

## 3. 8A Re-Entry (E-BUYFADE-REENTRY-001) → FALSIFIES (H-RE-NULL doğrulandı)
- S→S: tüm cooldown'larda incremental negatif; random-timing geçilemedi → CHURN.
- S→L val n=2, L kolları eligible=0 → INSUFFICIENT.
- Tek ilginç alt-sinyal: BAD_TIMING-stop sonrası re-entry (train +19bps n=16) — yeni
  prereg adayı. Rapor: `BUYFADE_REENTRY.md`.

## 4. Silence-Conditional Exit Timing (E-BUYFADE-SILEXIT-001) → REJECTED[econ] + T45_ROBUST
- Kazanç T0→T30'da (~10:1); T+30 medyan unrealized +22bps; T+45 çıkışı sağlam.
- En iyi aday bd_first_buy50 8/9 kriter geçti, val econ +1.37<3bps → red (gevşetilmedi);
  üstelik noisy'de de çalışıyor → silence'a özgü değil.
- Senaryo B (T+30 entry) −16/−13 → silence giriş sinyali değil (3. bağımsız teyit).
- Survivor-audit'li (pre-T30 SL'ler evrende). Mutation 16/16. Rapor: `BUYFADE_SILENCE_EXIT.md`.

## Yapısal dersler (bugün eklenenler)
1. Shadow-N küçükken route istatistiği genellemez — promotion öncesi tarihsel replay şart.
2. "Cascade-sonrası sessizlik" ailesi BİLGİ taşır ama giriş anında bilinemez — kullanım
   alanı geç-aşama yönetim/risk; o katmanda da T+45 zaten optimal bölgede.
3. Frozen econ/percentile eşikleri görevini yaptı: val-parlak delay_600s (untouched −12.3)
   ve 8/9-geçen bd-exit dürüstçe reddedildi.
4. Kontrol setleri mekanizma etiketini test eder: bd-exit noisy'de de çalışıyor —
  "silence-koşullu" etiketi kontrolsüz geçseydi yanlış bilgi üretilirdi.

## Operatör kararı bekleyenler
- BUY_FADE shadow route'u (ALL tarihsel negatif): gözlemde kalsın mı / evrilsin mi?
- Shadow'a observation-only `bd_first_buy50` çıkış gözlemcisi eklensin mi?
- Yeni prereg adayları: BAD_TIMING-re-entry · 4h-DOWN+silence continuation (H2) ·
  OI-genesis (veri birikince) · adaptif-referans drift monitörü.

## Test durumu
AMI 79 + BUYFADE 24 + SILEXIT 16 = **119 test, hepsi yeşil.** Forward pipeline: E-HOUR17 +
E-CONVCOMP binding VALID, n=0/20 (iddia yok).
