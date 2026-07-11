# FAILURE_ARCHIVE (index)

**Canonical kaynak:** `data/ami/knowledge.sqlite` → `failure_archive` tablosu (22 kayıt, 2026-07-06) — SİLİNMEZ. Bu dosya insan-okur indekstir; tam liste + retry koşulları DB'de ve DR/rapor dosyalarında.

## Kapalı aileler (yeniden test YASAK — yalnız kayıtlı retry-condition + YENİ prereg ile)
buy-side fade (ALL route tarihsel negatif) · reversal harvest · cross-asset transfer · gentleness · micro-timing · tight stop · partial exit · limit-entry · MFE50 ayırıcı (NO_EDGE) · latent 6A (NO_STABLE_STATE) · 6A-R2 risk overlay (INSUFFICIENT_SAMPLE) · buyfade timing/genesis/management (non-incremental) · S→S re-entry (churn) · silence-exit bd_first_buy50 (REJECTED econ) · silence T0-giriş (lookahead) · delayed-entry silence · failed_cascade_SHORT · early-exit (edge'i öldürüyor)

## Ek kayıtlar (DB id ile)
- id=22 (2026-07-06): pre-cascade dip-recovery sayısı (liq öncesi 2-3 düşüş+çıkış) — NO_EDGE; TRAIN-seçili config TEST perm_p=0.203 (core gate 0.411), grid işaretleri komşu configlerde kararsız → gürültü. Retry: ≥6 ay ek veri + tek önceden-sabitlenmiş config prereg. Rapor: `reports/research/s34/S34_PRE_CASCADE_DIP_RECOVERY.md`

## Kayıt kuralı
Yeni failed/falsified/rejected sonuç → DB'ye + bu indekse tek satır + retry-condition zorunlu. Eski kayıt silmek/sulandırmak yasak (master protokol §12).
