# AMI Research Protocols (bağlayıcı)

Kod karşılığı: `ami/research/registry.py` — protokol ihlali `ConstitutionViolation` fırlatır.

## 1. Deney yaşam döngüsü
Observation → Question → Hypotheses (PRIMARY + NULL + ALTERNATIVE/CONFOUNDER/DATA_QUALITY/EXECUTION/REGIME)
→ **ExperimentSpec.freeze()** → register → çalıştır → EvidenceBundle(attach) → Knowledge/Failure.

Freeze şunlar dolu olmadan İMKÂNSIZ: population, target, decision_criteria,
falsification_rule, untouched_data. Kayıttan sonra spec değişirse hash uyuşmaz → kanıt eklenemez.

## 2. Zorunlu deney standardı (Appendix D — S34 pratiğiyle hizalı)
- Kronolojik 70/30 TRAIN/TEST; eşikler yalnız TRAIN'de.
- MC permütasyon (≥500) + gerektiğinde max-stat / family düzeltmesi.
- No-overlap (tek-slot) EV her tradeable iddiada zorunlu.
- FEE ≥ 5bps net; execution model spec'te yazılı.
- Negative control şart (ör. skorun ters dilimi).
- Sonuç ne olursa olsun kayıt: destek → Knowledge, red → Failure Archive.
- Post-hoc ilginç bulgu = YENİ hipotez; aynı deneye eklenemez.

## 3. Kanıt bağımsızlığı (§6)
Aynı dataset/feature-ailesi/hedefin eşik varyantları TEK evidence family'dir.
`KnowledgeObject.evidence_families` bunu taşır; N tane benzer backtest replikasyon SAYILMAZ.

## 4. Promotion başvurusu (governor'a)
Başvuru = kod (`governor.promote`); anlatı kabul edilmez. Gate tablosu için
`AMI_KNOWLEDGE_SCHEMA.md`. Forward kanıt yalnızca FROZEN adaylarda birikir.

## 5. Mezarlık disiplini
Reddedilen fikir `store.archive_failure(...)` ile tip + retry-condition alır.
Yeni soru açmadan önce `store.is_known_failure(...)` kontrolü zorunlu
(tekrar-test yasağı: buy-side fade, reversal, cross-asset transfer, gentleness,
micro-timing, tight stops, partial exits, limit-entry — bkz. arşiv).

## 6. Kaynak bütçesi
Marketplace (60/25/15 exploitation/exploration/curiosity) sıralamayı verir;
tek Python prosesi kuralı ve RAM sınırı (max 2 pytest dosyası) geçerli kalır.
