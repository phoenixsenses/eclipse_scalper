# AMI Roadmap (whitepaper §90-99 hizalı)

| Faz | Kapsam | Durum (2026-07-02) |
|---|---|---|
| 0 | Anayasa, şemalar, lineage, failure archive, decision records | ✅ TAMAM (`ami/constitution`, `ami/enums`, `ami/knowledge`, docs/ami) |
| 1 | State foundation: unified StateObject, 1m..1W, data-health propagation | ✅ TAMAM (`ami/states/engine.py`; kural-bazlı taksonomi v0) |
| 2 | Structure engine: faz etiketleri, transition matrix, dual-direction, TF conflict | ✅ v0 (`ami/states/structure.py` + `StateBundle.conflict_report`; swing graph kısmi) |
| 3 | Trade lifecycle: post-entry snapshot, MFE classifier, observer çerçevesi | ✅ v0 (`ami/lifecycle/engine.py`; gerçek shadow ledger 120 trade replay; MFE+50 dataseti) |
| 4 | Research OS: soru/hipotez/prereg/kanıt/marketplace | ✅ TAMAM (`ami/research/*`; freeze mimari olarak zorunlu) |
| 5 | Epistemic Governor MVP: izinler, promotion/demotion, breaker, audit | ✅ TAMAM (`ami/governance/governor.py`; kalibrasyon motoru hariç) |
| 6 | ML + latent states: clustering/HMM, kalibrasyon, benzerlik | ⬜ BAŞLANMADI |
| 7 | World Model + Digital Twin: senaryolar, forecast, sayaç kalibrasyonu | ⬜ BAŞLANMADI (transition matrix ilk yapı taşı) |
| 8 | Autonomous Scientist: anomali→soru, agent reliability, otomatik öneri | ⬜ BAŞLANMADI (backlog + marketplace hazır altyapı) |
| 9 | Cross-market genelleme | ⬜ BAŞLANMADI |

## Sonraki somut adımlar (öncelik)
1. ~~Forward kanıt akışı~~ ✅ 2026-07-02: `ami/research/forward_pipeline.py` —
   E-HOUR17-FWD-001 + E-CONVCOMP-FWD-001 frozen binding'lerle canlı; her oturumda/cron'da
   `python -m ami.run_forward_pipeline` idempotent koşulur. (E-MECHCOMP bağlanmadı:
   shadow mech_score loglamıyor — loglama eklenince bağlanır.)
2. ~~Adversarial doğrulama~~ ✅ 2026-07-02: 20/20 mutation (`AMI_MUTATION_REPORT.md`).
3. ~~İlk preregistered alpha deneyi~~ ✅ 2026-07-02: E-MFE50-001 → **FALSIFIES**
   (tek-anlık feature'lar giveback'i ayıramıyor; retry: state-transition dizileri, YENİ prereg ile).
4. **Kalibrasyon motoru (Faz 6 ön):** DecisionTrace olasılıkları vs gerçekleşen 6h yön — Brier.
5. **S34_ALL.db → Knowledge Object toplu migrasyonu** (trust≥3, evidence-family gruplu).
6. **Shadow runner'a mech_score loglaması** (E-MECHCOMP bağlanabilsin — restart gerektirir).
7. **MFE50 v2:** lifecycle state-TRANSITION dizileriyle yeni prereg (failure archive retry koşulu).
8. State engine'i dashboard'a bağla (read-only, opsiyonel).
9. Faz 6 latent-state keşfi: mechanism_store clustering (FAZ KAPISI GEÇİLDİ — artık açılabilir).

## Faz 6A-R2 sonrası durum (2026-07-03)
- Risk/applicability doğrulaması: **FALSIFIES / INSUFFICIENT_SAMPLE**
  (`AMI_PHASE6AR2_RISK.md`, DR-0005). "regime+latent risk azaltır" iddiası
  frekans-normalize kontrollerde desteklenmedi; aday overlay validation erasında
  seçim üretemiyor (applicability çöküşü). Faz 6 satırı: latent hattı
  **market-description olarak sınırlı**; risk-overlay retry'ı forward shadow ≥6 ay
  birikince YENİ prereg ile.
- Yeni yeniden kullanılabilir altyapı: matched-count blocked bootstrap + random-veto +
  regime-only kontrol çerçevesi ve 13 yapısal guard (`ami/latent/risk_applicability.py`).
  BÜTÜN gelecek risk/applicability deneyleri bu guard'lardan geçmek zorunda.
- Drift monitor iyileştirme ihtiyacı (yeni madde): sabit-referans alarm satüre
  (13/13 UNUSABLE) — adaptif/rolling referans penceresi tasarımı, YENİ prereg ile.
- Faz 6B / World Model / Digital Twin: **KAPALI** (operatör kararı bekler).

## C-BUY-FADE paketi sonrası (2026-07-03)
- Yapısal + re-entry paketi kapandı (DR-0006): route ALL tarihsel NEGATİF; tek bilgi
  silence (T+30m) — giriş alfası değil. Operatör kararı bekleyen soru: BUY_FADE shadow
  route'u bu haliyle gözlemde kalmalı mı, silence-koşullu geç-aşama yönetime mi
  evrilmeli? (Otomatik değişiklik YAPILMADI.)
- Yeni prereg adayları: (1) BAD_TIMING-stop-sonrası S->S re-entry (train +19bps n=16);
  (2) 4h-DOWN+silence multi-hour continuation (H2; val h240 +48 ama n=8);
  (3) OI verisi birikince leverage-genesis boyutu.

## Silence-exit paketi sonrası (2026-07-03, DR-0007)
- T+45 çıkışı SAĞLAM; kazanç T0→T30'da. Silence = yönetim bilgisi (3. teyit: T+30 girişi
  −13/−16). bd_first_buy50 çıkışı 8/9 kontrol geçti, econ'da düştü VE noisy'de de çalışıyor
  (silence'a özgü değil) → forward-watch: shadow'a observation-only çıkış gözlemcisi
  (operatör onayı bekliyor; otomatik eklenmedi).
