# HISTORICAL_RESEARCH_WAVES

**Tarih:** 2026-07-03 · Phase 6-7 dalgaları; her dalga = prereg-freeze + master protokol §9 zorunlu kontrolleri. Mezarlık aileleri YENİDEN AÇILMAZ (failure archive + docs/ami/AMI_RESEARCH_PROTOCOLS.md §5); yalnız kayıtlı retry-condition sağlanırsa YENİ prereg ile.

| Dalga | Aileler | Önkoşul | Not |
|---|---|---|---|
| W1 | Cycle integrity & dedup + Q001–024 mevcut-verdict mapping | P3 | Mevcut sonuçların warehouse'a dürüst taşınması; yeni iddia yok |
| W2 | Unconditional LONG genesis + failed-fade LONG | P3 + candidate universe | **ERTELENDİ (OD-012, 2026-07-04):** mezarlık #8 (BUY-side fade)/#17 (BUYFADE LONG genesis T0) ile çakışıyor; #17 retry-koşulu (OI verisi) kontrol edildi, zayıf (38/252 anchor, 2 kopuk pencere) — OI verisi gerçek/sürekli birikmeden yeniden açılmaz |
| W3 | Entry timing (early/T0/delayed/late, iki yön) | P5 engines | **ZORLA AÇILMADI (2026-07-04, E-W3-ENTRY-TIMING-RECONCILIATION-001):** 10 mevcut rapor reconcile edildi, 9/10 zaten mezarlıkta (#1/#2/#8/#16) veya süperseded/N-yetersiz; tek açık soru (absorption-timing permission-feature, N=36-51) OI ile aynı yetersiz-örneklem riskini taşıyor — zorlanmadı |
| W4 | Post-event path taxonomy + structural location + event geometry | P3 paths | **TAMAMLANDI (2026-07-04, E-W4-POST-EVENT-PATH-TAXONOMY-001):** descriptive-only. Swing ufkunda REVERSAL 54.3%, anchor'lar rastgele-zamandan +10.6 puan daha REVERSAL-ağırlıklı (mean-reversion-after-cascade descriptive gerçek). C2-C4 yapısal-konum bayrakları DEGENERATE (eşik çok gevşek, düzeltilmedi). Takip adayı: sıkı eşikli yeni prereg VEYA W5 |
| W5 | Candle morphology + swing grammar + sweep + breakout/retest (chart W1) + unconditional SHORT genesis | P4-5 + Opus B | **KISMEN TAMAMLANDI (2026-07-04, E-W5A-MORPHOLOGY-SWING-GRAMMAR-001):** candle morphology + swing grammar çalıştırıldı — null-sonuç (REVERSAL oranını ayırt etmiyor). Sweep/breakout-retest (NOT_IMPLEMENTED altyapı) + unconditional SHORT genesis (whitepaper §29 feature'ları eksik) OD-014 ile backlog'a alındı, zorlanmadı. Descriptive → matched-control → verdict; observer YOK |
| W6 | Compression + channel + RS + session (chart W2) | P4-5 | **KISMEN TAMAMLANDI (2026-07-04, E-W6-COMPRESSION-RS-SESSION-001):** compression+RS+session çalıştırıldı. RS anlamlı (RS_ETH_STRONG REVERSAL %68.1 vs WEAK %53.4); compression cascade'lerde nadir (n=3, beklenen); session farksız (ASIA/US ~%56). Channel (CHANNEL_BOUNDARY NOT_IMPLEMENTED) OD-015 ile backlog'a alındı |
| W7 | Signal aging + market clock + state aging | P3 + P5 | |
| W8 | Competing-risk hold + progress-conditioned management | P7 timing metrics | **HOLD-BASELINE TAMAMLANDI (2026-07-04, E-W8-HOLD-BASELINE-001):** yalnız zorunlu ön-koşul (fixed-horizon MFE/MAE benchmark) çalıştırıldı, competing-risk/management hipotezi HENÜZ AÇILMADI. LONG (8/8 hücre) STABLE_BASELINE; SHORT (8/8 hücre) INSUFFICIENT_SAMPLE (N=32-50, 70/30 split MIN_BUCKET_N=20'yi karşılamıyor — dürüstçe raporlandı, birleştirilmedi). Negatif kontrol: direction-ratio eşleştirilemedi (BLOCKED_FOR_DIRECTION_MATCHING, fabrikasyon yapılmadı), chronological/session/vol-bucket eşleşti (0 shortfall). Management-rule testi (#3/#4/#10/#13/#18 mezarlığıyla çakışma riski) BU DALGADA AÇILMADI |
| W9 | Stop taxonomy + re-entry + setup cancellation | P7 | BAD_TIMING alt-sinyali yalnız OD-008 onayıyla |
| W10 | LONG↔SHORT transitions + multi-TF conflict | P7 | |
| W11 | Multi-sensor liq/cascade proxy reconstruction + real-vs-proxy karşılaştırma | P3 identity + event_family alanları | R-09 kontrolü burada test edilir |
| W12 | Position-aware action-value karşılaştırması | W4+W8 | Action-value output order üretemez (mutation test) |

Raporlama: her dalga → `reports/research/s34/` .md+.json + warehouse verdict + (varsa) failure archive; historical sonuç forward olarak KAYDEDİLMEZ.
