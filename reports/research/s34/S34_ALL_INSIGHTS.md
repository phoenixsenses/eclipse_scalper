# S34 — Tum SQL Data: Kapsamli Meta-Analiz

> Kaynak: `S34_ALL.db` (SADECE okundu). 44253 research satir / 258 rapor. 
> Uretim: 2026-07-01 22:42 UTC. Live/.env/sizing DOKUNULMADI.

## 1) Konsensus Tarayici (concept, raporlar arasi)

Bir concept ne kadar cok RAPORDA + yuksek WR + yuksek anlamlilik ile geciyorsa o kadar guvenilir.

| Concept | #satir | #rapor | ort WR | med total | %anlamli(mc<=.05) | %pozitif |
|---|--:|--:|--:|--:|--:|--:|
| short | 1278 | 44 | 15% | -369 | 62% | 13% |
| regime | 542 | 32 | 25% | +1685 | 84% | 15% |
| sync | 267 | 25 | 32% | +300 | 88% | 27% |
| score>=3 | 374 | 22 | 12% | +607 | 94% | 13% |
| silence | 542 | 21 | 18% | +1678 | 87% | 15% |
| score>=4 | 165 | 19 | 13% | +2610 | 86% | 15% |
| noisy | 294 | 22 | 32% | +996 | 63% | 27% |
| btc1m | 349 | 18 | 20% | +1394 | 77% | 11% |
| deep7d | 117 | 14 | 59% | +338 | 83% | 43% |
| buy-side | 423 | 25 | 31% | -1488 | 45% | 9% |
| ofi | 275 | 16 | 11% | +1923 | 71% | 11% |
| score>=5 | 47 | 9 | 6% | +1132 | 100% | 9% |
| cross-asset | 231 | 30 | 78% | -161 | 30% | 33% |
| echo | 144 | 10 | 49% | +2871 | 85% | 2% |
| navigation | 360 | 15 | 48% | +74 | 56% | 3% |
| funding | 149 | 10 | 46% | +5966 | 79% | 64% |
| density | 20 | 7 | 47% | +4021 | 100% | 20% |
| prebuild | 71 | 11 | 28% | +2160 | 63% | 7% |
| 100k | 110 | 16 | 26% | +1416 | 42% | 32% |
| fade | 227 | 12 | 50% | -926 | 56% | 28% |
| btc2m | 40 | 9 | 57% | +930 | 69% | 28% |
| limit-entry | 19 | 5 | 39% | +2407 | 100% | 47% |
| hour17 | 277 | 5 | 65% | +2160 | 79% | 98% |
| scale-in | 56 | 3 | 12% | +607 | 100% | 12% |
| shelf | 9 | 3 | 76% | +2759 | 89% | 100% |
| composite | 17 | 4 | 50% | +1017 | 60% | 59% |
| T15/bounce | 49 | 8 | 36% | +490 | 29% | 22% |
| rv5m | 7 | 4 | 71% | +2621 | 57% | 71% |
| whale | 9 | 2 | 77% | +2623 | 78% | 100% |
| vol-compress | 3 | 2 | 78% | +807 | 50% | 33% |
| reversal | 164 | 2 | 0% | -1400 | 0% | 6% |

## 2) Overfit / Suphe Dedektoru

Kucuk-N dev-sayi, cok-yuksek WR dusuk-N, veya mc_p olmayan buyuk-total.

| Report | key | N | WR | total | mc_p | flag |
|---|---|--:|--:|--:|--:|---|
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | same_symbol_transition.gap_by_transition | 283 | 100% | +1216576 | yok | mc_p-yok-buyuk-total |
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | same_symbol_transition.gap_by_transition | 261 | 100% | +1031886 | yok | mc_p-yok-buyuk-total |
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | same_symbol_transition.gap_by_transition | 233 | 100% | +897601 | yok | mc_p-yok-buyuk-total |
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | same_symbol_transition.gap_by_transition | 226 | 100% | +840603 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[3].all.all | 4261 | - | +118822 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[0].all.all | 4261 | - | +118822 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[3].all.by_symbol.ETHUS | 1642 | - | +102163 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[0].all.by_symbol.ETHUSDT | 1642 | - | +102163 | yok | mc_p-yok-buyuk-total |
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | event_end_vs_anchor.SELL.reclaim_delay_s | 544 | 99% | +86816 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[4].all.all | 2735 | - | +85458 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[1].all.all | 2735 | - | +85458 | yok | mc_p-yok-buyuk-total |
| S34_V02_EVENT_CHAIN_PUZZLE_TESTS | event_end_vs_anchor.BUY.reclaim_delay_se | 509 | 100% | +75944 | yok | mc_p-yok-buyuk-total |
| S34_NAVIGATION_BRIDGE | exit_preference_validation.actual_4h_min | 1038 | 100% | +70560 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[3].cal.all | 3335 | - | +70359 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[0].cal.all | 3335 | - | +70359 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[3].all.positive_fundin | 2167 | - | +69864 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[0].all.positive_funding_s | 2167 | - | +69864 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[4].all.by_symbol.ETHUS | 1031 | - | +62142 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | ranked_configs[1].all.by_symbol.ETHUSDT | 1031 | - | +62142 | yok | mc_p-yok-buyuk-total |
| FUNDING_EXTREME_MEAN_REVERSION | consistent_passes[3].cal.by_symbol.ETHUS | 1337 | - | +57965 | yok | mc_p-yok-buyuk-total |

> Toplam suphe flag'i: 4590 satir.

## 3) Celiski Bulucu (ayni concept, zit sonuc)

Hem guclu-pozitif hem olu sonucu olan concept'ler = metodoloji hassas (dikkat).

| Concept | anlamli-pozitif | olu(mc>.5 veya total<0) | celiski? |
|---|--:|--:|---|
| silence | 23 | 24 | EVET |
| sync | 31 | 51 | EVET |
| funding | 6 | 22 | EVET |
| hour17 | 218 | 5 | EVET |
| regime | 35 | 9 | EVET |
| score>=3 | 27 | 22 | EVET |
| composite | 3 | 7 | EVET |
| 100k | 20 | 19 | EVET |
| short | 43 | 235 | EVET |
| noisy | 40 | 14 | EVET |
| btc1m | 27 | 6 | EVET |
| btc2m | 7 | 3 | EVET |
| cross-asset | 4 | 101 | EVET |
| deep7d | 5 | 24 | EVET |
| fade | 5 | 134 | EVET |
| buy-side | 5 | 159 | EVET |
| navigation | 5 | 10 | EVET |

## 4) Mezarlik — Reddedilen Hipotezler (bir daha test etme)

mc_p>0.6 veya avg<-8bps (N>=15). Toplam 613 satir, 613 benzersiz.

| Report | label | N | WR | avg | mc_p |
|---|---|--:|--:|--:|--:|
| S34_FEATURE_FACTORY_PHASE1_ETH_BUY_200K | summaries[2] | 450 | 0% | -28 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP120 DEL | 434 | 0% | -27 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP60 DELA | 434 | 0% | -25 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP80 DELA | 434 | 0% | -25 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP120 DEL | 434 | 0% | -24 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP40 DELA | 434 | 0% | -23 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP80 DELA | 434 | 0% | -23 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP60 DELA | 434 | 0% | -22 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP40 DELA | 434 | 0% | -22 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP120 DEL | 434 | 0% | -21 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP60 DELA | 434 | 0% | -20 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP120 DEL | 434 | 0% | -20 | - |
| S34_SILENCE_CORE_FINAL | 150K prov6 fee=20 | 454 | 24% | -19 | 1.000 |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP40 DELA | 434 | 0% | -19 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP120 DEL | 434 | 0% | -19 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP80 DELA | 434 | 0% | -19 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP40 DELA | 434 | 0% | -19 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP60 DELA | 434 | 0% | -18 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 100000 TP120 DE | 273 | 0% | -28 | - |
| S34_BUY_LIQ_REVERSAL_SHORT_2026-06-07_15 | summaries[BUY_REVERSAL_SHORT 50000 TP80 DELA | 434 | 0% | -17 | - |

## 5) Research vs Gercek Paper (tahmin-gerceklik)

| Paper signal | paper N | paper WR | paper avg | ~research WR | ~research avg | not |
|---|--:|--:|--:|--:|--:|---|
| LONG_SILENCE | 399 | 44% | +14.9 | 1% | +25 | gap: gercek<research (silence lookahead) |
| SHORT_NEITHER | 96 | 71% | +78.3 | 1% | +12 | fark var |
| SHORT_NOISY | 253 | 55% | +24.3 | 1% | +12 | fark var |

## 6) Feature Konsensus Siralamasi (kazanan config'lerde geçme)

Kazanan config = N>=15, WR>=70, mc_p<=0.05 (445 satir). Feature ne kadar cok geciyorsa o kadar konsensus.

| Feature | kazananda #satir | tum-veride #satir | lift |
|---|--:|--:|--:|
| whale | 7 | 9 | 77.35x |
| shelf | 6 | 9 | 66.30x |
| rv5m | 4 | 7 | 56.83x |
| be_ratio | 6 | 14 | 42.62x |
| echo | 37 | 147 | 25.03x |
| hour17 | 60 | 285 | 20.94x |
| density | 9 | 84 | 10.65x |
| prebuild | 7 | 78 | 8.92x |
| regime | 65 | 747 | 8.65x |
| sync | 20 | 281 | 7.08x |
| funding | 5 | 174 | 2.86x |
| score | 42 | 2183 | 1.91x |
| silence | 11 | 611 | 1.79x |
| ofi | 6 | 393 | 1.52x |

## 7) Kapsama Haritasi (asset x yon)

| Asset | LONG satir | SHORT satir | notr |
|---|--:|--:|--:|
| ETH | 109 | 109 | 1021 |
| BTC | 700 | 472 | 2037 |
| SOL | 31 | 33 | 104 |

## 8) Kelly / Risk Kalibrasyonu (conviction sleeve'leri)

Kelly-yaklasik f* = WR - (1-WR)/R, R=avg_kazanc/avg_kayip (worst proxy). Sadece rehber.

| Config | N | WR | avg | worst | onerilen risk-tier |
|---|--:|--:|--:|--:|---|
| V_s8_ge3_full | 85 | 74% | +64 | +0 | MODERATE (f~0.29) |
| V_s8_ge4_full | 56 | 75% | +75 | +0 | MODERATE (f~0.35) |
| M2_s5_full | 26 | 85% | +104 | -107 | AGGRESSIVE (f~0.71) |
| V_100K_s3_noov | 53 | 66% | +55 | +0 | CONSERVATIVE (f~0.11) |
| gate4_noov | 21 | 71% | +55 | -184 | MODERATE (f~0.22) |

---
*Uretim: tools/analyze_s34_all_sql.py — sadece S34_ALL.db okundu.*