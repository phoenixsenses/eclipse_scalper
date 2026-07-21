# S34 — Temiz / Canonical Research View

> `research_clean` tablosu S34_ALL.db'ye eklendi. 43458 satir (bozuk-rapor 795 atildi). bps-temiz: 43416. 
> Uretim: 2026-07-01 22:59 UTC. Live/.env/sizing DOKUNULMADI.

**Filtreler:** non-bps raporlar atildi; wr 0-1 ise *100 normalize; bps_ok = |total|<50k & |avg|<2k & wr<=100. 
**trust (0-4):** N>=15 + mc_p<=.05 + bps_ok + (OOS/noov/holdout kanit).

## En Guvenilir Sonuclar (trust=4, bps-temiz, total'a gore)

| Report | key | dir | N | /ay | WR | avg | total | mc_p |
|---|---|---|--:|--:|--:|--:|--:|--:|
| S34_SILENCE_CORE_FINAL | sections.R.R_100K_holdall6 |  | 602 | 133.1 | 59% | +16 | +9546 | 0.004 |
| S34_SILENCE_CORE_FINAL | sections.H.H_100K_stopnone |  | 602 | 133.1 | 59% | +16 | +9546 | 0.004 |
| S34_SILENCE_CORE_FINAL | sections.R.R_150K_holdall6 |  | 454 | 100.4 | 59% | +20 | +9223 | 0.003 |
| S34_SILENCE_CORE_FINAL | sections.H.H_150K_stopnone |  | 454 | 100.4 | 59% | +20 | +9223 | 0.003 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_150K_base_all |  | 454 | 100.5 | 59% | +20 | +9187 | 0.003 |
| S34_ALPHA_ATTRIBUTION | sections.B.B_100K_sil_reg |  | 276 | 61.0 | 63% | +32 | +8693 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.R.R_100K_ideal_sil6 |  | 276 | 61.0 | 62% | +30 | +8337 | 0.000 |
| S34_ALPHA_ATTRIBUTION | sections.B.B_100K_sil |  | 331 | 73.2 | 60% | +25 | +8322 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.R.R_200K_holdall6 |  | 373 | 82.5 | 61% | +21 | +7755 | 0.005 |
| S34_SILENCE_CORE_FINAL | sections.H.H_200K_stopnone |  | 373 | 82.5 | 61% | +21 | +7755 | 0.005 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_200K_base_all |  | 373 | 82.6 | 61% | +21 | +7719 | 0.005 |
| S34_ALPHA_ATTRIBUTION | sections.C.C1_silence |  | 255 | 56.4 | 60% | +28 | +7225 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.H.H_100K_stop200 |  | 602 | 133.1 | 58% | +12 | +7166 | 0.013 |
| S34_SILENCE_CORE_FINAL | sections.H.H_150K_stop200 |  | 454 | 100.4 | 58% | +15 | +6662 | 0.018 |
| S34_SILENCE_PREDICTOR | sections.S2.S2_hour_FULL |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_150K_pred_FULL |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_SILENCE_PREDICTOR | sections.S6.S6_150K_full |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_SILENCE_PREDICTOR | sections.S2.S2_sync_k_FULL |  | 141 | 31.2 | 67% | +44 | +6185 | 0.000 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_200K_pred_FULL |  | 151 | 33.4 | 67% | +40 | +6119 | 0.000 |
| S34_ALPHA_ATTRIBUTION | sections.D.D_no_silence_6h |  | 194 | 42.9 | 63% | +31 | +5991 | 0.001 |
| S34_ALPHA_ATTRIBUTION | sections.D.D_sil_regime_6h |  | 142 | 31.4 | 63% | +42 | +5960 | 0.001 |
| S34_SILENCE_CORE_FINAL | sections.R.R_200K_ideal_sil6 |  | 142 | 31.4 | 63% | +42 | +5960 | 0.001 |
| S34_ALPHA_ATTRIBUTION | sections.C.C3_minus_not_EU |  | 187 | 41.4 | 63% | +32 | +5908 | 0.002 |
| S34_ALPHA_ATTRIBUTION | sections.C.C3_minus_not_bull |  | 145 | 32.1 | 64% | +40 | +5783 | 0.001 |
| S34_SILENCE_CORE_FINAL | sections.H.H_200K_stop200 |  | 373 | 82.5 | 59% | +15 | +5517 | 0.029 |
| S34_ALPHA_ATTRIBUTION | sections.A.A_min_LONG_l6h | LONG | 447 | 98.8 | 57% | +12 | +5416 | 0.047 |
| S34_ALPHA_ATTRIBUTION | sections.D.D_min_6h |  | 447 | 98.8 | 57% | +12 | +5416 | 0.047 |
| S34_SILENCE_PREDICTOR | sections.S2.S2_btc7d_FULL |  | 129 | 28.6 | 69% | +41 | +5313 | 0.000 |
| S34_ALPHA_ATTRIBUTION | sections.E.E_long_short | SHORT | 71 | 15.7 | 72% | +74 | +5247 | 0.000 |
| S34_ALPHA_ATTRIBUTION | sections.E.E_all_three |  | 71 | 15.7 | 72% | +74 | +5247 | 0.000 |

## En Guvenilir WR (trust=4, N>=20)

| Report | key | dir | N | WR | avg | total |
|---|---|---|--:|--:|--:|--:|
| S34_NEXT_TESTS_V1 | T6.combo_current_short_btc1M | SHORT | 25 | 96% | +164 | - |
| S34_ECHO_EXPANSION | sections.TAILWR.S4_regime_prebuild |  | 24 | 92% | +118 | - |
| S34_NEXT_TESTS_V1 | T6.combo_score1_btc7d_short_btc1M | SHORT | 33 | 91% | +139 | - |
| S34_CONVICTION_COMPOSITE | results.gate4_TEST |  | 20 | 90% | +113 | +2261 |
| S34_MEGA_V1 | A.A2_short_btc500K | SHORT | 20 | 90% | +134 | - |
| S34_NEXT_TESTS_V1 | T6.combo_score1_btc7d_short_btc2M | SHORT | 27 | 89% | +149 | - |
| S34_ALPHA_ATTRIBUTION | sections.E.E_echo_long | LONG | 26 | 88% | +96 | +2495 |
| S34_ECHO_LIVE_GAUNTLET | sections.TAIL.V_not_us1314 |  | 34 | 85% | +88 | - |
| S34_HORIZON | results.V.V_s8_ge4_TEST |  | 27 | 85% | +101 | +2718 |
| S34_FULL_SIGNAL_BOOST | hour17.200K.confidence.feature_ranking[2 |  | 20 | 85% | +112 | +2243 |
| S34_IDEAS_V2 | EW_echo_60_180_sil | LONG | 20 | 85% | +106 | - |
| S34_SIGNAL_MINING | rows[sync_ratio].test |  | 20 | 85% | +103 | +2052 |
| S34_ECHO_LIVE_GAUNTLET | sections.HOLD.HOLD_e3090_2h |  | 32 | 84% | +48 | - |
| S34_ECHO_LIVE_GAUNTLET | sections.HOLD.HOLD_e3090_3h |  | 32 | 84% | +82 | - |
| S34_ECHO_LIVE_GAUNTLET | sections.HOLD.HOLD_e3090_6h |  | 32 | 84% | +105 | - |
| S34_HORIZON | results.V.V_s7_ge4_TEST |  | 25 | 84% | +104 | +2610 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.confidence.feature_ranking[3 |  | 23 | 83% | +89 | +2055 |
| S34_IDEAS_V2 | LAG_no_btc | LONG | 23 | 83% | +91 | - |
| S34_CONVICTION_COMPOSITE | results.gate3_TEST |  | 28 | 82% | +101 | +2837 |
| S34_SIGNAL_MINING | rows[ofi_5m].test |  | 22 | 82% | +87 | +1923 |

## Trust Dagilimi

| trust | satir |
|--:|--:|
| 4 | 536 |
| 3 | 3960 |
| 2 | 22374 |
| 1 | 16588 |

## Canonical Alpha Registry (aile + temiz kanit sayisi)

| Signal | dir | durum | temiz-kanit(trust>=3) | not |
|---|---|---|--:|---|
| LONG_T15_BOUNCE | LONG | LEGACY_LIVE | 328 | T+15 bounce confirm (eski live LONG legi) |
| LONG_SILENCE | LONG | LEGACY_PAPER | 328 | silence LONG (lookahead cikti — arsiv) |
| LONG_HOUR17_HOLD6H | LONG | LIVE | 328 | Ana canli alpha: hour>=17 UTC + regime, T0, 6h, 30 |
| SHORT_NEITHER | SHORT | LIVE | 202 | BTC SELL>=2M confirm -> SHORT 2h |
| BUY_FADE_SHORT_H45_SL75 | SHORT | PAPER | 137 | BUY-side fade short 45m/75bps |
| LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE | LONG | PAPER | 328 | prebuild>=2 double cascade |
| LONG_ECHO_45_120_SILENCE | LONG | PAPER | 328 | echo 45-120m + silence |
| LONG_HOUR17_100K_COMPOSITE | LONG | PAPER | 328 | Frekans genisletme, composite score>=3 |
| LONG_HOUR17_COMPOSITE | LONG | PAPER | 328 | Composite conviction 0-8 (score>=3) + funding-veto |
| LONG_OFI_SILENCE_BUYERS | LONG | PAPER | 328 | silence + OFI buyers |
| SHORT_BTC1M_D10_H3 | SHORT | PAPER | 202 | BTC1M delay10 3h |
| SHORT_BTC1M_H4 | SHORT | PAPER | 202 | BTC1M confirm 4h |
| SHORT_NOISY_BTC1M_D5_H180 | SHORT | PAPER | 202 | noisy + BTC1M confirm, 180m hold (diversifier) |

---
*Uretim: tools/build_s34_clean_view.py — S34_ALL.db research_clean tablosu.*