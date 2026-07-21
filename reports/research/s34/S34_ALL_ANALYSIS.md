# S34 — Tum Alpha Ailesi: SQL Export + Analiz

> Uretim: 2026-07-01 21:59 UTC
> Kaynaklar: 332 research JSON (331 okundu, 0 atlandi) + 2 ledger.
> **research_results: 44253 satir / 258 rapor · paper_trades: 748 gercek trade**

DB: `S34_ALL.db` · SQL dump: `S34_ALL.sql` · bu analiz: `S34_ALL_ANALYSIS.md`

## Gercek Paper Trade PnL (ledger — sinyal basina)

| Signal | dir | N | WR | sum bps | avg bps | best | worst |
|---|---|--:|--:|--:|--:|--:|--:|
| LONG_SILENCE | LONG | 399 | 44.4% | +5943 | +14.9 | +609 | -466 |
| SHORT_NOISY | SHORT | 253 | 54.9% | +6147 | +24.3 | +407 | -239 |
| SHORT_NEITHER | SHORT | 96 | 70.8% | +7513 | +78.3 | +407 | -180 |

## En Yuksek TOTAL PnL — Research (N>=15, mc_p<=0.05)

| Report | Key | dir | N | /ay | WR | avg | TOTAL | mc_p |
|---|---|---|--:|--:|--:|--:|--:|--:|
| S34_SILENCE_CORE_FINAL | sections.R.R_100K_holdall6 |  | 602 | 133.1 | 59% | +16 | +9546 | 0.004 |
| S34_SILENCE_CORE_FINAL | sections.H.H_100K_stopnone |  | 602 | 133.1 | 59% | +16 | +9546 | 0.004 |
| S34_SILENCE_CORE_FINAL | sections.R.R_150K_holdall6 |  | 454 | 100.4 | 59% | +20 | +9223 | 0.003 |
| S34_SILENCE_CORE_FINAL | sections.H.H_150K_stopnone |  | 454 | 100.4 | 59% | +20 | +9223 | 0.003 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_150K_base_all |  | 454 | 100.5 | 59% | +20 | +9187 | 0.003 |
| S34_ALPHA_ATTRIBUTION | sections.B.B_100K_sil_reg |  | 276 | 61.0 | 63% | +32 | +8693 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.R.R_100K_ideal_sil6 |  | 276 | 61.0 | 62% | +30 | +8337 | 0.000 |
| S34_ALPHA_ATTRIBUTION | sections.B.B_100K_sil |  | 331 | 73.2 | 60% | +25 | +8322 | 0.000 |
| S34_EARLY_MGMT | sections.A.A1_100K_T0_raw |  | 215 | 47.7 | 63% | +38 | +8126 | 0.000 |
| S34_HORIZON | results.V.V_100K_s3_full |  | 130 | 28.9 | 68% | +60 | +7806 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.R.R_200K_holdall6 |  | 373 | 82.5 | 61% | +21 | +7755 | 0.005 |
| S34_SILENCE_CORE_FINAL | sections.H.H_200K_stopnone |  | 373 | 82.5 | 61% | +21 | +7755 | 0.005 |
| S34_SILENCE_PREDICTOR | sections.S5.S5_200K_base_all |  | 373 | 82.6 | 61% | +21 | +7719 | 0.005 |
| S34_ALPHA_ATTRIBUTION | sections.C.C1_silence |  | 255 | 56.4 | 60% | +28 | +7225 | 0.000 |
| S34_EARLY_MGMT | sections.C.C5_scalein100 |  | 126 | 28.0 | 71% | +57 | +7182 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.H.H_100K_stop200 |  | 602 | 133.1 | 58% | +12 | +7166 | 0.013 |
| S34_EARLY_MGMT | sections.C.C5_scalein150 |  | 126 | 28.0 | 71% | +57 | +7133 | 0.000 |
| S34_FULL_SIGNAL_BOOST | portfolio.portfolio_stats.all_three |  | 217 | 48.1 | 67% | +32 | +6984 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.entry.delay_1m.full |  | 156 | 34.5 | 62% | +43 | +6697 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.entry.d1_spread_clean.f |  | 156 | 34.5 | 62% | +43 | +6697 | 0.000 |
| S34_SILENCE_CORE_FINAL | sections.H.H_150K_stop200 |  | 454 | 100.4 | 58% | +15 | +6662 | 0.018 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.exit.profit_lock_200_10 |  | 156 | 34.5 | 62% | +42 | +6504 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.tail.vetoes.exclude_near_funding_3 |  | 143 | 31.7 | 64% | +45 | +6449 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.tail.vetoes.exclude_be_ratio_ge2.k |  | 151 | 33.4 | 63% | +42 | +6406 | 0.001 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.entry.delay_5m.full |  | 156 | 34.5 | 61% | +41 | +6386 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.confidence.base |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.tail.vetoes.exclude_spread_gt_0p35 |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.entry.delay_0m.full |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_FULL_SIGNAL_BOOST | hour17.150K.entry_exit.exit.hold_6h.full |  | 156 | 34.5 | 62% | +41 | +6360 | 0.000 |
| S34_EARLY_MGMT | sections.A.A1_150K_T0_raw |  | 156 | 34.6 | 62% | +41 | +6360 | 0.000 |

## En Yuksek WR — Research (N>=20)

| Report | Key | dir | N | WR | avg | TOTAL |
|---|---|---|--:|--:|--:|--:|
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[4].high |  | 135 | 6440% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[1].low |  | 136 | 6030% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[6].high |  | 136 | 5880% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[3].low |  | 135 | 5850% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | by_symbol.BTCUSDT.summary |  | 134 | 5820% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[8].high |  | 135 | 5700% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[2].high |  | 136 | 5660% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[3].low |  | 136 | 5660% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[8].high |  | 136 | 5660% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[0].low |  | 135 | 5630% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[1].low |  | 135 | 5630% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[5].high |  | 135 | 5630% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[6].high |  | 135 | 5630% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[5].low |  | 135 | 5560% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[4].low |  | 138 | 5510% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | holdout |  | 375 | 5330% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | overall |  | 541 | 5160% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[0].low |  | 136 | 5150% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[7].high |  | 136 | 5150% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[7].high |  | 139 | 5110% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | route_z_quartiles[8].low |  | 135 | 5110% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[7].low |  | 155 | 5100% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[3].high |  | 137 | 5040% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | by_symbol.ETHUSDT.summary |  | 256 | 5000% | - | - |
| S34_CONTINUOUS_ABSORPTION_REGRESSION | quartiles[2].low |  | 138 | 5000% | - | - |

## LONG vs SHORT (research, N>=10, mc_p<=0.05)

| dir | test sayisi | ort WR | ort avg | ort TOTAL |
|---|--:|--:|--:|--:|
| LONG | 161 | 73.2% | 69.6 | 4239.0 |
| SHORT | 107 | 74.3% | 97.4 | 1839.0 |

## Alpha Aileleri (taksonomi + gercek paper PnL)

| Signal | dir | universe | durum | paper N | paper WR | paper sum | not |
|---|---|---|---|--:|--:|--:|---|
| LONG_T15_BOUNCE | LONG | ETH SELL 200K | LEGACY_LIVE | - | - | - | T+15 bounce confirm (eski live LONG legi) |
| LONG_SILENCE | LONG | ETH SELL 200K | LEGACY_PAPER | 399 | 44% | +5943 | silence LONG (lookahead cikti — arsiv) |
| LONG_HOUR17_HOLD6H | LONG | ETH SELL 200K | LIVE | - | - | - | Ana canli alpha: hour>=17 UTC + regime, T0, 6h, 300bps stop, 15x |
| SHORT_NEITHER | SHORT | ETH SELL 200K | LIVE | 96 | 71% | +7513 | BTC SELL>=2M confirm -> SHORT 2h |
| BUY_FADE_SHORT_H45_SL75 | SHORT | ETH BUY 200K | PAPER | - | - | - | BUY-side fade short 45m/75bps |
| LONG_DOUBLE_CASCADE_PREBUILD2_SILENCE | LONG | ETH SELL 200K | PAPER | - | - | - | prebuild>=2 double cascade |
| LONG_ECHO_45_120_SILENCE | LONG | ETH SELL 200K | PAPER | - | - | - | echo 45-120m + silence |
| LONG_HOUR17_100K_COMPOSITE | LONG | ETH SELL 100-200K | PAPER | - | - | - | Frekans genisletme, composite score>=3 |
| LONG_HOUR17_COMPOSITE | LONG | ETH SELL 200K | PAPER | - | - | - | Composite conviction 0-8 (score>=3) + funding-veto |
| LONG_OFI_SILENCE_BUYERS | LONG | ETH SELL 200K | PAPER | - | - | - | silence + OFI buyers |
| SHORT_BTC1M_D10_H3 | SHORT | ETH SELL 200K | PAPER | - | - | - | BTC1M delay10 3h |
| SHORT_BTC1M_H4 | SHORT | ETH SELL 200K | PAPER | - | - | - | BTC1M confirm 4h |
| SHORT_NOISY_BTC1M_D5_H180 | SHORT | ETH SELL 200K | PAPER | - | - | - | noisy + BTC1M confirm, 180m hold (diversifier) |

## Rapor Basina Sonuc Sayisi (en cok 25)

| Report | sonuc |
|---|--:|
| S34_MAKER_FADE_DIAGNOSTICS | 12402 |
| S34_MICRO_ENTRY_SCALP | 2880 |
| S34_V02_ARMING_RETEST_PULLBACK | 2220 |
| S34_STATE_MACHINE_V7_FULL_DEVELOPMENT_SUITE | 1101 |
| S34_SELL_REVERSAL_FILTER_SWEEP_2026-06-07_15 | 1043 |
| S34_NAVIGATION_SCALP_AND_STRESS | 851 |
| S34_V_ENGINE_PORTFOLIO_MAP | 792 |
| S34_BUY_SIDE_STATE_MACHINE_GAUNTLET | 719 |
| S34_FEATURE_FACTORY_PHASE1_QUERY_RESULTS | 672 |
| S34_STATE_MACHINE_V8_ROBUSTNESS_SUITE | 589 |
| S34_LONG_RELAX_MANAGEMENT_SUITE | 549 |
| S34_V_ENGINE_POSITION_MANAGEMENT | 529 |
| S34_V4_DISSIPATION_MANAGEMENT_BACKTEST | 491 |
| S34_RESTORED_WINDOW_REPLAY_2026-06-07 | 464 |
| S34_V02_FREQUENCY_EXPANSION_TESTS | 441 |
| S34_CLUSTER_GEOMETRY_FEATURES | 416 |
| S34_V3_ROUTE_NODE_MAP | 404 |
| S34_STOP_TIGHTEN_ROBUSTNESS | 385 |
| S34_EARLY_BUILD_ENTRY | 384 |
| S34_STATE_MACHINE_V5_DEV_SUITE | 380 |
| S34_FULL_SIGNAL_BOOST | 376 |
| S34_STATE_MACHINE_V6_DEVELOPMENT_IDEAS | 366 |
| S34_STATE_MACHINE_V10_PROFIT_LOCK_FOLLOWUP | 365 |
| S34_BTC_1M_ROUTE_SWEEP | 336 |
| S34_MAKER_FADE_ETH_SELL_BRACKET | 324 |

## Ornek SQL sorgulari

```sql
-- en iyi LONG'lar:
SELECT report,key,n,wr,avg_bps,total_bps,mc_p FROM research_results WHERE direction='LONG' AND n>=15 AND mc_p<=0.05 ORDER BY total_bps DESC;
-- gercek paper trade'ler (hour17 composite):
SELECT * FROM paper_trades WHERE signal LIKE '%HOUR17%' ORDER BY closed_utc;
-- bir raporun tum sonuclari:
SELECT key,label,n,wr,avg_bps,total_bps FROM research_results WHERE report='S34_CONVICTION_COMPOSITE';
```
