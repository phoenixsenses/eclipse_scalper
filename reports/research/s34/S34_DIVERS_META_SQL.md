# S34 Diversification Meta (SQL)

> research_clean trust>=2, n>=8. risk_adj = total / max(|mdd or worst|, 50).

## interaction

| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| results.I.I_rv+shelf | S34_HORIZON | 45 | 10.0 | 77.8 | 87.9 | 3957.0 | None | None | 0.0 | 39.57 |
| results.I.I_rv+whale_lo | S34_HORIZON | 41 | 9.1 | 80.5 | 94.6 | 3877.0 | None | None | 0.0 | 38.77 |
| results.I.I_sync+be | S34_HORIZON | 49 | 10.9 | 79.6 | 72.4 | 3545.0 | None | None | 0.0 | 35.45 |
| results.I.I_sync+rv | S34_HORIZON | 41 | 9.1 | 78.0 | 84.8 | 3477.0 | None | None | 0.0 | 34.77 |
| results.I.I_shelf+whale_lo | S34_HORIZON | 35 | 7.8 | 80.0 | 95.3 | 3335.0 | None | None | 0.0 | 33.35 |
| results.I.I_sync+shelf | S34_HORIZON | 29 | 6.4 | 82.8 | 95.1 | 2759.0 | None | None | 0.0 | 27.59 |
| results.I.I_sync+whale_lo | S34_HORIZON | 30 | 6.7 | 86.7 | 87.4 | 2623.0 | None | None | 0.0 | 26.23 |
| results.I.I_rv+be | S34_HORIZON | 32 | 7.1 | 75.0 | 74.3 | 2377.0 | None | None | 0.0 | 23.77 |
| sections.S3.S3_hour_btc4h_TEST | S34_SILENCE_PREDICTOR | 14 | 10.3 | 85.7 | 132.2 | 1850.0 | -43.5 | -79.0 | 0.0 | 23.42 |
| results.I.I_be+whale_lo | S34_HORIZON | 31 | 6.9 | 80.6 | 68.1 | 2111.0 | None | None | 0.0 | 21.11 |
| results.I.I_shelf+be | S34_HORIZON | 22 | 4.9 | 86.4 | 81.0 | 1782.0 | None | None | 0.0 | 17.82 |
| sections.S3.S3_hour_btc4h_FULL | S34_SILENCE_PREDICTOR | 47 | 10.4 | 74.5 | 80.8 | 3798.0 | -154.6 | -253.0 | 0.0 | 15.01 |
| sections.S3.S3_hour_btc7d_TEST | S34_SILENCE_PREDICTOR | 8 | 5.9 | 87.5 | 93.3 | 746.0 | -15.6 | -16.0 | 0.014 | 14.92 |
| sections.S3.S3_hour_sync_k_FULL | S34_SILENCE_PREDICTOR | 38 | 8.4 | 78.9 | 68.9 | 2620.0 | -184.4 | -184.0 | 0.0 | 14.24 |
| sections.S3.S3_sync_k_btc7d_FULL | S34_SILENCE_PREDICTOR | 41 | 9.1 | 75.6 | 77.5 | 3179.0 | -138.8 | -227.0 | 0.0 | 14.0 |

## premium_sleeve

| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| results.M.M1_deep7d_s>=4 | S34_FRONTIER | 28 | 6.2 | 82.1 | 111.0 | 3109.0 | -64.0 | None | 0.0 | 48.58 |
| results.H.H_12h | S34_HORIZON | 54 | 12.0 | 66.7 | 84.0 | 4534.0 | None | None | 0.0 | 45.34 |
| results.gate4_TEST | S34_CONVICTION_COMPOSITE | 20 | 14.8 | 90.0 | 113.0 | 2261.0 | -44.8 | -51.0 | 0.0 | 44.33 |
| results.H.H_6h | S34_HORIZON | 54 | 12.0 | 75.9 | 78.9 | 4259.0 | None | None | 0.0 | 42.59 |
| results.V.V_s8_ge4_full | S34_HORIZON | 56 | 12.4 | 75.0 | 75.1 | 4205.0 | None | None | 0.0 | 42.05 |
| results.V.V_s7_ge4_full | S34_HORIZON | 50 | 11.1 | 76.0 | 78.1 | 3907.0 | None | None | 0.0 | 39.07 |
| results.H.H_24h | S34_HORIZON | 54 | 12.0 | 66.7 | 66.5 | 3592.0 | None | None | 0.03 | 35.92 |
| results.N.N1_h17-19_deep7d_s>=4 | S34_FRONTIER | 15 | 3.3 | 80.0 | 120.1 | 1801.0 | -64.0 | None | 0.002 | 28.14 |
| results.V.V_s8_ge4_TEST | S34_HORIZON | 27 | 20.0 | 85.2 | 100.7 | 2718.0 | None | None | 0.0 | 27.18 |
| results.V.V_s7_ge4_TEST | S34_HORIZON | 25 | 18.5 | 84.0 | 104.4 | 2610.0 | None | None | 0.0 | 26.1 |
| results.M.M2_s5_full | S34_FRONTIER | 26 | 5.8 | 84.6 | 104.4 | 2714.0 | -107.4 | None | 0.0 | 25.27 |
| results.N.N1_h20-23_deep7d_s>=4 | S34_FRONTIER | 13 | 2.9 | 84.6 | 100.6 | 1308.0 | -51.8 | None | 0.0 | 25.25 |
| hour17.150K.tail.vetoes.exclude_btc5m_lt_minus50.dropped | S34_FULL_SIGNAL_BOOST | 16 | 3.5 | 75.0 | 105.2 | 1682.8 | -67.1 | -67.1 | 0.0 | 25.08 |
| hour17.200K.tail.vetoes.exclude_btc5m_lt_minus50.dropped | S34_FULL_SIGNAL_BOOST | 14 | 3.1 | 85.7 | 107.3 | 1501.7 | -63.3 | -63.3 | 0.001 | 23.72 |
| results.M.M2_s4_full | S34_FRONTIER | 50 | 11.1 | 76.0 | 78.1 | 3907.0 | -183.9 | None | 0.0 | 21.25 |

## sizing_weighted

| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| oos_validation.test_auto_weights.primary_route | S34_LIQ_OUTCOME_CALCULATOR_O | 50 | None | 48.0 | 11.97917903766504 | 598.958951883252 | None | None | None | 5.99 |

## lean_composite

| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| hour17.150K.entry_exit.entry.d1_spread_clean.no_overlap | S34_FULL_SIGNAL_BOOST | 73 | 16.2 | 61.6 | 47.8 | 3492.1 | -185.1 | -320.9 | 0.001 | 10.88 |
| hour17.200K.entry_exit.entry.d1_spread_clean.full | S34_FULL_SIGNAL_BOOST | 126 | 27.9 | 65.1 | 43.8 | 5512.9 | -435.6 | -513.9 | 0.0 | 10.73 |
| hour17.200K.entry_exit.entry.d1_spread_clean.no_overlap | S34_FULL_SIGNAL_BOOST | 63 | 14.0 | 61.9 | 40.7 | 2563.4 | -185.1 | -313.0 | 0.011 | 8.19 |
| hour17.150K.entry_exit.entry.d1_spread_clean.full | S34_FULL_SIGNAL_BOOST | 156 | 34.5 | 62.2 | 42.9 | 6696.9 | -435.6 | -982.7 | 0.0 | 6.81 |

## portfolio

| key | report | N | /ay | WR | avg | TOT | worst | mdd | mc_p | risk_adj |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| hour17.150K.confidence.top_combos[3].full | S34_FULL_SIGNAL_BOOST | 17 | 3.8 | 100.0 | 109.1 | 1854.2 | 7.2 | 0.0 | 0.0 | 37.08 |
| hour17.150K.confidence.top_combos[0].full | S34_FULL_SIGNAL_BOOST | 23 | 5.1 | 82.6 | 137.8 | 3170.3 | -56.0 | -98.1 | 0.0 | 32.32 |
| hour17.150K.confidence.top_combos[1].full | S34_FULL_SIGNAL_BOOST | 14 | 3.1 | 92.9 | 113.8 | 1592.9 | -11.0 | -11.0 | 0.001 | 31.86 |
| hour17.150K.confidence.top_combos[0].no_overlap | S34_FULL_SIGNAL_BOOST | 11 | 2.4 | 81.8 | 142.1 | 1563.3 | -56.0 | -56.0 | 0.003 | 27.92 |
| hour17.150K.confidence.top_combos[1].no_overlap | S34_FULL_SIGNAL_BOOST | 11 | 2.4 | 100.0 | 118.5 | 1303.5 | 16.0 | 0.0 | 0.0 | 26.07 |
| results.P.P1_mini_score3 | S34_FRONTIER | 97 | 21.5 | 64.9 | 55.1 | 5345.0 | -206.3 | None | 0.0 | 25.91 |
| portfolio.portfolio_stats.all_three | S34_FULL_SIGNAL_BOOST | 217 | 48.1 | 66.8 | 32.2 | 6983.5 | -173.4 | -311.0 | 0.0 | 22.45 |
| hour17.150K.confidence.top_combos[3].no_overlap | S34_FULL_SIGNAL_BOOST | 12 | 2.7 | 100.0 | 92.9 | 1114.5 | 7.2 | 0.0 | 0.0 | 22.29 |
| hour17.150K.confidence.top_combos[5].full | S34_FULL_SIGNAL_BOOST | 18 | 4.0 | 77.8 | 82.4 | 1483.6 | -67.1 | -67.1 | 0.002 | 22.11 |
| results.T.T_portfolio | S34_HORIZON | 39 | 8.7 | 69.2 | 56.4 | 2199.0 | None | None | 0.0 | 21.99 |
| hour17.150K.confidence.top_combos[4].full | S34_FULL_SIGNAL_BOOST | 33 | 7.3 | 78.8 | 101.7 | 3356.8 | -154.6 | -154.6 | 0.0 | 21.71 |
| hour17.200K.confidence.top_combos[0].full | S34_FULL_SIGNAL_BOOST | 24 | 5.3 | 91.7 | 127.7 | 3064.3 | -154.6 | -163.1 | 0.0 | 18.79 |
| portfolio.portfolio_stats.h17_plus_buy | S34_FULL_SIGNAL_BOOST | 206 | 45.6 | 66.0 | 27.7 | 5712.6 | -173.4 | -311.0 | 0.0 | 18.37 |
| hour17.200K.confidence.top_combos[3].full | S34_FULL_SIGNAL_BOOST | 30 | 6.6 | 86.7 | 105.3 | 3159.4 | -154.6 | -177.0 | 0.0 | 17.85 |
| portfolio.portfolio_stats.buy_fade_only | S34_FULL_SIGNAL_BOOST | 177 | 39.2 | 66.7 | 24.6 | 4349.9 | -80.0 | -243.7 | 0.0 | 17.85 |
