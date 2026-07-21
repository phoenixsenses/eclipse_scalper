# S34 Market Confluence Scan

Generated: 2026-06-20T09:17:27.903963+00:00

Scope: ETH BUY feature-factory events, route `LONG_DELAY0_TP60`.

Confluence features use only BTC/SOL/ETH liquidation flow before the ETH event timestamp.

- Rows: `450`
- Predicate count: `136`

## OOS Candidates

| Rank | Candidate | Train N | Train Median | Train Cum | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | market_buy_liq_15m_ge_p75_303923 AND btc_eth_liq_ratio_15m_ge_p75_7 | 10 | +52.73 | +244.68 | 19 | +31.70 | +11.00 | +209.06 | +35.59 | 7/12 |
| 2 | market_buy_liq_15m_ge_p75_303923 AND btc_eth_liq_ratio_15m_ge_p50_2 | 24 | +52.39 | +477.89 | 46 | -8.07 | +10.29 | +473.23 | +290.85 | 14/22 |
| 3 | btc_buy_liq_15m_ge_p75_166951 AND btc_eth_liq_ratio_15m_ge_p75_7 | 19 | +52.38 | +485.65 | 23 | +52.21 | +15.92 | +366.14 | +187.47 | 10/15 |
| 4 | btc_buy_liq_15m_ge_p50_50092 AND market_buy_liq_15m_ge_p75_303923 | 41 | +52.34 | +773.69 | 67 | +31.70 | +15.80 | +1058.49 | +870.62 | 17/27 |
| 5 | btc_buy_liq_15m_ge_p75_166951 AND market_buy_liq_15m_ge_p75_303923 | 35 | +52.34 | +718.93 | 59 | +31.70 | +16.15 | +952.64 | +764.77 | 18/25 |
| 6 | btc_pre15_ge_20 AND market_buy_liq_15m_ge_p75_303923 | 37 | +52.34 | +686.43 | 48 | -1.10 | +13.42 | +643.93 | +457.78 | 17/23 |
| 7 | market_buy_liq_15m_ge_p75_303923 | 42 | +52.34 | +826.02 | 70 | +16.50 | +14.51 | +1015.57 | +827.70 | 18/27 |
| 8 | market_buy_liq_15m_ge_p50_111890 AND market_buy_liq_15m_ge_p75_303923 | 42 | +52.34 | +826.02 | 70 | +16.50 | +14.51 | +1015.57 | +827.70 | 18/27 |
| 9 | btc_pre15_ge_0 AND market_buy_liq_15m_ge_p75_303923 | 39 | +52.33 | +729.04 | 64 | +52.18 | +17.58 | +1124.88 | +937.01 | 18/26 |
| 10 | btc_buy_liq_15m_ge_p75_166951 | 48 | +52.23 | +1012.40 | 64 | +41.93 | +16.58 | +1061.15 | +873.28 | 18/26 |

## Real-Fill Parity

| Candidate | Total | Real Fill | No Fill | Test N | Test Median | Test Mean | Test Cum | Test Top3 Removed | Test Positive Days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| market_buy_liq_15m_ge_p75_303923 AND btc_eth_liq_ratio_15m_ge_p75_7 | 29 | 16 | 13 (44.8%) | 16 | +13.95 | +8.68 | +138.82 | -49.55 | 6/11 |
| market_buy_liq_15m_ge_p75_303923 AND btc_eth_liq_ratio_15m_ge_p50_2 | 70 | 34 | 36 (51.4%) | 34 | -9.43 | +2.90 | +98.64 | -91.68 | 8/15 |
| btc_buy_liq_15m_ge_p75_166951 AND btc_eth_liq_ratio_15m_ge_p75_7 | 42 | 16 | 26 (61.9%) | 16 | +13.95 | +8.68 | +138.82 | -49.55 | 6/11 |
| btc_buy_liq_15m_ge_p50_50092 AND market_buy_liq_15m_ge_p75_303923 | 108 | 44 | 64 (59.3%) | 44 | -0.53 | +11.71 | +515.29 | +302.06 | 9/16 |
| btc_buy_liq_15m_ge_p75_166951 AND market_buy_liq_15m_ge_p75_303923 | 94 | 41 | 53 (56.4%) | 41 | -2.88 | +10.47 | +429.08 | +230.76 | 10/16 |
| btc_pre15_ge_20 AND market_buy_liq_15m_ge_p75_303923 | 85 | 30 | 55 (64.7%) | 30 | -9.43 | +3.32 | +99.50 | -98.82 | 7/14 |
| market_buy_liq_15m_ge_p75_303923 | 112 | 47 | 65 (58.0%) | 47 | -2.88 | +10.15 | +477.10 | +263.87 | 9/16 |
| market_buy_liq_15m_ge_p50_111890 AND market_buy_liq_15m_ge_p75_303923 | 112 | 47 | 65 (58.0%) | 47 | -2.88 | +10.15 | +477.10 | +263.87 | 9/16 |
| btc_pre15_ge_0 AND market_buy_liq_15m_ge_p75_303923 | 103 | 42 | 61 (59.2%) | 42 | +16.30 | +13.00 | +546.06 | +332.84 | 10/16 |
| btc_buy_liq_15m_ge_p75_166951 | 112 | 42 | 70 (62.5%) | 42 | -5.11 | +8.97 | +376.55 | +178.22 | 10/16 |

## Read

This is a confluence research scan. Do not promote a candidate without separate forward pre-registration.
