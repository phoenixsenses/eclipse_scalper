# S34 V Engine Protective Stop

Generated: `2026-06-29T06:50:24.099115+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only stop overlays on the current live lifecycle `O20_W300_O5_C1`.

## Baseline

- eligible events: `48`
- filled rows: `23`
- baseline: N=23 sum=1112.6 med=37.0 T3R=433.6 max_loss=-144.4

## Stop Variants

| Rank | Variant | Exit N | Exit% | Summary | Delta sum | Delta T3R |
| ---: | --- | ---: | ---: | --- | ---: | ---: |
| 1 | `fixed_sl_150` | 1 | 4.3 | N=23 sum=1081.3 med=37.0 T3R=402.3 max_loss=-175.7 | -31.3 | -31.3 |
| 2 | `danger_15m_retlte-50_noreclaim` | 1 | 4.3 | N=23 sum=972.4 med=30.4 T3R=293.4 max_loss=-144.4 | -140.2 | -140.2 |
| 3 | `danger_15m_retlte-10_noreclaim` | 2 | 8.7 | N=23 sum=960.8 med=30.4 T3R=281.8 max_loss=-144.4 | -151.8 | -151.8 |
| 4 | `danger_15m_retlte-25_noreclaim` | 2 | 8.7 | N=23 sum=960.8 med=30.4 T3R=281.8 max_loss=-144.4 | -151.8 | -151.8 |
| 5 | `fixed_sl_100` | 2 | 8.7 | N=23 sum=954.4 med=30.4 T3R=275.4 max_loss=-116.3 | -158.2 | -158.2 |
| 6 | `hybrid_sl100_danger15m_ret-25_noreclaim` | 3 | 13.0 | N=23 sum=942.8 med=30.4 T3R=263.8 max_loss=-116.3 | -169.8 | -169.8 |
| 7 | `fixed_sl_120` | 2 | 8.7 | N=23 sum=876.9 med=30.4 T3R=197.9 max_loss=-165.9 | -235.7 | -235.7 |
| 8 | `fixed_sl_80` | 4 | 17.4 | N=23 sum=601.7 med=26.1 T3R=-77.3 max_loss=-108.1 | -510.9 | -510.9 |
| 9 | `hybrid_sl80_danger15m_ret-25_noreclaim` | 5 | 21.7 | N=23 sum=590.1 med=26.1 T3R=-88.9 max_loss=-108.1 | -522.5 | -522.5 |
| 10 | `fixed_sl_40` | 10 | 43.5 | N=21 sum=566.8 med=17.2 T3R=-112.2 max_loss=-53.9 | -545.8 | -545.8 |
| 11 | `fixed_sl_20` | 13 | 56.5 | N=23 sum=548.6 med=-24.0 T3R=-130.4 max_loss=-35.1 | -564.0 | -564.0 |
| 12 | `fixed_sl_30` | 11 | 47.8 | N=23 sum=546.0 med=-31.6 T3R=-133.0 max_loss=-40.2 | -566.6 | -566.6 |
| 13 | `fixed_sl_50` | 9 | 39.1 | N=23 sum=463.2 med=17.2 T3R=-215.8 max_loss=-70.4 | -649.4 | -649.4 |
| 14 | `fixed_sl_60` | 8 | 34.8 | N=23 sum=363.9 med=17.2 T3R=-315.1 max_loss=-107.6 | -748.7 | -748.7 |
| 15 | `hybrid_sl60_danger15m_ret-25_noreclaim` | 9 | 39.1 | N=23 sum=352.3 med=17.2 T3R=-326.7 max_loss=-107.6 | -760.3 | -760.3 |

## Read

- Best T3R-ranked stop overlay: `fixed_sl_150` -> N=23 sum=1081.3 med=37.0 T3R=402.3 max_loss=-175.7.
- For live safety, prefer an exchange-native hard SL even if it is not the top research overlay; process-only danger exits do not protect against outages.
