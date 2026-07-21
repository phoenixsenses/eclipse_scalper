# S34 Stress Reaction Deep Tests

Generated: `2026-06-29T13:31:19.383642+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

## BTC Stress Sweep

| Filter | N | Fixed 15m Reverse | TP150/SL30/15m | TP150/SL50/15m |
| --- | ---: | --- | --- | --- |
| btc4h_lt_50 | 209 | N=209 sum=1769.7 med=-1.2 T3R=1023.6 tail150=1 maxLoss=-150.8 | N=209 sum=1740.1 med=-6.6 T3R=1305.1 tail150=0 maxLoss=-35.0 | N=209 sum=1693.3 med=-1.2 T3R=1258.3 tail150=0 maxLoss=-55.0 |
| btc4h_lt_75 | 190 | N=190 sum=1757.1 med=-1.7 T3R=1011.0 tail150=1 maxLoss=-150.8 | N=190 sum=1742.1 med=-9.0 T3R=1307.1 tail150=0 maxLoss=-35.0 | N=190 sum=1684.7 med=-1.9 T3R=1249.7 tail150=0 maxLoss=-55.0 |
| btc4h_lt_100 | 129 | N=129 sum=1820.1 med=1.0 T3R=1074.0 tail150=1 maxLoss=-150.8 | N=129 sum=1868.7 med=-13.2 T3R=1433.7 tail150=0 maxLoss=-35.0 | N=129 sum=1686.0 med=0.2 T3R=1251.0 tail150=0 maxLoss=-55.0 |
| btc4h_lt_150 | 81 | N=81 sum=658.7 med=3.2 T3R=111.2 tail150=0 maxLoss=-102.0 | N=81 sum=793.0 med=-35.0 T3R=358.0 tail150=0 maxLoss=-35.0 | N=81 sum=724.5 med=3.5 T3R=289.5 tail150=0 maxLoss=-55.0 |
| btc4h_lt_200 | 70 | N=70 sum=436.0 med=1.5 T3R=-111.5 tail150=0 maxLoss=-102.0 | N=70 sum=535.0 med=-35.0 T3R=100.0 tail150=0 maxLoss=-35.0 | N=70 sum=484.3 med=-0.0 T3R=49.3 tail150=0 maxLoss=-55.0 |
| btc4h_lt_250 | 41 | N=41 sum=783.5 med=25.9 T3R=277.4 tail150=0 maxLoss=-102.0 | N=41 sum=910.6 med=-1.1 T3R=475.6 tail150=0 maxLoss=-35.0 | N=41 sum=1013.0 med=24.9 T3R=578.0 tail150=0 maxLoss=-55.0 |

## VDepth Cleaner (score>=3 & btc4h<-75)

| Filter | N | Fixed 15m Reverse | TP150/SL30/15m |
| --- | ---: | --- | --- |
| all | 190 | N=190 sum=1757.1 med=-1.7 T3R=1011.0 tail150=1 maxLoss=-150.8 | N=190 sum=1742.1 med=-9.0 T3R=1307.1 tail150=0 maxLoss=-35.0 |
| v_lt_25 | 136 | N=136 sum=1045.6 med=-1.7 T3R=434.0 tail150=0 maxLoss=-102.0 | N=136 sum=939.1 med=-7.7 T3R=504.1 tail150=0 maxLoss=-35.0 |
| v_25_40 | 39 | N=39 sum=743.8 med=-0.4 T3R=205.8 tail150=0 maxLoss=-113.1 | N=39 sum=464.9 med=-15.9 T3R=77.9 tail150=0 maxLoss=-35.0 |
| v_lt_40 | 175 | N=175 sum=1789.4 med=-1.2 T3R=1088.5 tail150=0 maxLoss=-113.1 | N=175 sum=1404.0 med=-8.8 T3R=969.0 tail150=0 maxLoss=-35.0 |
| v_lt_50 | 184 | N=184 sum=1953.9 med=-1.2 T3R=1207.7 tail150=0 maxLoss=-127.6 | N=184 sum=1844.0 med=-8.5 T3R=1409.0 tail150=0 maxLoss=-35.0 |
| v_lt_60 | 186 | N=186 sum=1895.8 med=-1.7 T3R=1149.7 tail150=0 maxLoss=-127.6 | N=186 sum=1788.2 med=-9.0 T3R=1353.2 tail150=0 maxLoss=-35.0 |
| exclude_danger_high | 175 | N=175 sum=1789.4 med=-1.2 T3R=1088.5 tail150=0 maxLoss=-113.1 | N=175 sum=1404.0 med=-8.8 T3R=969.0 tail150=0 maxLoss=-35.0 |

## Exit Sweep Top 20 (score>=3 & btc4h<-75)

N: `190`

| Horizon | TP | SL | Summary | Exits | Median exit sec |
| --- | ---: | ---: | --- | --- | ---: |
| 20m | 200.0 | 40.0 | N=190 sum=2844.6 med=-5.3 T3R=2259.6 tail150=0 maxLoss=-45.0 | `{'SL': 60, 'TIME': 116, 'TP': 14}` | 1198.0 |
| 20m | 200.0 | 25.0 | N=190 sum=2757.8 med=-30.0 T3R=2172.8 tail150=0 maxLoss=-30.0 | `{'SL': 96, 'TIME': 80, 'TP': 14}` | 668.0 |
| 20m | 200.0 | 30.0 | N=190 sum=2609.2 med=-12.2 T3R=2024.2 tail150=0 maxLoss=-35.0 | `{'SL': 84, 'TIME': 92, 'TP': 14}` | 1038.5 |
| 15m | 200.0 | 40.0 | N=190 sum=2534.0 med=-2.7 T3R=1949.0 tail150=0 maxLoss=-45.0 | `{'SL': 55, 'TIME': 123, 'TP': 12}` | 898.0 |
| 20m | 200.0 | 50.0 | N=190 sum=2512.5 med=-5.1 T3R=1927.5 tail150=0 maxLoss=-55.0 | `{'SL': 50, 'TIME': 126, 'TP': 14}` | 1198.0 |
| 15m | 200.0 | 30.0 | N=190 sum=2356.4 med=-9.0 T3R=1771.4 tail150=0 maxLoss=-35.0 | `{'SL': 81, 'TIME': 97, 'TP': 12}` | 898.0 |
| 15m | 200.0 | 50.0 | N=190 sum=2299.0 med=-1.9 T3R=1714.0 tail150=0 maxLoss=-55.0 | `{'SL': 46, 'TIME': 132, 'TP': 12}` | 898.0 |
| 15m | 200.0 | 25.0 | N=190 sum=2283.8 med=-20.5 T3R=1698.8 tail150=0 maxLoss=-30.0 | `{'SL': 94, 'TIME': 84, 'TP': 12}` | 668.0 |
| 20m | 150.0 | 40.0 | N=190 sum=2100.4 med=-5.3 T3R=1665.4 tail150=0 maxLoss=-45.0 | `{'SL': 60, 'TIME': 109, 'TP': 21}` | 1198.0 |
| 20m | 100.0 | 40.0 | N=190 sum=1880.8 med=-5.1 T3R=1595.8 tail150=0 maxLoss=-45.0 | `{'SL': 59, 'TIME': 92, 'TP': 39}` | 1040.0 |
| 20m | 150.0 | 25.0 | N=190 sum=2013.7 med=-30.0 T3R=1578.7 tail150=0 maxLoss=-30.0 | `{'SL': 96, 'TIME': 73, 'TP': 21}` | 642.5 |
| 20m | 125.0 | 40.0 | N=190 sum=1851.6 med=-5.3 T3R=1491.6 tail150=0 maxLoss=-45.0 | `{'SL': 60, 'TIME': 103, 'TP': 27}` | 1198.0 |
| 15m | 150.0 | 40.0 | N=190 sum=1919.7 med=-2.7 T3R=1484.7 tail150=0 maxLoss=-45.0 | `{'SL': 55, 'TIME': 117, 'TP': 18}` | 898.0 |
| 20m | 200.0 | 20.0 | N=190 sum=2050.8 med=-25.0 T3R=1465.8 tail150=0 maxLoss=-25.0 | `{'SL': 110, 'TIME': 70, 'TP': 10}` | 403.0 |
| 20m | 150.0 | 30.0 | N=190 sum=1865.1 med=-12.2 T3R=1430.1 tail150=0 maxLoss=-35.0 | `{'SL': 84, 'TIME': 85, 'TP': 21}` | 825.0 |
| 20m | 125.0 | 25.0 | N=190 sum=1764.9 med=-30.0 T3R=1404.9 tail150=0 maxLoss=-30.0 | `{'SL': 96, 'TIME': 67, 'TP': 27}` | 580.0 |
| 20m | 100.0 | 25.0 | N=190 sum=1632.1 med=-26.3 T3R=1347.1 tail150=0 maxLoss=-30.0 | `{'SL': 95, 'TIME': 59, 'TP': 36}` | 528.5 |
| 20m | 150.0 | 50.0 | N=190 sum=1768.4 med=-5.1 T3R=1333.4 tail150=0 maxLoss=-55.0 | `{'SL': 50, 'TIME': 119, 'TP': 21}` | 1198.0 |
| 15m | 150.0 | 30.0 | N=190 sum=1742.1 med=-9.0 T3R=1307.1 tail150=0 maxLoss=-35.0 | `{'SL': 81, 'TIME': 91, 'TP': 18}` | 825.0 |
| 20m | 100.0 | 50.0 | N=190 sum=1558.8 med=-4.0 T3R=1273.8 tail150=0 maxLoss=-55.0 | `{'SL': 49, 'TIME': 102, 'TP': 39}` | 1198.0 |

## MFE / MAE Timing (score>=3 & btc4h<-75, reverse 15m)

- `all`: `{'n': 190, 'mfe_median': 36.8, 'mae_median': -24.2, 'mfe_sec_median': 310.0, 'mae_sec_median': 406.5, 'tp150_hit_n': 18, 'tp150_sec_median': 588.5, 'sl30_hit_n': 81, 'sl30_sec_median': 306.0, 'final': {'n': 190, 'sum_bps': 1771.8, 'mean_bps': 9.3, 'median_bps': -1.9, 'win_rate': 0.479, 'max_loss_bps': -150.5, 'tail_lte_minus100_n': 4, 'tail_lte_minus150_n': 1, 'tail_lte_minus300_n': 0, 't3r_bps': 1026.5}}`
- `winners`: `{'n': 91, 'mfe_median': 72.7, 'mae_median': -12.0, 'mfe_sec_median': 704.0, 'mae_sec_median': 116.0, 'tp150_hit_n': 18, 'tp150_sec_median': 588.5, 'sl30_hit_n': 9, 'sl30_sec_median': 195.0, 'final': {'n': 91, 'sum_bps': 5231.2, 'mean_bps': 57.5, 'median_bps': 37.4, 'win_rate': 1.0, 'max_loss_bps': 0.0, 'tail_lte_minus100_n': 0, 'tail_lte_minus150_n': 0, 'tail_lte_minus300_n': 0, 't3r_bps': 4486.0}}`
- `losers`: `{'n': 26, 'mfe_median': 4.1, 'mae_median': -71.7, 'mfe_sec_median': 13.0, 'mae_sec_median': 842.5, 'tp150_hit_n': 0, 'tp150_sec_median': None, 'sl30_hit_n': 26, 'sl30_sec_median': 284.0, 'final': {'n': 26, 'sum_bps': -1868.6, 'mean_bps': -71.9, 'median_bps': -58.9, 'win_rate': 0.0, 'max_loss_bps': -150.5, 'tail_lte_minus100_n': 4, 'tail_lte_minus150_n': 1, 'tail_lte_minus300_n': 0, 't3r_bps': -1711.9}}`

## Event Chain Filters

| Filter | N | Fixed 15m Reverse | TP150/SL30/15m |
| --- | ---: | --- | --- |
| all | 190 | N=190 sum=1757.1 med=-1.7 T3R=1011.0 tail150=1 maxLoss=-150.8 | N=190 sum=1742.1 med=-9.0 T3R=1307.1 tail150=0 maxLoss=-35.0 |
| prior_15m_0 | 107 | N=107 sum=835.1 med=-4.5 T3R=250.3 tail150=0 maxLoss=-113.1 | N=107 sum=795.6 med=-7.9 T3R=360.6 tail150=0 maxLoss=-35.0 |
| prior_15m_ge1 | 83 | N=83 sum=922.1 med=-0.2 T3R=240.1 tail150=1 maxLoss=-150.8 | N=83 sum=946.4 med=-17.9 T3R=511.4 tail150=0 maxLoss=-35.0 |
| prior_1h_ge3 | 116 | N=116 sum=784.0 med=1.5 T3R=270.1 tail150=1 maxLoss=-150.8 | N=116 sum=941.3 med=-17.1 T3R=506.3 tail150=0 maxLoss=-35.0 |
| near_15m_thresholds_ge2 | 175 | N=175 sum=2112.5 med=-0.2 T3R=1366.3 tail150=1 maxLoss=-150.8 | N=175 sum=2045.7 med=-7.9 T3R=1610.7 tail150=0 maxLoss=-35.0 |
| near_15m_thresholds_ge3 | 137 | N=137 sum=2367.2 med=10.5 T3R=1621.0 tail150=1 maxLoss=-150.8 | N=137 sum=2127.0 med=0.0 T3R=1692.0 tail150=0 maxLoss=-35.0 |

## v0.2 Guard

| Filter | N | Normal 15m | Normal 2h |
| --- | ---: | --- | --- |
| all_v02 | 31 | N=31 sum=-180.0 med=0.8 T3R=-414.1 tail150=0 maxLoss=-106.0 | N=31 sum=697.6 med=12.0 T3R=52.6 tail150=2 maxLoss=-194.4 |
| tail_low | 11 | N=11 sum=-80.4 med=7.4 T3R=-185.7 tail150=0 maxLoss=-86.7 | N=11 sum=959.5 med=50.2 T3R=314.4 tail150=0 maxLoss=-12.6 |
| tail_high_unknown | 20 | N=20 sum=-99.6 med=-2.0 T3R=-294.3 tail150=0 maxLoss=-106.0 | N=20 sum=-261.9 med=0.0 T3R=-469.5 tail150=2 maxLoss=-194.4 |
| bid_ok | 11 | N=11 sum=-80.4 med=7.4 T3R=-185.7 tail150=0 maxLoss=-86.7 | N=11 sum=959.5 med=50.2 T3R=314.4 tail150=0 maxLoss=-12.6 |
| bid_thin | 20 | N=20 sum=-99.6 med=-2.0 T3R=-294.3 tail150=0 maxLoss=-106.0 | N=20 sum=-261.9 med=0.0 T3R=-469.5 tail150=2 maxLoss=-194.4 |
| bid_heavy | 1 | N=1 sum=19.5 med=19.5 T3R=19.5 tail150=0 maxLoss=19.5 | N=1 sum=266.9 med=266.9 T3R=266.9 tail150=0 maxLoss=266.9 |
| stress_ge2 | 17 | N=17 sum=-108.8 med=3.2 T3R=-212.1 tail150=0 maxLoss=-65.2 | N=17 sum=520.7 med=11.7 T3R=-124.3 tail150=1 maxLoss=-194.4 |
| stress_ge3 | 11 | N=11 sum=-101.5 med=5.6 T3R=-189.7 tail150=0 maxLoss=-65.2 | N=11 sum=444.3 med=0.9 T3R=-193.3 tail150=1 maxLoss=-194.4 |

## Fee Sensitivity (score>=3 & btc4h<-75, reverse TP150/SL30/15m)

| Fee | Summary | Exits |
| --- | --- | --- |
| fee_0bps | N=190 sum=2692.1 med=-4.0 T3R=2242.1 tail150=0 maxLoss=-30.0 | `{'SL': 81, 'TIME': 91, 'TP': 18}` |
| fee_2.5bps | N=190 sum=2217.1 med=-6.5 T3R=1774.6 tail150=0 maxLoss=-32.5 | `{'SL': 81, 'TIME': 91, 'TP': 18}` |
| fee_5bps | N=190 sum=1742.1 med=-9.0 T3R=1307.1 tail150=0 maxLoss=-35.0 | `{'SL': 81, 'TIME': 91, 'TP': 18}` |
| fee_8bps | N=190 sum=1172.1 med=-12.0 T3R=746.1 tail150=0 maxLoss=-38.0 | `{'SL': 81, 'TIME': 91, 'TP': 18}` |

## Big Winner / Loser Anatomy (score>=3 & btc4h<-75, reverse 15m)

Summary: N=190 sum=1757.1 med=-1.7 T3R=1011.0 tail150=1 maxLoss=-150.8
Winner profile: `{'n': 90, 'avg_threshold': 113333.3, 'avg_vdepth': 19.5, 'avg_prior4h': -218.6, 'avg_eth1h': -117.2, 'avg_btc4h': -194.9, 'avg_bid_depth': 148285.6, 'avg_book_imbalance': -0.2, 'tag_mix': {'SIZE_34X_FRAGILE': 87, 'TAIL_HIGH_OR_UNKNOWN': 87, 'RISK_OFF_REBOUND': 84, 'VDEPTH_DANGER_LOW': 69, 'BID_DEPTH_THIN': 57, 'EXIT_4H_ACTUAL_BETTER': 56, 'EXIT_2H_ACTUAL_BETTER': 34, 'BID_DEPTH_OK': 33}, 'threshold_mix': {'thr100000': 33, 'thr50000': 30, 'thr200000': 27}}`
Loser profile: `{'n': 26, 'avg_threshold': 117307.7, 'avg_vdepth': 21.4, 'avg_prior4h': -232.6, 'avg_eth1h': -110.2, 'avg_btc4h': -189.1, 'avg_bid_depth': 148591.5, 'avg_book_imbalance': 0.1, 'tag_mix': {'RISK_OFF_REBOUND': 26, 'SIZE_34X_FRAGILE': 24, 'TAIL_HIGH_OR_UNKNOWN': 24, 'VDEPTH_DANGER_LOW': 20, 'EXIT_4H_ACTUAL_BETTER': 17, 'BID_DEPTH_OK': 15, 'BID_DEPTH_THIN': 11, 'EXIT_2H_ACTUAL_BETTER': 9}, 'threshold_mix': {'thr200000': 9, 'thr50000': 9, 'thr100000': 8}}`
Tail profile: `{'n': 1, 'avg_threshold': 200000.0, 'avg_vdepth': 68.1, 'avg_prior4h': -119.4, 'avg_eth1h': -98.6, 'avg_btc4h': -115.8, 'avg_bid_depth': 95469.2, 'avg_book_imbalance': -0.6, 'tag_mix': {'BID_DEPTH_THIN': 1, 'EXIT_4H_ACTUAL_BETTER': 1, 'RISK_OFF_REBOUND': 1, 'SIZE_34X_FRAGILE': 1, 'TAIL_HIGH_OR_UNKNOWN': 1, 'VDEPTH_DANGER_HIGH': 1}, 'threshold_mix': {'thr200000': 1}}`

Worst 10:
- `{'event_id': 'ETHUSDT_SELL_200000_1782699234143', 'fold': 5, 'signal_utc': '2026-06-29T02:13:54.143000+00:00', 'value_bps': -150.8, 'stress_score': 3, 'threshold': 200000.0, 'vdepth': 68.1, 'prior4h': -119.4, 'eth1h': -98.6, 'btc4h': -115.8, 'bid_depth': 95469.2, 'book_imbalance': -0.632, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_HIGH', 'BID_DEPTH_THIN', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782699195732', 'fold': 5, 'signal_utc': '2026-06-29T02:13:15.732000+00:00', 'value_bps': -127.6, 'stress_score': 4, 'threshold': 100000.0, 'vdepth': 45.2, 'prior4h': -97.9, 'eth1h': -72.2, 'btc4h': -103.4, 'bid_depth': 137419.7, 'book_imbalance': 0.024, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_HIGH', 'BID_DEPTH_OK', 'BID_DEPTH_CORE', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782699186530', 'fold': 5, 'signal_utc': '2026-06-29T02:13:06.530000+00:00', 'value_bps': -113.1, 'stress_score': 4, 'threshold': 50000.0, 'vdepth': 34.8, 'prior4h': -85.6, 'eth1h': -68.7, 'btc4h': -91.7, 'bid_depth': 278379.4, 'book_imbalance': 0.186, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_CORE', 'BID_DEPTH_OK', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_LOW_CONTEXT', 'SIZE_15X_STABLE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782310759584', 'fold': 5, 'signal_utc': '2026-06-24T14:19:19.584000+00:00', 'value_bps': -102.0, 'stress_score': 3, 'threshold': 200000.0, 'vdepth': 18.7, 'prior4h': -148.5, 'eth1h': -119.5, 'btc4h': -255.1, 'bid_depth': 166592.1, 'book_imbalance': 0.569, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'BID_DEPTH_CORE', 'EXIT_2H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782324164473', 'fold': 5, 'signal_utc': '2026-06-24T18:02:44.473000+00:00', 'value_bps': -97.4, 'stress_score': 4, 'threshold': 50000.0, 'vdepth': 34.0, 'prior4h': -520.8, 'eth1h': -252.3, 'btc4h': -324.1, 'bid_depth': 221042.7, 'book_imbalance': 0.233, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_CORE', 'BID_DEPTH_OK', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_LOW_CONTEXT', 'SIZE_15X_STABLE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782310755533', 'fold': 5, 'signal_utc': '2026-06-24T14:19:15.533000+00:00', 'value_bps': -95.7, 'stress_score': 4, 'threshold': 100000.0, 'vdepth': 16.6, 'prior4h': -151.3, 'eth1h': -114.9, 'btc4h': -249.6, 'bid_depth': 181059.6, 'book_imbalance': -0.063, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'BID_DEPTH_CORE', 'EXIT_2H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782324211658', 'fold': 5, 'signal_utc': '2026-06-24T18:03:31.658000+00:00', 'value_bps': -69.9, 'stress_score': 4, 'threshold': 100000.0, 'vdepth': 22.2, 'prior4h': -511.4, 'eth1h': -235.9, 'btc4h': -318.6, 'bid_depth': 149889.4, 'book_imbalance': 0.007, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'BID_DEPTH_CORE', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782390445910', 'fold': 5, 'signal_utc': '2026-06-25T12:27:25.910000+00:00', 'value_bps': -69.1, 'stress_score': 4, 'threshold': 50000.0, 'vdepth': 9.3, 'prior4h': -154.7, 'eth1h': -26.7, 'btc4h': -102.1, 'bid_depth': 217141.9, 'book_imbalance': 0.662, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782316774193', 'fold': 5, 'signal_utc': '2026-06-24T15:59:34.193000+00:00', 'value_bps': -68.1, 'stress_score': 4, 'threshold': 200000.0, 'vdepth': 10.9, 'prior4h': -361.5, 'eth1h': -173.0, 'btc4h': -416.8, 'bid_depth': 11765.3, 'book_imbalance': -0.926, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1781620306905', 'fold': 4, 'signal_utc': '2026-06-16T14:31:46.905000+00:00', 'value_bps': -67.1, 'stress_score': 3, 'threshold': 200000.0, 'vdepth': 9.8, 'prior4h': -148.3, 'eth1h': -200.2, 'btc4h': -171.1, 'bid_depth': 25214.3, 'book_imbalance': -0.709, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
Best 10:
- `{'event_id': 'ETHUSDT_SELL_50000_1782394272242', 'fold': 5, 'signal_utc': '2026-06-25T13:31:12.242000+00:00', 'value_bps': 261.1, 'stress_score': 3, 'threshold': 50000.0, 'vdepth': 24.2, 'prior4h': -122.9, 'eth1h': -51.7, 'btc4h': -100.5, 'bid_depth': 31419.0, 'book_imbalance': -0.741, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782394284495', 'fold': 5, 'signal_utc': '2026-06-25T13:31:24.495000+00:00', 'value_bps': 242.9, 'stress_score': 4, 'threshold': 100000.0, 'vdepth': 39.4, 'prior4h': -132.2, 'eth1h': -60.8, 'btc4h': -109.2, 'bid_depth': 325274.6, 'book_imbalance': 0.807, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_CORE', 'BID_DEPTH_OK', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_LOW_CONTEXT', 'SIZE_15X_STABLE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782394305283', 'fold': 5, 'signal_utc': '2026-06-25T13:31:45.283000+00:00', 'value_bps': 242.1, 'stress_score': 3, 'threshold': 200000.0, 'vdepth': 45.6, 'prior4h': -135.4, 'eth1h': -63.0, 'btc4h': -112.0, 'bid_depth': 4368.9, 'book_imbalance': -0.97, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_HIGH', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782202432090', 'fold': 5, 'signal_utc': '2026-06-23T08:13:52.090000+00:00', 'value_bps': 196.9, 'stress_score': 3, 'threshold': 50000.0, 'vdepth': 28.8, 'prior4h': -306.8, 'eth1h': -99.6, 'btc4h': -202.5, 'bid_depth': 229234.2, 'book_imbalance': 0.592, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_CORE', 'BID_DEPTH_OK', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_LOW_CONTEXT', 'SIZE_15X_STABLE']}`
- `{'event_id': 'ETHUSDT_SELL_50000_1782319123478', 'fold': 5, 'signal_utc': '2026-06-24T16:38:43.478000+00:00', 'value_bps': 177.1, 'stress_score': 4, 'threshold': 50000.0, 'vdepth': 16.6, 'prior4h': -311.8, 'eth1h': -54.2, 'btc4h': -406.7, 'bid_depth': 300442.8, 'book_imbalance': 0.272, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782319424366', 'fold': 5, 'signal_utc': '2026-06-24T16:43:44.366000+00:00', 'value_bps': 173.5, 'stress_score': 4, 'threshold': 100000.0, 'vdepth': 20.2, 'prior4h': -337.3, 'eth1h': -54.2, 'btc4h': -425.3, 'bid_depth': 94814.2, 'book_imbalance': -0.642, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782202514090', 'fold': 5, 'signal_utc': '2026-06-23T08:15:14.090000+00:00', 'value_bps': 163.3, 'stress_score': 3, 'threshold': 200000.0, 'vdepth': 7.9, 'prior4h': -335.2, 'eth1h': -128.4, 'btc4h': -219.9, 'bid_depth': 2655.4, 'book_imbalance': -0.984, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782319601345', 'fold': 5, 'signal_utc': '2026-06-24T16:46:41.345000+00:00', 'value_bps': 155.6, 'stress_score': 4, 'threshold': 200000.0, 'vdepth': 18.4, 'prior4h': -341.9, 'eth1h': -21.8, 'btc4h': -420.9, 'bid_depth': 35378.2, 'book_imbalance': -0.803, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_4H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_200000_1782194864406', 'fold': 5, 'signal_utc': '2026-06-23T06:07:44.406000+00:00', 'value_bps': 151.8, 'stress_score': 4, 'threshold': 200000.0, 'vdepth': 11.8, 'prior4h': -125.4, 'eth1h': -63.7, 'btc4h': -130.2, 'bid_depth': 260942.6, 'book_imbalance': 0.326, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_OK', 'TAIL_REALIZED', 'EXIT_2H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
- `{'event_id': 'ETHUSDT_SELL_100000_1782194855206', 'fold': 5, 'signal_utc': '2026-06-23T06:07:35.206000+00:00', 'value_bps': 146.6, 'stress_score': 3, 'threshold': 100000.0, 'vdepth': 9.7, 'prior4h': -123.8, 'eth1h': -61.3, 'btc4h': -130.9, 'bid_depth': 70492.1, 'book_imbalance': -0.805, 'tags': ['RISK_OFF_REBOUND', 'VDEPTH_DANGER_LOW', 'BID_DEPTH_THIN', 'TAIL_REALIZED', 'EXIT_2H_ACTUAL_BETTER', 'TAIL_HIGH_OR_UNKNOWN', 'SIZE_34X_FRAGILE']}`
