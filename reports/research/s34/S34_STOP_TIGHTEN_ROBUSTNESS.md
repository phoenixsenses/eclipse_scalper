# S34 Stop-Tighten Robustness

Generated: `2026-06-28T21:14:08.662710+00:00`

Config: `O20_W300_O5_C1`

Baseline: N=22 sum=1120.7 med=39.4 T3R=441.8 max_loss=-144.4

| Rank | Variant | Trigger | Managed | Delta sum | Delta T3R | Delta max loss |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1 | `tight_trigger_sl80_5m_btc_down_continues` | 8/22 | N=22 sum=1162.4 med=39.4 T3R=483.4 max_loss=-102.9 | 41.7 | 41.6 | 41.5 |
| 2 | `tight_trigger_sl80_5m_no_reclaim_btc_down` | 6/22 | N=22 sum=1162.4 med=39.4 T3R=483.4 max_loss=-102.9 | 41.7 | 41.6 | 41.5 |
| 3 | `tight_trigger_sl100_5m_btc_down_continues` | 8/22 | N=22 sum=1155.8 med=39.4 T3R=476.8 max_loss=-109.5 | 35.1 | 35.0 | 34.9 |
| 4 | `tight_trigger_sl100_5m_no_reclaim_btc_down` | 6/22 | N=22 sum=1155.8 med=39.4 T3R=476.8 max_loss=-109.5 | 35.1 | 35.0 | 34.9 |
| 5 | `tight_trigger_sl120_7m_btc_down_continues` | 5/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 6 | `tight_trigger_sl120_7m_no_reclaim_btc_down` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 7 | `tight_trigger_sl120_7m_failed_v` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 8 | `tight_trigger_sl120_7m_weak_first` | 8/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 9 | `tight_trigger_sl80_10m_no_reclaim_btc_down` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 10 | `tight_trigger_sl100_10m_no_reclaim_btc_down` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 11 | `tight_trigger_sl120_10m_no_reclaim_btc_down` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 12 | `tight_trigger_sl80_10m_failed_v` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 13 | `tight_trigger_sl100_10m_failed_v` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 14 | `tight_trigger_sl120_10m_failed_v` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | 0.2 | 0.1 | 0.0 |
| 15 | `tight_trigger_sl120_5m_btc_down_continues` | 8/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | -21.2 | -21.3 | -21.4 |
| 16 | `tight_trigger_sl120_5m_no_reclaim_btc_down` | 6/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | -21.2 | -21.3 | -21.4 |
| 17 | `tight_trigger_sl80_10m_btc_down_continues` | 5/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | -21.2 | -21.3 | -21.4 |
| 18 | `tight_trigger_sl80_10m_weak_first` | 7/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | -21.2 | -21.3 | -21.4 |
| 19 | `tight_trigger_sl100_10m_btc_down_continues` | 5/22 | N=22 sum=1089.6 med=39.4 T3R=410.6 max_loss=-175.7 | -31.1 | -31.2 | -31.3 |
| 20 | `tight_trigger_sl100_10m_weak_first` | 7/22 | N=22 sum=1089.6 med=39.4 T3R=410.6 max_loss=-175.7 | -31.1 | -31.2 | -31.3 |
| 21 | `tight_trigger_sl120_10m_btc_down_continues` | 5/22 | N=22 sum=1084.8 med=39.4 T3R=405.8 max_loss=-180.5 | -35.9 | -36.0 | -36.1 |
| 22 | `tight_trigger_sl120_10m_weak_first` | 7/22 | N=22 sum=1084.8 med=39.4 T3R=405.8 max_loss=-180.5 | -35.9 | -36.0 | -36.1 |
| 23 | `tight_trigger_sl80_5m_failed_v` | 8/22 | N=22 sum=1053.5 med=39.4 T3R=374.5 max_loss=-102.9 | -67.2 | -67.3 | 41.5 |
| 24 | `tight_trigger_sl80_5m_weak_first` | 13/22 | N=22 sum=1053.5 med=39.4 T3R=374.5 max_loss=-102.9 | -67.2 | -67.3 | 41.5 |
| 25 | `tight_trigger_sl80_3m_failed_v` | 8/22 | N=22 sum=1039.5 med=39.4 T3R=360.5 max_loss=-105.7 | -81.2 | -81.3 | 38.7 |
| 26 | `tight_trigger_sl80_3m_weak_first` | 12/22 | N=22 sum=1039.5 med=39.4 T3R=360.5 max_loss=-105.7 | -81.2 | -81.3 | 38.7 |
| 27 | `tight_trigger_sl100_5m_failed_v` | 8/22 | N=22 sum=1025.4 med=39.4 T3R=346.4 max_loss=-113.2 | -95.3 | -95.4 | 31.2 |
| 28 | `tight_trigger_sl100_5m_weak_first` | 13/22 | N=22 sum=1025.4 med=39.4 T3R=346.4 max_loss=-113.2 | -95.3 | -95.4 | 31.2 |
| 29 | `tight_trigger_sl100_3m_failed_v` | 8/22 | N=22 sum=1007.5 med=39.4 T3R=328.5 max_loss=-125.0 | -113.2 | -113.3 | 19.4 |
| 30 | `tight_trigger_sl100_3m_weak_first` | 12/22 | N=22 sum=1007.5 med=39.4 T3R=328.5 max_loss=-125.0 | -113.2 | -113.3 | 19.4 |
| 31 | `tight_entry_sl60_10m_btc_down_continues` | 5/22 | N=22 sum=999.2 med=33.7 T3R=320.2 max_loss=-107.5 | -121.5 | -121.6 | 36.9 |
| 32 | `tight_trigger_sl80_3m_btc_down_continues` | 7/22 | N=22 sum=998.0 med=39.4 T3R=319.0 max_loss=-144.4 | -122.7 | -122.8 | 0.0 |
| 33 | `tight_trigger_sl80_3m_no_reclaim_btc_down` | 6/22 | N=22 sum=998.0 med=39.4 T3R=319.0 max_loss=-144.4 | -122.7 | -122.8 | 0.0 |
| 34 | `tight_entry_sl80_10m_btc_down_continues` | 5/22 | N=22 sum=991.2 med=33.7 T3R=312.2 max_loss=-108.1 | -129.5 | -129.6 | 36.3 |
| 35 | `tight_entry_sl80_10m_weak_first` | 7/22 | N=22 sum=991.2 med=33.7 T3R=312.2 max_loss=-108.1 | -129.5 | -129.6 | 36.3 |
| 36 | `tight_trigger_sl100_3m_btc_down_continues` | 7/22 | N=22 sum=978.7 med=39.4 T3R=299.7 max_loss=-144.4 | -142.0 | -142.1 | 0.0 |
| 37 | `tight_trigger_sl100_3m_no_reclaim_btc_down` | 6/22 | N=22 sum=978.7 med=39.4 T3R=299.7 max_loss=-144.4 | -142.0 | -142.1 | 0.0 |
| 38 | `tight_entry_sl100_5m_btc_down_continues` | 8/22 | N=22 sum=971.0 med=33.7 T3R=292.0 max_loss=-108.0 | -149.7 | -149.8 | 36.4 |
| 39 | `tight_entry_sl100_5m_no_reclaim_btc_down` | 6/22 | N=22 sum=971.0 med=33.7 T3R=292.0 max_loss=-108.0 | -149.7 | -149.8 | 36.4 |
| 40 | `tight_entry_sl100_10m_btc_down_continues` | 5/22 | N=22 sum=971.0 med=33.7 T3R=292.0 max_loss=-108.0 | -149.7 | -149.8 | 36.4 |

## Read

- Positive neighborhood cells: `14` / `128`.
- Positive delays: `[5, 7, 10]`; stops: `[80, 100, 120]`; conditions: `['btc_down_continues', 'failed_v', 'no_reclaim_btc_down', 'weak_first']`.
