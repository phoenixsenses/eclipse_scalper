# S34 V Engine Position Management

Generated: `2026-06-28T21:01:58.692509+00:00`

Config: `O20_W300_O5_C1`

Research-only. Tests post-fill management overlays on current live route.

Baseline: N=22 sum=1120.7 med=39.4 T3R=441.8 max_loss=-144.4

| Rank | Variant | Type | Trigger | Managed | Triggered original | Triggered managed | Delta sum | Delta T3R |
| ---: | --- | --- | ---: | --- | --- | --- | ---: | ---: |
| 1 | `tight_trigger_sl80_5m_no_reclaim_btc_down` | `tighten_stop` | 6/22 | N=22 sum=1162.4 med=39.4 T3R=483.4 max_loss=-102.9 | N=6 sum=170.7 med=56.1 T3R=-119.8 max_loss=-144.4 | N=6 sum=212.3 med=56.1 T3R=-78.3 max_loss=-102.9 | 41.7 | 41.6 |
| 2 | `tight_trigger_sl80_5m_btc_down_continues` | `tighten_stop` | 8/22 | N=22 sum=1162.4 med=39.4 T3R=483.4 max_loss=-102.9 | N=8 sum=219.8 med=48.6 T3R=-70.8 max_loss=-144.4 | N=8 sum=261.3 med=48.6 T3R=-29.3 max_loss=-102.9 | 41.7 | 41.6 |
| 3 | `tight_trigger_sl100_5m_no_reclaim_btc_down` | `tighten_stop` | 6/22 | N=22 sum=1155.8 med=39.4 T3R=476.8 max_loss=-109.5 | N=6 sum=170.7 med=56.1 T3R=-119.8 max_loss=-144.4 | N=6 sum=205.7 med=56.1 T3R=-84.9 max_loss=-109.5 | 35.1 | 35.0 |
| 4 | `tight_trigger_sl100_5m_btc_down_continues` | `tighten_stop` | 8/22 | N=22 sum=1155.8 med=39.4 T3R=476.8 max_loss=-109.5 | N=8 sum=219.8 med=48.6 T3R=-70.8 max_loss=-144.4 | N=8 sum=254.7 med=48.6 T3R=-35.9 max_loss=-109.5 | 35.1 | 35.0 |
| 5 | `tight_trigger_sl100_15m_no_reclaim_btc_down` | `tighten_stop` | 2/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=53.1 med=26.5 T3R=53.1 max_loss=-28.0 | 0.2 | 0.1 |
| 6 | `tight_trigger_sl40_30m_no_reclaim_btc_down` | `tighten_stop` | 2/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=53.1 med=26.5 T3R=53.1 max_loss=-28.0 | 0.2 | 0.1 |
| 7 | `tight_trigger_sl60_30m_no_reclaim_btc_down` | `tighten_stop` | 2/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=53.1 med=26.5 T3R=53.1 max_loss=-28.0 | 0.2 | 0.1 |
| 8 | `tight_trigger_sl80_30m_no_reclaim_btc_down` | `tighten_stop` | 2/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=53.1 med=26.5 T3R=53.1 max_loss=-28.0 | 0.2 | 0.1 |
| 9 | `tight_trigger_sl100_30m_no_reclaim_btc_down` | `tighten_stop` | 2/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=53.1 med=26.5 T3R=53.1 max_loss=-28.0 | 0.2 | 0.1 |
| 10 | `tight_trigger_sl80_10m_no_reclaim_btc_down` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=203.0 med=81.1 T3R=203.0 max_loss=-28.0 | N=3 sum=203.0 med=81.1 T3R=203.0 max_loss=-28.0 | 0.2 | 0.1 |
| 11 | `tight_trigger_sl100_10m_no_reclaim_btc_down` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=203.0 med=81.1 T3R=203.0 max_loss=-28.0 | N=3 sum=203.0 med=81.1 T3R=203.0 max_loss=-28.0 | 0.2 | 0.1 |
| 12 | `tight_trigger_sl40_30m_btc_down_continues` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | 0.2 | 0.1 |
| 13 | `tight_trigger_sl60_30m_btc_down_continues` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | 0.2 | 0.1 |
| 14 | `tight_trigger_sl80_30m_btc_down_continues` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | 0.2 | 0.1 |
| 15 | `tight_trigger_sl100_30m_btc_down_continues` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | 0.2 | 0.1 |
| 16 | `tight_trigger_sl40_30m_failed_v` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=17.3 med=-28.0 T3R=17.3 max_loss=-35.9 | N=3 sum=17.2 med=-28.0 T3R=17.2 max_loss=-35.9 | 0.2 | 0.1 |
| 17 | `tight_trigger_sl60_30m_failed_v` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=17.3 med=-28.0 T3R=17.3 max_loss=-35.9 | N=3 sum=17.2 med=-28.0 T3R=17.2 max_loss=-35.9 | 0.2 | 0.1 |
| 18 | `tight_trigger_sl80_30m_failed_v` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=17.3 med=-28.0 T3R=17.3 max_loss=-35.9 | N=3 sum=17.2 med=-28.0 T3R=17.2 max_loss=-35.9 | 0.2 | 0.1 |
| 19 | `tight_trigger_sl100_30m_failed_v` | `tighten_stop` | 3/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=3 sum=17.3 med=-28.0 T3R=17.3 max_loss=-35.9 | N=3 sum=17.2 med=-28.0 T3R=17.2 max_loss=-35.9 | 0.2 | 0.1 |
| 20 | `tight_trigger_sl80_10m_failed_v` | `tighten_stop` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=4 sum=167.1 med=26.6 T3R=-35.9 max_loss=-35.9 | N=4 sum=167.1 med=26.5 T3R=-35.9 max_loss=-35.9 | 0.2 | 0.1 |
| 21 | `tight_trigger_sl100_10m_failed_v` | `tighten_stop` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=4 sum=167.1 med=26.6 T3R=-35.9 max_loss=-35.9 | N=4 sum=167.1 med=26.5 T3R=-35.9 max_loss=-35.9 | 0.2 | 0.1 |
| 22 | `tight_trigger_sl100_15m_btc_down_continues` | `tighten_stop` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=4 sum=-86.9 med=-11.8 T3R=-144.4 max_loss=-144.4 | N=4 sum=-86.9 med=-11.8 T3R=-144.4 max_loss=-144.4 | 0.2 | 0.1 |
| 23 | `tight_trigger_sl100_15m_failed_v` | `tighten_stop` | 4/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=4 sum=167.1 med=26.6 T3R=-35.9 max_loss=-35.9 | N=4 sum=167.1 med=26.5 T3R=-35.9 max_loss=-35.9 | 0.2 | 0.1 |
| 24 | `tight_trigger_sl40_30m_weak_first` | `tighten_stop` | 6/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | 0.2 | 0.1 |
| 25 | `tight_trigger_sl60_30m_weak_first` | `tighten_stop` | 6/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | 0.2 | 0.1 |
| 26 | `tight_trigger_sl80_30m_weak_first` | `tighten_stop` | 6/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | 0.2 | 0.1 |
| 27 | `tight_trigger_sl100_30m_weak_first` | `tighten_stop` | 6/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | N=6 sum=-47.6 med=-0.5 T3R=-208.3 max_loss=-144.4 | 0.2 | 0.1 |
| 28 | `tight_trigger_sl100_15m_weak_first` | `tighten_stop` | 8/22 | N=22 sum=1120.9 med=39.4 T3R=441.9 max_loss=-144.4 | N=8 sum=109.2 med=20.1 T3R=-168.1 max_loss=-144.4 | N=8 sum=109.3 med=20.1 T3R=-168.0 max_loss=-144.4 | 0.2 | 0.1 |
| 29 | `tight_trigger_sl80_10m_btc_down_continues` | `tighten_stop` | 5/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | N=5 sum=63.0 med=4.4 T3R=-172.5 max_loss=-144.4 | N=5 sum=41.6 med=4.4 T3R=-193.8 max_loss=-165.8 | -21.2 | -21.3 |
| 30 | `tight_trigger_sl80_10m_weak_first` | `tighten_stop` | 7/22 | N=22 sum=1099.5 med=39.4 T3R=420.5 max_loss=-165.8 | N=7 sum=82.2 med=13.3 T3R=-195.0 max_loss=-144.4 | N=7 sum=60.9 med=13.3 T3R=-216.4 max_loss=-165.8 | -21.2 | -21.3 |
| 31 | `tight_trigger_sl100_10m_btc_down_continues` | `tighten_stop` | 5/22 | N=22 sum=1089.6 med=39.4 T3R=410.6 max_loss=-175.7 | N=5 sum=63.0 med=4.4 T3R=-172.5 max_loss=-144.4 | N=5 sum=31.7 med=4.4 T3R=-203.7 max_loss=-175.7 | -31.1 | -31.2 |
| 32 | `tight_trigger_sl100_10m_weak_first` | `tighten_stop` | 7/22 | N=22 sum=1089.6 med=39.4 T3R=410.6 max_loss=-175.7 | N=7 sum=82.2 med=13.3 T3R=-195.0 max_loss=-144.4 | N=7 sum=51.0 med=13.3 T3R=-226.3 max_loss=-175.7 | -31.1 | -31.2 |
| 33 | `partial25_15m_no_reclaim_btc_down` | `partial_reduce` | 2/22 | N=22 sum=1083.0 med=39.4 T3R=404.0 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=15.2 med=7.6 T3R=15.2 max_loss=-30.9 | -37.7 | -37.8 |
| 34 | `partial25_15m_btc_down_continues` | `partial_reduce` | 4/22 | N=22 sum=1075.7 med=39.4 T3R=396.7 max_loss=-152.4 | N=4 sum=-86.9 med=-11.8 T3R=-144.4 max_loss=-144.4 | N=4 sum=-132.1 med=-12.9 T3R=-152.4 max_loss=-152.4 | -45.0 | -45.1 |
| 35 | `partial25_30m_failed_v` | `partial_reduce` | 3/22 | N=22 sum=1064.3 med=33.9 T3R=385.3 max_loss=-144.4 | N=3 sum=17.3 med=-28.0 T3R=17.3 max_loss=-35.9 | N=3 sum=-39.4 med=-34.5 T3R=-39.4 max_loss=-35.6 | -56.4 | -56.5 |
| 36 | `partial25_30m_no_reclaim_btc_down` | `partial_reduce` | 2/22 | N=22 sum=1064.0 med=33.9 T3R=385.0 max_loss=-144.4 | N=2 sum=53.1 med=26.6 T3R=53.1 max_loss=-28.0 | N=2 sum=-3.8 med=-1.9 T3R=-3.8 max_loss=-34.5 | -56.7 | -56.8 |
| 37 | `tight_trigger_sl80_5m_failed_v` | `tighten_stop` | 8/22 | N=22 sum=1053.5 med=39.4 T3R=374.5 max_loss=-102.9 | N=8 sum=152.1 med=34.9 T3R=-138.5 max_loss=-144.4 | N=8 sum=84.7 med=12.3 T3R=-205.9 max_loss=-102.9 | -67.2 | -67.3 |
| 38 | `tight_trigger_sl80_5m_weak_first` | `tighten_stop` | 13/22 | N=22 sum=1053.5 med=39.4 T3R=374.5 max_loss=-102.9 | N=13 sum=226.5 med=17.2 T3R=-64.1 max_loss=-144.4 | N=13 sum=159.1 med=13.3 T3R=-131.5 max_loss=-102.9 | -67.2 | -67.3 |
| 39 | `partial25_15m_failed_v` | `partial_reduce` | 4/22 | N=22 sum=1049.2 med=39.4 T3R=370.2 max_loss=-144.4 | N=4 sum=167.1 med=26.6 T3R=-35.9 max_loss=-35.9 | N=4 sum=95.4 med=8.8 T3R=-30.9 max_loss=-30.9 | -71.5 | -71.6 |
| 40 | `partial25_30m_btc_down_continues` | `partial_reduce` | 3/22 | N=22 sum=1048.3 med=33.9 T3R=369.3 max_loss=-160.1 | N=3 sum=-91.3 med=-28.0 T3R=-91.3 max_loss=-144.4 | N=3 sum=-163.9 med=-34.5 T3R=-163.9 max_loss=-160.1 | -72.4 | -72.5 |

## Read

- Best overlay by managed T3R: `tight_trigger_sl80_5m_no_reclaim_btc_down` -> N=22 sum=1162.4 med=39.4 T3R=483.4 max_loss=-102.9; delta T3R `41.6`.
- Overlay is only useful if it improves T3R or materially reduces tail loss without consuming most expectancy.
