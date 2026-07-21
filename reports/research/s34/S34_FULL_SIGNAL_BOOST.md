# S34 Full Signal Boost Gauntlet

Generated: `2026-07-01T12:45:15.651357+00:00`

Research-only. No live executor, shadow runner, .env, leverage, or sizing changes.

## Scope

- months: `4.52`
- thresholds: `['150K', '200K']`

## Hour17 Threshold 150K

- base: N=156 /mo=34.5 WR=61.5% avg=40.8 total=6360.1 tail100=21 mc=0.0 wf=4/5
- base no-overlap: N=73 /mo=16.2 WR=60.3% avg=42.3 total=3084.9 tail100=10 mc=0.002 wf=5/5

### Feature Ranking

| Feature | Bin | TEST lift | TEST | FULL | NOOV |
|---|---:|---:|---|---|---|
| `btc7d` | `mid` | 93.4 | N=8 /mo=5.9 WR=87.5% avg=141.4 total=1131.0 tail100=0 mc=0.008 wf=4/5 | N=44 /mo=9.7 WR=81.8% avg=85.6 total=3767.7 tail100=0 mc=0.0 wf=5/5 | N=29 /mo=6.4 WR=86.2% avg=83.0 total=2407.0 tail100=0 mc=0.0 wf=5/5 |
| `btc4h` | `lo` | 70.4 | N=16 /mo=11.8 WR=87.5% avg=118.4 total=1894.2 tail100=0 mc=0.0 wf=5/5 | N=52 /mo=11.5 WR=76.9% avg=87.2 total=4535.3 tail100=3 mc=0.0 wf=5/5 | N=28 /mo=6.2 WR=78.6% avg=89.0 total=2492.3 tail100=3 mc=0.0 wf=5/5 |
| `funding_rate` | `mid` | 42.6 | N=13 /mo=9.6 WR=69.2% avg=90.6 total=1177.8 tail100=2 mc=0.013 wf=5/5 | N=49 /mo=10.8 WR=75.5% avg=72.2 total=3537.5 tail100=2 mc=0.0 wf=5/5 | N=30 /mo=6.6 WR=76.7% avg=62.2 total=1865.6 tail100=1 mc=0.001 wf=5/5 |
| `btc3d` | `lo` | 41.4 | N=23 /mo=17.0 WR=82.6% avg=89.4 total=2055.1 tail100=2 mc=0.0 wf=4/5 | N=59 /mo=13.1 WR=74.6% avg=87.9 total=5188.9 tail100=5 mc=0.0 wf=5/5 | N=32 /mo=7.1 WR=75.0% avg=79.5 total=2544.8 tail100=2 mc=0.0 wf=5/5 |
| `n2h` | `hi` | 31.5 | N=24 /mo=17.7 WR=75.0% avg=79.5 total=1908.6 tail100=2 mc=0.003 wf=4/5 | N=61 /mo=13.5 WR=70.5% avg=60.6 total=3694.7 tail100=6 mc=0.0 wf=5/5 | N=32 /mo=7.1 WR=59.4% avg=36.3 total=1161.6 tail100=4 mc=0.061 wf=4/5 |
| `book_imbalance` | `lo` | 29.5 | N=23 /mo=17.0 WR=78.3% avg=77.5 total=1782.3 tail100=3 mc=0.004 wf=5/5 | N=34 /mo=7.5 WR=70.6% avg=65.8 total=2238.2 tail100=4 mc=0.002 wf=4/5 | N=20 /mo=4.4 WR=60.0% avg=42.0 total=839.6 tail100=3 mc=0.082 wf=5/5 |
| `be_ratio_pre` | `mid` | 26.3 | N=20 /mo=14.8 WR=75.0% avg=74.3 total=1486.0 tail100=1 mc=0.014 wf=5/5 | N=56 /mo=12.4 WR=69.6% avg=54.9 total=3071.9 tail100=5 mc=0.002 wf=5/5 | N=40 /mo=8.9 WR=65.0% avg=58.4 total=2335.4 tail100=4 mc=0.004 wf=5/5 |
| `running_notional` | `lo` | 24.2 | N=11 /mo=8.1 WR=63.6% avg=72.2 total=793.9 tail100=1 mc=0.051 wf=4/5 | N=47 /mo=10.4 WR=57.4% avg=54.1 total=2542.6 tail100=4 mc=0.01 wf=4/5 | N=38 /mo=8.4 WR=57.9% avg=49.2 total=1869.8 tail100=4 mc=0.036 wf=4/5 |

### Top Combos

| Combo | FULL | NOOV |
|---|---|---|
| `btc4h=lo & btc3d=lo` | N=23 /mo=5.1 WR=82.6% avg=137.8 total=3170.3 tail100=0 mc=0.0 wf=5/5 | N=11 /mo=2.4 WR=81.8% avg=142.1 total=1563.3 tail100=0 mc=0.003 wf=5/5 |
| `btc7d=mid & btc4h=lo` | N=14 /mo=3.1 WR=92.9% avg=113.8 total=1592.9 tail100=0 mc=0.001 wf=5/5 | N=11 /mo=2.4 WR=100.0% avg=118.5 total=1303.5 tail100=0 mc=0.0 wf=5/5 |
| `btc3d=lo & n2h=hi` | N=21 /mo=4.6 WR=85.7% avg=110.0 total=2310.8 tail100=2 mc=0.0 wf=4/5 | N=13 /mo=2.9 WR=92.3% avg=105.2 total=1367.5 tail100=0 mc=0.003 wf=4/5 |
| `btc7d=mid & n2h=hi` | N=17 /mo=3.8 WR=100.0% avg=109.1 total=1854.2 tail100=0 mc=0.0 wf=5/5 | N=12 /mo=2.7 WR=100.0% avg=92.9 total=1114.5 tail100=0 mc=0.0 wf=5/5 |
| `btc4h=lo & n2h=hi` | N=33 /mo=7.3 WR=78.8% avg=101.7 total=3356.8 tail100=2 mc=0.0 wf=5/5 | N=19 /mo=4.2 WR=73.7% avg=91.2 total=1732.2 tail100=2 mc=0.005 wf=5/5 |
| `btc7d=mid & btc3d=lo` | N=18 /mo=4.0 WR=77.8% avg=82.4 total=1483.6 tail100=0 mc=0.002 wf=5/5 | N=13 /mo=2.9 WR=84.6% avg=75.9 total=987.1 tail100=0 mc=0.003 wf=4/5 |
| `btc7d=mid & funding_rate=mid` | N=0 | N=0 |
| `btc4h=lo & funding_rate=mid` | N=0 | N=0 |

### Hour Slices

| Slice | Stats |
|---|---|
| `17_19` | N=74 /mo=16.4 WR=67.6% avg=62.9 total=4651.1 tail100=5 mc=0.0 wf=5/5 |
| `20_21` | N=28 /mo=6.2 WR=57.1% avg=20.1 total=563.5 tail100=4 mc=0.181 wf=3/5 |
| `22_23` | N=54 /mo=12.0 WR=55.6% avg=21.2 total=1145.5 tail100=12 mc=0.179 wf=4/5 |

### Tail Vetoes

| Veto | Kept NOOV | Dropped |
|---|---|---|
| `exclude_be_ratio_ge2` | N=71 /mo=15.7 WR=62.0% avg=45.0 total=3195.2 tail100=10 mc=0.004 wf=5/5 | N=5 /mo=1.1 WR=20.0% avg=-9.2 total=-46.2 tail100=0 mc=None wf=1/5 |
| `exclude_btc_conc_ge1m` | N=73 /mo=16.2 WR=60.3% avg=42.3 total=3084.9 tail100=10 mc=0.002 wf=5/5 | N=1 /mo=0.2 WR=100.0% avg=108.3 total=108.3 tail100=0 mc=None wf=None |
| `exclude_sync_100_200k` | N=71 /mo=15.7 WR=60.6% avg=38.4 total=2728.6 tail100=9 mc=0.011 wf=5/5 | N=21 /mo=4.6 WR=71.4% avg=56.2 total=1179.7 tail100=2 mc=0.017 wf=5/5 |
| `exclude_spread_gt_0p35` | N=73 /mo=16.2 WR=60.3% avg=42.3 total=3084.9 tail100=10 mc=0.002 wf=5/5 | N=0 |
| `only_bid_depth_ge100k` | N=24 /mo=5.3 WR=58.3% avg=29.7 total=712.4 tail100=5 mc=0.146 wf=4/5 | N=102 /mo=22.6 WR=62.7% avg=45.6 total=4651.1 tail100=13 mc=0.003 wf=5/5 |
| `only_book_bid_support` | N=66 /mo=14.6 WR=57.6% avg=40.4 total=2663.3 tail100=10 mc=0.016 wf=5/5 | N=34 /mo=7.5 WR=70.6% avg=65.8 total=2238.2 tail100=4 mc=0.002 wf=4/5 |
| `exclude_sat_sun` | N=53 /mo=11.7 WR=56.6% avg=18.7 total=990.7 tail100=7 mc=0.109 wf=3/5 | N=46 /mo=10.2 WR=73.9% avg=96.1 total=4421.9 tail100=3 mc=0.0 wf=4/5 |
| `exclude_btc5m_lt_minus50` | N=67 /mo=14.8 WR=62.7% avg=38.0 total=2544.8 tail100=10 mc=0.019 wf=4/5 | N=16 /mo=3.5 WR=75.0% avg=105.2 total=1682.8 tail100=0 mc=0.0 wf=4/5 |
| `exclude_near_funding_30m` | N=68 /mo=15.1 WR=61.8% avg=47.2 total=3207.1 tail100=8 mc=0.005 wf=5/5 | N=13 /mo=2.9 WR=38.5% avg=-6.8 total=-88.9 tail100=4 mc=0.564 wf=2/5 |

### Entry Tests

| Test | FULL | NOOV |
|---|---|---|
| `delay_0m` | N=156 /mo=34.5 WR=61.5% avg=40.8 total=6360.1 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=60.3% avg=42.3 total=3084.9 tail100=10 mc=0.002 wf=5/5 |
| `delay_1m` | N=156 /mo=34.5 WR=62.2% avg=42.9 total=6696.9 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=61.6% avg=47.8 total=3492.1 tail100=9 mc=0.001 wf=5/5 |
| `delay_5m` | N=156 /mo=34.5 WR=60.9% avg=40.9 total=6386.5 tail100=19 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=60.3% avg=45.1 total=3293.3 tail100=7 mc=0.001 wf=5/5 |
| `delay_15m` | N=156 /mo=34.5 WR=57.1% avg=35.7 total=5564.2 tail100=22 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=57.5% avg=44.1 total=3217.2 tail100=11 mc=0.003 wf=5/5 |
| `delay_30m` | N=156 /mo=34.5 WR=55.1% avg=24.0 total=3751.4 tail100=21 mc=0.016 wf=4/5 | N=73 /mo=16.2 WR=58.9% avg=28.7 total=2096.6 tail100=10 mc=0.037 wf=5/5 |
| `delay_60m` | N=156 /mo=34.5 WR=51.3% avg=14.5 total=2268.5 tail100=22 mc=0.107 wf=4/5 | N=73 /mo=16.2 WR=56.2% avg=28.2 total=2059.5 tail100=9 mc=0.033 wf=5/5 |
| `d1_ofi_pos` | N=55 /mo=12.2 WR=58.2% avg=29.6 total=1627.5 tail100=8 mc=0.041 wf=5/5 | N=36 /mo=8.0 WR=58.3% avg=35.3 total=1270.3 tail100=5 mc=0.06 wf=3/5 |
| `d1_bid_support` | N=122 /mo=27.0 WR=59.0% avg=36.1 total=4406.0 tail100=17 mc=0.003 wf=5/5 | N=66 /mo=14.6 WR=57.6% avg=46.4 total=3061.4 tail100=9 mc=0.009 wf=5/5 |
| `d1_spread_clean` | N=156 /mo=34.5 WR=62.2% avg=42.9 total=6696.9 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=61.6% avg=47.8 total=3492.1 tail100=9 mc=0.001 wf=5/5 |
| `d1_bid100k` | N=54 /mo=12.0 WR=61.1% avg=34.3 total=1853.1 tail100=8 mc=0.02 wf=3/5 | N=24 /mo=5.3 WR=62.5% avg=37.3 total=895.7 tail100=5 mc=0.102 wf=4/5 |
| `d1_ofi_pos_bid_support` | N=44 /mo=9.7 WR=54.5% avg=24.1 total=1059.6 tail100=7 mc=0.111 wf=5/5 | N=32 /mo=7.1 WR=53.1% avg=31.2 total=999.2 tail100=5 mc=0.118 wf=2/5 |

### Exit Tests

| Test | FULL | NOOV |
|---|---|---|
| `hold_4h` | N=156 /mo=34.5 WR=65.4% avg=37.4 total=5835.9 tail100=19 mc=0.0 wf=4/5 | N=90 /mo=19.9 WR=65.6% avg=29.7 total=2669.4 tail100=11 mc=0.012 wf=5/5 |
| `hold_6h` | N=156 /mo=34.5 WR=61.5% avg=40.8 total=6360.1 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=60.3% avg=42.3 total=3084.9 tail100=10 mc=0.002 wf=5/5 |
| `hold_8h` | N=156 /mo=34.5 WR=49.4% avg=25.7 total=4004.9 tail100=32 mc=0.022 wf=4/5 | N=70 /mo=15.5 WR=50.0% avg=28.9 total=2023.7 tail100=14 mc=0.06 wf=4/5 |
| `hold_10h` | N=156 /mo=34.5 WR=54.5% avg=16.0 total=2500.0 tail100=39 mc=0.137 wf=3/5 | N=70 /mo=15.5 WR=57.1% avg=17.2 total=1204.6 tail100=18 mc=0.2 wf=3/5 |
| `profit_lock_100_50` | N=156 /mo=34.5 WR=66.7% avg=33.5 total=5226.9 tail100=19 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=64.4% avg=37.1 total=2707.7 tail100=9 mc=0.004 wf=4/5 |
| `profit_lock_150_75` | N=156 /mo=34.5 WR=62.2% avg=38.9 total=6072.2 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=61.6% avg=40.8 total=2975.1 tail100=10 mc=0.003 wf=5/5 |
| `profit_lock_200_100` | N=156 /mo=34.5 WR=62.2% avg=41.7 total=6503.7 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=61.6% avg=44.2 total=3224.7 tail100=10 mc=0.002 wf=5/5 |
| `time_damage_exit_if_neg_3h` | N=156 /mo=34.5 WR=49.4% avg=33.3 total=5194.4 tail100=21 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=47.9% avg=30.6 total=2232.5 tail100=12 mc=0.043 wf=5/5 |
| `stop_150` | N=156 /mo=34.5 WR=57.7% avg=24.1 total=3760.7 tail100=38 mc=0.016 wf=4/5 | N=73 /mo=16.2 WR=54.8% avg=11.3 total=828.5 tail100=22 mc=0.268 wf=2/5 |
| `stop_200` | N=156 /mo=34.5 WR=60.3% avg=32.6 total=5089.5 tail100=26 mc=0.001 wf=4/5 | N=73 /mo=16.2 WR=57.5% avg=21.9 total=1602.2 tail100=15 mc=0.128 wf=3/5 |
| `stop_300` | N=156 /mo=34.5 WR=61.5% avg=39.6 total=6182.3 tail100=22 mc=0.0 wf=4/5 | N=73 /mo=16.2 WR=60.3% avg=39.0 total=2846.9 tail100=11 mc=0.006 wf=4/5 |

### Worst Hour17 Cards

| UTC | Net | Hour | DOW | Sync | BE | BTC5m | Spread | BidDepth | Imb | OFI60 | ToFund |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04-01T22:50:34.386000+00:00 | -448.5 | 22 | 2 | 0.0 | 0.0 | -14.8 | 0.0 | 0.0 | 0.0 | -0.055 | 69.4 |
| 2026-04-01T22:19:13.347000+00:00 | -412.8 | 22 | 2 | 23801.5 | 0.07 | -15.7 | 0.0 | 0.0 | 0.0 | -0.005 | 100.8 |
| 2026-03-11T23:25:06.191000+00:00 | -189.0 | 23 | 2 | 3010.1 | 0.01 | -4.9 | 0.0 | 0.0 | 0.0 | 0.098 | 34.9 |
| 2026-06-15T21:56:23.438000+00:00 | -184.4 | 21 | 0 | 61374.6 | 0.33 | -9.0 | 0.055 | 121865.1 | 0.275 | -0.514 | 123.6 |
| 2026-06-15T22:23:59.381000+00:00 | -183.9 | 22 | 0 | 501844.1 | 1.01 | -17.7 | 0.055 | 69832.3 | -0.452 | -0.525 | 96.0 |
| 2026-04-22T23:21:42.809000+00:00 | -173.4 | 23 | 2 | 42006.0 | 0.06 | -23.6 | 0.042 | 398766.5 | 0.949 | 0.312 | 38.3 |
| 2026-06-15T19:18:51.620000+00:00 | -165.9 | 19 | 0 | 292730.3 | 0.52 | -34.4 | 0.055 | 218050.6 | 0.731 | -0.196 | 281.1 |
| 2026-02-23T23:44:12.087000+00:00 | -159.2 | 23 | 0 | 11655.4 | 0.01 | -21.6 | 0.0 | 0.0 | 0.0 | 0.401 | 15.8 |

## Hour17 Threshold 200K

- base: N=126 /mo=27.9 WR=65.1% avg=39.8 total=5015.8 tail100=18 mc=0.0 wf=5/5
- base no-overlap: N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5

### Feature Ranking

| Feature | Bin | TEST lift | TEST | FULL | NOOV |
|---|---:|---:|---|---|---|
| `funding_rate` | `lo` | 73.9 | N=6 /mo=4.4 WR=100.0% avg=142.9 total=857.1 tail100=0 mc=None wf=5/5 | N=35 /mo=7.8 WR=80.0% avg=79.3 total=2774.4 tail100=3 mc=0.004 wf=4/5 | N=20 /mo=4.4 WR=80.0% avg=73.7 total=1473.3 tail100=3 mc=0.054 wf=3/5 |
| `sync_ratio` | `hi` | 52.5 | N=14 /mo=10.3 WR=85.7% avg=121.5 total=1701.5 tail100=0 mc=0.001 wf=5/5 | N=44 /mo=9.7 WR=75.0% avg=75.3 total=3312.7 tail100=5 mc=0.0 wf=5/5 | N=28 /mo=6.2 WR=64.3% avg=50.4 total=1410.3 tail100=5 mc=0.032 wf=4/5 |
| `n2h` | `hi` | 43.1 | N=20 /mo=14.8 WR=85.0% avg=112.1 total=2243.0 tail100=0 mc=0.0 wf=4/5 | N=55 /mo=12.2 WR=74.5% avg=67.1 total=3689.0 tail100=4 mc=0.0 wf=5/5 | N=30 /mo=6.6 WR=66.7% avg=38.5 total=1156.0 tail100=3 mc=0.017 wf=5/5 |
| `sync_sell_pre` | `hi` | 42.9 | N=16 /mo=11.8 WR=81.2% avg=111.9 total=1790.1 tail100=0 mc=0.0 wf=5/5 | N=46 /mo=10.2 WR=67.4% avg=67.1 total=3088.2 tail100=4 mc=0.0 wf=5/5 | N=29 /mo=6.4 WR=58.6% avg=53.5 total=1551.3 tail100=3 mc=0.028 wf=5/5 |
| `minutes_to_funding` | `hi` | 33.3 | N=12 /mo=8.9 WR=75.0% avg=102.3 total=1227.4 tail100=1 mc=0.013 wf=4/5 | N=42 /mo=9.3 WR=64.3% avg=58.9 total=2472.6 tail100=4 mc=0.001 wf=3/5 | N=32 /mo=7.1 WR=62.5% avg=42.0 total=1342.6 tail100=4 mc=0.023 wf=3/5 |
| `btc7d` | `mid` | 32.9 | N=6 /mo=4.4 WR=83.3% avg=101.9 total=611.3 tail100=0 mc=None wf=4/5 | N=35 /mo=7.8 WR=80.0% avg=69.2 total=2420.4 tail100=1 mc=0.0 wf=5/5 | N=23 /mo=5.1 WR=82.6% avg=61.5 total=1415.3 tail100=1 mc=0.001 wf=5/5 |
| `be_ratio_pre` | `mid` | 21.1 | N=14 /mo=10.3 WR=85.7% avg=90.1 total=1261.7 tail100=0 mc=0.003 wf=5/5 | N=43 /mo=9.5 WR=74.4% avg=59.2 total=2545.7 tail100=5 mc=0.002 wf=5/5 | N=35 /mo=7.8 WR=74.3% avg=57.7 total=2017.8 tail100=4 mc=0.001 wf=5/5 |
| `ofi_0_60_ratio` | `lo` | 16.2 | N=16 /mo=11.8 WR=81.2% avg=85.2 total=1363.7 tail100=1 mc=0.007 wf=5/5 | N=45 /mo=10.0 WR=71.1% avg=64.3 total=2895.6 tail100=6 mc=0.001 wf=5/5 | N=29 /mo=6.4 WR=65.5% avg=50.5 total=1465.2 tail100=5 mc=0.04 wf=5/5 |

### Top Combos

| Combo | FULL | NOOV |
|---|---|---|
| `funding_rate=lo & sync_ratio=hi` | N=24 /mo=5.3 WR=91.7% avg=127.7 total=3064.3 tail100=1 mc=0.0 wf=5/5 | N=17 /mo=3.8 WR=94.1% avg=119.8 total=2036.5 tail100=1 mc=0.001 wf=5/5 |
| `funding_rate=lo & sync_sell_pre=hi` | N=22 /mo=4.9 WR=86.4% avg=128.5 total=2828.0 tail100=1 mc=0.0 wf=5/5 | N=17 /mo=3.8 WR=82.4% avg=116.1 total=1973.9 tail100=1 mc=0.002 wf=5/5 |
| `funding_rate=lo & minutes_to_funding=hi` | N=24 /mo=5.3 WR=79.2% avg=105.9 total=2541.2 tail100=2 mc=0.001 wf=5/5 | N=18 /mo=4.0 WR=77.8% avg=81.3 total=1462.7 tail100=2 mc=0.01 wf=4/5 |
| `funding_rate=lo & n2h=hi` | N=30 /mo=6.6 WR=86.7% avg=105.3 total=3159.4 tail100=1 mc=0.0 wf=5/5 | N=19 /mo=4.2 WR=78.9% avg=65.8 total=1249.3 tail100=1 mc=0.005 wf=4/5 |
| `sync_sell_pre=hi & minutes_to_funding=hi` | N=18 /mo=4.0 WR=66.7% avg=89.8 total=1616.9 tail100=1 mc=0.01 wf=5/5 | N=15 /mo=3.3 WR=60.0% avg=65.3 total=980.0 tail100=1 mc=0.05 wf=5/5 |
| `sync_ratio=hi & minutes_to_funding=hi` | N=17 /mo=3.8 WR=64.7% avg=84.3 total=1432.6 tail100=2 mc=0.024 wf=3/5 | N=13 /mo=2.9 WR=61.5% avg=61.9 total=804.2 tail100=2 mc=0.094 wf=3/5 |
| `sync_ratio=hi & n2h=hi` | N=26 /mo=5.8 WR=76.9% avg=84.4 total=2193.4 tail100=3 mc=0.001 wf=5/5 | N=19 /mo=4.2 WR=68.4% avg=56.0 total=1063.3 tail100=3 mc=0.031 wf=5/5 |
| `sync_ratio=hi & sync_sell_pre=hi` | N=35 /mo=7.8 WR=74.3% avg=80.0 total=2798.3 tail100=3 mc=0.0 wf=5/5 | N=23 /mo=5.1 WR=60.9% avg=54.4 total=1250.3 tail100=3 mc=0.037 wf=3/5 |

### Hour Slices

| Slice | Stats |
|---|---|
| `17_19` | N=61 /mo=13.5 WR=67.2% avg=56.3 total=3432.6 tail100=5 mc=0.0 wf=4/5 |
| `20_21` | N=21 /mo=4.7 WR=61.9% avg=24.3 total=510.2 tail100=3 mc=0.133 wf=4/5 |
| `22_23` | N=44 /mo=9.7 WR=63.6% avg=24.4 total=1073.0 tail100=10 mc=0.147 wf=5/5 |

### Tail Vetoes

| Veto | Kept NOOV | Dropped |
|---|---|---|
| `exclude_be_ratio_ge2` | N=62 /mo=13.7 WR=62.9% avg=35.2 total=2181.0 tail100=11 mc=0.013 wf=5/5 | N=5 /mo=1.1 WR=60.0% avg=134.5 total=672.5 tail100=0 mc=None wf=3/5 |
| `exclude_btc_conc_ge1m` | N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5 | N=3 /mo=0.7 WR=100.0% avg=248.1 total=744.3 tail100=0 mc=None wf=None |
| `exclude_sync_100_200k` | N=59 /mo=13.1 WR=62.7% avg=38.9 total=2296.4 tail100=10 mc=0.008 wf=5/5 | N=20 /mo=4.4 WR=75.0% avg=25.4 total=508.3 tail100=2 mc=0.082 wf=4/5 |
| `exclude_spread_gt_0p35` | N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5 | N=0 |
| `only_bid_depth_ge100k` | N=21 /mo=4.7 WR=61.9% avg=17.8 total=374.3 tail100=5 mc=0.234 wf=3/5 | N=79 /mo=17.5 WR=65.8% avg=35.9 total=2835.8 tail100=11 mc=0.007 wf=5/5 |
| `only_book_bid_support` | N=58 /mo=12.8 WR=62.1% avg=37.8 total=2190.1 tail100=10 mc=0.015 wf=5/5 | N=30 /mo=6.6 WR=70.0% avg=60.5 total=1814.6 tail100=3 mc=0.0 wf=5/5 |
| `exclude_sat_sun` | N=46 /mo=10.2 WR=58.7% avg=18.6 total=857.4 tail100=8 mc=0.136 wf=3/5 | N=36 /mo=8.0 WR=75.0% avg=72.1 total=2595.1 tail100=3 mc=0.0 wf=5/5 |
| `exclude_btc5m_lt_minus50` | N=55 /mo=12.2 WR=60.0% avg=19.8 total=1089.3 tail100=11 mc=0.09 wf=3/5 | N=14 /mo=3.1 WR=85.7% avg=107.3 total=1501.7 tail100=0 mc=0.001 wf=4/5 |
| `exclude_near_funding_30m` | N=58 /mo=12.8 WR=65.5% avg=44.1 total=2556.2 tail100=8 mc=0.004 wf=5/5 | N=9 /mo=2.0 WR=44.4% avg=-5.5 total=-49.6 tail100=3 mc=0.529 wf=1/5 |

### Entry Tests

| Test | FULL | NOOV |
|---|---|---|
| `delay_0m` | N=126 /mo=27.9 WR=65.1% avg=39.8 total=5015.8 tail100=18 mc=0.0 wf=5/5 | N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5 |
| `delay_1m` | N=126 /mo=27.9 WR=65.1% avg=43.8 total=5512.9 tail100=15 mc=0.0 wf=5/5 | N=63 /mo=14.0 WR=61.9% avg=40.7 total=2563.4 tail100=8 mc=0.011 wf=5/5 |
| `delay_5m` | N=126 /mo=27.9 WR=62.7% avg=40.2 total=5061.7 tail100=15 mc=0.0 wf=5/5 | N=63 /mo=14.0 WR=58.7% avg=35.6 total=2239.9 tail100=7 mc=0.019 wf=5/5 |
| `delay_15m` | N=126 /mo=27.9 WR=57.9% avg=33.3 total=4192.1 tail100=18 mc=0.003 wf=5/5 | N=63 /mo=14.0 WR=57.1% avg=30.2 total=1903.1 tail100=10 mc=0.043 wf=5/5 |
| `delay_30m` | N=126 /mo=27.9 WR=54.0% avg=24.8 total=3125.2 tail100=17 mc=0.015 wf=5/5 | N=63 /mo=14.0 WR=55.6% avg=19.4 total=1222.5 tail100=9 mc=0.127 wf=4/5 |
| `delay_60m` | N=126 /mo=27.9 WR=54.8% avg=18.3 total=2303.2 tail100=19 mc=0.052 wf=5/5 | N=63 /mo=14.0 WR=55.6% avg=20.0 total=1261.7 tail100=10 mc=0.13 wf=4/5 |
| `d1_ofi_pos` | N=40 /mo=8.9 WR=55.0% avg=23.6 total=944.2 tail100=6 mc=0.091 wf=4/5 | N=27 /mo=6.0 WR=48.1% avg=8.9 total=241.2 tail100=5 mc=0.34 wf=1/5 |
| `d1_bid_support` | N=96 /mo=21.3 WR=62.5% avg=37.8 total=3625.6 tail100=13 mc=0.006 wf=5/5 | N=58 /mo=12.8 WR=60.3% avg=44.3 total=2567.6 tail100=8 mc=0.005 wf=5/5 |
| `d1_spread_clean` | N=126 /mo=27.9 WR=65.1% avg=43.8 total=5512.9 tail100=15 mc=0.0 wf=5/5 | N=63 /mo=14.0 WR=61.9% avg=40.7 total=2563.4 tail100=8 mc=0.011 wf=5/5 |
| `d1_bid100k` | N=47 /mo=10.4 WR=63.8% avg=46.9 total=2206.1 tail100=6 mc=0.011 wf=5/5 | N=21 /mo=4.7 WR=61.9% avg=21.9 total=460.0 tail100=4 mc=0.175 wf=3/5 |
| `d1_ofi_pos_bid_support` | N=32 /mo=7.1 WR=50.0% avg=12.2 total=391.5 tail100=6 mc=0.277 wf=3/5 | N=25 /mo=5.5 WR=48.0% avg=8.9 total=222.2 tail100=5 mc=0.339 wf=2/5 |

### Exit Tests

| Test | FULL | NOOV |
|---|---|---|
| `hold_4h` | N=126 /mo=27.9 WR=63.5% avg=31.6 total=3978.0 tail100=15 mc=0.0 wf=5/5 | N=74 /mo=16.4 WR=63.5% avg=21.8 total=1616.8 tail100=9 mc=0.066 wf=5/5 |
| `hold_6h` | N=126 /mo=27.9 WR=65.1% avg=39.8 total=5015.8 tail100=18 mc=0.0 wf=5/5 | N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5 |
| `hold_8h` | N=126 /mo=27.9 WR=49.2% avg=26.5 total=3341.4 tail100=24 mc=0.018 wf=5/5 | N=61 /mo=13.5 WR=47.5% avg=15.5 total=947.9 tail100=12 mc=0.166 wf=4/5 |
| `hold_10h` | N=126 /mo=27.9 WR=55.6% avg=22.2 total=2796.9 tail100=30 mc=0.07 wf=3/5 | N=61 /mo=13.5 WR=54.1% avg=1.9 total=118.2 tail100=18 mc=0.449 wf=3/5 |
| `profit_lock_100_50` | N=126 /mo=27.9 WR=69.0% avg=28.1 total=3543.5 tail100=17 mc=0.001 wf=4/5 | N=63 /mo=14.0 WR=65.1% avg=23.4 total=1472.5 tail100=10 mc=0.042 wf=4/5 |
| `profit_lock_150_75` | N=126 /mo=27.9 WR=65.9% avg=37.5 total=4722.3 tail100=18 mc=0.001 wf=5/5 | N=63 /mo=14.0 WR=63.5% avg=29.8 total=1874.5 tail100=11 mc=0.03 wf=5/5 |
| `profit_lock_200_100` | N=126 /mo=27.9 WR=65.9% avg=40.9 total=5155.9 tail100=18 mc=0.001 wf=5/5 | N=63 /mo=14.0 WR=63.5% avg=35.7 total=2247.6 tail100=11 mc=0.015 wf=5/5 |
| `time_damage_exit_if_neg_3h` | N=126 /mo=27.9 WR=51.6% avg=30.0 total=3777.0 tail100=17 mc=0.003 wf=5/5 | N=63 /mo=14.0 WR=46.0% avg=20.1 total=1265.0 tail100=11 mc=0.13 wf=5/5 |
| `stop_150` | N=126 /mo=27.9 WR=61.9% avg=22.3 total=2810.5 tail100=29 mc=0.018 wf=4/5 | N=63 /mo=14.0 WR=57.1% avg=4.4 total=274.3 tail100=19 mc=0.405 wf=4/5 |
| `stop_200` | N=126 /mo=27.9 WR=63.5% avg=29.2 total=3674.2 tail100=22 mc=0.005 wf=5/5 | N=63 /mo=14.0 WR=58.7% avg=11.5 total=726.4 tail100=15 mc=0.277 wf=4/5 |
| `stop_300` | N=126 /mo=27.9 WR=65.1% avg=37.5 total=4726.4 tail100=19 mc=0.001 wf=4/5 | N=63 /mo=14.0 WR=61.9% avg=29.8 total=1876.0 tail100=12 mc=0.05 wf=4/5 |

### Worst Hour17 Cards

| UTC | Net | Hour | DOW | Sync | BE | BTC5m | Spread | BidDepth | Imb | OFI60 | ToFund |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04-01T22:50:34.386000+00:00 | -448.5 | 22 | 2 | 0.0 | 0.0 | -14.8 | 0.0 | 0.0 | 0.0 | -0.055 | 69.4 |
| 2026-03-11T23:25:06.191000+00:00 | -189.0 | 23 | 2 | 3010.1 | 0.01 | -4.9 | 0.0 | 0.0 | 0.0 | 0.098 | 34.9 |
| 2026-06-15T22:24:00.390000+00:00 | -183.9 | 22 | 0 | 553284.9 | 0.88 | -17.1 | 0.055 | 405357.7 | 0.591 | -0.528 | 96.0 |
| 2026-06-15T21:56:51.553000+00:00 | -175.7 | 21 | 0 | 61638.9 | 0.2 | -8.1 | 0.055 | 96445.3 | -0.072 | -0.193 | 123.1 |
| 2026-04-22T23:21:42.809000+00:00 | -173.4 | 23 | 2 | 42006.0 | 0.06 | -23.6 | 0.042 | 398766.5 | 0.949 | 0.312 | 38.3 |
| 2026-06-15T19:18:53.490000+00:00 | -161.8 | 19 | 0 | 292730.3 | 0.19 | -39.3 | 0.055 | 135044.3 | 0.263 | -0.138 | 281.1 |
| 2026-02-23T23:44:12.087000+00:00 | -159.2 | 23 | 0 | 11655.4 | 0.01 | -21.6 | 0.0 | 0.0 | 0.0 | 0.401 | 15.8 |
| 2026-04-19T17:51:16.652000+00:00 | -154.6 | 17 | 6 | 574198.2 | 1.17 | -34.3 | 0.044 | 345209.9 | 0.383 | -0.261 | 368.7 |

## SHORT_NOISY BTC-Confirmed

| Rank | Config | FULL | NOOV |
|---:|---|---|---|
| 1 | `btc1000k_d5_h180` | N=24 /mo=5.3 WR=79.2% avg=67.4 total=1617.7 tail100=3 mc=0.025 wf=4/5 | N=14 /mo=3.1 WR=92.9% avg=110.6 total=1549.0 tail100=1 mc=0.003 wf=5/5 |
| 2 | `btc1000k_d5_h120` | N=24 /mo=5.3 WR=62.5% avg=81.6 total=1957.9 tail100=2 mc=0.001 wf=4/5 | N=14 /mo=3.1 WR=71.4% avg=109.2 total=1528.2 tail100=0 mc=0.003 wf=5/5 |
| 3 | `btc1000k_d5_h240` | N=24 /mo=5.3 WR=66.7% avg=80.6 total=1934.7 tail100=5 mc=0.059 wf=4/5 | N=14 /mo=3.1 WR=78.6% avg=98.3 total=1376.2 tail100=1 mc=0.044 wf=5/5 |
| 4 | `btc1000k_d10_h120` | N=17 /mo=3.8 WR=64.7% avg=90.5 total=1538.6 tail100=1 mc=0.002 wf=5/5 | N=11 /mo=2.4 WR=63.6% avg=102.6 total=1128.7 tail100=0 mc=0.014 wf=5/5 |
| 5 | `btc2000k_d5_h120` | N=11 /mo=2.4 WR=72.7% avg=84.6 total=930.5 tail100=1 mc=0.03 wf=4/5 | N=7 /mo=1.6 WR=100.0% avg=144.7 total=1013.0 tail100=0 mc=None wf=5/5 |
| 6 | `btc1000k_d5_h90` | N=24 /mo=5.3 WR=58.3% avg=48.9 total=1173.6 tail100=1 mc=0.011 wf=5/5 | N=15 /mo=3.3 WR=53.3% avg=66.4 total=995.5 tail100=0 mc=0.024 wf=5/5 |
| 7 | `btc500k_d15_h240` | N=19 /mo=4.2 WR=68.4% avg=92.9 total=1764.4 tail100=4 mc=0.046 wf=4/5 | N=14 /mo=3.1 WR=71.4% avg=71.0 total=994.1 tail100=3 mc=0.122 wf=3/5 |
| 8 | `btc2000k_d5_h180` | N=11 /mo=2.4 WR=72.7% avg=49.6 total=546.0 tail100=2 mc=0.163 wf=4/5 | N=7 /mo=1.6 WR=100.0% avg=136.5 total=955.5 tail100=0 mc=None wf=5/5 |
| 9 | `btc1000k_d10_h240` | N=17 /mo=3.8 WR=76.5% avg=119.0 total=2022.2 tail100=2 mc=0.026 wf=4/5 | N=11 /mo=2.4 WR=72.7% avg=81.0 total=891.4 tail100=1 mc=0.119 wf=4/5 |
| 10 | `btc500k_d10_h120` | N=26 /mo=5.8 WR=61.5% avg=62.4 total=1621.6 tail100=3 mc=0.016 wf=4/5 | N=18 /mo=4.0 WR=50.0% avg=48.2 total=866.7 tail100=2 mc=0.099 wf=3/5 |

## BUY-Side Fade

| Variant | FULL | NOOV |
|---|---|---|
| `all_t0_h45_sl75` | N=415 /mo=91.9 WR=47.0% avg=-4.0 total=-1666.5 tail100=0 mc=0.897 wf=1/5 | N=326 /mo=72.2 WR=45.7% avg=-5.9 total=-1922.5 tail100=0 mc=0.954 wf=0/5 |
| `silent30_t0_h45_sl75_lookahead_label` | N=184 /mo=40.7 WR=66.8% avg=24.8 total=4560.8 tail100=0 mc=0.0 wf=5/5 | N=177 /mo=39.2 WR=66.7% avg=24.6 total=4349.9 tail100=0 mc=0.0 wf=5/5 |
| `silent30_confirm_t30_h45_sl75_tradeable` | N=184 /mo=40.7 WR=41.3% avg=-9.2 total=-1689.9 tail100=0 mc=0.988 wf=0/5 | N=177 /mo=39.2 WR=40.1% avg=-10.1 total=-1784.7 tail100=0 mc=0.994 wf=0/5 |
| `silent30_ask_depth_hi_t0` | N=42 /mo=9.3 WR=76.2% avg=30.3 total=1270.8 tail100=0 mc=0.0 wf=5/5 | N=41 /mo=9.1 WR=75.6% avg=28.0 total=1148.2 tail100=0 mc=0.0 wf=5/5 |

## Portfolio

- best SHORT config: `btc1000k_d5_h180`

| Portfolio | Stats |
|---|---|
| `h17_only` | N=63 /mo=14.0 WR=61.9% avg=33.6 total=2117.7 tail100=11 mc=0.021 wf=5/5 |
| `short_noisy_only` | N=14 /mo=3.1 WR=92.9% avg=110.6 total=1549.0 tail100=1 mc=0.003 wf=5/5 |
| `buy_fade_only` | N=177 /mo=39.2 WR=66.7% avg=24.6 total=4349.9 tail100=0 mc=0.0 wf=5/5 |
| `h17_plus_short` | N=74 /mo=16.4 WR=64.9% avg=45.8 total=3388.7 tail100=12 mc=0.001 wf=5/5 |
| `h17_plus_buy` | N=206 /mo=45.6 WR=66.0% avg=27.7 total=5712.6 tail100=11 mc=0.0 wf=5/5 |
| `all_three` | N=217 /mo=48.1 WR=66.8% avg=32.2 total=6983.5 tail100=12 mc=0.0 wf=5/5 |

## Route Counts In All-Three

```json
{
  "BUY_FADE_T0_LABEL": 145,
  "H17_LONG": 61,
  "SHORT_NOISY_BEST": 11
}
```

## Read

- Treat T0 silence-labelled BUY fade variants as research labels unless explicitly shown as confirmed/tradeable.
- Promotion needs forward paper/shadow accumulation and operator sign-off.
- Sizing/tail-budget remains separate and urgent; this report only ranks signals.
