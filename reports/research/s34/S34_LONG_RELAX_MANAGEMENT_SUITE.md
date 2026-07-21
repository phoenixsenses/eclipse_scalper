# S34 Long Relax + Management Suite

Generated: `2026-07-01T08:45:03.484243+00:00`

## Baselines
- `long_current_relax_or`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3
- `long_strict_btc7d`: N=28 WR=71.4% avg=+72.5 sum=+2029.1 T3R=+1073.4 worst=-305.3 tail100=3 /mo=6.4
- `short_current_btc2m_d5_h2`: N=7 WR=85.7% avg=+99.0 sum=+693.2 T3R=-58.4 worst=-169.6 tail100=1 /mo=7.0
- `short_btc1m_d5_h4`: N=12 WR=66.7% avg=+150.0 sum=+1800.4 T3R=+327.1 worst=-165.8 tail100=3 /mo=3.3

## long_gate
- `current_relax_or`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3 | hold N=18 WR=77.8% avg=+85.1 sum=+1531.5 T3R=+743.4 worst=-91.5 tail100=0 /mo=8.6
- `strict_btc7d_only`: N=28 WR=71.4% avg=+72.5 sum=+2029.1 T3R=+1073.4 worst=-305.3 tail100=3 /mo=6.4 | hold N=9 WR=66.7% avg=+104.1 sum=+937.1 T3R=+162.3 worst=-91.5 tail100=0 /mo=9.0
- `btc4h_only`: N=53 WR=66.0% avg=+35.8 sum=+1895.8 T3R=+940.2 worst=-305.3 tail100=7 /mo=12.2 | hold N=16 WR=87.5% avg=+99.4 sum=+1590.0 T3R=+801.8 worst=-52.3 tail100=0 /mo=7.5
- `btc4h_and_btc7d`: N=23 WR=73.9% avg=+70.8 sum=+1627.4 T3R=+671.8 worst=-305.3 tail100=3 /mo=5.3 | hold N=7 WR=85.7% avg=+145.8 sum=+1020.6 T3R=+245.8 worst=-52.3 tail100=0 /mo=7.0
- `no_btc_regime`: N=64 WR=60.9% avg=+29.0 sum=+1857.2 T3R=+901.5 worst=-305.3 tail100=9 /mo=14.7 | hold N=20 WR=70.0% avg=+70.4 sum=+1407.1 T3R=+618.9 worst=-91.5 tail100=0 /mo=9.6
- `sync300_or`: N=75 WR=68.0% avg=+37.2 sum=+2789.3 T3R=+1833.6 worst=-311.4 tail100=8 /mo=17.2 | hold N=23 WR=78.3% avg=+77.5 sum=+1783.4 T3R=+995.2 worst=-91.5 tail100=0 /mo=10.9
- `sync500_or`: N=91 WR=69.2% avg=+30.5 sum=+2775.4 T3R=+1806.4 worst=-406.9 tail100=11 /mo=20.8 | hold N=28 WR=82.1% avg=+72.2 sum=+2020.8 T3R=+1135.1 worst=-313.6 tail100=1 /mo=28.0
- `score4_or`: N=30 WR=76.7% avg=+60.7 sum=+1820.7 T3R=+1099.7 worst=-147.4 tail100=3 /mo=7.0 | hold N=9 WR=66.7% avg=+39.9 sum=+359.3 T3R=-175.2 worst=-131.5 tail100=2 /mo=4.1
- `allow_mon_wed_or`: N=81 WR=63.0% avg=+36.1 sum=+2924.8 T3R=+1928.5 worst=-305.3 tail100=9 /mo=18.6 | hold N=25 WR=72.0% avg=+70.2 sum=+1754.3 T3R=+841.3 worst=-101.8 tail100=1 /mo=25.0

## tail_detector_kept
- `exclude_eth1h_lt_-80` kept: N=29 WR=65.5% avg=+45.7 sum=+1323.9 T3R=+465.7 worst=-131.5 tail100=2 /mo=6.9 | dropped: N=29 WR=65.5% avg=+33.6 sum=+973.6 T3R=+252.5 worst=-305.3 tail100=5 /mo=6.8
- `exclude_eth4h_lt_-150` kept: N=36 WR=63.9% avg=+32.8 sum=+1181.4 T3R=+377.0 worst=-305.3 tail100=5 /mo=8.4 | dropped: N=22 WR=68.2% avg=+50.7 sum=+1116.1 T3R=+328.0 worst=-213.9 tail100=2 /mo=5.1
- `exclude_btc1h_lt_-50` kept: N=28 WR=60.7% avg=+26.5 sum=+741.5 T3R=+50.8 worst=-305.3 tail100=4 /mo=6.4 | dropped: N=30 WR=70.0% avg=+51.9 sum=+1556.0 T3R=+669.4 worst=-213.9 tail100=3 /mo=7.1
- `exclude_btc4h_lt_-150` kept: N=44 WR=63.6% avg=+29.6 sum=+1304.3 T3R=+499.9 worst=-305.3 tail100=6 /mo=10.1 | dropped: N=14 WR=71.4% avg=+70.9 sum=+993.2 T3R=+235.9 worst=-194.2 tail100=1 /mo=3.4
- `exclude_sync_100_200` kept: N=40 WR=67.5% avg=+54.6 sum=+2183.5 T3R=+1227.8 worst=-305.3 tail100=6 /mo=9.2 | dropped: N=18 WR=61.1% avg=+6.3 sum=+114.0 T3R=-165.1 worst=-213.9 tail100=1 /mo=4.6
- `only_n2h_ge5` kept: N=31 WR=74.2% avg=+56.0 sum=+1737.4 T3R=+875.6 worst=-305.3 tail100=5 /mo=7.2 | dropped: N=27 WR=55.6% avg=+20.7 sum=+560.1 T3R=-40.4 worst=-213.9 tail100=2 /mo=6.3
- `exclude_slow_elapsed_gt180s` kept: N=51 WR=64.7% avg=+35.2 sum=+1793.4 T3R=+919.9 worst=-305.3 tail100=6 /mo=11.8 | dropped: N=7 WR=71.4% avg=+72.0 sum=+504.1 T3R=-134.4 worst=-194.2 tail100=1 /mo=1.8
- `only_echo` kept: N=43 WR=67.4% avg=+36.7 sum=+1576.5 T3R=+714.7 worst=-305.3 tail100=7 /mo=9.9 | dropped: N=15 WR=60.0% avg=+48.1 sum=+721.0 T3R=+88.9 worst=-91.5 tail100=0 /mo=3.5
- `only_prebuild` kept: N=45 WR=68.9% avg=+48.6 sum=+2185.9 T3R=+1230.3 worst=-305.3 tail100=6 /mo=10.3 | dropped: N=13 WR=53.8% avg=+8.6 sum=+111.6 T3R=-287.7 worst=-194.2 tail100=1 /mo=3.4
- `exclude_high_vol30_gt60` kept: N=39 WR=66.7% avg=+43.5 sum=+1695.4 T3R=+739.8 worst=-305.3 tail100=4 /mo=9.2 | dropped: N=19 WR=63.2% avg=+31.7 sum=+602.1 T3R=+13.9 worst=-213.9 tail100=3 /mo=4.9

## dynamic_hold_long
- `h30m`: N=58 WR=65.5% avg=+2.2 sum=+129.4 T3R=-270.4 worst=-241.6 tail100=3 /mo=13.3 | hold N=18 WR=77.8% avg=+18.0 sum=+324.4 T3R=+22.0 worst=-117.4 tail100=1 /mo=8.6
- `h60m`: N=58 WR=58.6% avg=+7.8 sum=+452.4 T3R=-336.2 worst=-251.6 tail100=3 /mo=13.3 | hold N=18 WR=61.1% avg=+10.0 sum=+179.5 T3R=-174.1 worst=-137.0 tail100=1 /mo=8.6
- `h90m`: N=58 WR=53.4% avg=+10.6 sum=+614.3 T3R=-214.8 worst=-182.5 tail100=4 /mo=13.3 | hold N=18 WR=44.4% avg=+9.3 sum=+166.6 T3R=-242.7 worst=-104.2 tail100=1 /mo=8.6
- `h120m`: N=58 WR=60.3% avg=+17.1 sum=+992.4 T3R=+129.0 worst=-273.3 tail100=5 /mo=13.3 | hold N=18 WR=72.2% avg=+35.5 sum=+639.9 T3R=+167.9 worst=-54.1 tail100=0 /mo=8.6
- `h180m`: N=58 WR=67.2% avg=+32.6 sum=+1888.8 T3R=+1059.4 worst=-293.1 tail100=7 /mo=13.3 | hold N=18 WR=77.8% avg=+68.4 sum=+1231.1 T3R=+553.7 worst=-112.9 tail100=1 /mo=8.6
- `h240m`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3 | hold N=18 WR=77.8% avg=+85.1 sum=+1531.5 T3R=+743.4 worst=-91.5 tail100=0 /mo=8.6
- `h360m`: N=58 WR=67.2% avg=+39.4 sum=+2284.8 T3R=+1341.8 worst=-274.6 tail100=7 /mo=13.3 | hold N=18 WR=88.9% avg=+79.4 sum=+1429.7 T3R=+810.3 worst=-108.1 tail100=1 /mo=8.6
- `h480m`: N=58 WR=51.7% avg=+21.9 sum=+1267.6 T3R=+286.8 worst=-304.1 tail100=11 /mo=13.3 | hold N=18 WR=61.1% avg=+43.6 sum=+784.8 T3R=+314.4 worst=-161.5 tail100=2 /mo=8.6

## dynamic_hold_short
- `h30m`: N=7 WR=42.9% avg=+15.1 sum=+105.9 T3R=-251.1 worst=-136.3 tail100=1 /mo=7.0 | hold N=3 WR=33.3% avg=-24.9 sum=-74.7 T3R=-74.7 worst=-136.3 tail100=1 /mo=3.0
- `h60m`: N=7 WR=57.1% avg=+19.8 sum=+138.9 T3R=-268.5 worst=-158.6 tail100=1 /mo=7.0 | hold N=3 WR=33.3% avg=-26.3 sum=-79.0 T3R=-79.0 worst=-158.6 tail100=1 /mo=3.0
- `h90m`: N=7 WR=85.7% avg=+71.5 sum=+500.8 T3R=-37.0 worst=-131.1 tail100=1 /mo=7.0 | hold N=3 WR=66.7% avg=-20.1 sum=-60.2 T3R=-60.2 worst=-131.1 tail100=1 /mo=3.0
- `h120m`: N=7 WR=85.7% avg=+99.0 sum=+693.2 T3R=-58.4 worst=-169.6 tail100=1 /mo=7.0 | hold N=3 WR=66.7% avg=-19.5 sum=-58.4 T3R=-58.4 worst=-169.6 tail100=1 /mo=3.0
- `h180m`: N=7 WR=71.4% avg=+75.6 sum=+529.3 T3R=-102.0 worst=-118.8 tail100=1 /mo=7.0 | hold N=3 WR=66.7% avg=-22.5 sum=-67.5 T3R=-67.5 worst=-118.8 tail100=1 /mo=3.0
- `h240m`: N=7 WR=42.9% avg=-2.2 sum=-15.7 T3R=-548.3 worst=-196.5 tail100=3 /mo=7.0 | hold N=3 WR=0.0% avg=-127.9 sum=-383.6 T3R=-383.6 worst=-196.5 tail100=2 /mo=3.0

## confidence
- `conf_0_2`: N=3 WR=33.3% avg=-15.2 sum=-45.5 T3R=-45.5 worst=-80.2 tail100=0 /mo=1.8 | hold N=1 WR=100.0% avg=+60.2 sum=+60.2 T3R=+60.2 worst=60.2 tail100=0 /mo=1.0
- `conf_3_4`: N=20 WR=55.0% avg=+30.1 sum=+601.6 T3R=-2.2 worst=-91.5 tail100=0 /mo=5.1 | hold N=6 WR=50.0% avg=+3.4 sum=+20.4 T3R=-177.2 worst=-91.5 tail100=0 /mo=6.0
- `conf_5_plus`: N=35 WR=74.3% avg=+49.8 sum=+1741.4 T3R=+879.6 worst=-305.3 tail100=7 /mo=8.0 | hold N=11 WR=100.0% avg=+138.7 sum=+1526.0 T3R=+737.8 worst=17.6 tail100=0 /mo=11.0
- `conf_ge4`: N=47 WR=68.1% avg=+46.0 sum=+2162.6 T3R=+1206.9 worst=-305.3 tail100=7 /mo=10.8 | hold N=15 WR=73.3% avg=+75.8 sum=+1136.3 T3R=+348.2 worst=-137.7 tail100=2 /mo=6.6
- `conf_ge5`: N=35 WR=74.3% avg=+49.8 sum=+1741.4 T3R=+879.6 worst=-305.3 tail100=7 /mo=8.0 | hold N=11 WR=100.0% avg=+138.7 sum=+1526.0 T3R=+737.8 worst=17.6 tail100=0 /mo=11.0

## exit_by_state
- `hold_all_4h`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3 | hold N=18 WR=77.8% avg=+85.1 sum=+1531.5 T3R=+743.4 worst=-91.5 tail100=0 /mo=8.6
- `exit_on_noisy_follow`: N=58 WR=32.8% avg=+15.8 sum=+918.8 T3R=-26.4 worst=-90.5 tail100=0 /mo=13.3 | hold N=18 WR=50.0% avg=+42.2 sum=+760.0 T3R=+147.6 worst=-60.1 tail100=0 /mo=8.6
- `reverse_short_on_noisy_follow`: N=58 WR=43.1% avg=-5.6 sum=-326.6 T3R=-1271.8 worst=-267.6 tail100=13 /mo=13.3 | hold N=18 WR=50.0% avg=+9.5 sum=+171.6 T3R=-440.8 worst=-214.9 tail100=4 /mo=8.6
- `silence_only_hold`: N=19 WR=84.2% avg=+102.5 sum=+1946.7 T3R=+1001.5 worst=-33.8 tail100=0 /mo=4.8 | hold N=6 WR=100.0% avg=+134.3 sum=+806.0 T3R=+195.3 worst=12.0 tail100=0 /mo=6.0

## route_fusion
- `long_only`: N=46 WR=67.4% avg=+34.4 sum=+1580.8 T3R=+776.4 worst=-305.3 tail100=6 /mo=10.6 | hold N=14 WR=78.6% avg=+66.7 sum=+933.3 T3R=+416.2 worst=-91.5 tail100=0 /mo=14.0
- `short_current_only`: N=4 WR=75.0% avg=+17.8 sum=+71.3 T3R=-169.6 worst=-169.6 tail100=1 /mo=4.0 | hold N=2 WR=50.0% avg=-36.9 sum=-73.8 T3R=-73.8 worst=-169.6 tail100=1 /mo=2.0
- `short_1m_h4_only`: N=5 WR=20.0% avg=-86.7 sum=-433.5 T3R=-330.6 worst=-165.8 tail100=3 /mo=1.4 | hold N=2 WR=0.0% avg=-150.0 sum=-299.9 T3R=-299.9 worst=-165.8 tail100=2 /mo=2.0
- `long_plus_short_current`: N=48 WR=66.7% avg=+30.4 sum=+1460.7 T3R=+656.3 worst=-305.3 tail100=7 /mo=11.0 | hold N=15 WR=73.3% avg=+41.7 sum=+626.0 T3R=+198.3 worst=-169.6 tail100=1 /mo=15.0
- `long_plus_short_1m_h4`: N=48 WR=60.4% avg=+17.9 sum=+860.0 T3R=+55.5 worst=-305.3 tail100=9 /mo=11.0 | hold N=15 WR=60.0% avg=+8.6 sum=+128.9 T3R=-260.8 worst=-165.8 tail100=3 /mo=15.0
- `fusion_priority_short1m_h4_then_long`: N=48 WR=60.4% avg=+17.9 sum=+860.0 T3R=+55.5 worst=-305.3 tail100=9 /mo=11.0 | hold N=15 WR=60.0% avg=+8.6 sum=+128.9 T3R=-260.8 worst=-165.8 tail100=3 /mo=15.0

## adaptive_stop_long
- `sl50`: N=58 WR=39.7% avg=+15.1 sum=+875.0 T3R=-70.2 worst=-62.0 tail100=0 /mo=13.3 | hold N=18 WR=55.6% avg=+41.2 sum=+742.2 T3R=+51.5 worst=-60.9 tail100=0 /mo=8.6
- `sl75`: N=58 WR=46.6% avg=+17.7 sum=+1025.3 T3R=+80.1 worst=-89.1 tail100=0 /mo=13.3 | hold N=18 WR=61.1% avg=+44.2 sum=+795.0 T3R=+104.3 worst=-89.1 tail100=0 /mo=8.6
- `sl100`: N=58 WR=56.9% avg=+25.5 sum=+1478.5 T3R=+533.3 worst=-112.1 tail100=18 /mo=13.3 | hold N=18 WR=66.7% avg=+52.2 sum=+939.5 T3R=+248.8 worst=-109.5 tail100=4 /mo=8.6
- `sl150`: N=58 WR=62.1% avg=+30.4 sum=+1761.8 T3R=+806.2 worst=-176.7 tail100=11 /mo=13.3 | hold N=18 WR=72.2% avg=+68.2 sum=+1226.7 T3R=+438.6 worst=-165.1 tail100=2 /mo=8.6
- `sl200`: N=58 WR=65.5% avg=+35.8 sum=+2079.0 T3R=+1123.3 worst=-212.8 tail100=9 /mo=13.3 | hold N=18 WR=77.8% avg=+76.5 sum=+1376.8 T3R=+588.6 worst=-207.0 tail100=1 /mo=8.6
- `long_conf_ge5_sl75_else_sl150`: N=58 WR=50.0% avg=+24.3 sum=+1410.4 T3R=+465.2 worst=-91.5 tail100=0 /mo=13.3 | hold N=18 WR=61.1% avg=+45.8 sum=+823.6 T3R=+132.8 worst=-91.5 tail100=0 /mo=8.6
- `long_tail_detector_sl75_else_hold`: N=58 WR=55.2% avg=+27.8 sum=+1614.8 T3R=+669.6 worst=-131.5 tail100=2 /mo=13.3 | hold N=18 WR=66.7% avg=+50.2 sum=+902.8 T3R=+212.1 worst=-91.5 tail100=0 /mo=8.6

## adaptive_stop_short
- `sl50`: N=7 WR=28.6% avg=+15.7 sum=+109.8 T3R=-226.9 worst=-59.4 tail100=0 /mo=7.0 | hold N=3 WR=0.0% avg=-56.9 sum=-170.8 T3R=-170.8 worst=-59.4 tail100=0 /mo=3.0
- `sl75`: N=7 WR=57.1% avg=+35.0 sum=+244.7 T3R=-243.0 worst=-81.8 tail100=0 /mo=7.0 | hold N=3 WR=33.3% avg=-22.1 sum=-66.3 T3R=-66.3 worst=-81.8 tail100=0 /mo=3.0
- `sl100`: N=7 WR=71.4% avg=+90.4 sum=+632.8 T3R=-118.9 worst=-107.5 tail100=2 /mo=7.0 | hold N=3 WR=33.3% avg=-39.6 sum=-118.9 T3R=-118.9 worst=-107.5 tail100=2 /mo=3.0
- `sl150`: N=7 WR=71.4% avg=+76.6 sum=+536.3 T3R=-215.3 worst=-156.0 tail100=2 /mo=7.0 | hold N=3 WR=33.3% avg=-71.8 sum=-215.3 T3R=-215.3 worst=-156.0 tail100=2 /mo=3.0
- `sl200`: N=7 WR=85.7% avg=+93.9 sum=+657.4 T3R=-94.3 worst=-205.5 tail100=1 /mo=7.0 | hold N=3 WR=66.7% avg=-31.4 sum=-94.3 T3R=-94.3 worst=-205.5 tail100=1 /mo=3.0

## multi_stage
- `entry_delay_0m`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3 | hold N=18 WR=77.8% avg=+85.1 sum=+1531.5 T3R=+743.4 worst=-91.5 tail100=0 /mo=8.6
- `entry_delay_5m`: N=58 WR=67.2% avg=+39.8 sum=+2309.4 T3R=+1383.9 worst=-237.1 tail100=7 /mo=13.3 | hold N=18 WR=83.3% avg=+88.5 sum=+1592.2 T3R=+860.2 worst=-68.2 tail100=0 /mo=8.6
- `entry_delay_15m`: N=58 WR=63.8% avg=+39.4 sum=+2286.1 T3R=+1355.8 worst=-178.0 tail100=6 /mo=13.3 | hold N=18 WR=72.2% avg=+82.5 sum=+1484.5 T3R=+698.6 worst=-87.6 tail100=0 /mo=8.6
- `entry_delay_30m`: N=58 WR=60.3% avg=+32.3 sum=+1875.0 T3R=+1176.9 worst=-179.8 tail100=6 /mo=13.3 | hold N=18 WR=66.7% avg=+61.9 sum=+1113.6 T3R=+579.2 worst=-93.6 tail100=0 /mo=8.6
- `entry_delay_60m`: N=58 WR=62.1% avg=+26.9 sum=+1560.2 T3R=+1053.9 worst=-177.9 tail100=6 /mo=13.3 | hold N=18 WR=72.2% avg=+69.9 sum=+1259.1 T3R=+788.6 worst=-53.5 tail100=0 /mo=8.6
- `half_t0_half_t15`: N=58 WR=67.2% avg=+39.5 sum=+2291.8 T3R=+1348.9 worst=-218.7 tail100=7 /mo=13.3 | hold N=18 WR=83.3% avg=+83.8 sum=+1508.0 T3R=+721.0 worst=-89.6 tail100=0 /mo=8.6
- `enter_t15_only_if_pullback`: N=26 WR=57.7% avg=+21.2 sum=+551.4 T3R=-187.9 worst=-178.0 tail100=6 /mo=6.2 | hold N=8 WR=62.5% avg=+54.1 sum=+433.0 T3R=-157.7 worst=-116.6 tail100=1 /mo=3.7
- `enter_t15_after_bounce`: N=32 WR=68.8% avg=+54.2 sum=+1734.7 T3R=+970.4 worst=-85.3 tail100=0 /mo=7.4 | hold N=10 WR=70.0% avg=+77.2 sum=+771.7 T3R=+240.9 worst=-43.5 tail100=0 /mo=10.0

## position_sizing
- `flat_long_current`: N=58 WR=65.5% avg=+39.6 sum=+2297.5 T3R=+1341.9 worst=-305.3 tail100=7 /mo=13.3
- `confidence_0p5_to_1p5`: N=58 WR=65.5% avg=+57.2 sum=+3317.2 T3R=+1954.6 worst=-458.0 tail100=8 /mo=13.3
- `confidence_ge5_only`: N=35 WR=74.3% avg=+49.8 sum=+1741.4 T3R=+879.6 worst=-305.3 tail100=7 /mo=8.0
- `half_size_tail_risk`: N=58 WR=65.5% avg=+30.2 sum=+1751.1 T3R=+892.9 worst=-152.7 tail100=4 /mo=13.3
- `compound_35_current_env_1190`:
  - `start`: `35.0`
  - `notional`: `1190.0`
  - `end`: `308.4`
  - `pnl`: `273.4`
  - `min_equity`: `47.22`
- `compound_35_balanced_16p3`:
  - `start`: `35.0`
  - `notional`: `16.3`
  - `end`: `38.74`
  - `pnl`: `3.74`
  - `min_equity`: `35.17`

## feature_bins
- `state`:
  - `NOISY_EARLY_EXIT`: N=39 WR=56.4% avg=+9.0 sum=+350.8 T3R=-296.6 worst=-305.3 tail100=7 /mo=9.0
  - `TIME_EXIT`: N=19 WR=84.2% avg=+102.5 sum=+1946.7 T3R=+1001.5 worst=-33.8 tail100=0 /mo=4.8
- `session`:
  - `ASIA`: N=22 WR=54.5% avg=+31.0 sum=+682.2 T3R=-273.5 worst=-305.3 tail100=4 /mo=5.6
  - `OFF`: N=1 WR=100.0% avg=+173.7 sum=+173.7 T3R=+173.7 worst=173.7 tail100=0 /mo=1.0
  - `US`: N=35 WR=71.4% avg=+41.2 sum=+1441.6 T3R=+835.5 worst=-147.4 tail100=3 /mo=8.1
- `dow`:
  - `Fri`: N=12 WR=75.0% avg=+74.0 sum=+888.5 T3R=+138.7 worst=-147.4 tail100=1 /mo=2.9
  - `Sat`: N=8 WR=37.5% avg=-15.8 sum=-126.2 T3R=-449.7 worst=-305.3 tail100=1 /mo=2.0
  - `Sun`: N=12 WR=50.0% avg=+37.7 sum=+453.0 T3R=-181.9 worst=-131.5 tail100=2 /mo=2.8
  - `Thu`: N=15 WR=66.7% avg=+17.7 sum=+265.1 T3R=-137.4 worst=-213.9 tail100=3 /mo=3.6
  - `Tue`: N=11 WR=90.9% avg=+74.3 sum=+817.1 T3R=+303.8 worst=-80.2 tail100=0 /mo=2.8
- `sync_bucket`:
  - `sync_0_100`: N=40 WR=67.5% avg=+54.6 sum=+2183.5 T3R=+1227.8 worst=-305.3 tail100=6 /mo=9.2
  - `sync_100_200`: N=18 WR=61.1% avg=+6.3 sum=+114.0 T3R=-165.1 worst=-213.9 tail100=1 /mo=4.6
- `n2h_bucket`:
  - `n2h_0_2`: N=14 WR=50.0% avg=-4.4 sum=-61.8 T3R=-373.5 worst=-213.9 tail100=1 /mo=3.3
  - `n2h_3_4`: N=13 WR=61.5% avg=+47.8 sum=+621.9 T3R=+52.0 worst=-108.1 tail100=1 /mo=3.3
  - `n2h_5p`: N=31 WR=74.2% avg=+56.0 sum=+1737.4 T3R=+875.6 worst=-305.3 tail100=5 /mo=7.2
- `vol30_bucket`:
  - `vol_35_60`: N=22 WR=68.2% avg=+54.5 sum=+1198.2 T3R=+480.8 worst=-305.3 tail100=1 /mo=5.2
  - `vol_gt60`: N=19 WR=63.2% avg=+31.7 sum=+602.1 T3R=+13.9 worst=-213.9 tail100=3 /mo=4.9
  - `vol_le35`: N=17 WR=64.7% avg=+29.3 sum=+497.3 T3R=-211.0 worst=-137.7 tail100=3 /mo=4.5
- `echo`:
  - `echo`: N=43 WR=67.4% avg=+36.7 sum=+1576.5 T3R=+714.7 worst=-305.3 tail100=7 /mo=9.9
  - `no_echo`: N=15 WR=60.0% avg=+48.1 sum=+721.0 T3R=+88.9 worst=-91.5 tail100=0 /mo=3.5
- `prebuild`:
  - `no_prebuild`: N=13 WR=53.8% avg=+8.6 sum=+111.6 T3R=-287.7 worst=-194.2 tail100=1 /mo=3.4
  - `prebuild`: N=45 WR=68.9% avg=+48.6 sum=+2185.9 T3R=+1230.3 worst=-305.3 tail100=6 /mo=10.3

## Decision
- Decision: `NO_LIVE_CHANGE`
- Reason: Requires N>=30, WR>=70%, avg>70bps, T3R>0. If no candidate passes, keep current live unchanged.
- Promotion candidate: none.