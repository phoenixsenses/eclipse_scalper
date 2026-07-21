# S34 V Engine Cancel/Replace

Generated: `2026-06-28T20:00:46.153387+00:00`

Protocol: `S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_V28_40_P4D`

Research-only. Tests whether unfilled O20 maker entries should be cancelled or replaced with a more aggressive limit after a fixed wait.

Events: `47`

## Ranked Configs

| Rank | Config | Fill% | Initial | Replacement | Filled | Replacement only | No-fill CF | Missed CF |
| ---: | --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| 1 | `O20_W300_O5_C1` | 46.8 | 10 | 12 | N=22 sum=1120.7 med=39.4 T3R=441.8 | N=12 sum=392.9 med=22.1 T3R=-45.9 | N=2 sum=132.4 med=66.2 T3R=132.4 | 132.4 |
| 2 | `O20_W300_O5_C2` | 46.8 | 9 | 13 | N=22 sum=1082.3 med=39.4 T3R=403.4 | N=13 sum=503.8 med=17.2 T3R=-10.3 | N=2 sum=132.4 med=66.2 T3R=132.4 | 132.4 |
| 3 | `O20_W120_O5_C1` | 46.8 | 6 | 16 | N=22 sum=1069.0 med=33.7 T3R=388.8 | N=16 sum=612.0 med=23.7 T3R=94.8 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 4 | `O20_W60_O5_C2` | 46.8 | 6 | 16 | N=22 sum=1057.7 med=33.7 T3R=377.5 | N=16 sum=600.7 med=19.5 T3R=79.8 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 5 | `O20_W120_O5_C2` | 46.8 | 6 | 16 | N=22 sum=1051.2 med=33.7 T3R=371.0 | N=16 sum=594.2 med=19.5 T3R=77.1 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 6 | `O20_W60_O5_C1` | 46.8 | 6 | 16 | N=22 sum=1035.5 med=33.7 T3R=355.3 | N=16 sum=578.5 med=23.7 T3R=55.7 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 7 | `O20_W1200_O15_C1` | 40.4 | 15 | 4 | N=19 sum=866.8 med=37.0 T3R=336.2 | N=4 sum=186.0 med=47.0 T3R=22.8 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 8 | `O20_W600_O15_C1` | 40.4 | 13 | 6 | N=19 sum=859.1 med=37.0 T3R=328.5 | N=6 sum=225.4 med=25.4 T3R=62.2 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 9 | `O20_W1200_O15_C2` | 40.4 | 15 | 4 | N=19 sum=855.4 med=37.0 T3R=327.7 | N=4 sum=186.0 med=47.0 T3R=22.8 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 10 | `O20_W30_O5_C1` | 46.8 | 3 | 19 | N=22 sum=992.2 med=30.3 T3R=325.6 | N=19 sum=960.6 med=30.2 T3R=294.0 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 11 | `O20_W300_O15_C1` | 40.4 | 10 | 9 | N=19 sum=852.4 med=37.0 T3R=321.7 | N=9 sum=124.6 med=23.6 T3R=-67.5 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 12 | `O20_W600_O5_C1` | 44.7 | 13 | 8 | N=21 sum=997.8 med=37.0 T3R=321.3 | N=8 sum=364.1 med=22.1 T3R=31.2 | N=3 sum=284.6 med=75.2 T3R=284.6 | 284.6 |
| 13 | `O20_W300_O15_C2` | 40.4 | 9 | 10 | N=19 sum=847.1 med=37.0 T3R=321.3 | N=10 sum=268.6 med=25.4 T3R=-12.4 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 14 | `O20_W600_O15_C2` | 40.4 | 12 | 7 | N=19 sum=842.7 med=37.0 T3R=315.0 | N=7 sum=74.4 med=23.6 T3R=-88.8 | N=7 sum=568.8 med=57.3 T3R=115.5 | 568.8 |
| 15 | `O20_W30_O5_C2` | 46.8 | 3 | 19 | N=22 sum=974.8 med=30.3 T3R=307.9 | N=19 sum=943.2 med=30.2 T3R=276.3 | N=1 sum=57.3 med=57.3 T3R=57.3 | 57.3 |
| 16 | `O20_W1200_O10_C2` | 40.4 | 15 | 4 | N=19 sum=835.0 med=37.0 T3R=307.3 | N=4 sum=165.6 med=42.0 T3R=18.3 | N=6 sum=536.1 med=66.2 T3R=82.9 | 536.1 |
| 17 | `O20_W60_O15_C1` | 40.4 | 6 | 13 | N=19 sum=832.7 med=35.0 T3R=307.2 | N=13 sum=375.7 med=27.2 T3R=95.0 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 18 | `O20_W120_O15_C1` | 40.4 | 6 | 13 | N=19 sum=832.7 med=35.0 T3R=307.2 | N=13 sum=375.7 med=27.2 T3R=95.0 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 19 | `O20_W60_O15_C2` | 40.4 | 6 | 13 | N=19 sum=831.7 med=35.0 T3R=306.1 | N=13 sum=374.7 med=27.2 T3R=93.9 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 20 | `O20_W120_O15_C2` | 40.4 | 6 | 13 | N=19 sum=831.7 med=35.0 T3R=306.1 | N=13 sum=374.7 med=27.2 T3R=93.9 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 21 | `O20_W30_O15_C1` | 40.4 | 3 | 16 | N=19 sum=815.5 med=35.0 T3R=302.1 | N=16 sum=783.9 med=37.3 T3R=270.5 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 22 | `O20_W30_O15_C2` | 40.4 | 3 | 16 | N=19 sum=814.4 med=35.0 T3R=301.1 | N=16 sum=782.8 med=37.3 T3R=269.5 | N=6 sum=542.2 med=66.2 T3R=89.0 | 542.2 |
| 23 | `O20_W1200_O10_C1` | 42.6 | 15 | 5 | N=20 sum=820.4 med=33.7 T3R=289.8 | N=5 sum=139.6 med=22.2 T3R=-7.7 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 24 | `O20_W1200_O5_C1` | 42.6 | 15 | 5 | N=20 sum=812.3 med=33.7 T3R=281.7 | N=5 sum=131.5 med=27.0 T3R=-9.8 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 25 | `O20_W300_O10_C2` | 40.4 | 9 | 10 | N=19 sum=802.4 med=37.0 T3R=281.6 | N=10 sum=223.9 med=20.8 T3R=-38.3 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 26 | `O20_W600_O5_C2` | 44.7 | 12 | 9 | N=21 sum=953.1 med=37.0 T3R=279.6 | N=9 sum=184.8 med=14.1 T3R=-145.0 | N=4 sum=311.2 med=66.2 T3R=26.6 | 311.2 |
| 27 | `O20_W60_O10_C2` | 40.4 | 6 | 13 | N=19 sum=800.7 med=30.5 T3R=278.2 | N=13 sum=343.6 med=22.2 T3R=79.8 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 28 | `O20_W600_O10_C2` | 40.4 | 12 | 7 | N=19 sum=805.7 med=37.0 T3R=278.0 | N=7 sum=37.4 med=19.4 T3R=-107.0 | N=6 sum=536.1 med=66.2 T3R=82.9 | 536.1 |
| 29 | `O20_W600_O10_C1` | 42.6 | 13 | 7 | N=20 sum=807.0 med=33.7 T3R=276.4 | N=7 sum=173.3 med=19.1 T3R=28.9 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 30 | `O20_W300_O10_C1` | 42.6 | 10 | 10 | N=20 sum=791.4 med=33.7 T3R=260.8 | N=10 sum=63.6 med=18.7 T3R=-112.3 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 31 | `O20_W1200_O5_C2` | 42.6 | 15 | 5 | N=20 sum=786.0 med=33.7 T3R=258.3 | N=5 sum=116.6 med=17.2 T3R=-14.5 | N=5 sum=537.0 med=75.2 T3R=83.8 | 537.0 |
| 32 | `O20_W60_O10_C1` | 42.6 | 6 | 14 | N=20 sum=779.4 med=30.5 T3R=256.9 | N=14 sum=322.3 med=20.7 T3R=58.5 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 33 | `O20_W120_O10_C2` | 40.4 | 6 | 13 | N=19 sum=774.3 med=30.5 T3R=251.8 | N=13 sum=317.2 med=22.2 T3R=53.4 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 34 | `O20_W600_CANCEL_C2` | 25.5 | 12 | 0 | N=12 sum=768.3 med=43.2 T3R=240.6 | N=0 sum=0.0 med=None T3R=0.0 | N=23 sum=-22.0 med=11.9 T3R=-527.0 | -22.0 |
| 35 | `O20_W120_O10_C1` | 42.6 | 6 | 14 | N=20 sum=753.0 med=30.5 T3R=230.5 | N=14 sum=295.9 med=20.7 T3R=32.1 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 36 | `O20_W30_O10_C2` | 40.4 | 3 | 16 | N=19 sum=724.0 med=30.5 T3R=215.2 | N=16 sum=692.4 med=33.2 T3R=183.6 | N=5 sum=509.6 med=75.2 T3R=56.4 | 509.6 |
| 37 | `O20_W300_CANCEL_C1` | 21.3 | 10 | 0 | N=10 sum=727.8 med=43.2 T3R=197.1 | N=0 sum=0.0 med=None T3R=0.0 | N=25 sum=-22.2 med=0.8 T3R=-527.2 | -22.2 |
| 38 | `O20_W30_O10_C1` | 42.6 | 3 | 17 | N=20 sum=700.2 med=30.5 T3R=191.4 | N=17 sum=668.6 med=30.5 T3R=159.8 | N=4 sum=510.5 med=113.7 T3R=57.3 | 510.5 |
| 39 | `O20_W1200_CANCEL_C1` | 31.9 | 15 | 0 | N=15 sum=680.8 med=37.0 T3R=150.2 | N=0 sum=0.0 med=None T3R=0.0 | N=15 sum=395.6 med=22.9 T3R=-109.4 | 395.6 |
| 40 | `O20_W300_CANCEL_C2` | 19.1 | 9 | 0 | N=9 sum=578.5 med=41.7 T3R=144.0 | N=0 sum=0.0 med=None T3R=0.0 | N=26 sum=122.5 med=6.4 T3R=-400.3 | 122.5 |
| 41 | `O20_W1200_CANCEL_C2` | 31.9 | 15 | 0 | N=15 sum=669.4 med=37.0 T3R=141.7 | N=0 sum=0.0 med=None T3R=0.0 | N=16 sum=422.1 med=24.7 T3R=-82.9 | 422.1 |
| 42 | `O20_W600_CANCEL_C1` | 27.7 | 13 | 0 | N=13 sum=633.8 med=41.7 T3R=103.1 | N=0 sum=0.0 med=None T3R=0.0 | N=21 sum=145.9 med=11.9 T3R=-359.1 | 145.9 |
| 43 | `O20_W60_CANCEL_C1` | 12.8 | 6 | 0 | N=6 sum=457.0 med=40.8 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=35 sum=-135.8 med=0.8 T3R=-658.6 | -135.8 |
| 44 | `O20_W120_CANCEL_C1` | 12.8 | 6 | 0 | N=6 sum=457.0 med=40.8 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=30 sum=153.4 med=8.3 T3R=-369.4 | 153.4 |
| 45 | `O20_W60_CANCEL_C2` | 12.8 | 6 | 0 | N=6 sum=457.0 med=40.8 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=35 sum=-135.8 med=0.8 T3R=-658.6 | -135.8 |
| 46 | `O20_W120_CANCEL_C2` | 12.8 | 6 | 0 | N=6 sum=457.0 med=40.8 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=31 sum=-0.9 med=4.7 T3R=-523.6 | -0.9 |
| 47 | `O20_W30_CANCEL_C1` | 6.4 | 3 | 0 | N=3 sum=31.6 med=30.4 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=39 sum=199.4 med=3.1 T3R=-445.5 | 199.4 |
| 48 | `O20_W30_CANCEL_C2` | 6.4 | 3 | 0 | N=3 sum=31.6 med=30.4 T3R=31.6 | N=0 sum=0.0 med=None T3R=0.0 | N=39 sum=199.4 med=3.1 T3R=-445.5 | 199.4 |

## Read

- Best T3R-ranked cancel/replace config: `O20_W300_O5_C1` -> N=22 sum=1120.7 med=39.4 T3R=441.8.
- Cancel-only control `O20_W30_CANCEL_C1` -> N=3 sum=31.6 med=30.4 T3R=31.6.
- Cancel-only control `O20_W30_CANCEL_C2` -> N=3 sum=31.6 med=30.4 T3R=31.6.
- A positive result here must beat fixed O20/O20 shadow after skew removal, not just increase fill count.
