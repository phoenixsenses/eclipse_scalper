# S34 BUY-Side Fade Follow-Up

Generated: `2026-07-01T08:28:35.019923+00:00`

Research-only. No live/shadow runner, order logic, env, leverage, or sizing was changed.

## Baseline
`F_silence_short_h60`: N=184 WR=67.9% avg=+22.7 sum=+4183.1 T3R=+3426.3 worst=-334.3 tail100=5

## 1. Tail Anatomy
- Tail subset: N=5 WR=0.0% avg=-187.0 sum=-934.9 T3R=-541.0 worst=-334.3 tail100=5
- Non-tail subset: N=179 WR=69.8% avg=+28.6 sum=+5118.0 T3R=+4361.2 worst=-86.6 tail100=0

### Worst Tail Examples
| UTC | Net | Session | DOW | BTC4h | BTC7d | Sync | Prebuild | Echo | AskDepth |
|---|---:|---|---|---:|---:|---:|---:|---|---:|
| 2026-03-03T15:25:26.165000+00:00 | -334.3 | US | Tue | -24.4 | +467.7 | 16518 | 0 | True |  |
| 2026-04-07T21:35:39.511000+00:00 | -206.6 | OFF | Tue | +168.2 | +242.3 | 32036 | 2 | True |  |
| 2026-02-25T15:31:26.229000+00:00 | -141.5 | US | Wed | +276.2 | -23.7 | 31406 | 3 | True |  |
| 2026-06-15T14:46:52.186000+00:00 | -130.8 | US | Mon | +130.5 | +421.1 | 41726 | 1 | True | 4452 |
| 2026-04-13T15:47:30.507000+00:00 | -121.6 | US | Mon | +130.4 | +292.9 | 1009 | 0 | True | 77297 |

### Tail Splits
#### session
- `ASIA`: N=55 WR=70.9% avg=+23.8 sum=+1310.1 T3R=+828.3 worst=-69.0 tail100=0
- `OFF`: N=21 WR=76.2% avg=+23.6 sum=+496.6 T3R=+183.0 worst=-206.6 tail100=1
- `US`: N=108 WR=64.8% avg=+22.0 sum=+2376.5 T3R=+1725.1 worst=-334.3 tail100=4

#### dow
- `Fri`: N=18 WR=72.2% avg=+27.1 sum=+486.9 T3R=+132.1 worst=-39.6 tail100=0
- `Mon`: N=36 WR=58.3% avg=+9.0 sum=+325.0 T3R=+5.5 worst=-130.8 tail100=2
- `Sat`: N=18 WR=61.1% avg=+15.5 sum=+278.2 T3R=+18.4 worst=-86.6 tail100=0
- `Sun`: N=20 WR=75.0% avg=+27.6 sum=+551.9 T3R=+297.0 worst=-50.0 tail100=0
- `Thu`: N=23 WR=78.3% avg=+36.4 sum=+837.9 T3R=+233.7 worst=-85.2 tail100=0
- `Tue`: N=35 WR=65.7% avg=+13.4 sum=+468.8 T3R=+33.1 worst=-334.3 tail100=2
- `Wed`: N=34 WR=70.6% avg=+36.3 sum=+1234.3 T3R=+626.2 worst=-141.5 tail100=1

#### btc4h
- `btc4h_neg`: N=45 WR=64.4% avg=+9.7 sum=+435.7 T3R=+60.2 worst=-334.3 tail100=1
- `btc4h_pos`: N=139 WR=69.1% avg=+27.0 sum=+3747.4 T3R=+2990.6 worst=-206.6 tail100=4

#### btc7d
- `btc7d_neg`: N=84 WR=72.6% avg=+30.0 sum=+2520.7 T3R=+1849.3 worst=-141.5 tail100=1
- `btc7d_pos`: N=100 WR=64.0% avg=+16.6 sum=+1662.4 T3R=+1058.4 worst=-334.3 tail100=4

#### sync
- `sync_ge200k`: N=60 WR=76.7% avg=+32.5 sum=+1952.5 T3R=+1433.9 worst=-84.1 tail100=0
- `sync_lt200k`: N=124 WR=63.7% avg=+18.0 sum=+2230.6 T3R=+1473.8 worst=-334.3 tail100=5

#### prebuild
- `prebuild_ge2`: N=54 WR=70.4% avg=+33.4 sum=+1805.4 T3R=+1286.8 worst=-206.6 tail100=2
- `prebuild_lt2`: N=130 WR=66.9% avg=+18.3 sum=+2377.7 T3R=+1620.9 worst=-334.3 tail100=3

#### echo
- `echo_45_120`: N=77 WR=70.1% avg=+21.9 sum=+1685.3 T3R=+1040.1 worst=-334.3 tail100=5
- `no_echo`: N=107 WR=66.4% avg=+23.3 sum=+2497.8 T3R=+1902.4 worst=-86.6 tail100=0

#### ask_depth
- `ask_depth_hi`: N=64 WR=67.2% avg=+19.9 sum=+1271.3 T3R=+805.1 worst=-121.6 tail100=1
- `ask_depth_lo`: N=120 WR=68.3% avg=+24.3 sum=+2911.8 T3R=+2240.4 worst=-334.3 tail100=4

#### book_imbalance
- `imbalance_ask_or_flat`: N=142 WR=67.6% avg=+24.8 sum=+3518.1 T3R=+2761.4 worst=-334.3 tail100=3
- `imbalance_bid`: N=42 WR=69.0% avg=+15.8 sum=+664.9 T3R=+309.2 worst=-130.8 tail100=2

## 2. Stop / Time-Stop Sweep
| Variant | N | WR | Avg | Sum | T3R | Worst | TailN | SL exits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h45_sl75 | 184 | 67.9% | +24.8 | +4567.7 | +3918.1 | -80.0 | 0 | 12 |
| h45_sl50 | 184 | 66.8% | +24.6 | +4528.2 | +3878.6 | -55.0 | 0 | 26 |
| h45_sl150 | 184 | 68.5% | +24.6 | +4524.5 | +3874.9 | -155.0 | 3 | 3 |
| h45_slnone | 184 | 68.5% | +24.6 | +4521.1 | +3871.4 | -255.8 | 2 | 0 |
| h30_slnone | 184 | 69.6% | +23.9 | +4389.6 | +3791.1 | -101.6 | 1 | 0 |
| h45_sl100 | 184 | 67.9% | +23.8 | +4371.8 | +3722.2 | -105.0 | 10 | 10 |
| h30_sl150 | 184 | 69.6% | +23.3 | +4290.4 | +3691.9 | -155.0 | 2 | 1 |
| h30_sl75 | 184 | 69.6% | +22.9 | +4215.7 | +3617.2 | -80.0 | 0 | 9 |
| h30_sl50 | 184 | 68.5% | +22.6 | +4155.3 | +3556.8 | -55.0 | 0 | 18 |
| h60_sl150 | 184 | 67.9% | +23.0 | +4223.9 | +3467.2 | -155.0 | 7 | 5 |
| h60_sl100 | 184 | 67.9% | +22.9 | +4216.3 | +3459.6 | -105.0 | 12 | 12 |
| h60_sl75 | 184 | 66.8% | +22.9 | +4215.8 | +3459.0 | -80.0 | 0 | 18 |
| h30_sl100 | 184 | 69.6% | +22.0 | +4054.7 | +3456.1 | -105.0 | 8 | 8 |
| h60_slnone | 184 | 67.9% | +22.7 | +4183.1 | +3426.3 | -334.3 | 5 | 0 |
| h60_sl50 | 184 | 64.7% | +22.4 | +4115.9 | +3359.1 | -55.0 | 0 | 34 |
| h15_slnone | 184 | 69.0% | +13.6 | +2501.6 | +2128.5 | -108.8 | 1 | 0 |
| h15_sl150 | 184 | 69.0% | +13.2 | +2434.8 | +2061.8 | -155.0 | 2 | 1 |
| h15_sl75 | 184 | 69.0% | +13.2 | +2423.5 | +2050.5 | -80.0 | 0 | 7 |
| h15_sl50 | 184 | 68.5% | +13.1 | +2405.8 | +2032.8 | -55.0 | 0 | 15 |
| h15_sl100 | 184 | 69.0% | +12.5 | +2293.2 | +1920.2 | -105.0 | 6 | 6 |

## 3. Exit Shape
| Horizon | Stats |
|---|---|
| h5 | N=184 WR=62.5% avg=+5.6 sum=+1026.3 T3R=+816.6 worst=-91.2 tail100=0 |
| h10 | N=184 WR=67.4% avg=+9.4 sum=+1731.9 T3R=+1460.1 worst=-109.2 tail100=2 |
| h15 | N=184 WR=69.0% avg=+13.6 sum=+2501.6 T3R=+2128.5 worst=-108.8 tail100=1 |
| h20 | N=184 WR=76.6% avg=+19.8 sum=+3646.5 T3R=+3191.9 worst=-96.7 tail100=0 |
| h30 | N=184 WR=69.6% avg=+23.9 sum=+4389.6 T3R=+3791.1 worst=-101.6 tail100=1 |
| h45 | N=184 WR=68.5% avg=+24.6 sum=+4521.1 T3R=+3871.4 worst=-255.8 tail100=2 |
| h60 | N=184 WR=67.9% avg=+22.7 sum=+4183.1 T3R=+3426.3 worst=-334.3 tail100=5 |
| h90 | N=184 WR=63.6% avg=+18.7 sum=+3445.7 T3R=+2675.1 worst=-390.5 tail100=10 |
| h120 | N=184 WR=63.6% avg=+18.0 sum=+3307.8 T3R=+2380.4 worst=-548.1 tail100=16 |

## 4. Silence Window / Confirmation Cost
| Variant | Stats |
|---|---|
| silence10_t0_short_h60 | N=258 WR=62.4% avg=+12.3 sum=+3184.2 T3R=+2427.4 worst=-378.8 tail100=18 |
| silence10_confirmed_entry_h60 | N=258 WR=49.6% avg=-1.7 sum=-426.0 T3R=-1144.3 worst=-512.8 tail100=21 |
| silence15_t0_short_h60 | N=228 WR=64.5% avg=+17.6 sum=+4003.9 T3R=+3247.1 worst=-334.3 tail100=12 |
| silence15_confirmed_entry_h60 | N=228 WR=45.6% avg=-2.7 sum=-608.2 T3R=-1251.5 worst=-216.3 tail100=16 |
| silence20_t0_short_h60 | N=206 WR=66.5% avg=+21.6 sum=+4455.2 T3R=+3698.4 worst=-334.3 tail100=7 |
| silence20_confirmed_entry_h60 | N=206 WR=40.3% avg=-9.2 sum=-1893.5 T3R=-2498.0 worst=-262.4 tail100=16 |
| silence30_t0_short_h60 | N=184 WR=67.9% avg=+22.7 sum=+4183.1 T3R=+3426.3 worst=-334.3 tail100=5 |
| silence30_confirmed_entry_h60 | N=184 WR=39.7% avg=-10.2 sum=-1868.9 T3R=-2419.3 worst=-291.1 tail100=12 |
| silence45_t0_short_h60 | N=153 WR=70.6% avg=+23.8 sum=+3645.2 T3R=+2993.9 worst=-206.6 tail100=4 |
| silence45_confirmed_entry_h60 | N=153 WR=36.6% avg=-14.2 sum=-2179.6 T3R=-2601.1 worst=-430.3 tail100=10 |

## 5. SELL Live-Family Overlap
- Approx SELL-family event count: `109`
- `overlap_30m` overlap_rate=0.043: overlap N=8 WR=87.5% avg=+72.1 sum=+577.0 T3R=+47.2 worst=-39.6 tail100=0; no_overlap N=176 WR=67.0% avg=+20.5 sum=+3606.1 T3R=+2934.7 worst=-334.3 tail100=5
- `overlap_60m` overlap_rate=0.076: overlap N=14 WR=71.4% avg=+19.5 sum=+272.9 T3R=-256.9 worst=-334.3 tail100=1; no_overlap N=170 WR=67.6% avg=+23.0 sum=+3910.2 T3R=+3238.8 worst=-206.6 tail100=4
- `overlap_120m` overlap_rate=0.174: overlap N=32 WR=68.8% avg=+39.2 sum=+1255.7 T3R=+610.5 worst=-334.3 tail100=1; no_overlap N=152 WR=67.8% avg=+19.3 sum=+2927.4 T3R=+2297.2 worst=-206.6 tail100=4
- `overlap_240m` overlap_rate=0.245: overlap N=45 WR=66.7% avg=+31.1 sum=+1398.9 T3R=+753.7 worst=-334.3 tail100=1; no_overlap N=139 WR=68.3% avg=+20.0 sum=+2784.1 T3R=+2154.0 worst=-206.6 tail100=4

## 6. Ask-Depth / Absorption Mirror
### ask_depth
- `ask_ge100k`: N=55 WR=70.9% avg=+25.2 sum=+1386.7 T3R=+920.5 worst=-86.6 tail100=0
- `ask_ge50k`: N=9 WR=44.4% avg=-12.8 sum=-115.4 T3R=-234.6 worst=-121.6 tail100=1
- `ask_lt50k`: N=120 WR=68.3% avg=+24.3 sum=+2911.8 T3R=+2240.4 worst=-334.3 tail100=4
### imbalance
- `ask_imbalance`: N=38 WR=71.1% avg=+27.4 sum=+1041.1 T3R=+578.9 worst=-86.6 tail100=0
- `balanced`: N=108 WR=67.6% avg=+24.8 sum=+2673.7 T3R=+2002.3 worst=-334.3 tail100=3
- `bid_imbalance`: N=38 WR=65.8% avg=+12.3 sum=+468.3 T3R=+112.5 worst=-130.8 tail100=2
### spread
- `spread_le5`: N=184 WR=67.9% avg=+22.7 sum=+4183.1 T3R=+3426.3 worst=-334.3 tail100=5

## 7. Cross-Asset BUY Resonance
### sync_buckets
- `sync_200_500k`: N=31 WR=71.0% avg=+34.7 sum=+1074.6 T3R=+621.2 worst=-84.1 tail100=0
- `sync_ge500k`: N=29 WR=82.8% avg=+30.3 sum=+877.9 T3R=+503.7 worst=-43.6 tail100=0
- `sync_lt200k`: N=124 WR=63.7% avg=+18.0 sum=+2230.6 T3R=+1473.8 worst=-334.3 tail100=5
### btc_regime
- `both_nonpos`: N=24 WR=70.8% avg=+21.1 sum=+507.2 T3R=+138.4 worst=-69.0 tail100=0
- `btc4h_pos_7d_pos`: N=79 WR=65.8% avg=+21.9 sum=+1733.9 T3R=+1129.9 worst=-206.6 tail100=3
- `mixed`: N=81 WR=69.1% avg=+24.0 sum=+1942.0 T3R=+1270.6 worst=-334.3 tail100=2

## 8. Navigation Labels
| Label | Stats |
|---|---|
| BUY_CONTINUATION_DANGER | N=316 WR=39.9% avg=-26.1 sum=-8257.8 T3R=-8936.8 worst=-378.8 tail100=61 |
| BUY_SILENCE_FADE_WATCH | N=247 WR=64.4% avg=+18.4 sum=+4532.7 T3R=+3596.1 worst=-334.3 tail100=10 |
| BUY_PREBUILD2_FADE_WATCH | N=69 WR=66.7% avg=+31.1 sum=+2146.0 T3R=+1406.4 worst=-212.9 tail100=3 |
| BUY_SYNC_HIGH_TAIL_WARNING | N=209 WR=56.5% avg=-9.4 sum=-1957.4 T3R=-2487.9 worst=-378.8 tail100=32 |

## Conclusions
- Best stop/time variant by T3R is `h45_sl75`: N=184 WR=67.9% avg=+24.8 sum=+4567.7 T3R=+3918.1 worst=-80.0 tail100=0.
- Best fixed exit by T3R is `h45`: N=184 WR=68.5% avg=+24.6 sum=+4521.1 T3R=+3871.4 worst=-255.8 tail100=2.
- BUY-side fade has a real-looking 1h edge, but tails are structural enough that it should remain shadow until stop/confirmation rules are forward-tested.
- BUY continuation remains a danger/navigation label, not a long alpha.