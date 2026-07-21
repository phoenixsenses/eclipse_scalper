# S34 Exit Giveback Sweep

Generated: 2026-06-17T09:25:24.207517+00:00

Scope: ETHUSDT BUY feature factory outcomes, 450 events, Phase-1 simplified cost model.

No runner/config changes. This is descriptive + retrospective sweep only.

## Definitions

- Giveback loss: MFE reached at least 50% of route TP, but final net_bps < 0.
- Protected from SL: BE exit where the same path without BE would have reached SL.
- Missed profit to SL: SL exit after the path had at least +20 bps MFE.
- Trailing half-MFE: after +30 bps MFE, stop follows at 50% of maximum MFE.

## 1. Giveback Loss Rate

| Route | N | TP | Giveback Losses | Rate | Giveback Cum Loss | Total Negative | Total Negative Cum |
|---|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60 | 450 | 60 | 133 | 29.6% | -1210.72 | 227 | -5298.98 |
| LONG_DELAY60_TP120 | 450 | 120 | 59 | 13.1% | -544.28 | 321 | -8256.89 |
| SHORT_DELAY0_TP40_CONTROL | 450 | 40 | 57 | 12.7% | -1453.72 | 346 | -15600.70 |

## 2. BE Threshold Sweep

| Route | BE | N | Cum | Mean | Median | WR | TP/BE/SL/TIME | Protected From SL | Missed Profit To SL |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| LONG_DELAY0_TP60 | 20 | 450 | +6147.46 | +13.66 | -8.00 | 44.7% | 186/184/53/27 | 77 | 0 |
| LONG_DELAY0_TP60 | 25 | 450 | +5976.02 | +13.28 | -8.00 | 46.9% | 194/152/69/35 | 61 | 16 |
| LONG_DELAY0_TP60 | 30 | 450 | +6375.00 | +14.17 | -5.09 | 49.6% | 204/133/76/37 | 54 | 23 |
| LONG_DELAY0_TP60 | 35 | 450 | +6537.92 | +14.53 | +21.81 | 53.6% | 215/96/92/47 | 38 | 39 |
| LONG_DELAY0_TP60 | 40 | 450 | +6345.46 | +14.10 | +31.06 | 54.7% | 220/73/103/54 | 27 | 50 |
| LONG_DELAY60_TP120 | 20 | 450 | +2193.16 | +4.87 | -8.00 | 24.2% | 66/229/97/58 | 102 | 0 |
| LONG_DELAY60_TP120 | 25 | 450 | +2262.40 | +5.03 | -8.00 | 26.9% | 71/193/117/69 | 82 | 20 |
| LONG_DELAY60_TP120 | 30 | 450 | +2221.06 | +4.94 | -8.00 | 28.7% | 72/169/128/81 | 71 | 31 |
| LONG_DELAY60_TP120 | 35 | 450 | +2618.04 | +5.82 | -8.00 | 30.7% | 76/152/135/87 | 64 | 38 |
| LONG_DELAY60_TP120 | 40 | 450 | +2565.23 | +5.70 | -8.00 | 32.2% | 77/133/143/97 | 56 | 46 |
| SHORT_DELAY0_TP40_CONTROL | 20 | 450 | -12248.15 | -27.22 | -48.34 | 18.4% | 74/77/276/23 | 33 | 0 |
| SHORT_DELAY0_TP40_CONTROL | 25 | 450 | -12383.40 | -27.52 | -48.40 | 20.4% | 82/55/287/26 | 22 | 11 |
| SHORT_DELAY0_TP40_CONTROL | 30 | 450 | -12357.80 | -27.46 | -48.46 | 23.1% | 93/30/298/29 | 11 | 22 |
| SHORT_DELAY0_TP40_CONTROL | 35 | 450 | -12378.17 | -27.51 | -48.52 | 24.4% | 99/15/304/32 | 5 | 28 |
| SHORT_DELAY0_TP40_CONTROL | 40 | 450 | -12341.39 | -27.43 | -48.55 | 26.2% | 103/0/309/38 | 0 | 33 |

## 3. Partial Exit Sweep

| Route | Scenario | N | Cum | Mean | Median | WR | Delta vs Full TP60 | Delta vs Full TP120 | Better/Worse vs TP60 | Better/Worse vs TP120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LONG_DELAY0_TP60 | 50% TP60 + 50% TP120/BE | 450 | +6754.03 | +15.01 | +22.00 | 62.0% | +379.03 | -519.36 | 241/172 | 233/180 |
| LONG_DELAY60_TP120 | 50% TP60 + 50% TP120/BE | 450 | +1961.58 | +4.36 | +5.74 | 52.2% | -21.33 | -259.48 | 265/142 | 268/139 |

## 4. Trailing Half-MFE Sweep

| Route | Current Cum | Trailing Cum | Delta | Current Median | Trailing Median | Better/Worse Count | Trailing Exits |
|---|---:|---:|---:|---:|---:|---:|---|
| LONG_DELAY0_TP60 | +6228.28 | +6882.26 | +653.97 | -5.09 | +13.19 | 214/60 | {'SL': 76, 'TRAIL': 198, 'TP': 153, 'TIME': 23} |
| LONG_DELAY60_TP120 | +2016.71 | +2619.06 | +602.36 | -8.78 | +11.07 | 310/70 | {'SL': 128, 'TRAIL': 252, 'TP': 36, 'TIME': 34} |
| SHORT_DELAY0_TP40_CONTROL | -12412.91 | -11451.61 | +961.30 | -48.46 | -48.00 | 330/16 | {'TP': 77, 'SL': 298, 'TRAIL': 48, 'TIME': 27} |

## Honest Read

These are retrospective sweeps on the same sample used to inspect the issue. They are useful for identifying failure modes, not for changing the live runner directly.
