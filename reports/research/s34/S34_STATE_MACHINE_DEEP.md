# S34 State Machine Deep Research

Generated: `2026-06-30T15:23:54.722314+00:00`  |  Fee: 5.0 bps

## Q1. Stop Loss for SHORT

| SL bps | N | WR | Mean | T3R | Trigger% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 34 | 64.7% | +72.9 | +1394 | 32.4% |
| 75 | 34 | 70.6% | +81.2 | +1676 | 17.6% |
| 100 | 34 | 73.5% | +80.5 | +1652 | 14.7% |
| 125 | 34 | 73.5% | +76.9 | +1527 | 14.7% |
| 150 | 34 | 76.5% | +84.9 | +1802 | 2.9% |
| 175 | 34 | 76.5% | +85.9 | +1836 | 0.0% |
| 200 | 34 | 76.5% | +85.9 | +1836 | 0.0% |
| 250 | 34 | 76.5% | +85.9 | +1836 | 0.0% |

## Q2. BTC Threshold Sensitivity

| BTC Thr | N all | N hold | WR hold | Mean hold | T3R hold |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200000 | 124 | 38 | 65.8% | +67.0 | +1462 |
| 300000 | 88 | 27 | 63.0% | +67.9 | +776 |
| 500000 | 68 | 21 | 66.7% | +72.3 | +462 |
| 750000 | 49 | 15 | 73.3% | +102.4 | +480 |
| 1000000 | 39 | 12 | 75.0% | +107.2 | +229 |

## Q3. BTC Entry Choice (First / Largest / Last)

| Entry | N | N hold | Avg Slippage | Hold WR | Hold Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| first | 68 | 34 | -55.7bps | 52.9% | +30.2 |
| largest | 68 | 34 | -72.9bps | 44.1% | +3.1 |
| last | 68 | 34 | -84.2bps | 38.2% | -16.8 |

## Q4. ETH Notional Sweet Spot

| Bucket | N sil | WR sil hold | N nei | WR nei hold |
| --- | ---: | ---: | ---: | ---: |
| 200K-500K | 144 | 71.4% | 57 | 76.9% |
| 500K-1M | 17 | 100.0% | 4 | 66.7% |
| 1M-2M | 3 | 0.0% | 2 | 100.0% |
| 2M+ | 3 | 100.0% | 5 | 75.0% |

## Q5. n2h (Sequential Cascade) Effect on NEITHER

| n2h >= | N NEITHER all | WR hold | Mean hold | T3R hold |
| ---: | ---: | ---: | ---: | ---: |
| >=0 | 68 | 76.5% | +85.9 | +1836 |
| >=1 | 62 | 78.1% | +86.8 | +1693 |
| >=2 | 55 | 82.8% | +99.4 | +1797 |
| >=3 | 46 | 80.8% | +106.3 | +1678 |
| >=4 | 39 | 87.0% | +121.9 | +1719 |
| >=5 | 35 | 90.5% | +117.3 | +1511 |

## Q6. Silence Window Sensitivity

| Window | N sil | WR sil hold | N nei | WR nei hold |
| --- | ---: | ---: | ---: | ---: |
| 15min | 219 | 60.6% | 51 | 62.5% |
| 20min | 203 | 60.7% | 59 | 72.2% |
| 30min | 167 | 70.6% | 68 | 66.7% |
| 45min | 137 | 73.8% | 83 | 68.0% |
| 60min | 120 | 77.8% | 86 | 65.4% |

## Q7. Session Breakdown

| Session | N sil | WR sil hold | N nei | WR nei hold |
| --- | ---: | ---: | ---: | ---: |
| ASIA | 44 | 92.3% | 10 | 66.7% |
| EUROPE | 39 | 44.4% | 10 | 75.0% |
| US | 61 | 81.8% | 42 | 82.6% |
| OFF | 23 | 60.0% | 6 | 50.0% |

## Q8. BTC Regime Effect

| Condition | N all | N hold | WR hold | Mean | T3R |
| --- | ---: | ---: | ---: | ---: | ---: |
| btc_up | 7 | 3 | 66.7% | +79.8 | +239 |
| btc_down | 61 | 31 | 77.4% | +86.5 | +1597 |
| score3_btc_up | 4 | 2 | 100.0% | +126.0 | +252 |
| score3_btc_dn | 47 | 26 | 80.8% | +104.0 | +1618 |

## Q9. P&L Distribution and Max Drawdown (Holdout Portfolio)

Portfolio: SILENCE LONG + NEITHER(score≥3) SHORT  |  N=66
Sum=+5811bps  WR=77.3%  Mean=+88.0  Median=+54.8

**Max drawdown: 554 bps**

| Percentile | bps |
| --- | ---: |
| p5 | -116.8 |
| p10 | -50.1 |
| p25 | +7.0 |
| p75 | +168.7 |
| p90 | +292.6 |

Worst consecutive loss streaks:

- 2 trades: -554 bps
- 3 trades: -168 bps
- 1 trades: -118 bps
- 1 trades: -118 bps
- 1 trades: -117 bps

## Q10. NOISY Recovery

| Filter | N all | N hold | WR hold | Mean hold | T3R hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 185 | 58 | 48.3% | +9.5 | -308 |
| score>=2 | 154 | 49 | 49.0% | +12.9 | -226 |
| score>=3 | 100 | 32 | 40.6% | +2.5 | -629 |
| score>=4 | 46 | 19 | 42.1% | +4.6 | -490 |
| vd>=30 | 64 | 17 | 58.8% | +19.1 | -126 |
| btc4h<0 | 140 | 44 | 40.9% | -0.7 | -738 |
| US_session | 96 | 28 | 46.4% | +13.6 | -197 |
| score>=3+btc4h<0 | 87 | 28 | 32.1% | -6.3 | -885 |
| score>=3+US | 72 | 24 | 41.7% | +9.8 | -343 |
| vd>=30+btc4h<0 | 48 | 11 | 36.4% | -19.7 | -328 |
