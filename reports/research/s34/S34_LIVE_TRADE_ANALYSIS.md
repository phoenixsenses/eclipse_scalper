# S34 Live Paper Trade Analysis

Generated: `2026-06-27T10:46:07.497139+00:00`

Analyzes actual runner paper trades to find conditions separating winners from losers.

**Total closed:** 105  **WR:** 53%  **Median:** +31.1 bps  **Cum:** +1819.2 bps

## By Rule

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 | 25 | -38.9 | -80.9 | 0.28 |
| ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 24 | 41.1 | 453.6 | 0.583 |
| SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 | 24 | 48.5 | 640.4 | 0.625 |
| ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30 | 19 | 52.3 | 743.3 | 0.789 |
| BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30 | 6 | 38.6 | 214.0 | 0.667 |
| ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60 | 2 | -21.6 | -43.3 | 0.0 |
| ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40 | 2 | -55.1 | -110.2 | 0.0 |
| ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30 | 1 | -11.4 | -11.4 | 0.0 |
| SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30 | 1 | 55.1 | 55.1 | 1.0 |
| SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30 | 1 | -41.6 | -41.6 | 0.0 |

## Winner vs Loser — Regime Conditions

Winners N=56, Losers N=49

| field | winner_median | loser_median | winner_mean | loser_mean |
| --- | --- | --- | --- | --- |
| day_trend_pct | 1.609 | 2.029 | 2.05 | 2.311 |
| day_range_pct | 3.611 | 3.445 | 4.285 | 4.367 |
| buy_liq_notional | 8158358.386 | 11893097.735 | 16404292.634 | 15565861.03 |
| agg_trade_count | 507671.5 | 523032.0 | 533958.196 | 621624.429 |
| cascade_notional | 621433.87 | 333271.359 | 1251077.191 | 730227.952 |
| liq_count | 12.5 | 6.0 | 12.964 | 8.429 |
| liq_max_notional | 325488.442 | 262845.0 | 805992.435 | 449223.274 |
| hold_sec | 250.504 | 518.006 | 636.351 | 1131.287 |
| entry_hour_utc | 13.0 | 15.0 | 12.857 | 14.633 |

## By Day Trend (%)

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| [-inf,0.0) | 10 | 21.6 | 239.5 | 0.5 |
| [0.0,1.0) | 17 | 51.7 | 540.4 | 0.706 |
| [1.0,2.0) | 29 | 10.4 | 551.4 | 0.517 |
| [2.0,4.0) | 26 | 17.9 | 369.4 | 0.5 |
| [4.0,+inf) | 23 | -10.7 | 118.4 | 0.478 |

## By Day Range (%)

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| [-inf,2.5) | 15 | 51.7 | 470.7 | 0.667 |
| [2.5,3.0) | 17 | -12.7 | -33.8 | 0.412 |
| [3.0,4.0) | 28 | 50.3 | 1014.8 | 0.607 |
| [4.0,6.0) | 20 | -46.8 | -38.2 | 0.45 |
| [6.0,+inf) | 25 | 42.0 | 405.8 | 0.52 |

## By Cascade Notional

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| [-inf,100000.0) | 8 | -26.2 | -88.4 | 0.25 |
| [100000.0,200000.0) | 10 | -47.0 | -302.2 | 0.1 |
| [1000000.0,+inf) | 32 | 50.8 | 1021.3 | 0.688 |
| [200000.0,500000.0) | 32 | 47.7 | 947.6 | 0.625 |
| [500000.0,1000000.0) | 23 | -5.1 | 240.8 | 0.478 |

## By Entry Hour (UTC)

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| [-inf,4.0) | 8 | 19.5 | 111.2 | 0.5 |
| [12.0,16.0) | 41 | 49.9 | 949.6 | 0.585 |
| [16.0,20.0) | 17 | 34.6 | 186.5 | 0.529 |
| [20.0,+inf) | 21 | -10.7 | 19.1 | 0.381 |
| [4.0,8.0) | 12 | 44.5 | 392.6 | 0.667 |
| [8.0,12.0) | 6 | 20.8 | 160.1 | 0.5 |

## By Exit Reason

| key | n | median | cum | wr |
| --- | --- | --- | --- | --- |
| TP | 51 | 56.3 | 3387.0 | 1.0 |
| SL | 26 | -51.6 | -1349.7 | 0.0 |
| BE | 19 | -10.7 | -196.0 | 0.053 |
| TIME | 9 | -12.7 | -22.1 | 0.444 |

## Top 10 Winners

- `P188` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **+125.7bps** exit=TP trend=1.05% range=3.13% cascade=9586666
- `P111` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **+116.0bps** exit=TP trend=1.48% range=3.45% cascade=55220
- `P138` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **+114.5bps** exit=TP trend=2.97% range=3.90% cascade=201207
- `P146` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **+113.2bps** exit=TP trend=4.18% range=5.12% cascade=1661065
- `P566` BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL3 **+101.4bps** exit=TP trend=-2.55% range=6.29% cascade=3169049
- `P143` ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **+100.4bps** exit=TP trend=3.70% range=4.66% cascade=452211
- `P060` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **+99.7bps** exit=TP trend=1.37% range=3.18% cascade=751665
- `P550` SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **+94.7bps** exit=TP trend=0.21% range=3.28% cascade=473378
- `P567` SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **+93.3bps** exit=TP trend=-2.95% range=8.22% cascade=420045
- `P456` ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL4 **+87.7bps** exit=TP trend=0.19% range=1.44% cascade=539703

## Bottom 10 Losers

- `P418` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **-53.4bps** exit=SL trend=2.03% range=4.26% cascade=90926
- `P056` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **-53.5bps** exit=SL trend=2.08% range=3.18% cascade=146468
- `P690` SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **-55.4bps** exit=SL trend=9.21% range=11.81% cascade=581218
- `P150` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **-55.6bps** exit=SL trend=5.37% range=6.85% cascade=76856
- `P064` ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **-56.8bps** exit=SL trend=4.01% range=4.43% cascade=296517
- `P065` ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30 **-56.8bps** exit=SL trend=4.01% range=4.43% cascade=296517
- `P114` ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **-57.2bps** exit=SL trend=2.68% range=4.24% cascade=2423558
- `P678` ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40 **-59.2bps** exit=SL trend=0.68% range=4.83% cascade=694422
- `P466` SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **-62.9bps** exit=SL trend=-0.63% range=2.89% cascade=533577
- `P349` SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30 **-63.2bps** exit=SL trend=4.04% range=4.39% cascade=690696
