# S34 Signal Mining — cascade oncesi/ani/sonrasi

> hour17 200K baz, 126 event, 4.5 ay. Holdout 70/30 (TRAIN yon secer, TEST raporlar).
> PRE/AT=T0 entry, POST=T+5 delayed. FEE=5bps. Tarih 2026-07-01

Delta = TEST(favorable yon avg) - TEST(diger yon avg). Full = tum-veri favorable yon.

| Rank | Window | Signal | Fav | TEST N | TEST WR | TEST avg | Delta | Full N | Full avg | Full mc_p |
|--:|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| 1 | pre | agg-trade count pre-5m | hi | 6 | 83.3% | +195.6 | +150.4 | 50 | +75.9 | 0.0 |
| 2 | pre | ETH ret pre-1h | lo | 7 | 100.0% | +164.7 | +117.3 | 51 | +71.3 | 0.0 |
| 3 | pre | BTC_conc/rn | hi | 16 | 93.8% | +122.8 | +93.0 | 60 | +71.2 | 0.0 |
| 4 | pre | sync_sell/rn | hi | 20 | 85.0% | +102.6 | +71.0 | 64 | +69.7 | 0.0 |
| 5 | pre | realized vol 5m | hi | 38 | 73.7% | +69.0 | +69.0 | 82 | +59.4 | 0.0 |
| 6 | pre | BTC ret 5m | lo | 8 | 87.5% | +123.4 | +68.9 | 52 | +69.6 | 0.0 |
| 7 | pre | ETH ret pre-15m | lo | 11 | 81.8% | +115.7 | +65.7 | 55 | +67.6 | 0.0 |
| 8 | post | bid depth change 0->5m | hi | 18 | 88.9% | +95.7 | +62.1 | 36 | +57.0 | 0.002 |
| 9 | pre | 24h cascade density | hi | 33 | 75.8% | +75.4 | +48.8 | 88 | +49.3 | 0.0 |
| 10 | pre | agg-trade OFI pre-5m | hi | 22 | 81.8% | +87.4 | +43.8 | 66 | +48.7 | 0.0 |
| 11 | post | price reclaim vs anchor @T+5 bps | lo | 17 | 76.5% | +82.5 | +35.2 | 61 | +66.1 | 0.0 |
| 12 | post | BTC ret 0->5m post | lo | 17 | 76.5% | +82.5 | +35.2 | 61 | +66.1 | 0.0 |
| 13 | post | agg-trade OFI post-5m (buyers) | lo | 23 | 78.3% | +75.6 | +31.9 | 67 | +58.1 | 0.0 |
| 14 | pre | bid/ask imbalance @T0 | lo | 26 | 76.9% | +77.2 | +26.1 | 40 | +68.4 | 0.0 |
| 15 | pre | bid_qty/ask_qty @T0 | lo | 26 | 76.9% | +77.2 | +26.1 | 40 | +68.4 | 0.0 |
| 16 | pre | prebuildup 30m count | hi | 29 | 69.0% | +74.5 | +23.3 | 89 | +46.2 | 0.002 |
| 17 | post | follow-on liq 1-5m notional | hi | 20 | 70.0% | +73.3 | +21.7 | 64 | +60.1 | 0.0 |
| 18 | pre | ETH ret pre-5m | lo | 13 | 61.5% | +81.7 | +19.3 | 57 | +71.3 | 0.0 |
| 19 | at | running_accel | hi | 20 | 75.0% | +73.0 | +8.5 | 64 | +44.3 | 0.0 |
| 20 | post | book imbalance @T+5 | lo | 30 | 70.0% | +63.7 | +3.4 | 44 | +57.2 | 0.0 |
| 21 | at | max_single/rn | lo | 19 | 68.4% | +69.8 | +1.7 | 63 | +41.0 | 0.004 |
| 22 | at | cascade running_notional | hi | 22 | 72.7% | +62.5 | -15.4 | 66 | +45.4 | 0.0 |
| 23 | pre | bid depth USD @T0 | hi | 21 | 66.7% | +62.0 | -15.6 | 35 | +44.6 | 0.038 |
| 24 | at | max single liq | hi | 20 | 75.0% | +61.5 | -15.7 | 64 | +40.0 | 0.012 |
| 25 | at | liq count | hi | 11 | 72.7% | +51.3 | -24.9 | 57 | +38.7 | 0.006 |
| 26 | pre | agg-trade OFI pre-15m | lo | 18 | 61.1% | +44.6 | -46.3 | 62 | +37.9 | 0.016 |
| 27 | pre | spread_pct @T0 | lo | 0 | - | - | -69.0 | 14 | +24.0 | 0.208 |
| 28 | pre | funding_rate | lo | 0 | - | - | -69.0 | 44 | +30.3 | 0.042 |
| 29 | at | ETH drop during cascade bps | lo | 12 | 50.0% | +17.8 | -74.8 | 56 | +34.5 | 0.02 |
| 30 | at | liq/sec rate | hi | 22 | 63.6% | +32.8 | -85.9 | 66 | +49.4 | 0.0 |

---
*Script: tools/research_s34_signal_mining.py*