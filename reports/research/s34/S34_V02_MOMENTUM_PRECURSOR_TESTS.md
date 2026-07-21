# S34 v0.2 Momentum Precursor Tests

Generated: `2026-06-29T14:15:44.955329+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Rule: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

Events: `11`

2h anchor-mark outcome: N=11 sum=959.5 med=50.2 T3R=314.5 maxLoss=-12.6

## 1. Momentum Onset Coverage

- `onset20`: `{'n': 11, 'onset_n': 10, 'coverage': 0.909, 'delay_median_sec': 85.0, 'delay_p25_sec': 51.0, 'delay_p75_sec': 154.0}`
- `onset40`: `{'n': 11, 'onset_n': 5, 'coverage': 0.455, 'delay_median_sec': 365.0, 'delay_p25_sec': 214.0, 'delay_p75_sec': 429.0}`

## 2. Winner vs Loser Pre-Onset Effects

### `onset20`
- onset N `10`, good N `9`, bad N `1`
| Feature | Good median | Bad median | Delta median | AUC good>bad |
| --- | ---: | ---: | ---: | ---: |
| `onset20_pre5_bid_notional_delta` | -18475.9 | 386542.3 | -405018.2 | 0.0 |
| `onset20_w5_0_taker_buy_notional` | 1066132.1 | 245765.9 | 820366.2 | 1.0 |
| `onset20_w2_0_eth_ret_bps` | 3.4 | 1.0 | 2.4 | 0.889 |
| `onset20_pre5_ask_notional_delta` | -17850.1 | 36912.5 | -54762.6 | 0.222 |
| `onset20_w5_0_taker_flow_imbalance` | 0.7 | 0.2 | 0.5 | 0.778 |
| `onset20_w5_0_eth_ret_bps` | 5.7 | 1.4 | 4.3 | 0.778 |
| `onset20_pre5_book_imbalance_delta` | -0.1 | -0.0 | -0.1 | 0.333 |
| `onset20_w2_0_taker_flow_imbalance` | 0.6 | 0.7 | -0.1 | 0.333 |
| `onset20_w5_0_taker_sell_notional` | 150610.4 | 166402.6 | -15792.2 | 0.444 |
| `onset20_w5_0_btc_ret_bps` | 4.9 | 1.3 | 3.6 | 0.556 |
| `onset20_anchor_to_ref_sec` | 91.5 | 79.5 | 12.0 | 0.556 |
| `onset20_pre5_spread_bps_delta` | -0.0 | -0.0 | 0.0 | 0.5 |
| `onset20_pre5_micro_minus_mid_bps_delta` | 0.0 | -0.0 | 0.0 | 0.5 |
| `onset20_w5_0_sell_liq_notional` | 0.0 | 0.0 | 0.0 | 0.5 |
| `onset20_w2_0_sell_liq_notional` | 0.0 | 0.0 | 0.0 | 0.5 |

### `onset40`
- onset N `5`, good N `5`, bad N `0`
| Feature | Good median | Bad median | Delta median | AUC good>bad |
| --- | ---: | ---: | ---: | ---: |


## 3. Simple Indicator Screens (onset20)

| Indicator | Yes N | Yes 2h | No N | No 2h |
| --- | ---: | --- | ---: | --- |
| `BID_REPLENISH_5S` | 5 | N=5 sum=220.9 med=22.9 T3R=3.0 maxLoss=-12.6 | 5 | N=5 sum=586.4 med=64.7 T3R=28.9 maxLoss=4.7 |
| `SPREAD_COMPRESS_5S` | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None | 10 | N=10 sum=807.3 med=37.2 T3R=169.7 maxLoss=-12.6 |
| `MICROPRICE_UP_5S` | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None | 10 | N=10 sum=807.3 med=37.2 T3R=169.7 maxLoss=-12.6 |
| `TAKER_BUY_DOMINANT_5S` | 8 | N=8 sum=314.5 med=23.5 T3R=54.8 maxLoss=-12.6 | 2 | N=2 sum=492.8 med=246.4 T3R=492.8 maxLoss=225.9 |
| `SELL_LIQ_QUIET_5S` | 10 | N=10 sum=807.3 med=37.2 T3R=169.7 maxLoss=-12.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |
| `BTC_UP_5S` | 10 | N=10 sum=807.3 med=37.2 T3R=169.7 maxLoss=-12.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |
| `ETH_UP_5S` | 10 | N=10 sum=807.3 med=37.2 T3R=169.7 maxLoss=-12.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |

## 4. Simple Indicator Screens (onset40)

| Indicator | Yes N | Yes 2h | No N | No 2h |
| --- | ---: | --- | ---: | --- |
| `BID_REPLENISH_5S` | 4 | N=4 sum=371.4 med=44.5 T3R=15.6 maxLoss=15.6 | 1 | N=1 sum=50.2 med=50.2 T3R=50.2 maxLoss=50.2 |
| `SPREAD_COMPRESS_5S` | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 |
| `MICROPRICE_UP_5S` | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 |
| `TAKER_BUY_DOMINANT_5S` | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |
| `SELL_LIQ_QUIET_5S` | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |
| `BTC_UP_5S` | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |
| `ETH_UP_5S` | 5 | N=5 sum=421.6 med=50.2 T3R=39.8 maxLoss=15.6 | 0 | N=0 sum=0.0 med=None T3R=None maxLoss=None |

## 5. Event Cards

Worst 10:
- `{'event_id': 'V02:5941567:1782470373530', 'anchor_utc': '2026-06-26T10:39:33.530000+00:00', 'net_2h_bps': -12.6, 'vdepth_bps': 34.4, 'bid_depth_usd': 293221.2, 'onset20_delay': 79.0, 'onset40_delay': None, 'pre5_bid_delta': 386542.3, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.193, 'w5_sell_liq': 0.0, 'w5_btc_ret': 1.3}`
- `{'event_id': 'V02:5921614:1776484476116', 'anchor_utc': '2026-04-18T03:54:36.116000+00:00', 'net_2h_bps': 4.7, 'vdepth_bps': 35.6, 'bid_depth_usd': 302670.7, 'onset20_delay': 325.0, 'onset40_delay': None, 'pre5_bid_delta': -193568.6, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.477, 'w5_sell_liq': 0.0, 'w5_btc_ret': 1.0}`
- `{'event_id': 'V02:5938863:1781659021753', 'anchor_utc': '2026-06-17T01:17:01.753000+00:00', 'net_2h_bps': 15.6, 'vdepth_bps': 28.4, 'bid_depth_usd': 139272.4, 'onset20_delay': 386.0, 'onset40_delay': 429.0, 'pre5_bid_delta': 187175.6, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.765, 'w5_sell_liq': 0.0, 'w5_btc_ret': 1.1}`
- `{'event_id': 'V02:5938614:1781584271525', 'anchor_utc': '2026-06-16T04:31:11.525000+00:00', 'net_2h_bps': 22.9, 'vdepth_bps': 28.2, 'bid_depth_usd': 185449.9, 'onset20_delay': 54.0, 'onset40_delay': None, 'pre5_bid_delta': 15638.0, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.886, 'w5_sell_liq': 0.0, 'w5_btc_ret': 0.8}`
- `{'event_id': 'V02:5922320:1776696071191', 'anchor_utc': '2026-04-20T14:41:11.191000+00:00', 'net_2h_bps': 24.2, 'vdepth_bps': 34.1, 'bid_depth_usd': 192825.1, 'onset20_delay': 154.0, 'onset40_delay': 365.0, 'pre5_bid_delta': -39599.9, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.711, 'w5_sell_liq': 0.0, 'w5_btc_ret': 5.5}`
- `{'event_id': 'V02:5922313:1776694119155', 'anchor_utc': '2026-04-20T14:08:39.155000+00:00', 'net_2h_bps': 50.2, 'vdepth_bps': 31.4, 'bid_depth_usd': 177537.4, 'onset20_delay': 44.0, 'onset40_delay': 550.0, 'pre5_bid_delta': 298316.3, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.839, 'w5_sell_liq': 0.0, 'w5_btc_ret': 4.9}`
- `{'event_id': 'V02:5921158:1776347534594', 'anchor_utc': '2026-04-16T13:52:14.594000+00:00', 'net_2h_bps': 64.7, 'vdepth_bps': 29.4, 'bid_depth_usd': 382659.3, 'onset20_delay': 195.0, 'onset40_delay': 214.0, 'pre5_bid_delta': -19765.7, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.378, 'w5_sell_liq': 0.0, 'w5_btc_ret': 9.7}`
- `{'event_id': 'V02:5941473:1782442110475', 'anchor_utc': '2026-06-26T02:48:30.475000+00:00', 'net_2h_bps': 144.8, 'vdepth_bps': 39.6, 'bid_depth_usd': 135918.4, 'onset20_delay': 91.0, 'onset40_delay': None, 'pre5_bid_delta': 190300.2, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.757, 'w5_sell_liq': 0.0, 'w5_btc_ret': 6.9}`
- `{'event_id': 'V02:5939881:1781964531159', 'anchor_utc': '2026-06-20T14:08:51.159000+00:00', 'net_2h_bps': 152.2, 'vdepth_bps': 29.2, 'bid_depth_usd': 225003.7, 'onset20_delay': None, 'onset40_delay': None, 'pre5_bid_delta': None, 'pre5_micro_delta': None, 'w5_flow_imb': None, 'w5_sell_liq': None, 'w5_btc_ret': None}`
- `{'event_id': 'V02:5940282:1782084822690', 'anchor_utc': '2026-06-21T23:33:42.690000+00:00', 'net_2h_bps': 225.9, 'vdepth_bps': 29.8, 'bid_depth_usd': 136804.1, 'onset20_delay': 20.0, 'onset40_delay': None, 'pre5_bid_delta': -62510.2, 'pre5_micro_delta': -0.0, 'w5_flow_imb': -0.038, 'w5_sell_liq': 0.0, 'w5_btc_ret': 0.9}`
Best 10:
- `{'event_id': 'V02:5941599:1782479934877', 'anchor_utc': '2026-06-26T13:18:54.877000+00:00', 'net_2h_bps': 266.9, 'vdepth_bps': 36.3, 'bid_depth_usd': 829882.1, 'onset20_delay': 51.0, 'onset40_delay': 67.0, 'pre5_bid_delta': -18475.9, 'pre5_micro_delta': -0.0, 'w5_flow_imb': -0.052, 'w5_sell_liq': 0.0, 'w5_btc_ret': 7.6}`
- `{'event_id': 'V02:5940282:1782084822690', 'anchor_utc': '2026-06-21T23:33:42.690000+00:00', 'net_2h_bps': 225.9, 'vdepth_bps': 29.8, 'bid_depth_usd': 136804.1, 'onset20_delay': 20.0, 'onset40_delay': None, 'pre5_bid_delta': -62510.2, 'pre5_micro_delta': -0.0, 'w5_flow_imb': -0.038, 'w5_sell_liq': 0.0, 'w5_btc_ret': 0.9}`
- `{'event_id': 'V02:5939881:1781964531159', 'anchor_utc': '2026-06-20T14:08:51.159000+00:00', 'net_2h_bps': 152.2, 'vdepth_bps': 29.2, 'bid_depth_usd': 225003.7, 'onset20_delay': None, 'onset40_delay': None, 'pre5_bid_delta': None, 'pre5_micro_delta': None, 'w5_flow_imb': None, 'w5_sell_liq': None, 'w5_btc_ret': None}`
- `{'event_id': 'V02:5941473:1782442110475', 'anchor_utc': '2026-06-26T02:48:30.475000+00:00', 'net_2h_bps': 144.8, 'vdepth_bps': 39.6, 'bid_depth_usd': 135918.4, 'onset20_delay': 91.0, 'onset40_delay': None, 'pre5_bid_delta': 190300.2, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.757, 'w5_sell_liq': 0.0, 'w5_btc_ret': 6.9}`
- `{'event_id': 'V02:5921158:1776347534594', 'anchor_utc': '2026-04-16T13:52:14.594000+00:00', 'net_2h_bps': 64.7, 'vdepth_bps': 29.4, 'bid_depth_usd': 382659.3, 'onset20_delay': 195.0, 'onset40_delay': 214.0, 'pre5_bid_delta': -19765.7, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.378, 'w5_sell_liq': 0.0, 'w5_btc_ret': 9.7}`
- `{'event_id': 'V02:5922313:1776694119155', 'anchor_utc': '2026-04-20T14:08:39.155000+00:00', 'net_2h_bps': 50.2, 'vdepth_bps': 31.4, 'bid_depth_usd': 177537.4, 'onset20_delay': 44.0, 'onset40_delay': 550.0, 'pre5_bid_delta': 298316.3, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.839, 'w5_sell_liq': 0.0, 'w5_btc_ret': 4.9}`
- `{'event_id': 'V02:5922320:1776696071191', 'anchor_utc': '2026-04-20T14:41:11.191000+00:00', 'net_2h_bps': 24.2, 'vdepth_bps': 34.1, 'bid_depth_usd': 192825.1, 'onset20_delay': 154.0, 'onset40_delay': 365.0, 'pre5_bid_delta': -39599.9, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.711, 'w5_sell_liq': 0.0, 'w5_btc_ret': 5.5}`
- `{'event_id': 'V02:5938614:1781584271525', 'anchor_utc': '2026-06-16T04:31:11.525000+00:00', 'net_2h_bps': 22.9, 'vdepth_bps': 28.2, 'bid_depth_usd': 185449.9, 'onset20_delay': 54.0, 'onset40_delay': None, 'pre5_bid_delta': 15638.0, 'pre5_micro_delta': 0.0, 'w5_flow_imb': 0.886, 'w5_sell_liq': 0.0, 'w5_btc_ret': 0.8}`
- `{'event_id': 'V02:5938863:1781659021753', 'anchor_utc': '2026-06-17T01:17:01.753000+00:00', 'net_2h_bps': 15.6, 'vdepth_bps': 28.4, 'bid_depth_usd': 139272.4, 'onset20_delay': 386.0, 'onset40_delay': 429.0, 'pre5_bid_delta': 187175.6, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.765, 'w5_sell_liq': 0.0, 'w5_btc_ret': 1.1}`
- `{'event_id': 'V02:5921614:1776484476116', 'anchor_utc': '2026-04-18T03:54:36.116000+00:00', 'net_2h_bps': 4.7, 'vdepth_bps': 35.6, 'bid_depth_usd': 302670.7, 'onset20_delay': 325.0, 'onset40_delay': None, 'pre5_bid_delta': -193568.6, 'pre5_micro_delta': -0.0, 'w5_flow_imb': 0.477, 'w5_sell_liq': 0.0, 'w5_btc_ret': 1.0}`
