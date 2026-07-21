# S34 v0.2 Momentum Arming Live-Like Tests

Generated: `2026-06-29T14:22:21.589587+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Rule: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

Events: `11`

Anchor 2h mark outcome: N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6

## 1. Causal Arming Screens

| Config | Armed | No-arm | Delay med | Anchor->Arm | Armed anchor 2h | No-arm anchor 2h | Arm->60s | Arm->15m | Arm->2h |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| `ARM_BASE` | 11 | 0 | 23.0 | N=11 sum=-24.6 med=0.3 WR=0.545 T3R=-37.7 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-6.5 med=0.9 WR=0.545 T3R=-44.2 maxLoss=-39.2 | N=11 sum=-59.2 med=0.5 WR=0.545 T3R=-152.9 maxLoss=-59.5 | N=11 sum=920.2 med=45.2 WR=0.909 T3R=278.9 maxLoss=-64.4 |
| `ETH_BTC_UP_ONLY` | 11 | 0 | 8.0 | N=11 sum=-6.3 med=-1.4 WR=0.364 T3R=-20.7 maxLoss=-7.9 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-22.4 med=-3.0 WR=0.364 T3R=-54.6 maxLoss=-31.9 | N=11 sum=-82.4 med=2.0 WR=0.545 T3R=-179.2 maxLoss=-70.1 | N=11 sum=945.1 med=42.7 WR=0.909 T3R=299.9 maxLoss=-25.4 |
| `QUIET_ETH_BTC_UP` | 11 | 0 | 14.0 | N=11 sum=-29.4 med=0.8 WR=0.545 T3R=-44.3 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-9.8 med=-0.9 WR=0.455 T3R=-52.0 maxLoss=-39.2 | N=11 sum=-60.8 med=0.5 WR=0.545 T3R=-154.5 maxLoss=-57.9 | N=11 sum=952.4 med=45.2 WR=0.909 T3R=311.1 maxLoss=-32.1 |
| `FLOW_POSITIVE_ONLY` | 11 | 0 | 5.0 | N=11 sum=-8.7 med=-0.4 WR=0.455 T3R=-18.6 maxLoss=-7.9 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-11.8 med=-2.4 WR=0.364 T3R=-51.4 maxLoss=-39.5 | N=11 sum=-90.0 med=2.0 WR=0.545 T3R=-184.8 maxLoss=-79.8 | N=11 sum=965.9 med=42.7 WR=0.909 T3R=323.7 maxLoss=-14.1 |
| `ARM_FLOW_STRONG` | 11 | 0 | 33.0 | N=11 sum=-13.7 med=-1.0 WR=0.273 T3R=-41.5 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=12.9 med=2.3 WR=0.636 T3R=-49.8 maxLoss=-39.2 | N=11 sum=-10.6 med=-6.3 WR=0.455 T3R=-139.3 maxLoss=-41.3 | N=11 sum=900.3 med=36.8 WR=0.909 T3R=254.3 maxLoss=-82.0 |
| `ARM_BUY_250K` | 11 | 0 | 23.0 | N=11 sum=-18.7 med=0.3 WR=0.545 T3R=-37.7 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-13.0 med=0.9 WR=0.545 T3R=-50.7 maxLoss=-39.2 | N=11 sum=-66.0 med=-6.3 WR=0.455 T3R=-159.7 maxLoss=-59.5 | N=11 sum=914.2 med=45.2 WR=0.909 T3R=278.9 maxLoss=-64.4 |
| `ARM_BUY_500K` | 11 | 0 | 23.0 | N=11 sum=-11.8 med=0.8 WR=0.545 T3R=-35.2 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-20.4 med=-0.7 WR=0.455 T3R=-55.3 maxLoss=-39.2 | N=11 sum=-72.4 med=-6.4 WR=0.455 T3R=-166.1 maxLoss=-60.2 | N=11 sum=904.3 med=45.2 WR=0.909 T3R=275.4 maxLoss=-64.4 |
| `ARM_ETHBTC_2BPS` | 11 | 0 | 30.0 | N=11 sum=-5.5 med=1.0 WR=0.727 T3R=-27.4 maxLoss=-30.0 | N=11 sum=959.5 med=50.2 WR=0.909 T3R=314.5 maxLoss=-12.6 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None | N=11 sum=-7.7 med=1.2 WR=0.545 T3R=-47.2 maxLoss=-39.2 | N=11 sum=-79.3 med=0.5 WR=0.545 T3R=-159.2 maxLoss=-59.5 | N=11 sum=885.4 med=36.8 WR=0.909 T3R=250.4 maxLoss=-82.0 |

## 2. Chronological Split From Arming

60/40 chronological split. This is small-N only, but catches obvious regime concentration.

### `ARM_BASE`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-38.8 med=-1.4 WR=0.333 T3R=-40.3 maxLoss=-29.4 | N=5 sum=-52.3 med=-6.0 WR=0.0 T3R=-39.1 maxLoss=-30.6 |
| `60s` | N=6 sum=-16.5 med=1.2 WR=0.667 T3R=-42.5 maxLoss=-39.2 | N=5 sum=10.0 med=-0.9 WR=0.4 T3R=-6.7 maxLoss=-5.2 |
| `5m` | N=6 sum=-81.0 med=-7.4 WR=0.333 T3R=-104.2 maxLoss=-55.0 | N=5 sum=30.1 med=3.7 WR=0.6 T3R=-26.6 maxLoss=-23.0 |
| `15m` | N=6 sum=-27.3 med=-15.1 WR=0.333 T3R=-99.1 maxLoss=-39.7 | N=5 sum=-31.9 med=4.1 WR=0.8 T3R=-59.0 maxLoss=-59.5 |
| `2h` | N=6 sum=198.7 med=25.9 WR=1.0 T3R=44.9 maxLoss=3.3 | N=5 sum=721.5 med=155.9 WR=0.8 T3R=80.2 maxLoss=-64.4 |

### `ETH_BTC_UP_ONLY`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-41.2 med=-3.7 WR=0.0 T3R=-35.9 maxLoss=-25.3 | N=5 sum=-32.2 med=-2.3 WR=0.2 T3R=-32.7 maxLoss=-26.9 |
| `60s` | N=6 sum=-25.1 med=-2.6 WR=0.333 T3R=-46.5 maxLoss=-31.9 | N=5 sum=2.7 med=-3.0 WR=0.4 T3R=-8.0 maxLoss=-4.9 |
| `5m` | N=6 sum=-97.4 med=-3.6 WR=0.5 T3R=-121.2 maxLoss=-90.4 | N=5 sum=53.4 med=3.3 WR=0.6 T3R=-18.2 maxLoss=-14.6 |
| `15m` | N=6 sum=-71.3 med=-19.9 WR=0.333 T3R=-135.8 maxLoss=-70.1 | N=5 sum=-11.1 med=11.9 WR=0.8 T3R=-51.5 maxLoss=-53.5 |
| `2h` | N=6 sum=178.5 med=26.3 WR=1.0 T3R=39.0 maxLoss=5.7 | N=5 sum=766.6 med=155.4 WR=0.8 T3R=121.4 maxLoss=-25.4 |

### `QUIET_ETH_BTC_UP`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-35.6 med=-1.4 WR=0.333 T3R=-37.1 maxLoss=-29.4 | N=5 sum=-34.2 med=-2.2 WR=0.2 T3R=-35.6 maxLoss=-30.6 |
| `60s` | N=6 sum=-26.1 med=-1.7 WR=0.5 T3R=-51.4 maxLoss=-39.2 | N=5 sum=16.3 med=-0.9 WR=0.4 T3R=-6.7 maxLoss=-5.2 |
| `5m` | N=6 sum=-71.8 med=-7.4 WR=0.333 T3R=-95.0 maxLoss=-55.0 | N=5 sum=41.5 med=3.7 WR=0.6 T3R=-16.2 maxLoss=-12.6 |
| `15m` | N=6 sum=-30.9 med=-16.9 WR=0.333 T3R=-102.7 maxLoss=-39.7 | N=5 sum=-29.9 med=4.5 WR=0.8 T3R=-57.4 maxLoss=-57.9 |
| `2h` | N=6 sum=192.0 med=25.9 WR=1.0 T3R=38.2 maxLoss=3.3 | N=5 sum=760.4 med=155.9 WR=0.8 T3R=119.1 maxLoss=-32.1 |

### `FLOW_POSITIVE_ONLY`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-27.2 med=-1.4 WR=0.167 T3R=-28.0 maxLoss=-20.4 | N=5 sum=-34.1 med=-2.3 WR=0.2 T3R=-34.9 maxLoss=-28.7 |
| `60s` | N=6 sum=-21.7 med=-1.7 WR=0.333 T3R=-46.3 maxLoss=-39.5 | N=5 sum=9.9 med=-2.8 WR=0.4 T3R=-8.0 maxLoss=-4.9 |
| `5m` | N=6 sum=-93.0 med=-5.2 WR=0.5 T3R=-122.3 maxLoss=-85.9 | N=5 sum=48.1 med=3.3 WR=0.6 T3R=-18.9 maxLoss=-14.7 |
| `15m` | N=6 sum=-77.1 med=-20.6 WR=0.333 T3R=-138.1 maxLoss=-79.8 | N=5 sum=-12.9 med=12.8 WR=0.8 T3R=-52.2 maxLoss=-54.2 |
| `2h` | N=6 sum=191.3 med=28.4 WR=1.0 T3R=47.7 maxLoss=3.7 | N=5 sum=774.6 med=155.3 WR=0.8 T3R=132.4 maxLoss=-14.1 |

### `ARM_FLOW_STRONG`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-11.1 med=1.2 WR=0.667 T3R=-30.1 maxLoss=-29.4 | N=5 sum=-3.1 med=-6.0 WR=0.2 T3R=-13.4 maxLoss=-7.2 |
| `60s` | N=6 sum=-18.7 med=1.7 WR=0.667 T3R=-47.2 maxLoss=-39.2 | N=5 sum=31.6 med=3.7 WR=0.6 T3R=-14.7 maxLoss=-8.0 |
| `5m` | N=6 sum=-81.9 med=-8.7 WR=0.167 T3R=-98.7 maxLoss=-55.0 | N=5 sum=24.9 med=-2.2 WR=0.4 T3R=-21.1 maxLoss=-17.4 |
| `15m` | N=6 sum=-28.4 med=-14.0 WR=0.333 T3R=-96.6 maxLoss=-39.7 | N=5 sum=17.8 med=0.7 WR=0.6 T3R=-47.6 maxLoss=-41.3 |
| `2h` | N=6 sum=195.7 med=25.5 WR=1.0 T3R=50.3 maxLoss=7.6 | N=5 sum=704.6 med=155.8 WR=0.8 T3R=58.6 maxLoss=-82.0 |

### `ARM_BUY_250K`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-38.8 med=-1.4 WR=0.333 T3R=-40.3 maxLoss=-29.4 | N=5 sum=-56.3 med=-6.2 WR=0.0 T3R=-39.1 maxLoss=-30.6 |
| `60s` | N=6 sum=-16.5 med=1.2 WR=0.667 T3R=-42.5 maxLoss=-39.2 | N=5 sum=3.5 med=-0.9 WR=0.4 T3R=-13.2 maxLoss=-8.0 |
| `5m` | N=6 sum=-81.0 med=-7.4 WR=0.333 T3R=-104.2 maxLoss=-55.0 | N=5 sum=24.2 med=-2.2 WR=0.4 T3R=-26.6 maxLoss=-23.0 |
| `15m` | N=6 sum=-27.3 med=-15.1 WR=0.333 T3R=-99.1 maxLoss=-39.7 | N=5 sum=-38.7 med=4.1 WR=0.6 T3R=-65.8 maxLoss=-59.5 |
| `2h` | N=6 sum=198.7 med=25.9 WR=1.0 T3R=44.9 maxLoss=3.3 | N=5 sum=715.5 med=155.9 WR=0.8 T3R=80.2 maxLoss=-64.4 |

### `ARM_BUY_500K`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-38.5 med=-1.4 WR=0.333 T3R=-40.0 maxLoss=-29.4 | N=5 sum=-62.2 med=-10.2 WR=0.0 T3R=-41.0 maxLoss=-30.6 |
| `60s` | N=6 sum=-18.8 med=0.1 WR=0.5 T3R=-44.1 maxLoss=-39.2 | N=5 sum=-1.6 med=-0.9 WR=0.4 T3R=-15.5 maxLoss=-10.3 |
| `5m` | N=6 sum=-85.4 med=-7.4 WR=0.333 T3R=-108.6 maxLoss=-55.0 | N=5 sum=20.8 med=-3.6 WR=0.4 T3R=-28.6 maxLoss=-23.6 |
| `15m` | N=6 sum=-28.7 med=-15.8 WR=0.333 T3R=-100.5 maxLoss=-39.7 | N=5 sum=-43.7 med=4.1 WR=0.6 T3R=-70.8 maxLoss=-60.2 |
| `2h` | N=6 sum=196.7 med=25.9 WR=1.0 T3R=42.9 maxLoss=3.3 | N=5 sum=707.6 med=155.9 WR=0.8 T3R=78.7 maxLoss=-64.4 |

### `ARM_ETHBTC_2BPS`
| Horizon | Cal | Hold |
| --- | --- | --- |
| `30s` | N=6 sum=-30.8 med=-2.7 WR=0.167 T3R=-41.6 maxLoss=-29.4 | N=5 sum=-61.1 med=-8.5 WR=0.0 T3R=-44.4 maxLoss=-30.6 |
| `60s` | N=6 sum=-12.7 med=1.4 WR=0.667 T3R=-40.5 maxLoss=-39.2 | N=5 sum=5.0 med=-0.9 WR=0.4 T3R=-11.1 maxLoss=-9.6 |
| `5m` | N=6 sum=-90.1 med=-9.6 WR=0.167 T3R=-105.0 maxLoss=-55.0 | N=5 sum=16.0 med=3.7 WR=0.6 T3R=-36.4 maxLoss=-23.0 |
| `15m` | N=6 sum=-35.7 med=-15.5 WR=0.333 T3R=-99.1 maxLoss=-39.7 | N=5 sum=-43.6 med=0.7 WR=0.8 T3R=-59.0 maxLoss=-59.5 |
| `2h` | N=6 sum=187.8 med=24.6 WR=1.0 T3R=42.4 maxLoss=3.3 | N=5 sum=697.6 med=149.6 WR=0.8 T3R=62.6 maxLoss=-82.0 |

## 3. Maker Lifecycle Alternatives

Offline maker-fill simulation using the same O20/O5 lifecycle primitives. These are not live changes.

| Variant | Signals | Fill rate | Initial | Replacement | Filled summary |
| --- | ---: | ---: | ---: | ---: | --- |
| `CURRENT_O20_W300_O5` | 11 | 1.0 | 5 | 6 | N=11 sum=1087.4 med=46.8 WR=1.0 T3R=406.9 maxLoss=13.8 |
| `ARM_BASE_O5_ELSE_CURRENT` | 11 | 1.0 | 1 | 10 | N=11 sum=1028.6 med=46.8 WR=1.0 T3R=361.2 maxLoss=9.8 |
| `ARM_BASE_O0_ELSE_CURRENT` | 11 | 1.0 | 1 | 10 | N=11 sum=904.8 med=46.8 WR=0.909 T3R=268.8 maxLoss=-82.9 |
| `ARM_BASE_O5_ELSE_CANCEL300` | 11 | 1.0 | 1 | 10 | N=11 sum=1028.6 med=46.8 WR=1.0 T3R=361.2 maxLoss=9.8 |
| `ARM_BASE_CANCEL_IF_NOARM300` | 11 | 1.0 | 1 | 10 | N=11 sum=1028.6 med=46.8 WR=1.0 T3R=361.2 maxLoss=9.8 |

## 4. ARM_BASE Event Rows

- `{'event_id': 'V02:5921158:1776347534594', 'anchor_utc': '2026-04-16T13:52:14.594000+00:00', 'armed': True, 'arming_delay_sec': 71, 'eth_5s_bps': 6.8, 'btc_5s_bps': 2.7, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 7391483.7, 'taker_imbalance_5s': 0.636, 'anchor_to_arm_gross_bps': -30.0, 'anchor_to_2h_net_bps': 64.7, 'arm_to_60s_net_bps': -39.2, 'arm_to_15m_net_bps': -35.7, 'arm_to_2h_net_bps': 80.7}`
- `{'event_id': 'V02:5921614:1776484476116', 'anchor_utc': '2026-04-18T03:54:36.116000+00:00', 'armed': True, 'arming_delay_sec': 32, 'eth_5s_bps': 2.7, 'btc_5s_bps': 1.0, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 735069.3, 'taker_imbalance_5s': 0.147, 'anchor_to_arm_gross_bps': 1.1, 'anchor_to_2h_net_bps': 4.7, 'arm_to_60s_net_bps': -4.2, 'arm_to_15m_net_bps': 7.5, 'arm_to_2h_net_bps': 3.3}`
- `{'event_id': 'V02:5922313:1776694119155', 'anchor_utc': '2026-04-20T14:08:39.155000+00:00', 'armed': True, 'arming_delay_sec': 6, 'eth_5s_bps': 0.6, 'btc_5s_bps': 0.1, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 1168860.3, 'taker_imbalance_5s': 0.241, 'anchor_to_arm_gross_bps': 0.8, 'anchor_to_2h_net_bps': 50.2, 'arm_to_60s_net_bps': 18.3, 'arm_to_15m_net_bps': 70.7, 'arm_to_2h_net_bps': 45.2}`
- `{'event_id': 'V02:5922320:1776696071191', 'anchor_utc': '2026-04-20T14:41:11.191000+00:00', 'armed': True, 'arming_delay_sec': 8, 'eth_5s_bps': 3.6, 'btc_5s_bps': 2.4, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 2112563.7, 'taker_imbalance_5s': 0.585, 'anchor_to_arm_gross_bps': -0.8, 'anchor_to_2h_net_bps': 24.2, 'arm_to_60s_net_bps': 6.1, 'arm_to_15m_net_bps': -39.7, 'arm_to_2h_net_bps': 27.9}`
- `{'event_id': 'V02:5938614:1781584271525', 'anchor_utc': '2026-06-16T04:31:11.525000+00:00', 'armed': True, 'arming_delay_sec': 29, 'eth_5s_bps': 4.7, 'btc_5s_bps': 1.7, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 1662937.5, 'taker_imbalance_5s': 0.474, 'anchor_to_arm_gross_bps': -1.4, 'anchor_to_2h_net_bps': 22.9, 'arm_to_60s_net_bps': 0.9, 'arm_to_15m_net_bps': -6.4, 'arm_to_2h_net_bps': 23.8}`
- `{'event_id': 'V02:5938863:1781659021753', 'anchor_utc': '2026-06-17T01:17:01.753000+00:00', 'armed': True, 'arming_delay_sec': 30, 'eth_5s_bps': 2.4, 'btc_5s_bps': 2.1, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 282755.6, 'taker_imbalance_5s': 0.155, 'anchor_to_arm_gross_bps': 0.3, 'anchor_to_2h_net_bps': 15.6, 'arm_to_60s_net_bps': 1.6, 'arm_to_15m_net_bps': -23.7, 'arm_to_2h_net_bps': 17.8}`
- `{'event_id': 'V02:5939881:1781964531159', 'anchor_utc': '2026-06-20T14:08:51.159000+00:00', 'armed': True, 'arming_delay_sec': 14, 'eth_5s_bps': 0.7, 'btc_5s_bps': 0.3, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 1213692.3, 'taker_imbalance_5s': 0.489, 'anchor_to_arm_gross_bps': -2.8, 'anchor_to_2h_net_bps': 152.2, 'arm_to_60s_net_bps': -5.2, 'arm_to_15m_net_bps': 13.8, 'arm_to_2h_net_bps': 155.9}`
- `{'event_id': 'V02:5940282:1782084822690', 'anchor_utc': '2026-06-21T23:33:42.690000+00:00', 'armed': True, 'arming_delay_sec': 7, 'eth_5s_bps': 4.0, 'btc_5s_bps': 4.8, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 155063.4, 'taker_imbalance_5s': 0.052, 'anchor_to_arm_gross_bps': 5.7, 'anchor_to_2h_net_bps': 225.9, 'arm_to_60s_net_bps': -1.5, 'arm_to_15m_net_bps': 0.5, 'arm_to_2h_net_bps': 214.0}`
- `{'event_id': 'V02:5941473:1782442110475', 'anchor_utc': '2026-06-26T02:48:30.475000+00:00', 'armed': True, 'arming_delay_sec': 39, 'eth_5s_bps': 7.3, 'btc_5s_bps': 3.8, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 483034.8, 'taker_imbalance_5s': 0.313, 'anchor_to_arm_gross_bps': -4.9, 'anchor_to_2h_net_bps': 144.8, 'arm_to_60s_net_bps': 13.3, 'arm_to_15m_net_bps': -59.5, 'arm_to_2h_net_bps': 144.6}`
- `{'event_id': 'V02:5941567:1782470373530', 'anchor_utc': '2026-06-26T10:39:33.530000+00:00', 'armed': True, 'arming_delay_sec': 23, 'eth_5s_bps': 1.5, 'btc_5s_bps': 1.1, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 705093.3, 'taker_imbalance_5s': 0.068, 'anchor_to_arm_gross_bps': 6.0, 'anchor_to_2h_net_bps': -12.6, 'arm_to_60s_net_bps': 4.3, 'arm_to_15m_net_bps': 4.1, 'arm_to_2h_net_bps': -64.4}`
- `{'event_id': 'V02:5941599:1782479934877', 'anchor_utc': '2026-06-26T13:18:54.877000+00:00', 'armed': True, 'arming_delay_sec': 12, 'eth_5s_bps': 7.4, 'btc_5s_bps': 2.1, 'sell_liq_5s_usd': 0.0, 'taker_buy_5s_usd': 1183112.6, 'taker_imbalance_5s': 0.061, 'anchor_to_arm_gross_bps': 1.4, 'anchor_to_2h_net_bps': 266.9, 'arm_to_60s_net_bps': -0.9, 'arm_to_15m_net_bps': 9.2, 'arm_to_2h_net_bps': 271.4}`

## 5. Interpretation

- This is causal/live-like: the arming timestamp is found by scanning forward from the anchor, not by labeling a future rebound onset.
- Treat as navigation/management evidence only. N is still 11, so no live gating or order-logic change is justified by this report alone.
