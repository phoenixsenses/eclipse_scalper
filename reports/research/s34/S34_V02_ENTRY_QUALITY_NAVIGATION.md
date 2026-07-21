# S34 v0.2 Entry Quality Navigation

Generated: `2026-06-29T14:35:34.349898+00:00`

Status: `RESEARCH_ONLY_NO_LIVE_CHANGE`

Rule: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`

Closed filled rows: `11`

## 1. Overall Current Lifecycle Outcomes

| Horizon | Result |
| --- | --- |
| `net_30s_bps` | N=11 sum=-6.8 med=1.5 WR=0.545 T3R=-37.7 maxLoss=-19.8 |
| `net_60s_bps` | N=11 sum=-22.6 med=2.1 WR=0.545 T3R=-75.2 maxLoss=-35.8 |
| `net_5m_bps` | N=11 sum=1.7 med=11.5 WR=0.545 T3R=-97.7 maxLoss=-74.4 |
| `net_15m_bps` | N=11 sum=-25.9 med=2.1 WR=0.545 T3R=-134.0 maxLoss=-57.6 |
| `net_2h_bps` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 |

## 2. Fill Phase vs Arming

| Bucket | 2h result |
| --- | --- |
| `POST_ARM_FILL` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 |

## 3. Fill Leg

| Bucket | 2h result |
| --- | --- |
| `initial` | N=5 sum=607.3 med=81.7 WR=1.0 T3R=69.8 maxLoss=27.5 |
| `replacement` | N=6 sum=487.6 med=38.0 WR=1.0 T3R=60.0 maxLoss=13.9 |

## 4. Entry Quality Score Buckets

| Bucket | 2h result |
| --- | --- |
| `ENTRY_QUALITY_LOW` | N=10 sum=1047.9 med=62.0 WR=1.0 T3R=362.9 maxLoss=13.9 |
| `ENTRY_QUALITY_MID` | N=1 sum=47.0 med=47.0 WR=1.0 T3R=47.0 maxLoss=47.0 |

## 5. Retest Quality Score Buckets

| Bucket | 2h result |
| --- | --- |
| `RETEST_QUALITY_HIGH` | N=6 sum=648.8 med=99.2 WR=1.0 T3R=116.8 maxLoss=27.5 |
| `RETEST_QUALITY_LOW` | N=2 sum=46.1 med=23.0 WR=1.0 T3R=46.1 maxLoss=17.2 |
| `RETEST_QUALITY_MID` | N=3 sum=400.0 med=81.7 WR=1.0 T3R=400.0 maxLoss=13.9 |

## 6. Retest Depth Buckets

| Bucket | 2h result |
| --- | --- |
| `DEEP_RETEST_GE20` | N=2 sum=331.9 med=165.9 WR=1.0 T3R=331.9 maxLoss=27.5 |
| `LIGHT_RETEST_2_10` | N=4 sum=245.9 med=38.0 WR=1.0 T3R=17.2 maxLoss=17.2 |
| `MID_RETEST_10_20` | N=4 sum=503.2 med=116.6 WR=1.0 T3R=42.3 maxLoss=42.3 |
| `TOUCH_RETEST_0_2` | N=1 sum=13.9 med=13.9 WR=1.0 T3R=13.9 maxLoss=13.9 |

## 7. Fill Delay After Arming

| Bucket | 2h result |
| --- | --- |
| `FAST_0_60S` | N=2 sum=386.1 med=193.0 WR=1.0 T3R=386.1 maxLoss=81.7 |
| `LATE_GT900S` | N=2 sum=46.1 med=23.0 WR=1.0 T3R=46.1 maxLoss=17.2 |
| `NORMAL_60_300S` | N=3 sum=221.2 med=42.3 WR=1.0 T3R=221.2 maxLoss=27.5 |
| `SLOW_300_900S` | N=4 sum=441.5 med=99.9 WR=1.0 T3R=13.9 maxLoss=13.9 |

## 8. Healthy Retest

| Bucket | 2h result |
| --- | --- |
| `False` | N=4 sum=432.2 med=55.3 WR=1.0 T3R=17.2 maxLoss=17.2 |
| `True` | N=7 sum=662.7 med=47.0 WR=1.0 T3R=130.7 maxLoss=13.9 |

## 9. Panic Retest

| Bucket | 2h result |
| --- | --- |
| `True` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 |

## 10. Retest Quality Short Horizon Diagnostics

| Bucket | 60s | 15m |
| --- | --- | --- |
| `RETEST_QUALITY_HIGH` | N=6 sum=2.4 med=3.2 WR=0.667 T3R=-14.3 maxLoss=-10.8 | N=6 sum=38.5 med=4.9 WR=0.667 T3R=-19.9 maxLoss=-12.3 |
| `RETEST_QUALITY_LOW` | N=2 sum=-7.1 med=-3.5 WR=0.5 T3R=-7.1 maxLoss=-14.5 | N=2 sum=-10.9 med=-5.4 WR=0.5 T3R=-10.9 maxLoss=-31.4 |
| `RETEST_QUALITY_MID` | N=3 sum=-17.9 med=-20.0 WR=0.333 T3R=-17.9 maxLoss=-35.8 | N=3 sum=-53.5 med=-53.3 WR=0.333 T3R=-53.5 maxLoss=-57.6 |

## 11. Original Momentum-Oriented Tag Separators (2h)

| Tag | Yes | No |
| --- | --- | --- |
| `BID_DEPTH_RETAINED` | N=7 sum=591.6 med=47.0 WR=1.0 T3R=130.7 maxLoss=13.9 | N=4 sum=503.3 med=90.9 WR=1.0 T3R=17.2 maxLoss=17.2 |
| `BID_STILL_THERE` | N=8 sum=744.4 med=64.3 WR=1.0 T3R=212.4 maxLoss=13.9 | N=3 sum=350.5 med=28.9 WR=1.0 T3R=350.5 maxLoss=17.2 |
| `BID_VANISHED` | N=3 sum=350.5 med=28.9 WR=1.0 T3R=350.5 maxLoss=17.2 | N=8 sum=744.4 med=64.3 WR=1.0 T3R=212.4 maxLoss=13.9 |
| `BTC_NOT_CONFIRMING` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `POST_ARM_FILL` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `RETEST_NOT_CHASE` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `SECOND_FLUSH` | N=5 sum=473.1 med=42.3 WR=1.0 T3R=44.7 maxLoss=17.2 | N=6 sum=621.8 med=99.2 WR=1.0 T3R=89.8 maxLoss=13.9 |
| `SELL_LIQ_QUIET_15S` | N=4 sum=259.3 med=44.6 WR=1.0 T3R=17.2 maxLoss=17.2 | N=7 sum=835.6 med=81.7 WR=1.0 T3R=152.0 maxLoss=13.9 |
| `SELL_LIQ_RESTART_15S` | N=7 sum=835.6 med=81.7 WR=1.0 T3R=152.0 maxLoss=13.9 | N=4 sum=259.3 med=44.6 WR=1.0 T3R=17.2 maxLoss=17.2 |
| `SPREAD_CLEAN` | N=7 sum=942.9 med=151.4 WR=1.0 T3R=257.9 maxLoss=17.2 | N=4 sum=152.0 med=28.2 WR=1.0 T3R=13.9 maxLoss=13.9 |

## 12. Retest-Oriented Tag Separators (2h)

| Tag | Yes | No |
| --- | --- | --- |
| `BID_DEPTH_RETAINED` | N=7 sum=591.6 med=47.0 WR=1.0 T3R=130.7 maxLoss=13.9 | N=4 sum=503.3 med=90.9 WR=1.0 T3R=17.2 maxLoss=17.2 |
| `BID_STILL_THERE` | N=8 sum=744.4 med=64.3 WR=1.0 T3R=212.4 maxLoss=13.9 | N=3 sum=350.5 med=28.9 WR=1.0 T3R=350.5 maxLoss=17.2 |
| `BID_VANISHED` | N=3 sum=350.5 med=28.9 WR=1.0 T3R=350.5 maxLoss=17.2 | N=8 sum=744.4 med=64.3 WR=1.0 T3R=212.4 maxLoss=13.9 |
| `FAST_RETEST_FILL` | N=5 sum=607.3 med=81.7 WR=1.0 T3R=69.8 maxLoss=27.5 | N=6 sum=487.6 med=38.0 WR=1.0 T3R=60.0 maxLoss=13.9 |
| `LARGE_SELL_LIQ_RESTART` | N=1 sum=81.7 med=81.7 WR=1.0 T3R=81.7 maxLoss=81.7 | N=10 sum=1013.2 med=44.6 WR=1.0 T3R=328.2 maxLoss=13.9 |
| `LATE_RETEST_FILL` | N=2 sum=46.1 med=23.0 WR=1.0 T3R=46.1 maxLoss=17.2 | N=9 sum=1048.8 med=81.7 WR=1.0 T3R=363.8 maxLoss=13.9 |
| `NO_DEEP_SECOND_FLUSH` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `NO_LARGE_SELL_LIQ_RESTART` | N=10 sum=1013.2 med=44.6 WR=1.0 T3R=328.2 maxLoss=13.9 | N=1 sum=81.7 med=81.7 WR=1.0 T3R=81.7 maxLoss=81.7 |
| `POST_ARM_FILL` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `PULLBACK_FILL` | N=11 sum=1094.9 med=47.0 WR=1.0 T3R=409.9 maxLoss=13.9 | N=0 sum=0.0 med=None WR=None T3R=0.0 maxLoss=None |
| `RETEST_BAND_2_25` | N=10 sum=1081.0 med=64.3 WR=1.0 T3R=396.0 maxLoss=17.2 | N=1 sum=13.9 med=13.9 WR=1.0 T3R=13.9 maxLoss=13.9 |
| `SPREAD_CLEAN` | N=7 sum=942.9 med=151.4 WR=1.0 T3R=257.9 maxLoss=17.2 | N=4 sum=152.0 med=28.2 WR=1.0 T3R=13.9 maxLoss=13.9 |

## 13. Event Cards

- `{'event_id': 'V02:5921158:1776347534594', 'anchor_utc': '2026-04-16T13:52:14.594000+00:00', 'fill_leg': 'initial', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 37.0, 'fill_minus_arm_sec': 32.4, 'entry_vs_arm_bps': -18.0, 'entry_quality_score': 1, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 6, 'retest_quality_bucket': 'RETEST_QUALITY_MID', 'retest_depth_bucket': 'MID_RETEST_10_20', 'fill_minus_arm_bucket': 'FAST_0_60S', 'healthy_retest': False, 'panic_retest': True, 'btc_ret_15s_bps': -13.5, 'eth_ret_15s_bps': -12.3, 'sell_liq_15s_usd': 6397733.4, 'taker_imbalance_15s': -0.594, 'bid_depth_fill_vs_anchor': 0.851, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING', 'SECOND_FLUSH'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'FAST_RETEST_FILL'], 'retest_negative_tags': ['LARGE_SELL_LIQ_RESTART'], 'net_60s_bps': -35.8, 'net_15m_bps': -57.6, 'net_2h_bps': 81.7}`
- `{'event_id': 'V02:5921614:1776484476116', 'anchor_utc': '2026-04-18T03:54:36.116000+00:00', 'fill_leg': 'initial', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 151.0, 'fill_minus_arm_sec': 146.9, 'entry_vs_arm_bps': -20.8, 'entry_quality_score': 1, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 8, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'DEEP_RETEST_GE20', 'fill_minus_arm_bucket': 'NORMAL_60_300S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -5.0, 'eth_ret_15s_bps': -11.6, 'sell_liq_15s_usd': 169101.9, 'taker_imbalance_15s': -0.339, 'bid_depth_fill_vs_anchor': 0.854, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING', 'SECOND_FLUSH'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'FAST_RETEST_FILL', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': 4.2, 'net_15m_bps': 28.1, 'net_2h_bps': 27.5}`
- `{'event_id': 'V02:5922313:1776694119155', 'anchor_utc': '2026-04-20T14:08:39.155000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 1951.0, 'fill_minus_arm_sec': 1946.8, 'entry_vs_arm_bps': -8.2, 'entry_quality_score': -1, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 3, 'retest_quality_bucket': 'RETEST_QUALITY_LOW', 'retest_depth_bucket': 'LIGHT_RETEST_2_10', 'fill_minus_arm_bucket': 'LATE_GT900S', 'healthy_retest': False, 'panic_retest': True, 'btc_ret_15s_bps': -17.9, 'eth_ret_15s_bps': -25.8, 'sell_liq_15s_usd': 70340.3, 'taker_imbalance_15s': -0.345, 'bid_depth_fill_vs_anchor': 0.112, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING', 'BID_VANISHED'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': ['BID_VANISHED', 'LATE_RETEST_FILL'], 'net_60s_bps': 7.4, 'net_15m_bps': -31.4, 'net_2h_bps': 28.9}`
- `{'event_id': 'V02:5922320:1776696071191', 'anchor_utc': '2026-04-20T14:41:11.191000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 597.0, 'fill_minus_arm_sec': 592.8, 'entry_vs_arm_bps': -0.7, 'entry_quality_score': 2, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 6, 'retest_quality_bucket': 'RETEST_QUALITY_MID', 'retest_depth_bucket': 'TOUCH_RETEST_0_2', 'fill_minus_arm_bucket': 'SLOW_300_900S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -29.4, 'eth_ret_15s_bps': -36.5, 'sell_liq_15s_usd': 7192.5, 'taker_imbalance_15s': -0.215, 'bid_depth_fill_vs_anchor': 0.926, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': -20.0, 'net_15m_bps': -53.3, 'net_2h_bps': 13.9}`
- `{'event_id': 'V02:5938614:1781584271525', 'anchor_utc': '2026-06-16T04:31:11.525000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 804.0, 'fill_minus_arm_sec': 799.5, 'entry_vs_arm_bps': -6.0, 'entry_quality_score': 5, 'entry_quality_bucket': 'ENTRY_QUALITY_MID', 'retest_quality_score': 8, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'LIGHT_RETEST_2_10', 'fill_minus_arm_bucket': 'SLOW_300_900S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -2.8, 'eth_ret_15s_bps': -8.9, 'sell_liq_15s_usd': 0.0, 'taker_imbalance_15s': -0.824, 'bid_depth_fill_vs_anchor': 1.143, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'SELL_LIQ_QUIET_15S', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['BTC_NOT_CONFIRMING'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': 5.2, 'net_15m_bps': -9.7, 'net_2h_bps': 47.0}`
- `{'event_id': 'V02:5938863:1781659021753', 'anchor_utc': '2026-06-17T01:17:01.753000+00:00', 'fill_leg': 'initial', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 264.0, 'fill_minus_arm_sec': 247.2, 'entry_vs_arm_bps': -19.6, 'entry_quality_score': 4, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 9, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'MID_RETEST_10_20', 'fill_minus_arm_bucket': 'NORMAL_60_300S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -7.3, 'eth_ret_15s_bps': -13.1, 'sell_liq_15s_usd': 0.0, 'taker_imbalance_15s': -0.309, 'bid_depth_fill_vs_anchor': 1.451, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'SELL_LIQ_QUIET_15S', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['BTC_NOT_CONFIRMING', 'SECOND_FLUSH'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'FAST_RETEST_FILL', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': -10.8, 'net_15m_bps': 2.1, 'net_2h_bps': 42.3}`
- `{'event_id': 'V02:5939881:1781964531159', 'anchor_utc': '2026-06-20T14:08:51.159000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 333.0, 'fill_minus_arm_sec': 321.8, 'entry_vs_arm_bps': -2.9, 'entry_quality_score': 4, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 7, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'LIGHT_RETEST_2_10', 'fill_minus_arm_bucket': 'SLOW_300_900S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -5.6, 'eth_ret_15s_bps': -4.8, 'sell_liq_15s_usd': 0.0, 'taker_imbalance_15s': -0.149, 'bid_depth_fill_vs_anchor': 0.714, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'SELL_LIQ_QUIET_15S', 'BID_STILL_THERE', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['BTC_NOT_CONFIRMING'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'SPREAD_CLEAN', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': 2.1, 'net_15m_bps': 22.6, 'net_2h_bps': 152.8}`
- `{'event_id': 'V02:5940282:1782084822690', 'anchor_utc': '2026-06-21T23:33:42.690000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 645.0, 'fill_minus_arm_sec': 640.3, 'entry_vs_arm_bps': -10.7, 'entry_quality_score': 3, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 8, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'MID_RETEST_10_20', 'fill_minus_arm_bucket': 'SLOW_300_900S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -8.3, 'eth_ret_15s_bps': -17.2, 'sell_liq_15s_usd': 32.3, 'taker_imbalance_15s': -0.619, 'bid_depth_fill_vs_anchor': 2.49, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': 7.3, 'net_15m_bps': 7.7, 'net_2h_bps': 227.8}`
- `{'event_id': 'V02:5941473:1782442110475', 'anchor_utc': '2026-06-26T02:48:30.475000+00:00', 'fill_leg': 'initial', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 290.0, 'fill_minus_arm_sec': 273.5, 'entry_vs_arm_bps': -12.1, 'entry_quality_score': 3, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 9, 'retest_quality_bucket': 'RETEST_QUALITY_HIGH', 'retest_depth_bucket': 'MID_RETEST_10_20', 'fill_minus_arm_bucket': 'NORMAL_60_300S', 'healthy_retest': True, 'panic_retest': True, 'btc_ret_15s_bps': -3.1, 'eth_ret_15s_bps': -8.3, 'sell_liq_15s_usd': 18687.9, 'taker_imbalance_15s': -0.565, 'bid_depth_fill_vs_anchor': 1.942, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'BID_STILL_THERE', 'BID_DEPTH_RETAINED', 'SPREAD_CLEAN', 'FAST_RETEST_FILL', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': [], 'net_60s_bps': -5.6, 'net_15m_bps': -12.3, 'net_2h_bps': 151.4}`
- `{'event_id': 'V02:5941567:1782470373530', 'anchor_utc': '2026-06-26T10:39:33.530000+00:00', 'fill_leg': 'replacement', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 4285.0, 'fill_minus_arm_sec': 4276.5, 'entry_vs_arm_bps': -2.1, 'entry_quality_score': 1, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 4, 'retest_quality_bucket': 'RETEST_QUALITY_LOW', 'retest_depth_bucket': 'LIGHT_RETEST_2_10', 'fill_minus_arm_bucket': 'LATE_GT900S', 'healthy_retest': False, 'panic_retest': True, 'btc_ret_15s_bps': -12.7, 'eth_ret_15s_bps': -33.7, 'sell_liq_15s_usd': 0.0, 'taker_imbalance_15s': -0.549, 'bid_depth_fill_vs_anchor': 0.134, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'SELL_LIQ_QUIET_15S', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['BTC_NOT_CONFIRMING', 'BID_VANISHED', 'SECOND_FLUSH'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'SPREAD_CLEAN', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': ['BID_VANISHED', 'LATE_RETEST_FILL'], 'net_60s_bps': -14.5, 'net_15m_bps': 20.5, 'net_2h_bps': 17.2}`
- `{'event_id': 'V02:5941599:1782479934877', 'anchor_utc': '2026-06-26T13:18:54.877000+00:00', 'fill_leg': 'initial', 'phase': 'POST_ARM_FILL', 'fill_delay_sec': 37.0, 'fill_minus_arm_sec': 27.1, 'entry_vs_arm_bps': -20.2, 'entry_quality_score': -1, 'entry_quality_bucket': 'ENTRY_QUALITY_LOW', 'retest_quality_score': 6, 'retest_quality_bucket': 'RETEST_QUALITY_MID', 'retest_depth_bucket': 'DEEP_RETEST_GE20', 'fill_minus_arm_bucket': 'FAST_0_60S', 'healthy_retest': False, 'panic_retest': True, 'btc_ret_15s_bps': -31.6, 'eth_ret_15s_bps': -19.0, 'sell_liq_15s_usd': 97627.8, 'taker_imbalance_15s': -0.349, 'bid_depth_fill_vs_anchor': 0.154, 'spread_fill_minus_anchor_bps': 0.0, 'positive_tags': ['POST_ARM_FILL', 'SPREAD_CLEAN', 'RETEST_NOT_CHASE'], 'negative_tags': ['SELL_LIQ_RESTART_15S', 'BTC_NOT_CONFIRMING', 'BID_VANISHED', 'SECOND_FLUSH'], 'retest_positive_tags': ['POST_ARM_FILL', 'PULLBACK_FILL', 'RETEST_BAND_2_25', 'NO_DEEP_SECOND_FLUSH', 'SPREAD_CLEAN', 'FAST_RETEST_FILL', 'NO_LARGE_SELL_LIQ_RESTART'], 'retest_negative_tags': ['BID_VANISHED'], 'net_60s_bps': 37.9, 'net_15m_bps': 57.4, 'net_2h_bps': 304.4}`

## 14. Interpretation

- These are navigation labels for the current v0.2 alpha, not new entry filters.
- Because N is 11, tags are for dashboard/shadow observation only.
- A tag becomes actionable only after forward OOS confirms it on new fills.
