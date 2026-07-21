# S34 Fast Scalp Candidate Validation

Generated: `2026-06-27T11:36:40.183981+00:00`
Lookback: `120d`

Focused validation of SELL-liq -> SHORT fast scalp candidates. Real bookTicker fills via shadow-runner helpers. No runner/config changes.

## Summary

| Candidate | Signals | Closed | No-fill | Median | Mean | WR | Cum | Top3 removed | Pos days | Med TTM | Fast5 | Exits | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | 58 | 46 | 21% | +1.0 | +0.5 | 50% | +22 | -169 | 7/19 | 1.4m | 100% | SL=9 TIME=21 TP=16 | reject_outlier_dependent |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | 107 | 86 | 20% | -10.5 | -6.6 | 36% | -572 | -765 | 5/24 | 1.4m | 100% | SL=17 TIME=51 TP=18 | reject_negative_median |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | 105 | 71 | 32% | -13.7 | -10.6 | 27% | -752 | -880 | 4/25 | 1.5m | 100% | SL=9 TIME=53 TP=9 | reject_negative_median |

## Half Split

| Candidate | Half | N | Median | WR | Cum | Top3 removed | Pos days |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | first_half | 23 | -3.9 | 43% | -66 | -189 | 3/10 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | second_half | 23 | +4.7 | 57% | +88 | -91 | 5/10 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | first_half | 43 | -6.8 | 33% | -249 | -363 | 2/15 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | second_half | 43 | -15.0 | 40% | -323 | -516 | 3/10 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | first_half | 35 | -13.2 | 23% | -408 | -501 | 3/15 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | second_half | 36 | -14.1 | 31% | -343 | -471 | 1/10 |

## Regime / Geometry Splits

| Candidate | Slice | N | Median | WR | Cum | Pos days |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | liq_count:5-15 | 19 | +14.0 | 58% | +135 | 6/10 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | liq_count:<5 | 9 | -15.0 | 33% | -118 | 3/9 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | liq_count:>=15 | 18 | +1.1 | 50% | +5 | 8/12 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | notional:<500K | 41 | +3.3 | 51% | +71 | 9/19 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | range:250-500 | 26 | +4.0 | 58% | +118 | 8/14 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | range:<250 | 12 | -4.9 | 50% | +19 | 3/8 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | range:>=500 | 8 | -32.0 | 25% | -115 | 1/5 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | single_share:50-80 | 12 | +13.0 | 58% | +53 | 5/8 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | single_share:<50 | 13 | +10.6 | 54% | +60 | 6/10 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | single_share:>=80 | 21 | -10.9 | 43% | -91 | 4/13 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | trend:bear | 39 | +3.3 | 51% | +69 | 5/15 |
| SOL_SELL_LIQ_SHORT_200K_FAST_TP40_SL30_BE40 | trend:bull | 7 | -1.2 | 43% | -47 | 2/5 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | liq_count:5-15 | 49 | -10.3 | 41% | -201 | 6/20 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | liq_count:<5 | 20 | -17.8 | 25% | -251 | 3/15 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | liq_count:>=15 | 17 | -3.9 | 35% | -120 | 5/13 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | notional:<500K | 83 | -10.7 | 36% | -531 | 5/24 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | range:250-500 | 42 | -3.0 | 45% | -66 | 6/16 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | range:<250 | 32 | -15.7 | 25% | -335 | 4/17 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | range:>=500 | 12 | -35.6 | 33% | -170 | 1/5 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | single_share:50-80 | 15 | -20.7 | 20% | -228 | 3/11 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | single_share:<50 | 33 | -3.4 | 42% | -57 | 6/18 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | single_share:>=80 | 38 | -15.0 | 37% | -287 | 6/19 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | trend:bear | 66 | -4.7 | 38% | -247 | 7/19 |
| SOL_SELL_LIQ_SHORT_100K_FAST_TP40_SL30_BE40 | trend:bull | 20 | -20.4 | 30% | -324 | 3/11 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | liq_count:5-15 | 19 | -17.3 | 11% | -295 | 2/11 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | liq_count:>=15 | 49 | -13.2 | 33% | -457 | 6/22 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | notional:>=1M | 71 | -13.7 | 27% | -752 | 4/25 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | range:250-500 | 25 | -13.8 | 32% | -172 | 5/16 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | range:<250 | 42 | -12.2 | 24% | -506 | 6/19 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | single_share:50-80 | 25 | -13.8 | 24% | -255 | 3/17 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | single_share:<50 | 33 | -12.2 | 33% | -308 | 5/15 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | single_share:>=80 | 13 | -13.7 | 15% | -189 | 2/11 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | trend:bear | 62 | -13.4 | 29% | -657 | 3/20 |
| BTC_SELL_LIQ_SHORT_1M_FAST_TP40_SL40_BE40 | trend:bull | 9 | -13.8 | 11% | -94 | 1/8 |

## Interpretation

- `candidate_for_paper_shadow` means the route is strong enough for a separately pre-registered paper/shadow bucket, not live capital.
- `watch_no_fill_high` means performance is positive but bookTicker coverage bias is too large to ignore.
- `thin` means N is too small for a route decision.
- This report does not change any live executor allow-list.
