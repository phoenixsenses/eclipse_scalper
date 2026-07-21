# S34 Live-Rule Behaviour Profile (monitoring / risk, NOT a new-edge search)

Generated: `2026-06-29T06:18:13.027246+00:00`  |  ETH-SELL deep-V>= 28.0bps 200K 4h fade (live rule family), cost 8.1bps

This DESCRIBES the live rule; it does not claim an edge (permutation-null already showed none).

## Overall: N=173, win 0.561, median 20.5, max_loss -410.0, losers(<-100) 32 (18.5%)

## Ripple directions -- when it fires

| Session | N | win | median |
| --- | ---: | ---: | ---: |
| ASIA | 32 | 0.562 | 17.3 |
| EUROPE | 29 | 0.517 | 11.2 |
| OFF | 20 | 0.75 | 47.3 |
| US | 92 | 0.533 | 15.0 |

| Month | N | win | median |
| --- | ---: | ---: | ---: |
| 2026-02 | 35 | 0.543 | 51.5 |
| 2026-03 | 61 | 0.59 | 40.2 |
| 2026-04 | 27 | 0.593 | 39.3 |
| 2026-06 | 50 | 0.52 | 12.3 |

## Failure geometry -- what the WORST trades (<-100) share (live risk watch-out)
- worst N=32 | winners N=97
- sync_k (concurrent cross-asset sell-liq): worst 227.6 vs winners 238.9
- depth_bps: worst 44.0 vs winners 39.6
- btc_ret_10m: worst -39.2 vs winners -48.6
- worst-trade session mode: US
