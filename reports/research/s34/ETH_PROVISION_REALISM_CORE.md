# ETH Provision Realism Test

Generated: `2026-06-29T06:14:24.001428+00:00`

`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - no live executor, paper state, or runtime state was touched.

- ETH events: `92`
- model: `fill requires quote penetration by queue_cross_bps; one-sided losses scaled by adverse_mult`

## Pass-Like Configs

Configs with cal+hold positive sum/T3R and hold N>=40: `11`

| Config | Fill counts | Cal | Hold |
| --- | --- | --- | --- |
| `eth_provision_o2_h300s_q0_fee-0.5_adv1` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=32 sum=145.1 mean=4.5 med=5.0 T3R=130.1 WR=0.969 maxL=-9.9 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| `eth_provision_o2_h300s_q0.5_fee-0.5_adv1` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| `eth_provision_o2_h300s_q0_fee-0.5_adv1.2` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=32 sum=143.6 mean=4.5 med=5.0 T3R=128.6 WR=0.969 maxL=-11.4 | N=60 sum=92.5 mean=1.5 med=5.0 T3R=77.5 WR=0.933 maxL=-146.3 |
| `eth_provision_o2_h300s_q0.5_fee-0.5_adv1.2` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=91.4 mean=2.9 med=5.0 T3R=76.4 WR=0.906 maxL=-28.5 | N=60 sum=92.5 mean=1.5 med=5.0 T3R=77.5 WR=0.933 maxL=-146.3 |
| `eth_provision_o2_h300s_q1_fee-0.5_adv1` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=89.8 mean=1.5 med=5.0 T3R=74.8 WR=0.917 maxL=-122.3 |
| `eth_provision_o2_h300s_q0_fee0_adv1` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=32 sum=113.6 mean=3.6 med=4.0 T3R=101.6 WR=0.969 maxL=-10.4 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| `eth_provision_o2_h300s_q0.5_fee0_adv1` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| `eth_provision_o2_h300s_q1_fee-0.5_adv1.2` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=91.4 mean=2.9 med=5.0 T3R=76.4 WR=0.906 maxL=-28.5 | N=60 sum=55.3 mean=0.9 med=5.0 T3R=40.3 WR=0.917 maxL=-146.3 |
| `eth_provision_o2_h300s_q0_fee0_adv1.2` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=32 sum=112.1 mean=3.5 med=4.0 T3R=100.1 WR=0.969 maxL=-11.9 | N=60 sum=34.5 mean=0.6 med=4.0 T3R=22.5 WR=0.933 maxL=-146.8 |
| `eth_provision_o2_h300s_q0.5_fee0_adv1.2` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=60.9 mean=1.9 med=4.0 T3R=48.9 WR=0.906 maxL=-29.0 | N=60 sum=34.5 mean=0.6 med=4.0 T3R=22.5 WR=0.933 maxL=-146.8 |
| `eth_provision_o2_h300s_q1_fee0_adv1` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=32.3 mean=0.5 med=4.0 T3R=20.3 WR=0.917 maxL=-122.8 |

## Ranked Configs

| Rank | Config | Fill counts | All | Cal | Hold |
| ---: | --- | --- | --- | --- | --- |
| 1 | `eth_provision_o2_h300s_q0_fee-0.5_adv1` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=92 sum=267.2 mean=2.9 med=5.0 T3R=252.2 WR=0.946 maxL=-122.3 | N=32 sum=145.1 mean=4.5 med=5.0 T3R=130.1 WR=0.969 maxL=-9.9 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 2 | `eth_provision_o2_h300s_q0.5_fee-0.5_adv1` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=221.1 mean=2.4 med=5.0 T3R=206.1 WR=0.924 maxL=-122.3 | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=122.1 mean=2.0 med=5.0 T3R=107.1 WR=0.933 maxL=-122.3 |
| 3 | `eth_provision_o2_h300s_q0_fee-0.5_adv1.2` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=92 sum=236.2 mean=2.6 med=5.0 T3R=221.2 WR=0.946 maxL=-146.3 | N=32 sum=143.6 mean=4.5 med=5.0 T3R=128.6 WR=0.969 maxL=-11.4 | N=60 sum=92.5 mean=1.5 med=5.0 T3R=77.5 WR=0.933 maxL=-146.3 |
| 4 | `eth_provision_o2_h300s_q0.5_fee-0.5_adv1.2` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=183.9 mean=2.0 med=5.0 T3R=168.9 WR=0.924 maxL=-146.3 | N=32 sum=91.4 mean=2.9 med=5.0 T3R=76.4 WR=0.906 maxL=-28.5 | N=60 sum=92.5 mean=1.5 med=5.0 T3R=77.5 WR=0.933 maxL=-146.3 |
| 5 | `eth_provision_o2_h300s_q1_fee-0.5_adv1` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=188.9 mean=2.1 med=5.0 T3R=173.9 WR=0.913 maxL=-122.3 | N=32 sum=99.0 mean=3.1 med=5.0 T3R=84.0 WR=0.906 maxL=-24.2 | N=60 sum=89.8 mean=1.5 med=5.0 T3R=74.8 WR=0.917 maxL=-122.3 |
| 6 | `eth_provision_o2_h300s_q0_fee0_adv1` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=92 sum=177.7 mean=1.9 med=4.0 T3R=165.7 WR=0.946 maxL=-122.8 | N=32 sum=113.6 mean=3.6 med=4.0 T3R=101.6 WR=0.969 maxL=-10.4 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 7 | `eth_provision_o2_h300s_q0.5_fee0_adv1` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=132.6 mean=1.4 med=4.0 T3R=120.6 WR=0.924 maxL=-122.8 | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=64.1 mean=1.1 med=4.0 T3R=52.1 WR=0.933 maxL=-122.8 |
| 8 | `eth_provision_o2_h300s_q1_fee-0.5_adv1.2` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=146.7 mean=1.6 med=5.0 T3R=131.7 WR=0.913 maxL=-146.3 | N=32 sum=91.4 mean=2.9 med=5.0 T3R=76.4 WR=0.906 maxL=-28.5 | N=60 sum=55.3 mean=0.9 med=5.0 T3R=40.3 WR=0.917 maxL=-146.3 |
| 9 | `eth_provision_o2_h300s_q0_fee0_adv1.2` | `{'BOTH_FILLED': 87, 'BID_ONLY_LONG_INVENTORY': 4, 'ASK_ONLY_SHORT_INVENTORY': 1}` | N=92 sum=146.7 mean=1.6 med=4.0 T3R=134.7 WR=0.946 maxL=-146.8 | N=32 sum=112.1 mean=3.5 med=4.0 T3R=100.1 WR=0.969 maxL=-11.9 | N=60 sum=34.5 mean=0.6 med=4.0 T3R=22.5 WR=0.933 maxL=-146.8 |
| 10 | `eth_provision_o2_h300s_q0.5_fee0_adv1.2` | `{'BOTH_FILLED': 85, 'ASK_ONLY_SHORT_INVENTORY': 2, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=95.4 mean=1.0 med=4.0 T3R=83.4 WR=0.924 maxL=-146.8 | N=32 sum=60.9 mean=1.9 med=4.0 T3R=48.9 WR=0.906 maxL=-29.0 | N=60 sum=34.5 mean=0.6 med=4.0 T3R=22.5 WR=0.933 maxL=-146.8 |
| 11 | `eth_provision_o2_h300s_q1_fee0_adv1` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=100.9 mean=1.1 med=4.0 T3R=88.9 WR=0.913 maxL=-122.8 | N=32 sum=68.5 mean=2.1 med=4.0 T3R=56.5 WR=0.906 maxL=-24.7 | N=60 sum=32.3 mean=0.5 med=4.0 T3R=20.3 WR=0.917 maxL=-122.8 |
| 12 | `eth_provision_o2_h300s_q1_fee0_adv1.2` | `{'BOTH_FILLED': 84, 'ASK_ONLY_SHORT_INVENTORY': 3, 'BID_ONLY_LONG_INVENTORY': 5}` | N=92 sum=58.7 mean=0.6 med=4.0 T3R=46.7 WR=0.913 maxL=-146.8 | N=32 sum=60.9 mean=1.9 med=4.0 T3R=48.9 WR=0.906 maxL=-29.0 | N=60 sum=-2.2 mean=-0.0 med=4.0 T3R=-14.2 WR=0.917 maxL=-146.8 |

## Read

- The prior optimistic lead was touch-fill with maker rebate. This test requires price to cross beyond the quote.
- If only queue_cross=0 survives, the lead is mostly touch-fill/queue-priority artifact.
- If it survives queue_cross>=0.5 with fee/rebate stress and adverse_mult>=1.0, it becomes a real shadow candidate, not live.
