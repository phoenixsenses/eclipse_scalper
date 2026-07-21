# S34 Maker Absorption Permission

Generated: `2026-06-28T23:17:42.806893+00:00`

Research-only. Applies T=0 book absorption to current maker lifecycle and state-machine outcomes. No live/paper state changed.

Config: `O20_W300_O5_C1`
Eligible: `47`; book-covered eligible: `22`; filled: `22`

Baseline filled H2: N=22 sum=1120.7 med=39.4 T3R=441.8 max_loss=-144.4

## Cuts

- `imbalance_med`: `0.1`
- `bid_depth_med`: `135423.8`
- `imbalance_p25`: `-0.4`
- `bid_depth_p25`: `74885.0`

## Permission Gates On Maker Lifecycle

| Gate | Eligible | Filled | Fill% | Filled summary | No-fill anchor CF |
| --- | ---: | ---: | ---: | --- | --- |
| `imbalance_gate=bid_support` | 11 | 11 | 100.0 | N=11 sum=742.6 med=27.0 T3R=140.8 max_loss=-35.9 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `imbalance_gate=ask_heavy` | 11 | 11 | 100.0 | N=11 sum=378.1 med=41.7 T3R=38.4 max_loss=-144.4 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `imbalance_gate=no_book` | 25 | 0 | 0.0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=25 sum=-821.3 med=-28.5 T3R=-1166.1 max_loss=-272.8 |
| `bid_depth_gate=deep_bid` | 11 | 11 | 100.0 | N=11 sum=1081.4 med=46.3 T3R=402.5 max_loss=13.3 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `bid_depth_gate=shallow_bid` | 11 | 11 | 100.0 | N=11 sum=39.4 med=30.4 T3R=-126.0 max_loss=-144.4 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `bid_depth_gate=no_book` | 25 | 0 | 0.0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=25 sum=-821.3 med=-28.5 T3R=-1166.1 max_loss=-272.8 |
| `absorption_gate=absorbed` | 9 | 9 | 100.0 | N=9 sum=812.7 med=46.3 T3R=210.9 max_loss=13.3 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `absorption_gate=mixed` | 8 | 8 | 100.0 | N=8 sum=164.4 med=23.1 T3R=-168.4 max_loss=-144.4 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `absorption_gate=vacuum_like` | 5 | 5 | 100.0 | N=5 sum=143.6 med=37.0 T3R=2.4 max_loss=-28.0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None |
| `absorption_gate=no_book` | 25 | 0 | 0.0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=25 sum=-821.3 med=-28.5 T3R=-1166.1 max_loss=-272.8 |

## Permission + State Machine

| Gate | Filled | Baseline H2 | State machine | Actions |
| --- | ---: | --- | --- | --- |
| `imbalance_gate=bid_support` | 11 | N=11 sum=742.6 med=27.0 T3R=140.8 max_loss=-35.9 | N=11 sum=1552.4 med=130.1 T3R=633.0 max_loss=-35.9 | `baseline_h2:1, danger_stop_tighten:1, recovery_extend_h4:9` |
| `imbalance_gate=ask_heavy` | 11 | N=11 sum=378.1 med=41.7 T3R=38.4 max_loss=-144.4 | N=11 sum=385.5 med=37.1 T3R=12.1 max_loss=-102.9 | `danger_stop_tighten:2, recovery_extend_h4:9` |
| `imbalance_gate=no_book` | 0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | `` |
| `bid_depth_gate=deep_bid` | 11 | N=11 sum=1081.4 med=46.3 T3R=402.5 max_loss=13.3 | N=11 sum=1791.0 med=164.6 T3R=871.6 max_loss=6.0 | `danger_stop_tighten:1, recovery_extend_h4:10` |
| `bid_depth_gate=shallow_bid` | 11 | N=11 sum=39.4 med=30.4 T3R=-126.0 max_loss=-144.4 | N=11 sum=146.9 med=15.2 T3R=-114.3 max_loss=-102.9 | `baseline_h2:1, danger_stop_tighten:2, recovery_extend_h4:8` |
| `bid_depth_gate=no_book` | 0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | `` |
| `absorption_gate=absorbed` | 9 | N=9 sum=812.7 med=46.3 T3R=210.9 max_loss=13.3 | N=9 sum=1586.0 med=164.6 T3R=666.6 max_loss=6.0 | `danger_stop_tighten:1, recovery_extend_h4:8` |
| `absorption_gate=mixed` | 8 | N=8 sum=164.4 med=23.1 T3R=-168.4 max_loss=-144.4 | N=8 sum=142.9 med=18.6 T3R=-99.2 max_loss=-102.9 | `baseline_h2:1, danger_stop_tighten:1, recovery_extend_h4:6` |
| `absorption_gate=vacuum_like` | 5 | N=5 sum=143.6 med=37.0 T3R=2.4 max_loss=-28.0 | N=5 sum=209.0 med=55.6 T3R=-52.2 max_loss=-28.0 | `danger_stop_tighten:1, recovery_extend_h4:4` |
| `absorption_gate=no_book` | 0 | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | N=0 sum=0.0 med=None T3R=0.0 max_loss=None | `` |

## Read

- Maker filled bid_support vs ask_heavy delta T3R: `102.4`; delta max_loss `108.5`.
- Bid_support + state-machine: N=11 sum=1552.4 med=130.1 T3R=633.0 max_loss=-35.9.
- A live-adjacent permission gate must improve filled P&L without just deleting the fills that carry expectancy.
