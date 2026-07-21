# S34 V Engine State-Machine Management

Generated: `2026-06-28T21:31:14.522301+00:00`

Config: `O20_W300_O5_C1`

Research-only. Combines stop-tighten and winner-extension overlays; no live/paper state changed.

Baseline H2: N=22 sum=1120.7 med=39.4 T3R=441.8 max_loss=-144.4

## State Counts

- Rows: `22`
- 5m danger triggers: `6`
- 30m recovery triggers: `18`
- Danger/recovery overlap: `3`

## Variants

| Rank | Variant | Actions | Summary | Delta sum | Delta T3R | Delta max loss |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | `state_machine_recovery_priority` | `baseline_h2:1, danger_stop_tighten:3, recovery_extend_h4:18` | N=22 sum=1937.9 med=57.4 T3R=1018.5 max_loss=-102.9 | 817.2 | 576.7 | 41.5 |
| 2 | `winner_extension_only_30m_anchor_and_btc_h4` | `recovery_extend_h4:18, baseline_h2:4` | N=22 sum=1896.4 med=57.4 T3R=977.0 max_loss=-144.4 | 775.7 | 535.2 | 0.0 |
| 3 | `state_machine_danger_priority` | `baseline_h2:1, danger_stop_tighten:6, recovery_extend_h4:15` | N=22 sum=1754.4 med=57.4 T3R=939.1 max_loss=-102.9 | 633.7 | 497.3 | 41.5 |
| 4 | `stop_only_5m_no_reclaim_btc_down_trigger_sl80` | `danger_stop_tighten:6, baseline_h2:16` | N=22 sum=1162.3 med=39.4 T3R=483.4 max_loss=-102.9 | 41.6 | 41.6 | 41.5 |

## Read

- Best combined path by T3R: `state_machine_recovery_priority` -> N=22 sum=1937.9 med=57.4 T3R=1018.5 max_loss=-102.9.
- If combined state-machine underperforms winner-extension-only, keep stop-tighten as a separate safety shadow rather than coupling it into the exit engine.
