# S34 State Machine v0.1 Shadow Protocol

Status: `EXPLORATORY_FROZEN`

Protocol id: `S34_STATE_MACHINE_V0_1_ETH_SELL_MAKER_LONG_RECOVERY_PRIORITY`

Created: 2026-06-29

## Purpose

Freeze the first complete V Engine position-management state machine. This
protocol combines the strongest winner-extension branch with the stop-tighten
defensive branch. It is observation-only and does not authorize live executor
changes.

## Parent Route

`S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_W300_O5`

## Frozen State Machine

| State | Check | Condition | Action |
| --- | --- | --- | --- |
| Recovery | `30 minutes` after maker fill | ETH anchor reclaimed AND BTC is not `btc_down_continues` | Hold to fixed `4h` after fill |
| Danger | `5 minutes` after maker fill | ETH anchor not reclaimed AND BTC is `btc_down_continues` | Tighten stop to trigger price `-80 bps` |
| Neutral | otherwise | neither state active | Keep baseline fixed `2h` after fill |

Decision priority:

```text
Recovery first, then danger, then neutral.
```

## Evidence That Motivated The Freeze

Source report:

```text
reports/research/s34/S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.md
```

Historical result:

```text
baseline H2:              N=22 sum=+1120.7 med=+39.4 T3R=+441.8 max_loss=-144.4
state_machine_recovery:   N=22 sum=+1937.9 med=+57.4 T3R=+1018.5 max_loss=-102.9
delta:                    +817.2 sum, +576.7 T3R, max_loss improved by +41.5
```

## Discipline

- Do not put this into live exit logic yet.
- Do not tune confirmation time, stop distance, or horizon using forward
  observations.
- Forward observations start from zero after this protocol is committed.
- Report baseline H2, state-machine P&L, action counts, delta sum, delta T3R,
  max loss, and top-3-winner-removed cumulative.
- If the state machine underperforms the simple 4h route in forward shadow,
  prefer the simpler route.

## Monitoring

Current historical research:

```text
reports/research/s34/S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.md
reports/research/s34/S34_V_ENGINE_STATE_MACHINE_MANAGEMENT.json
```

Refresh command:

```text
python tools/s34_v_engine_state_machine_management.py
```

## Kill Criteria

Keep as observation-only unless a separately locked forward sample passes. Kill
the protocol if 60-day forward `state_machine_T3R - baseline_H2_T3R < 0`, or if
max loss is not better than baseline.
