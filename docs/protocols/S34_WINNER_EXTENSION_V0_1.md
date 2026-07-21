# S34 Winner Extension v0.1 Shadow Protocol

Status: `EXPLORATORY_FROZEN`

Protocol id: `S34_WINNER_EXTENSION_V0_1_ETH_SELL_MAKER_LONG_30M_ANCHOR_BTC_H4`

Created: 2026-06-29

## Purpose

Freeze the first winner-extension overlay for the current V Engine route. This
protocol tests whether a confirmed recovery state should be held longer than
the current fixed 2h exit. It is observation-only and does not authorize live
executor changes.

## Parent Route

`S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_W300_O5`

## Frozen Overlay

| Field | Value |
| --- | --- |
| Symbol | `ETHUSDT` |
| Parent trigger | ETH SELL liquidation, running notional `>= 200000`, `28 <= vdepth_bps < 40`, `prior_4h_bps < -50` |
| Parent execution | maker LONG, O20 initial limit, after 300s replace to O5, cross margin `1 bps` |
| Baseline exit | fixed `2h` after maker fill |
| Overlay check time | `30 minutes` after maker fill |
| Recovery condition | ETH anchor reclaimed AND BTC is not in `btc_down_continues` |
| Overlay exit | fixed `4h` after maker fill |
| Permission | `SHADOW_WINNER_EXTENSION_V0_1` observation only |

## Evidence That Motivated The Freeze

Source report:

```text
reports/research/s34/S34_V_ENGINE_WINNER_EXTENSION.md
```

Best historical overlay:

```text
30m anchor_and_btc, 4h hold
baseline H2: N=22 sum=+1120.9 med=+39.4 T3R=+441.9 max_loss=-144.4
overlay H4:  N=18 sum=+2022.8 med=+82.0 T3R=+1103.6 max_loss=-24.2
delta:       +901.9 sum, +661.7 T3R
```

## Discipline

- Do not put this into live exit logic yet.
- Do not tune the confirmation time or holding horizon using forward
  observations.
- Forward observations start from zero after this protocol is committed.
- Report baseline 2h and overlay 4h P&L, trigger count, delta sum, delta T3R,
  max loss, and top-3-winner-removed cumulative.
- If the recovery condition is not present at 30m, baseline 2h remains the
  comparison path; this protocol only studies the extension subset.

## Monitoring

Current historical research:

```text
reports/research/s34/S34_V_ENGINE_WINNER_EXTENSION.md
reports/research/s34/S34_V_ENGINE_WINNER_EXTENSION.json
```

Refresh command:

```text
python tools/s34_v_engine_winner_extension.py
```

## Kill Criteria

Keep as observation-only unless a separately locked forward sample passes. Kill
the overlay if 60-day forward `overlay_T3R - baseline_T3R < 0` or if the
overlay increases max loss versus the 2h baseline.
