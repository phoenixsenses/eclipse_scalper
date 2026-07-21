# S34 Stop-Tighten v0.1 Shadow Protocol

Status: `EXPLORATORY_FROZEN`

Protocol id: `S34_STOP_TIGHTEN_V0_1_ETH_SELL_MAKER_LONG_5M_BTC_DOWN_TRIGSL80`

Created: 2026-06-29

## Purpose

Freeze the first post-fill management overlay for the current V Engine live
route. This protocol is observation-only. It does not authorize live parameter
changes.

## Parent Route

`S34_V_ENGINE_V0_1_ETH_SELL_MAKER_LONG_H2_O20_W300_O5`

## Frozen Overlay

| Field | Value |
| --- | --- |
| Symbol | `ETHUSDT` |
| Parent trigger | ETH SELL liquidation, running notional `>= 200000`, `28 <= vdepth_bps < 40`, `prior_4h_bps < -50` |
| Parent execution | maker LONG, O20 initial limit, after 300s replace to O5, cross margin `1 bps` |
| Parent exit | fixed `2h` after fill |
| Overlay check time | `5 minutes` after maker fill |
| Danger condition | `BTC down continues` AND ETH anchor not reclaimed |
| Stop action | tighten stop to `trigger_price - 80 bps` |
| Permission | `SHADOW_STOP_TIGHTEN_V0_1` observation only |

## Evidence That Motivated The Freeze

Source report:

```text
reports/research/s34/S34_V_ENGINE_POSITION_MANAGEMENT.md
```

Best historical overlay:

```text
tight_trigger_sl80_5m_no_reclaim_btc_down
baseline: N=22 sum=+1120.7 med=+39.4 T3R=+441.8 max_loss=-144.4
managed:  N=22 sum=+1162.4 med=+39.4 T3R=+483.4 max_loss=-102.9
delta:    +41.7 sum, +41.6 T3R
```

## Discipline

- Do not put this into live stop logic yet.
- Do not tune the delay or stop distance using forward observations.
- Forward observations start from zero after this protocol is committed.
- Report baseline and managed P&L, trigger count, delta sum, delta T3R, and max loss.

## Monitoring

Ledger:

```text
reports/research/s34/S34_STOP_TIGHTEN_V0_1_LEDGER.jsonl
reports/research/s34/S34_STOP_TIGHTEN_V0_1_LEDGER.csv
```

Brief:

```text
reports/research/s34/S34_STOP_TIGHTEN_V0_1_BRIEF.md
reports/research/s34/S34_STOP_TIGHTEN_V0_1_BRIEF.json
```

Refresh command:

```text
python tools/s34_stop_tighten_shadow_observer.py
```

## Kill Criteria

Keep as observation-only unless a separately locked forward sample passes. Kill
the overlay if 60-day forward `managed_T3R - baseline_T3R < 0`.
