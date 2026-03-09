# PASSIVE_THEN_TAKER ETH 60s Family Map

Date:
- 2026-03-09

Branch:
- `codex/research/pocket-promotion-checklist`

Purpose:
- turn the current ETH 60s pocket evidence into a tighter rollout map
- separate true execution flips from fillability-only improvements

Scope:
- symbol:
  - `ETHUSDT`
- rule:
  - `micro_edge_v3_passive_alpha`
- horizon:
  - `h=60`
- window:
  - `7D`

## Family map

| Pocket | `min_imbalance` | `min_trade_intensity` | `max_spread` | Baseline pass | Passive-then-taker pass | Baseline net (bps) | Passive-then-taker net (bps) | Baseline fill | Passive-then-taker fill | Read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pocket B | `0.30` | `8000` | `0.00020` | `1/3` | `3/3` | `-2.044` | `+0.902` | `79.97%` | `100%` | clean rescue |
| Pocket C | `0.30` | `8000` | `0.00025` | `n/a` | `3/3` | `n/a` | `+3.792` | `n/a` | `100%` | strongest passing pocket seen |
| Soft | `0.40` | `2500` | `0.00030` | `0/3` | `2/3` | `-2.776` | `+0.049` | `55.91%` | `100%` | partial rescue, thin edge |
| Mid | `0.50` | `3500` | `0.00030` | `0/3` | `0/3` | `-4.311` | `-0.493` | `57.76%` | `100%` | fillability fix only |
| Tight-mid | `0.50` | `5000` | `0.00025` | `0/3` | `3/3` | `-2.029` | `+2.132` | `60.73%` | `100%` | real flip |

## Read of the map

What is stable:
- `passive_then_taker` consistently pushes fillability to `100%`
- the best outcomes cluster where the pocket is tighter on spread and/or higher on intensity
- the strongest positive net outcomes are not the broadest pockets

What is not stable:
- the effect does not generalize across all ETH 60s pockets
- `MID` proves that better fills alone do not imply a promotable pocket
- `SOFT` is directionally improved but still too thin to sell as a robust passing family

## Pocket classes

Promotable experimental candidates:
- `Pocket B`
- `Pocket C`
- `Tight-mid`

Observe-only pockets:
- `Soft`

Do-not-promote pockets:
- `Mid`

## Rollout language

Use:
- `ETH 60s tighter subfamily experimental candidate`
- `execution shape effect is real, but localized`

Do not use:
- `ETH 60s general solution`
- `broad passive rescue across the family`

## Promotion boundary

Promote only inside this boundary:
- `ETHUSDT`
- `micro_edge_v3_passive_alpha`
- `h=60`
- pockets with at least one of:
  - `min_trade_intensity >= 5000`
  - `max_spread <= 0.00025`
- and no evidence of `0/3` behavior on adjacent checks

Hold outside the boundary:
- softer spread caps around `0.00030` without compensating tightness
- mid-strength pockets where fillability improves but net stays negative

## Research conclusion

Conclusion:
- `passive_then_taker` has cleared the bar for a narrow ETH 60s tighter pocket experiment
- it has not cleared the bar for family-wide ETH promotion

Best current interpretation:
- this is an execution-shape win
- but only inside the tighter pocket subfamily

## Artifact references

- `reports/ETH_POCKET_B_7D_BASELINE_SPLIT2.json`
- `reports/ETH_POCKET_B_7D_PASSIVE_THEN_TAKER.json`
- `reports/ETH_POCKET_C_7D_PASSIVE_THEN_TAKER.json`
- `reports/ETH_POCKET_SOFT_7D_BASELINE.json`
- `reports/ETH_POCKET_SOFT_7D_PASSIVE_THEN_TAKER.json`
- `reports/ETH_POCKET_MID_7D_BASELINE.json`
- `reports/ETH_POCKET_MID_7D_PASSIVE_THEN_TAKER.json`
- `reports/ETH_POCKET_TIGHTMID_7D_BASELINE.json`
- `reports/ETH_POCKET_TIGHTMID_7D_PASSIVE_THEN_TAKER.json`
