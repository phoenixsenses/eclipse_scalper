# Alpha Gate Edge Evaluation

## Overall

- Build UTC: 2026-02-19T03:31:44.118828+00:00
- Mode: offline_replay
- Input: `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\tmp\tests\alpha_gate_edge_group_filter\data\canonical\canonical_merged.parquet`
- Regime Input: `C:\Users\Windows 11\.vscode\CryptoLion\eclipse_scalper\tmp\tests\alpha_gate_edge_group_filter\data\canonical\canonical_regimes.parquet`
- Journal: `logs\execution_journal.jsonl`
- Symbols: ETHUSDT
- Horizons: 5, 15
- Event mode: change (resolved=change)
- Min gap sec: 0
- Group include regex: None
- Group exclude regex: LIQ_CASCADE
- Events before group filter: 3
- Events after group filter: 1
- Allowed events: 1
- Join match rate: 1.000000

## Join Diagnostics

| Scope | Events | Matched Canonical | Unmatched Canonical | Match Rate |
|---|---:|---:|---:|---:|
| overall | 1 | 1 | 0 | 1.000000 |

| Horizon | Events Matched Canonical | Events with Horizon | Events Dropped Horizon | Availability Rate | Drop Rate |
|---|---:|---:|---:|---:|---:|
| 5m | 1 | 1 | 0 | 1.000000 | 0.000000 |
| 15m | 1 | 1 | 0 | 1.000000 | 0.000000 |

## Top Rules by avg_return_15m

| Rule ID | symbol | Events | Avg Return 15m | Net Avg 15m @2bps | Win Rate 15m | P05 15m | P01 15m |
|---|---|---:|---:|---:|---:|---:|---:|
| ETH_VOL_PRICE_DOWN_LONG | ETHUSDT | 1 | 0.000720 | 0.000520 | - | - | - |

## Top Rules by avg_return_60m

| Rule ID | symbol | Events | Avg Return 60m | Net Avg 60m @2bps | Win Rate 60m | P05 60m | P01 60m |
|---|---|---:|---:|---:|---:|---:|---:|
| - | - | 0 | - | - | - | - | - |

## Stability Slice: by (rule_id, symbol) top 15m

| Rule ID | symbol | Events | Avg Return 15m | Net Avg 15m @2bps | Win Rate 15m | P05 15m | P01 15m |
|---|---|---:|---:|---:|---:|---:|---:|
| ETH_VOL_PRICE_DOWN_LONG | ETHUSDT | 1 | 0.000720 | 0.000520 | 1.000000 | 0.000720 | 0.000720 |

## Stability Slice: by (rule_id, symbol) top 60m

| Rule ID | symbol | Events | Avg Return 60m | Net Avg 60m @2bps | Win Rate 60m | P05 60m | P01 60m |
|---|---|---:|---:|---:|---:|---:|---:|
| - | - | 0 | - | - | - | - | - |

## Stability Slice: by (rule_id, regime_label) top 15m

| Rule ID | regime_label | Events | Avg Return 15m | Net Avg 15m @2bps | Win Rate 15m | P05 15m | P01 15m |
|---|---|---:|---:|---:|---:|---:|---:|
| ETH_VOL_PRICE_DOWN_LONG | VOLATILE | 1 | 0.000720 | 0.000520 | 1.000000 | 0.000720 | 0.000720 |

## Stability Slice: by (rule_id, regime_label) top 60m

| Rule ID | regime_label | Events | Avg Return 60m | Net Avg 60m @2bps | Win Rate 60m | P05 60m | P01 60m |
|---|---|---:|---:|---:|---:|---:|---:|
| - | - | 0 | - | - | - | - | - |

## Walk-forward Stability

- walkforward_days=7 folds=0
| Rule ID | Symbol | Folds | Sign Flips 60m | AvgRet 60m Variance |
|---|---|---:|---:|---:|
| - | - | 0 | - | - |

