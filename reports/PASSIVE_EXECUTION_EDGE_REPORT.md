# PASSIVE_EXECUTION_EDGE_REPORT

## Re-evaluation (DB + existing debug regime logs)

### Existing regime debug context (h60 files)
| file | n | avg_gross | avg_cost | avg_net |
|---|---:|---:|---:|---:|
| micro_edge_debug_regime_h60_halfspread.jsonl | 702 | +0.000203 | +0.000937 | -0.000734 |
| micro_edge_debug_regime_h60_maker.jsonl | 702 | +0.000203 | +0.000200 | +0.000003 |
| micro_edge_debug_regime_h60_mid.jsonl | 702 | +0.000203 | +0.000800 | -0.000597 |
| micro_edge_debug_regime_h60_taker.jsonl | 702 | +0.000203 | +0.001200 | -0.000997 |

### Current 1440m / 1s bucket / 60s horizon / gates A results
Gates: spread<=0.0003, trade_intensity>=2500, imbalance>=0.3; rule=intensity_spike_imbalance_cont; side=auto

| model | symbol | n | win_rate | avg_gross | avg_cost | avg_net | break_even_cost_bps | fill_rate_attempt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| taker | BTCUSDT | 849 | 5.54% | +0.000127 | +0.001200 | -0.001073 | 1.27 | 100.00% |
| taker | ETHUSDT | 726 | 9.37% | +0.000147 | +0.001200 | -0.001053 | 1.47 | 100.00% |
| maker | BTCUSDT | 849 | 41.46% | +0.000126 | +0.000200 | -0.000074 | 1.26 | 100.00% |
| maker | ETHUSDT | 726 | 44.21% | +0.000146 | +0.000200 | -0.000054 | 1.46 | 100.00% |
| mid | BTCUSDT | 849 | 12.13% | +0.000126 | +0.000800 | -0.000674 | 1.26 | 100.00% |
| mid | ETHUSDT | 726 | 17.08% | +0.000146 | +0.000800 | -0.000654 | 1.46 | 100.00% |
| halfspread | BTCUSDT | 849 | 9.66% | +0.000126 | +0.000902 | -0.000776 | 1.26 | 100.00% |
| halfspread | ETHUSDT | 726 | 14.88% | +0.000146 | +0.000937 | -0.000791 | 1.46 | 100.00% |
| passive_realistic | BTCUSDT | 509 | 41.85% | +0.000045 | +0.000108 | -0.000064 | 0.45 | 43.43% |
| passive_realistic | ETHUSDT | 493 | 45.84% | +0.000106 | +0.000117 | -0.000010 | 1.06 | 55.21% |

## Answer
- Net edge under `passive_realistic` is not positive on this run (cross-symbol avg_net=-0.000037).
- Required break-even total cost from observed gross is ~0.76 bps.
- Observed attempt fill-rate under passive model is 49.32%; lower fill-rate can preserve quality, but excess adverse selection removes edge.
- On this dataset, passive execution materially improves vs taker, but remains slightly negative net; profitability requires either lower effective cost or slightly stronger gross edge / longer horizon.
