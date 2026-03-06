# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_SELL_fee0p8_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +7.410877e-05 | 61.11% | gate_reject | 0.80 | 68.87% | 59.45% | 1.600 | 0.319 | 0.570 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +6.543015e-05 | 61.11% | gate_reject | 0.80 | 64.20% | 58.93% | 1.600 | 0.331 | 0.633 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +4.646457e-05 | 50.00% | mixed | 0.80 | 31.11% | 65.87% | 1.600 | 0.336 | 0.211 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +5.488560e-05 | 55.56% | gate_reject | 0.80 | 80.56% | 60.94% | 1.600 | 0.307 | 0.460 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +5.553226e-05 | 61.11% | gate_reject | 0.80 | 78.87% | 60.79% | 1.600 | 0.295 | 0.453 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +5.523427e-05 | 72.22% | gate_reject | 0.80 | 72.97% | 59.49% | 1.600 | 0.316 | 0.547 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +5.563312e-05 | 72.22% | gate_reject | 0.80 | 76.54% | 59.83% | 1.600 | 0.308 | 0.526 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+5.818411e-05
- top10_mean_pass_rate_core=61.90%

## PASSIVE_POCKET_RANKING_SELL_fee0p9_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -9.155955e-06 | 50.00% | gate_reject | 0.90 | 80.13% | 61.15% | 1.800 | 0.309 | -0.255 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.835334e-05 | 44.44% | gate_reject | 0.90 | 68.18% | 60.05% | 1.800 | 0.322 | -0.179 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -1.891300e-05 | 44.44% | gate_reject | 0.90 | 76.04% | 60.20% | 1.800 | 0.310 | -0.317 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.896550e-05 | 44.44% | gate_reject | 0.90 | 63.60% | 59.42% | 1.800 | 0.332 | -0.129 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -2.422304e-05 | 44.44% | gate_reject | 0.90 | 72.44% | 59.55% | 1.800 | 0.319 | -0.322 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -2.475021e-05 | 44.44% | fees_dominate | 0.90 | 30.12% | 64.84% | 1.800 | 0.342 | -0.422 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -1.517461e-05 | 38.89% | gate_reject | 0.90 | 78.44% | 60.58% | 1.800 | 0.298 | -0.291 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-1.850509e-05
- top10_mean_pass_rate_core=44.44%

## PASSIVE_POCKET_RANKING_SELL_fee1p0_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -4.486506e-05 | 38.89% | gate_reject | 1.00 | 69.20% | 58.64% | 2.000 | 0.316 | -0.536 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -3.451328e-05 | 27.78% | gate_reject | 1.00 | 76.88% | 58.56% | 2.000 | 0.305 | -0.527 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -3.580307e-05 | 27.78% | gate_reject | 1.00 | 73.39% | 57.73% | 2.000 | 0.313 | -0.521 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -3.997382e-05 | 27.78% | gate_reject | 1.00 | 79.22% | 59.51% | 2.000 | 0.292 | -0.524 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -4.725646e-05 | 27.78% | gate_reject | 1.00 | 64.59% | 57.81% | 2.000 | 0.328 | -0.641 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -5.406581e-05 | 27.78% | fees_dominate | 1.00 | 31.70% | 64.70% | 2.000 | 0.332 | -0.796 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -4.023757e-05 | 22.22% | gate_reject | 1.00 | 80.81% | 59.96% | 2.000 | 0.305 | -0.676 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=-4.238787e-05
- top10_mean_pass_rate_core=28.57%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

