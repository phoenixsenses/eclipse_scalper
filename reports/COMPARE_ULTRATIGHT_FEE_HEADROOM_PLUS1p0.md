# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_SELL_fee0p5_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +4.602898e-05 | 77.78% | gate_reject | 0.50 | 73.64% | 58.65% | 1.000 | 0.314 | 0.690 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +4.332508e-05 | 72.22% | gate_reject | 0.50 | 77.11% | 59.43% | 1.000 | 0.307 | 0.767 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +3.128088e-05 | 72.22% | gate_reject | 0.50 | 79.38% | 59.84% | 1.000 | 0.293 | 0.687 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +2.789427e-05 | 77.78% | gate_reject | 0.50 | 80.91% | 60.00% | 1.000 | 0.306 | 0.697 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +1.003180e-05 | 55.56% | gate_reject | 0.50 | 69.42% | 59.32% | 1.000 | 0.317 | 0.608 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +9.694839e-06 | 55.56% | gate_reject | 0.50 | 64.95% | 58.65% | 1.000 | 0.329 | 0.627 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.386051e-05 | 44.44% | mixed | 0.50 | 31.64% | 64.19% | 1.000 | 0.334 | 0.016 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+2.205648e-05
- top10_mean_pass_rate_core=65.08%

## PASSIVE_POCKET_RANKING_SELL_fee0p6_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +7.389162e-05 | 61.11% | gate_reject | 0.60 | 69.48% | 60.00% | 1.200 | 0.316 | 0.477 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +6.970940e-05 | 61.11% | gate_reject | 0.60 | 64.94% | 58.78% | 1.200 | 0.328 | 0.478 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +5.930626e-05 | 61.11% | mixed | 0.60 | 31.48% | 65.82% | 1.200 | 0.331 | 0.341 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +5.160813e-05 | 66.67% | gate_reject | 0.60 | 73.61% | 59.66% | 1.200 | 0.313 | 0.258 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +5.087787e-05 | 61.11% | gate_reject | 0.60 | 79.34% | 60.75% | 1.200 | 0.292 | 0.316 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +5.023957e-05 | 55.56% | gate_reject | 0.60 | 77.07% | 60.34% | 1.200 | 0.305 | 0.307 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +4.987560e-05 | 55.56% | gate_reject | 0.60 | 80.86% | 61.01% | 1.200 | 0.304 | 0.277 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+5.792978e-05
- top10_mean_pass_rate_core=60.32%

## PASSIVE_POCKET_RANKING_SELL_fee0p7_ultratight_pass0p50.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +3.022322e-05 | 55.56% | fees_dominate | 0.70 | 31.58% | 63.24% | 1.400 | 0.330 | -0.078 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | +2.761285e-05 | 55.56% | gate_reject | 0.70 | 79.41% | 59.06% | 1.400 | 0.291 | -0.113 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +2.721506e-05 | 50.00% | gate_reject | 0.70 | 81.04% | 59.42% | 1.400 | 0.303 | -0.165 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +2.615953e-05 | 50.00% | gate_reject | 0.70 | 77.13% | 58.31% | 1.400 | 0.305 | -0.167 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +1.383599e-05 | 50.00% | gate_reject | 0.70 | 73.66% | 57.95% | 1.400 | 0.313 | -0.224 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +6.931456e-06 | 50.00% | gate_reject | 0.70 | 64.85% | 57.94% | 1.400 | 0.326 | -0.229 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +5.936403e-06 | 50.00% | gate_reject | 0.70 | 69.41% | 58.23% | 1.400 | 0.315 | -0.275 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+1.970207e-05
- top10_mean_pass_rate_core=51.59%

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

