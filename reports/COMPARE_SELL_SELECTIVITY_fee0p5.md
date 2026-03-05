# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_SELL_fee0p5_base.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +7.299907e-06 | 50.00% | mixed | 0.50 | 2.08% | 55.63% | 1.000 | 0.259 | 0.094 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.622818e-06 | 50.00% | mixed | 0.50 | 1.92% | 59.71% | 1.000 | 0.276 | 0.071 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -3.020497e-06 | 44.44% | mixed | 0.50 | 2.57% | 56.53% | 1.000 | 0.257 | 0.025 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -9.737651e-06 | 44.44% | fees_dominate | 0.50 | 3.23% | 55.30% | 1.000 | 0.262 | -0.095 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -4.433591e-06 | 38.89% | fees_dominate | 0.50 | 2.97% | 57.35% | 1.000 | 0.256 | -0.048 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -1.593188e-05 | 33.33% | fees_dominate | 0.50 | 2.38% | 60.82% | 1.000 | 0.271 | -0.156 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -3.246928e-05 | 22.22% | fees_dominate | 0.50 | 3.47% | 61.72% | 1.000 | 0.340 | -0.301 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (57.14%)
- top10_mean_npa_core=-7.952882e-06
- top10_mean_pass_rate_core=40.48%

## PASSIVE_POCKET_RANKING_SELL_fee0p5_tight.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -3.549931e-05 | 27.78% | fees_dominate | 0.50 | 3.48% | 61.64% | 1.000 | 0.343 | -0.773 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -5.742778e-05 | 27.78% | fees_dominate | 0.50 | 40.89% | 59.41% | 1.000 | 0.296 | -0.730 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -6.771251e-05 | 27.78% | fees_dominate | 0.50 | 32.57% | 58.64% | 1.000 | 0.308 | -0.789 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -3.975214e-05 | 22.22% | fees_dominate | 0.50 | 49.30% | 58.14% | 1.000 | 0.294 | -0.693 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -4.535706e-05 | 22.22% | gate_reject | 0.50 | 63.10% | 60.05% | 1.000 | 0.283 | -0.699 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -4.580205e-05 | 22.22% | gate_reject | 0.50 | 60.10% | 59.35% | 1.000 | 0.274 | -0.714 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -4.806481e-05 | 22.22% | gate_reject | 0.50 | 55.71% | 58.97% | 1.000 | 0.284 | -0.722 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (57.14%)
- top10_mean_npa_core=-4.851652e-05
- top10_mean_pass_rate_core=24.60%

## PASSIVE_POCKET_RANKING_SELL_fee0p5_ultratight.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +2.544234e-05 | 61.11% | mixed | 0.50 | 31.55% | 64.90% | 1.000 | 0.334 | 0.325 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +2.383328e-05 | 61.11% | gate_reject | 0.50 | 64.50% | 57.96% | 1.000 | 0.329 | 0.340 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +1.810046e-05 | 55.56% | gate_reject | 0.50 | 68.99% | 58.96% | 1.000 | 0.315 | 0.374 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +1.457209e-05 | 55.56% | gate_reject | 0.50 | 80.63% | 60.24% | 1.000 | 0.305 | 0.505 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -3.118132e-06 | 50.00% | gate_reject | 0.50 | 76.69% | 58.96% | 1.000 | 0.305 | 0.242 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -4.157959e-06 | 44.44% | gate_reject | 0.50 | 78.95% | 59.32% | 1.000 | 0.292 | 0.369 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.281645e-05 | 44.44% | gate_reject | 0.50 | 73.23% | 58.30% | 1.000 | 0.313 | 0.211 |

Diagnosis
- dominant_failure_reason_top=gate_reject (85.71%)
- top10_mean_npa_core=+8.836519e-06
- top10_mean_pass_rate_core=53.17%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

