# COMPARE_RANK_RUNS

## PASSIVE_POCKET_RANKING_BUY.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | +4.538812e-05 | 77.78% | mixed | 0.00 | 2.42% | 56.05% | 0.000 | 0.256 | 0.791 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | +4.528485e-05 | 83.33% | mixed | 0.00 | 2.88% | 57.05% | 0.000 | 0.256 | 0.865 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | +4.685324e-05 | 61.11% | mixed | 0.00 | 4.22% | 62.70% | 0.000 | 0.349 | 0.406 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | +3.655886e-05 | 77.78% | mixed | 0.00 | 2.23% | 60.76% | 0.000 | 0.271 | 0.645 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | +2.474410e-05 | 66.67% | mixed | 0.00 | 2.60% | 61.40% | 0.000 | 0.271 | 0.520 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | +5.797911e-07 | 50.00% | mixed | 0.00 | 3.55% | 55.57% | 0.000 | 0.260 | 0.277 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -9.906162e-07 | 44.44% | mixed | 0.00 | 3.34% | 57.79% | 0.000 | 0.254 | 0.273 |

Diagnosis
- dominant_failure_reason_top=mixed (100.00%)
- top10_mean_npa_core=+2.834548e-05
- top10_mean_pass_rate_core=65.87%

## PASSIVE_POCKET_RANKING_BUY_fee0p5.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.396300e-05 | 44.44% | fees_dominate | 0.50 | 1.97% | 61.77% | 1.000 | 0.273 | -0.238 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -8.671313e-06 | 38.89% | fees_dominate | 0.50 | 2.17% | 56.39% | 1.000 | 0.256 | -0.210 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -2.097545e-05 | 33.33% | fees_dominate | 0.50 | 3.39% | 56.41% | 1.000 | 0.260 | -0.444 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -3.485628e-05 | 33.33% | fees_dominate | 0.50 | 2.47% | 62.93% | 1.000 | 0.270 | -0.526 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -5.749314e-05 | 33.33% | fees_dominate | 0.50 | 3.65% | 63.16% | 1.000 | 0.340 | -0.745 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -2.310910e-05 | 27.78% | fees_dominate | 0.50 | 2.71% | 57.73% | 1.000 | 0.254 | -0.328 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -2.450064e-05 | 27.78% | fees_dominate | 0.50 | 3.16% | 58.66% | 1.000 | 0.254 | -0.436 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top10_mean_npa_core=-2.622413e-05
- top10_mean_pass_rate_core=34.13%

## PASSIVE_POCKET_RANKING_BUY_fee1p0.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -6.144945e-05 | 11.11% | fees_dominate | 1.00 | 3.49% | 60.38% | 2.000 | 0.342 | -1.336 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -7.702550e-05 | 11.11% | fees_dominate | 1.00 | 2.11% | 55.43% | 2.000 | 0.256 | -1.354 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -9.011828e-05 | 11.11% | fees_dominate | 1.00 | 2.56% | 57.01% | 2.000 | 0.256 | -1.411 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -8.275374e-05 | 5.56% | fees_dominate | 1.00 | 3.21% | 55.64% | 2.000 | 0.262 | -1.515 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -9.427038e-05 | 5.56% | fees_dominate | 1.00 | 2.40% | 61.46% | 2.000 | 0.271 | -1.600 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -8.136624e-05 | 0.00% | fees_dominate | 0.00 | 2.98% | 58.05% | 2.000 | 0.255 | -1.486 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.000202e-04 | 0.00% | fees_dominate | 0.00 | 1.92% | 60.13% | 2.000 | 0.273 | -1.484 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top10_mean_npa_core=-8.385769e-05
- top10_mean_pass_rate_core=6.35%

## Cross-Run Diagnosis

- BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name).

