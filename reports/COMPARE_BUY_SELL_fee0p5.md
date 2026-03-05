# COMPARE_RANK_RUNS

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

## PASSIVE_POCKET_RANKING_SELL_fee0p5.json

rows_total=7 top_n=7

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | 120 | 0.40 | 2500 | 0.000500 | -4.063021e-06 | 50.00% | fees_dominate | 0.50 | 2.55% | 56.46% | 1.000 | 0.258 | -0.162 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000500 | -1.022187e-05 | 44.44% | fees_dominate | 0.50 | 2.06% | 55.19% | 1.000 | 0.260 | -0.087 |
| ETHUSDT | 120 | 0.50 | 2500 | 0.000300 | -1.640492e-05 | 38.89% | fees_dominate | 0.50 | 1.87% | 60.13% | 1.000 | 0.277 | -0.208 |
| ETHUSDT | 120 | 0.30 | 2500 | 0.000500 | -1.647348e-05 | 38.89% | fees_dominate | 0.50 | 2.98% | 57.57% | 1.000 | 0.256 | -0.240 |
| ETHUSDT | 120 | 0.50 | 3500 | 0.000300 | -1.983611e-05 | 38.89% | fees_dominate | 0.50 | 3.40% | 61.36% | 1.000 | 0.341 | -0.357 |
| ETHUSDT | 120 | 0.20 | 2500 | 0.000500 | -1.601079e-05 | 27.78% | fees_dominate | 0.50 | 3.24% | 55.17% | 1.000 | 0.263 | -0.248 |
| ETHUSDT | 120 | 0.40 | 2500 | 0.000300 | -3.237914e-05 | 27.78% | fees_dominate | 0.50 | 2.32% | 61.40% | 1.000 | 0.273 | -0.430 |

Diagnosis
- dominant_failure_reason_top=fees_dominate (100.00%)
- top10_mean_npa_core=-1.648419e-05
- top10_mean_pass_rate_core=38.10%

## Cross-Run Diagnosis

- BUY/SELL delta (top-10 mean): delta_npa_core=-9.739940e-06, delta_pass_rate_core=-3.97%

