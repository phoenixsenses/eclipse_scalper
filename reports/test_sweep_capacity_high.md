# FILTER_SWEEP_PASSIVE_REALISTIC

rows=1 pass=0
capacity_filter splits=4 seeds=11,22 min_n=500 min_n_frac=0.0 min_attempt_fill_rate=0.1 max_insufficient_fill_rate=0.5

| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | v2_min_score | v2_min_persistence | filled_n | filled_avg_net | filled_p90_net | net_per_attempt | cap_attempt_fill_rate | insufficient_fill_rate | cap_ok | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETHUSDT | 60 | 0.50 | 2500 | 0.000500 | 0.000000 | 0.000000 | 100 | +0.000200 | +0.000300 | +0.000120 | 45.00% | 100.00% | NO | NO |
