# CAPACITY_THRESHOLD_CALIBRATION

candidates=2 rule=micro_edge_v3_passive_alpha fee=1.0 adverse=1.2

| min_n_frac | candidate_count | capacity_pass_pct | median_attempts_per_min | median_attempt_fill_rate | median_net_per_attempt | median_effective_min_n | dominance_mode | pass_rate_p25 | pass_rate_p50 | pass_rate_p75 | note |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0.000500 | 2 | 100.00% | 1.75 | 47.50% | +2.000000e-05 | 0 | min_n | 50.00% | 50.00% | 50.00% | rows may look identical while min_n dominates effective_min_n |
| 0.003000 | 2 | 0.00% | 1.75 | 47.50% | -1.000000e-05 | 0 | min_n | 0.00% | 0.00% | 0.00% | rows may look identical while min_n dominates effective_min_n |
