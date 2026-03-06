# CAPACITY_THRESHOLD_CALIBRATION

candidates=7 rule=micro_edge_v3_passive_alpha fee=1.0 adverse=1.2

| min_n_frac | candidate_count | capacity_pass_pct | median_attempts_per_min | median_attempt_fill_rate | median_net_per_attempt | median_effective_min_n | dominance_mode | pass_rate_p25 | pass_rate_p50 | pass_rate_p75 | note |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| 0.000100 | 7 | 100.00% | 0.07 | 58.57% | -6.683047e-05 | 20 | min_n | 16.67% | 27.78% | 38.89% | rows may look identical while min_n dominates effective_min_n |
| 0.000250 | 7 | 100.00% | 0.07 | 58.57% | -6.683047e-05 | 45 | frac_component | 16.67% | 27.78% | 38.89% | frac component influences effective_min_n |
| 0.000500 | 7 | 85.71% | 0.07 | 58.57% | -6.683047e-05 | 89 | frac_component | 11.11% | 16.67% | 27.78% | frac component influences effective_min_n |
| 0.000750 | 7 | 42.86% | 0.07 | 58.57% | -6.683047e-05 | 134 | frac_component | 5.56% | 11.11% | 11.11% | frac component influences effective_min_n |
| 0.001000 | 7 | 0.00% | 0.07 | 58.57% | -6.683047e-05 | 178 | frac_component | 0.00% | 0.00% | 0.00% | frac component influences effective_min_n |
| 0.002000 | 7 | 0.00% | 0.07 | 58.57% | -6.683047e-05 | 355 | frac_component | 0.00% | 0.00% | 0.00% | frac component influences effective_min_n |
| 0.003000 | 7 | 0.00% | 0.07 | 58.57% | -6.683047e-05 | 533 | frac_component | 0.00% | 0.00% | 0.00% | frac component influences effective_min_n |
