# Forced Flow Candidate Harness

- verdict: `SHADOW_CANDIDATE`

## Best

`{'symbol': 'SOLUSDT', 'liq_side': 'BUY', 'direction': 'SHORT', 'threshold': 50000.0, 'horizon_sec': 900, 'exec_model': 'passive_then_taker', 'events': 46, 'filled': 46, 'fill_rate': 1.0, 'gross_wr': 71.73913043478261, 'gross_mean_bps': 16.493913736847755, 'gross_median_bps': 16.314949517220242, 'folds': [{'fold': 1, 'n': 9, 'wr': 55.55555555555556, 'mean_bps': 22.611048902776545, 'median_bps': 5.970149253731004}, {'fold': 2, 'n': 9, 'wr': 77.77777777777777, 'mean_bps': 18.166070065151956, 'median_bps': 15.856066437541791}, {'fold': 3, 'n': 9, 'wr': 33.333333333333336, 'mean_bps': -10.506177553123866, 'median_bps': -8.028443628857367}, {'fold': 4, 'n': 9, 'wr': 100.0, 'mean_bps': 25.12926072124998, 'median_bps': 27.204715484016653}, {'fold': 5, 'n': 10, 'wr': 90.0, 'mean_bps': 26.01182126705052, 'median_bps': 21.40287698855411}], 'fees': {'2.0': {'net_mean_bps': 14.493913736847755, 'net_median_bps': 14.314949517220244, 'net_wr': 71.73913043478261, 'folds_positive': 4}, '4.0': {'net_mean_bps': 12.493913736847755, 'net_median_bps': 12.314949517220244, 'net_wr': 65.21739130434783, 'folds_positive': 4}, '8.0': {'net_mean_bps': 8.493913736847755, 'net_median_bps': 8.314949517220244, 'net_wr': 58.69565217391305, 'folds_positive': 4}, '10.0': {'net_mean_bps': 6.493913736847756, 'net_median_bps': 6.314949517220244, 'net_wr': 56.52173913043478, 'folds_positive': 4}}}`

## Results

| symbol | side | dir | threshold | h | exec | events | filled | fill_rate | WR | mean_bps | median_bps |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| SOLUSDT | BUY | SHORT | 25000 | 900 | passive_then_taker | 76 | 76 | 100.00% | 67.11% | 14.35 | 14.15 |
| SOLUSDT | BUY | SHORT | 50000 | 900 | passive_then_taker | 46 | 46 | 100.00% | 71.74% | 16.49 | 16.31 |
| SOLUSDT | BUY | SHORT | 100000 | 900 | passive_then_taker | 24 | 24 | 100.00% | 70.83% | 16.25 | 14.50 |