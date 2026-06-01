# Forced Flow Candidate Harness

- verdict: `SHADOW_CANDIDATE`

## Best

`{'symbol': 'SOLUSDT', 'liq_side': 'BUY', 'direction': 'SHORT', 'threshold': 50000.0, 'horizon_sec': 900, 'exec_model': 'passive', 'events': 46, 'filled': 37, 'fill_rate': 0.8043478260869565, 'gross_wr': 75.67567567567568, 'gross_mean_bps': 16.999621204177345, 'gross_median_bps': 16.504582287148807, 'folds': [{'fold': 1, 'n': 7, 'wr': 71.42857142857143, 'mean_bps': 19.18047352558339, 'median_bps': 6.946856547412559}, {'fold': 2, 'n': 7, 'wr': 57.142857142857146, 'mean_bps': 13.293933722443821, 'median_bps': 3.4296201514806133}, {'fold': 3, 'n': 8, 'wr': 62.5, 'mean_bps': -4.435257838431942, 'median_bps': 4.523032141217269}, {'fold': 4, 'n': 7, 'wr': 100.0, 'mean_bps': 32.70858079089467, 'median_bps': 33.79900856241678}, {'fold': 5, 'n': 8, 'wr': 87.5, 'mean_bps': 26.02339137369551, 'median_bps': 21.40287698855411}], 'fees': {'2.0': {'net_mean_bps': 14.999621204177346, 'net_median_bps': 14.504582287148807, 'net_wr': 75.67567567567568, 'folds_positive': 4}, '4.0': {'net_mean_bps': 12.999621204177346, 'net_median_bps': 12.504582287148807, 'net_wr': 67.56756756756756, 'folds_positive': 4}, '8.0': {'net_mean_bps': 8.999621204177345, 'net_median_bps': 8.504582287148807, 'net_wr': 59.45945945945946, 'folds_positive': 4}, '10.0': {'net_mean_bps': 6.999621204177346, 'net_median_bps': 6.504582287148807, 'net_wr': 56.75675675675676, 'folds_positive': 4}}}`

## Results

| symbol | side | dir | threshold | h | exec | events | filled | fill_rate | WR | mean_bps | median_bps |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| SOLUSDT | BUY | SHORT | 25000 | 900 | passive | 76 | 64 | 84.21% | 68.75% | 13.94 | 13.96 |
| SOLUSDT | BUY | SHORT | 50000 | 900 | passive | 46 | 37 | 80.43% | 75.68% | 17.00 | 16.50 |
| SOLUSDT | BUY | SHORT | 100000 | 900 | passive | 24 | 21 | 87.50% | 71.43% | 16.83 | 12.88 |