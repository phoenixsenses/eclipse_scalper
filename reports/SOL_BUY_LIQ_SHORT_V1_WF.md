# Forced Flow Candidate Harness

- verdict: `SHADOW_CANDIDATE`

## Best

`{'symbol': 'SOLUSDT', 'liq_side': 'BUY', 'direction': 'SHORT', 'threshold': 50000.0, 'horizon_sec': 900, 'exec_model': 'taker', 'events': 46, 'filled': 46, 'fill_rate': 1.0, 'gross_wr': 73.91304347826087, 'gross_mean_bps': 15.784917022604914, 'gross_median_bps': 15.334047148768251, 'folds': [{'fold': 1, 'n': 9, 'wr': 77.77777777777777, 'mean_bps': 25.73510219256413, 'median_bps': 9.293680297397572}, {'fold': 2, 'n': 9, 'wr': 77.77777777777777, 'mean_bps': 18.771204383947303, 'median_bps': 20.570269110311976}, {'fold': 3, 'n': 9, 'wr': 33.333333333333336, 'mean_bps': -12.388552536284292, 'median_bps': -8.732977387163503}, {'fold': 4, 'n': 9, 'wr': 100.0, 'mean_bps': 24.50240396627987, 'median_bps': 28.451394891624037}, {'fold': 5, 'n': 10, 'wr': 80.0, 'mean_bps': 21.652476098126293, 'median_bps': 17.495391138057236}], 'fees': {'2.0': {'net_mean_bps': 13.784917022604914, 'net_median_bps': 13.334047148768251, 'net_wr': 71.73913043478261, 'folds_positive': 4}, '4.0': {'net_mean_bps': 11.784917022604914, 'net_median_bps': 11.334047148768251, 'net_wr': 67.3913043478261, 'folds_positive': 4}, '8.0': {'net_mean_bps': 7.784917022604914, 'net_median_bps': 7.334047148768251, 'net_wr': 60.869565217391305, 'folds_positive': 4}, '10.0': {'net_mean_bps': 5.784917022604914, 'net_median_bps': 5.334047148768251, 'net_wr': 56.52173913043478, 'folds_positive': 4}}}`

## Results

| symbol | side | dir | threshold | h | exec | events | filled | fill_rate | WR | mean_bps | median_bps |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| SOLUSDT | BUY | SHORT | 25000 | 300 | taker | 76 | 76 | 100.00% | 65.79% | 9.24 | 7.87 |
| SOLUSDT | BUY | SHORT | 25000 | 900 | taker | 76 | 76 | 100.00% | 68.42% | 13.22 | 12.93 |
| SOLUSDT | BUY | SHORT | 50000 | 300 | taker | 46 | 46 | 100.00% | 76.09% | 13.44 | 15.45 |
| SOLUSDT | BUY | SHORT | 50000 | 900 | taker | 46 | 46 | 100.00% | 73.91% | 15.78 | 15.33 |
| SOLUSDT | BUY | SHORT | 100000 | 300 | taker | 24 | 24 | 100.00% | 70.83% | 11.56 | 13.51 |
| SOLUSDT | BUY | SHORT | 100000 | 900 | taker | 24 | 24 | 100.00% | 70.83% | 14.41 | 11.34 |