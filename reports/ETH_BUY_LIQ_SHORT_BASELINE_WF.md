# Forced Flow Candidate Harness

- verdict: `SHADOW_CANDIDATE`

## Best

`{'symbol': 'ETHUSDT', 'liq_side': 'BUY', 'direction': 'SHORT', 'threshold': 500000.0, 'horizon_sec': 900, 'exec_model': 'taker', 'events': 121, 'filled': 121, 'fill_rate': 1.0, 'gross_wr': 63.63636363636363, 'gross_mean_bps': 7.522522297506553, 'gross_median_bps': 11.548009166585963, 'folds': [{'fold': 1, 'n': 24, 'wr': 66.66666666666667, 'mean_bps': 7.5354438415051455, 'median_bps': 16.106246451256894}, {'fold': 2, 'n': 24, 'wr': 62.5, 'mean_bps': -19.23345294362497, 'median_bps': 11.178396754435534}, {'fold': 3, 'n': 24, 'wr': 70.83333333333333, 'mean_bps': 17.81856535084757, 'median_bps': 9.882049082253234}, {'fold': 4, 'n': 24, 'wr': 79.16666666666667, 'mean_bps': 33.55494430633593, 'median_bps': 27.608606922226425}, {'fold': 5, 'n': 25, 'wr': 40.0, 'mean_bps': -1.6794726129294102, 'median_bps': -4.253421215935176}], 'fees': {'2.0': {'net_mean_bps': 5.522522297506553, 'net_median_bps': 9.548009166585963, 'net_wr': 62.8099173553719, 'folds_positive': 3}, '4.0': {'net_mean_bps': 3.522522297506553, 'net_median_bps': 7.548009166585963, 'net_wr': 58.67768595041322, 'folds_positive': 3}, '8.0': {'net_mean_bps': -0.4774777024934471, 'net_median_bps': 3.548009166585963, 'net_wr': 55.37190082644628, 'folds_positive': 2}, '10.0': {'net_mean_bps': -2.4774777024934473, 'net_median_bps': 1.5480091665859632, 'net_wr': 52.892561983471076, 'folds_positive': 2}}}`

## Results

| symbol | side | dir | threshold | h | exec | events | filled | fill_rate | WR | mean_bps | median_bps |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ETHUSDT | BUY | SHORT | 200000 | 120 | taker | 438 | 438 | 100.00% | 57.53% | -0.62 | 4.22 |
| ETHUSDT | BUY | SHORT | 200000 | 900 | taker | 438 | 438 | 100.00% | 61.19% | 5.64 | 9.58 |
| ETHUSDT | BUY | SHORT | 500000 | 120 | taker | 121 | 121 | 100.00% | 60.33% | 0.05 | 4.64 |
| ETHUSDT | BUY | SHORT | 500000 | 900 | taker | 121 | 121 | 100.00% | 63.64% | 7.52 | 11.55 |
| ETHUSDT | BUY | SHORT | 1000000 | 120 | taker | 50 | 50 | 100.00% | 52.00% | -7.36 | 2.48 |
| ETHUSDT | BUY | SHORT | 1000000 | 900 | taker | 50 | 50 | 100.00% | 64.00% | 4.88 | 13.04 |