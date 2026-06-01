# S53 Purity + OFI

Distributions: `{'purity': {'min': 1, 'p25': 3.0, 'p50': 3.0, 'p75': 4.0, 'p90': 4.0}, 'ofi': {'min': -0.4697422727168306, 'p25': -0.04083924232889871, 'p50': 0.14512761045961162, 'p75': 0.24801085283304683, 'p90': 0.32104359497386564}, 'btc_ofi': {'min': -0.5898170595174513, 'p25': -0.19635124874304824, 'p50': -0.06957648417864788, 'p75': 0.14358916639021851, 'p90': 0.24698598916047276}}`

| filter | h | N | WR | mean_ret_bps |
|---|---:|---:|---:|---:|
| purity >= 6 | 60s | 0 | nan% | +nan |
| purity >= 6 | 120s | 0 | nan% | +nan |
| purity >= 7 | 60s | 0 | nan% | +nan |
| purity >= 7 | 120s | 0 | nan% | +nan |
| purity >= 8 | 60s | 0 | nan% | +nan |
| purity >= 8 | 120s | 0 | nan% | +nan |
| purity >= 9 | 60s | 0 | nan% | +nan |
| purity >= 9 | 120s | 0 | nan% | +nan |
| purity >= 10 | 60s | 0 | nan% | +nan |
| purity >= 10 | 120s | 0 | nan% | +nan |
| purity < 5 | 60s | 38 | 55.3% | +0.33 |
| purity < 5 | 120s | 38 | 47.4% | +1.49 |
| ofi < -0.3 | 60s | 4 | 50.0% | -0.75 |
| ofi < -0.3 | 120s | 4 | 0.0% | -11.47 |
| ofi < -0.5 | 60s | 0 | nan% | +nan |
| ofi < -0.5 | 120s | 0 | nan% | +nan |
| ofi < -0.7 | 60s | 0 | nan% | +nan |
| ofi < -0.7 | 120s | 0 | nan% | +nan |
| ofi > 0.3 | 60s | 8 | 62.5% | +3.80 |
| ofi > 0.3 | 120s | 8 | 75.0% | +16.87 |
| purity >= 7 and ofi < -0.3 | 60s | 0 | nan% | +nan |
| purity >= 7 and ofi < -0.3 | 120s | 0 | nan% | +nan |
| purity >= 7 and ofi < -0.5 | 60s | 0 | nan% | +nan |
| purity >= 7 and ofi < -0.5 | 120s | 0 | nan% | +nan |
| btc_ofi < -0.3 | 60s | 10 | 70.0% | +3.93 |
| btc_ofi < -0.3 | 120s | 10 | 80.0% | +6.18 |
| eth_ofi < -0.3 and btc_ofi < -0.3 | 60s | 0 | nan% | +nan |
| eth_ofi < -0.3 and btc_ofi < -0.3 | 120s | 0 | nan% | +nan |