# Policy Outcomes

## Overall

- alpha_gate_checks_total: 2
- alpha_gate_pass_total: 2
- trade_rate_after_pass: 0.500000

## Horizons

| Horizon | Count | Avg | Win | Sharpe | P05 | P01 |
|---|---:|---:|---:|---:|---:|---:|
| 5m | 2 | 0.006491 | 1.000000 | 1.849738 | 0.003333 | 0.003052 |
| 15m | 2 | 0.009473 | 1.000000 | 17.965127 | 0.008998 | 0.008956 |
| 60m | 1 | 0.010000 | 1.000000 | - | 0.010000 | 0.010000 |

## Net Return Stress

| Horizon | Cost bps | Net Avg |
|---|---:|---:|
| 5m | 0 | 0.006491 |
| 5m | 2 | 0.006291 |
| 5m | 5 | 0.005991 |
| 15m | 0 | 0.009473 |
| 15m | 2 | 0.009273 |
| 15m | 5 | 0.008973 |
| 60m | 0 | 0.010000 |
| 60m | 2 | 0.009800 |
| 60m | 5 | 0.009500 |

## By Run ID

| Run ID | Mode | 60m Avg |
|---|---|---:|
| R-OFF | off | 0.010000 |
| R-ON | override | - |

## Warnings

- small_sample:5m:events=2<min_events_warning=30
- small_sample:15m:events=2<min_events_warning=30
- small_sample:60m:events=1<min_events_warning=30
- horizon_availability_low:60m:0.500000<0.90

