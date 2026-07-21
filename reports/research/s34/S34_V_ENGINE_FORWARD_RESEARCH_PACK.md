# S34 V Engine Forward Research Pack

Generated: `2026-06-29T08:09:29.830444+00:00`

Status: `RESEARCH_OBSERVATION_ONLY_NO_LIVE_CHANGE`. Research only. No live executor, order logic, leverage, size, or .env changes.

## Bull Regime Tags

| Tag | N | Sum bps | Median | Win | T3R | Max loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NEUTRAL | 4 | 108.1 | 26.6 | 1.0 | 13.3 | 13.3 |
| RISK_OFF | 7 | 973.5 | 149.9 | 1.0 | 294.5 | 17.2 |

## Exit Management Sweep

| Variant | N | Sum bps | Median | Win | T3R | Max loss | Exit reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tp300_sl150_4h | 11 | 1780.1 | 165.6 | 1.0 | 895.1 | 7.3 | {'TIME': 8, 'TP': 3} |
| fixed_4h | 11 | 1740.8 | 165.6 | 1.0 | 822.6 | 7.3 | {'TIME': 11} |
| fixed_8h | 11 | 1406.4 | 146.7 | 0.818 | 542.2 | -211.3 | {'TIME': 11} |
| trail100_after150_4h | 11 | 1361.0 | 130.9 | 1.0 | 744.5 | 7.3 | {'TIME': 5, 'TRAIL': 6} |
| fixed_2h | 11 | 1089.9 | 46.5 | 1.0 | 406.3 | 12.9 | {'TIME': 11} |
| sl150_2h | 11 | 1089.9 | 46.5 | 1.0 | 406.3 | 12.9 | {'TIME': 11} |
| partial_tp150_2h | 11 | 1014.4 | 81.2 | 1.0 | 455.0 | 12.9 | {'TIME': 6, 'PARTIAL_TP_TIME': 5} |

## Sizing Equity

| Mode | N | Notional | Margin | Sum bps | PnL USDT | End equity | Max DD % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BALANCED | 11 | 16.3 | 0.4 | 1081.6 | 1.763 | 36.763 | 0.0 |
| CURRENT_ENV | 11 | 1190.0 | 29.75 | 1081.6 | 128.711 | 163.71 | 0.0 |
| STOP_ASSISTED | 11 | 39.8 | 1.0 | 1081.6 | 4.304 | 39.305 | 0.0 |
| SURVIVAL | 11 | 11.0 | 0.3 | 1081.6 | 1.192 | 36.19 | 0.0 |

## Fill / Queue Realism

- Fill delay: median `333.0` sec, max `4285.0` sec, >300s `6`, >900s `2`.
- Maker vs anchor counterfactual gain: N=11 sum=122.6 median=16.5 bps.
- Queue stress -10bps: sum=971.6 median=36.3 T3R=322.6.
- Queue stress -20bps: sum=861.6 median=26.3 T3R=242.6.

## Decision Gate

- Status: `WAIT_FORWARD_OOS`; span `71.0` days; N `11`; weeks `4`; reasons `['60D_DECISION_NOT_MET']`.
