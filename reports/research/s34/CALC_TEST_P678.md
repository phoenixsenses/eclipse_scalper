# S34 Liquidation Outcome Calculator

- generated_at_utc: `2026-06-26T19:12:26+00:00`
- preset: `-`
- source_signal: `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40:5941630`
- source_trade: `P678`
- scope: `BTCUSDT BUY N=127  |  BTCUSDT SELL N=113  |  ETHUSDT BUY N=450  |  ETHUSDT SELL N=222  |  SOLUSDT BUY N=104  |  SOLUSDT SELL N=105`
- selection_mode: `knn`
- decision_card: `RESEARCH_ONLY`
- model_tag: `KNN_USEFUL`
- candidate_events: `222`
- matched_events: `10`
- confidence: `too_thin`
- filters: `symbol=ETHUSDT; side=SELL; cluster_notional>=500000`

## Forward Return Distribution

| Horizon | N | Mean | Median | P25 | P75 | Positive Rate |
|---|---:|---:|---:|---:|---:|---:|
| 60s | 10 | -13.12 | -8.29 | -24.63 | -3.60 | 20.0% |
| 300s | 10 | -12.70 | -4.44 | -18.61 | +3.08 | 30.0% |
| 900s | 10 | +2.52 | +5.18 | -29.20 | +32.71 | 50.0% |
| 3600s | 10 | +9.01 | -5.94 | -18.49 | +26.47 | 40.0% |

## Route Simulation

| Route | N | Median Net | Mean Net | Cum Net | Top3 Removed | WR | TP/BE/SL/TIME | MFE Median | MAE Median |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|

## Decision Card

- verdict: `RESEARCH_ONLY`
- model_tag: `KNN_USEFUL`
- recommended_route: `SHORT_DELAY0_TP60`
- reasons: `best_route_wr=50%; source_signal_targets_loaded`
- warnings: `sample_too_small_n=10; outlier_dependent_top3_removed_negative`
| `LONG_DELAY0_TP40_CONTROL` | 10 | -48.58 | -19.77 | -197.69 | -306.12 | 30.0% | 3/1/6/0 | +7.15 | -40.58 |
| `SHORT_DELAY0_TP60` | 10 | -0.04 | +4.84 | +48.44 | -114.62 | 50.0% | 4/1/3/2 | +50.70 | -7.15 |
| `SHORT_DELAY0_TP80` | 10 | -8.51 | +4.24 | +42.42 | -176.22 | 40.0% | 3/2/3/2 | +50.70 | -7.15 |

## Similarity

- candidate_n: `222`
- selected_n: `10`
- k: `10`

| Feature | Target | Weight | Scale |
|---|---:|---:|---:|
| `log_cluster_notional` | +694421.93 | 2.00 | +0.97 |
| `max_single_liq_share` | +68.02 | 0.80 | +51.12 |
| `intensity_per_sec` | +16017.48 | 1.00 | +10093.87 |
| `inter_cluster_gap_sec` | +170.72 | 0.70 | +36205.24 |
| `day_range_bps` | +483.42 | 1.00 | +269.16 |

## Nearest Analogs

| Event | UTC | Notional | Day Trend | Day Range | Symbol Pre15 | Distance |
|---|---|---:|---:|---:|---:|---:|
| `ETHUSDT_SELL_5922123` | 2026-04-19T22:15:19.314000+00:00 | 1033923 | -395.56 | +451.90 | -56.82 | 0.2195 |
| `ETHUSDT_SELL_5922399` | 2026-04-20T21:18:47.347000+00:00 | 703267 | +251.51 | +380.85 | -87.52 | 0.2597 |
| `ETHUSDT_SELL_5937300` | 2026-06-11T15:01:00.082000+00:00 | 612248 | +124.89 | +318.09 | -61.89 | 0.2787 |
| `ETHUSDT_SELL_5939070` | 2026-06-17T18:32:32.727000+00:00 | 690885 | -166.21 | +391.89 | -6.80 | 0.3003 |
| `ETHUSDT_SELL_5912818` | 2026-03-18T14:52:48.113000+00:00 | 651440 | -476.58 | +660.44 | -73.85 | 0.3073 |
| `ETHUSDT_SELL_5906037` | 2026-02-23T01:45:26.932000+00:00 | 775256 | -452.81 | +492.22 | -14.07 | 0.3435 |
| `ETHUSDT_SELL_5904223` | 2026-02-16T18:36:38.607000+00:00 | 706057 | +56.03 | +439.54 | -25.08 | 0.3465 |
| `ETHUSDT_SELL_5907539` | 2026-02-28T06:55:08.148000+00:00 | 708939 | -400.83 | +460.72 | -46.24 | 0.3544 |
| `ETHUSDT_SELL_5941599` | 2026-06-26T13:16:54.455000+00:00 | 684787 | -218.97 | +488.61 | -25.64 | 0.3659 |
| `ETHUSDT_SELL_5922917` | 2026-04-22T16:26:47.095000+00:00 | 793966 | +294.42 | +474.52 | -27.95 | 0.3709 |

## Read

This is a conditional historical distribution, not a price forecast. It is paper/research only.
If confidence is `thin` or `too_thin`, treat the output as a hypothesis prompt, not evidence.
