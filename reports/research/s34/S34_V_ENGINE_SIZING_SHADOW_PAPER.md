# S34 V Engine Sizing Shadow Paper

Generated: `2026-07-21T08:12:48.400584+00:00`

Status: `SHADOW_PAPER_SIZING_ONLY_NO_ORDER`. Same v0.2 shadow fills, separate sizing ledgers. No live order/config change.

| Mode | N | Notional | Margin | Lev | Sum bps | Median | Win | PnL USDT | End Equity | Max DD % | Max Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BALANCED | 13 | 16.3 | 0.4 | 40.0 | 1089.6 | 46.3 | 0.923 | 1.776 | 36.776 | 0.335 | -0.117 |
| CURRENT_ENV | 13 | 1190.0 | 29.75 | 40.0 | 1089.6 | 46.3 | 0.923 | 129.663 | 164.662 | 24.446 | -8.556 |
| STOP_ASSISTED | 13 | 39.8 | 1.0 | 40.0 | 1089.6 | 46.3 | 0.923 | 4.336 | 39.337 | 0.818 | -0.286 |
| SURVIVAL | 13 | 11.0 | 0.3 | 40.0 | 1089.6 | 46.3 | 0.923 | 1.201 | 36.199 | 0.226 | -0.079 |

Separate sizing shadow-paper for the same v0.2 alpha fills. CURRENT_ENV mirrors configured live sizing for comparison; BALANCED/SURVIVAL are system risk recommendations. No order is sent and no live config is changed.
