# S34 Feature Factory Real-Fill Parity - Top 5 OOS Filters

Scope: recompute top 5 OOS candidates with real historical `book_ticker` bid/ask fills where available.

Caveat: `book_ticker` starts later than the full feature set, so older events are counted as `NO_FILL_DATA` and excluded from real-fill metrics. No modeled spread fallback is used.

| Rank | Route | Filter | Total | Real Fill | No Fill | OOS Test Median | Real Median | Real Mean | Real Cum | Fill Penalty Mean | Entry Adv | Exit Adv | Spread | Fee |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | LONG_DELAY0_TP60 | cluster_notional >= 1000000 AND day_trend_bps >= 0 | 53 | 30 | 23 | +53.51 | +50.85 | +32.09 | +962.80 | -1.82 | +0.85 | -2.73 | +0.06 | +8.00 |
| 2 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_buy_liq_notional >= 5000000 | 63 | 36 | 27 | +52.63 | +48.62 | +26.12 | +940.46 | -0.33 | +0.90 | -1.29 | +0.06 | +8.00 |
| 3 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_trend_bps >= 0 | 97 | 51 | 46 | +52.63 | +48.45 | +25.99 | +1325.66 | -0.46 | +1.11 | -1.62 | +0.06 | +8.00 |
| 4 | LONG_DELAY0_TP60 | cluster_notional >= 500000 AND day_trend_bps >= 100 | 69 | 34 | 35 | +52.77 | +47.69 | +22.99 | +781.66 | +0.20 | +1.30 | -1.16 | +0.06 | +8.00 |
| 5 | LONG_DELAY0_TP60 | btc_pre_15m_bps >= 0 AND day_range_bps >= 500 | 96 | 10 | 86 | +52.81 | +54.33 | +27.14 | +271.44 | -0.75 | +0.57 | -1.42 | +0.10 | +8.00 |

## Read

A candidate passes this gate only if real-fill median remains positive, mean remains positive, and no-fill coverage is not the dominant sample. This is still historical replay, not live paper validation.

## Gate Verdict

Passed real-fill parity:

- `cluster_notional >= 1000000 AND day_trend_bps >= 0`
- `cluster_notional >= 500000 AND day_buy_liq_notional >= 5000000`
- `cluster_notional >= 500000 AND day_trend_bps >= 0`
- `cluster_notional >= 500000 AND day_trend_bps >= 100`

These four retained positive real-fill median and mean after executable bid/ask fills. The true spread cost was small, around `0.06 bps`, and fees remained the dominant fixed cost at `8 bps`. Entry adverse selection was modest in this historical subset, roughly `0.85-1.30 bps` on average.

Rejected / under-covered:

- `btc_pre_15m_bps >= 0 AND day_range_bps >= 500`

This filter still looks positive on the 10 real-fill rows, but `NO_FILL_DATA` is `86/96` (`89.6%`). That is too much missing bookTicker coverage to trust the parity result.

Current best candidate for future paper exploration:

`ETH_BUY_LIQ_LONG_CLUSTER_500K_DAY_TREND_UP_TP60_SL40_BE30`

Reason: `cluster_notional >= 500000 AND day_trend_bps >= 0` has the largest real-fill sample among the passing rows (`51`), positive median (`+48.45 bps`), positive mean (`+25.99 bps`), and positive cumulative real-fill net (`+1325.66 bps`). It is also mechanically interpretable: large ETH BUY liquidation cluster during a non-negative day trend.

Do not add this directly to the pre-registered sample. If promoted, it should become a separate exploratory paper rule with its own label and sample counter.
