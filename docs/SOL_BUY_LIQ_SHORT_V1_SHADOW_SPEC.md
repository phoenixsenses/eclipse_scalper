# SOL_BUY_LIQ_SHORT_V1 Shadow Spec

## Status

`SHADOW_ONLY`

This spec is for signal emission and research logging only. It should not place live orders, size positions, or alter production routing.

## Signal Definition

Emit a shadow signal when all trigger conditions are true:

| Field | Requirement |
| --- | --- |
| symbol | `SOLUSDT` |
| source event | forced liquidation |
| liquidation side | `BUY` |
| liquidation notional | `>= 50000` USDT |
| signal direction | `SHORT` |
| research horizon | `900` seconds |

Recommended signal id:

```text
SOL_BUY_LIQ_SHORT_V1:{event_ts_ms}:{event_id_or_hash}
```

## Required Logged Fields

Core event fields:

- `signal_id`
- `signal_family = "SOL_BUY_LIQ_SHORT_V1"`
- `status = "SHADOW_ONLY"`
- `ts_ms`
- `symbol`
- `liq_side`
- `liq_notional`
- `direction = "SHORT"`
- `entry_reference_price`
- `horizon_sec = 900`

Market context fields:

- `spread_bps_at_signal`
- `best_bid_at_signal`
- `best_ask_at_signal`
- `quote_intensity_1s`
- `quote_intensity_5s`
- `book_imbalance_at_signal`
- `depth_bid_top_at_signal`
- `depth_ask_top_at_signal`

Overlap fields:

- `eth_detector_overlap_60s`
- `eth_big_buy_overlap_60s`
- `btc_big_buy_overlap_60s`
- `eth_big_sell_overlap_60s`
- `btc_big_sell_overlap_60s`

Forward labels:

- `mark_60s`
- `mark_120s`
- `mark_300s`
- `mark_900s`
- `return_bps_60s_short`
- `return_bps_120s_short`
- `return_bps_300s_short`
- `return_bps_900s_short`
- `max_favorable_bps_900s`
- `max_adverse_bps_900s`

Execution simulation fields:

- `taker_net_bps_rt_2`
- `taker_net_bps_rt_4`
- `taker_net_bps_rt_8`
- `taker_net_bps_rt_10`
- `passive_fill_simulated`
- `passive_fill_delay_sec`
- `passive_then_taker_net_bps`

## Promotion Gates

Do not promote beyond shadow unless all gates pass on forward-only data collected after this spec date:

| Gate | Requirement |
| --- | --- |
| sample size | at least 100 new events |
| folds | at least 5 chronological folds |
| positive folds | at least 4/5 positive net folds at 900s |
| gross mean | at least +8 bps at 900s |
| fee survival | positive mean after 8 bps round-trip cost |
| tail risk | no mature fold worse than -5 bps mean once that fold has at least 20 events |
| overlap | edge remains positive both with and without ETH/BTC big-buy overlap |
| execution | passive or taker route selected by realized net bps, not gross bps |

## Current Research Snapshot

Backtest result from `reports/SOL_BUY_LIQ_SHORT_V1_WF.md`:

- 46 events
- 73.91% win rate
- +15.78 bps gross mean
- +15.33 bps gross median
- 4/5 positive chronological folds
- +7.78 bps mean after 8 bps round-trip fee stress
- one bad fold: 9 events, 33.33% win rate, -12.39 bps mean

The bad fold is the main reason this remains shadow-only.

