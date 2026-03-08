# Research Feature Capability Map

## Purpose

This map labels the current research-side feature surface by upstream dependency strength.

The categories are:

- `mark_only`
- `trade_flow`
- `trade_plus_liq`
- `requires_book`

This is necessary because the current live writer does not publish a true top-of-book table by default. `mark_prices` is often acting as a proxy source for book-like consumers.

## Current Reality

Default live writer:
- `agg_trades`
- `mark_prices`
- `liquidations`

Usually missing from the default writer:
- true top-of-book depth table with real `bid/ask/bid_qty/ask_qty`

Implication:
- trade-flow research is reliable
- liquidation-assisted research is usable
- full book-depth research should be treated as conditional

## Feature Map

### From `core/micro_features.py`

`mark_price`
- category: `mark_only`
- source: `mark_prices`
- note: strongest low-risk feature in the current live setup

`trade_intensity`
- category: `trade_flow`
- source: `agg_trades`
- note: currently computed as 30s trade count converted to per-minute equivalent

`imbalance`
- category: `trade_flow`
- source: `agg_trades`
- note: derived from taker buy/sell flow, not order-book depth

`imbalance_signed`
- category: `trade_flow`
- source: `agg_trades`
- note: directional version of the same trade-flow signal

`spread`
- category: `trade_flow`
- source: `agg_trades + mark_prices`
- note: in current engine this is a proxy based on trade-last vs mark-last, not true bid/ask spread

`age_sec`
- category: `mark_only`
- source: latest observed DB timestamp
- note: health/readiness oriented, not alpha-oriented

### From `tools/build_micro_features.py`

`mid`
- category: `mark_only`
- source: true book if present, otherwise `mark_prices` or trade fallback
- note: valid but interpretation changes depending on source

`spread`
- category: `requires_book`
- source: best when `bid_px/ask_px` exist
- fallback: proxy if true book is absent
- note: do not treat proxy spread as real quoted spread in research conclusions

`microprice`
- category: `requires_book`
- source: `bid_px`, `ask_px`, `bid_qty`, `ask_qty`
- fallback: collapses toward `mid`
- note: low confidence without top-of-book depth

`buy_qty`
- category: `trade_flow`
- source: `agg_trades`

`sell_qty`
- category: `trade_flow`
- source: `agg_trades`

`trade_count`
- category: `trade_flow`
- source: `agg_trades`

`qty_sum`
- category: `trade_flow`
- source: `agg_trades`

`vwap`
- category: `trade_flow`
- source: `agg_trades`

`ofi`
- category: `trade_flow`
- source: `agg_trades`
- note: this is trade-flow imbalance in the current pipeline, not classical book-event OFI

`ofi_norm`
- category: `trade_flow`
- source: `agg_trades`

`trade_intensity_qty_per_sec`
- category: `trade_flow`
- source: `agg_trades`

`trade_intensity_trades_per_sec`
- category: `trade_flow`
- source: `agg_trades`

`top_depth_imbalance`
- category: `requires_book`
- source: `bid_qty`, `ask_qty`
- note: low confidence or empty when real book depth is absent

`rv_short`
- category: `mark_only`
- source: rolling log returns on `mid`
- note: useful even without book depth if `mid` is mark-derived

`liq_count`
- category: `trade_plus_liq`
- source: `liquidations`

`liq_qty`
- category: `trade_plus_liq`
- source: `liquidations`

`liq_sell_qty`
- category: `trade_plus_liq`
- source: `liquidations`
- note: sell-side liquidation pressure proxy

`liq_buy_qty`
- category: `trade_plus_liq`
- source: `liquidations`
- note: buy-side liquidation pressure proxy

`liq_imbalance`
- category: `trade_plus_liq`
- source: `liquidations`
- note: normalized `(sell_liq_qty - buy_liq_qty) / total_liq_qty`; useful for liquidation cascade directionality

`liq_rate_per_sec`
- category: `trade_plus_liq`
- source: `liquidations`
- note: liquidation flow intensity per second

`bid_px`
- category: `requires_book`

`ask_px`
- category: `requires_book`

`bid_qty`
- category: `requires_book`

`ask_qty`
- category: `requires_book`

## Research Guidance

Prioritize these first:
- `mark_price`
- `trade_intensity`
- `imbalance`
- `imbalance_signed`
- `vwap`
- `ofi`
- `ofi_norm`
- `liq_count`
- `liq_qty`
- `liq_sell_qty`
- `liq_buy_qty`
- `liq_imbalance`
- `liq_rate_per_sec`
- `rv_short`

Treat these as conditional:
- `spread`
- `microprice`
- `top_depth_imbalance`
- direct `bid/ask`-driven features

## Rule For Person 1

Before promoting any feature or signal:

1. identify its dependency category
2. confirm whether the current source DB really satisfies that category
3. if the feature is `requires_book` and live data only provides mark/trade proxies, label the result as proxy-based in reports
