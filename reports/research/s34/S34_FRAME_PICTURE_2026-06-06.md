# S34 Frame Picture - BTC / ETH / SOL

Generated: 2026-06-06 15:55:22 UTC
Local latest mark timestamp: 2026-06-06 15:55:14 UTC

## Executive Frame

Verdict: **YELLOW for data collection, RED for S34 execution until liquidation transport is restored.**

S34 should be framed as a cascade strategy with BTC as the market-state leader, ETH as the primary execution target, and SOL as a secondary/high-beta confirmation leg. The current data layer is no longer dead: REST fallback is writing fresh mark prices for BTC/ETH/SOL and fresh SOL agg trades, but BTC/ETH aggTrade freshness is not yet as clean as markPrice. The hard blocker is still liquidation transport: local liquidation rows are stale, so S34 cannot honestly confirm forced-flow cascades from current local data.

Picture artifact: `reports\research\s34\S34_FRAME_PICTURE_2026-06-06.png`

## External Market Context

- CoinDesk reported a fresh liquidation-heavy regime on 2026-06-03: roughly $1.84B liquidated in 24h, with long liquidations led by BTC, ETH and SOL, and Binance handling a large share of the cascade.
- Coinalyze currently shows large derivatives surface activity: BTC, ETH and SOL all appear among the largest 24h liquidation / OI names, with BTC 24h liquidations around $197M, ETH around $145M, SOL around $31M on its public dashboard snapshot.

Sources: CoinDesk liquidation report and Coinalyze futures dashboard.

## Local BTC / ETH / SOL Snapshot

| Symbol | Last mark | 1h % | 6h % | 24h % | 1h agg notional | 1h taker imbalance | 24h agg notional | 24h taker imbalance | 24h marks | last agg | 30d local liq rows | last local liq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| BTCUSDT | 60699.5096 | -0.33 | -0.32 | 0.23 | $0.0M | $0.0M | $5.91B | $-70.0M | 12,178 | 2026-06-05 21:04:27 UTC | 0 | 2026-04-27 14:27:26 UTC |
| ETHUSDT | 1558.2231 | -0.29 | -0.49 | -1.56 | $0.0M | $0.0M | $4.86B | $157.3M | 12,177 | 2026-06-05 22:26:17 UTC | 0 | 2026-04-27 14:27:26 UTC |
| SOLUSDT | 61.9684 | -1.16 | -0.78 | -3.52 | $129.5M | $0.0M | $2.59B | $-58.5M | 12,177 | 2026-06-06 15:55:14 UTC | 0 | 2026-04-27 14:24:51 UTC |

Interpretation:

- Positive taker imbalance means buyer-initiated notional dominates; negative means seller-initiated notional dominates.
- mark rows are current and usable for regime framing; agg rows need per-symbol freshness checks because BTC/ETH agg are currently stale relative to SOL.
- the liquidation column is the key S34 red flag.

## Lead-Lag Frame

| Pair | Same-minute corr | Best leader lag | Best lag corr | S34 meaning |
|---|---:|---:|---:|---|
| BTCUSDT->ETHUSDT | 0.881 | 1m | 0.081 | core S34 premise |
| BTCUSDT->SOLUSDT | 0.876 | 1m | 0.103 | confirmation / secondary propagation |
| ETHUSDT->SOLUSDT | 0.894 | 1m | 0.116 | confirmation / secondary propagation |

The lead-lag table should not be read as a standalone alpha verdict. It is a frame check: if BTC->ETH correlation/lag is weak or unstable, S34's BTC_WATCH premise needs caution; if BTC->ETH remains the strongest propagation lane, S34 architecture is directionally supported.

## Current System Truth

- Collector health: `ok`
- Watchdog overall: `GREEN`
- REST fallback active: `True`
- Rows since current collector start: `{"agg_trades":2866886,"mark_prices":36473,"liquidations":0}`
- Liquidation transport available: `False`
- Latest detector signal: 2026-04-15 00:05:30 UTC
- Detector signals in last 30d: 0

## Best Frame For S34

1. BTC is the regime/fuel instrument, not the final confirmation. Watch BTC return shock, BTC taker sell/buy pressure, and BTC->ETH propagation.
2. ETH is the primary S34 execution instrument. It should require BTC lead plus ETH local confirmation, not ETH alone.
3. SOL is a high-beta secondary leg. Use it as confirmation when BTC shock propagates broadly; do not let SOL override BTC/ETH unless a separate SOL-specific edge is validated.
4. Liquidation flow remains the missing forced-flow sensor. Without it, S34 can frame risk but cannot honestly claim cascade confirmation.
5. The next engineering step should be liquidation-only restoration, isolated from agg/mark collector stability.

## Actionable Decision

- Data collection: **continue running**. mark data is useful and fresh; agg freshness needs a focused follow-up for BTC/ETH.
- S34 live/paper signal generation: **do not trust yet** until liquidation transport or validated liquidation substitute is online.
- Next PR: **liquidation-only restoration**. Do not stop the working agg/mark fallback while doing it.

## Honest Limits

- This report uses local mark/agg data and public derivatives context. It does not repair liquidation transport.
- The local DB has historical liquidation rows, but current liquidation feed is unavailable.
- Lead-lag computed here is a framing diagnostic, not a full walk-forward alpha proof.
