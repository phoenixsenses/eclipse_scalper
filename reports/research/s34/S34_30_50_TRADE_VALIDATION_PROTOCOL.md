# S34-Inspired Manual Validation Protocol

Generated: 2026-06-06

## Purpose

Validate whether the S34-inspired BTC/ETH/SOL frame can repeatedly produce usable ETHUSDT short/long setups without the live liquidation feed.

This is not full S34 validation. Full S34 requires live liquidation confirmation. This protocol tests the incomplete-sensor frame:

```text
BTC/ETH/SOL structure + mark/agg context + predefined levels
```

## Sample Target

```text
Minimum: 30 trades
Preferred: 50 trades
One setup at a time
No rule changes during the first 30 trades
```

## Risk Rule

Use micro size until the first 30 trades are complete.

Do not use 40x for the validation set. High leverage will contaminate the test because liquidation/risk pressure may force bad manual decisions.

Recommended validation risk:

```text
Max account risk per trade: 3-5 USDT or less
Preferred leverage: 5x-10x
One position at a time
Always SL + TP placed immediately
No averaging down
No revenge trades
```

## Trade Types

### A. Short Continuation

Use when:

```text
BTC fails reclaim
ETH rejects S34 short zone
SOL remains weak or fails reclaim
```

Required order structure:

```text
Entry: predefined rejection zone
SL: predefined invalidation level
TP1: first liquidity / structure target
TP2: optional deeper cascade target
```

### B. Long Reversal

Use only when:

```text
ETH flushes into support
BTC stops falling / reclaims
SOL stops underperforming
Price reclaims breakdown level
```

Longs are secondary. Current S34 frame is naturally more short/cascade-oriented.

## Required Log Fields

Every trade must be logged before outcome is known:

```text
trade_id
timestamp_utc
symbol
direction
setup_type
entry
stop_loss
tp1
tp2
leverage
position_value_usdt
account_equity_before
btc_state
eth_state
sol_state
liquidation_feed_available
s34_confidence
reason_for_entry
exit_price
exit_reason
gross_pnl_usdt
net_pnl_usdt
notes
```

## Current Trade: Trial 001

```text
trade_id: S34M-001
symbol: ETHUSDT
direction: SHORT
setup_type: short continuation / rejection zone
entry: 1562.00
stop_loss: 1574.00
tp1: 1535.00
tp2: none active
leverage: ~10x
position_value_usdt: ~390.7
liquidation_feed_available: false
s34_confidence: incomplete-sensor frame only
status: open
```

## Scoring

After 30 trades:

```text
Kill if:
- Net PnL <= 0
- Win rate < 45%
- Average loss > average win
- More than 5 rule violations
- Edge only exists when manually moved after entry

Continue if:
- Net PnL > 0
- Win rate >= 50% OR R:R compensates below-50 WR
- Drawdown controlled
- Rules are repeatable
```

After 50 trades:

```text
Forward-validate only if:
- Net PnL positive after fees
- Max drawdown tolerable
- At least 2 setup types or 1 very stable setup type
- No dependency on hindsight chart reading
```

## Non-Negotiables

```text
Do not move SL wider after entry.
Do not add to losing position.
Do not enter without TP/SL.
Do not trade while emotionally reacting to previous trade.
Do not count undocumented trades.
```

## Interpretation

If this 30-50 trade set works, it validates the S34-inspired frame as a candidate manual alpha.

It still does not validate full S34 until liquidation transport is restored and the same setups are compared with liquidation-confirmed setups.
