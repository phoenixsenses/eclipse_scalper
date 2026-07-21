# S34 Execution Playbook

This is the operating standard for S34 paper/testnet/live-readiness work. It is not a live-order instruction sheet.

## Current Mode

- Mode: `PAPER_ONLY_NO_ORDERS`
- Data source: local `liquidations` and `mark_prices`
- Signal source: ETH BUY liquidation clusters
- Current paper direction: ETH LONG
- Leverage target for future execution testing: `10x`
- Real Binance orders: disabled

## Core Terms

- Leverage: borrowed exposure multiplier. At `10x`, 10 USDT margin controls about 100 USDT notional.
- Notional: full position value. Example: 100 USDT notional ETH position.
- Margin: capital locked for the position.
- Isolated margin: risk is contained to that position's margin.
- Cross margin: position shares whole futures wallet. Higher operational risk.
- Entry: the opening order.
- TP: take profit order that closes the position in profit.
- SL: stop loss order that closes the position if thesis is invalidated.
- BE: break-even stop. After price moves enough in favor, SL is moved near entry.
- Reduce-only: exit order that can only reduce/close a position, never increase it.
- Maker fee: limit order provides liquidity.
- Taker fee: market order removes liquidity.
- Slippage: difference between intended and actual fill price.
- Funding fee: periodic perp payment between longs and shorts.
- R multiple: reward/risk unit. A 1R loss equals planned trade risk.

## Default Risk Standard

- Simulated equity: `100 USDT`
- Leverage: `10x`
- Risk per trade: `0.25% equity`
- Daily max loss: `1.0% equity`
- Daily max SL count: `3`
- Max open S34 paper trades: `1`
- Cooldown: after `2` consecutive SL, pause new entries for `6h`
- No new setup if mark data at signal time is stale by more than `30s`

Sizing formula:

```text
risk_usdt = equity * risk_per_trade_pct / 100
notional_usdt = risk_usdt / (sl_bps / 10000)
margin_required = notional_usdt / leverage
```

With default `100 USDT` equity and `40 bps` SL:

```text
risk_usdt = 0.25
notional_usdt = 62.50
margin_required_at_10x = 6.25
```

## S34 Paper Setup Shape

The runner opens paper trials only when a rule sees an ETH BUY liquidation cluster.

Current paper routes:

- `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30`
- `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30`

Each trial records:

- `P###` trial id
- signal timestamp
- liquidation side and notional
- entry mark price
- TP, SL, BE trigger
- simulated notional and margin
- exit reason: `TP`, `SL`, `BE`, `TIME`, or risk skip reason
- gross/net bps
- gross/net USDT

## Hard Rules

- Never enter without an immediately defined SL.
- Exits must be reduce-only in any future testnet/live executor.
- If SL placement fails in testnet/live, close the entry immediately.
- Do not widen SL after entry.
- Do not increase leverage after a loss.
- Do not manually override a paper signal and call it S34 evidence.
- Manual trades belong in the manual journal; system trials belong in `S34_SHADOW_PAPER_JOURNAL.csv`.

## Promotion Gates

- Gate 1: at least `50` system-generated paper trials.
- Gate 2: positive net bps after `8 bps` cost.
- Gate 3: no severe data-health gaps during entries.
- Gate 4: risk-gated paper journal stable for at least `72h`.
- Gate 5: testnet only, tiny notional, `10x`, with mandatory SL and reduce-only exits.
- Gate 6: live canary only after at least `100` paper/testnet trials with positive expectancy.

Until these gates pass, S34 is research/paper-only.
