# S34 v9 Kill Criteria & Forward Review

Generated: `2026-06-29T07:15:29.146404+00:00`

Mode: `RESEARCH_RISK_ONLY_NO_LIVE_CHANGE`

## 1. Kill Criteria

- hard truth: No realized-PnL kill rule can fire before the first tail; only sizing/stop can bound first-tail loss.
- recommended rule: `FIRST_TAIL_OR_10PCT_DD_PAUSE`
- logic: pause after any closed trade <= -100 bps OR realized equity drawdown >= 10%; operator review required to resume
- reason: FIRST_TAIL stops repeat-tail exposure; DD threshold catches non-tail loss clusters. It still cannot protect the first tail.

### Current Env Notional

- notional: `$1190.0`
- max DD: `$-17.2` = `49.1%` equity

| Rule | Result |
| --- | --- |
| `ANY_LOSS_PAUSE` | trigger=True idx=1 dd%=11.6 pnl=$-4.1 |
| `DRAWDOWN_10PCT` | trigger=True idx=1 dd%=11.6 pnl=$-4.1 |
| `FIRST_TAIL_PAUSE` | trigger=True idx=16 dd%=49.1 pnl=$74.4 |
| `DRAWDOWN_20PCT` | trigger=True idx=16 dd%=49.1 pnl=$74.4 |
| `DRAWDOWN_40PCT` | trigger=True idx=16 dd%=49.1 pnl=$74.4 |
| `ROLLING_2_NEGATIVE` | trigger=True idx=17 dd%=49.1 pnl=$80.6 |
| `ROLLING_3_NEGATIVE` | trigger=True idx=18 dd%=49.1 pnl=$77.3 |
| `ROLLING_5_NEGATIVE` | trigger=False idx=None dd%=49.1 pnl=$132.4 |

### Conservative Weighted Size

- notional: `$16.3`
- max DD: `$-0.2` = `0.7%` equity

| Rule | Result |
| --- | --- |
| `ANY_LOSS_PAUSE` | trigger=True idx=1 dd%=0.2 pnl=$-0.1 |
| `FIRST_TAIL_PAUSE` | trigger=True idx=16 dd%=0.7 pnl=$1.0 |
| `ROLLING_2_NEGATIVE` | trigger=True idx=17 dd%=0.7 pnl=$1.1 |
| `ROLLING_3_NEGATIVE` | trigger=True idx=18 dd%=0.7 pnl=$1.1 |
| `ROLLING_5_NEGATIVE` | trigger=False idx=None dd%=0.7 pnl=$1.8 |
| `DRAWDOWN_10PCT` | trigger=False idx=None dd%=0.7 pnl=$1.8 |
| `DRAWDOWN_20PCT` | trigger=False idx=None dd%=0.7 pnl=$1.8 |
| `DRAWDOWN_40PCT` | trigger=False idx=None dd%=0.7 pnl=$1.8 |

## 2. Tick-Level Atomicity Scan

- status: `NO_TICK_CATASTROPHIC_GAP_FOUND`
- filled N: `23` covered N: `23`
- worst tick adverse: `-22.7 bps`
- <= -5bps N: `10`
- <= -25bps N: `0`
- <= -150bps N: `0`
- read: Bounded by filled lifecycle rows; uses agg_trades and book_ticker inside the unprotected 2s window.

## 3. Forward Review Gate

- status: `FROZEN_DECISION_GATE`
- non-negotiable: Passing this gate authorizes only operator review, not automatic live scaling.
- `30D_INTERIM` minimum: >=30 calendar days AND >=10 independent closed forward fills
- `60D_DECISION` minimum: >=60 calendar days AND >=20 independent closed forward fills across >=2 UTC weeks

## Final Read

Kill rules cannot prevent the first tail; use weighted/tail sizing for first-loss survival, then FIRST_TAIL_OR_10PCT_DD_PAUSE for repeat-risk control.
