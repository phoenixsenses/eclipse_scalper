# S34 Exploratory Paper Rule Pre-Registration - SOL 200K BUY-Liq LONG

Date: 2026-06-19

Status: exploratory paper only. This is not part of the locked 50K/TP120 pre-registered ETH validation sample.

## Rule

`SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30`

- Symbol: `SOLUSDT`
- Liquidation side: `BUY`
- Direction: `LONG`
- Cluster threshold: `>= 200,000 USDT`
- Bucket: `300s`
- Minimum gap: `900s`
- Entry delay: `0s`
- TP: `60 bps`
- SL: `40 bps`
- BE trigger: `30 bps`
- Fill model: real bookTicker taker entry/exit, no modeled fallback
- Regime: no global ETH regime gate
- Max open trades: rule-scoped default `1`
- Daily max SL: rule-scoped default `3`

## Why This Rule Exists

The cross-symbol real-fill scan found that the ETH BUY-liq surface is mostly already covered by existing live exploratory ETH rules, while SOL showed an independent positive pocket:

| Candidate | Real N | Days | Median net bps | Mean net bps | Cum net bps | WR | Second-half N | Second-half median | No-fill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SOL BUY >=200K LONG TP60/SL40/BE30 | 35 | 12 | +50.01 | +21.75 | +761.39 | 62.9% | 25 | +48.95 | 31.4% |

Source: `reports/research/s34/S34_CROSS_SYMBOL_BUY_REALFILL_SCAN.md`

BTC 1M also passed the screen, but SOL 200K had a stronger median, lower no-fill rate, and provides a non-ETH orthogonal exploratory surface. SOL 500K was stronger by point estimate but only had N=12, so it is too small for live paper promotion.

## Evaluation Discipline

This rule is exploratory. It does not change or reset:

- `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` pre-reg N/40 counter
- Existing ETH exploratory rule counters
- Existing feature-factory research reports

The rule should be assessed only after it has enough clean live paper data:

- Minimum N: 30 clean closed trades
- Minimum spread: at least 8 distinct trading days
- Median net bps must remain positive
- Top 3 winners removed cumulative net bps must remain positive
- No-fill/quarantine rate must be monitored; if it exceeds 25% and correlates with cascade intensity, the sample is biased

No TP/SL/BE tuning is authorized from this document.
