# S37 — Mirror S34 Long Research

**Hypothesis**: Large ETH SELL liquidations → LONG ETH → mean reversion profit.

Total ETH SELL liq in window: 43994
Baseline (all SELL liq, h=60s): WR=51.4%

## Results by Notional Threshold (all hours)

| threshold | h | N | WR | mean_ret |
|---|---:|---:|---:|---:|
| >100k | 60s | 850 | 50.1% | -0.87 bps |
| >100k | 120s | 850 | 52.5% | -0.56 bps |
| >200k | 60s | 370 | 51.6% | -1.07 bps |
| >200k | 120s | 370 | 52.2% | -1.17 bps |
| >500k | 60s | 101 | 53.5% | -0.59 bps |
| >500k | 120s | 101 | 54.5% | -2.11 bps |
| >1000k | 60s | 31 | 48.4% | -2.45 bps |
| >1000k | 120s | 31 | 45.2% | -4.84 bps |

## Results by Notional Threshold (13-17 UTC session only)

| threshold | h | N | WR | mean_ret |
|---|---:|---:|---:|---:|
| >100k session | 60s | 234 | 47.9% | -1.83 bps |
| >100k session | 120s | 234 | 52.1% | -0.39 bps |
| >200k session | 60s | 101 | 51.5% | -2.19 bps |
| >200k session | 120s | 101 | 54.5% | -0.78 bps |
| >500k session | 60s | 30 | 50.0% | -2.86 bps |
| >500k session | 120s | 30 | 43.3% | -7.60 bps |
| >1000k session | 60s | 12 | 58.3% | -2.30 bps |
| >1000k session | 120s | 12 | 50.0% | -7.08 bps |

## Notional Distribution

Total SELL liq: 43994
- >100k: 850 events
- >200k: 370 events
- >500k: 101 events
- >1000k: 31 events

## Verdict: WEAK — marginal edge at thresh=1000k, h=60s, session