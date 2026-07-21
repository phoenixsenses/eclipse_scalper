# S34 Cross-Asset Lead-Lag (is the precursor in another asset?)

Generated: `2026-06-28T22:35:27.794910+00:00`  |  window 45d, 1-min returns

## A. Lead-lag cross-correlation (corr lead[t] vs follow[t+lag]); peak at +lag => lead leads follow

| Pair (lead->follow) | peak lag (min) | peak corr | corr@0 | corr@+1 | corr@+2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC->ETH | 0 | 0.86 | 0.86 | 0.026 | -0.02 |
| BTC->SOL | 0 | 0.83 | 0.83 | 0.03 | -0.017 |
| ETH->SOL | 0 | 0.869 | 0.869 | 0.025 | -0.018 |
| ETH->BTC | 0 | 0.86 | 0.86 | 0.031 | -0.025 |

## B. BTC precursor around ETH deep-V SELL cascades
- N=169; BTC return before cross: [-5m]=-39.2bps, [-1m]=-19.8bps
- BTC[-5m] median: winners=-44.6 vs runaways=-35.5
