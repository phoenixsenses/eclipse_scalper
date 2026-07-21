# S34 SELL Liquidation Filter Sweep

Date: 2026-06-16  
Window: 2026-06-07, 2026-06-11, 2026-06-14, 2026-06-15  
Data: `data/microstructure.db` (`liquidations`, `mark_prices`)  
Model: simplified mark-price replay, flat 8 bps round trip

## Filter Sweep Top Results

| Rank | Threshold | TP | Delay | Filter | N | Days | Mean Net bps | Median Net bps | Cum Net bps | WR | Exits |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 100K | 80 | 0s | delay2_short_confirm_ge_8 | 16 | 3 | +15.46 | -8.00 | +247.38 | 43.8% | TIME 1 / TP 7 / BE 4 / SL 4 |
| 2 | 100K | 80 | 0s | btc_pre15_le_0_delay1_ge_5 | 15 | 4 | +14.67 | -8.00 | +220.08 | 40.0% | TIME 1 / TP 6 / BE 5 / SL 3 |
| 3 | 200K | 60 | 0s | delay2_short_confirm_ge_8 | 8 | 3 | +14.18 | +22.66 | +113.46 | 50.0% | TP 4 / BE 2 / SL 2 |
| 4 | 200K | 80 | 0s | delay2_short_confirm_ge_8 | 8 | 3 | +11.94 | -8.00 | +95.52 | 37.5% | TP 3 / BE 3 / SL 2 |
| 5 | 200K | 60 | 0s | delay1_short_confirm_ge_5 | 9 | 4 | +11.50 | -8.00 | +103.53 | 44.4% | TP 4 / BE 3 / SL 2 |
| 6 | 200K | 60 | 0s | btc_pre15_le_0_delay1_ge_5 | 9 | 4 | +11.50 | -8.00 | +103.53 | 44.4% | TP 4 / BE 3 / SL 2 |
| 7 | 100K | 60 | 0s | btc_pre15_le_0_delay1_ge_5 | 15 | 4 | +11.48 | -8.00 | +172.20 | 40.0% | mixed |
| 12 | 100K | 40 | 0s | btc_pre15_le_0_delay1_ge_5 | 15 | 4 | +7.75 | +32.40 | +116.18 | 60.0% | mixed |

## Lookahead Caveat

Several top filters use labels like `delay1_short_confirm` and `delay2_short_confirm` while the sweep entry delay remains `0s`. If the confirmation observes price movement after the signal but still enters at the original signal timestamp, that result is not deployable. The deployable version must wait until the confirmation timestamp, then enter after the confirmation.

A quick deployable re-check using entry after the confirmation did not preserve the apparent edge. This means the SELL side needs a stricter path-quality study before any live paper variant is added.

## Read

The filtered SELL results are interesting as research hints, but not yet valid strategy candidates. The most plausible direction remains SELL liquidation -> SHORT, but the cleanest results currently depend on confirmation logic that must be re-tested without lookahead.

Verdict for now: continue research only. Do not add an active SELL paper rule yet.
