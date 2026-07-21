# S34 SELL Liquidation Replay

Date: 2026-06-16  
Window: 2026-06-07, 2026-06-11, 2026-06-14, 2026-06-15  
Data: `data/microstructure.db` (`liquidations`, `mark_prices`)  
Model: simplified mark-price replay, flat 8 bps round trip, no real bid/ask fill, no live risk-gate parity

## Raw SELL Replay

The raw replay tested ETH SELL liquidation clusters as both continuation shorts and reversal longs. The only mildly positive pocket was SELL liquidation -> SHORT at the 200K threshold. Lower thresholds and SELL->LONG were not attractive.

| Rank | Side | Direction | Threshold | TP | N | Days | Mean Net bps | Median Net bps | Cum Net bps | WR | Exits |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SELL | SHORT | 200K | 80 | 20 | 4 | +5.04 | -5.53 | +100.83 | 45.0% | TP 3 / TIME 12 / BE 3 / SL 2 |
| 2 | SELL | SHORT | 200K | 60 | 20 | 4 | +2.32 | -5.53 | +46.41 | 45.0% | TP 3 / TIME 12 / BE 3 / SL 2 |
| 3 | SELL | SHORT | 200K | 120 | 20 | 4 | +2.21 | -5.53 | +44.28 | 45.0% | TIME 15 / BE 3 / SL 2 |
| 4 | SELL | SHORT | 50K | 80 | 49 | 4 | -4.21 | -8.04 | -206.13 | 34.7% | TIME 38 / TP 3 / SL 6 / BE 2 |
| 5 | SELL | SHORT | 50K | 60 | 49 | 4 | -4.71 | -8.04 | -230.89 | 34.7% | TIME 33 / TP 8 / SL 6 / BE 2 |
| 6 | SELL | SHORT | 100K | 80 | 34 | 4 | -5.02 | -12.06 | -170.52 | 35.3% | TIME 25 / TP 3 / SL 6 |

## Daily Split For Best Raw Pocket

Best raw pocket: `SELL -> SHORT`, threshold 200K, TP80.

| Day | N | Mean Net bps | Median Net bps | Cum Net bps | WR | Exits |
|---|---:|---:|---:|---:|---:|---|
| 2026-06-07 | 6 | +16.28 | +5.89 | +97.69 | 66.7% | TP 1 / TIME 4 / BE 1 |
| 2026-06-11 | 6 | -10.62 | -14.48 | -63.70 | 16.7% | TIME-heavy |
| 2026-06-14 | 1 | -48.93 | -48.93 | -48.93 | 0.0% | SL 1 |
| 2026-06-15 | 7 | +16.54 | +4.62 | +115.77 | 57.1% | mixed |

## Read

SELL liquidation continuation short exists as a weak pocket, but it is not clean: median is negative, most exits are TIME/BE, and the edge is carried by 2026-06-07 and 2026-06-15. This is much weaker than the BUY-side 200K/BTC-pre continuation candidate.

Verdict for now: do not add a live SELL rule from the raw replay alone.
