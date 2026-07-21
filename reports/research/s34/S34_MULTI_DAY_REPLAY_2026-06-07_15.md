# S34 Multi-Day Replay - 2026-06-07 / 06-11 / 06-14 / 06-15

Scope: read-only replay over existing microstructure.db. No production runner/config changes.

Model caveat: this is a simplified mark-price replay with flat 8 bps round-trip cost. It does not fully model live paper behavior: real bid/ask fills, adverse selection, rule-scoped risk gates, cooldown, max-open slot timing, and live cursor sequencing are not identical here. Treat this as directional route research, not validation.

| day | rule | signals | regime-pass | mean net bps | median net bps | cum net bps | exits | ETH BUY liq notional |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2026-06-07 | 50K_TP120 | 39 | 30 | -0.67 | -9.49 | -20.05 | TIME 6 / BE 8 / SL 12 / TP 4 | 78.86M |
| 2026-06-07 | 200K_TP60 | 14 | 7 | 31.60 | 53.31 | 221.23 | TP 5 / BE 1 / SL 1 | 78.86M |
| 2026-06-11 | 50K_TP120 | 28 | 13 | -16.95 | -10.07 | -220.34 | SL 6 / TIME 1 / TP 1 / BE 5 | 15.54M |
| 2026-06-11 | 200K_TP60 | 11 | 6 | 1.08 | -8.31 | 6.48 | TP 2 / BE 2 / SL 2 | 15.54M |
| 2026-06-14 | 50K_TP120 | 11 | 4 | 3.30 | -29.44 | 13.20 | BE 1 / TP 1 / SL 2 | 19.86M |
| 2026-06-14 | 200K_TP60 | 6 | 3 | -0.61 | -10.49 | -1.83 | BE 1 / TP 1 / SL 1 | 19.86M |
| 2026-06-15 | 50K_TP120 | 28 | 20 | 44.98 | 61.84 | 899.67 | TIME 5 / TP 8 / BE 2 / SL 5 | 70.98M |
| 2026-06-15 | 200K_TP60 | 17 | 14 | 30.60 | 52.52 | 428.46 | TP 10 / BE 2 / SL 2 | 70.98M |

## Read

- 50K/TP120 is regime-dependent and unstable in this small replay: weak on 06-07, bad on 06-11, small positive on 06-14, very strong on 06-15.
- 200K/TP60 is more consistent in this replay slice: positive on 06-07, flat-positive on 06-11, flat-negative on 06-14, strong on 06-15.
- 06-15 is not enough to declare edge. It is the best day and can dominate interpretation if treated carelessly.
- The next useful replay is condition analysis: separate strong trend continuation days from chop/failed continuation days, then compare 50K vs 200K under the same regime labels.

Artifacts:
- JSON: `reports/research/s34/S34_MULTI_DAY_REPLAY_2026-06-07_15.json`
