# Alpha Discovery Tests - 2026-06-04

## Scope

No new data sources were used. This run tested existing local data only:

- sibling lanes around known forced-flow candidates
- cross-asset forced-flow transfer
- S34 quality filters
- anti-alpha rejection filters
- fold stability
- 2/4/8/10 bps fee survival
- current shadow telemetry outcomes

Command:

```powershell
python tools\alpha_discovery_tests.py --db data\microstructure.db --telemetry-path logs\telemetry.jsonl --max-events 300 --min-n 20 --out-md reports\ALPHA_DISCOVERY_TESTS.md --out-json reports\ALPHA_DISCOVERY_TESTS.json
```

## Result

- candidates tested: `690`
- promote to shadow: `19`
- watch only: `69`
- confirmed rejection filters: `3`
- shadow telemetry outcomes: `0` rows so far

## Best New Shadow Candidates

| rank | candidate | n | win rate | mean bps | net after 8 bps | net8 folds |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `BTCUSDT_BUY100000_SHORT_900_UTC07` | 22 | 90.91% | +45.19 | +37.19 | 5/5 |
| 2 | `ETHUSDT_BUY250000_SHORT_900_UTC14` | 33 | 75.76% | +43.13 | +35.13 | 4/5 |
| 3 | `ETHUSDT_BUY1000000_SHORT_900_SESSION_US` | 20 | 70.00% | +34.74 | +26.74 | 4/5 |
| 4 | `SOLUSDT_BUY50000_SHORT_900_FUNDING_NEGATIVE` | 20 | 85.00% | +31.83 | +23.83 | 4/5 |
| 5 | `ETHUSDT_BUY250000_SHORT_900_UTC19` | 26 | 73.08% | +30.45 | +22.45 | 5/5 |
| 6 | `ETHUSDT_S34_SHORT_900_SESSION_US` | 25 | 72.00% | +29.01 | +21.01 | 5/5 |
| 7 | `SOLUSDT_BUY25000_SHORT_900_FUNDING_NEGATIVE` | 34 | 82.35% | +27.39 | +19.39 | 4/5 |
| 8 | `ETHUSDT_S34_SHORT_900_BASIS_POSITIVE` | 31 | 80.65% | +27.38 | +19.38 | 4/5 |
| 9 | `ETHUSDT_BUY500000_SHORT_900_SESSION_US` | 62 | 72.58% | +25.86 | +17.86 | 4/5 |
| 10 | `BTCUSDT_SELL250000_LONG_900_UTC13` | 23 | 69.57% | +25.44 | +17.44 | 4/5 |

## Interpretation

The strongest new discovery is a BTC forced-flow lane:

`BTCUSDT_BUY100000_SHORT_900_UTC07`

This was not in the prior shadow set. It has only 22 events, but the fold and fee profile are clean enough to shadow-test.

The ETH forced-flow result strengthened:

- `ETHUSDT_BUY250000_SHORT_900_UTC14` remains high value.
- `ETHUSDT_BUY250000_SHORT_900_UTC19` is a new sibling lane worth shadowing.
- `ETHUSDT_BUY1000000_SHORT_900_SESSION_US` suggests larger ETH forced-buy events in US session may be cleaner than the 500k version.

The SOL result strengthened:

- `SOLUSDT_BUY50000_SHORT_900_FUNDING_NEGATIVE` remains good.
- `SOLUSDT_BUY25000_SHORT_900_FUNDING_NEGATIVE` expands the threshold lower, but should be tracked separately because it may be noisier.

S34 filters are now very clear:

- good: `SESSION_US`, `BASIS_POSITIVE`, `CONFIDENCE_MEDIUM`
- bad: `CLUSTERED`, `SESSION_NON_US`, `BASIS_NONPOSITIVE`

## Confirmed Rejection Filters

| candidate | n | win rate | mean bps | net after 8 bps |
| --- | ---: | ---: | ---: | ---: |
| `ETHUSDT_S34_SHORT_900_CLUSTERED` | 44 | 47.73% | -5.67 | -13.67 |
| `ETHUSDT_S34_SHORT_900_SESSION_NON_US` | 48 | 52.08% | -8.16 | -16.16 |
| `ETHUSDT_S34_SHORT_900_BASIS_NONPOSITIVE` | 42 | 42.86% | -12.26 | -20.26 |

These should be treated as skip/rejection filters for S34 shadow analysis.

## Action Completed

These new lanes were added to the shadow emitter as `SHADOW_ONLY`:

- `BTCUSDT_BUY100000_SHORT_900_UTC07`
- `ETHUSDT_BUY250000_SHORT_900_UTC19`
- `ETHUSDT_BUY1000000_SHORT_900_SESSION_US`
- `SOLUSDT_BUY25000_SHORT_900_FUNDING_NEGATIVE`
- `ETHUSDT_S34_SHORT_900_CONFIDENCE_MEDIUM`

Runtime family names:

- `BTC_BUY100K_SHORT_900_UTC07`
- `ETH_BUY250K_SHORT_900_UTC19`
- `ETH_BUY1M_SHORT_900_SESSION_US`
- `SOL_BUY25K_SHORT_900_FUNDING_NEGATIVE`
- `S34_SHORT_900_CONFIDENCE_MEDIUM`

No live execution was promoted. All additions remain `SHADOW_ONLY`.
