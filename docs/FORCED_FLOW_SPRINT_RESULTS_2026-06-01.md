# Forced Flow Sprint Results - 2026-06-01

## Executive Decisions

| Path | Decision | Reason |
| --- | --- | --- |
| SOL BUY liquidation -> SHORT | Promote to shadow candidate | Strongest untapped path found: 50k threshold, 900s horizon, 46 events, 73.91% win rate, +15.78 bps gross mean, survives 2-10 bps round-trip fee stress with 4/5 positive folds. |
| ETH S34 single-large branch | Promote to forward validation | DB-native validation shows single-large sharply separates from clustered states: 21 events, 80.95% win rate at 120s, +10.78 bps mean; 80.95% at 900s, +17.37 bps mean. |
| S34 bad-fingerprint rejection | Keep, but narrow to single-large / basis-positive | Clustered states are actively bad in the DB-native validator, while basis-positive single-large is best. Book-state subgroups are too small after current recovery annotation. |
| True L1 book / queue-state capture | Build instrumentation, not alpha yet | bookTicker is live and dense, but simple bookTicker filters did not improve Stage C. The value is now in queue-state/recovery diagnostics around known events, not generic imbalance filters. |
| Venue / fee / routing edge | Keep as first-class gate | Fees decide viability. SOL survives up to 10 bps round-trip gross stress; broad ETH liquidation does not. Passive and passive-then-taker need more realistic fill and fee handling before promotion. |
| Passive-then-taker refresh | Freeze | 21-day validation produced 0/12 passes with insufficient fill-rate failures. Treat as harness/audit debt before any live claim. |
| Event-lane conditional discovery | Blocked by tooling/data path | `check_event_lanes` timed out on 24h and returned no data on 60m. Needs indexing or a narrower lane extractor before more discovery. |
| DeFi liquidation linkage | Blocked by missing collector/credentials | No local DeFi liquidation table and no runnable collector without a tokenized Gateway endpoint. |
| SOL forced-flow transfer later | Pull forward to shadow | Contrary to the original ordering, SOL already has enough seed history for a shadow-only candidate. |

## Highest-Value New Path

### `SOL_BUY_LIQ_SHORT_V1`

Candidate:

- Symbol: `SOLUSDT`
- Trigger: forced BUY liquidation notional `>= 50000`
- Direction: SHORT
- Horizon: 900 seconds
- Initial mode: `SHADOW_ONLY`

Walk-forward taker result:

| Metric | Value |
| --- | ---: |
| Events | 46 |
| Filled | 46 |
| Win rate | 73.91% |
| Gross mean | +15.78 bps |
| Gross median | +15.33 bps |
| Positive folds | 4/5 |
| 2 bps RT net mean | +13.78 bps |
| 4 bps RT net mean | +11.78 bps |
| 8 bps RT net mean | +7.78 bps |
| 10 bps RT net mean | +5.78 bps |

Fold risk:

| Fold | N | Win Rate | Mean Bps |
| --- | ---: | ---: | ---: |
| 1 | 9 | 77.78% | +25.74 |
| 2 | 9 | 77.78% | +18.77 |
| 3 | 9 | 33.33% | -12.39 |
| 4 | 9 | 100.00% | +24.50 |
| 5 | 10 | 80.00% | +21.65 |

This is not live-trading ready because one fold is materially bad and sample size is still small. It is good enough to emit shadow signals, persist realized forward returns, and test routing/fee assumptions on every new event.

## ETH S34 Validation

DB-native validator result:

| Branch | Horizon | N | Win Rate | Mean Bps |
| --- | ---: | ---: | ---: | ---: |
| all | 120s | 73 | 54.79% | +2.00 |
| all | 900s | 73 | 60.27% | +5.55 |
| single_large | 60s | 21 | 85.71% | +8.02 |
| single_large | 120s | 21 | 80.95% | +10.78 |
| single_large | 300s | 21 | 85.71% | +12.26 |
| single_large | 900s | 21 | 80.95% | +17.37 |
| clustered | 120s | 44 | 36.36% | -5.47 |
| clustered | 900s | 44 | 50.00% | -4.54 |
| single_large_basis_pos | 120s | 10 | 90.00% | +11.97 |
| single_large_basis_pos | 900s | 10 | 90.00% | +31.78 |

The actionable edge is not "S34 generally"; it is single-large, preferably basis-positive. Clustered/resumed/slow-burn style states should be rejected until proven otherwise.

## SOL Independence Check

The SOL candidate has no overlap with formal ETH detector rows, but it often occurs during broader liquidation stress:

| Slice | N | Win Rate | Mean Bps |
| --- | ---: | ---: | ---: |
| all | 46 | 73.91% | +15.78 |
| no ETH detector overlap | 46 | 73.91% | +15.78 |
| ETH big BUY overlap | 32 | 71.88% | +16.18 |
| no ETH big BUY overlap | 14 | 78.57% | +14.88 |
| BTC big BUY overlap | 25 | 72.00% | +15.61 |
| no BTC big BUY overlap | 21 | 76.19% | +15.99 |

Interpretation: this is not just a duplicate of the formal ETH detector, but it is not fully independent of broad market stress. Shadow logging should include ETH/BTC forced-flow overlap flags.

## Negative / Blocked Results

| Test | Result |
| --- | --- |
| Stage C bookTicker filters | Failed to improve the Apr11+ baseline enough. High quote intensity gave only +0.349 bps uplift. |
| S69 OBI entry | Insufficient evidence because the current DB has `book_ticker`, not full orderbook snapshots. |
| S37 mirror long | Weak; ETH SELL liquidation -> LONG did not hold. |
| S44 BTC contagion | No edge; isolated events did not beat contagion-filtered events. |
| S45 fragility reversal | No edge; prior extreme/cold -> goldilocks transition was bad. |
| S70 funding regime | Insufficient evidence. |
| Passive pocket 21D refresh | 0/12 pass count across baseline and passive-then-taker variants. |
| Stage F breakthrough | Timed out after 304s; no result used. |
| Event lanes | 24h timed out, 60m returned no data. |
| DeFi linkage | Missing local table and collector credentials. |

## Artifacts

Primary reports:

- `reports/SOL_BUY_LIQ_SHORT_V1_WF.md`
- `reports/SOL_BUY_LIQ_SHORT_V1_PASSIVE.md`
- `reports/SOL_BUY_LIQ_SHORT_V1_PTT.md`
- `reports/SOL_ETH_CONFLICT_CHECK.md`
- `reports/S34_SINGLE_LARGE_V1_VALIDATE.md`
- `reports/ETH_BUY_LIQ_SHORT_BASELINE_WF.md`
- `reports/S49_SINGLE_LARGE_VALIDATION.md`
- `reports/BOOKTICKER_VALIDATION.md`
- `deep_discovery/reports/stage_c_book_ticker.md`

New tools:

- `tools/forced_flow_candidate_harness.py`
- `tools/s34_single_large_v1_validate.py`
- `tools/sol_eth_conflict_check.py`
- `tools/research_sol_forced_flow_transfer.py`

## Next Steps

1. Add a shadow emitter for `SOL_BUY_LIQ_SHORT_V1` that logs triggers, no orders.
2. Add a forward-validation lane for `S34_SINGLE_LARGE_V1`, restricted to single-large and basis-positive states.
3. Build real L1 queue-state capture around liquidation events: spread recovery time, bid/ask depth recovery, quote intensity decay, and passive fill probability.
4. Audit passive pocket validation because the 21-day run produced zero passes and no usable live candidate.
5. Fix event-lane extraction with narrower indexed queries before running conditional discovery again.
6. Implement DeFi liquidation ingestion only after credentials and schema are available.

