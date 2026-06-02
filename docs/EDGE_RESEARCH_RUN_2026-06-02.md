# Edge Research Run - 2026-06-02

## Scope

No new data sources were added. DeFi liquidation linkage and deeper L1/depth capture remain parked for later.

This run used only the current local SQLite data in `data/microstructure.db` and added two research-only tools:

- `tools/research_narrow_event_lanes.py`
- `tools/research_lane_candidate_validation.py`

## New Finding

The strongest new idea is not a new raw signal. It is lane-conditioned forced-flow:

> ETH forced BUY liquidation shorts work much better during specific US-session and UTC-hour lanes.

This is separate from the prior S34 single-large result and gives us another possible alpha path:

- broad event: ETH BUY liquidation
- direction: SHORT
- horizon: 900s
- best current lane: UTC hour 14
- more robust current lane: US session

## Targeted Validation Results

| candidate | decision | n | win rate | mean bps | uplift bps | folds positive | net after 8 bps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_BUY250K_SHORT_900_UTC14` | shadow test | 34 | 76.47% | +43.28 | +36.34 | 4/5 | +35.28 |
| `SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE` | shadow test | 20 | 85.00% | +31.83 | +16.05 | 5/5 | +23.83 |
| `S34_SHORT_900_SESSION_US` | shadow test | 25 | 72.00% | +29.01 | +24.44 | 5/5 | +21.01 |
| `S34_SHORT_900_BASIS_POSITIVE` | shadow test | 31 | 80.65% | +27.38 | +22.81 | 4/5 | +19.38 |
| `ETH_BUY500K_SHORT_900_SESSION_US` | shadow test | 62 | 72.58% | +25.86 | +18.34 | 4/5 | +17.86 |
| `S34_SHORT_900_SINGLE_LARGE` | shadow test | 21 | 80.95% | +17.32 | +12.75 | 4/5 gross, 3/5 after 8 bps | +9.32 |
| `ETH_BUY500K_SHORT_900_UTC14` | reject for now | 15 | 80.00% | +60.81 | +53.29 | 5/5 | +52.81 |
| `BTC_SELL100K_LONG_900_UTC13` | watch only | 44 | 75.00% | +21.14 | +14.65 | 3/5 | +13.14 |

## Interpretation

### Promote To Shadow

1. `ETH_BUY250K_SHORT_900_UTC14`
   - Highest mean and uplift.
   - Sample is still modest, but it has 34 events and 4/5 positive chronological folds.
   - Treat as a narrow event-lane alpha candidate.

2. `ETH_BUY500K_SHORT_900_SESSION_US`
   - Lower mean than the UTC14 lane but more robust sample at 62 events.
   - This is the better generalization candidate.

3. `S34_SHORT_900_SESSION_US`
   - 5/5 positive folds.
   - This suggests the existing S34 family should be time-lane aware.

4. `S34_SHORT_900_BASIS_POSITIVE`
   - Reinforces the earlier basis-positive S34 branch.
   - Keep as a quality filter.

5. `SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE`
   - Improves the already-promising SOL forced-flow candidate.
   - Small sample at 20 events, but all folds are positive gross.

### Do Not Promote Yet

1. `ETH_BUY500K_SHORT_900_UTC14`
   - Looks excellent but only has 15 events.
   - Keep as a sub-lane to monitor inside the broader ETH forced-flow shadow lane.

2. `BTC_SELL100K_LONG_900_UTC13`
   - Good aggregate stats but only 3/5 positive folds.
   - Watch only, no shadow promotion yet.

## New Alpha Map

| rank | alpha path | status |
| ---: | --- | --- |
| 1 | ETH BUY liquidation -> SHORT, UTC14 / US session | new shadow candidate |
| 2 | SOL BUY liquidation -> SHORT, funding negative | stronger version of existing SOL shadow candidate |
| 3 | ETH S34 -> SHORT, US session | new S34 time-lane filter |
| 4 | ETH S34 -> SHORT, basis positive | confirmed S34 quality filter |
| 5 | ETH S34 -> SHORT, single large | confirmed but fee-sensitive |
| 6 | BTC SELL liquidation -> LONG, UTC13 | watch only |

## Next Actions

1. Add shadow-only specs for:
   - `ETH_BUY250K_SHORT_900_UTC14`
   - `ETH_BUY500K_SHORT_900_SESSION_US`
   - `SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE`
   - `S34_SHORT_900_SESSION_US`
   - `S34_SHORT_900_BASIS_POSITIVE`
2. Keep all of them out of live execution.
3. Track forward-only samples until each has at least 100 new events or a clear failure.
4. Keep DeFi and deeper L1/depth capture parked until after the shadow lanes show whether these event families still fire cleanly.

## Artifacts

- `reports/NARROW_EVENT_LANE_ALPHA_SCAN.md`
- `reports/NARROW_EVENT_LANE_ALPHA_SCAN.json`
- `reports/LANE_CANDIDATE_VALIDATION.md`
- `reports/LANE_CANDIDATE_VALIDATION.json`

