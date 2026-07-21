# S34 Claude Comprehensive Handoff

Generated: `2026-06-30T09:32:32.257671+00:00`

This is a research-only synthesis for Claude. Live executor/config/order logic were not changed.

## Executive Conclusion

- Current live alpha remains `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`.
- The best durable concept is not a new live rule yet; it is a navigation map: SELL silence/reclaim is fade-friendly, SELL propagation is fade-danger/momentum-watch, BUY-side mirror is not deployable.
- Frequency expansion should come from separate shadow lanes after execution gauntlet, not from loosening the live lane blindly.
- The latest frequency suite found map value but no immediate live promotion.

## Latest Frequency Expansion Top Cells

| rank | cell | N | sum | median | T3R | hold T3R |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `tests.event_end_vs_maker.taker.event_end_H4` | 341 | 11525.2 | 34.0 | 10099.3 | 3098.6 |
| 2 | `tests.event_end_vs_maker.taker.event_end_H2` | 341 | 10408.2 | 26.9 | 9231.4 | 2239.1 |
| 3 | `tests.event_end_vs_maker.taker.event_end_H1` | 341 | 9582.2 | 21.7 | 8467.4 | 2129.8 |
| 4 | `tests.event_end_vs_maker.taker.reclaim_H4` | 341 | 9032.6 | 29.9 | 7470.2 | 2552.8 |
| 5 | `tests.event_end_vs_maker.taker.reclaim_H2` | 341 | 7670.0 | 19.2 | 6533.0 | 1454.4 |
| 6 | `tests.event_end_vs_maker.taker.reclaim_H1` | 341 | 6423.6 | 13.4 | 5320.5 | 1492.4 |
| 7 | `tests.event_end_vs_maker.taker.event_end_M15` | 341 | 5115.9 | 13.4 | 4583.9 | 1575.1 |
| 8 | `tests.threshold_expansion.50000.cells.tau30_H2` | 479 | 4507.4 | 9.6 | 2613.5 | -124.8 |
| 9 | `tests.threshold_expansion.50000.cells.tau30_H4` | 479 | 4331.8 | 4.4 | 2614.8 | -409.0 |
| 10 | `tests.threshold_expansion.300000.cells.tau600_H4` | 269 | 4007.0 | 6.3 | 2436.2 | 537.5 |
| 11 | `tests.threshold_expansion.300000.cells.tau900_H4` | 263 | 3731.6 | 12.7 | 2141.2 | 220.7 |
| 12 | `tests.threshold_expansion.300000.cells.tau900_H2` | 263 | 3051.1 | 9.4 | 2021.7 | -588.2 |
| 13 | `tests.threshold_expansion.300000.cells.tau600_H2` | 269 | 2906.9 | 10.9 | 1918.8 | -712.0 |
| 14 | `tests.event_end_vs_maker.taker.reclaim_M15` | 341 | 2890.9 | 7.9 | 2368.1 | 1014.7 |
| 15 | `tests.deepbid_ablation.buckets.spread_bps.LOW_<=0.1` | 102 | 2872.5 | 22.9 | 1666.5 | 1201.7 |
| 16 | `tests.threshold_expansion.300000.cells.tau30_H2` | 134 | 2809.8 | 21.4 | 1758.6 | -126.0 |
| 17 | `tests.sell_silence_lane_expansion.lanes.tau30_all_silence` | 193 | 2795.4 | 26.7 | 1522.6 | 993.3 |
| 18 | `tests.threshold_expansion.200000.cells.tau30_H4` | 193 | 2795.4 | 26.7 | 1522.6 | 993.3 |
| 19 | `tests.threshold_expansion.300000.cells.tau60_H2` | 170 | 2758.2 | 17.9 | 1692.6 | 66.6 |
| 20 | `tests.threshold_expansion.150000.cells.tau30_H4` | 222 | 2628.6 | 12.9 | 1117.7 | -655.9 |

## Settled Results From This Research Block

1. Four-arm symmetry: SELL->LONG is the only materially positive V02-family arm; BUY->SHORT mirror is weak and T3R-negative.
2. Event-chain puzzle: same-side continuation cells are strongly negative for fade; cross-asset next SELL propagation flips SELL fade from positive to dangerous.
3. Propagation suite: silence after SELL shock is fade-friendly; propagation is momentum-watch but not yet causally/execution validated.
4. Candidate gauntlet: broad in-sample candidates failed multiple-comparison correction; strongest raw cells are navigation hypotheses.
5. Execution gauntlet: with book staleness guard, only SELL silence fade remains marginal; propagation momentum collapses under causal executable entry.
6. Next-navigation tests: SELL silence/reclaim at 30-60s is the cleanest navigation state; BUY silence fade remains bad.
7. Frequency expansion: threshold/deepbid/event_end/maker mapping expanded the surface; no new lane is live-ready.

## Detailed Numeric Read

### A. Current Live Lane And Direct Symmetry

- Live configured rule: `S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID`.
- Four-arm test:
  - `SELL -> LONG` baseline: N=11, H4 sum `+1705.9`, median `+161.6`, T3R `+795.5`.
  - `SELL -> SHORT` negative control: N=11, H4 sum `-1554.2`, T3R `-1465.3`.
  - `BUY -> SHORT` mirror: N=17, H4 sum `+95.9`, median `+9.7`, T3R `-443.1`.
  - `BUY -> LONG` negative control: N=13, H4 sum `-463.8`, T3R `-705.8`.
  - Multiple-comparison permutation: observed max T3R `+795.5`, null p95 `+227.1`, p-right `0.002`; this is driven by the existing SELL->LONG lane, not by BUY mirror.
- Interpretation: mirror short does not currently justify a separate live/paper lane. Frequency expansion should not start by mirroring BUY->SHORT.

### B. Event Chain / Puzzle Results

- Same-symbol transition result:
  - `SELL -> BUY` fade: N=233, sum `+13589.9`, median `+53.8`, T3R `+12186.1`.
  - `BUY -> SELL` fade: N=226, sum `+6359.0`, median `+38.7`, T3R `+4935.2`.
  - `SELL -> SELL`: N=283, sum `-15052.0`, median `-28.5`, T3R `-15978.5`.
  - `BUY -> BUY`: N=261, sum `-20520.3`, median `-53.3`, T3R `-21425.3`.
- Anchor vs lifecycle:
  - SELL anchor H4: N=585, sum `-827.0`, T3R `-2230.8`.
  - SELL event_end H4: N=585, sum `+5702.0`, T3R `+4275.2`.
  - SELL reclaim H4: N=543, sum `+3219.3`, T3R `+1633.2`.
  - BUY anchor H4: `-13218.6`; BUY event_end H4: `-5379.7`; BUY reclaim H4: `-6295.2`.
- Cross-asset propagation:
  - ETH SELL fade if no cross-asset next SELL propagation: N=395, sum `+7456.9`, T3R `+6088.9`.
  - ETH SELL fade if cross-asset next SELL propagation exists: N=190, sum `-8283.9`, T3R `-9434.6`.
- Interpretation: cascade lifecycle matters more than raw threshold. SELL shock that ends/reclaims is fade-friendly; same-side/cross-asset propagation is danger.

### C. Propagation Puzzle Suite

- Overall next-same-side rate:
  - SELL propagation rate around `0.49`.
  - BUY propagation rate around `0.506`.
- SELL propagation:
  - Propagation fade H4: N=282, sum `-7612.2`, T3R `-8782.7`.
  - No-propagation fade H4: N=303, sum `+6785.2`, T3R `+5448.3`.
  - Propagation momentum SHORT H1: N=282, sum `+5895.2`, T3R `+4650.2`.
- BUY propagation:
  - Propagation fade SHORT H4: N=277, sum `-12249.1`, T3R `-13586.4`.
  - Propagation momentum LONG H1: N=277, sum `+5934.5`, T3R `+4920.6`.
- SELL silence after shock:
  - Silence: N=287, fade H4 sum `+7743.1`, T3R `+6406.2`.
  - Noisy: N=298, fade H4 sum `-8570.1`, T3R `-9740.6`.
- Composite pressure score:
  - SELL LOW score fade H4: N=97, sum `+2993.0`, T3R `+1958.8`.
  - SELL HIGH score fade H4: N=276, sum `-7832.4`, T3R `-9002.9`.
- Interpretation: propagation is a strong navigation/danger state. It is not yet a deployable momentum alpha because causal executable tests weaken it.

### D. Candidate Gauntlet And Execution Reality

- Broad candidate gauntlet top cells:
  - `SELL_SILENCE_FADE_LONG_H4`, tau 1800: N=345, sum `+9503.0`, median `+29.5`, T3R `+8135.0`, hold T3R `+2819.5`.
  - `BUY_PROPAGATION_MOMENTUM_LONG_H1`, tau 3600: N=194, sum `+7672.0`, T3R `+6658.1`.
  - `SELL_PROPAGATION_MOMENTUM_SHORT_H1`, tau 3600: N=185, sum `+7763.4`, T3R `+6518.4`.
- Multiple-comparison corrected permutation:
  - Observed max T3R `+8135.0`.
  - Null p95 max T3R `+10632.9`.
  - Corrected p-right `0.383`.
  - Verdict: broad candidate search did not clear corrected null.
- Executable/staleness-guarded gauntlet:
  - `SELL_SILENCE_FADE_LONG_H4`, tau 1800, taker0: N=131, sum `+1160.8`, median `+15.0`, T3R `+13.6`, hold T3R `-244.4`.
  - Maker O5: N=90, sum `+900.1`, T3R `+44.5`, fill rate `0.261`.
  - `SELL_PROPAGATION_MOMENTUM_SHORT_H1`, taker0: N=98, sum `-736.7`, median `-10.2`, T3R `-1516.4`.
  - `BUY_PROPAGATION_MOMENTUM_LONG_H1`, taker0: N=99, sum `-1294.3`, median `-17.8`, T3R `-1835.0`.
  - `BUY_SILENCE_FADE_SHORT_H4`, taker0: N=83, sum `-492.0`, median `-1.7`, T3R `-1279.4`.
- Interpretation: the map is real, but executable alpha is not yet proven. The broad negative-control patterns were mostly early-label/entry-price artifacts.

### E. Latest Frequency Expansion Suite

This new suite is a fast mapping pass using `MARK_TAKER_PROXY` and `MARK_MAKER_PULLBACK_PROXY`; it is explicitly not a final live execution test.

- Best broad lifecycle cells:
  - SELL event_end H4: N=341, sum `+11525.2`, median `+34.0`, T3R `+10099.3`, hold T3R `+3098.6`.
  - SELL event_end H2: N=341, sum `+10408.2`, median `+26.9`, T3R `+9231.4`, hold T3R `+2239.1`.
  - SELL reclaim H4: N=341, sum `+9032.6`, median `+29.9`, T3R `+7470.2`, hold T3R `+2552.8`.
- Current V02 vs outside lane:
  - `tau30_all_silence`: N=193, sum `+2795.4`, median `+26.7`, T3R `+1522.6`, hold T3R `+993.3`.
  - `tau30_inside_current_v02_shadow_times`: N=5, sum `+468.6`, median `+99.7`, T3R `+40.8`, hold T3R `+315.7`.
  - `tau30_outside_current_v02_shadow_times`: N=188, sum `+2326.8`, median `+25.7`, T3R `+1054.0`, hold T3R `+677.6`.
  - `tau60_outside_current_v02_shadow_times`: N=232, sum `+1453.0`, median `+18.8`, T3R `+204.4`, hold T3R `+504.4`.
  - Later tau outside lanes degrade: tau300 outside T3R `-2894.2`; tau600 outside T3R `-485.8`.
- Threshold expansion:
  - 50K tau30 H2/H4 look positive in-sample but hold T3R is negative (`-124.8`, `-409.0`).
  - 200K tau30/60/600 H4 survive as map cells, with hold T3R `+993.3`, `+943.1`, `+679.4`.
  - 300K tau600 H4: N=269, sum `+4007.0`, median `+6.3`, T3R `+2436.2`, hold T3R `+537.5`.
  - 300K tau900 H4: N=263, sum `+3731.6`, median `+12.7`, T3R `+2141.2`, hold T3R `+220.7`.
- DeepBid / book ablation on SELL silence tau60:
  - bid_depth HIGH `>207439`: N=35, sum `+1399.3`, median `+22.3`, T3R `+455.7`, hold T3R `+492.7`.
  - bid_depth LOW `<=115762.6`: N=34, sum `+1047.7`, median `+42.2`, T3R `+54.1`, hold T3R `+56.7`.
  - bid_depth MID: N=33, sum `+425.5`, T3R `-268.5`, hold T3R `-598.1`.
  - spread clean bucket: N=102, sum `+2872.5`, median `+22.9`, T3R `+1666.5`, hold T3R `+1201.7`.
- Cross-asset lead:
  - Most cross-asset lead cells are not robust; many positive medians have negative T3R.
  - SOL SELL prev900s ETH fade H4: N=37, sum `+1093.2`, median `+35.3`, T3R `+315.8`, hold T3R `+17.6`; too small and weak as standalone.
- Interpretation: there is a plausible frequency-expansion map outside current V02, especially early SELL silence/reclaim. But because this is mark/proxy and not book-staleness/queue-realistic, the correct next action is shadow/execution gauntlet, not live promotion.

## Current Working Model

Cascade is no longer treated as a single event. The better model is a state sequence:

```text
SELL shock -> silence/reclaim -> fade-friendly recovery
SELL shock -> same-side/cross-asset propagation -> fade danger / momentum watch
BUY shock -> mirror does not symmetrically validate
```

## Open Questions For Next Round

1. Can SELL silence/reclaim be made executable with maker pullback/reclaim entry without losing holdout T3R?
2. Can propagation pressure be detected before 900-1800s using tick/book features rather than future event labels?
3. Is deep_bid an independent causal condition, or just a proxy for the current V02 sample?
4. Can cross-asset lead be converted into a permission/avoidance tag rather than a directional entry?
5. Should H4 shadow be promoted only as management/navigation for current V02, not a separate live entry?

## Source Reports

- `D:\eclipse_scalper\reports\research\s34\S34_V02_MANAGEMENT_NAVIGATION_SUITE.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_FOUR_ARM_SYMMETRY_TESTS.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_EVENT_CHAIN_PUZZLE_TESTS.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_PROPAGATION_PUZZLE_SUITE.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_PROPAGATION_CANDIDATE_GAUNTLET.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_CANDIDATE_EXECUTION_GAUNTLET.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_NEXT_NAVIGATION_TESTS.md` (exists)
- `D:\eclipse_scalper\reports\research\s34\S34_V02_FREQUENCY_EXPANSION_TESTS.md` (exists)

## Guardrail

No candidate here should be promoted live without: causal entry, executable book fill, chronological holdout positive, T3R positive after top winners removed, and preferably forward shadow.
