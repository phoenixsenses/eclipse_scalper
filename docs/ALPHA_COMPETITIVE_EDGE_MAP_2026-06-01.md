# Alpha And Competitive Edge Map

Date: 2026-06-01

Purpose: map the current Eclipse alpha stack, competitive advantages, blockers, and the most promising paths for genuinely new opportunity discovery.

## Executive Read

Eclipse currently has one real but economically constrained microstructure edge family: ETH forced-flow / cascade continuation. The strongest measured high-frequency variant is statistically real, stable across development splits, and execution-sensitive. It is not viable at standard taker fees. It may become viable with VIP taker fees, better venue economics, or a genuinely stronger conditioning dimension.

The broader S34 event-level ETH cascade blueprint is more strategic and more differentiated. It has a coherent forced-flow thesis, stronger descriptive event returns, and rich shadow features, but it is still sample-limited and not yet forward-validated enough for aggressive promotion.

Most broad alternative alpha families tested so far are killed or weak in their current forms: naive lead-lag, broad funding reversion, broad OHLCV/stat-arb, liquidation reversal passive pockets, and generic 100ms candidate discovery.

The next untapped opportunities are not likely to come from another generic indicator sweep. They are more likely to come from data-right advantages: real top-of-book capture, event-lane context, DeFi liquidation linkage, venue/fee routing, and execution toxicity intelligence.

## Current Alpha Inventory

| Edge | Status | Evidence | Economic Read | Main Blocker |
|---|---|---|---|---|
| ETH microstructure burst continuation, taker onset | Conditional | `EE_TAKER_VIP_ONSET`: +3.12 bps gross, 4/4 WF splits, N=3068 | Viable only near <=2 bps/side taker; fails standard fees | fee tier, holdout not run |
| ETH microstructure amplification stack | Real but insufficient | best dev stack +4.29 bps gross, 4/4 positive WF splits | gross edge too small for standard 10 bps RT cost | amplification ceiling |
| S34 ETH forced-flow continuation short | Research-grade candidate | conditioned sample around 65-68% WR, N around 43-46; invalidation branch separates strongly | strategically interesting, not yet deployable | sample size, live execution, feed quality |
| S35 BTC-led S34 overlay | Monitor-only | btc_led N=4, 75% WR vs eth_only 66.67% | possible sizing/confidence overlay only | tiny sample; independent lead-lag audit negative |
| Passive-then-taker ETH 60s pockets | Experimental | multiple 7D tight pockets flipped from negative to positive | useful execution profile in narrow pocket family | short window, needs refresh/replication |
| BTC cascade continuation short | Hypothesis-ready | strategy slot says next high-priority slot, weaker than ETH | likely learning value more than immediate alpha | backtest/validation pending |
| ETH long mirror | Hypothesis stage | same forced-flow family, regime-opposed | diversification candidate | no signed evidence package |
| Funding regime reversion | Killed in current broad form | initial sweep killed by WR, WF degradation, parameter instability, recent absence | not promotable as generic funding signal | instability; low persistence |
| Naive cross-asset lead-lag | Killed in current form | all credible N>=200 rows net negative | wrong sign after costs | framing/economics failure |
| Liquidation reversal passive pockets | Blocked | raw coverage exists but tradeable pockets fail fill-rate gates | not usable with current execution shape | fillability/execution shape |
| 100ms alpha discovery grid | Empty | 600 candidates, 0 robust selected | no current robust alpha | data/sample/candidate quality |
| Event lanes | Context layer | spread stress, return shock, volatility burst, volume vacuum, book proxy pressure bridge exists | potentially valuable as filters/regime labels | descriptive, not causal yet |
| Fill toxicity / latency stress lanes | Monitoring edge | tools/reporting exist, current toxicity rows 0 | competitive defense and future execution gate | needs realized trade sample |

## Competitive Edges We Actually Have

1. Event-specific microstructure doctrine.
   Eclipse is not built around generic bars. The best work is centered on forced flow, liquidation seeds, intensity decay, cascade fingerprints, resumption, purity, entropy, and event-specific invalidation.

2. A strong kill discipline.
   Several plausible alpha families have been killed rather than stretched: funding reversion, naive lead-lag, broad OHLCV/stat-arb, generic 100ms discovery, and passive liquidation reversal pockets.

3. Execution-aware research.
   The repo distinguishes signal return from realized fill economics. The execution engineering report explicitly found passive adverse selection and separated taker/VIP economics from standard-fee failure.

4. Shadow feature richness around S34.
   Current detector/research fields include purity, loss risk, fingerprint class, entropy, OFI-like flow, first-of-day, BTC-led labels, entry tier, invalidation state, and resumption behavior.

5. Operational stack depth.
   There is infrastructure for paper/live runtime, live parity gates, detector signal state, risk gates, canary policy, event lanes, stream health, alert contracts, and post-trade diagnostics.

6. The ability to preserve holdouts.
   Reports explicitly preserve May 08-May 15 holdout windows when development evidence is insufficient. That is a real research advantage.

## Competitive Weaknesses

1. Fee economics dominate the highest-confidence short-horizon signal.
   A +3 to +4.3 bps gross edge is real but fragile under normal retail taker fees.

2. True book data is missing or proxy-only in key places.
   `book_proxy_pressure`, proxy spread, and OFI-like features are not the same as true bid/ask/depth event flow. This limits mechanism certainty.

3. Live feed availability has been unstable.
   Stream health reports show starved liquidation, aggTrade, and mark streams in some snapshots. A forced-flow strategy dies immediately if forceOrder/data freshness dies.

4. S34 is still short-sample.
   The most interesting event-level edge is based on roughly 45 days / about 43-46 conditioned observations in the core reports. It is coherent, not proven.

5. Cross-asset assumptions have been weakened.
   Independent BTC->ETH lead-lag checks were strongly negative in the naive trade framing, and detector BTC-led metadata has shown inconsistency.

6. Current event lanes are mostly descriptive.
   They help explain regimes, but they are not yet signed causal filters or deployable strategy gates.

## Edge Families By Opportunity Quality

### Tier 1: Highest Expected Value

#### Real L1 Book / Queue State Edge

Untapped question: does S34 win because forced flow decelerates before liquidity and maker inventory normalize?

Why it matters:
- This directly tests the central S34 mechanism.
- Current `spread`, `microprice`, `top_depth_imbalance`, and `book_proxy_pressure` are proxy-limited.
- A real book feed can separate clean losers before entry, identify maker return speed, and quantify capacity.

Research program:
- Persist best bid/ask, bid/ask quantity, spread, microprice, queue imbalance, quote update rate, and quote recovery after liquidation seeds.
- Rebuild S34 features into pre-entry book recovery clocks.
- Test whether clean vs resumed branches are separable before entry.

Decision gate:
- Promote only if book-derived variables improve S34 OOS expectancy or reduce bad-branch entry rate without overcutting sample.

#### Venue/Fee/Routing Edge

Untapped question: can the real signal become profitable through venue choice rather than signal discovery?

Why it matters:
- Execution engineering found VIP taker economics are the only viable path for `EE_TAKER_VIP_ONSET`.
- This is a business/market-access edge, not a model edge.

Research program:
- Compare Binance VIP, OKX, Bybit, Gate.io, and maker rebate structures.
- Re-run onset strategy under realistic exchange-specific fee/slippage/latency.
- Track whether the same forced-flow event appears earlier/cleaner on another venue.

Decision gate:
- Need confirmed <=2 bps/side taker economics or equivalent routing improvement before treating high-frequency taker onset as live-candidate.

#### S34 Bad-Fingerprint Classifier

Untapped question: can slow-burn / low-purity / resumed bad branches be filtered before entry?

Why it matters:
- S34 deterioration looked like signal-mix deterioration, not clean edge death.
- Fingerprint cluster 0 had 75.76% WR while cluster 1 had 28.57% WR in the frontier report.
- High entropy and purity branches look materially better than low-quality branches.

Research program:
- Freeze pre-entry-only fingerprint definitions.
- Test purity, rise time, entropy, OFI proxy, liquidity state, and resumption risk without post-entry leakage.
- Build a bad-family reject model first, not a more selective “perfect signal” model.

Decision gate:
- Reject model must preserve enough frequency while improving forward net EV, not just historical WR.

### Tier 2: Attractive But Needs Care

#### Passive-Then-Taker Execution Profile

Untapped question: is passive-then-taker a reusable execution alpha for narrow ETH 60s pockets?

Why it matters:
- It rescued multiple 7D ETH pockets and improved fillability to 100% in the documented decision.
- It may be a way to monetize otherwise fill-toxic surfaces.

Research program:
- Refresh over 21D/60D windows.
- Compare against pure taker, passive, scratch, and passive-then-taker.
- Test fee sensitivity and adverse selection by pocket tightness.

Decision gate:
- Median pocket family result must stay positive after refresh; not enough for one isolated pocket to pass.

#### DeFi Liquidation Linkage

Untapped question: do on-chain liquidation waves or collateral stress explain ETH-specific continuation better than CEX forceOrder alone?

Why it matters:
- S34 theory specifically suspects ETH platform feedback.
- DeFi liquidation data was ranked second-highest information value after bookTicker.

Research program:
- Add Aave/Compound/Maker-style liquidation feeds or snapshots.
- Align DeFi liquidation clusters with ETH CEX liquidation seeds.
- Test whether DeFi stress predicts clean continuation, resumption, or buyer absorption.

Decision gate:
- Must improve branch separation or pre-seed fragility, not merely correlate with volatility.

#### SOL Forced-Flow Transfer

Untapped question: is SOL a better future cascade asset than BTC once ETH edge decays?

Why it matters:
- S34 honest assessment names SOL as the next candidate in theory due to retail leverage and slower repair dynamics.
- It may diversify away from ETH-specific saturation.

Research program:
- Wait for 60-90 deduplicated SOL buy-liquidation seeds above a defensible threshold.
- Mirror ETH testing order: seed-only continuation, fragility, intensity decay, invalidation, execution.

Decision gate:
- Review only if SOL conditioned 60-signal WR exceeds roughly 64% and ETH rolling quality degrades.

### Tier 3: Keep As Context, Not Primary Search

#### Funding/Basis Regime Context

The broad funding reversion strategy is killed, but funding may still be useful as a context variable for cascade fragility or crowdedness. Do not revive it as a standalone directional signal without a new hypothesis.

#### Cross-Asset Overlay

Naive lead-lag is killed. S35 should remain an S34 context label, not a standalone cross-asset strategy, until larger event-conditioned samples justify it.

#### Event Lanes As Filters

Spread stress, return shock, volatility burst, volume vacuum, and book proxy pressure should be used to annotate candidate rows and find conditional pockets. They should not mutate live behavior until bridge reports show stable OOS uplift.

## Paths To Entirely New Untapped Opportunities

1. Build a true book-state dataset and convert S34 from a liquidation/intensity strategy into a liquidity-recovery strategy.

2. Treat exchange fee tier and venue routing as alpha infrastructure. The same signal changes from dead to viable depending on taker cost.

3. Search for bad-state avoidance rather than stronger entries. Filtering slow-burn, low-purity, resumed, maker-recovered states may unlock more EV than finding rarer perfect entries.

4. Add Ethereum-native stress data. If ETH forced-flow edge is platform-specific, CEX-only data may be missing the actual causal variable.

5. Use event lanes to discover conditional alpha surfaces. Ask “when does this pocket work?” before asking “is this pocket globally good?”

6. Separate signal alpha from execution alpha. Passive, taker, passive-then-taker, scratch, and time-exit variants should be evaluated as different products, not as implementation details.

7. Explore aftermath regimes. The S34 second-leg / post-cascade pattern suggests there may be a separate T+25m to T+45m regime after the first forced-flow trade exits.

8. Create a live deterioration classifier. Fill toxicity, latency stress, spread stress, and stream health can become a competitive defense layer that preserves edge by refusing to trade when the market plumbing is unfavorable.

## Recommended Research Order

1. Confirm data plumbing: true top-of-book persistence, forceOrder health, aggTrade health, mark health.
2. Run the preserved holdout only after the fee/venue path is realistic.
3. Build S34 book-state mechanism tests.
4. Build the S34 bad-fingerprint reject model using only pre-entry fields.
5. Refresh passive-then-taker pocket family over longer windows.
6. Add DeFi liquidation context for ETH.
7. Prepare SOL forced-flow transfer only after enough SOL seed history exists.
8. Revisit funding and cross-asset only as context overlays, not standalone strategies.

## Source Trail

- `execution_engineering/FINAL_EXECUTION_ENGINEERING_REPORT.md`
- `signal_amplification/reports/GATE_B_PRE_HOLDOUT.md`
- `docs/research/S34_SIGNAL_DEFINITION.md`
- `docs/research/S34_HONEST_ASSESSMENT_APR2026.md`
- `docs/research/S34_NEXT_FRONTIER.md`
- `docs/research/S35_FULL_DESIGN.md`
- `docs/strategies/STRATEGY_SLOTS.md`
- `docs/strategies/EDGE_TAXONOMY.md`
- `docs/RESEARCH_FEATURE_CAPABILITY_MAP.md`
- `reports/research/lead_lag/09_initial_sweep.md`
- `reports/research/lead_lag/11_s34_premise_check.md`
- `reports/research/funding_reversion/09_initial_sweep.md`
- `docs/PASSIVE_THEN_TAKER_DECISION.md`
- `docs/RESEARCH_LIQ_REVERSAL_SURFACE_NOTES.md`
- `docs/RESEARCH_EVENT_SIGNAL_BRIDGE.md`
- `docs/RESEARCH_BOOK_PROXY_PRESSURE_LANE.md`
- `docs/RESEARCH_FILL_TOXICITY_LANE.md`
- `docs/RESEARCH_LATENCY_STRESS_LANE.md`
- `reports/STREAM_HEALTH.md`
- `reports/STARVATION_VERDICT.md`
- `reports/alpha_discovery_ETHUSDT_100ms.md`
