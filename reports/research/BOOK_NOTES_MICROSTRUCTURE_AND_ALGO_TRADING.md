# Eclipse / S34: A Practitioner's Field Notes
## Market Microstructure, Algorithmic Edge, and the Discipline of Forward Validation

**Started:** 2026-06-09  
**Last updated:** 2026-06-28  
**Author:** Eclipse Research Team

**Source texts:**
- Harris, L. — *Trading and Exchanges: Market Microstructure for Practitioners*
- Chan, E. — *Algorithmic Trading: Winning Strategies and Their Rationale*

**What this document is:**  
Not a book summary. A living extraction — each concept filtered through one question: what does this mean for Eclipse/S34 right now? Theory earns its place only if it changes how we build or evaluate.

---

## Chapter 1: The Hypothesis

### 1.1 Why Liquidation Cascades

Every position in a perpetual futures market is either voluntary or forced. When price moves sharply against leveraged longs or shorts, exchanges liquidate those positions automatically. A large enough cluster of forced liquidations — a cascade — creates a temporary dislocation: order flow that is not information-driven, not directional in the usual sense, but mechanical. It has to happen regardless of fundamentals.

The S34 hypothesis is that this mechanical flow produces a predictable short-term price reaction that can be traded. Not because cascades always reverse. Not because they always continue. But because the market structure around a large forced-flow event is temporarily different from normal, and that difference is measurable.

This is a microstructure alpha, not a macro alpha. It does not require a view on where ETH is going in a week. It requires a view on where ETH is going in the next 3 to 10 minutes after a significant liquidation cluster.

### 1.2 The Core Signal

The raw input is Binance Futures liquidation data: forced order events with symbol, side, quantity, and timestamp. A BUY-side forced order means a short position was liquidated — the exchange bought back the short. A SELL-side forced order means a long position was liquidated.

A large cluster of BUY-side liquidations (shorts getting squeezed) is taken as a candidate signal for a LONG entry. A large cluster of SELL-side liquidations (longs getting washed) is a candidate for a SHORT entry.

The threshold for "large enough" is the first calibration parameter. Too low: noise fills the signal. Too high: too few events for statistical work.

### 1.3 What This Document Tracks

This is a record of what we have learned, built, and validated — in order. It covers:
- what market microstructure theory predicts about our signal
- what our edge discovery sprints found empirically
- what execution engineering revealed about real-world costs
- the pre-registration framework we use to prevent self-deception
- current route status across all eight registered routes
- what remains open and what the next decision points are

---

## Chapter 2: Microstructure Foundations (Harris)

### 2.1 Liquidity Is Multi-Dimensional

Liquidity is not "can I trade." It has at least four dimensions:
- **Tightness** — the bid/ask spread, the immediate cost of crossing
- **Depth** — how much size is available near the best price
- **Immediacy** — how fast the trade can be executed at a given price
- **Resiliency** — how quickly the order book recovers after absorbing a large order

S34 implication: a liquidation cascade is the moment when all four dimensions deteriorate simultaneously. The event that generates our signal is also the moment with the worst execution conditions. Paper fills at mark price during liquidation bursts are optimistic by construction. Real fills during a 5M notional cascade cluster will cross spread, may move price further, and will face a book that has just been partially consumed.

This is not a reason to abandon the strategy. It is a reason to model execution realistically before trusting paper results.

### 2.2 Bid/Ask Spread as Adverse Selection Compensation

Market makers set spreads partly to protect themselves against informed flow. When they cannot distinguish informed from uninformed order flow, they widen the spread to compensate for the expected losses against traders who know something they do not.

S34 implication: during a liquidation cascade, market makers face a wall of forced flow. They cannot easily determine whether the cascade is mechanical (uninformed) or preceded by informed short-selling. In practice, cascades attract both. Spreads widen. Passive fills during the worst part of the cascade may be fills against continuing adverse flow, not fills that represent a good entry.

The execution engineering sprint (Phase 4, May 2026) confirmed this: passive maker fills during high-volatility cascade windows had significantly worse real outcomes than mark-price paper fills assumed. Taker fills at the entry moment — crossing spread deliberately — were more reliable because they guaranteed participation at a defined price.

### 2.3 Stop Levels Are Not Private

Markets contain participants who infer where large clusters of stops and liquidation triggers are placed. Round numbers, recent swing highs and lows, and well-known technical levels attract both legitimate orders and predatory order flow designed to trigger those orders.

S34 implication: our SL placement must be tested against stop-hunting patterns. A fixed 40 bps SL placed mechanically at every entry will cluster at predictable price levels in volatile sessions. If those levels are hunted systematically, our SL will be triggered more often than a random-walk model predicts. MAE path analysis (maximum adverse excursion before eventual TP) should be examined to determine whether the market frequently dips beyond our SL before recovering — which would indicate stop hunting rather than genuine signal failure.

This analysis has not yet been done formally. It is a required next step before SL geometry is finalized for any live route.

### 2.4 Transaction Cost Is Strategy-Specific

Benchmark costs like "fees paid per trade" are insufficient. Real transaction cost for a short-hold directional strategy includes:
- spread cost at entry and exit
- fee structure (maker vs. taker)
- market impact of entry size on a thin book
- adverse selection: whether the fill price reflects information that will move against you
- slippage between signal trigger and actual entry

S34 implication: the current paper runner subtracts a fixed 8 bps round-trip fee. This models the fee component only. It does not model spread, market impact, or adverse selection. The execution engineering sprint demonstrated that at standard taker fees (2 × 2 bps maker rebate subtracted, net ~4 bps gross fee on a taker entry/exit), the best live-calibrated strategies are borderline. At VIP-tier fees (net ~2 bps taker), strategies become consistently viable.

The 8 bps fixed paper deduction is not wrong, but it is not the whole picture.

### 2.5 Performance Evaluation Is Vulnerable to Period Selection

A strategy that looks good because it was measured in a period where conditions happened to match its strengths is not proven. Period selection bias is one of the most common sources of false confidence in systematic trading.

S34 implication: our historical backtests concentrated in certain months where ETH BUY cascades were frequent and directional. The out-of-sample forward validation protocol exists precisely to counter this. No route is considered validated until it accumulates N ≥ 40 forward trades under live conditions with the rule parameters fixed — not adjusted after each losing stretch.

---

## Chapter 3: Algorithmic Trading Methodology (Chan)

### 3.1 Backtesting Pitfalls Are Almost Always Optimistic

The most dangerous backtesting errors, in order of how hard they are to detect:

**Lookahead bias** — using information at time T that would not have been available until T+k. Example: using the day's final trend percentage to filter entries made during that day. This was present in an early S34 regime filter and was corrected. The no-lookahead day-so-far filter replaced it.

**Data snooping / overfitting** — running many parameter combinations and selecting the best performer without accounting for the number of tests. A strategy that looks optimal at TP=60, SL=30, BE=30 after testing a 4×4×4 grid has 64 implicit comparisons. The probability that the best result is noise is not small.

**Survivorship bias** — when historical data excludes symbols or events that failed, the remaining sample looks better than reality.

**Execution mismatch** — assuming fills at prices that could not have been achieved. Mark price entries during liquidation spikes are the clearest example.

S34 implication: all four biases have been partially addressed. Lookahead is controlled. Parameter grid searches are run once and frozen; subsequent modifications require a formal amendment with a pre-stated hypothesis. Binance liquidation data covers all events, not survivors. Execution mismatch remains the most underaddressed risk.

### 3.2 Statistical Significance Before Trust

A sample of 40 trades with a 55% win rate does not prove edge. The binomial standard error at N=40 with p=0.55 is approximately 0.079. A 95% confidence interval spans from roughly 0.40 to 0.70. The true win rate could easily be 50% or lower.

Approximate forward sample requirements for a cascade-style strategy with ~50% base win rate:
- N = 40: first kill check only — is the route catastrophically wrong?
- N = 100: early read — is the sign correct?
- N = 300: directional confidence — can we say "this route has edge"?
- N = 600+: meaningful statistical discussion
- N = 1,000+: serious live-risk discussion

These thresholds are not pessimistic. They are the minimum required to distinguish signal from noise at meaningful confidence levels.

Current status (June 2026): no route has reached N=300. ETH BUY is the furthest along. Declaring edge anywhere in the portfolio before N=300 per route would be premature.

### 3.3 Mean Reversion vs. Momentum — Know Which You Are Trading

Chan distinguishes mean-reversion and momentum systems because their failure modes are opposite. A mean-reversion strategy fails when the market continues trending; a momentum strategy fails when the market snaps back.

S34 implication: the direction of our trade depends critically on what side of a cascade we enter. A large BUY-side cascade (shorts liquidated) is both:
- a momentum signal: the cascade accelerates the move that triggered it
- a mean-reversion opportunity: extreme forced flow may overshoot, creating a rebound

Early S34 BUY entries were implicitly momentum — entering after a cascade expecting continuation. SELL entries are implicitly counter-trend — entering short after a SELL-side cascade expecting a reversion or continuation of the downmove.

Both can work. Both require different TP/SL geometry and different N thresholds before trust. Mixing them in the same evaluation pool is a methodological error. The current system correctly separates BUY and SELL routes.

### 3.4 Stop Logic Is Not Automatically Beneficial

A stop loss reduces tail risk but also crystallizes noise losses. In a mean-reverting system, the optimal stop may be wider than intuition suggests — the entry may temporarily exceed the SL before recovering to TP. A mechanical fixed-bps SL applied uniformly may systematically exit trades that would have reached TP given slightly more room.

S34 implication: BE (breakeven) triggers introduce a third failure mode that is empirically visible in our data. ETH SELL trades in June 2026 showed a pattern where the entry occurred 60+ bps after a cascade had already moved, leaving only 41 bps of remaining expected move toward TP. The BE trigger fired before TP, then price reversed — producing BE exits that were effectively losses after fees. This is not a SL problem. It is a late-entry problem. The cascade had already spent most of its energy before our entry.

### 3.5 Backtest Execution Must Match Live Execution

If the backtest fills at mark-price and live execution fills at taker price after a 200ms latency, the strategy's apparent edge in backtest may not survive live conditions. Every 1 bps of unmodeled execution cost directly reduces the realized edge.

S34 implication: the execution engineering sprint (9 phases, May 2026) addressed this directly. The main findings:
- At standard taker fees: the best route (ETH BUY onset) produces +3.12 bps gross but nets to approximately -0.88 bps after fees
- At VIP taker fees (achievable at ~$5M monthly volume): the same route nets approximately +1.1 bps
- Passive maker fills were intermittently viable but introduced significant missed-fill risk (no-fill rate rising to 40%+ in thin conditions)

The execution layer is not trivially solved. It requires VIP fee status to unlock the net edge.

---

## Chapter 4: The S34 Signal Architecture

### 4.1 Data Pipeline

The collection stack runs continuously and captures three data streams:

**Liquidation stream** — Binance Futures websocket, all forced order events per symbol. Stored as individual events with timestamp, symbol, side (BUY/SELL), price, and quantity.

**Mark price stream** — 250ms snapshots of mark price per symbol. Used for P&L calculation, regime detection, and TP/SL evaluation.

**Book ticker stream** — best bid/ask with quantities at each update. Approximately 2 ticks per second per symbol. Used for spread analysis and execution modeling. Data volume: ~1.6 billion rows for ETH/SOL/BTC over two months.

**Symbols currently tracked:** ETHUSDT, SOLUSDT, BTCUSDT.

All data is stored in SQLite with read-only access from research and chart processes. The collector supervisor monitors all data processes and restarts on failure.

### 4.2 Signal Detection

A cascade event is triggered when cumulative liquidation notional within a rolling 5-minute window exceeds a symbol-specific threshold:
- ETH: 500K or 1M USDT (route-dependent)
- SOL: 100K or 200K USDT
- BTC: 1M USDT

Additional filters applied at signal time:
- Minimum gap since last signal (prevents clustering)
- Regime gate: trend ≥ 1.0%, range ≥ 2.5%, daily cumulative BUY liq ≥ 5M, daily agg count ≥ 250K (ETH BUY routes)
- Distributed filter on BTC: no single order may exceed 50% of the 5-minute bucket notional

### 4.3 Trade Parameters (Fixed per Route)

Each route has pre-registered TP, SL, and BE parameters:
- **ETH BUY 500K:** TP=60 bps, SL=40 bps, BE=30 bps
- **ETH BUY 500K DAYTREND:** TP=60 bps, SL=40 bps, BE=30 bps
- **SOL BUY 200K:** TP=60 bps, SL=40 bps, BE=30 bps
- **SOL BUY 100K:** TP=60 bps, SL=40 bps, BE=30 bps
- **BTC BUY 1M distributed:** TP=60 bps, SL=30 bps, BE=30 bps
- **ETH SELL 500K:** TP=60 bps, SL=40 bps, BE=40 bps
- **ETH SELL 1M:** TP=80 bps, SL=40 bps, BE=40 bps
- **SOL SELL 100K:** TP=60 bps, SL=30 bps, BE=40 bps
- **SOL SELL 200K:** TP=60 bps, SL=30 bps, BE=30 bps

Parameters are frozen at amendment filing. No mid-route adjustments.

### 4.4 The Prediction Layer (Calculator v2)

A KNN (K-nearest neighbors) model is trained on historical features at signal time and predicts the probability of TP vs. SL. Features include:
- cluster liquidation notional
- pre-signal price momentum (15-minute)
- day trend direction
- cluster duration
- symbol context

**Current model tags (June 2026):**
- ETH SELL: `KNN_USEFUL` — directional accuracy 0.72, meaningful lift over base rate
- ETH BUY: `REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY` — model depends heavily on recent data; regime shift in April 2026 made older samples less informative
- SOL BUY: `BASE_RATE_ONLY` — KNN provides no consistent lift over base rate
- BTC SELL: `BASE_RATE_ONLY` — 34 test events, 1 unique prediction; majority-class artifact confirmed

The calculator is used in the live chart intelligence panel for decision support. It is not wired into execution.

### 4.5 Shadow Tags and Context Enrichment

Four shadow tags annotate each signal with session and structural context:

- `CONTEXT_CO_CASCADE` — whether a co-cascade occurred on another symbol within 30 seconds
- `CONTEXT_SESSION` — Asian / Europe / US session at signal time
- `CONTEXT_GAP_LAST_SEC` — whether a significant price gap preceded the cascade
- `CONTEXT_IDIOSYNCRASY` — whether the cascade appears idiosyncratic (no co-cascade, no prior trend)

All four tags are accumulating data. No rules have been modified based on shadow tags yet — N is below the 50-per-bucket threshold required for a decision.

---

## Chapter 5: Edge Discovery — What We Found

### 5.1 The Discovery Sprint (May 2026)

An 8-stage discovery sprint tested candidate signals beyond the core liquidation cascade. The goal was to find additional alpha sources that could supplement or diversify S34.

**Stage results:**

| Signal | Result |
|---|---|
| Liquidation reversal (core S34) | Real, +1.56 bps gross — but not tradeable at standard fees alone |
| Vol state regime filter | Broken — vol_state computation had a data dependency error; discarded |
| Liquidation intensity scoring | Marginal lift over binary threshold; not enough to justify complexity |
| Cascade precursors (pre-signal) | Undirected — precursor events do not predict direction reliably |
| Book pressure lane | Real signal in certain regimes; added as a shadow feature, not a standalone route |
| Funding rate context | Directional modifier, not a standalone trigger |
| Return shock detection | Correlation with cascades too high; largely redundant |
| Cross-symbol lead-lag | BTC pre-signal has modest predictive value for ETH within 15 minutes |

**Overall finding:** the liquidation cascade is the primary signal. Supplementary signals provide context and modest filtering ability but do not independently generate tradeable edge at our N counts.

### 5.2 Pine Indicator Training (June 2026)

A composite indicator (VWAP deviation + EMA slope + delta imbalance) was trained walk-forward on 1-minute local data. The training target was predicting cascade-aligned TP outcomes.

Results:
- Gross performance: approximately +4 bps per signal
- At standard fees: dead (fees consume the gross edge)
- At VIP fees: borderline viable

The indicator confirmed that cascade direction is regime-dependent — the same signal in a trending vs. ranging session has materially different expected outcomes. Regime conditioning is necessary, not optional.

### 5.3 BTC 1M Distributed Route — Historical Contradiction

The BTC BUY 1M DISTRIBUTED route was pre-registered based on a backtest claiming WR=60%, median=+42.3 bps (N=83). A fresh historical analysis run in June 2026 using the correct pre-reg parameters (SL=30, distributed filter applied) produced:

- Historical N: 44 events (not 83)
- Win rate: 25%
- Median: -8 bps
- Cumulative: -297 bps

The discrepancy between the pre-reg backtest and the fresh analysis is unexplained. Likely causes: different minimum gap parameter, different distributed filter application, or different data window. The pre-reg claim is not reproduced.

**Current status:** forward validation sample N=3 (since 2026-06-25). First kill check at N=40, expected late July 2026. Route is paper only. No live execution.

### 5.4 ETH SELL Timing Problem

An analysis of June 2026 ETH SELL trades revealed a structural late-entry problem. Cascade SELL events were triggering the signal after the cascade had already moved price 60+ bps downward. At entry, only ~41 bps of expected move remained toward TP=60/80 bps.

The BE trigger (40 bps) fires before TP when the entry is this late. Price reaches BE, triggers the stop move-up, then recovers — producing BE exits that are effectively flat or slightly negative after fees.

**Root cause confirmed:** late entry into a partially-spent cascade. Three potential fixes identified:
1. Pre-entry move filter: discard signals where price has already moved >X bps before entry
2. Smaller TP: calibrate TP to the remaining expected move, not the full cascade move
3. Entry delay refinement: tighten the entry window to catch earlier in the cascade

No fix will be implemented until N ≥ 30 for SELL routes and a formal amendment is filed. Current SELL N: approximately 11–15 depending on route.

---

## Chapter 6: Execution Engineering

### 6.1 The Cost Problem

The core finding of the execution engineering sprint (9 phases, May 2026):

**At standard fees (~4 bps round-trip taker):** the best route (ETH BUY onset, defined as entries within the first 30 seconds of a cascade) generates +3.12 bps gross in 4/4 development splits. Net after fees: approximately -0.88 bps. Not viable.

**At VIP fees (~2 bps round-trip taker):** the same route nets approximately +1.1 bps. Viable, but requires $5M+ monthly trading volume to qualify.

This means the edge is real but thin. Standard retail fee access cannot capture it. VIP access — or a strategy modification that reduces taker exposure (more passive fills, less frequent trading) — is required.

### 6.2 Maker vs. Taker Fill Analysis

Four fill models were tested across all routes:

| Model | Description | Result |
|---|---|---|
| `mark_mid` | Fill at mark price, optimistic baseline | Best apparent performance, not executable |
| `taker` | Cross spread + taker fee | Viable only at VIP fees |
| `passive` | Limit order, fill only if price revisits | Viable at standard fees but no-fill rate 35–45% |
| `latency` | Entry after N seconds of confirmed cascade | Worse than onset for BUY; potentially better for SELL |

The no-fill rate on passive fills is the critical constraint. When no-fill exceeds 40%, the strategy's effective sample drops dramatically and adverse selection on the fills that do execute increases.

### 6.3 Onset vs. Distributed Entries

An "onset" entry (entering at the first qualified bar of a cascade) outperforms a "distributed" entry (averaging across the cascade window) for BUY routes. The early part of the cascade has the most residual expected move. Late entries capture less.

For SELL routes the picture is reversed — SELL cascades tend to have more complex structure, and onset entries during the initial SELL wave may be premature if the cascade has multiple legs.

### 6.4 The VIP Threshold Problem

The execution engineering conclusion sets a hard practical constraint: to capture S34 edge at current signal frequency and TP geometry, the system requires VIP-tier Binance fees. This requires either:

1. Scaling capital to generate $5M+ monthly volume organically at our current trade frequency
2. Increasing trade frequency substantially (increases adverse selection risk)
3. Improving gross edge enough that standard fees become viable (requires route improvement)
4. Finding a fee arrangement outside of standard retail access

This constraint is not hidden in the data. It is the most important practical finding of the entire execution sprint. No deployment decision should ignore it.

---

## Chapter 7: The Pre-Registration Framework

### 7.1 Why Pre-Registration Exists

Without pre-registration, the natural tendency of any systematic trading researcher is to:
- adjust parameters after seeing results
- add filters after observing losing trades
- declare success when a particular period looks good
- ignore evidence that contradicts the preferred hypothesis

Pre-registration is borrowed from clinical research. Before the trade data arrives, we state:
- exactly which rule will fire
- exactly what parameters it uses
- exactly what N constitutes a kill check
- what would cause us to conclude the route has failed

Then we do not change these until the stated N is reached.

### 7.2 Amendment Protocol

Any change to a live or paper route requires a formal amendment:
- State the hypothesis for the change
- State what evidence supports it
- State the new parameters
- File the amendment with a hash of the runner code at filing time
- Restart the clock — forward N resets to zero

**Amendment history (Eclipse S34):**
- Amendments 1–3: ETH BUY route geometry refinements, regime filter corrections
- Amendment 4: Added SOL BUY 200K and BTC BUY 1M DISTRIBUTED
- Amendment 5: Internal calibration update
- Amendment 6: Added ETH SELL 500K, ETH SELL 1M, SOL SELL 200K
- Amendment 7: Added SOL BUY 100K at priority=15
- Amendment 8 (filed 2026-06-26): Added SOL SELL 100K; SELL route clocks reset

**Current runner SHA256 (Amendment 8):** `037b2fe0774a6f2f88534163aa99d6e882dce617a879d76120d88250358a1197`

### 7.3 Kill Check Criteria

**MAIN routes (N=100 target, calibration at N=40):**
- K1 at N=40: if cumulative net bps < -150 AND win rate < 35%, route is terminated
- K2 at N=100: full calibration review; holdout evaluation begins

**SELL_EXP routes (N=30 target):**
- Kill criteria: median ≤ 0 OR top-3-outliers-removed median ≤ 0 OR fewer than 8 trading days elapsed
- Promotion criteria: median > 0 AND top-3-removed > 0 AND ≥ 8 days

### 7.4 What Pre-Registration Does Not Prevent

Pre-registration controls the timing and justification of parameter changes. It does not prevent:
- genuinely bad routes from running until N=40/100
- regime shifts that invalidate the hypothesis before N is reached
- data collection failures that corrupt the forward sample

The framework requires discipline, not perfection. The value is in forcing the question: "would we have made this decision before seeing this specific losing streak?" If no, the amendment is not justified.

---

## Chapter 8: Route Status — June 2026

### 8.1 ETH BUY 500K DAYTREND (MAIN, Clock reset Amendment 3)

Pre-reg parameters: TP=60, SL=40, BE=30  
Forward N: accumulating since 2026-06-25 clock reset  
Model tag: REGIME_SHIFT_RECENCY_HELPFUL_PRELIMINARY  
Early pattern: regime-on sessions in European trading hours show strongest performance  
Kill check: N=40

**Status:** Active accumulation. No decision possible until N=40.

### 8.2 SOL BUY 200K (MAIN, Amendment 4)

Pre-reg parameters: TP=60, SL=40, BE=30  
Forward N: most advanced of all MAIN routes  
Model tag: BASE_RATE_ONLY  
Early pattern: signal frequency moderate; geometry CLEAN flag associated with better outcomes  
Kill check: N=40 reached or approaching

**Status:** Most mature BUY route. Closest to early read threshold.

### 8.3 SOL BUY 100K (MAIN, Amendment 7, priority=15)

Pre-reg parameters: TP=60, SL=40, BE=30  
Forward N: lower (later start)  
Model tag: BASE_RATE_ONLY  
Note: fires after SOL 200K on same cascade; complements rather than competes

**Status:** Active accumulation. Early stage.

### 8.4 BTC BUY 1M DISTRIBUTED (MAIN, Amendment 4)

Pre-reg parameters: TP=60, SL=30, BE=30, max_single_liq_share_pct=50%  
Forward N: 3 (as of 2026-06-27)  
Historical backtest contradiction: pre-reg claims WR=60%, N=83; fresh analysis shows WR=25%, N=44  
Kill check: N=40, expected late July 2026  
Paper only — not in live executor

**Status:** Under close watch. Historical data does not support pre-reg claims. Forward accumulation is the only path to a verdict.

### 8.5 ETH SELL 500K (SELL_EXP, Amendment 6, priority=10)

Pre-reg parameters: TP=60, SL=40, BE=40  
Clock start: 2026-06-26  
Research finding: regime-agnostic, robust across bull/bear sessions  
Model tag: KNN_USEFUL (directional accuracy 0.72)  
Kill criteria: median ≤ 0 at N=30

**Status:** Most promising SELL route by research. Clock recently started. N accumulating.

### 8.6 ETH SELL 1M (SELL_EXP, Amendment 6, priority=8)

Pre-reg parameters: TP=80, SL=40, BE=40  
Clock start: 2026-06-26  
Research finding: stronger on bull days (WR=86%), weaker on strong-bear days; high-range days best  
Late-entry timing problem observed in June trades  
Kill criteria: median ≤ 0 at N=30

**Status:** Active. Late-entry problem identified — root cause understood, fix cannot be implemented until N≥30 and amendment filed.

### 8.7 SOL SELL 200K (SELL_EXP, Amendment 6, priority=10)

Pre-reg parameters: TP=60, SL=30, BE=30  
Clock start: 2026-06-26  
Research finding: bear-day dependent; bull-day performance ≈ -10 bps; bear filter tested but gain (+0.8 bps) did not justify signal loss  
Kill criteria: median ≤ 0 at N=30

**Status:** Active. Bear-day dependency is a risk. No filter added; watching forward data.

### 8.8 SOL SELL 100K (SELL_EXP, Amendment 8, priority=15)

Pre-reg parameters: TP=60, SL=30, BE=40  
Clock start: 2026-06-26  
Research finding: range-dependent; low-vol sessions marginal (+13 bps), high-vol strong (+51 bps)  
Kill criteria: median ≤ 0 at N=30

**Status:** Newest route. Lowest priority on same cascade. N accumulation just begun.

---

## Chapter 9: The Prediction Layer in Practice

### 9.1 What the Calculator Is

The S34 outcome calculator is a KNN model that takes the feature vector at signal time and returns a predicted outcome probability (TP vs. SL/BE). It is used in the live chart intelligence panel to display a confidence signal alongside each trade candidate.

It is a decision-support tool. It does not execute. It does not block trades. It provides an additional signal that a human (or eventually an automated gate) can incorporate.

### 9.2 When KNN Is Useful and When It Is Not

The ETH SELL calculator (tag: KNN_USEFUL) demonstrates that KNN can provide meaningful lift when:
- the feature space is sufficiently different between winning and losing trades
- the training set has enough diversity (not dominated by one regime)
- the features are computable without lookahead at signal time

The BTC SELL calculator failure (tag: BASE_RATE_ONLY) demonstrates that KNN fails when:
- there are too few unique signals for the model to distinguish patterns (34 test events, 1 unique prediction)
- the feature space collapses to majority-class behavior

The lesson: KNN is not a universal signal enhancer. It requires a sufficient and diverse feature space. With small N and regime homogeneity, it degenerates to base-rate majority prediction.

### 9.3 Population Scan vs. Signal Target

One subtle error caught during calculator v2 development: the `log_cluster_notional` feature (derived from the minimum notional filter used to define the population of signals) was being passed to the model as a candidate signal feature. This creates circular logic — a feature that partially defines whether an event is a signal cannot also predict signal outcome.

The fix: features derived from the signal definition filter are excluded from the KNN input. Only features representing external market context (price momentum, session, co-cascade) are valid inputs.

This error is easy to make and hard to detect without explicit feature provenance tracking.

---

## Chapter 10: What We Know Now

### 10.1 Confirmed

- The liquidation cascade signal is real. Gross edge is positive and directionally consistent across multiple N counts and temporal splits.
- Execution cost is the critical constraint. At standard fees, the edge is insufficient. At VIP fees, it is viable.
- Regime conditioning matters. The same signal in a trending vs. ranging, bull vs. bear session has meaningfully different outcomes. Unconditional analysis masks this.
- Pre-registration works. Several amendment candidates that felt justified in the moment were correctly deferred because N was insufficient. The framework prevented several likely false positives.
- SELL alpha exists. ETH SELL is regime-agnostic and the KNN model provides useful lift. The SELL routes were the right expansion direction.

### 10.2 Open Questions

**Q1: Can the late-entry SELL problem be fixed without losing signal frequency?**  
Pre-entry move filter or TP geometry adjustment. Testable at N=30. Amendment planned.

**Q2: Does the BTC 1M distributed route have any historical basis?**  
Historical analysis contradicts pre-reg claims. Forward N=3 cannot resolve this. Kill check at N=40 (late July 2026) is the earliest decision point.

**Q3: What are the shadow tag conditional outcomes?**  
CONTEXT_IDIOSYNCRASY=36% WR (SELL worst), CONTEXT_SESSION (Europe BUY best) — early data, N below 50-per-bucket threshold. No action possible yet.

**Q4: Is there SOL SELL bear-day filter value?**  
Research showed +0.8 bps gain from filter, not worth the signal loss. Revisit at N=30 forward.

**Q5: What happens to execution at scale?**  
Current paper trades are not sized. Actual position sizing and market impact at any meaningful capital level have not been modeled. This must be addressed before any live scale-up.

### 10.3 What The Books Would Say About Our Current Position

Harris would say: you have identified a forced-flow signal in a domain (crypto perpetual futures) where liquidation events are unusually large, frequent, and observable. The signal is plausible. Your primary remaining risk is execution — adverse selection during the cascade window is precisely when execution quality deteriorates most. Do not declare edge until you have live taker fills, not mark-price paper fills.

Chan would say: your sample sizes are too small for confidence. Your pre-registration framework is correct. Your most important discipline right now is to not loosen gates prematurely because of a few good trades, and to not abandon routes prematurely because of a bad stretch. Systematic trading requires accepting statistical uncertainty for longer than feels comfortable.

Both would agree: the framework is sound. The evidence is preliminary. The next 3 months of forward accumulation matter more than any backtest refinement.

### 10.4 Next Decision Gates

| Date | Gate | Route |
|---|---|---|
| Late July 2026 | N=40 kill check | BTC BUY 1M DISTRIBUTED |
| When N≥30 (SELL) | Kill/promote decision | All 4 SELL_EXP routes |
| When N≥30 (SELL) | Amendment: late-entry fix | ETH SELL 1M |
| When N≥50/bucket | Shadow tag conditioning | CONTEXT_IDIOSYNCRASY, CONTEXT_SESSION |
| When N≥300 (ETH BUY) | Directional confidence claim | ETH BUY 500K DAYTREND |
| TBD | Execution realism upgrade | All routes (MAE/MFE path journal) |
| TBD | VIP fee access | Prerequisite for live deployment |

### 10.5 The Correct Posture

The edge is real enough to be worth continuing. The data is not sufficient for confidence. The execution problem is solved in principle but not in practice. The framework is working.

Keep collecting. Keep the gates fixed. Do not adjust parameters between kill checks. The value of the next N=50 forward trades is greater than any further backtest refinement.

---

## Appendix A: Glossary

**Cascade** — a cluster of forced liquidations within a short time window that exceeds a notional threshold  
**Amendment** — a formal parameter change to a pre-registered route, with restated hypothesis and N clock reset  
**Kill check** — a pre-stated decision point where route continuation is evaluated against pre-defined criteria  
**BE (breakeven)** — a stop-move mechanism that shifts SL to entry price after the trade moves N bps in favor  
**MAE** — maximum adverse excursion; the worst drawdown experienced during a trade before exit  
**MFE** — maximum favorable excursion; the best profit point experienced during a trade before exit  
**WR** — win rate; fraction of closed trades that exited at TP  
**bps** — basis points; 1 bps = 0.01%  
**VIP** — Binance VIP fee tier requiring $5M+ monthly volume; reduces taker fees to ~1 bps per side  
**KNN** — K-nearest neighbors; the prediction model used in the outcome calculator  
**Shadow tag** — a metadata annotation on each trade recording session, co-cascade, and structural context  
**Forward N** — number of trades accumulated since the current amendment was filed  
**Distributed filter** — BTC route constraint: no single liquidation order may exceed 50% of the 5-minute bucket notional  

## Appendix B: Route Quick Reference (Amendment 8)

| Route | Type | TP | SL | BE | N target | Kill at |
|---|---|---|---|---|---|---|
| ETH BUY 500K DAYTREND | MAIN | 60 | 40 | 30 | 100 | N=40 |
| SOL BUY 200K | MAIN | 60 | 40 | 30 | 100 | N=40 |
| SOL BUY 100K | MAIN | 60 | 40 | 30 | 100 | N=40 |
| BTC BUY 1M DISTRIBUTED | MAIN | 60 | 30 | 30 | 100 | N=40 |
| ETH SELL 500K | SELL_EXP | 60 | 40 | 40 | 30 | N=30 |
| ETH SELL 1M | SELL_EXP | 80 | 40 | 40 | 30 | N=30 |
| SOL SELL 200K | SELL_EXP | 60 | 30 | 30 | 30 | N=30 |
| SOL SELL 100K | SELL_EXP | 60 | 30 | 40 | 30 | N=30 |

All parameters in basis points. All clocks active from Amendment 8 filing (2026-06-26).

---

*This document is updated as forward data accumulates and new decisions are made. It is not a marketing document. It is not a performance record. It is a working hypothesis under test.*
