# ECLIPSE RESEARCH BIBLE
## E-DER / Eclipse Scalper — Research Charter, Knowledge Base, Decision Log, and AI Operating Protocol

**Version:** 0.1  
**Last updated:** 2026-08-19  
**Status:** Living document  
**Purpose:** Give any future ChatGPT/Codex/research assistant enough context to continue the project intelligently without requiring the user to restate prior work.

---

# 0. BOOTSTRAP INSTRUCTION FOR ANY FUTURE AI

Read this document before proposing research, code, experiments, prompts, or conclusions.

Treat it as the canonical project context unless a later explicitly dated decision supersedes it.

The assistant is expected to be **proactive**. Do not wait for the user to invent every next idea. When the next scientifically justified step is clear:

1. State it.
2. Explain why it is the next step.
3. Identify what evidence would support or falsify it.
4. Identify the data requirements.
5. Identify the main failure modes.
6. Suggest the smallest valid experiment.
7. Preserve anti-data-mining discipline.
8. Do not ask the user to repeat information already contained here.

Never convert uncertainty into a story merely because the story is intuitive.

---

# 1. PROJECT IDENTITY

**Project:** Eclipse Scalper  
**Research branch:** Echo Delayed Exhaustion Rebound (**E-DER**)

## Market / environment

- Binance Futures perpetuals.
- BTCUSDT / ETHUSDT were central historically; later research universe expanded to roughly 56 symbols.
- Core frequency: 1 minute.
- Multiscale work often uses 1 / 3 / 5 / 10 minute views.
- Fixed research boundary: +240 minutes.
- Entry convention where frozen: next exact minute OPEN.
- Historical transaction-cost assumption often used: 10 bps.
- Current state is research / validation, not live deployment.
- Windows, Python, VS Code.
- Historical repository path: `D:\eclipse_scalper`.
- Historical SQLite DB: `D:\eclipse_scalper\data\microstructure_02.db`.

Do not assume paths or schemas still exist without auditing them.

---

# 2. PROJECT PHILOSOPHY

The project is **not** trying to find a profitable pattern at any cost.

Primary objective:

> Determine whether E-DER represents a reproducible, interpretable, out-of-sample market state, and if so, what can and cannot be identified about the mechanism.

Priorities:

- falsification over confirmation,
- chronological OOS over random CV,
- measurement validity before feature engineering,
- identification boundaries before causal narratives,
- minimum adequate models before complex models,
- fixed rules before tuning,
- discovery separated from confirmation,
- explicit failure reporting,
- resistance to data mining.

A negative result is acceptable. A positive result found through post-hoc threshold search is not.

---

# 3. NON-NEGOTIABLE RESEARCH RULES

Do **not**:

- optimize E-DER thresholds using +240m returns,
- choose features because they improve E-DER returns,
- use random train/test splits on dependent market time series,
- use in-sample residuals as “surprise,”
- treat repeated snapshots from the same trade as independent observations,
- multiply 25 events by multiple snapshots and call that independent N,
- optimize nonlinear impact exponents on the historical 25 E-DER events,
- treat historical post-hoc mechanism findings as confirmatory,
- silently redefine event, entry, exit, or holding period,
- create a composite “absorption score” simply because several components exist,
- infer a unique mechanism when mechanisms are observationally equivalent,
- assume missing fields existed,
- assume current Binance semantics applied historically,
- assume two Binance feeds observe the same liquidity universe,
- infer exact cancellations/additions when the feed cannot identify them,
- treat `forceOrder` as a complete liquidation tape,
- treat `aggTrade` as the underlying trader/order process unless historical semantics are proven,
- use E-DER outcome to choose model class, lag length, calibration window, feature set, or measurement variant.

If a hypothesis fails, report the failure plainly. Do not search the same historical sample for a replacement profitable rule.

## Statistical unit

The primary independent unit is the **trade/event**, not each within-trade snapshot. Repeated horizons and snapshots are dependent.

## Historical vs confirmatory

The mechanism hypothesis developed after observing Channel A behavior. Therefore:

- 25 historical E-DER events = post-hoc historical mechanism evidence,
- fresh forward E-DER events = true confirmatory sample.

---

# 4. FROZEN E-DER CANDIDATE

Current frozen logic, approximately:

- reset present,
- parent SELL liquidation,
- echo normalized liquidation flow stronger than parent: `q_echo >= q_parent`,
- multiscale event direction = SHORT,
- pre-parent stress anchors >= 2,
- LONG hypothesis after delay,
- delayed entry = +31m OPEN,
- fixed exit = +240m OPEN,
- price acceptance is diagnostic only.

## Historical sample

- 25 independent historical E-DER events.
- Legacy chronological OOS subset: 17 trades.

Approximate historical values:

### +31m delayed LONG
- Avg +305.10 bps
- Median +124.36 bps
- WR 64%

### Immediate LONG
- Avg +183.00 bps
- Median −34.39 bps
- WR 48%

### BTC-neutral
- Avg +281.95 bps
- WR 60%

### Legacy chronological OOS, 17 trades
Fixed +240m:
- Avg +377.93 bps
- Median +124.36 bps
- WR 70.6%
- 0 early exits

Earlier generic early-exit systems damaged performance relative to fixed hold:

**Event Score**
- Avg about +10.88 bps
- WR about 52.9%
- 7 early exits
- paired damage about −891.41 bps

**Forecast**
- Avg about +6.94 bps
- WR about 41.2%
- 14 early exits
- paired damage about −450.49 bps

Historical lesson:

> Generic deterioration/forecast exits repeatedly exited before the fixed-boundary rebound.

---

# 5. CHANNEL A — WHAT FAILED

Channel A expected:

> The delayed E-DER LONG rebounds while forced SELL-liquidation flow does not renew.

Definitions:

**CONFIRMING**
- price above entry,
- cumulative post-entry SELL-liquidation/min <= frozen 30m pre-entry rate.

**CONTRADICTING**
- price below entry,
- post-entry SELL-liquidation/min > pre-entry rate.

**NEUTRAL**
- response legs disagree or price unchanged.

Outcome:
- next exact OPEN after observation to unchanged fixed-boundary OPEN.

Expected ordering `CONFIRMING > NEUTRAL > CONTRADICTING` failed at all six snapshots. Holm-adjusted p <= 0.05 occurred 0/6 times.

Examples:

- 5m: Confirming +113.73, Neutral +66.80, Contradicting +1217.74 bps
- 15m: +22.49, +212.44, +1011.07
- 60m: +77.91, +1.26, +238.93
- 120m: +61.74, +76.47, +397.13

Correct interpretation:

Channel A **failed as a deterioration model**.

It does **not** prove:

- more SELL liquidation is bullish,
- contradiction means stronger E-DER,
- renewed liquidation causes rebound,
- absorption is proven,
- dynamic exits should reverse the old rule.

What it established:

1. `renewed SELL liquidation = thesis broken` was too crude.
2. adverse-flow cells behaved unexpectedly.
3. this justified mechanism research.
4. the next question became whether adverse flow becomes progressively less effective at pushing price lower.

---

# 6. S35 DIRECTION

A Codex prompt was already prepared for **S35 E-DER Post-Entry Flow–Impact Compression Diagnostic V1**.

Restrictions:

- research-only,
- no dynamic-exit design,
- no exit simulation,
- no execution-file changes,
- same 25 historical events,
- same entry,
- same fixed boundary,
- no threshold optimization,
- statistical unit = trade/event,
- state uses only information available at observation,
- subsequent return is outcome only.

Primary question:

> Does increasing post-entry forced SELL liquidation become progressively less effective at pushing price lower?

Requested measures included:

1. normalized SELL-liquidation pressure,
2. immediate downside price response,
3. SELL-flow price effectiveness,
4. change in effectiveness,
5. if data permits: bid depth, imbalance, spread, aggressive trade imbalance, liquidity replenishment.

Robustness requested:
- counts,
- mean,
- median,
- trimmed mean,
- quartiles,
- WR,
- leave-one-trade-out,
- leave-one-symbol-out,
- tail dependence,
- BTC-neutral,
- multiple-testing correction.

Critical instruction:

> Do not optimize anything. If the hypothesis fails, report the failure plainly rather than searching for an alternative profitable rule.

---

# 7. CURRENT CONCEPTUAL UPGRADE

The project evolved from:

> More SELL liquidation after entry means deterioration.

To:

> What matters may be the effectiveness of adverse flow, not merely its amount.

Then further to:

> Impact compression is itself not a mechanism; multiple mechanisms can produce it.

Current central research objects:

1. **Flow Surprise**
2. **Impact Surprise**
3. **Response Path**
4. **Measurement Sensitivity**
5. **Competing-Mechanism Falsification**
6. **Mechanism Identification**

---

# 8. FLOW SURPRISE

Preferred V1 primitive is fixed-clock, dimensionless aggressive-flow imbalance rather than event-count statistics.

Candidate:

\[
X_t = \frac{BuyNotional_t - SellNotional_t}{BuyNotional_t + SellNotional_t}
\]

Then:

\[
\widehat X_t = E[X_t\mid\mathcal F_{t-1}]
\]

and:

\[
U^{flow}_t = X_t - \widehat X_t
\]

Interpretation:

- raw SELL high but surprise small → persistent/predictable selling,
- SELL surprise large → unusual public adverse flow conditional on observable history.

Do not interpret Flow Surprise as:

- latent trader intent,
- identified liquidation metaorder,
- informed trading,
- causal forced-flow innovation.

It is an innovation in the **observed public flow process**.

Why fixed-window notional is preferred:

`aggTrade`/exchange aggregation may distort:

- event count,
- run length,
- inter-arrival duration,
- event-lag ACF,
- burstiness,
- Hawkes branching estimates.

Fixed-window signed notional should be more stable, but this must be audited empirically.

Trade/event-time sign prediction remains a secondary challenger only if raw trade semantics and integrity are proven.

---

# 9. IMPACT SURPRISE

Core concept:

\[
R_{t,h} = \Delta m^{actual}_{t,h} - \widehat E[\Delta m_{t,h}\mid flow, liquidity, history, activity, regime]
\]

Mechanism work should prefer **midquote**, not trade/Open price.

For adverse SELL state:

- `R < 0`: more adverse response than expected,
- `R ≈ 0`: ordinary response,
- `R > 0`: less adverse response than expected.

Do **not** automatically call positive residual:

- absorption,
- resilience alpha,
- seller exhaustion,
- market-maker support.

Safer description:

> Conditional response anomaly / unexpected-resilience evidence.

---

# 10. RESPONSE PATH

Do not call every horizon “market impact.”

Prefer:

- ~5m: immediate / short-horizon response,
- ~15–30m: digestion / transition,
- ~60–120m: subsequent response path,
- +240m: frozen trading outcome.

Clock time stays frozen for E-DER alignment.

Activity/event time is a descriptor:

- trade count,
- market volume,
- book-update count,
- liquidation snapshot count,
- inter-arrival structure,
- burstiness.

---

# 11. MINIMUM NULL-MODEL ARCHITECTURE

Goal: **minimum adequate null**, not the most complex or highest-R² model.

Too weak → ordinary behavior becomes fake surprise.  
Too flexible → genuine anomaly gets absorbed.

## Impact benchmark ladder

**I0 — OFI/depth linear baseline**

Conceptually:

\[
\Delta m = \alpha + \beta OFI + \gamma LiquidityState + \varepsilon
\]

**I1 — MLOFI Ridge**
- only if historical data validly supports multi-level flow,
- only if it materially fixes generic OOS conditional-calibration failures.

**I2 — history-augmented impact**
- only if I0/I1 retain systematic OOS bias related to flow history.

Do not escalate complexity for tiny global RMSE gains.

## Flow benchmark ladder

**F0**
- simple finite-lag / regularized minute-level flow predictor.

**F1**
- event/trade-time predictor only if raw-event semantics support it.

**F2**
- DAR/MTD/regime-aware model only if simpler models fail generic chronological OOS calibration.

No E-DER outcome may influence model selection.

---

# 12. OOS CALIBRATION RULES

All residuals used as surprise must be **genuinely chronological OOS**.

- no random CV,
- no observation predicts itself,
- use walk-forward,
- account for overlapping labels/horizons,
- residual calibration uses strictly prior OOS errors.

Possible descriptive surprise percentile:

\[
U_{t,h} = F^{prior,OOS}_h(R_{t,h})
\]

This percentile is **not a p-value**.

Global RMSE is insufficient. Check residual calibration conditioned on:

- adverse/favorable flow,
- liquidity state,
- activity,
- relative tick / microstructure regime,
- stress state,
- time / feed semantic regime.

A model that is globally good but systematically biased in adverse-flow tails is not a valid E-DER null.

---

# 13. BINANCE DATA SEMANTICS — CRITICAL LESSONS

## forceOrder

Treat public forceOrder as an **observed forced-liquidation pressure proxy**, not a complete liquidation tape.

Audit historical payload fields such as:
- `q`,
- `z`,
- `l`,
- `p`,
- `ap`.

Prove how historical liquidation notional was computed.

Do not claim:
- true liquidation volume,
- complete forced-flow count,
- exact cascade intensity.

## aggTrade

Do not assume:
- one aggTrade = one trader,
- one aggTrade = one taker order,
- event count = underlying order count.

Historical aggregation semantics must be audited by exact period/version.

## RPI / observable liquidity universe

Current USD-M documentation indicates ordinary book streams and richer RPI-related visibility can represent different observable liquidity sets. Trade fields can also include flow that does not align perfectly with ordinary-book visibility.

Key principle:

> Two Binance feeds do not necessarily observe the same liquidity universe.

Never assume compatibility merely because both came from Binance.

## Historical schemas

Never assume a field present today exists in:
- old local DB,
- historical archives,
- old aggTrade records,
- old raw trades.

Audit explicitly.

---

# 14. DATA-RESOLUTION CLASSES

## Level A — high-quality incremental/event L2

Potentially supports:
- reconstructed displayed book,
- OFI/MLOFI,
- update-sequence validation,
- detailed displayed-book transitions.

Still do not infer exact cancellation vs execution cause unless the data identifies it.

## Level B — regular multi-level snapshots

Supports:
- spread,
- depth,
- imbalance,
- shape,
- sparsity,
- displayed-depth transitions.

Does **not** support exact:
- add rate,
- cancel rate,
- replenishment rate,
- event causality.

## Level C — top-of-book/bookTicker

Supports:
- best bid/ask,
- spread,
- top-level quantity,
- best-level imbalance.

Mechanism identification is weak.

---

# 15. TICK / MICROSTRUCTURE REGIME

Candidate continuous descriptor:

\[
RelativeTick_{i,t} = 10^4\frac{TickSize_{i,t}}{Mid_{i,t}}
\]

Also audit:
- spread in ticks,
- one-tick-spread fraction,
- book sparsity,
- multi-level shape.

Do not import equity thresholds directly into crypto.

Do not backfill current tick size into historical periods without evidence.

Cross-symbol common structure should be **tested, not assumed**.

Same-symbol expected-impact models are preferred where feasible. Pooled models require generic OOS evidence and appropriate controls.

---

# 16. REACTION TURNOVER

Neutral latent concept:

> **Adverse-Flow Response Turnover**

Question:

> While adverse flow continues, does the market’s active/passive response change in a way that reduces the marginal downside effectiveness of that flow?

Potential vector:

\[
\mathbf T_t = (ActiveSellPressure, PassiveBidResponse, OppositeActiveFlow, LOBState, ImpactResidual)
\]

Do not collapse immediately into one score.

Different possible mechanisms:

- SELL flow weakens,
- SELL continues but displayed bid state improves,
- opposite active BUY flow appears,
- history-dependent impact naturally falls,
- hidden liquidity matters,
- common-market reversal matters,
- measurement artefact creates apparent resilience.

---

# 17. MECHANISM MAP

**A. Cascade continuation** — adverse flow persists, impact worsens, liquidity degrades.  
**B. Unexpected resilience** — actual downside response is smaller than state-conditioned expectation.  
**C. Predictable/asymmetric liquidity** — persistent same-sign flow becomes lower-impact.  
**D. Mechanical impact decay** — prior impact naturally relaxes.  
**E. Active displayed replenishment** — displayed passive-side capacity stabilizes/improves.  
**F. Hidden/latent liquidity** — visible depth understates executable capacity.  
**G. Common-market reversal** — BTC/systemic move explains local resistance.  
**H. Data/feed artefact** — censoring, timing, aggregation, missing L2, wrong semantics.  
**I. Impact-function regime shift** — response becomes concave/pinned.  
**J. Liquidity-tail/informativeness regime** — extreme flow behaves differently.  
**K. Dynamic liquidity price** — impact coefficient changes through time.  
**L. Same-sign history-conditioned suppression** — persistent SELL has falling marginal impact.  
**M. Counter-flow/liquidity-molasses** — adverse flow offset by passive/quote dynamics.  
**N. Tick/queue-regime effect** — same flow has different meaning under different microstructure regimes.  
**O. Expected-flow adaptation** — SELL high but already expected.  
**P. Surprise-flow resilience** — unexpected SELL high but downside response unusually small.  
**Q. Book–trade observability mismatch** — flow/book feeds observe different liquidity universes.  
**R. Feed-transformation artefact** — exchange aggregation/timing/visibility creates apparent anomaly.

These are not mutually exclusive.

---

# 18. FALSIFICATION MATRIX

| Explanation | What would seriously weaken it? |
|---|---|
| Predictable SELL / asymmetric impact | Unexpected SELL remains high but positive Impact Surprise survives |
| Displayed passive resilience | Spread/depth/book state keeps deteriorating while anomaly persists |
| Hidden/latent liquidity | Usually not fully falsifiable from legacy public data; richer-view sensitivity can weaken it |
| Normal transient impact decay | Anomaly survives a history/decay-aware generic OOS null |
| Common-market reversal | Anomaly survives market-neutral/idiosyncratic adjustment |
| Feed artefact | Valid alternate measurement pipelines retain the same anomaly |
| Simple cascade continuation | Adverse pressure continues but expected deterioration / price effectiveness no longer appears |
| Unique “absorption” story | If competing mechanisms remain observationally equivalent, unique absorption is NOT IDENTIFIABLE |

---

# 19. CLAIM TAXONOMY

## DIRECTLY OBSERVABLE
Examples:
- stored signed flow,
- stored midquote,
- displayed depth,
- spread,
- observed liquidation proxy.

## STATISTICALLY ESTIMABLE
Examples:
- expected flow,
- Flow Surprise,
- expected price response,
- Impact Surprise,
- OOS residual percentile.

These are model-dependent.

## MECHANISM-COMPATIBLE
Example:

> Positive Impact Surprise co-occurs with improving displayed bid depth.

Compatible with displayed passive resilience; not causal proof.

## SUPPORTED
Use only when required data exists, generic OOS evidence supports the pattern, admissible measurement sensitivity does not destroy it, and major identifiable alternatives have been tested.

## NOT SUPPORTED
Required data exists but the specified pattern is absent or reversed.

## NOT IDENTIFIABLE
Use when necessary data is missing or competing mechanisms cannot be separated by the stored observations.

Critical rule:

> NOT IDENTIFIABLE is not the same as NOT SUPPORTED.

---

# 20. OBSERVATIONAL EQUIVALENCE

A state such as:

\[
SELL\ high,\quad displayed\ bid\ weak,\quad actual\ downside\ small
\]

may be compatible with:
- hidden liquidity,
- predictable flow,
- transient decay,
- common-market reversal,
- feed mismatch,
- genuine passive resilience.

Research goal:

> Determine which mechanism classes can be rejected, which remain compatible, and which are not identifiable.

Not:

> Choose the most intuitive story.

---

# 21. MEASUREMENT UNCERTAINTY

Separate:

**Market uncertainty** — market stochasticity.  
**Model uncertainty** — expected-response model error.  
**Measurement uncertainty** — flow/book/timestamps incomplete, aggregated, delayed, censored, or semantically changed.

Do not hide all three inside one residual.

Do not create arbitrary `Residual × ReliabilityScore` adjustments.

Where richer data permits, compare only scientifically admissible measurement variants:

- raw trades vs stored/aggregate flow,
- ordinary book vs richer observable book,
- exchange-event time vs receive time,
- total flow vs semantically aligned flow,
- legacy-compatible pipeline vs richer-observability pipeline.

Possible descriptive sensitivity envelope:

\[
[\min_k R^{(k)},\max_k R^{(k)}]
\]

Do not search dozens of variants for the most favorable residual.

---

# 22. MODERN VALIDATION PERIOD

Treat modern richer data as a **validation laboratory**, not a truth oracle.

Construct:

**Legacy-emulation view** — deliberately restricted to what the old pipeline could observe.  
**Richer-observability view** — newer/richer fields and streams where available.

Compare generic market measurements and OOS residuals.

Do not automatically transport a modern correction factor backward into historical E-DER. Transportability must be demonstrated.

Validation adequacy depends on state-support overlap, not arbitrary calendar duration.

Relevant axes:
- symbol/liquidity tier,
- adverse-flow quantile,
- volatility/activity,
- displayed depth/spread,
- relative tick,
- feed semantic regime.

---

# 23. SYSTEMIC EVENT DEPENDENCE

25 E-DER trades may not equal 25 independent economic shocks.

Future robustness should consider:
- leave-one-event-out,
- leave-one-symbol-out,
- leave-one-systemic-episode-out.

Potential systemic episodes should be defined outcome-free, for example by overlap in required analysis-support intervals, not tuned minute thresholds.

---

# 24. MULTIPLE TESTING

5 / 15 / 30 / 60 / 120 minute horizons overlap and are dependent.

Avoid large test families.

Potential future preregistered hierarchy:

**Family A — Immediate response**  
Is short-horizon realised response unusual relative to generic expected impact?

**Family B — Response path**  
Does later response path differ from generic reference trajectory?

Historical 25-event work remains post-hoc exploratory if this family structure was developed after Channel A.

---

# 25. LITERATURE BASE ALREADY STUDIED

## Major books

### Bouchaud, Bonart, Donier, Gould — *Trades, Quotes and Prices*
Key themes: order-flow persistence, impact, metaorders, propagator models, latent liquidity, Hawkes, market response.

### Joel Hasbrouck — *Empirical Market Microstructure*
Key themes: empirical identification, trade/quote inference, liquidity, depth, resiliency, price innovations, econometric limitations.

### Abergel et al. — *Limit Order Books*
Key themes: empirical LOB structure, queue models, Hawkes, simulation, predictability, order-book dynamics.

### *Econophysics of Order-driven Markets*
Especially relevant: resilience, price-impact decay, distinction between book resilience and impact decay, activity, tick effects, data quality.

These books were systematically traversed previously.

Caveat:
- PDF parsers may corrupt some mathematical glyphs.
- Critical formulas/figures should be reopened and visually verified before relying on exact notation.
- Do not claim perfect verbatim memory of every symbol.

## Important research lines already considered

- Cont, Kukanov, Stoikov — OFI and price impact
- Xu, Gould, Howison — MLOFI
- Eisler, Bouchaud, Kockelkoren — all-event impact/history dependence
- Lillo/Farmer — long-memory order flow
- Tóth et al. — participant reaction/impact
- Taranto et al. — transient vs history-dependent impact
- Bechler/Ludkovski — order flow and resiliency
- queue imbalance / tick-regime literature
- Hawkes / queue-reactive Hawkes literature
- funding/liquidity spiral literature
- matching / data snooping / multiple-testing literature
- recent crypto/perpetual microstructure research
- Binance feed semantics / RPI / aggregation issues
- hidden-liquidity and measurement-error work

Do not over-weight recent SSRN/preprint work relative to canonical literature.

---

# 26. CURRENT STAGE GATE

The broad literature-review phase is considered largely mature.

The next correct stage is:

# DATA FEASIBILITY AUDIT V1

Do not start Flow Surprise / Impact Surprise model fitting before this audit.

---

# 27. DATA FEASIBILITY AUDIT V1 — REQUIRED QUESTIONS

Audit the actual repository/database/collectors.

## Storage
- tables,
- columns,
- date ranges,
- symbols,
- missing intervals,
- duplicates,
- collector restarts,
- timestamp basis/time zone.

## Trades
- raw `trades`?
- `aggTrade`?
- both?
- exchange timestamp?
- receive timestamp?
- trade ID?
- aggregate ID?
- buyer-maker?
- quote quantity?
- `nq`?
- RPI fields?
- schema changes?

## Order book
- bookTicker?
- snapshot?
- partial depth?
- diff-depth?
- multiple levels?
- update IDs?
- sequence continuity?
- reconstruction logic?
- sampling interval?
- missing-update detection?

## Liquidation
- exact source/stream?
- exact payload fields?
- notional formula?
- `q`, `z`, `l`, `p`, `ap` availability?
- timestamp semantics?
- collector gaps/throttling?

## Symbol metadata
- tick size history?
- step size history?
- contract/filter changes?
- listing dates?

## 25 historical E-DER events
For each event:
- which trade-flow variables are observable?
- which book-state variables are observable?
- which measurement uncertainties apply?
- which mechanism claims are testable?
- which are NOT IDENTIFIABLE?

---

# 28. ABSOLUTE AUDIT RULE

Before calculating:

- OFI,
- MLOFI,
- Flow Surprise,
- Impact Surprise,
- replenishment,
- cancellations,
- absorption,
- hidden liquidity,
- reaction turnover,
- any dynamic exit,

the audit must first prove the stored data supports the definition.

Canonical instruction:

> **Do not calculate a feature until you prove the stored data supports its definition.**

---

# 29. FUTURE AI PROACTIVE-RESEARCH MANDATE

The assistant should proactively look for:

- hidden assumptions,
- measurement incompatibilities,
- feed-semantic changes,
- dependence / pseudo-replication,
- untested competing mechanisms,
- better nulls,
- falsification opportunities,
- missing robustness,
- opportunities to simplify,
- claims that should be downgraded to NOT IDENTIFIABLE.

Every proactive idea should state:

1. scientific motivation,
2. data requirement,
3. falsification condition,
4. data-mining risk,
5. whether it alters any frozen rule,
6. whether it is historical exploratory or forward confirmatory.

Do not propose complexity for its own sake.

---

# 30. RESEARCH LEDGER FORMAT

Every new idea should be logged with:

- **ID**
- **Status:** FROZEN / ACTIVE / EXPLORATORY / POST-HOC / REJECTED / NOT IDENTIFIABLE / COMPLETED
- **Question**
- **Null**
- **Alternative**
- **Data required**
- **Test**
- **Falsification**
- **Forbidden tuning**
- **Historical role**
- **Forward role**
- **Decision**

---

# 31. DECISION LOG — CURRENT IMPORTANT DECISIONS

**D1** Renewed SELL liquidation is not automatically deterioration.  
**D2** Raw liquidation amount is not the primary mechanism variable.  
**D3** Impact compression is not itself proof of absorption.  
**D4** Visible book state is not total executable/latent liquidity.  
**D5** Flow Surprise and Impact Surprise are separate innovations.  
**D6** Impact Surprise is a conditional response anomaly, not automatically alpha or mechanism proof.  
**D7** Measurement sensitivity stays separate from residual magnitude.  
**D8** Modern rich data is a validation laboratory, not a historical correction oracle.  
**D9** Event-count/run-length/Hawkes measures are secondary unless event semantics are proven.  
**D10** Fixed-clock signed-notional flow is the current preferred Flow Surprise primitive.  
**D11** Minimum adequate null is preferred over a complex predictive model.  
**D12** Historical 25-event mechanism analysis is post-hoc; forward sample is confirmatory.  
**D13** Use NOT IDENTIFIABLE whenever the data cannot separate mechanisms.  
**D14** The next stage is Data Feasibility Audit V1.

---

# 32. WHAT THE ASSISTANT SHOULD DO NEXT

1. Inspect the actual Eclipse repository/database.
2. Build a factual data inventory.
3. Build a feed-semantics timeline.
4. Build an event-by-event E-DER observability matrix.
5. Classify candidate variables as:
   - VALID,
   - PROXY ONLY,
   - UNSUPPORTED,
   - NOT IDENTIFIABLE.
6. Only then design Flow Surprise / Impact Surprise V1.

If the audit reveals a specific unresolved Binance/API/statistics issue, return to targeted research for that issue.

---

# 33. STANDARD FOR CLAIM LANGUAGE

Prefer:

- consistent with,
- supports,
- does not support,
- conditional response anomaly,
- displayed-book recovery,
- observed forced-liquidation pressure proxy,
- public aggregated-flow process,
- mechanism-compatible,
- not identifiable.

Avoid without much stronger evidence:

- proved,
- absorption confirmed,
- seller exhaustion confirmed,
- market makers absorbed it,
- true liquidation volume,
- hidden alpha discovered,
- caused the rebound.

---

# 34. FINAL RESEARCH PRINCIPLE

Always ask:

> What would have to be true for this interpretation to be wrong?

Test that before promoting the interpretation.

The long-term goal is not the most exciting E-DER story.

It is the most defensible one.

---

# END OF V0.1
