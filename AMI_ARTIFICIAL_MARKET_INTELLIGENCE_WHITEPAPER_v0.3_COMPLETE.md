# AMI — Artificial Market Intelligence
## Whitepaper, Scientific Constitution and Canonical Engineering Specification

**Document:** `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.3_COMPLETE.md`  
**Version:** `0.3.0`  
**Status:** `FOUNDATIONAL WHITEPAPER / CANONICAL ENGINEERING SPECIFICATION / LIVING RESEARCH CONSTITUTION`  
**Initial research laboratory:** S34 liquidation and cascade intelligence  
**Long-term scope:** General-purpose, multi-market Artificial Market Intelligence  
**Date:** 2026-07-03


## Revision 0.3 — Cycle Intelligence and Decision Research

Version 0.3 extends the original AMI constitution with a complete market-cycle research architecture.

The central revision is conceptual:

```text
Previous framing:
Event → Direction → Entry → Exit

Revised framing:
Cycle state
+ state age
+ structural location
+ event geometry
+ market clock
+ position path
+ execution feasibility
+ evidence independence
+ uncertainty
↓
Available actions
↓
Conditional action values
↓
Abstain / observe / research / permitted action
```

The new specification adds:

- position-aware decision research for FLAT, ALREADY_LONG and ALREADY_SHORT states;
- unconditional LONG-genesis research without conditioning on a future event;
- failed-fade and squeeze-continuation LONG routes;
- independent market-cycle identification and event deduplication;
- post-event path taxonomy with soft, horizon-dependent labels;
- signal aging, state dwell time and market-clock normalization;
- explicit separation of scalp, intraday and swing routes;
- competing-risk and progress-conditioned hold research;
- structural-location, event-geometry and micro-mechanism layers;
- execution mechanics, fill probability, latency and capacity research;
- evidence-contamination, multiple-testing and researcher-exposure ledgers;
- causal-assumption, OOD, novelty and uncertainty registries;
- sequential policy evaluation rather than hindsight-best action selection;
- a comprehensive forward observatory and research dashboard;
- a canonical scientific-question system spanning the complete LONG–SHORT cycle;
- and a BUY-fade case study demonstrating why information can be real yet operationally late.

No component in this revision grants live permission. The entire extension remains research-only until explicit, versioned promotion gates are passed.

---

## Abstract

Artificial Market Intelligence, or **AMI**, is a proposed scientific and engineering architecture for building systems that do more than generate trading signals.

AMI is designed to observe markets, construct multi-timeframe world representations, discover hidden structural states, generate scientific questions, formulate falsifiable hypotheses, design experiments, learn causal and probabilistic relationships, maintain a governed body of market knowledge, and convert validated knowledge into operational decisions.

The central idea is that financial markets should not be represented as a collection of independent indicators or isolated entry rules. Markets should be modelled as evolving systems composed of:

- global regimes;
- sessions;
- structural phases;
- order-book dynamics;
- participant behaviour;
- inventory pressure;
- leverage;
- funding;
- open interest;
- information transfer;
- liquidity stress;
- cascades;
- trade lifecycle states;
- portfolio states;
- and changing scientific beliefs.

Within this architecture, liquidation is not treated as the beginning of a strategy. It is treated as one visible consequence of a deeper market process. LONG and SHORT are not independent strategies. They are possible actions produced by different phases of the same evolving structure. Entry, hold and exit are not one decision. They are separate inference problems.

AMI also introduces a second level of intelligence: it does not merely research markets; it researches its own research process. It evaluates which validation methods predict forward survival, which agents generate false discoveries, which theories decay, which assumptions have become invalid, and which unanswered questions have the highest information value.

The governing principle is:

> **AMI should not merely accumulate market facts. It should govern belief under uncertainty.**

The long-term objective is not simply a profitable trading bot. It is a continuously learning, multi-timeframe, probabilistic, causality-aware digital twin and autonomous scientist of financial markets.

---

# Reader’s Guide

This document has two functions.

## Function One — Whitepaper

It explains:

- why AMI should exist;
- what philosophical problem it solves;
- how its view of markets differs from ordinary trading systems;
- how LONG and SHORT become connected structural phases;
- how scientific discovery, machine learning and epistemic governance fit together;
- and why trading is only one downstream application.

## Function Two — Canonical System Specification

It defines:

- architectural layers;
- state models;
- knowledge schemas;
- research protocols;
- promotion and demotion rules;
- system boundaries;
- data contracts;
- implementation phases;
- operational safeguards;
- and instructions for engineering agents.

This document is therefore both a conceptual whitepaper and a practical blueprint.

---

# Table of Contents

## Volume I — Foundations

1. The Problem with Conventional Trading Systems  
2. The AMI Thesis  
3. Philosophy of Ignorance  
4. Markets as Evolving Systems  
5. Intelligence Beyond PnL  
6. System Identity and Boundaries  

## Volume II — Market World Model

7. Reality Interface  
8. Global Market State  
9. Multi-Timeframe Representation  
10. Market Physics  
11. Structure Engine  
12. Microstructure  
13. Order Book Evolution  
14. Maker and Inventory Models  
15. Funding and OI  
16. Cross-Exchange Networks  
17. Liquidity Stress and Cascades  

## Volume III — Structural Trading Intelligence

18. LONG and SHORT as Connected Phases  
19. LONG Research Engine  
20. SHORT Research Engine  
21. Structural Transition Engine  
22. Swing Structure  
23. Trade Lifecycle Engine  
24. Position Management  
25. Exit Intelligence  
26. Portfolio Brain  

## Volume IV — Scientific Intelligence

27. Research Operating System  
28. Autonomous Scientist  
29. Question and Hypothesis Generation  
30. Theory Builder  
31. Counterfactual Research  
32. Research Marketplace  
33. Meta-Research  

## Volume V — Epistemic Governance

34. Epistemic Core  
35. Knowledge Objects  
36. Evidence Hierarchy  
37. Belief Revision  
38. Contradiction Resolution  
39. Promotion and Demotion  
40. Epistemic Risk  
41. Scientific Constitution  

## Volume VI — Machine Intelligence

42. Latent States  
43. World Model  
44. Digital Twin  
45. Probability Engine  
46. Market Memory  
47. Explainability  
48. Self-Improvement  

## Volume VII — Engineering Blueprint

49. Service Map  
50. Data Contracts  
51. Event Bus  
52. Storage Architecture  
53. APIs  
54. Validation Standards  
55. Roadmap  
56. Build Instructions  

## Volume VIII — Cycle Intelligence and Forward Research

57. Why Event-Centric Research Is Incomplete  
58. Canonical Market-Cycle State Machine  
59. Position-Aware Action Space  
60. Unified LONG and SHORT Research Architecture  
61. Signal Aging and Market Clock  
62. Scalp, Intraday and Swing Route Separation  
63. Dynamic Hold, Competing Risks and Progress  
64. Structural Location, Event Geometry and Mechanism  
65. Cycle Integrity, Event Overlap and Censoring  
66. Regime Transitions and Multi-Horizon Conflict  
67. Execution, Latency, Capacity and Venue  
68. Forward Intelligence Observatory and Dashboard  
69. Canonical Research Question System  
70. Evidence Independence and Epistemic Safety  
71. OOD, Uncertainty, Calibration and Abstention  
72. Sequential Policies, Benchmarks and Regret  
73. Portfolio, Sequence and Capital-Path Intelligence  
74. Market-Structure, Shock and Derivatives Context  
75. Active Data Acquisition and Failure Meta-Research  
76. BUY-Fade Case Study  
77. Implementation Programme  
78. Version 0.3 Definition of Done  

---

# Volume I — Foundations

# 1. The Problem with Conventional Trading Systems

Most trading systems are built around a narrow loop:

```text
Indicator
    ↓
Signal
    ↓
Entry
    ↓
Exit
    ↓
PnL
```

This architecture is easy to implement but scientifically weak.

It creates several recurring failures.

## 1.1 It confuses prediction with explanation

A feature may correlate with an outcome without explaining why the outcome occurs. A backtest can therefore look strong while remaining fragile under regime change.

## 1.2 It treats the market as static

Thresholds are usually estimated from one historical period and then treated as if the relationship were timeless. In reality, exchange structure, participants, liquidity, fees, leverage and dominant venues change.

## 1.3 It compresses the full trade lifecycle into one rule

A correct entry does not guarantee a correct hold. A correct hold does not guarantee a correct exit. Many systems discover entry alpha but destroy it with management that belongs to a different timeframe.

## 1.4 It hides uncertainty

A model may output LONG with no representation of:

- confidence;
- applicable regime;
- evidence quality;
- data health;
- known contradictions;
- execution uncertainty;
- or expiration risk.

## 1.5 It forgets failed research

Rejected ideas disappear from notebooks, only to be rediscovered and retested later. This wastes time and creates false novelty.

## 1.6 It rewards storytelling

After seeing the result, researchers construct a mechanism that appears to explain it. This is not prediction. It is retrospective narrative.

## 1.7 It treats PnL as the only intelligence metric

A system can make money during one regime while learning nothing general. It can also produce valuable scientific knowledge without immediate PnL.

AMI is designed to solve these failures architecturally rather than procedurally.

---

# 2. The AMI Thesis

The AMI thesis is:

> **Markets are evolving, partially observed, multi-agent systems. Their behaviour should be modelled through states, transitions, mechanisms, uncertainty and scientific revision rather than isolated entry rules.**

The implications are substantial.

## 2.1 Liquidation becomes a symptom

Instead of:

```text
Liquidation occurred
Should we trade?
```

AMI asks:

```text
Which market process created the liquidation?
Which phase of the process is active?
Is the process accelerating, exhausting or reversing?
What state transition is now most probable?
```

## 2.2 LONG and SHORT become structural outputs

LONG and SHORT are not separate universes.

A complete structural cycle can contain both:

```text
Distribution
    ↓
Breakdown
    ↓
Cascade
    ↓
SHORT pressure
    ↓
Exhaustion
    ↓
Absorption
    ↓
Reclaim
    ↓
LONG recovery
```

## 2.3 Trading becomes one application

AMI may eventually support:

- trade selection;
- market monitoring;
- risk forecasting;
- regime analysis;
- execution planning;
- anomaly detection;
- research automation;
- cross-market theory building;
- and market simulation.

Trading remains important, but it no longer defines the full system.

---

# 3. Philosophy of Ignorance

AMI begins with an explicit philosophy:

> The purpose of intelligence is not to create the appearance of certainty. It is to reduce uncertainty while representing what remains unknown.

Every conclusion is temporary.

Every theory is revisable.

Every strategy is treated as a hypothesis awaiting more evidence.

Every operational permission can be withdrawn.

## 3.1 Three epistemic categories

```text
KNOWN
Supported by current evidence

KNOWN UNKNOWN
A defined question with missing evidence or missing data

POTENTIAL UNKNOWN
An anomaly or residual that existing models do not explain
```

AMI must not convert the second or third category into confidence merely because an operational decision is desired.

## 3.2 The ability to say “I do not know”

A mature system should be capable of outputs such as:

```text
Direction: UNKNOWN
Reason: conflicting 4h and 1D states
Data health: DEGRADED
Forward evidence: insufficient
Permitted action: OBSERVE_ONLY
```

This is not failure. It is epistemic competence.

---

# 4. Markets as Evolving Systems

AMI represents the market as interacting layers.

```text
GLOBAL MARKET
      ↓
SESSION
      ↓
ASSET STRUCTURE
      ↓
MICROSTRUCTURE
      ↓
BOOK
      ↓
FLOW
      ↓
LEVERAGE
      ↓
PRESSURE
      ↓
CASCADE
      ↓
TRADE
      ↓
POSITION
      ↓
EXIT
      ↓
LEARNING
```

Each layer has its own state, timescale and uncertainty.

## 4.1 States, not isolated values

AMI should not merely observe:

```text
Funding = -0.01
```

It should represent:

```text
Funding State:
Negative
Falling
Historically crowded
OI expanding
Price no longer falling
Possible seller exhaustion
```

## 4.2 Transitions, not static labels

The most important object is often not the current state but the transition:

```text
BOOK_PULLING
    ↓
BOOK_STABLE
    ↓
ABSORBING
```

A transition may create an opportunity even when none of the individual states is sufficient alone.

## 4.3 Duration matters

The same state may have different meaning depending on age.

```text
Fresh expansion
Mature expansion
Late expansion
Exhausted expansion
```

Every state must therefore carry duration and transition history.

---

# 5. Intelligence Beyond PnL

AMI should be evaluated through multiple dimensions.

## 5.1 Scientific intelligence

- Does a hypothesis replicate?
- Does a theory produce correct new predictions?
- Does the system detect contradictions?
- Does it revise beliefs quickly?

## 5.2 Predictive intelligence

- Are probabilities calibrated?
- Are transition forecasts accurate?
- Are tail events represented?

## 5.3 Epistemic intelligence

- Does the system know when confidence is unjustified?
- Does it identify stale knowledge?
- Does it preserve uncertainty?

## 5.4 Operational intelligence

- Does validated knowledge survive real execution?
- Are decisions traceable?
- Does the system demote failing rules?

## 5.5 Economic intelligence

- Does the system produce positive expectancy?
- Does it use capital efficiently?
- Does it reduce drawdown and tail risk?

PnL is essential, but it is only one layer of intelligence.

---

# 6. The AMI Stack

The complete AMI stack is:

```text
EPISTEMIC GOVERNOR
        │
        ├── governs what may be believed
        ├── governs what may be researched
        └── governs what may influence decisions

AUTONOMOUS SCIENTIST
        │
        ├── observes anomalies
        ├── creates questions
        ├── designs experiments
        └── updates theories

WORLD MODEL
        │
        ├── represents market states
        ├── forecasts transitions
        └── simulates futures

STRUCTURE ENGINE
        │
        ├── connects LONG and SHORT
        ├── maps swings
        └── models phases

TRADE LIFECYCLE ENGINE
        │
        ├── entry
        ├── hold
        ├── scale
        ├── lock
        ├── exit
        └── reverse

PORTFOLIO BRAIN
        │
        ├── exposure
        ├── correlation
        ├── risk
        └── capital allocation

REALITY INTERFACE
        │
        └── external market observations
```

---

# Volume II — Market World Model

# 7. Reality Interface as a Scientific Instrument

The quality of AMI’s intelligence is bounded by the quality of its observations.

The Reality Interface must be treated like a scientific instrument.

It should measure:

```text
Trades
Order books
Book ticker
Liquidations
Funding
OI
Spot/perpetual basis
Options
Cross-exchange flows
Macro
News
On-chain
Sentiment
Execution telemetry
```

Every feed must expose staleness, gaps, schema, latency and source identity.

A missing feed is not a zero.

A stale value is not a current value.

A reconstructed value is not an observed value.

---

# 8. Global Market State

Before evaluating ETH, SOL or any individual trade, AMI should answer:

```text
What kind of day is this?
```

Components:

- global risk appetite;
- BTC trend and dominance;
- market breadth;
- leverage level;
- volatility;
- stablecoin flows;
- macro-event proximity;
- options expiry;
- weekend/session effects;
- correlation regime;
- liquidity fragmentation.

The Global State constrains local interpretation.

Example:

```text
GLOBAL_STATE:
Risk-Off
High Volatility
Leverage Contracting
US Session
Macro Event +4h
BTC Dominant
```

A local LONG setup may still exist, but its expected duration and risk should change.

---

# 9. Multi-Timeframe Intelligence

Required timeframes:

```text
1m → 5m → 15m → 1h → 4h → 1D → 1W
```

The 1D timeframe is mandatory because a state that appears to be a failed intraday SHORT may be the beginning of a daily LONG recovery.

## 9.1 Timeframe roles

```text
1m
Execution and microstructure

5m
Scalp pressure

15m
Immediate transition

1h
Intraday structure

4h
Primary swing

1D
Daily direction and structural regime

1W
Background market cycle
```

## 9.2 Timeframe nesting

Example:

```text
5m SHORT scalp
1h exhaustion
4h reclaim
1D LONG recovery
```

The correct action may be:

```text
Take short profit
Do not hold short
Wait for long confirmation
```

This requires timeframe-aware lifecycle intelligence, not one universal label.

---

# 10. Market Physics

AMI should search for recurring principles.

Candidate theories include:

- momentum conservation;
- momentum exhaustion;
- liquidity attraction;
- liquidity vacuum;
- inventory mean reversion;
- dealer hedging;
- volatility expansion;
- volatility decay;
- order-flow memory;
- cascade propagation;
- reflexivity;
- crowding release;
- attention rotation.

Each proposed law must generate predictions.

Example:

```text
Theory:
Forced selling is absorbed by passive buyers.

Predictions:
1. Price impact per liquidation dollar declines.
2. Bid pull stops.
3. New SELL liquidations fail to create new lows.
4. Reversal probability rises.
5. The effect survives chronological holdout.
```

---

# 11. Market Structure Engine

Canonical phases:

```text
Accumulation
Compression
Early Expansion
Expansion
Mature Trend
Distribution
Breakdown
Cascade
Exhaustion
Absorption
Reclaim
Recovery
Reaccumulation
Reversal
Range
Dislocation
```

These phases form a probabilistic graph.

AMI should estimate:

- current phase;
- phase confidence;
- age;
- likely next phases;
- expected duration;
- preferred direction;
- invalidation.

---

# 12. Structural LONG and SHORT

The structure engine is the bridge between both directions.

## 12.1 SHORT path

```text
Distribution
    ↓
Failed strength
    ↓
Liquidity withdrawal
    ↓
Breakdown
    ↓
Cascade
    ↓
SHORT
```

## 12.2 LONG path

```text
Seller exhaustion
    ↓
Absorption
    ↓
No new low
    ↓
Reclaim
    ↓
Recovery
    ↓
LONG
```

## 12.3 Full swing

```text
SHORT opportunity
    ↓
SHORT management
    ↓
SHORT weakening
    ↓
Transition uncertainty
    ↓
LONG confirmation
    ↓
LONG swing
```

The system must permit `NO_TRADE` between the two.

---

# 13. The LONG Research Programme

LONG must be studied as both:

1. an independent alpha family;
2. a transition emerging from SHORT failure.

## 13.1 Independent LONG families

- compression breakout;
- spot-led demand;
- funding/OI dislocation;
- order-book absorption;
- cross-exchange lead;
- structural reclaim;
- stablecoin or institutional flow;
- options/dealer hedging.

## 13.2 SHORT-failure LONG

Research triggers:

- SHORT reached MFE and gave back;
- breakdown failed;
- new SELL liquidation created no new low;
- BTC synchronization recovered;
- book stopped pulling;
- OFI flipped;
- basis normalized;
- funding stayed negative while price stabilized.

## 13.3 LONG horizon map

Every candidate must be tested at:

```text
30m
1h
2h
4h
6h
12h
24h
```

This is how AMI determines whether it is a scalp, swing or daily structure.

## 13.4 LONG failure map

AMI should also study:

- failed reclaim;
- absorption collapse;
- BTC divergence;
- OI expansion against price;
- renewed book pull;
- renewed cascade;
- daily resistance;
- macro shock.

---

# 14. The SHORT Research Programme

SHORT should be developed through:

- buyer exhaustion;
- failed breakout;
- distribution;
- liquidity withdrawal;
- maker retreat;
- positive funding crowding;
- OI expansion without acceptance;
- cross-exchange weakness;
- delayed cascade propagation.

SHORT is not simply the inverse of LONG. Its speed, tail and execution characteristics can be different.

---

# 15. Microstructure as Motion

The order book must be modelled as a sequence.

```text
Book appears
Book thickens
Book pulls
Book refills
Book migrates
Book flips
Book collapses
Book recovers
```

Potential features:

- pull speed;
- refill speed;
- queue survival;
- wall migration;
- impact per aggressive dollar;
- liquidity holes;
- depth recovery;
- spread response;
- imbalance persistence.

---

# 16. Maker and Inventory Inference

Maker identity is not observed directly.

AMI should infer maker state probabilistically.

```text
Aggressive sells arrive
Bid remains
Impact falls
→ probable absorption
```

```text
Aggressive sells arrive
Bid disappears
Impact rises
→ probable retreat
```

Inventory theories should remain separate from direct observations. The Epistemic Governor must prevent inferred maker narratives from being promoted as causal fact without stronger evidence.

---

# 17. Funding and OI as Dynamic Systems

Required states include:

```text
Funding level
Funding slope
Funding velocity
Funding acceleration
Funding divergence

OI level
OI slope
OI acceleration
OI expansion
OI collapse
Price/OI divergence
Funding/OI interaction
```

The purpose is not to ask whether funding is positive.

The purpose is to ask:

```text
What leverage process is evolving?
```

---

# 18. Cross-Exchange Intelligence

AMI should represent exchanges as a network.

```text
CME
  ↓
Bybit
  ↓
Binance
  ↓
Altcoins
```

Questions:

- Which venue leads?
- Is stress universal or isolated?
- Does spot lead perpetuals?
- Is one exchange absorbing another?
- Does liquidation synchronize?

Cross-exchange data can reveal whether an event is local microstructure or broad market information.

---

# 19. Liquidity Stress and Cascade Evolution

Cascade phases:

```text
Pre-Cascade
Early Cascade
Acceleration
Peak
Exhaustion
Recovery
Echo
Failed Cascade
```

The event itself is not the alpha.

The alpha may exist in:

- the state producing it;
- the way it evolves;
- the transition after it;
- the trade lifecycle created by it.

---

# Volume III — Trade Lifecycle Intelligence

# 20. Entry, Hold and Exit Are Different Problems

```text
ENTRY ALPHA
≠
HOLD ALPHA
≠
EXIT ALPHA
```

A trade that moves +50 bps and later closes -50 bps may have:

- correct entry;
- missing hold intelligence;
- late exit;
- or a direction transition.

AMI must not label the entire trade “wrong” without locating the failed stage.

---

# 21. Trade Lifecycle States

```text
OPEN
HEALTHY
ACCELERATING
STALLING
WEAKENING
EXHAUSTED
RECOVERING
LOCKED
REVERSING
INVALIDATED
CLOSED
```

Transitions should be learned from post-entry features.

---

# 22. Post-Entry Research

Snapshots:

```text
Entry
+20
+40
+50
+75
+100 bps
3m
5m
10m
15m
30m
60m
90m
120m
MFE
MAE
Exit
```

At each snapshot, record:

- book state;
- pull/refill;
- taker flow;
- CVD;
- OFI;
- BTC synchronization;
- OI;
- funding;
- basis;
- volatility;
- liquidation pressure;
- structural phase.

---

# 23. MFE State Classifier

When a trade reaches +50 bps:

```text
Will it:
A. continue;
B. return to breakeven;
C. finish negative;
D. time-exit positive;
E. create a reversal?
```

The classifier must use only information available at that moment.

This is a direct route to discovering exit alpha.

---

# 24. Management Actions

```text
HOLD
SCALE
REDUCE
LOCK
EXIT
REVERSE
NO_ACTION
```

Each candidate action must be tested against:

- expectancy;
- drawdown;
- tail;
- execution;
- complexity;
- forward survival;
- epistemic risk.

---

# 25. Position and Portfolio Intelligence

Trade management cannot ignore portfolio context.

A good trade may be rejected because:

- another position expresses the same state;
- portfolio exposure is concentrated;
- daily loss throttle is active;
- data health is degraded;
- the active theory is under demotion review.

AMI manages positions and beliefs together.

---

# Volume IV — Scientific Intelligence

# 26. Research OS

The Research OS formalizes:

```text
Observation
Question
Hypothesis
Experiment
Evidence
Replication
Theory
Knowledge
Application
```

No stage should be skipped.

---

# 27. Autonomous Scientist

AMI should eventually generate its own questions.

Example:

```text
Observation:
SHORT_NOISY frequently reaches positive MFE and gives back.

Questions:
Is the route scalp-only?
Does BTC recovery predict giveback?
Does book absorption predict LONG transition?
Is the effect daily rather than intraday?
```

It then generates competing hypotheses and preregistered experiments.

---

# 28. Theory Builder

A theory should contain:

```text
Mechanism
Scope
Predictions
Supporting evidence
Contradictions
Alternative theories
Revision history
```

A theory that explains everything after the fact is useless.

A useful theory predicts something new.

---

# 29. Counterfactual Research

AMI should ask:

```text
What if the trade was not opened?
What if the opposite direction was used?
What if the exit occurred at MFE?
What if the scale-in occurred?
What if BTC had not recovered?
```

Observed evidence and simulated counterfactuals must remain clearly separated.

---

# 30. Research Economics

Every experiment consumes:

- compute;
- time;
- data;
- statistical power;
- researcher attention;
- overfit budget.

Questions should be ranked by:

```text
Information gain
Economic value
Risk reduction
Contradiction resolution
Novelty
Generalization
Falsifiability
Data readiness
Cost
```

---

# Volume V — Epistemic Governance

# 31. Why an Epistemic Governor Is Necessary

Without governance, an autonomous scientist can create thousands of plausible stories.

The Epistemic Governor controls:

- what may be claimed;
- what may be used;
- what must be retested;
- what has expired;
- what is forbidden from live decisions.

It is the scientific control plane above all other intelligence.

---

# 32. Knowledge Objects

Every claim should carry:

```text
Claim
Type
Evidence
Replications
Holdout
Forward evidence
Contradictions
Scope
Assumptions
Confidence
Expiration
Falsification
Permissions
Genealogy
```

A finding without this context is not AMI knowledge.

---

# 33. Belief Revision

New evidence can:

```text
REINFORCE
RESTRICT
REVISE
REJECT
```

Old versions remain in history.

The system should never silently rewrite its past.

---

# 34. Promotion and Demotion

Typical path:

```text
Observation
Preliminary
Replicated
Holdout Validated
Forward Validating
Operational Candidate
Provisionally Accepted
```

Demotion triggers:

- forward degradation;
- data-quality discovery;
- execution mismatch;
- regime shift;
- contradiction;
- drift;
- calibration failure.

---

# Volume VI — Machine Intelligence

# 35. Latent State Discovery

Machine learning can discover states humans did not define.

Candidate tools:

- clustering;
- HMM;
- switching state-space models;
- change-point detection;
- sequence embeddings;
- contrastive learning.

Models may discover first; humans name later.

---

# 36. World Model and Digital Twin

The World Model receives current states and produces scenario distributions.

```text
Current state
    ↓
Continuation scenario
Reversal scenario
Compression scenario
Panic scenario
```

The Digital Twin compares simulated futures with reality and updates calibration.

---

# 37. Market Memory

AMI should retrieve similar historical states while adjusting for:

- regime;
- symbol;
- venue;
- timeframe;
- execution;
- feature version;
- data quality.

Similarity does not equal identity. Retrieval confidence must be explicit.

---

# 38. Explainability

Every recommendation should show:

- supporting evidence;
- counter-evidence;
- active states;
- probabilities;
- uncertainty;
- Knowledge Objects used;
- rejected alternatives;
- permitted action level.

---

# Volume VII — Engineering and Delivery

# 39. System Build Philosophy

The whitepaper must not be interpreted as:

```text
Implement everything immediately.
```

It means:

```text
Build foundations first.
Preserve scientific boundaries.
Add autonomy gradually.
Never bypass live guardrails.
```

---

# 40. Recommended Build Order

```text
Phase 0
Constitution, schemas, lineage

Phase 1
State foundation

Phase 2
Structure and LONG/SHORT transition engine

Phase 3
Trade lifecycle

Phase 4
Research OS

Phase 5
Epistemic Governor

Phase 6
ML and latent states

Phase 7
World Model

Phase 8
Autonomous Scientist

Phase 9
Cross-market generalization
```

---

# 41. Deliverable Standard

Each engineering phase should return:

- code;
- schemas;
- tests;
- migration notes;
- documentation;
- validation report;
- updated roadmap;
- updated Knowledge Objects;
- explicit list of untouched live components.

---

# 42. Whitepaper Conclusion

AMI is an attempt to replace the idea of a trading bot with a scientific market intelligence.

It treats:

- data as observation;
- features as evidence;
- states as representations;
- transitions as processes;
- theories as temporary;
- uncertainty as mandatory;
- trading as downstream;
- learning as continuous.

Its ambition is not to be certain.

Its ambition is to know when certainty is justified.

---

# Canonical Technical Specification

The complete canonical technical specification follows. It defines the detailed architecture, governance model, schemas, research rules and implementation roadmap that operationalize the whitepaper above.

---

## Canonical System Specification, Scientific Constitution and Engineering Roadmap

**Document ID:** `AMI-SPEC-0001`  
**Version:** `0.1.0-draft`  
**Status:** `CANONICAL_DRAFT`  
**Primary format:** Living Markdown specification  
**Scope:** Market intelligence, scientific research, probabilistic world modelling, structural LONG/SHORT discovery, trade lifecycle intelligence, portfolio decision support and epistemic governance  
**Initial application family:** S34 liquidation/cascade research  
**Long-term objective:** General-purpose, multi-market Artificial Market Intelligence  
**Last updated:** 2026-07-02

---

## Document Control

This document is the canonical architectural reference for AMI. It is not a strategy memo, a list of features, or a one-time research prompt. It defines a continuously evolving scientific and engineering system.

Every material change should produce:

```text
Version change
Decision record
Reason for change
Affected components
Affected Knowledge Objects
Required migrations
Validation status
```

Recommended repository layout:

```text
docs/ami/
├── AMI_SYSTEM_SPECIFICATION.md
├── AMI_CHANGELOG.md
├── AMI_DECISION_RECORDS/
├── AMI_KNOWLEDGE_SCHEMA.md
├── AMI_STATE_TAXONOMY.md
├── AMI_RESEARCH_PROTOCOLS.md
└── AMI_ROADMAP.md
```

This document may later be divided into several volumes, but this file remains the master index and constitutional source.

---

# Executive Summary

AMI is not a liquidation bot and not merely a trading system.

AMI is a continuously learning scientific system that:

1. observes markets through many data interfaces;
2. converts raw observations into multi-timeframe market states;
3. discovers hidden structures and market mechanisms;
4. generates its own research questions;
5. forms falsifiable hypotheses and theories;
6. designs and executes statistically rigorous experiments;
7. represents uncertainty explicitly;
8. revises or expires beliefs when evidence changes;
9. predicts probabilistic state transitions;
10. generates LONG, SHORT, NO-TRADE and management intelligence;
11. explains every operational recommendation through traceable evidence;
12. treats trading as one downstream application of broader market cognition.

The central shift is:

```text
OLD QUESTION
A liquidation happened. Should we trade it?

NEW QUESTION
Which structural process has the market followed?
Which state is active now?
What are the most likely next transitions across 1m, 5m, 15m,
1h, 4h, 1D and 1W?
What action is justified by the available evidence?
```

The system is built around five inseparable ideas:

```text
WORLD MODEL
How the market is represented

STRUCTURE ENGINE
How market phases and LONG/SHORT transitions are modelled

TRADE LIFECYCLE ENGINE
How an open position changes state after entry

AUTONOMOUS SCIENTIST
How the system asks and tests its own questions

EPISTEMIC GOVERNOR
How the system knows what is proven, uncertain, expired or forbidden
```

AMI must never confuse a promising backtest with knowledge, a correlation with a mechanism, or a mechanism with a live permission.

---

# Part I — Identity, Mission and Boundaries

## 1. System Identity

**Name:** Artificial Market Intelligence  
**Short name:** AMI  
**Initial domain:** Crypto derivatives and spot markets  
**Future domains:** Equities, FX, commodities, options and cross-asset markets

S34 is not AMI itself. S34 is one alpha family, one empirical laboratory and one early application running inside AMI.

```text
AMI
├── S34 Cascade Intelligence
├── LONG Structure Research
├── SHORT Structure Research
├── Order-Book Mechanism Research
├── OI / Funding Intelligence
├── Cross-Exchange Intelligence
├── Trade Lifecycle Intelligence
└── Future Alpha Families
```

## 2. Mission

AMI’s mission is:

> To observe, model, explain, simulate and predict the structural evolution of financial markets while governing every belief under uncertainty.

## 3. Non-Goals

AMI is not designed to:

- maximize the number of strategies;
- produce a trade for every market event;
- replace scientific validation with model complexity;
- treat PnL as the only measure of intelligence;
- promote rules because they are narratively attractive;
- hide failed experiments;
- claim universal validity from a single market regime;
- use machine learning as an opaque permission to trade;
- optimize execution before proving selection alpha;
- overwrite live safety constraints without explicit governance.

## 4. Core Constitutional Principles

```text
1. No claim without provenance.
2. No operational promotion without untouched validation.
3. No mechanism claim from correlation alone.
4. No confidence without calibration.
5. No theory without falsifiable predictions.
6. No live rule without execution validation.
7. No failed idea is silently deleted.
8. No contradiction is ignored.
9. No unknown is converted into certainty.
10. No agent may override the evidence hierarchy.
11. No economic attractiveness substitutes for scientific validity.
12. Every decision must be traceable to active evidence.
13. Every model must expose where it is likely to fail.
14. Every state must be timeframe-aware.
15. LONG and SHORT must be studied as connected structural phases,
    not merely opposite labels.
16. Research, shadow, paper and live permissions remain architecturally separate.
17. The system must be able to say “I do not know.”
```

---

# Part II — AMI as a Layered Operating System

## 5. Top-Level Architecture

```text
                             EPISTEMIC GOVERNOR
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
   RESEARCH OS                    WORLD MODEL                 DECISION SYSTEM
        │                             │                             │
Autonomous Scientist      Multi-Timeframe State Graph    Trade / Portfolio Brain
        │                             │                             │
        └─────────────── KNOWLEDGE GRAPH & MARKET MEMORY ──────────┘
                                      │
                              REALITY INTERFACE
                                      │
                                  MARKETS
```

The Epistemic Governor is a control plane, not a reporting dashboard. It can restrict research conclusions, block promotion, downgrade active knowledge and suspend operational permissions.

## 6. Functional Planes

AMI consists of six functional planes.

### 6.1 Observation Plane

Collects raw external reality:

- trades;
- order books;
- book ticker;
- liquidations;
- mark and index prices;
- spot and perpetual prices;
- funding;
- open interest;
- options;
- cross-exchange feeds;
- on-chain flows;
- macro and economic calendar;
- news and sentiment;
- execution telemetry.

### 6.2 Representation Plane

Transforms raw data into:

- normalized features;
- event objects;
- state objects;
- regime objects;
- structural phases;
- uncertainty estimates;
- multi-timeframe representations.

### 6.3 Scientific Plane

Contains:

- Observation Registry;
- Question Generator;
- Hypothesis Engine;
- Experiment Generator;
- Theory Builder;
- Replication Engine;
- Contradiction Resolver;
- Meta-Research Engine;
- Research Marketplace.

### 6.4 Predictive Plane

Contains:

- State Transition Graph;
- Latent State Engine;
- World Model;
- Scenario Generator;
- Counterfactual Engine;
- Calibration Engine;
- Probability Forecast Engine.

### 6.5 Decision Plane

Contains:

- Entry Intelligence;
- LONG Engine;
- SHORT Engine;
- NO-TRADE Engine;
- Trade Lifecycle Engine;
- Position Management;
- Exit Intelligence;
- Portfolio Brain;
- Risk Engine.

### 6.6 Governance Plane

Contains:

- Epistemic Governor;
- Knowledge permissions;
- Promotion/demotion gates;
- audit logs;
- live safety boundaries;
- model and data lineage;
- constitutional policies.

---

# Part III — Reality Interface and Data Foundation

## 7. Layer 0 — Reality Interface

AMI’s first obligation is to observe reality without silently fabricating continuity.

Primary sources:

```text
Exchanges
├── Spot trades
├── Perpetual trades
├── Order books
├── Book ticker
├── Liquidations
├── Funding
├── OI
├── Mark / index price
└── Exchange metadata

External market context
├── Macro calendar
├── News
├── Options
├── On-chain
├── Stablecoin flows
├── ETF / institutional flows
└── Sentiment
```

## 8. Sensor Design Principles

Every sensor must expose:

```yaml
sensor_id:
source:
symbol:
venue:
data_type:
timestamp_exchange:
timestamp_received:
timestamp_persisted:
latency_ms:
sequence_id:
quality_status:
schema_version:
collector_version:
gap_status:
staleness_seconds:
```

No feature may use a sensor without an explicit freshness contract.

## 9. Data Quality States

```text
HEALTHY
DEGRADED
STALE
GAPPED
SCHEMA_MISMATCH
UNVERIFIED
RECONSTRUCTED
UNAVAILABLE
```

A stale value must never be silently reused as current data.

The previous `vol_state` failure is the canonical example of why AMI needs data-quality states and provenance-aware feature computation.

## 10. Data Lineage

```text
Raw packet
    ↓
Normalized record
    ↓
Feature
    ↓
State
    ↓
Experiment
    ↓
Knowledge Object
    ↓
Decision
```

Every downstream result must be reversible to:

- source table;
- raw rows;
- exact feature code;
- code commit;
- dataset hash;
- experiment configuration;
- execution model version.

## 11. Storage Tiers

```text
HOT
Recent data required for real-time state

WARM
Forward-validation and active research window

COLD
Historical archive and replication

DERIVED
Feature stores and mechanism stores

KNOWLEDGE
Claims, theories, experiments, permissions and genealogy
```

Retention must be data-class specific. High-volume book/trade data should not share retention policy with lightweight OI, funding or state summaries.

## 12. Event-Centred Mechanism Store

Each important market event should have an aligned window:

```text
T-60m
T-30m
T-10m
T-5m
T-2m
T-1m
T0
T+30s
T+1m
T+2m
T+5m
T+10m
T+30m
T+1h
T+2h
T+4h
T+6h
T+12h
T+24h
```

The mechanism store must support both:

- event samples;
- time-matched non-event controls.

This is necessary to distinguish a genuine pre-cascade signature from ordinary market conditions.

---

# Part IV — World Representation and Multi-Timeframe Intelligence

## 13. Layer 1 — World Representation

AMI should not operate directly on isolated numbers. It should build typed state representations.

Example:

```text
GLOBAL
Risk-Off · High Volatility · US Session

BTC
Trend Recovering · Funding Positive · OI Expanding

ETH
Post-Cascade Absorption · Perp Discount · Seller Exhaustion

SOL
Compression · Low Participation · Neutral
```

## 14. State Object

```yaml
state_id:
entity:
timeframe:
state_family:
state_label:
start_time:
age:
confidence:
probability:
supporting_features:
contradicting_features:
parent_states:
expected_transitions:
applicable_actions:
data_quality:
knowledge_objects_used:
```

## 15. Multi-Timeframe State Fusion

Required timeframes:

```text
1m
5m
15m
1h
4h
1D
1W
```

Optional extensions:

```text
tick
10s
30s
30m
2h
8h
3D
1M
```

Each timeframe answers a different question.

| Timeframe | Primary purpose |
|---|---|
| Tick–1m | Microstructure and execution state |
| 5m | Scalp pressure and immediate reaction |
| 15m | Intraday momentum transition |
| 1h | Local swing formation |
| 4h | Primary swing structure |
| 1D | Daily regime, structural direction and carry |
| 1W | Background cycle and macro market phase |

## 16. Timeframe Mapping Requirement

Every signal or state must be tested across horizons.

Example:

```text
SHORT_NOISY
5m: SHORT scalp probability 68%
2h: no durable edge
4h: transition state
1D: LONG recovery probability 63%
```

AMI must not assume a state has one universal direction. A state may be bearish at 15m and bullish at 1D.

## 17. Timeframe Conflict Resolution

```text
1m SHORT
15m SHORT
4h NEUTRAL
1D LONG
```

This is not necessarily a contradiction. It can be a nested structure.

Required outputs:

```yaml
dominant_timeframe:
action_timeframe:
higher_timeframe_constraint:
expected_holding_period:
conflict_score:
alignment_score:
```

## 18. Global Market State

Before symbol-level interpretation, AMI should model:

- risk-on / risk-off;
- volatility regime;
- liquidity regime;
- leverage regime;
- BTC dominance;
- stablecoin flows;
- session;
- weekend effect;
- macro event proximity;
- options expiry;
- news shock;
- market breadth;
- cross-asset correlation.

Output:

```text
GLOBAL_STATE
Expected volatility
Expected persistence
Expected tail risk
Preferred horizon
Allowed research interpretations
```

---

# Part V — Market Physics, Structure and Evolution

## 19. Market Physics Layer

AMI should move from isolated features toward testable market laws.

Candidate law families:

```text
Momentum persistence
Momentum exhaustion
Liquidity attraction
Liquidity vacuum
Inventory mean reversion
Dealer hedging
Volatility expansion
Volatility decay
Cascade propagation
Reflexivity
Order-flow memory
Crowding release
Absorption
Pressure transfer
Information diffusion
Attention rotation
```

A “market law” is not accepted because it sounds plausible. It must generate falsifiable predictions.

## 20. Structure Engine

Canonical structural phases:

```text
ACCUMULATION
COMPRESSION
EARLY_EXPANSION
EXPANSION
MATURE_TREND
DISTRIBUTION
BREAKDOWN
CASCADE
EXHAUSTION
ABSORPTION
RECLAIM
RECOVERY
REACCUMULATION
REVERSAL
RANGE
DISLOCATION
```

These phases form a graph, not a rigid linear sequence.

## 21. Structural Cycle

```text
Build Energy
    ↓
Compression
    ↓
Release
    ↓
Expansion
    ↓
Trend
    ↓
Distribution
    ↓
Breakdown / Cascade
    ↓
Exhaustion
    ↓
Absorption
    ↓
Recovery
    ↓
Build Again
```

## 22. Energy and Entropy

### Energy representation

Potential components:

- leverage buildup;
- range compression;
- resting liquidity;
- unresolved imbalance;
- crowding;
- volatility potential;
- funding pressure.

### Entropy representation

Measures whether the market is:

```text
LOW ENTROPY
Structured, persistent, predictable

HIGH ENTROPY
Noisy, fragmented, unstable
```

The same alpha may require different management under different entropy states.

## 23. Market Memory

Markets may retain event influence over time.

Each event should have a decay function:

```yaml
event_type:
initial_impact:
half_life:
direction:
affected_states:
memory_strength:
invalidation_condition:
```

Examples:

- liquidation cascade memory;
- prior day high/low memory;
- large wall memory;
- funding extreme memory;
- macro shock memory.

## 24. Agent and Ecology Model

AMI should represent inferred participant classes:

```text
Retail
Whales
Passive makers
Market makers
Arbitrageurs
Options dealers
Liquidation engine
Trend followers
Mean-reversion participants
Institutional flows
```

These are probabilistic inferred roles, not known identities.

Example ecology:

```text
Retail panic selling
        ↓
Maker absorption
        ↓
Inventory accumulation
        ↓
Hedging / price recovery
        ↓
Short covering
        ↓
Expansion
```

---

# Part VI — Unified LONG/SHORT Structural Intelligence

## 25. Foundational Principle

LONG and SHORT are not independent strategies.

They are possible actions produced by the same evolving structure.

```text
Distribution
    ↓
Breakdown
    ↓
SELL cascade
    ↓
SHORT pressure
    ↓
Seller exhaustion
    ↓
Absorption
    ↓
Reclaim
    ↓
LONG opportunity
```

Similarly:

```text
Accumulation
    ↓
Expansion
    ↓
BUY euphoria
    ↓
Liquidity exhaustion
    ↓
Failed breakout
    ↓
Distribution
    ↓
SHORT opportunity
```

## 26. Dual-Direction State Matrix

Every state must produce both LONG and SHORT estimates.

```yaml
state:
p_long:
p_short:
p_no_trade:
best_horizon_long:
best_horizon_short:
expected_mfe_long:
expected_mfe_short:
expected_mae_long:
expected_mae_short:
confidence:
failure_modes:
```

## 27. Structural Asymmetry

LONG and SHORT should not be assumed symmetric.

Potential asymmetry:

```text
SELL panic
Fast collapse
Slow recovery
LONG opportunity

BUY euphoria
Slow grind
Fast collapse
SHORT opportunity
```

AMI must measure:

- speed asymmetry;
- volatility asymmetry;
- depth asymmetry;
- liquidation asymmetry;
- recovery asymmetry;
- execution asymmetry.

## 28. LONG Research Engine

### 28.1 Primary objective

Discover how LONG states emerge, survive, mature and fail.

### 28.2 Core questions

```text
Why does LONG begin?
Why does LONG fail?
When is LONG a scalp?
When is LONG a 4h swing?
When is LONG a 1D structure?
What is the earliest reliable LONG state?
What kills a LONG state?
Which SHORT failures generate LONG?
Which LONG states later generate SHORT?
```

### 28.3 LONG genesis families

#### A. Seller-exhaustion LONG

Conditions to study:

- new SELL liquidations produce less price impact;
- no new low despite forced sell flow;
- taker sell aggression decays;
- CVD remains negative while price stabilizes;
- bid depth stops pulling;
- absorption increases;
- BTC stops falling;
- basis normalizes;
- funding remains negative but price no longer declines.

#### B. Failed-breakdown LONG

- breakdown level reclaimed;
- retest holds;
- volume confirmation;
- OFI flips positive;
- BTC synchronization recovers.

#### C. Compression-release LONG

- low background volatility;
- local volatility expansion;
- positive flow transition;
- OI expansion with price acceptance;
- no liquidation dependency required.

#### D. Cross-timeframe LONG

- 5m bearish exhaustion;
- 1h recovery;
- 4h swing reversal;
- 1D structural support;
- 1W non-hostile background.

#### E. Independent LONG mechanisms

LONG must also be discovered outside SHORT failure:

- order-book absorption without cascade;
- funding/OI dislocation;
- cross-exchange lead;
- spot-led demand;
- stablecoin or institutional flow;
- options/dealer hedging;
- range breakout with acceptance.

## 29. SHORT Research Engine

SHORT should be modelled through equivalent but not mechanically mirrored families:

- buyer exhaustion;
- failed breakout;
- liquidity withdrawal;
- maker retreat;
- OI expansion without price acceptance;
- positive funding crowding;
- cross-exchange weakness;
- distribution;
- post-euphoria collapse.

## 30. SHORT-to-LONG Transition Research Protocol

Anchor timestamps:

```text
T0 cascade
T+5m
T+15m
T+30m
T+60m
T+120m
time of SHORT MFE
time of momentum stall
time of failed breakdown
time of reclaim
time of absorption
```

Test LONG outcomes:

```text
30m
1h
2h
4h
6h
12h
24h
```

For every transition candidate, measure:

- MFE;
- MAE;
- time to MFE;
- probability of new low;
- probability of reclaim;
- execution feasibility;
- adverse selection;
- regime dependence;
- no-overlap portfolio effect.

## 31. Complete Swing Structure Objective

AMI should eventually represent a complete swing:

```text
SHORT setup
    ↓
SHORT entry
    ↓
SHORT healthy
    ↓
SHORT weakening
    ↓
SHORT exit / profit lock
    ↓
transition uncertainty
    ↓
LONG confirmation
    ↓
LONG entry
    ↓
LONG expansion
    ↓
LONG exhaustion
    ↓
next transition
```

The system should not force a reversal trade. `NO_TRADE` is a valid transition state.

---

# Part VII — Microstructure and Mechanism Intelligence

## 32. Market Microstructure Layer

Core features:

```text
Trade arrival rate
Order arrival rate
Cancellation rate
Aggressive volume
Passive volume
Queue dynamics
Tick speed
Spread dynamics
Book imbalance
Book slope
Depth
Liquidity concentration
Price impact
Hidden-liquidity proxies
```

## 33. Order Book Evolution

The book is a video, not a photograph.

```text
BOOK_APPEARS
BOOK_THICKENS
BOOK_PULLS
BOOK_REFILLS
BOOK_MIGRATES
BOOK_FLIPS
BOOK_COLLAPSES
BOOK_RECOVERS
```

Measurements:

- pull speed;
- refill speed;
- wall migration;
- depth recovery;
- liquidity holes;
- price impact per unit flow;
- queue survival;
- absorption persistence.

## 34. Maker Behaviour and Inventory Pressure

Direct maker identity is unavailable. AMI should infer behaviour from observable consequences.

```text
Large bid survives sell aggression
→ probable absorption

Large bid disappears before impact
→ probable retreat

Repeated passive filling
→ possible inventory accumulation

Inventory accumulation + hedging
→ possible delayed directional pressure
```

Every maker claim must retain a low causal confidence unless stronger evidence is available.

## 35. Flow State

Components:

- taker aggression;
- OFI;
- CVD;
- retail-flow proxy;
- large-trade share;
- participation breadth;
- flow acceleration;
- flow exhaustion;
- cross-asset flow.

## 36. Funding and OI Dynamics

Funding is a process, not one number.

Required dynamics:

```text
Funding level
Funding slope
Funding velocity
Funding acceleration
Funding divergence

OI level
OI slope
OI velocity
OI acceleration
OI collapse
OI expansion
Price/OI divergence
Funding/OI interaction
```

## 37. Cross-Exchange Network

Model:

- Binance;
- Bybit;
- OKX;
- Hyperliquid;
- CME where available;
- spot/perpetual dislocation;
- information lead-lag;
- liquidation synchronization;
- fragmented vs universal stress.

Output:

```yaml
network_state:
leader_venue:
lagging_venues:
synchronization:
dispersion:
stress_propagation:
confidence:
```

## 38. Liquidity Stress and Cascade Evolution

Stress sources:

- stop density;
- liquidation density;
- funding stress;
- OI stress;
- basis stress;
- options gamma;
- book holes;
- cross-venue imbalance.

Cascade phases:

```text
PRE_CASCADE
EARLY_CASCADE
ACCELERATION
PEAK
EXHAUSTION
RECOVERY
SECONDARY_ECHO
FAILED_CASCADE
```

---

# Part VIII — Prediction, Decision and Trade Lifecycle

## 39. Prediction Engine

AMI produces distributions, not one deterministic call.

```text
Continuation 52%
Reversal 27%
Range 14%
Panic extension 7%
```

Required forecast horizons:

```text
5m
15m
30m
1h
4h
1D
```

Each forecast must include:

- calibration;
- confidence interval;
- dominant evidence;
- counter-evidence;
- regime;
- action relevance.

## 40. Decision Engine

Possible outputs:

```text
LONG
SHORT
NO_TRADE
WAIT_FOR_CONFIRMATION
REDUCE
SCALE
LOCK_PROFIT
EXIT
REVERSE
SUSPEND
```

The Decision Engine may not use Knowledge Objects beyond their permitted use.

## 41. Entry Intelligence

Entry output should contain:

```yaml
direction:
entry_confidence:
expected_duration:
expected_mfe:
expected_mae:
expected_volatility:
expected_rr:
recommended_entry_type:
invalidating_state:
no_trade_probability:
```

## 42. Trade Lifecycle Engine

Entry alpha, hold alpha and exit alpha are separate.

```text
ENTRY ALPHA ≠ HOLD ALPHA ≠ EXIT ALPHA
```

Trade states:

```text
OPEN
HEALTHY
ACCELERATING
STALLING
WEAKENING
EXHAUSTED
RECOVERING
LOCKED
REVERSING
INVALIDATED
CLOSED
```

## 43. Post-Entry Feature Store

Capture snapshots at:

```text
Entry
+20 bps
+40 bps
+50 bps
+75 bps
+100 bps
3m
5m
10m
15m
30m
60m
90m
120m
MFE time
MAE time
Exit
```

Features:

- book pull/refill;
- absorption;
- taker flow;
- OFI/CVD;
- spread;
- volatility;
- BTC synchronization;
- funding;
- OI;
- basis;
- liquidation pressure;
- structural phase;
- state transition probability.

## 44. MFE State Classifier

For each profit milestone:

```text
Trade reached +50 bps.

Next:
A. Continues to +100/+150
B. Returns to breakeven
C. Finishes negative
D. Time-exits positive
E. Produces reversal opportunity
```

The classifier should distinguish continuation from giveback using only information available at the milestone.

## 45. Management Action Model

```text
HOLD
SCALE
REDUCE
LOCK
EXIT
REVERSE
NO_ACTION
```

Every proposed action must be evaluated against:

- baseline expectancy;
- drawdown;
- tail;
- execution feasibility;
- forward performance;
- psychological/operational complexity;
- state uncertainty.

## 46. Exit Intelligence

Exit families:

- momentum exit;
- liquidity exit;
- book exit;
- volatility exit;
- structure exit;
- synchronization exit;
- funding/OI exit;
- risk exit;
- time exit.

An exit reason must be explicit. “TP hit” is not a mechanism explanation.

## 47. Current Management Research Lessons

Current S34 research provides useful priors, not universal truths:

- fast bar-based trailing can destroy a slower swing edge;
- tight stops can be incompatible with dip-holding mechanisms;
- limit orders may lose more through missed fills than they gain in price;
- fixed 6h hold has remained difficult to beat in the tested universe;
- wide profit lock `200/100` is a better observer candidate than `100/50`;
- scale-in near `-100 bps` within two hours is an in-sample management candidate requiring forward observation;
- selection appears more valuable than micro-timing in the current cascade family.

These findings must remain scoped to their evidence and execution assumptions.

---

# Part IX — Portfolio Brain and Risk

## 48. Portfolio State

```yaml
open_positions:
gross_exposure:
net_exposure:
directional_concentration:
state_correlation:
alpha_family_correlation:
liquidity_risk:
tail_budget:
daily_loss:
available_risk:
regime:
epistemic_risk:
```

## 49. Portfolio Actions

```text
ADMIT
REJECT
DEFER
PRIORITIZE
NET
HEDGE
REDUCE
THROTTLE
PAUSE
```

## 50. Correlation Beyond Returns

AMI must measure:

- shared trigger;
- shared data dependency;
- shared regime dependency;
- shared failure mode;
- shared execution risk;
- shared theory.

Two routes with low return correlation may still be epistemically the same alpha.

## 51. Economic Risk vs Epistemic Risk

```text
Economic Risk
How much can be lost if the trade fails?

Epistemic Risk
How likely is the underlying belief to be wrong or misapplied?
```

Position admission should account for both.

## 52. Tail Budget and Operational Safety

Operational safety rules should remain outside research optimization.

Examples:

- leverage guardrails;
- position-size caps;
- atomic stop protection;
- daily throttle;
- maximum concurrent slots;
- data-health kill switches;
- execution mismatch suspension.

No research agent may silently alter these.

---

# Part X — Research Operating System and Autonomous Scientist

## 53. Scientific Discovery Loop

```text
Observe
    ↓
Detect anomaly
    ↓
Generate question
    ↓
Generate competing hypotheses
    ↓
Rank experiments
    ↓
Freeze protocol
    ↓
Execute
    ↓
Evaluate
    ↓
Replicate
    ↓
Update knowledge
    ↓
Generate better question
```

## 54. Observation Engine

Sources of research questions:

- unexplained residuals;
- large prediction errors;
- repeated giveback patterns;
- state transitions;
- contradictions;
- regime degradation;
- unknown clusters;
- execution anomalies;
- new data feeds;
- failure archive revisitation.

## 55. Question Generator

A research question object:

```yaml
question_id:
question:
origin_observation:
scientific_value:
economic_value:
risk_reduction_value:
novelty:
data_readiness:
required_sample:
estimated_cost:
falsifiability:
priority:
dependencies:
```

## 56. Hypothesis Generator

For every question, generate:

- primary hypothesis;
- null hypothesis;
- alternative mechanisms;
- confounder hypothesis;
- data-quality explanation;
- execution explanation;
- regime explanation.

## 57. Experiment Generator

The system must pre-register:

```text
Population
Target
Features
Threshold-selection method
Untouched data
Chronological split
Negative controls
Effect-size requirement
Minimum sample
Multiple-testing correction
Execution model
Decision criteria
Falsification rule
```

## 58. Research Marketplace

Questions compete for limited research resources.

Priority dimensions:

```text
Information gain
Economic value
Risk reduction
Contradiction resolution
Generalization
Infrastructure value
Data-acquisition value
Novelty
Falsifiability
Cost
Sample requirement
Overfit risk
```

Recommended research allocation:

```text
60% exploitation research
25% exploration research
15% curiosity research
```

## 59. Curiosity Engine

Curiosity research explores high-novelty questions that are not immediately monetizable.

Examples:

- unexplained state cluster;
- unexpected cross-timeframe reversal;
- feature family never tested jointly;
- market behaviour that violates an accepted theory.

## 60. Theory Builder

A theory is a structured explanatory object.

```yaml
theory_id:
statement:
mechanism:
scope:
predictions:
supporting_evidence:
contradictions:
alternative_theories:
status:
revision_history:
```

A theory must produce new predictions before it is considered useful.

## 61. Meta-Research Engine

AMI must research its own scientific process:

- which validation schemes best predict forward survival;
- which feature families overfit most often;
- which agents produce false discoveries;
- which sample sizes are misleading;
- which metrics predict operational failure;
- which experiments consume resources without information gain.

## 62. Failure Archive

Failed ideas are retained with:

```yaml
idea:
reason_rejected:
data_period:
regimes_tested:
failure_type:
retry_condition:
related_theories:
```

Failure types:

```text
NO_EDGE
OVERFIT
LOOKAHEAD
EXECUTION_FAILURE
REGIME_LIMITED
INSUFFICIENT_SAMPLE
DATA_UNAVAILABLE
DUPLICATE_HYPOTHESIS
WRONG_TIMEFRAME
WRONG_DIRECTION
```

---

# Part XI — Machine Learning, World Model and Digital Twin

## 63. Role of Machine Learning

ML is not granted automatic authority.

Preferred initial roles:

- quality scoring;
- probability calibration;
- nonlinear interaction discovery;
- latent-state discovery;
- anomaly detection;
- similarity retrieval;
- state-transition modelling;
- forecast distributions;
- research prioritization.

Black-box direct trade control requires a much higher evidence standard.

## 64. Latent State Engine

Candidate methods:

- clustering;
- hidden Markov models;
- switching state-space models;
- change-point detection;
- representation learning;
- contrastive learning;
- sequence models.

Output:

```yaml
latent_state_id:
observed_signature:
transition_profile:
timeframe:
regime:
economic_interpretation:
stability:
confidence:
```

Latent states should be discovered before being narratively named.

## 65. State Transition Graph

```text
STATE_A
├── 54% STATE_B
├── 26% STATE_C
├── 12% STATE_D
└── 8% UNKNOWN
```

Graph dimensions:

- probability;
- expected duration;
- transition trigger;
- timeframe;
- regime;
- direction;
- uncertainty.

## 66. World Model

The World Model simulates possible market evolution.

Inputs:

- current multi-timeframe states;
- structural phase;
- order-book dynamics;
- flow;
- OI/funding;
- cross-exchange state;
- macro context;
- portfolio state.

Outputs:

```text
Scenario A: continuation
Scenario B: reversal
Scenario C: compression
Scenario D: panic
```

## 67. Digital Twin

```text
Real market
    ↓
AMI state estimate
    ↓
Simulated futures
    ↓
Observed future
    ↓
Model error
    ↓
Calibration and revision
```

The Digital Twin is useful only if its forecast error and uncertainty are continuously measured.

## 68. Counterfactual Engine

Questions:

```text
What if no trade was opened?
What if direction was reversed?
What if entry was delayed?
What if size was reduced?
What if a scale-in occurred?
What if BTC had not recovered?
What if funding had remained unchanged?
```

Counterfactual conclusions must distinguish simulation from observed evidence.

## 69. Market Memory Retrieval

For the current state:

```text
Find historically similar states
Measure transition outcomes
Adjust for regime and execution
Report similarity uncertainty
```

Nearest-neighbour retrieval must avoid lookahead and duplicate-event leakage.

## 70. Explainability

Every model output should provide:

- key evidence;
- counter-evidence;
- closest historical analogues;
- state transition rationale;
- uncertainty;
- known failure modes;
- applicable Knowledge Objects.

---

# Part XII — Epistemic Core

The following section is incorporated as the authoritative starting specification for AMI’s Epistemic Governor.


# Epistemic Core — How AMI Knows, Doubts and Changes Its Mind

AMI yalnızca piyasa hakkında sonuç üreten bir sistem olmayacaktır.

Aynı zamanda:

* sonuçların hangi veriden üretildiğini,
* hangi varsayımlara dayandığını,
* ne kadar güçlü kanıtla desteklendiğini,
* hangi piyasa koşullarında geçerli olduğunu,
* hangi bulgularla çürütülebileceğini,
* ne zaman yeniden test edilmesi gerektiğini,
* hangi uygulamalarda kullanılmasına izin verildiğini

yöneten bir bilimsel bilgi sistemi olacaktır.

Bu katmanın merkezi bileşeni:

# Epistemic Governor

olacaktır.

Epistemic Governor, AMI içindeki diğer sistemlerin üzerinde çalışan epistemik kontrol düzlemidir.

```text
                    EPISTEMIC GOVERNOR
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
  World Model        Autonomous Scientist   Decision Engines
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                 Knowledge Permissions
                            │
             Research / Shadow / Operational
```

World Model bir state hakkında yüksek güven iddia edemez, Autonomous Scientist yeterli kanıt olmadan teori kabul edemez ve Trading Engine epistemik statüsü yetersiz bir bilgiyi operasyonel kurala dönüştüremez.

---

# 1. Three Learning Loops

AMI içinde üç farklı fakat birbirine bağlı öğrenme döngüsü bulunacaktır.

## 1.1 Market Learning Loop

```text
Market
   ↓
Observation
   ↓
Question
   ↓
Hypothesis
   ↓
Experiment
   ↓
Evidence
   ↓
Market Knowledge
```

Bu döngü piyasa davranışlarını araştırır.

## 1.2 Meta-Research Loop

```text
Research Process
       ↓
Method Performance
       ↓
Validation Quality
       ↓
Failure Analysis
       ↓
Improved Research Method
```

Bu döngü, sistemin araştırma yapma biçimini geliştirir.

## 1.3 Epistemic Governance Loop

```text
Existing Knowledge
        ↓
New Evidence
        ↓
Contradiction Detection
        ↓
Confidence Update
        ↓
Revision / Restriction / Expiration
```

Bu döngü, mevcut bilgilerin geçerliliğini sürekli denetler.

Üçüncü döngünün görevi yalnızca yeni bilgi eklemek değildir.

Aynı zamanda:

```text
What is no longer true?
What was never sufficiently proven?
What is valid only in a narrow regime?
What must no longer influence decisions?
```

sorularını yanıtlamaktır.

---

# 2. Epistemic Authority

Epistemic Governor’ın sistem üzerinde gerçek yetkisi bulunmalıdır.

Yalnızca öneri vermemeli; bazı eylemleri engelleyebilmelidir.

Örneğin:

```text
Claim Status:
PRELIMINARY

Requested Action:
Promote to live

Governor Decision:
DENIED
```

veya:

```text
Claim Status:
FORWARD_VALIDATING

Permitted Uses:
Research
Shadow
Observer

Forbidden Uses:
Real orders
Sizing decisions
Leverage changes
```

Her Knowledge Object için izin verilen kullanım alanları tanımlanmalıdır:

```text
RESEARCH_ONLY
BACKTEST_ALLOWED
SHADOW_ALLOWED
OBSERVER_ALLOWED
PAPER_ALLOWED
OPERATIONAL_CANDIDATE
LIVE_ALLOWED
SIZING_ALLOWED
PORTFOLIO_ALLOWED
```

Böylece araştırma bulgusu ile operasyonel karar arasındaki sınır mimari olarak korunur.

---

# 3. Knowledge Object Model

AMI içindeki her bilimsel iddia yapılandırılmış bir Knowledge Object olmalıdır.

```text
Knowledge ID:
Claim:
Claim Type:
Status:
Direction:
Mechanism:
Target Variable:
Effect Size:
Uncertainty Interval:

Evidence Count:
Independent Replications:
Chronological Holdouts:
Forward Evidence:
Negative Controls:
Contradictions:

Applicable Symbols:
Applicable Exchanges:
Applicable Regimes:
Applicable Sessions:
Applicable Timeframes:
Applicable Volatility States:

Execution Assumptions:
Data Assumptions:
Latency Assumptions:
Fee Assumptions:
Liquidity Assumptions:

Confidence:
Calibration Class:
Generalization Score:
Novelty Score:
Economic Relevance:
Expiration Risk:

Created At:
Last Tested:
Last Verified:
Next Review Date:

Falsification Conditions:
Known Alternative Explanations:
Parent Observation:
Parent Hypothesis:
Derived Predictions:
Child Knowledge Objects:
Downstream Applications:

Permitted Uses:
Forbidden Uses:
Owner:
Version:
```

Knowledge Object yalnızca sonuç değil, sonucun bütün bilimsel bağlamını taşımalıdır.

---

# 4. Claim Types

Her iddia aynı türde değildir.

AMI bunları ayırmalıdır:

```text
DESCRIPTIVE
Veride ne gözlendi?

PREDICTIVE
Hangi koşul gelecekteki sonucu tahmin ediyor?

MECHANISTIC
Bu ilişki neden oluşuyor?

CAUSAL
Bir değişken diğerini gerçekten etkiliyor mu?

OPERATIONAL
Bu bilgi bir karar kuralına dönüştürülebilir mi?

META_RESEARCH
Hangi araştırma yöntemi daha güvenilir?
```

Örneğin:

> SELL cascade sonrası fiyat sıklıkla toparlanıyor.

descriptive olabilir.

> Book absorption mevcutsa toparlanma olasılığı artıyor.

predictive olabilir.

> Makers forced sell flow’u absorbe ettiği için fiyat etkisi azalıyor.

mechanistic bir iddiadır.

Bu üçü aynı kanıt standardıyla kabul edilmemelidir.

---

# 5. Evidence Hierarchy

AMI açık bir kanıt hiyerarşisine sahip olmalıdır.

```text
Level 0 — Anecdotal Observation
Level 1 — In-Sample Pattern
Level 2 — Chronological Validation
Level 3 — Untouched Holdout
Level 4 — Independent Replication
Level 5 — Forward Shadow Validation
Level 6 — Controlled Paper Validation
Level 7 — Small-Scale Operational Validation
Level 8 — Multi-Regime Replication
Level 9 — Cross-Symbol / Cross-Market Generalization
```

Her iddia için:

```text
Current Evidence Level:
Required Evidence Level:
Evidence Gap:
```

alanları tutulmalıdır.

Örneğin bir bulgunun ekonomik olarak çok güçlü görünmesi, onun kanıt seviyesini yükseltmemelidir.

---

# 6. Evidence Quality Is Not Evidence Quantity

On benzer backtest, on bağımsız kanıt değildir.

AMI evidence dependency modeline sahip olmalıdır.

Aşağıdaki testler birbirinden bağımsız sayılmamalıdır:

```text
Same dataset
Same feature family
Same event universe
Same target
Slightly different threshold
Slightly different TP
```

Bunlar tek bir evidence family olarak gruplanmalıdır.

```text
Evidence Family ID:
Shared Dataset:
Shared Assumptions:
Shared Failure Modes:
Independence Score:
```

Bu, yüzlerce benzer testin sahte güven oluşturmasını önler.

---

# 7. Data Lineage and Provenance

Her iddia doğrudan ham veriye kadar izlenebilmelidir.

```text
Knowledge Object
      ↓
Experiment
      ↓
Feature Set
      ↓
Dataset Version
      ↓
Source Tables
      ↓
Raw Market Events
```

Tutulması gereken alanlar:

```text
Dataset Hash
Code Commit
Feature Version
Experiment Config
Random Seed
Query Definition
Data Time Range
Excluded Periods
Missing Data Policy
Execution Model Version
```

Aynı deney gelecekte aynı veri ve kodla tekrar çalıştırılabilmelidir.

Epistemik güven, yalnızca istatistiksel sonuçtan değil, yeniden üretilebilirlikten de gelmelidir.

---

# 8. Assumption Registry

Her Knowledge Object açık varsayımlar taşımalıdır.

Örneğin:

```text
Assumption:
BookTicker fill is representative of executable fill.

Assumption:
Binance flow adequately represents total market pressure.

Assumption:
No major structural market change occurred during the sample.

Assumption:
Latency remains below 500 ms.

Assumption:
Fees and slippage remain within modeled range.
```

Bir varsayım bozulduğunda, ona bağlı bütün bilgiler otomatik olarak yeniden değerlendirilmelidir.

```text
Assumption invalidated
        ↓
Dependent knowledge found
        ↓
Confidence reduced
        ↓
Operational permissions suspended
        ↓
Retest scheduled
```

---

# 9. Confidence Must Be Decomposed

Tek bir `Confidence: 82%` alanı yetersiz olabilir.

Güven farklı bileşenlere ayrılmalıdır:

```text
Statistical Confidence
Replication Confidence
Forward Confidence
Mechanism Confidence
Execution Confidence
Regime Confidence
Generalization Confidence
Data Quality Confidence
```

Örneğin:

```text
Statistical Confidence: High
Mechanism Confidence: Low
Forward Confidence: Medium
Execution Confidence: High
Generalization Confidence: Unknown
```

Bu, sistemin hangi konuda emin olduğunu daha doğru gösterir.

---

# 10. Confidence Calibration

AMI yalnızca güven puanı üretmemeli; güven puanlarının doğruluğunu da takip etmelidir.

Örneğin sistem 100 farklı hipoteze ortalama %70 güven verdiyse, bunların yaklaşık %70’inin daha sonraki doğrulamalarda hayatta kalması beklenir.

Takip edilecek metrikler:

```text
Brier Score
Calibration Error
Reliability Curve
Overconfidence Rate
Underconfidence Rate
Confidence by Evidence Level
Confidence by Research Agent
Confidence by Feature Family
```

AMI’nin en önemli yeteneklerinden biri doğru cevap vermek kadar, **ne kadar emin olması gerektiğini doğru tahmin etmek** olacaktır.

---

# 11. Uncertainty Budget

Her karar sınırsız belirsizlik taşıyamaz.

Bir operational candidate için maksimum belirsizlik bütçesi tanımlanmalıdır.

```text
Total Uncertainty =
Data Uncertainty
+ Model Uncertainty
+ Regime Uncertainty
+ Execution Uncertainty
+ Structural Uncertainty
```

Belirsizlik bütçesi aşılırsa:

```text
LIVE promotion denied
Size capped
Shadow only
Additional evidence required
```

kararlarından biri uygulanmalıdır.

---

# 12. Temporal Decay and Knowledge Expiration

Piyasa bilgisi kalıcı kabul edilmemelidir.

Her Knowledge Object için decay profile tutulmalıdır.

```text
Decay Type:
Slow
Moderate
Fast
Event-Driven
Regime-Dependent

Half-Life:
30 days
90 days
180 days
Unknown
```

Örneğin mikrostructure bilgileri hızlı, daha temel piyasa mekanizmaları daha yavaş eskime gösterebilir.

Expiration yalnızca zamana bağlı olmamalıdır.

Aşağıdaki olaylar da yeniden doğrulama tetiklemelidir:

```text
Exchange rule change
Fee change
Tick-size change
Liquidity migration
New dominant venue
Volatility regime shift
Model performance drift
Execution degradation
Major market structure change
```

---

# 13. Belief Revision Engine

Yeni kanıt geldiğinde sistemin dört temel seçeneği bulunmalıdır:

```text
REINFORCE
RESTRICT
REVISE
REJECT
```

## Reinforce

Yeni kanıt mevcut iddiayı destekler.

## Restrict

İddia yalnızca daha dar koşullarda geçerlidir.

```text
Old:
Works on ETH SELL cascades.

New:
Works only during US session and positive BTC synchronization.
```

## Revise

Mekanizma veya yön yeniden tanımlanır.

```text
Old:
Funding level drives continuation.

New:
Funding acceleration matters, not funding level.
```

## Reject

İddia mevcut biçimiyle terk edilir.

Eski versiyon silinmez. Tarihsel olarak saklanır.

---

# 14. Contradiction Resolution Protocol

İki güvenilir bulgu çeliştiğinde sistem birini otomatik olarak silmemelidir.

İlk soru şu olmalıdır:

> Under what conditions can both findings be true?

Çelişki çözüm süreci:

```text
Detect contradiction
        ↓
Compare populations
        ↓
Compare regimes
        ↓
Compare timeframes
        ↓
Compare execution models
        ↓
Compare feature definitions
        ↓
Search for hidden moderator
        ↓
Unify or separate claims
```

Örnek:

```text
Claim A:
Funding improves SHORT outcomes.

Claim B:
Funding has no effect.

Resolution:
Funding matters only when OI is expanding and BTC synchronization is positive.
```

Çelişkiler hata değil, yeni state veya moderating variable keşfinin kaynağı olarak görülmelidir.

---

# 15. Adversarial Epistemics

AMI kendi hipotezlerini yalnızca doğrulamaya çalışmamalıdır.

Her güçlü hipoteze karşı otomatik bir red-team süreci oluşturulmalıdır.

```text
Proponent Agent
Best argument supporting the hypothesis

Skeptic Agent
Best argument against the hypothesis

Alternative Agent
Competing mechanism

Leakage Agent
Search for lookahead and contamination

Execution Agent
Search for unrealistic fill assumptions

Regime Agent
Search for period dependency

Statistician Agent
Search for multiple-testing and low-power problems
```

Bir iddia epistemik incelemeden geçmeden yüksek statüye ulaşamamalıdır.

---

# 16. Falsification Protocol

Her hipotez testten önce aşağıdaki alanları doldurmalıdır:

```text
What result supports the hypothesis?
What result weakens the hypothesis?
What result falsifies the hypothesis?
What metrics are frozen before testing?
What data remains untouched?
What effect size is economically meaningful?
What minimum sample is required?
What alternative explanation exists?
What negative control will be used?
What result would trigger replication?
```

Test tamamlandıktan sonra başarı kriterleri değiştirilememelidir.

Post-hoc bulunan ilginç ilişkiler yeni hipotez olarak kaydedilmeli ve yeni veri üzerinde ayrıca test edilmelidir.

---

# 17. Prediction Registry

Her mekanizma teorisi, deneyden önce tahmin üretmelidir.

```text
Theory ID:
Prediction ID:
Expected Direction:
Expected Effect Size:
Expected Time Horizon:
Expected Regime:
Failure Condition:
Test Dataset:
Outcome:
```

Başarısız tahminler gizlenmemelidir.

Bir teorinin değeri yalnızca açıklama kalitesinden değil, gelecekte doğru tahmin üretme yeteneğinden ölçülmelidir.

---

# 18. Theory Ledger

Her teori için bir bilanço tutulmalıdır.

```text
Correct Predictions:
Incorrect Predictions:
Unresolved Predictions:
Supported Mechanisms:
Contradicted Mechanisms:
Applicable Regimes:
Failed Regimes:
Theory Age:
Last Revision:
```

Teoriler sürekli yeni açıklamalar ekleyerek kurtarılmamalıdır.

Çok fazla başarısız tahmin üreten teori:

```text
WEAKENED
RESTRICTED
REPLACED
DEPRECATED
```

statülerinden birine düşürülmelidir.

---

# 19. Research Genealogy

Her alpha ve her teori tam bir soy ağacına sahip olmalıdır.

```text
Observation
    ↓
Question
    ↓
Hypothesis Family
    ↓
Experiment Series
    ↓
Contradictions
    ↓
Mechanism Theory
    ↓
Predictive Rule
    ↓
Operational Candidate
    ↓
Shadow Validation
    ↓
Live Application
```

Bir operasyonel strateji için sistem şu soruya cevap verebilmelidir:

> Why does this rule exist, which evidence supports it, which assumptions does it depend on, and what would cause it to be disabled?

---

# 20. Unknown Engine

Unknown Engine üç temel sınıfta çalışmalıdır.

## Known Unknowns

Sorusu belirli fakat cevabı eksik konular.

```text
Does the edge survive weekends?
Does it replicate on SOL?
Does it remain after fee changes?
```

## Structural Blind Spots

Sistemin ölçemediği alanlar.

```text
Missing OI history
No reliable cross-exchange order book
Unobserved maker identity
Incomplete options data
```

## Potential Unknowns

Sistemin mevcut modellerinin açıklayamadığı anomaliler.

Kaynaklar:

```text
Large residual errors
Persistent miscalibration
Unclassified event clusters
Sudden theory decay
Contradictory valid signals
New latent states
Unusual transition paths
```

Potential Unknown’lar otomatik olarak araştırma sorularına dönüştürülmelidir.

---

# 21. Knowledge Compatibility Graph

Her bilgi diğer bilgilerle uyumlu olmayabilir.

Örneğin:

```text
Claim A requires:
High volatility regime

Claim B requires:
Low volatility regime
```

Bu iki bilgi aynı anda aynı karar içinde kullanılamaz.

Knowledge Graph yalnızca ilişkileri değil, uyumluluk kurallarını da taşımalıdır:

```text
SUPPORTS
CONTRADICTS
DEPENDS_ON
REQUIRES
INVALIDATES
RESTRICTS
SUPERSEDES
CAN_COEXIST_WITH
CANNOT_COEXIST_WITH
```

Decision Engine, birbirini dışlayan knowledge object’leri fark etmeden birleştirememelidir.

---

# 22. Decision Traceability

AMI tarafından üretilen her karar açıklanabilir olmalıdır.

```text
Decision:
LOCK_PROFIT

Active Evidence:
BTC synchronization lost
Sell aggression declined
Bid absorption increased
Volatility decayed

Knowledge Objects Used:
K-1042
K-1177
K-1189

Confidence:
74%

Uncertainty:
Moderate

Rejected Alternatives:
HOLD_SHORT
FLIP_LONG
```

Sistem yalnızca karar vermemeli; kararın epistemik kaynağını göstermelidir.

---

# 23. Epistemic Promotion Gates

Knowledge Object’lerin statü geçişleri açık kurallara bağlanmalıdır.

Örnek:

```text
PRELIMINARY
    ↓
Minimum sample met
    ↓
REPLICATED
    ↓
Untouched holdout survived
    ↓
HOLDOUT_VALIDATED
    ↓
Forward evidence accumulated
    ↓
FORWARD_VALIDATING
    ↓
Execution assumptions verified
    ↓
OPERATIONAL_CANDIDATE
```

Hiçbir agent manuel veya anlatısal ikna yoluyla bu kapıları atlayamamalıdır.

Promotion kararı kodlanmış kriterlere dayanmalıdır.

---

# 24. Automatic Demotion

Promotion kadar demotion da otomatik olmalıdır.

Tetikleyiciler:

```text
Forward degradation
Confidence calibration failure
Regime shift
Execution mismatch
Repeated contradiction
Feature drift
Sample contamination discovery
Data-quality failure
Replication failure
```

Sonuç:

```text
LIVE_ALLOWED
      ↓
OPERATIONAL_CANDIDATE
      ↓
SHADOW_ONLY
      ↓
RESEARCH_ONLY
```

Sistem bir bilgiye bağlanmamalı; kanıt zayıfladığında kullanım yetkisini geri almalıdır.

---

# 25. Research Marketplace Improvements

Research Marketplace yalnızca tek bir priority score kullanmamalıdır.

Bir araştırmanın birden fazla değeri olabilir:

```text
Scientific Value
Economic Value
Risk Reduction Value
Contradiction Resolution Value
Generalization Value
Infrastructure Value
Data Acquisition Value
```

Örneğin doğrudan alpha üretmeyen bir deney, on farklı teoriyi çürütebilecekse çok yüksek bilimsel değere sahip olabilir.

Önerilen puanlama:

```text
Priority =
Information Gain
× Decision Relevance
× Falsifiability
× Generalization Potential
× Contradiction Resolution Value
× Data Readiness
÷
Research Cost
÷
Multiple-Testing Risk
÷
Execution Complexity
```

Ayrıca portfolio yaklaşımı kullanılmalıdır:

```text
60% exploitation research
Known strong areas

25% exploration research
New mechanisms and states

15% curiosity research
Low-probability, high-novelty questions
```

---

# 26. Epistemic Risk Engine

Trading Risk Engine para kaybı riskini ölçer.

Epistemic Risk Engine yanlış bilgiyle karar verme riskini ölçmelidir.

Risk türleri:

```text
Overfit Risk
Leakage Risk
Regime Risk
Execution Risk
Measurement Risk
Selection Bias
Survivorship Bias
Narrative Risk
Multiple-Testing Risk
Automation Bias
False Certainty Risk
```

Bir trade ekonomik olarak iyi görünse bile dayandığı bilginin epistemic risk’i yüksekse sistem pozisyon açmayabilir.

---

# 27. Agent Reliability Scores

AMI içindeki araştırma agent’ları da değerlendirilmelidir.

```text
Agent:
Hypothesis Generator

Historical Accuracy:
Calibration:
Replication Survival:
False Discovery Rate:
Novelty:
Cost Efficiency:
Common Failure Modes:
```

Sürekli başarısız veya aşırı güvenli hipotez üreten agent’ın etkisi azaltılmalıdır.

Böylece sistem yalnızca market knowledge değil, araştırmacı bileşenlerinin güvenilirliğini de öğrenir.

---

# 28. Intelligence Metrics

AMI’nin başarısı dört kategoride ölçülmelidir.

## Scientific Metrics

```text
Replication Rate
False Discovery Rate
Theory Survival Rate
Prediction Accuracy
Contradiction Resolution Rate
Knowledge Half-Life
```

## Epistemic Metrics

```text
Uncertainty Calibration
Overconfidence Rate
Unknown Detection Rate
Belief Revision Speed
Expired Knowledge Detection
Unsupported Claim Rate
```

## Research Metrics

```text
Information Gain per Experiment
Cost per Valid Discovery
Duplicate Experiment Rate
Time to Falsification
Research Backlog Quality
Experiment Power Efficiency
```

## Operational Metrics

```text
Forward Survival
Execution Robustness
Decision Traceability
Promotion Failure Rate
Demotion Response Time
PnL Conditional on Evidence Level
```

Özellikle şu metrik izlenmelidir:

```text
Operational damage caused by weak knowledge
```

Çünkü sistemin başarısı yalnızca doğru şeyler keşfetmesi değil, yeterince doğrulanmamış bilgilerin karar sistemine sızmasını engellemesidir.

---

# 29. Epistemic Constitution

AMI’nin değiştirilemez veya yalnızca özel yönetişimle değiştirilebilir bazı temel kuralları bulunmalıdır.

Örnek anayasal ilkeler:

```text
No claim without provenance.

No operational promotion without untouched validation.

No mechanism claim from correlation alone.

No confidence without calibration.

No theory without falsifiable predictions.

No live rule without execution validation.

No failed idea is silently deleted.

No contradiction is ignored.

No unknown is converted into certainty.

No agent may override evidence hierarchy.

No economic attractiveness may substitute for scientific validity.
```

Bu ilkeler projenin her bileşeni için bağlayıcı olmalıdır.

---

# 30. Final Architectural Role

Epistemic Governor, AMI’nin yalnızca bir bölümü olmayacaktır.

Bütün sistemin üzerinde çalışan epistemik işletim sistemi olacaktır.

```text
                         EPISTEMIC GOVERNOR
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
Research Permissions       Knowledge Permissions       Decision Permissions
       │                          │                          │
Autonomous Scientist          World Model              Trading / Portfolio
```

Görevi:

```text
Know what is known.
Represent what is uncertain.
Detect what has changed.
Restrict what is overclaimed.
Expire what is outdated.
Preserve what has failed.
Promote only what survives evidence.
```

AMI’nin gücü, her soruya cevap vermesinden gelmeyecektir.

Asıl güç:

> Hangi cevabın kanıtlandığını, hangisinin geçici olduğunu, hangisinin yalnızca belirli koşullarda geçerli olduğunu, hangisinin artık kullanılmaması gerektiğini ve hangi konuda henüz hiçbir şey bilmediğini doğru biçimde ayırt edebilmesinden gelecektir.

Bu nedenle AMI’nin merkezi zekâsı yalnızca World Model veya Autonomous Scientist değildir.

Merkezi zekâ:

> **The ability to govern belief under uncertainty.**

olacaktır.

Bence bunun ardından yazılması gereken en kritik teknik bölüm **Epistemic Governor veri şeması, statü geçiş tablosu ve otomatik promotion/demotion kuralları** olur.


---

# Part XIII — Additional Epistemic Extensions

The Epistemic Core above defines the control plane. The following extensions connect it directly to AMI’s state, research and operational architecture.

## 71. Knowledge Object as an Executable Contract

A Knowledge Object is not passive text. It should expose machine-enforceable functions:

```text
is_applicable(context)
is_fresh(current_time)
is_permitted(action)
required_evidence_gap()
dependent_assumptions()
conflicting_knowledge()
demotion_triggers()
```

## 72. Contextual Applicability

```yaml
context:
  symbol: ETHUSDT
  venue: BINANCE
  session: US
  timeframe: 4h
  regime: HIGH_VOL_RISK_OFF
  execution_mode: MARKET
  data_health: HEALTHY
```

A claim may be true but not applicable to the current context.

## 73. Epistemic Conflict Score

For a proposed action:

```text
Support score
Contradiction score
Unknown score
Data-quality penalty
Regime-distance penalty
Execution-distance penalty
```

The Decision Engine should expose these separately, not hide them in one confidence number.

## 74. Knowledge Freeze and Reproducibility

Before forward validation:

```text
Freeze feature definition
Freeze target
Freeze thresholds
Freeze execution assumptions
Freeze management rule
Freeze promotion criteria
```

Any change creates a new candidate version and resets forward evidence.

## 75. Epistemic Circuit Breakers

Immediate permission suspension triggers:

- source data stale;
- schema change;
- execution slippage outside validated band;
- state distribution drift;
- calibration failure;
- unresolved contradiction;
- unexpected tail;
- code/data lineage missing.

---

# Part XIV — Engineering Architecture

## 76. Proposed Service Map

```text
ami-reality-gateway
ami-data-health
ami-feature-factory
ami-event-store
ami-state-engine
ami-structure-engine
ami-long-engine
ami-short-engine
ami-lifecycle-engine
ami-world-model
ami-research-os
ami-epistemic-governor
ami-knowledge-graph
ami-portfolio-brain
ami-explainability
ami-dashboard
```

Initial implementation can remain modular within one repository and process architecture. The service map describes logical boundaries, not an immediate microservice mandate.

## 77. Event Bus

Canonical events:

```text
MarketObservationReceived
DataQualityChanged
FeatureComputed
StateChanged
StructureTransitioned
CascadeDetected
TradeCandidateCreated
PositionStateChanged
ResearchQuestionCreated
ExperimentCompleted
KnowledgeUpdated
PermissionChanged
ModelDriftDetected
DecisionGenerated
```

## 78. Core Stores

```text
RAW MARKET STORE
FEATURE STORE
EVENT / MECHANISM STORE
STATE STORE
TRADE LIFECYCLE STORE
EXPERIMENT REGISTRY
KNOWLEDGE GRAPH
THEORY LEDGER
FAILURE ARCHIVE
MODEL REGISTRY
AUDIT LOG
```

## 79. Versioning

Version independently:

- raw schema;
- feature schema;
- state taxonomy;
- model;
- experiment;
- Knowledge Object;
- strategy candidate;
- execution model;
- system specification.

## 80. APIs

Illustrative interfaces:

```python
observe(context) -> ObservationBatch
infer_states(observations) -> StateBundle
forecast(state_bundle, horizons) -> ForecastDistribution
propose_actions(forecast, portfolio) -> ActionCandidates
authorize(action, knowledge_context) -> PermissionDecision
generate_questions(anomalies) -> ResearchBacklog
run_experiment(experiment_spec) -> EvidenceBundle
revise_knowledge(evidence_bundle) -> KnowledgeRevision
```

## 81. Auditability

Every operational recommendation should produce an immutable decision packet:

```yaml
decision_id:
timestamp:
context:
states:
forecasts:
candidate_actions:
selected_action:
knowledge_objects:
permission_decision:
uncertainty:
alternatives:
model_versions:
data_versions:
```

---

# Part XV — Validation and Promotion Standard

## 82. Validation Ladder

```text
Observation
In-sample exploration
Chronological split
Untouched holdout
Independent replication
Realistic execution
No-overlap portfolio simulation
Forward shadow
Paper
Small-scale live
Multi-regime validation
Operational knowledge
```

## 83. Minimum Promotion Questions

Before promotion:

```text
Does it survive untouched data?
Does it survive realistic costs?
Does it survive top-winner removal?
Does it survive no-overlap?
Does it survive plausible threshold changes?
Is it independent evidence or the same dataset family?
Is the mechanism stable?
Does the state occur live?
Are data and execution assumptions currently valid?
What would automatically demote it?
```

## 84. Forward Validation Design

Forward validation should record:

- all eligible events, not only selected winners;
- missed data;
- state at decision time;
- model score;
- action permission;
- hypothetical and actual execution;
- counterfactual outcomes;
- candidate version;
- feature version.

## 85. Shadow Observers

Observers may test:

- alternative management;
- scale-in;
- locks;
- LONG transition;
- direction reversal;
- timeframe mapping;
- ML quality score.

Observers may not modify real orders.

## 86. Multiple Testing Control

Required tools may include:

- family-level permutation;
- max-stat correction;
- false discovery rate;
- preregistration;
- holdout quarantine;
- evidence-family grouping.

---

# Part XVI — Current S34 Knowledge Integration

This section records current working evidence as examples of AMI Knowledge Objects. It is not a permanent declaration of truth.

## 87. Current Mechanism Lessons

### 87.1 Post-cascade selection is currently stronger than pre-cascade trading

Observed research:

- pre-cascade detection can increase cascade probability;
- honest all-alert trading did not produce positive expectancy;
- early entry is therefore a navigation signal, not currently a trade rule.

Status:

```text
HOLDOUT_SUPPORTED
FORWARD_SCOPE_REQUIRED
RESEARCH_ONLY
```

### 87.2 Micro-timing and execution engineering did not create the edge

Current tested findings:

- 2s–15m delays did not create broad alpha;
- limit entries suffered from missed fills;
- VWAP entry added little;
- dynamic TP did not beat baseline;
- volatility stops damaged the tested dip-holding mechanism.

Interpretation:

> For the current S34 family, selection appears more important than entry micro-engineering.

### 87.3 Book pull is more robust than refill

- `bk_pull` retained useful separation in the gated universe;
- `bk_refill` changed sign between populations;
- refill should not be promoted as a universal mechanism.

### 87.4 Mechanism composite

A book/funding/flow mechanism composite produced promising holdout behaviour in the current research universe.

Required next state:

```text
SHADOW_ALLOWED
FORWARD_VALIDATING
NOT_LIVE_ALLOWED
```

### 87.5 Funding

Funding level currently appears stronger than funding velocity in the tested mechanism research, but this remains scoped to the tested event family and period.

### 87.6 Management

- uniform 6h hold remains a strong baseline;
- tight trailing and early loser cuts were harmful in tested data;
- wide profit-lock `200/100` is an observer candidate;
- scale-in at `-100 bps` within two hours is an in-sample observer candidate;
- neither may change live sizing without forward evidence and risk approval.

### 87.7 OI and basis

OI and spot/basis collection has restarted. Historical gaps mean future OI mechanism research requires forward accumulation.

## 88. Current Closed Hypotheses

Examples that should remain in the Failure Archive:

- naive early pre-cascade entry;
- broad micro-timing optimization;
- tight volatility stop;
- partial exits in tested form;
- limit-entry improvement in tested form;
- refill as a robust universal separator;
- several previously rejected reversal/fade/cross-asset variants.

## 89. Current Open Questions

- Does the mechanism composite survive forward?
- Does scale-in `-100/2h` remain beneficial under real sequential conditions?
- Can post-entry state transitions predict MFE giveback?
- Does SHORT_NOISY contain distinct scalp, swing and daily LONG subtypes?
- Which OI/funding dynamics precede durable LONG recovery?
- Can a full SHORT → exhaustion → LONG swing graph be learned?
- Are structural transitions stable across ETH, BTC and SOL?
- Which findings generalize outside crypto?

---

# Part XVII — Implementation Roadmap

## 90. Phase 0 — Constitutional Foundation

Deliverables:

- canonical repository structure;
- Epistemic Constitution;
- Knowledge Object schema;
- experiment registry;
- status and permission enums;
- data lineage minimum standard;
- failure archive;
- decision records.

## 91. Phase 1 — State Foundation

- unified state object;
- multi-timeframe feature alignment;
- 1m/5m/15m/1h/4h/1D/1W state store;
- data-health state propagation;
- initial hand-defined state taxonomy.

## 92. Phase 2 — Structure Engine

- structure phase labels;
- transition matrix;
- LONG/SHORT dual-direction outputs;
- timeframe conflict model;
- swing graph.

## 93. Phase 3 — Trade Lifecycle

- post-entry snapshots;
- MFE/MAE state classifier;
- route-specific lifecycle models;
- observer framework;
- explanation packets.

## 94. Phase 4 — Research OS

- question registry;
- hypothesis templates;
- preregistered experiment generator;
- evidence ingestion;
- contradiction detection;
- research marketplace.

## 95. Phase 5 — Epistemic Governor MVP

- permission checks;
- promotion/demotion;
- assumption dependency;
- freshness/expiration;
- Knowledge Object audit;
- live boundary enforcement.

## 96. Phase 6 — ML and Latent States

- feature embeddings;
- clustering;
- HMM/change-point candidates;
- calibration;
- historical analogue retrieval;
- model registry.

## 97. Phase 7 — World Model and Digital Twin

- probabilistic scenarios;
- state transition forecasting;
- counterfactual engine;
- model error and calibration feedback.

## 98. Phase 8 — Autonomous Scientist

- anomaly-triggered questions;
- competing hypothesis generation;
- experiment ranking;
- theory ledger;
- automatic research proposals;
- human approval boundaries.

## 99. Phase 9 — Cross-Market Generalization

- cross-symbol;
- cross-exchange;
- equities/FX/commodities;
- universal vs asset-specific market laws;
- transfer learning with explicit generalization scores.

---

# Part XVIII — Required Schemas

## 100. Knowledge Object Example

```yaml
knowledge_id: K-S34-BOOK-PULL-001
claim: >
  Low pre-cascade bid-depth withdrawal is associated with higher
  post-cascade reversal outcomes in the gated ETH SELL universe.
claim_type: PREDICTIVE
status: HOLDOUT_VALIDATED
mechanism: LIQUIDITY_REMAINS_AND_ABSORBS_FORCED_FLOW
effect_size:
  delta_bps: 70
evidence:
  families: 2
  chronological_holdouts: 1
  forward_events: 0
contradictions:
  - refill did not remain stable across universes
scope:
  symbols: [ETHUSDT]
  timeframes: [event, 6h]
  regimes: [tested_gate_only]
confidence:
  statistical: HIGH
  mechanism: MEDIUM
  forward: LOW
  generalization: UNKNOWN
permissions:
  allowed: [RESEARCH_ONLY, SHADOW_ALLOWED]
  forbidden: [LIVE_ALLOWED, SIZING_ALLOWED]
falsification:
  - negative or null forward effect after minimum sample
  - effect disappears under executable timestamps
assumptions:
  - book data health is HEALTHY
  - feature definition remains frozen
last_verified: 2026-07-02
```

## 101. Research Question Example

```yaml
question_id: Q-LONG-TRANSITION-001
question: >
  Does failed SHORT_NOISY continuation produce a 4h or 1D LONG state
  after breakdown reclaim?
origin:
  anomaly: positive MFE followed by negative time exit
hypotheses:
  H1: short is scalp-only
  H2: failed short transitions to long
  H3: route contains multiple subtypes
priority:
  information_gain: high
  economic_value: high
  data_readiness: medium
protocol:
  frozen_features:
    - MFE time
    - reclaim
    - book pull
    - BTC sync
    - OFI flip
  horizons: [30m, 1h, 2h, 4h, 6h, 12h, 24h]
```

## 102. Decision Trace Example

```yaml
decision: WAIT_FOR_CONFIRMATION
direction_candidates:
  LONG: 0.55
  SHORT: 0.21
  NO_TRADE: 0.24
active_states:
  5m: SELLER_EXHAUSTION
  1h: RECLAIM_ATTEMPT
  4h: RANGE
  1D: RECOVERY_BIAS
support:
  - bid pull stopped
  - price impact declining
  - BTC synchronization recovering
counterevidence:
  - OFI remains negative
knowledge_used:
  - K-S34-BOOK-PULL-001
permission:
  result: SHADOW_ONLY
uncertainty: HIGH
```

---

# Part XIX — Intelligence Metrics

## 103. Scientific Metrics

- replication rate;
- false discovery rate;
- theory survival;
- prediction success;
- contradiction resolution;
- knowledge half-life.

## 104. Predictive Metrics

- Brier score;
- log loss;
- calibration error;
- transition accuracy;
- horizon-specific forecast accuracy;
- tail-event recall.

## 105. Epistemic Metrics

- overconfidence;
- underconfidence;
- unsupported claim rate;
- unknown detection;
- revision latency;
- stale knowledge detection.

## 106. Research Metrics

- information gain per experiment;
- cost per valid discovery;
- duplicate experiment rate;
- time to falsification;
- data-readiness utilization;
- percent of questions producing reusable knowledge.

## 107. Operational Metrics

- forward survival;
- execution mismatch;
- promotion failure;
- demotion response;
- decision trace completeness;
- PnL by evidence level;
- damage caused by weak knowledge.

## 108. System Intelligence Score

A composite score may eventually combine:

```text
Scientific validity
Prediction calibration
Knowledge growth
Adaptation speed
Operational safety
Generalization
Research efficiency
```

PnL should remain a downstream metric, not the sole definition of intelligence.

---

# Part XX — Research Backlog Generated by This Specification

## 109. Highest-Priority Near-Term Questions

1. Can post-entry state transitions identify profitable MFE giveback exits without damaging 6h swing expectancy?
2. Does SHORT_NOISY split into:
   - short scalp,
   - no-trade subtype,
   - failed-short LONG transition,
   - daily LONG structure?
3. Does the mechanism composite survive forward?
4. Does `bk_pull` retain direction and effect under new forward data?
5. Does OI contraction or expansion distinguish exhaustion from continuation?
6. Can funding/OI/book interactions create an independent LONG family?
7. Can a state transition model outperform static rule selection?
8. Which validation metrics best predict forward survival?
9. What evidence threshold should permit sizing influence?
10. Which current claims have high expiration risk?

## 110. Medium-Term Questions

- cross-exchange lead-lag;
- maker inventory proxy;
- universal structure across assets;
- 1D/1W state fusion;
- theory prediction accuracy;
- latent states;
- digital-twin calibration;
- autonomous question quality.

---

# Part XXI — Definition of Done

AMI is not “done” when it places trades.

A meaningful early AMI version is complete when it can:

```text
1. Observe data with explicit health and provenance.
2. Represent synchronized multi-timeframe states.
3. Produce LONG, SHORT and NO-TRADE probabilities.
4. Track a trade through lifecycle states.
5. Store every claim as a governed Knowledge Object.
6. Generate and rank research questions.
7. Execute preregistered experiments.
8. Detect contradictions and revise beliefs.
9. Explain every recommendation.
10. Enforce research/shadow/live permission boundaries.
```

A mature AMI version additionally:

- discovers latent states;
- predicts state transitions;
- builds and revises theories;
- simulates alternative futures;
- learns research methods;
- transfers knowledge across markets;
- maintains calibrated uncertainty.

---

# Closing Vision

The long-term objective is not merely to build a profitable trading strategy.

The objective is to build a continuously learning digital twin and autonomous scientist of financial markets: a probabilistic, multi-agent, multi-timeframe, causality-aware intelligence capable of observing how markets emerge, evolve, interact and transition through hidden structural states.

LONG and SHORT are not independent strategies. They are complementary manifestations of the same evolving structure.

Liquidation is not the beginning of the analysis. It is one visible symptom of a deeper process.

Trading is not the definition of AMI. Trading is one application of its knowledge.

AMI’s central intelligence is not certainty.

It is:

> **the ability to govern belief under uncertainty, discover what remains unknown, and continuously improve how knowledge is created.**

---

# Appendix A — Canonical State Families

```text
GLOBAL_STATE
REGIME_STATE
SESSION_STATE
STRUCTURE_STATE
MICROSTRUCTURE_STATE
BOOK_STATE
MAKER_STATE
FLOW_STATE
LEVERAGE_STATE
NETWORK_STATE
PRESSURE_STATE
CASCADE_STATE
LONG_STATE
SHORT_STATE
TRADE_STATE
POSITION_STATE
EXIT_STATE
PORTFOLIO_STATE
KNOWLEDGE_STATE
RESEARCH_STATE
```

# Appendix B — Canonical Actions

```text
OBSERVE
RESEARCH
WAIT
NO_TRADE
OPEN_LONG
OPEN_SHORT
HOLD
SCALE
REDUCE
LOCK
EXIT
REVERSE
PAUSE
DEMOTE
PROMOTE
RETEST
EXPIRE
```

# Appendix C — Canonical Knowledge Statuses

```text
OBSERVATION
OPEN_QUESTION
HYPOTHESIS
TESTING
PRELIMINARY
REPLICATED
HOLDOUT_VALIDATED
FORWARD_VALIDATING
OPERATIONAL_CANDIDATE
PROVISIONALLY_ACCEPTED
REGIME_LIMITED
CONTRADICTED
WEAKENED
DEPRECATED
REJECTED
RETRY_WHEN_CONDITIONS_CHANGE
```

# Appendix D — Mandatory Experiment Checklist

```text
[ ] Question is explicit
[ ] Null and alternatives exist
[ ] Population is frozen
[ ] Target is frozen
[ ] Metrics are frozen
[ ] Economic effect size is defined
[ ] Untouched data exists
[ ] Negative control exists
[ ] Multiple testing is addressed
[ ] Execution assumptions are stated
[ ] Data quality is verified
[ ] Falsification rule is explicit
[ ] Result enters Knowledge Graph
[ ] Failure enters Failure Archive
```

# Appendix E — AMI Build Instruction for an Engineering Agent

An engineering agent receiving this document should:

1. treat it as a system specification, not a request to implement everything at once;
2. produce a gap analysis against the current repository;
3. identify reusable existing S34 components;
4. propose a phased implementation plan;
5. preserve live guardrails;
6. implement research and governance foundations before autonomous execution;
7. create tests for data lineage, permissions and state transitions;
8. never modify live order logic, leverage, sizing or `.env` without explicit operator approval;
9. update this specification and the changelog after every major architectural decision;
10. return concrete artifacts, code, schemas, tests and research reports for each completed phase.


---

# Appendix F — Direct Build Brief for Claude / Codex

The engineering agent should begin by producing a **repository gap analysis**, not by writing autonomous trading code.

Required first response:

```text
1. Existing components that already satisfy AMI requirements
2. Missing foundations
3. Data-quality risks
4. Knowledge-governance risks
5. Proposed directory structure
6. Proposed schemas
7. Proposed Phase 0 implementation
8. Explicit files that must not be modified
9. Test plan
10. Migration and rollback plan
```

## Mandatory guardrails

```text
Do not modify live executor.
Do not modify .env.
Do not modify leverage.
Do not modify real sizing.
Do not send exchange orders.
Do not promote research findings.
Do not silently redefine existing forward candidates.
Do not mix exploratory and untouched datasets.
```

## First implementation package

The first code package should contain only:

```text
ami/knowledge/
ami/research/
ami/states/
ami/governance/
schemas/
tests/
docs/ami/
```

Minimum initial objects:

```text
KnowledgeObject
ResearchQuestion
Hypothesis
ExperimentSpec
EvidenceBundle
TheoryObject
StateObject
PermissionDecision
DecisionTrace
```

Minimum initial tests:

```text
Knowledge provenance required
Invalid promotion rejected
Stale data blocks applicability
Contradiction lowers permission
Candidate version change resets forward evidence
Research-only knowledge cannot authorize live use
```

## Build principle

AMI should emerge through a sequence of validated capabilities.

```text
First:
Know what we observed.

Then:
Know what we claim.

Then:
Know how uncertain the claim is.

Then:
Know which state is active.

Then:
Know which transition is probable.

Only then:
Use that knowledge operationally.
```

---

# Appendix G — Living Document Protocol

Every future research session should update this document through a structured patch.

```yaml
change_id:
date:
author:
section_changed:
reason:
new_evidence:
affected_knowledge:
status_change:
implementation_change:
validation_required:
```

New ideas should not replace old sections without genealogy.

Use:

```text
ADD
REVISE
RESTRICT
DEPRECATE
REJECT
```

The whitepaper should evolve as AMI evolves.

---

# Final Declaration

AMI is not a promise that markets can be perfectly predicted.

It is a commitment to build a system that:

```text
observes honestly,
tests rigorously,
remembers failures,
represents uncertainty,
changes its mind,
and earns the right to act.
```

---

# Volume VIII — Cycle Intelligence, Position-Aware Decisions and Forward Research

# 57. Why Event-Centric Research Is Incomplete

An event is not a strategy. It is one observation inside a longer market process.

A BUY-side liquidation event may occur during:

- a healthy LONG continuation;
- a mature but still valid LONG;
- temporary exhaustion;
- distribution;
- an intraday pullback inside a higher-timeframe uptrend;
- a full trend reversal;
- a liquidity vacuum;
- a failed fade that becomes a squeeze;
- or unresolved range noise.

Therefore, the same event can have different economic meanings for different position states.

```text
FLAT trader:
Should I enter LONG, enter SHORT or wait?

Existing LONG:
Should I hold, reduce, exit, or reverse?

Existing SHORT:
Should I hold, add, reduce, exit, or prepare to reverse?
```

AMI must not confuse these problems.

The system must represent three distinct forms of value:

```text
Entry value
Management value
Risk-information value
```

A market observation can be valuable even when it does not justify a new trade.

---

# 58. Canonical Market-Cycle State Machine

The primary research object is the market cycle, not the isolated signal.

```text
LONG_GENESIS
→ LONG_EXPANSION
→ LONG_MATURE
→ EXHAUSTION_CANDIDATE
→ EVENT_PENDING / NO_EVENT
→ EVENT_ACTIVE
→ POST_EVENT_UNRESOLVED
   ├─ HEALTHY_LONG_CONTINUATION
   ├─ FAILED_FADE_LONG
   ├─ SHALLOW_PULLBACK
   ├─ DEEP_PULLBACK_FULL_RECLAIM
   ├─ TEMPORARY_FADE_RANGE
   ├─ CLEAN_SHORT_CORRECTION
   ├─ MULTI_HOUR_SHORT
   ├─ FULL_TREND_REVERSAL
   ├─ TWO_STAGE_DECLINE
   └─ NO_RESOLUTION_NOISE
→ TRANSITION
   ├─ LONG_RECLAIM
   ├─ SHORT_RELOAD
   ├─ WAIT
   └─ CYCLE_CLOSED
```

## 44.1 State requirements

Every state record should contain:

```yaml
state_id:
cycle_id:
state_family:
state_label:
state_start_ts:
state_age:
first_known_ts:
confidence:
label_confidence:
primary_timeframe:
supporting_timeframes:
structural_location:
market_clock_age:
data_quality:
evidence_status:
```

## 44.2 Duration-aware transitions

State transitions should not be assumed memoryless.

A LONG_MATURE state that has existed for five minutes is not equivalent to one that has existed for nine hours. AMI should support semi-Markov or other duration-aware research models.

Required fields:

```text
state_dwell_time
signal_age
cycle_age
time_since_last_progress
time_since_last_confirmation
```

## 44.3 Soft and horizon-dependent path labels

One event can be:

```text
T+30m: CLEAN_SHORT_CORRECTION
T+4h: DEEP_PULLBACK_FULL_RECLAIM
T+1D: HEALTHY_LONG_CONTINUATION
```

Therefore the taxonomy must store:

```yaml
classification_horizon:
primary_path:
secondary_path:
path_probability:
label_confidence:
taxonomy_version:
```

Hard labels must not erase ambiguity.

---

# 59. Position-Aware Action Space

AMI should rank feasible actions rather than force a direction prediction.

Canonical actions:

```text
HOLD_LONG
ADD_LONG
PARTIAL_EXIT_LONG
EXIT_LONG
REENTER_LONG
HOLD_SHORT
ADD_SHORT
PARTIAL_EXIT_SHORT
EXIT_SHORT
REENTER_SHORT
ENTER_LONG_NOW
ENTER_LONG_LATER
ENTER_SHORT_NOW
ENTER_SHORT_LATER
WAIT
NO_ACTION
```

For each action, the research layer should estimate:

```yaml
conditional_ev:
median:
mean:
tail_risk:
expected_log_growth:
execution_cost:
opportunity_cost:
confidence:
evidence_level:
permitted_use:
```

## 45.1 FLAT state

The key question is not whether the event predicts a direction. It is whether any executable action beats WAIT.

Mandatory comparisons:

```text
WAIT
ENTER_LONG_NOW
ENTER_LONG_LATER
ENTER_SHORT_NOW
ENTER_SHORT_LATER
```

## 45.2 ALREADY_LONG state

Mandatory comparisons:

```text
HOLD_LONG
PARTIAL_EXIT_LONG
EXIT_LONG
EXIT_LONG_THEN_WAIT
REVERSE_SHORT
```

The system must separate LONG-exit alpha from SHORT-entry alpha.

## 45.3 ALREADY_SHORT state

Mandatory comparisons:

```text
HOLD_SHORT
PARTIAL_EXIT_SHORT
EXIT_SHORT
ADD_SHORT
EXIT_SHORT_THEN_LONG
```

A signal that is negative as a fresh SHORT entry can still be useful as management information for a pre-existing SHORT.

## 45.4 Position path state

Position state includes more than direction.

```yaml
position_age:
entry_ts:
entry_distance_from_event:
position_origin_route:
current_pnl:
max_pnl_since_entry:
max_drawdown_since_entry:
pnl_giveback:
entry_quality_percentile:
partial_exit_state:
reentry_count:
```

A new LONG and a twelve-hour-old LONG should not automatically receive the same action.

---

# 60. Unified LONG and SHORT Research Architecture

LONG and SHORT are connected phases, but they should not be forced into false symmetry.

## 46.1 Unconditional LONG genesis

Pre-event LONG research must not select only LONGs that are later followed by an event.

Two populations are mandatory:

```text
EVENT_CONDITIONED_LONG_GENESIS
UNCONDITIONAL_ALL_TIME_LONG_GENESIS
```

The unconditional detector must run across all eligible timestamps. It must retain:

- LONG genesis followed by an event;
- LONG genesis not followed by an event;
- false signals;
- early stops;
- healthy continuations;
- and unresolved cases.

Conditioning on a future event can create selection or collider bias. Such research cannot be promoted as a real-time LONG detector without an unconditional control population.

## 46.2 LONG genesis candidates

Research candidates include:

```text
compression breakout
local-low reclaim
1h structure reversal
4h higher-low
daily reclaim
seller-exhaustion recovery
BTC-led recovery
flow-confirmed expansion
OI-supported acceptance
```

## 46.3 Failed Fade → LONG Continuation

A failed SHORT setup may contain opposite-direction information.

Candidate confirmations:

```text
event-high reclaim
event-high acceptance for 1m / 5m
failure to produce downside progress
first higher-low after the event
reclaim-retest hold
SHORT stop followed by structural reclaim
OI expansion while price holds above event high
```

Mandatory benchmarks:

```text
NO_TRADE
T0_SHORT
HOLD_EXISTING_LONG
FAILED_FADE_LONG
WAIT_THEN_LONG
```

## 46.4 LONG horizon, management and stop taxonomy

LONG research must be as deep as SHORT research.

Horizons:

```text
5m / 15m / 30m / 1h / 2h / 4h / 6h / 12h / 1D / 2D / 3D / 7D
```

Management candidates:

```text
fixed hold
time-stop
milestone lock
MFE giveback
partial exit
event-triggered exit
higher-timeframe core position
structural break exit
```

Neutral stop labels:

```text
LONG_BAD_TIMING_CANDIDATE
LONG_WRONG_DIRECTION_CANDIDATE
LIQUIDITY_SWEEP_CANDIDATE
BTC_SHOCK_CANDIDATE
STRUCTURAL_INVALIDATION
UNKNOWN
```

Outcome-derived labels must be separated from real-time candidate labels.

## 46.5 Directional asymmetry

AMI should test, not assume, mirror symmetry.

Possible verdicts:

```text
STRUCTURALLY_SYMMETRIC
PARTIALLY_SYMMETRIC
DIRECTIONALLY_ASYMMETRIC
INSUFFICIENT_SAMPLE
```

A BUY-event fade and a SELL-event reversal may require completely different models, horizons and execution assumptions.

---

# 61. Signal Aging and Market Clock

A correct signal can become economically useless before it becomes statistically confirmable.

## 47.1 Signal lifecycle

Every observer should record:

```yaml
signal_birth_ts:
first_known_ts:
first_executable_ts:
last_valid_ts:
signal_age_at_entry:
time_since_last_progress:
time_since_last_confirmation:
```

Research questions include:

- What is the half-life of event information?
- Does EV decay smoothly or collapse at discrete boundaries?
- Does a new same-direction event refresh the old signal or create a new signal?
- Can LONG-exit information remain valid longer than SHORT-entry information?
- Can a confirmation be scientifically correct but economically late?

## 47.2 Alternative market clocks

Wall-clock time is only one representation of market time.

Store:

```text
wall_clock_age
trade_count_age
volume_age
realized_vol_age
liquidation_count_age
book_update_age
```

Ten quiet minutes in Asia and ten minutes during a high-intensity US session may represent different amounts of economic time.

Research should compare fixed wall-clock horizons with:

- volume time;
- trade-count time;
- volatility time;
- liquidation-event time;
- order-book-update time.

## 47.3 Signal age versus state age

Signal age and state age are separate.

```text
Event age = 2 minutes
LONG_MATURE state age = 9 hours
```

Transition research must preserve both.

---

# 62. Scalp, Intraday and Swing Are Different Routes

Scalp and swing should not be defined only by holding the same entry for different durations.

Canonical route families:

```text
SHORT_SCALP
SHORT_INTRADAY
SHORT_SWING
LONG_SCALP
LONG_INTRADAY
LONG_SWING
```

## 48.1 SCALP characteristics

```text
microstructure displacement
fast confirmation
small adverse tolerance
strict no-progress exit
high latency sensitivity
high fee/slippage sensitivity
limited re-entry
```

## 48.2 SWING characteristics

```text
structural transition
higher-timeframe alignment
wider invalidation
slower confirmation
lower sensitivity to small fill differences
longer causal-attribution decay
```

Research must determine:

- whether scalp and swing originate from the same population;
- whether the first decline is a scalp and the second decline a separate swing route;
- when a scalp may be upgraded to a swing;
- when a swing should be downgraded to a scalp;
- whether route class must be frozen at entry or may change through a versioned state transition.

Route-class changes must never be selected with hindsight.

---

# 63. Dynamic Hold, Competing Risks and Progress

Fixed exit grids are useful baselines, but open trades face competing termination risks.

Potential competing events:

```text
TP
SL
event-high reclaim
opposite liquidation
BTC reversal
regime transition
no-progress
new independent event
scheduled boundary
exogenous shock
```

## 49.1 Competing-risk fields

```text
hazard_tp
hazard_sl
hazard_reclaim
hazard_opposite_event
hazard_regime_change
survival_probability
remaining_expected_move
```

Research should estimate:

```text
P(TP before SL | currently available path)
P(reclaim before target | currently available path)
remaining EV after T+5 / T+15 / T+30
```

## 49.2 Progress-conditioned management

Record:

```text
expected_progress_1m / 3m / 5m / 10m
actual_progress
progress_ratio
time_since_new_extreme
distance_from_expected_path
```

Key questions:

- If a SHORT fails to create a new low in five minutes, how much does expectancy fall?
- If a LONG reclaim fails to gain acceptance, should the trade be cancelled?
- Does a no-progress exit avoid losers without sacrificing slow winners?
- Does progress need normalization by volatility, session or market clock?

## 49.3 Sequential policy caution

A dynamic hold rule is a sequential policy, not a collection of hindsight-best exits.

Every policy decision must store:

```yaml
policy_id:
policy_version:
decision_ts:
available_information:
chosen_action:
alternative_actions:
realized_regret:
```

---

# 64. Structural Location, Event Geometry and Mechanism

## 50.1 Structural location

An event’s location may matter more than its raw size.

Record:

```text
distance_to_previous_day_high
distance_to_previous_day_low
distance_to_weekly_high
distance_to_weekly_low
distance_to_vwap
distance_to_anchored_vwap
range_percentile
distance_to_breakout_level
distance_to_volume_node
ATR_extension
trend_channel_extension
```

Example hypothesis:

```text
Large event in range middle
→ noise / failed fade / continuation LONG

Smaller event at mature extension and structural resistance
→ clean SHORT scalp
```

## 50.2 Event geometry

Record:

```text
liquidation_count
largest_liquidation_share
event_duration
inter_arrival_time
notional_concentration
price_displacement
price_impact_per_dollar
notional_to_5m_volume
notional_to_local_depth
rolling_notional_percentile
cross_symbol_synchronization
market_wide_share
```

Raw dollar thresholds should be compared with normalized definitions.

## 50.3 Neutral micro-mechanism taxonomy

Initial labels must remain neutral:

```text
MECH_A — aggression persists while price progress decays
MECH_B — aggression itself decays
MECH_C — liquidity withdraws and price crosses a thin book
MECH_MIXED_OR_UNKNOWN
```

Only after independent evidence should these be interpreted as:

```text
absorption
aggression exhaustion
liquidity vacuum
```

Manipulation labels such as spoofing or iceberg must never be inferred from insufficient data.

## 50.4 Order-book resilience

Snapshot imbalance may be weaker than resilience dynamics.

Research fields:

```text
refill_speed
cancellation_intensity
queue_persistence
depth_recovery_half_life
trade_to_cancel_ratio
same_level_replenishment
```

---

# 65. Cycle Integrity, Event Overlap and Censoring

## 51.1 Cycle-level deduplication

Every event should belong to an independently defined cycle.

```yaml
cycle_id:
cycle_start_ts:
cycle_end_ts:
cycle_phase:
event_order_in_cycle:
event_count_in_cycle:
primary_event:
largest_event:
final_event:
```

Mandatory sample counts:

```text
event_N
independent_cycle_N
independent_day_N
independent_week_N
effective_sample_size
```

The same cycle must never be split across train and validation.

## 51.2 Event overlap and state reset

A new event may:

- refresh the active state;
- create a new independent state;
- strengthen the existing hypothesis;
- invalidate it;
- censor the old outcome;
- or force WAIT.

State-reset rules must be frozen before outcome calculation.

## 51.3 Right censoring and outcome integrity

Long horizons require explicit censoring.

Do not treat as ordinary completed outcomes:

- events near the end of the dataset;
- events crossing a data outage;
- outcomes dominated by a new independent shock;
- multi-day paths whose causal attribution to the initial event has decayed.

Support:

```text
RIGHT_CENSORED
DATA_CENSORED
SHOCK_CENSORED
NEW_CYCLE_CENSORED
COMPLETED
```

Survival analysis should be compared with fixed-endpoint results.

---

# 66. Regime Transitions, Conflict and Multi-Horizon Direction

Static labels are insufficient.

Store transitions:

```text
UP→UP
UP→RANGE
UP→DOWN
RANGE→UP
RANGE→DOWN
DOWN→RANGE
DOWN→UP
```

Also store derivatives:

```text
TREND_STRENGTH_RISING
TREND_STRENGTH_FALLING
VOL_CONTRACTION→EXPANSION
STRESS_RISING
STRESS_PEAKING
STRESS_DECAYING
```

## 52.1 Evidence conflict

Direction, horizon and timing must be separate outputs.

```text
1h SHORT
24h LONG
```

can be simultaneously correct.

Conflict examples:

- microstructure SHORT versus higher-timeframe LONG;
- price confirmation versus flow disagreement;
- strong geometry at poor structural location;
- BTC weakness versus ETH-specific acceptance.

AMI should calculate an evidence-conflict score and narrow the permitted action set as conflict rises.

Conflict should generally increase abstention, not automatically change size.

---

# 67. Execution, Latency, Capacity and Venue

## 53.1 Entry mechanics

Timing is not the same as order mechanics.

Compare:

```text
market entry
maker limit
pullback limit
breakdown stop-entry
breakdown-retest entry
partial fill
```

Mandatory fields:

```text
eligible_N
attempted_N
filled_N
missed_N
partial_fill_N
fill_rate
fill_adverse_selection
missed_winner_cost
```

A limit entry can appear superior because it fills mainly on bad trades. This must be tested explicitly.

## 53.2 Data availability latency

Every feature must carry:

```text
event_ts
exchange_ts
receive_ts
processed_ts
feature_known_at
staleness_ms
coverage_state
```

An analytically valid feature may still be operationally unavailable at decision time.

## 53.3 Venue propagation

A single-venue event may be local noise or a lagging observation of a market-wide process.

Research states:

```text
VENUE_LOCAL
VENUE_LEADING
VENUE_LAGGING
MARKET_WIDE
UNKNOWN
```

If data are unavailable, use:

```text
BLOCKED_BY_CROSS_VENUE_DATA
```

## 53.4 Capacity

Research-only capacity metrics:

```text
maximum_executable_size
slippage_by_notional
partial_fill_risk
capacity_adjusted_EV
```

Capacity research can never grant sizing permission.

---

# 68. Forward Intelligence Observatory and Dashboard

The forward system should collect more than closed trade PnL.

It should preserve the full cycle:

```text
pre-event LONG genesis
→ LONG expansion and maturity
→ event
→ all executable SHORT timing candidates
→ management alternatives
→ SHORT exit
→ re-entry candidates
→ SHORT-to-LONG transition
→ later cycle resolution
```

## 54.1 Immutable forward event record

```yaml
event_id:
cycle_id:
event_family:
symbol:
venue:
event_ts:
event_side:
event_notional:
route_version:
collector_version:
schema_version:
code_commit:
data_health_at_event:
feature_coverage:
missing_features:
session:
regime:
timeframe_alignment:
```

## 54.2 Snapshot schedule

Pre-event:

```text
T−7D / T−3D / T−2D / T−1D / T−12h / T−8h / T−6h / T−4h
T−3h / T−2h / T−1h / T−30m / T−15m / T−10m / T−5m / T−3m / T−1m
```

Post-event:

```text
T+30s / T+1m / T+2m / T+3m / T+5m / T+10m / T+15m / T+30m
T+45m / T+60m / T+75m / T+90m / T+2h / T+3h / T+4h / T+6h
T+8h / T+12h / T+24h / T+2D / T+3D / T+7D
```

Future snapshots remain PENDING. Missing data must never be replaced with zero.

## 54.3 Observer classes

```text
ACTUAL_SHADOW
OBSERVER_HYPOTHETICAL
HISTORICAL_REPLAY
```

These classes must never be combined in performance aggregates.

## 54.4 Observer families

- early SHORT;
- T0 SHORT;
- near-delayed SHORT;
- post-event continuation SHORT;
- pre-event LONG;
- failed-fade LONG;
- LONG exit;
- fixed and structural exit;
- re-entry;
- direction transition;
- general opposite-liquidation exit;
- stop taxonomy.

## 54.5 Required dashboard pages

1. AMI Command Center  
2. Forward Experiments  
3. Market-Cycle Explorer  
4. Position-Aware Action Board  
5. Entry Timing Laboratory  
6. Exit and Management Laboratory  
7. LONG/SHORT Transition Map  
8. Re-entry and Churn  
9. Silence Intelligence  
10. Multi-Timeframe Matrix  
11. Event Geometry and Structural Location  
12. Mechanism and Book Resilience  
13. Data Health and Drift  
14. Knowledge and Permissions  
15. Research Question Registry  
16. Evidence Independence and Contamination  
17. Policy Evaluation and Regret  
18. Portfolio Overlap and Sequence Risk

## 54.6 Suggested logical tables

```text
ami_forward_events
ami_market_cycles
ami_forward_snapshots
ami_forward_timeframe_states
ami_forward_observer_entries
ami_forward_observer_exits
ami_forward_long_genesis
ami_forward_silence
ami_forward_reentries
ami_forward_stop_taxonomy
ami_forward_path_labels
ami_forward_action_values
ami_forward_experiment_progress
ami_forward_question_progress
ami_evidence_contamination
ami_researcher_exposure
ami_forward_incidents
```

All writes must be idempotent and versioned.

---

# 69. Canonical Research Question System

AMI must remember unanswered questions as carefully as it remembers answers.

## 55.1 Question object

```yaml
question_id:
title:
canonical_question:
question_family:
parent_question:
child_questions:
origin:
origin_experiment:
origin_observation:
created_at:

cycle_state:
position_state:
direction:
timeframe:
horizon:

primary_hypothesis:
null_hypothesis:
alternative_explanations:

required_data:
required_features:
feature_availability:
minimum_event_n:
minimum_cycle_n:
minimum_days:
required_regimes:
required_forward_duration:

current_evidence:
contradicting_evidence:
evidence_level:
known_leakage_risks:

status:
priority:
information_value:
economic_value:
risk_reduction_value:
scientific_value:
estimated_cost:

blocked_by:
retry_condition:
next_valid_test:
dependencies:
duplicate_of:
related_questions:

permitted_actions:
owner:
last_reviewed:
next_review:
```

## 55.2 Question statuses

```text
NEW
TRIAGED
READY_FOR_PREREG
BLOCKED_BY_DATA
BLOCKED_BY_SAMPLE
BLOCKED_BY_CYCLE_N
BLOCKED_BY_REGIME
BLOCKED_BY_FORWARD
FORWARD_ACCUMULATING
EXPERIMENT_FROZEN
IN_TEST
ANSWERED_SUPPORTED
ANSWERED_FALSIFIED
ANSWERED_INSUFFICIENT
DORMANT
RETIRED
DUPLICATE
```

A falsified question is not deleted. It remains part of the system’s scientific memory.

## 55.3 Question families

The canonical registry covers:

- pre-event LONG genesis;
- LONG maturity and exhaustion;
- early, T0, delayed and late SHORT entry;
- SHORT horizon and management;
- LONG horizon and management;
- silence onset, maturity and breakdown;
- failed-fade LONG;
- LONG→SHORT and SHORT→LONG transitions;
- LONG and SHORT re-entry;
- stop taxonomy;
- multi-timeframe structure;
- regime and regime transition;
- event geometry and structural location;
- path taxonomy and path quality;
- signal aging and market-clock normalization;
- scalp/intraday/swing separation;
- competing risks and progress-conditioned exits;
- execution mechanics and capacity;
- cycle deduplication, overlap and censoring;
- counterfactual controls and causal assumptions;
- evidence contamination and multiple testing;
- OOD, uncertainty and abstention;
- sequential policy evaluation;
- portfolio interaction and non-ergodic capital path;
- data acquisition and failure meta-analysis;
- asset and venue transferability;
- and researcher exposure.

## 55.4 Research-ready gate

A question can become READY_FOR_PREREG only when:

- the population is explicit;
- the event and independent-cycle units are defined;
- position state is defined;
- timestamp-safe features are available;
- baseline and counterfactual controls are defined;
- minimum event and cycle sample sizes are frozen;
- leakage and contamination risks are recorded;
- previous evidence has been reviewed;
- and genuinely new untouched or forward information is available.

---

# 70. Evidence Independence and Epistemic Safety

## 56.1 Evidence contamination ledger

Every hypothesis must record:

```yaml
hypothesis_birth_ts:
hypothesis_origin_split:
splits_seen_before_freeze:
contaminated_splits:
eligible_validation_splits:
fresh_forward_required:
maximum_evidence_ceiling:
```

Evidence statuses:

```text
INDEPENDENT_EVIDENCE
REUSED_EVIDENCE
CONTAMINATED_FOR_CONFIRMATION
FORWARD_ONLY_CONFIRMATION_REQUIRED
```

An untouched split that inspired a new hypothesis is no longer untouched for that hypothesis.

## 56.2 Evidence dependency graph

Claims can share the same events, cycles, features and outcomes.

Record:

```text
evidence_set_id
shared_event_ratio
shared_cycle_ratio
shared_feature_dependency
shared_outcome_dependency
effective_independent_evidence_count
```

Confidence should be based on effective independent evidence, not the raw number of claims.

## 56.3 Multiple-testing governance

Record:

```text
family_id
variants_tested
effective_trials
family_adjusted_significance
threshold_stability
researcher_freedom_score
minimum_economic_effect
```

Use family-level discovery control, threshold-neighborhood stability and explicit search-space accounting.

## 56.4 Researcher exposure

```text
BLINDLY_PREREGISTERED
RESULT_INFORMED_HYPOTHESIS
POST_HOC_EXPLORATION
INDEPENDENT_REPLICATION
```

Every manual override and definition change must be logged.

## 56.5 Causal-assumption registry

AMI must distinguish:

```text
PREDICTIVE_ASSOCIATION
MECHANISM_CONSISTENT
CAUSALLY_SUPPORTED
```

A liquidation event can be a consequence of a prior market state rather than the cause of the later move.

A causal registry should track assumed arrows and possible confounders:

```text
prior trend
leverage and OI
structural location
BTC move
session
liquidity depth
funding
news shock
```

---

# 71. OOD, Uncertainty, Calibration and Abstention

## 57.1 Novelty detection

Record:

```text
distance_to_historical_support
nearest_cycle_distance
feature_density
regime_novelty
execution_novelty
data_quality_novelty
```

States:

```text
IN_DISTRIBUTION
WEAK_SUPPORT
OUT_OF_DISTRIBUTION
NOVEL_REGIME
```

OOD events should generally produce abstention and separate observer reporting.

## 57.2 Uncertainty decomposition

Do not compress uncertainty into one number.

```text
epistemic_uncertainty
market_uncertainty
data_quality_uncertainty
execution_uncertainty
label_uncertainty
regime_uncertainty
```

Different uncertainty compositions may imply different permitted actions.

## 57.3 Confidence calibration

Research:

- whether predicted probabilities match realized frequencies;
- whether confidence is monotonic with economic value;
- whether conflicting action values should produce WAIT;
- whether small cells require shrinkage;
- and whether a confirmation’s information value exceeds its delay cost.

## 57.4 Hierarchical evidence

Narrow cells should be partially pooled toward parent populations rather than accepted raw or discarded automatically.

Example hierarchy:

```text
All BUY-fade
└─ Silence
   └─ Clean silence
      └─ 4h DOWN
         └─ Europe session
```

Partial pooling is a research tool, not a promotion shortcut.

---

# 72. Sequential Policies, Benchmarks and Regret

Action-value matrices do not constitute a strategy unless the action selection rule is frozen before outcomes.

## 58.1 Policy object

```yaml
policy_id:
policy_version:
state_inputs:
action_set:
decision_order:
transition_rules:
invalidations:
training_window:
validation_window:
complexity_budget:
```

## 58.2 Mandatory benchmarks

```text
NO_TRADE
WAIT
HOLD_EXISTING_POSITION
SIMPLE_1H_TREND
SIMPLE_4H_TREND
FIXED_T45
CURRENT_BEST_ROUTE
RANDOM_MATCHED_ENTRY
```

## 58.3 Regret accounting

Report:

```text
absolute_return
incremental_return
regret_vs_best_feasible_action
complexity_cost
data_cost
execution_cost
```

The comparison must be against actions feasible with information available at the decision timestamp, not hindsight-best actions.

---

# 73. Portfolio, Sequence and Capital-Path Intelligence

## 59.1 Portfolio overlap

Measure:

```text
trigger_overlap
cycle_overlap
regime_overlap
return_correlation
failure_mode_overlap
incremental_portfolio_contribution
```

A new route can be profitable standalone yet redundant to the existing portfolio.

## 59.2 Sequence risk

Research:

- whether losers cluster by cycle, regime, day or week;
- whether repeated stops are the same failed state traded repeatedly;
- whether cooldown should be based on state failure rather than trade count;
- whether bottom-tail clusters survive cycle deduplication;
- whether route suspension should use independent cycles instead of the last X trades.

## 59.3 Non-ergodic capital path

Research-only metrics:

```text
expected_log_growth
drawdown_duration
recovery_time
loss_cluster
ruin_probability
capital_lockup
```

Positive mean bps does not guarantee a healthy capital path.

---

# 74. Market-Structure, Shock and Derivatives Context

## 60.1 Market-structure versioning

Record:

```text
market_structure_version
exchange_api_version
fee_schedule_version
contract_version
event_definition_version
```

Tick-size, fee, liquidation-feed and exchange-market-share changes can invalidate historical comparability.

## 60.2 Scheduled boundaries

Track:

- funding timestamps;
- daily and weekly closes;
- session transitions;
- weekends;
- known macro boundaries.

Boundary effects should be treated as population definitions, not post-hoc stories.

## 60.3 Exogenous shocks

States:

```text
NORMAL_MARKET
SCHEDULED_BOUNDARY
EXOGENOUS_SHOCK
DATA_INCIDENT
UNKNOWN_ANOMALY
```

Shock-contaminated events may require abstention or separate research populations.

## 60.4 Derivatives surface

If data become available, study:

```text
futures curve
basis term structure
options IV
skew
expiry proximity
gamma concentration
```

Until available, mark relevant questions:

```text
BLOCKED_BY_DERIVATIVES_DATA
```

---

# 75. Active Data Acquisition and Failure Meta-Research

## 61.1 Data acquisition priority

For each candidate source, estimate:

```text
questions_unblocked
expected_information_gain
implementation_cost
storage_cost
latency_cost
priority
```

Candidates may include:

- OI;
- basis;
- L2 depth;
- cross-venue liquidation;
- options;
- spot/perpetual decomposition;
- macro event tags.

No collector should be enabled automatically.

## 61.2 Failure meta-analysis

The Failure Archive should support higher-level questions:

- Which rejection reason is most common?
- Which question families repeatedly fail economically?
- Which features never produce incremental value?
- Which research designs survive forward more often?
- Which failed routes create useful veto or anti-alpha knowledge?
- Is a new hypothesis merely a renamed previous failure?

AMI should learn about its research process, not only the market.

## 61.3 Asset and venue transferability

Possible verdicts:

```text
ASSET_SPECIFIC
TRANSFERABLE_WITH_RECALIBRATION
CROSS_ASSET_STABLE
FAILED_TRANSFER
```

No threshold or mechanism should be transferred automatically from ETH to BTC, SOL or another venue.

---

# 76. BUY-Fade Case Study — What AMI Learned

The BUY-fade programme is a useful demonstration of AMI’s scientific principles.

## 62.1 Baseline route

Canonical research population:

```text
ETH BUY cascade ≥ 200K
EUROPE and bear-squeeze vetoes
T0 SHORT
45-minute hold
SL75
5 bps fee
```

The broad route failed to generalize:

```text
Train:      −9.5 bps/trade
Validation: −1.1 bps/trade
Untouched: −10.7 bps/trade
```

The original small positive sample did not survive the larger history.

## 62.2 Silence contained real information

The silence subset remained materially better than the noisy subset and survived matched-control analysis.

However, silence was knowable only after the event.

```text
Silence is real descriptive information.
Silence is not a valid T0 entry feature.
```

## 62.3 T+30 delayed entry failed

The independent observer-entry control showed:

```text
T+30 entry train:      −15.9 bps
T+30 entry validation: −13.3 bps
```

By the time silence was confirmed, the economic movement was largely over.

## 62.4 Profit decomposition

For silence-confirmed trades:

```text
Split       T0→T30     T30→T45
Train       +36.7      +0.6
Validation  +31.7      +3.0
Untouched   +22.9      +5.4
```

Approximately ninety percent of the economic movement occurred before silence became fully knowable.

This illustrates a critical AMI principle:

> Information may be statistically real yet operationally late.

## 62.5 T45 was robust

The 45–120 minute area behaved as a plateau. Longer extensions deteriorated in validation. T45 therefore remained a robust baseline rather than an obvious management error.

## 62.6 First new BUY≥50K exit

The best breakdown-style observer passed most technical checks but failed the frozen economic threshold:

```text
Validation incremental: +1.37 bps
Required threshold:      +3.00 bps
Verdict: REJECTED [econ]
```

It also improved the noisy control population, suggesting a possible general SHORT-management mechanism rather than a silence-specific mechanism.

Correct next status:

```text
OBSERVATION_ONLY
FORWARD_NOT_VALIDATED
NO_ORDER_EFFECT
```

## 62.7 Scientific interpretation

Current knowledge:

```text
Broad T0 SHORT route: negative
Silence information: real but late
T+30 entry: negative
T45 exit: robust
Hold extension: non-incremental
General re-entry: churn
LONG maturity alone: non-incremental
```

Therefore the highest-value new hypotheses are not minor T0 optimizations. They are:

- whether the event is primarily a LONG risk-routing event;
- whether failed fades create LONG continuation;
- whether structural location identifies a narrow SHORT scalp population;
- whether 4h-DOWN plus silence contains a separate continuation route;
- whether general opposite-liquidation arrival is useful management information;
- and whether unconditional LONG genesis is economically stronger than the event fade.

---

# 77. Implementation Programme

## Phase A — Evidence safety

1. Evidence contamination ledger  
2. Researcher exposure log  
3. Multiple-testing family registry  
4. Market-structure and data-version ledger  
5. Causal-assumption registry

## Phase B — Cycle integrity

1. Independent cycle engine  
2. Event overlap and reset rules  
3. Censoring and outcome integrity  
4. State dwell time  
5. Soft path taxonomy

## Phase C — Economic route separation

1. Unconditional LONG genesis  
2. Failed-fade LONG  
3. Position-aware action comparison  
4. Scalp/intraday/swing route separation  
5. Structural-location and event-geometry research

## Phase D — Dynamic lifecycle research

1. Signal aging  
2. Market-clock normalization  
3. Competing-risk hold  
4. Progress-conditioned management  
5. Entry mechanics and fill probability

## Phase E — Forward observatory

1. Immutable event and cycle records  
2. Snapshot scheduler  
3. Observer engine  
4. Question-progress service  
5. Evidence-independence dashboard  
6. Daily and weekly scientific reports

## Phase F — Advanced decision research

1. OOD and uncertainty decomposition  
2. Hierarchical shrinkage  
3. Sequential policy evaluation  
4. Portfolio incrementality  
5. Non-ergodic capital-path analysis

No phase grants live permission automatically.

---

# 78. Definition of Done for Version 0.3

Version 0.3 is implemented only when:

- existing v0.2 objects and verdicts remain unchanged;
- the Question Registry supports all new families;
- event-level and cycle-level samples are separated;
- position-aware states are recorded;
- unconditional LONG genesis is measurable;
- failed-fade LONG observer exists;
- signal and state age are separately recorded;
- market-clock fields are available;
- scalp, intraday and swing routes are distinct;
- path taxonomy is horizon-aware and versioned;
- structural location and normalized geometry are stored;
- overlap, reset and censoring rules are enforced;
- data latency is explicit through `feature_known_at`;
- ACTUAL_SHADOW, OBSERVER and HISTORICAL_REPLAY remain separate;
- evidence contamination and researcher exposure are visible;
- OOD and uncertainty do not collapse into one confidence score;
- action-value output cannot create an order;
- policy evaluation is sequential and versioned;
- benchmark and regret reports exist;
- all writes are idempotent;
- mutation and integration suites pass;
- live and shadow operational components remain unchanged;
- rollback is documented;
- and every output remains operationally forbidden until explicit promotion.

---

# Appendix H — Canonical Database Extensions

```sql
-- Logical specification only. Exact dialect and migration style must follow the repository standard.

CREATE TABLE IF NOT EXISTS ami_market_cycles (...);
CREATE TABLE IF NOT EXISTS ami_forward_events (...);
CREATE TABLE IF NOT EXISTS ami_forward_snapshots (...);
CREATE TABLE IF NOT EXISTS ami_forward_timeframe_states (...);
CREATE TABLE IF NOT EXISTS ami_forward_observer_entries (...);
CREATE TABLE IF NOT EXISTS ami_forward_observer_exits (...);
CREATE TABLE IF NOT EXISTS ami_forward_long_genesis (...);
CREATE TABLE IF NOT EXISTS ami_forward_silence (...);
CREATE TABLE IF NOT EXISTS ami_forward_reentries (...);
CREATE TABLE IF NOT EXISTS ami_forward_stop_taxonomy (...);
CREATE TABLE IF NOT EXISTS ami_forward_path_labels (...);
CREATE TABLE IF NOT EXISTS ami_forward_action_values (...);
CREATE TABLE IF NOT EXISTS ami_forward_experiment_progress (...);
CREATE TABLE IF NOT EXISTS ami_forward_question_progress (...);
CREATE TABLE IF NOT EXISTS ami_evidence_contamination (...);
CREATE TABLE IF NOT EXISTS ami_researcher_exposure (...);
CREATE TABLE IF NOT EXISTS ami_forward_incidents (...);
```

All tables require:

```text
primary key
schema version
feature version
observer or policy version
source hash
created_at
updated_at
activation timestamp
provenance
```

---

# Appendix I — Required Mutation and Integration Tests

At minimum, the system must reject or detect:

1. use of future event knowledge in a pre-event LONG detector;
2. reporting event-conditioned LONG as unconditional LONG;
3. counting multiple events from one cycle as independent;
4. splitting one cycle across train and validation;
5. defining a failed fade using future highs;
6. merging FLAT, LONG and SHORT action values;
7. merging LONG-exit alpha with SHORT-entry alpha;
8. using future regime transitions;
9. assigning causal mechanism labels without evidence;
10. replacing missing depth with zero;
11. skipping matched and placebo controls;
12. hiding cycle_N while displaying event_N;
13. promoting a tiny path class;
14. leaking outcome taxonomy into features;
15. reporting oracle entry as executable entry;
16. mixing selection, timing, execution and management attribution;
17. allowing action ranking to generate an order;
18. treating standalone PnL as portfolio incrementality;
19. assuming mirror-direction symmetry;
20. starting experiments automatically from the Question Registry;
21. adding historical replay to forward N;
22. deleting false LONG signals;
23. excluding failed-fade LONG losers;
24. optimizing current-position state with future outcome;
25. modifying live or shadow routes;
26. using a hypothesis-origin split as independent confirmation;
27. counting dependent claims as independent evidence;
28. failing to log result-informed hypothesis changes;
29. using OOD samples as ordinary in-distribution evidence;
30. selecting hindsight-best actions per event;
31. ignoring data staleness and receive-time latency;
32. using incomplete long-horizon outcomes as completed trades;
33. resetting a signal clock post-hoc;
34. hiding missed-winner cost from maker-entry studies;
35. granting size permission from capacity research.

---

# Appendix J — Canonical Research Priorities

## P0 — Scientific integrity

- holdout contamination;
- evidence dependency;
- researcher exposure;
- multiple-testing governance;
- data and market-structure versioning;
- cycle deduplication;
- overlap and censoring;
- data availability latency.

## P1 — Highest information value

- unconditional LONG genesis;
- failed-fade LONG continuation;
- position-aware event value;
- structural location;
- post-event path taxonomy;
- scalp/swing separation;
- signal aging;
- competing-risk hold.

## P2 — Execution reality

- entry mechanics;
- fill probability;
- venue propagation;
- capacity;
- scheduled boundaries;
- exogenous shocks.

## P3 — Advanced intelligence

- OOD and novelty;
- uncertainty decomposition;
- state dwell time;
- order-book resilience;
- cross-asset factor decomposition;
- derivatives context;
- dynamic policy evaluation;
- portfolio and non-ergodic path analysis.

## P4 — Self-improving research

- active data acquisition;
- failure meta-analysis;
- transferability;
- hierarchical shrinkage;
- edge lifecycle;
- agent and researcher reliability.

---

# Appendix K — Source Integration and Provenance

This version integrates:

- the complete AMI Artificial Market Intelligence Whitepaper v0.2;
- the cycle-intelligence, position-aware action and canonical-question extension;
- the signal-aging, market-clock, scalp/swing, hold-risk, location, execution and governance extension;
- the comprehensive Forward Intelligence Observatory and dashboard specification;
- the BUY-fade structural, re-entry and silence-exit findings supplied during the 2026-07-03 research session;
- and the advanced evidence-safety, causal, OOD, policy and meta-research additions defined in Version 0.3.

The original source-language research notes are preserved below to prevent loss of detail and research genealogy.

---

# Appendix L — Source Research Expansion A (Original Turkish)

Evet, hâlâ önemli boşluklar var. Fakat eksik olan şey yalnızca “birkaç yeni feature” değil. Envanter **SHORT tarafını çok ayrıntılı**, LONG tarafını ise daha çok “event öncesi fırsat ve event sonrası dönüş” şeklinde ele alıyor. Ayrıca sorular hâlâ büyük ölçüde event merkezli; piyasanın event oluşmadan devam ettiği veya event’in başarısız olduğu yollar yeterince temsil edilmiyor. Mevcut 395 soruluk yapı bunun için çok güçlü bir temel. 

# En büyük yapısal eksik

Şu anda ana zincir şöyle:

```text
LONG başlangıcı
→ LONG olgunlaşması
→ BUY-fade event
→ SHORT düzeltme
→ yeniden LONG
```

Fakat gerçek piyasa ağacı şöyle olmalı:

```text
LONG genesis
       ↓
LONG expansion
       ↓
┌──────────────────────────────────┐
│ 1. Sağlıklı continuation         │
│ 2. Geçici exhaustion/pullback    │
│ 3. Distribution                  │
│ 4. Gerçek reversal               │
│ 5. Event oluşmadan trend ölümü   │
│ 6. Event oluşup fade başarısızlığı│
└──────────────────────────────────┘
       ↓
LONG devam / flat / scalp SHORT / swing SHORT
```

Yani temel soru yalnızca **“SHORT çalışıyor mu?”** olmamalı:

> Bu event sonrasında hangi piyasa dalına geçtik ve mevcut pozisyon durumuna göre en doğru aksiyon nedir?

Bu ayrım yapılmadan aynı event bazen LONG exit, bazen no-trade, bazen scalp SHORT, bazen swing SHORT, bazen de LONG continuation sinyali gibi davranabilir.

---

# 1. Pozisyon-durumuna göre karar ağacı eksik

Aynı sinyalin anlamı, sistemin event geldiğinde flat, LONG veya SHORT olmasına göre değişir.

Eklenmesi gereken sorular:

### Flat durumunda

396. Flat iken event sonrası en iyi aksiyon LONG, SHORT veya WAIT mi?

397. Flat iken yalnız LONG exit bilgisi taşıyan event’i trade etmeye çalışmak negatif expectancy mi yaratıyor?

398. Flat trader ile zaten LONG taşıyan trader için event’in ekonomik değeri farklı mı?

### Zaten LONG durumunda

399. Event yalnız risk azaltma sinyali mi?

400. LONG’un maliyet avantajı ve mevcut unrealized PnL exit kararını değiştirmeli mi?

401. Çok kârlı LONG’da partial exit, yeni açılmış LONG’da full exit daha mı doğru?

402. Event sonrası LONG kapatılmalı ama SHORT açılmamalı mı?

403. LONG exit edge’i ile reverse-to-SHORT edge’i arasındaki gerçek fark nedir?

### Zaten SHORT durumunda

404. Event yeni SHORT entry’den çok mevcut SHORT’u yönetme sinyali mi?

405. SHORT zaten event öncesi açılmışsa event size artırma, azaltma veya hiçbir şey yapmama sinyali mi?

406. Aynı sinyal flat girişte negatifken açık SHORT management’ında pozitif olabilir mi?

Bu bölüm çok kritik. Çünkü **entry alpha**, **position-management alpha** ve **risk information** birbirinden ayrı Knowledge Object olmalı.

---

# 2. LONG tarafında “başarısız fade / squeeze continuation” eksik

Mevcut sorular LONG’un başlangıcını ve event sonrası LONG’a dönüşü inceliyor. Fakat event geldiği hâlde piyasanın düşmeyip yukarı devam ettiği durum ayrı bir alpha olabilir.

Eklenmesi gereken blok:

## Failed Fade → LONG Continuation

407. BUY-fade event sonrasında fiyat event high üzerinde acceptance üretirse LONG continuation oluşuyor mu?

408. İlk 1–5 dakikada downside progress oluşmaması LONG sinyali mi?

409. Yüksek BUY aggression’a rağmen SHORT MFE oluşmaması squeeze riskini gösteriyor mu?

410. Event high reclaim değil, event high üzerinde hold daha güçlü LONG confirmation mı?

411. Event sonrası short sellers eklenirken fiyat düşmüyorsa trapped-short LONG fırsatı var mı?

412. OI artıyor ve fiyat event high üzerinde kalıyorsa squeeze continuation oluşuyor mu?

413. Failed fade sonrası ilk pullback LONG entry için kullanılabilir mi?

414. Failed fade LONG’unun expectancy’si doğrudan event öncesi LONG’dan daha mı yüksek?

415. Failed fade LONG route’u, negatif T0 SHORT trade’lerinin karşı tarafı mı?

416. T0 SHORT stoplarının bir bölümü aslında sistematik LONG alpha mı içeriyor?

Bu muhtemelen gözden kaçan en önemli LONG alanlarından biri. Negatif SHORT sonucu yalnızca “SHORT alpha yok” demeyebilir; bazı hücrelerde **karşı yönde continuation bilgisi** olabilir.

---

# 3. LONG tarafında SHORT kadar ayrıntılı management ve stop taxonomy yok

SHORT tarafında stop nedeni, re-entry, horizon, exit ve management ayrıntılı. LONG tarafı aynı simetriye sahip değil.

Eklenmesi gereken bölümler:

## LONG horizon

417. LONG genesis sonrası ekonomik hareket kaç dakika, saat veya gün sürüyor?

418. LONG MFE’nin ne kadarı BUY-fade event öncesinde oluşuyor?

419. LONG event gelmeden doğal olarak plato yapıyor mu?

420. LONG winner’larının ne kadarı event oluşmadan sona eriyor?

421. LONG horizon genesis tipine göre değişiyor mu?

422. Spot-led LONG daha uzun, perp-led LONG daha kısa mı sürüyor?

## LONG stop taxonomy

423. LONG stop seller continuation nedeniyle mi oluştu?

424. LONG stop doğru yön fakat erken giriş nedeniyle mi oluştu?

425. LONG stop yalnız liquidity sweep nedeniyle mi oluştu?

426. LONG stop BTC kaynaklı cross-market shock nedeniyle mi oluştu?

427. LONG stop sonrası reclaim gerçek zamanda ayırt edilebilir mi?

428. LONG stop sonrası aynı-yön re-entry churn mü, yoksa belirli genesis türlerinde alpha mı?

429. LONG’un WRONG_DIRECTION ve BAD_TIMING sınıfları bağımsız üretilebilir mi?

## LONG management

430. LONG’da event öncesi milestone lock gerekli mi?

431. LONG MFE giveback kuralı event bilgisinden daha güçlü mü?

432. Event sonrası partial exit + trailing remainder, full exit’ten daha mı iyi?

433. Higher-timeframe UP durumunda LONG tail pozisyonu korunmalı mı?

434. LONG’da time-stop hangi genesis türlerinde çalışıyor?

435. LONG ilk belirli süre içinde expansion üretmezse çıkılmalı mı?

Bu simetri kurulmazsa sistem istemeden SHORT araştırma motoru olarak kalır.

---

# 4. Event gerçekleşmeyen LONG’lar eksik

Pre-event LONG araştırmasının büyük tehlikesi şu:

Event’i görüp geçmişe doğru bakarak LONG başlangıcı bulursanız, yalnızca **gelecekte event oluşmuş LONG’ları** incelersiniz. Bu seçim yanlılığı yaratabilir.

Eklenmesi gereken sorular:

436. Aynı LONG genesis oluşup sonrasında BUY-fade event gelmeyen örneklerde sonuç nedir?

437. Event’e bağlanan LONG’lar bütün LONG genesis evrenini temsil ediyor mu?

438. Event oluşmayan sağlıklı LONG’lar daha büyük winner mı?

439. Event gerçekleşeceği bilgisi olmadan genesis gerçek zamanda seçilebilir mi?

440. LONG detector bütün timestamp’lerde çalıştırıldığında false-positive oranı nedir?

441. Event-conditioned LONG sonucu unconditional LONG sonucundan ne kadar farklı?

442. Gelecekte event oluşmasıyla koşullamak collider veya selection bias yaratıyor mu?

443. LONG fırsatı yalnız retrospectively event’e bağlandığı için iyi görünüyor olabilir mi?

Bu blok zorunlu. LONG başlangıcı araştırması mutlaka:

```text
Event-conditioned study
vs
Unconditional all-time LONG genesis study
```

şeklinde iki ayrı deney olmalı.

---

# 5. Cycle-level deduplication eksik

Aynı piyasa swing’i içinde birkaç BUY-side event oluşabilir. Bunları bağımsız örnek saymak pseudo-replication yaratabilir.

Ek sorular:

444. Bir LONG–SHORT cycle içinde ortalama kaç event oluşuyor?

445. Aynı swing içindeki ikinci ve üçüncü event bağımsız bilgi taşıyor mu?

446. Event-level N ile bağımsız cycle-level N arasındaki fark nedir?

447. Sonuç birkaç yoğun güne veya birkaç büyük swing’e mi bağımlı?

448. Aynı cycle içindeki event’lere cluster-robust değerlendirme uygulandığında sonuç korunuyor mu?

449. Gün, hafta ve market-cycle bazında effective sample size nedir?

450. İlk event mi, en büyük event mi, son event mi ekonomik olarak daha anlamlı?

451. Event tekrarları alpha mı, yoksa aynı bilginin tekrar sayılması mı?

Dashboard’da hem `event_N` hem `independent_cycle_N` gösterilmeli.

---

# 6. Post-event outcome taxonomy yeterince açık değil

Şu anda fade, continuation ve transition soruları farklı bölümlerde dağılmış durumda. Önce mutually exclusive outcome sınıfları oluşturulmalı.

Önerilen sınıflar:

```text
A. Immediate continuation up
B. Shallow pullback → continuation up
C. Deep pullback → full reclaim
D. Temporary fade → range
E. Clean short correction
F. Multi-hour short continuation
G. Full trend reversal
H. Two-stage decline
I. No-resolution / noise
```

Ek sorular:

452. Her event yalnızca bir nihai path sınıfına atanabilir mi?

453. Path sınıfı T+5, T+15, T+30’da ne kadar erken tahmin edilebilir?

454. Hangi path sınıfları birbirinden gerçek zamanda ayrıştırılamıyor?

455. T+30 silence aynı nihai sınıf içinde mi etkili, yoksa yalnız sınıf dağılımını mı değiştiriyor?

456. LONG exit, SHORT entry ve LONG re-entry farklı path sınıflarında mı çalışıyor?

457. Two-stage decline tek trade mi, iki ayrı trade mi olmalı?

458. Shallow pullback ile full reversal arasındaki en erken ayrım noktası nedir?

Bu taxonomy olmadan “ortalama event” üzerinde çalışmak birbirine zıt path’leri karıştırır.

---

# 7. Static regime yerine regime transition eksik

Envanter UP/DOWN/RANGE durumlarını soruyor ama asıl bilgi çoğu zaman rejimin kendisinden değil, **rejim değişiminden** gelir.

Ek sorular:

459. Event sırasında rejim UP’tan RANGE’e mi geçiyor?

460. RANGE’den DOWN’a transition SHORT için daha güçlü mü?

461. DOWN’dan RANGE’e transition SHORT exit sinyali mi?

462. Event öncesi volatility contraction’dan expansion’a geçiş var mı?

463. Trend-strength yalnız seviyesiyle değil türeviyle bilgi taşıyor mu?

464. 4h hâlâ UP etiketi taşırken 4h trend gücü hızla çöküyor olabilir mi?

465. Rejim etiketi gecikmeli olduğu için transition bilgisi kayboluyor mu?

466. State’in seviyesi mi, değişim yönü mü daha yüksek expectancy üretiyor?

Örneğin:

```text
4h_UP
```

tek başına yeterli olmayabilir.

```text
4h_UP fakat slope↓, participation↓, volatility↑
```

çok farklı bir state olabilir.

---

# 8. Absorption, exhaustion ve liquidity vacuum birbirinden ayrılmalı

Şu an price impact azalması ve aggression decay soruları var. Fakat üç farklı mekanizma karışabilir:

### Absorption

Agresif alımlar var, büyük pasif satıcı bunları emiyor.

### Exhaustion

Agresif alıcılar artık azalıyor.

### Liquidity vacuum

Satıcı güçlenmedi; yalnızca bid tarafı çekildiği için fiyat düşüyor.

Bunların trade anlamları farklıdır.

Ek sorular:

467. Event sonrası düşüş gerçek SELL aggression’dan mı, bid withdrawal’dan mı geliyor?

468. BUY aggression sürerken fiyat ilerlemiyorsa absorption var mı?

469. BUY aggression tamamen kesiliyorsa exhaustion mı?

470. Book depth kaybolduğu için oluşan düşüş daha hızlı fakat daha kısa mı?

471. Absorption sonrası SHORT, exhaustion sonrası SHORT’tan farklı horizon mı taşıyor?

472. Vacuum decline sonrasında V-reversal riski daha yüksek mi?

473. Passive seller refill rate tepe mekanizmasını ayırıyor mu?

474. Aynı silence etiketi farklı mikro-mekanizmaları bir araya getiriyor olabilir mi?

Silence tek başına “aktivite yok” diyebilir; fakat neden aktivite olmadığı ayrıştırılmalı.

---

# 9. Event geometry daha fazla geliştirilmelidir

Sadece eşik ve notional değil, event’in şekli önemlidir.

Ek feature/sorular:

475. Event tek büyük liquidation mı, çok sayıda küçük liquidation zinciri mi?

476. Event concentration sonucu değiştiriyor mu?

477. Cascade süresi ve inter-arrival time SHORT horizon’ını tahmin ediyor mu?

478. Event notional / traded volume oranı daha anlamlı mı?

479. Event notional / visible depth oranı impact’i açıklıyor mu?

480. Aynı notional farklı volatility ortamlarında aynı anlama mı geliyor?

481. Event’in price displacement per dollar değeri exhaustion’ı gösteriyor mu?

482. Event sırasında fiyatın ilerlememesi failed continuation bilgisi mi?

483. Event’in ilk yarısı ve ikinci yarısı arasında impact decay var mı?

484. Event threshold’ları sabit dolar yerine volatility/depth normalize edilmeli mi?

485. Event’in market-wide mi idiosyncratic mi olduğu path’i değiştiriyor mu?

Burada ham `50K/200K/500K` eşikleri yanında normalize edilmiş ölçekler kullanılmalı:

```text
liq_notional / 5m_volume
liq_notional / local_depth
liq_notional / rolling_liq_percentile
price_impact / aggressive_dollar
```

---

# 10. LONG–SHORT simetrisi değil, asimetrisi araştırılmalı

Her LONG sinyalinin SHORT aynası veya her BUY event’in SELL aynası aynı şekilde davranmaz. Kripto piyasasında yukarı ve aşağı hareketlerin mekanikleri farklı olabilir.

Ek sorular:

486. SELL-side karşı event aynı state machine’i üretiyor mu?

487. BUY-event sonrası SHORT ile SELL-event sonrası LONG yapısal olarak simetrik mi?

488. Short squeeze ve long liquidation dinamikleri farklı horizon mı oluşturuyor?

489. LONG genesis seller exhaustion’dan, SHORT genesis buyer exhaustion’dan aynı doğrulukla bulunabiliyor mu?

490. Funding ve leverage nedeniyle SHORT→LONG transition daha hızlı mı?

491. LONG→SHORT reversal daha yavaş ve distribution tabanlı mı?

492. Aynı feature’ın işareti çevrilince sonuç korunuyor mu?

493. Simetri varsayımı yanlışsa hangi modüller tamamen ayrı tutulmalı?

Burada “mirror test” başarısızlığı da değerli bilgi olmalı; sistemi zorla simetrik hâle getirmemeliyiz.

---

# 11. Counterfactual ve negatif kontroller eksik

Silence gerçek bilgi taşıyor gibi görünüyor; ancak event dışı kontroller daha da genişletilebilir.

Ek sorular:

494. Aynı saat ve rejimde random timestamp’lerde benzer silence ne ifade ediyor?

495. Benzer fiyat yükselişi olup liquidation event olmayan kontrollerde sonuç nedir?

496. Aynı notional event fakat farklı price response taşıyan kontrollerde sonuç nedir?

497. Event timestamp’i birkaç dakika ileri/geri kaydırıldığında edge korunuyor mu?

498. Placebo threshold’larda benzer sonuç çıkıyor mu?

499. BTC’de aynı mekanizma yoksa ETH-spesifik neden nedir?

500. Silence yalnız yüksek-momentum hareketlerden sonra doğal olarak mı oluşuyor?

501. Event olmasaydı fiyatın counterfactual path’i ne olurdu?

En az üç kontrol evreni gerekir:

```text
Event vs random-time control
Event vs matched-move-no-event control
Event vs same-event-different-response control
```

---

# 12. Endpoint PnL yerine path quality eksik

Median ve cumulative PnL tek başına yeterli değil. Trade’in içeride nasıl hareket ettiği de öğrenilmeli.

Ek sorular:

502. MFE’den önce MAE ne kadar oluşuyor?

503. Winner’lar önce stop bölgesine gidip sonra mı kazanıyor?

504. Time-to-positive ne kadar?

505. Time-under-water ne kadar?

506. İlk 5 dakikalık path nihai winner/loser’ı tahmin ediyor mu?

507. Winner ve loser path’leri ne zaman ayrışıyor?

508. Aynı final PnL’ye sahip trade’lerin risk path’i farklı mı?

509. SHORT için `MAE → MFE` sırası timing problemini gösteriyor mu?

510. LONG için smooth expansion ile violent recovery aynı kategoriye mi düşüyor?

511. Path entropy veya directional efficiency outcome’u tahmin ediyor mu?

Bu bilgiler stop, size ve entry zamanlamasını final PnL’den daha iyi geliştirebilir.

---

# 13. Entry–selection–exit katkısı ayrıştırılmalı

Bir route pozitif veya negatif çıktığında nedenini tam bilmiyoruz:

```text
Selection iyi / entry kötü
Selection kötü / exit iyi
Direction doğru / timing yanlış
Direction yanlış / management sonucu gizliyor
```

Ek sorular:

512. Aynı event selection üzerinde idealized entry sweep sonucu nedir?

513. Aynı entry üzerinde farklı selection’ların katkısı nedir?

514. Aynı selection ve entry üzerinde exit katkısı nedir?

515. Toplam expectancy’nin ne kadarı direction, timing ve management’tan geliyor?

516. Negatif T0 route doğru direction fakat kötü fill nedeniyle mi negatif?

517. T+30 entry selection bilgisini kullanıyor fakat ekonomik hareketi kaçırdığı için mi negatif?

518. LONG exit bilgisi güçlü olduğu hâlde SHORT entry neden zayıf?

519. Oracle direction ile executable direction arasındaki kayıp ne?

Her experiment raporunda şu attribution bulunmalı:

```text
Selection contribution
Timing contribution
Execution contribution
Management contribution
```

---

# 14. Action-value karşılaştırması eksik

Şu anda route’lar çoğunlukla bağımsız test ediliyor. Asıl karşılaştırılması gereken aynı anda mümkün olan aksiyonlar:

```text
HOLD LONG
EXIT LONG
PARTIAL EXIT LONG
REVERSE SHORT
WAIT
ENTER SHORT LATER
RE-ENTER LONG
```

Ek sorular:

520. Event geldiğinde en iyi aksiyonun conditional expected value’su nedir?

521. EXIT LONG ile REVERSE SHORT arasındaki incremental değer nedir?

522. WAIT, zayıf LONG ve SHORT işlemlerinin ikisini de yeniyor mu?

523. Partial exit’in değeri yalnız risk azaltımından mı geliyor?

524. Flat kalmanın opportunity cost’u nedir?

525. En iyi aksiyon mevcut pozisyon, rejim ve cycle phase’e göre nasıl değişiyor?

526. Sistem yalnız direction tahmin etmek yerine action ranking yapmalı mı?

Bu, AMI’yi predictor olmaktan çıkarıp **decision engine** hâline getirir.

---

# 15. Portfolio ve diğer alpha’larla ilişki eksik

Route tek başına çalışsa bile mevcut S34 alpha’larıyla aynı risk faktörüne maruz kalabilir.

Ek sorular:

527. Bu LONG/SHORT cycle diğer liquidation route’larla aynı event’leri mi trade ediyor?

528. Yeni LONG alpha mevcut ETH continuation alpha’sıyla duplicate mi?

529. Yeni SHORT state mevcut panic/continuation lane’leriyle korelasyonlu mu?

530. Aynı anda birden fazla route sinyal verirse hangisi capital priority almalı?

531. Event LONG pozisyonunu kapatırken başka sistem hâlâ LONG diyorsa ne yapılmalı?

532. Portfolio seviyesinde incremental Sharpe veya drawdown katkısı nedir?

533. Route tek başına pozitif olsa bile existing portfolio’ya incremental değer katıyor mu?

534. Aynı market-cycle riskinin iki kez alınması nasıl engellenecek?

---

# En önemli kavramsal değişiklik

Mevcut yapı:

```text
Sinyal → yön → entry → exit
```

yerine şu yapıya geçmeli:

```text
Cycle state
↓
Possible next paths
↓
Current position state
↓
Available actions
↓
Conditional action value
↓
Execution feasibility
↓
Risk-adjusted decision
```

Önerdiğim ana state machine:

```text
LONG_GENESIS
→ LONG_EXPANSION
→ LONG_MATURE
→ EXHAUSTION_CANDIDATE
→ EVENT_PENDING
→ EVENT_ACTIVE
→ POST_EVENT_UNRESOLVED
   ├─ FAILED_FADE_LONG
   ├─ SHALLOW_PULLBACK
   ├─ SHORT_SCALP
   ├─ SHORT_CONTINUATION
   ├─ FULL_REVERSAL
   └─ RANGE_NOISE
→ TRANSITION
   ├─ LONG_RECLAIM
   ├─ SHORT_RELOAD
   └─ WAIT
```

Her state için sistem şu üç soruyu cevaplamalı:

```text
1. Flat olsaydım ne yapardım?
2. LONG olsaydım ne yapardım?
3. SHORT olsaydım ne yapardım?
```

# Bence yeni en yüksek öncelik sırası

1. **Failed Fade → LONG Continuation**
   Negatif T0 SHORT örneklerinin karşı tarafında sistematik LONG alpha var mı?

2. **Event-conditioned olmayan LONG genesis**
   Gelecekte event oluşacağını bilmeden LONG başlangıcı gerçekten seçilebiliyor mu?

3. **Position-aware action study**
   Event flat trader için entry değil, açık LONG için exit alpha mı?

4. **Cycle-level deduplication**
   391 event gerçekte kaç bağımsız piyasa cycle’ı?

5. **Post-event path taxonomy**
   Temporary pullback, failed fade, clean correction ve full reversal ayrılmalı.

6. **LONG stop/horizon/management simetrisi**
   SHORT tarafındaki araştırma derinliği LONG tarafına taşınmalı.

7. **Absorption–exhaustion–vacuum ayrımı**
   Silence’ın altında hangi mikro-mekanizma var?

8. **Action-value comparison**
   HOLD LONG / EXIT / WAIT / SHORT arasından gerçekten hangisi üstün?

En büyük olası keşif şu olabilir:

> BUY-fade event’in kendisi doğrudan SHORT alpha olmayabilir. Fakat LONG’un riskini azaltan, failed fade durumunda LONG continuation’ı doğrulayan ve yalnız dar DOWN-state’lerde SHORT’a dönüşen bir **routing event’i** olabilir.

Bu okuma, şimdiye kadarki negatif SHORT sonuçlarıyla silence’ın gerçek bilgi taşıması bulgusunu aynı anda açıklayabilecek en güçlü üst hipotez.

---

# Appendix M — Source Research Expansion B (Original Turkish)

Dosya artık oldukça kapsamlı: position-aware aksiyonlar, failed-fade LONG, unconditional LONG genesis, cycle deduplication, path taxonomy, regime transition, mikro-mekanizma, event geometry, karşı-olgusal kontroller, path quality, attribution ve action-value katmanları zaten yerleştirilmiş. 

Kalan boşluklar daha çok **zamanın nasıl ölçüldüğü, sinyalin yaşlanması, scalp ile swing’in gerçekten ayrılması, hold sırasında yarışan riskler ve gerçek fill edilebilirlik** tarafında.

# 1. Signal aging — sinyal ne kadar süre canlı?

Şu anda T+1m, T+5m, T+30m gibi girişler var. Fakat bunlar sinyalin bilgi ömrünü doğrudan modellemiyor.

Eklenmeli:

### SIGNAL_AGE_AND_DECAY

535. Event bilgisinin half-life’ı kaç dakika?

536. Entry EV zamanla düzgün biçimde mi azalıyor, yoksa belirli kırılma noktalarında mı çöküyor?

537. Event’ten geçen süre mi, son price progress’ten geçen süre mi daha önemli?

538. Event sonrası ilk yeni high/low sinyal saatini sıfırlıyor mu?

539. Yeni aynı-yön liquidation gelmesi eski event’i yeniliyor mu, yoksa bağımsız yeni event mi yaratıyor?

540. Confirmation geç geldiyse sinyal doğrulanmış fakat ekonomik olarak ölmüş olabilir mi?

541. Her aksiyon için `first_tradable_ts` ve `last_tradable_ts` nedir?

542. LONG exit bilgisi, SHORT entry bilgisinden daha uzun ömürlü olabilir mi?

543. Failed-fade LONG confirmation’ı kaç dakika sonra stale hâle geliyor?

544. Sinyalin yaşı ile MFE, MAE ve time-to-positive nasıl değişiyor?

Her observer’da bulunmalı:

```text
signal_birth_ts
first_known_ts
first_executable_ts
last_valid_ts
signal_age_at_entry
time_since_last_progress
time_since_last_confirmation
```

Bu önemli, çünkü **geç doğrulanan doğru bilgi yine de kötü trade olabilir**.

---

# 2. Wall-clock yerine market-clock

Asia’da sakin 10 dakika ile US açılışındaki 10 dakika aynı ekonomik zaman değildir.

Eklenmeli:

### MARKET_TIME_NORMALIZATION

545. Hold süresi wall-clock, volume-time veya event-time ile mi ölçülmeli?

546. Son 10.000 işlem boyunca hold, sabit 10 dakikadan daha stabil mi?

547. Belirli traded-volume tamamlanana kadar hold daha genellenebilir mi?

548. Volatility-time kullanıldığında T+45 plato sonucu değişiyor mu?

549. Düşük aktivitede 30 dakika, yüksek aktivitede 5 dakikaya eşdeğer olabilir mi?

550. Silence gerçek zaman sessizliği mi, işlem sayısı sessizliği mi?

551. Session’lar arasındaki timing farkı yalnız aktivite hızından mı kaynaklanıyor?

Kaydedilecek alternatif clock’lar:

```text
wall_clock_age
trade_count_age
volume_age
realized_vol_age
liquidation_count_age
book_update_age
```

Bu katman özellikle scalp ve hold sonuçlarını ciddi biçimde değiştirebilir.

---

# 3. Scalp ile swing sadece farklı hold süresi değildir

En önemli eksiklerden biri bu.

Şu yanlış olabilir:

```text
Aynı entry
→ 15m tutarsak scalp
→ 4h tutarsak swing
```

Gerçekte scalp ve swing farklı route’lardır:

```text
SCALP
microstructure displacement
fast confirmation
small adverse tolerance
no-progress exit
high execution sensitivity
```

```text
SWING
structural state transition
higher-timeframe alignment
wider invalidation
slower confirmation
lower execution sensitivity
```

Eklenmeli:

### SCALP_SWING_ROUTE_SEPARATION

552. SHORT scalp ile SHORT swing aynı event population’ından mı doğuyor?

553. Scalp için gerekli confirmation, swing için gerekli confirmation’dan farklı mı?

554. Scalp winner’larının swing winner’larıyla overlap oranı nedir?

555. İlk düşüş scalp, ikinci düşüş swing route’u mu?

556. Scalp için event-high rejection yeterliyken swing için 1h structure break gerekli mi?

557. Higher-timeframe UP içinde SHORT scalp pozitif, SHORT swing negatif mi?

558. Higher-timeframe DOWN içinde scalp çıkışı büyük continuation’ı gereksiz yere kesiyor mu?

559. Scalp sinyali swing’e ne zaman upgrade edilebilir?

560. Swing sinyali ne zaman scalp’e downgrade edilmelidir?

561. Route başlangıçta mı belirlenmeli, yoksa trade sırasında mı sınıf değiştirmeli?

562. Scalp target spread, fee ve slippage sonrasında hâlâ yeterince büyük mü?

563. Scalp’te maksimum kabul edilebilir entry latency nedir?

564. Scalp yalnız tek giriş hakkına mı sahip olmalı?

565. Scalp stop sonrası aynı cycle içinde ikinci giriş tamamen yasaklanmalı mı?

Her route açıkça ayrı tutulmalı:

```text
SHORT_SCALP
SHORT_INTRADAY
SHORT_SWING
LONG_SCALP
LONG_INTRADAY
LONG_SWING
```

---

# 4. Hold problemi için “competing risks” modeli

Fixed hold grid’i tek başına yetersiz kalabilir. Trade açıkken birden fazla sonlandırıcı olay yarışır:

```text
TP
SL
event-high reclaim
opposite liquidation
BTC reversal
regime transition
no-progress
new independent event
funding/session boundary
```

Eklenmeli:

### COMPETING_RISK_HOLD_MODEL

566. Trade’i en sık hangi olay bitiriyor?

567. T+15 itibarıyla TP’den önce reclaim olma olasılığı nedir?

568. T+30 itibarıyla continuation hazard’ı ne kadar kalıyor?

569. İlk 10 dakikada progress yoksa sonraki winner olasılığı nedir?

570. Hızlı winner ve yavaş winner ayrı population mı?

571. Slow winner’ları korumak için tutmak, loser exposure’ını gereğinden fazla mı artırıyor?

572. Yeni event geldiğinde hold clock sıfırlanmalı mı?

573. Aynı cycle içindeki yeni event mevcut trade’i güçlendiriyor mu, bozuyor mu?

574. Event-high reclaim, fiyat stopa ulaşmadan ekonomik invalidation mı?

575. Opposite-flow arrival sabit hold’dan daha iyi exit üretiyor mu?

576. Hold kararı her dakika yeniden hesaplanan conditional EV ile mi yönetilmeli?

577. `P(TP before SL | current path)` kullanılabilir mi?

578. Path sınıfı değiştiğinde route otomatik değil, research düzeyinde yeniden etiketlenmeli mi?

Önerilen metrikler:

```text
hazard_tp
hazard_sl
hazard_reclaim
hazard_opposite_event
hazard_regime_change
survival_probability
remaining_expected_move
```

Bu yaklaşım “45 dakika mı, 60 dakika mı?” sorusundan daha güçlüdür.

---

# 5. Progress-adjusted hold ve no-progress exit

Bir trade yalnız zamana göre değil, beklenen hareketi üretip üretmediğine göre yönetilmeli.

Eklenmeli:

### PROGRESS_CONDITIONAL_MANAGEMENT

579. İlk 1m/3m/5m/10m minimum progress eşiği nedir?

580. SHORT event sonrası ilk 5 dakikada yeni low üretemiyorsa expectancy ne kadar düşüyor?

581. LONG reclaim sonrası ilk 5 dakikada acceptance oluşmazsa trade iptal edilmeli mi?

582. Price progress var fakat order-flow confirmation yoksa hold edilmeli mi?

583. Order-flow doğru fakat fiyat ilerlemiyorsa absorption mı, failed trade mi?

584. Beklenen MFE path’inin gerisinde kalan trade erken kapatılmalı mı?

585. Trade başlangıçta hızlı çalışıp sonra duruyorsa partial exit gerekli mi?

586. Trade önce ters gidip sonra çalışıyorsa kötü entry mi, normal path mi?

587. No-progress exit slow winner’ları ne kadar feda ediyor?

588. Progress eşiği volatility ve session’a göre normalize edilmeli mi?

Örnek:

```text
expected_progress_5m
actual_progress_5m
progress_ratio
time_since_new_extreme
distance_from_expected_path
```

---

# 6. Structural location — event nerede oldu?

Event geometry eklenmiş, fakat **event’in piyasa haritasındaki konumu** ayrı bir katman olmalı.

Aynı 500K event:

* range ortasında,
* previous-day high’da,
* weekly resistance’ta,
* breakout sonrasında,
* VWAP altında,
* low-volume node üzerinde

aynı anlamı taşımaz.

Eklenmeli:

### STRUCTURAL_LOCATION

589. Event previous-day high/low’a ne kadar yakın?

590. Event weekly high/low yakınında mı?

591. Event range ortasında olduğunda edge kayboluyor mu?

592. Event 4h resistance üzerinde acceptance sonrası mı oluştu?

593. Event breakout level’ın altında, üzerinde veya retest’inde mi?

594. Anchored VWAP uzaklığı outcome’u değiştiriyor mu?

595. Volume-profile HVN/LVN konumu fade sonucunu etkiliyor mu?

596. Event local liquidity pool tüketildikten sonra mı oluşuyor?

597. Event prior swing extension’ın kaç ATR ötesinde?

598. Trend channel’ın aşırı uzamış bölgesinde mi?

599. Location bilgisi event notional’dan daha mı değerli?

600. Failed fade LONG en çok resistance gerçekten kırıldığında mı oluşuyor?

Kaydedilecekler:

```text
distance_to_pdh
distance_to_pdl
distance_to_weekly_high
distance_to_weekly_low
distance_to_vwap
distance_to_anchored_vwap
range_percentile
distance_to_breakout_level
distance_to_volume_node
atr_extension
```

Bence bu, mevcut paketteki en önemli eksik feature ailesidir.

---

# 7. Entry türleri ayrı araştırılmalı

Şu anda entry zamanı ayrıntılı; fakat **entry mekanizması** da kritik.

Aynı sinyalde:

```text
market entry
limit pullback
breakdown stop-entry
retest entry
maker-only entry
```

çok farklı sonuç verebilir.

### ENTRY_MECHANICS

601. T0 market entry ile event-high retest limit entry arasında fark ne?

602. Breakdown market entry adverse selection üretiyor mu?

603. Breakdown sonrası retest beklemek fill frequency’yi çok mu düşürüyor?

604. Pullback limit entry’nin gerçek fill probability’si nedir?

605. İyi görünen limit entry yalnız kötü trade’lerde fill oluyor olabilir mi?

606. Partial fill sonucu nasıl etkiliyor?

607. Missed winner maliyeti nedir?

608. Maker fill beklerken sinyal yaşlanıyor mu?

609. Entry improvement, fill-rate kaybını ekonomik olarak karşılıyor mu?

610. Failed-fade LONG için breakout entry mi, retest entry mi daha iyi?

611. SHORT scalp için market entry zorunlu fakat swing için limit entry daha mı uygun?

612. Entry order’ın queue position tahmini mümkün mü?

Zorunlu rapor:

```text
eligible_N
attempted_N
filled_N
missed_N
partial_fill_N
fill_rate
fill_adverse_selection
missed_winner_cost
```

Scalp tarafında backtest alpha’nın kaybolacağı ilk yer burasıdır.

---

# 8. State reset ve overlapping events

Bir event’in outcome horizon’ı tamamlanmadan ikinci event oluşabilir. Bu durumda hangi event’in sonucu ölçülüyor?

### EVENT_OVERLAP_AND_STATE_RESET

613. Yeni event önceki event’in outcome’unu sansürlüyor mu?

614. Aynı yön event eski state’i güçlendiriyor mu, sıfırlıyor mu?

615. Karşı yön event mevcut path’i geçersiz mi kılıyor?

616. Overlapping horizon’larda PnL hangi event’e atanmalı?

617. Event stack sayısı outcome’u değiştiriyor mu?

618. İlk event ile son event arasında bilgi üstünlüğü var mı?

619. Bir cycle’da yalnız ilk tradable event mi kullanılmalı?

620. Event’ler belirli zaman aralığında tek meta-event olarak birleştirilmeli mi?

621. İkinci event geldiğinde mevcut trade yeni route’a mı dönüşüyor?

622. Yeni event sonrası hold clock sıfırlanırsa lookahead veya overfitting riski oluşuyor mu?

623. Opposing event sonrasında zorunlu WAIT state gerekli mi?

Bu bölüm cycle deduplication’dan farklıdır:

* **Deduplication:** örneklem bağımsızlığı.
* **State reset:** açık trade ve aktif hipotezin ne zaman geçersizleştiği.

---

# 9. Right-censoring ve sonuçların kirlenmesi

Uzun horizon araştırmasında bazı trade’ler gerçekten tamamlanmamış olabilir.

### CENSORING_AND_OUTCOME_INTEGRITY

624. Veri sonuna yakın event’ler doğru şekilde right-censored mı?

625. Data outage sırasında açık kalan trade outcome’u kullanılmalı mı?

626. Yeni bağımsız structural shock eski event outcome’unu geçersiz kılıyor mu?

627. Makro haber sonrası hareket hâlâ ilk event’e mi yazılıyor?

628. 1D–7D sonuçlarda başka event’ler sonucu domine ediyor mu?

629. Uzun horizon’da causal attribution ne kadar hızlı zayıflıyor?

630. Censored trade’leri kayıp/zero kabul etmek sonucu bozuyor mu?

631. Survival analysis ile fixed endpoint sonucu farklı mı?

632. LONG genesis ile BUY-fade event arasına başka büyük cycle girmiş olabilir mi?

Bu olmazsa özellikle multi-hour ve multi-day hold sonuçları yanıltıcı olabilir.

---

# 10. Position state içinde eksik olan: position age ve entry quality

Dosyada FLAT / ALREADY_LONG / ALREADY_SHORT var. Buna ek olarak mevcut pozisyonun yaşı ve maliyet kalitesi de gereklidir.

### POSITION_PATH_STATE

633. Yeni açılmış LONG ile 12 saattir taşınan LONG aynı şekilde yönetilmeli mi?

634. Pozisyon event’e çok yakın açılmışsa yanlış timing nedeniyle mi riskli?

635. Position cost basis event level’a göre nerede?

636. Unrealized PnL aynı olsa bile PnL path’i farklıysa aksiyon değişmeli mi?

637. Pozisyon daha önce büyük MFE verip geri çekilmişse exit önceliği artmalı mı?

638. Pozisyon hiç kâra geçmemişse event farklı mı yorumlanmalı?

639. Already-SHORT pozisyon event’ten önce mi, sonra mı açıldı?

640. Kötü SHORT entry event geldiğinde kurtarılmalı mı, kapatılmalı mı?

641. Partial exit sonrası kalan pozisyon yeni state mi sayılmalı?

642. Re-entry sonrası position age sıfırlanmalı mı?

Ek alanlar:

```text
position_age
entry_distance_from_event
current_pnl
max_pnl_since_entry
max_drawdown_since_entry
pnl_giveback
entry_quality_percentile
position_origin_route
```

---

# 11. Scheduled boundary risk

Hold sonuçları funding, günlük kapanış veya makro olay sınırlarını geçiyor olabilir.

### SCHEDULED_BOUNDARY_EFFECTS

643. Trade funding timestamp’ini geçtiğinde sonuç değişiyor mu?

644. Funding öncesi ve sonrası liquidation davranışı farklı mı?

645. UTC günlük kapanışı hold path’ini değiştiriyor mu?

646. Weekly close yakınındaki event’ler farklı mı?

647. US macro announcement çevresindeki event’ler ayrı population mı?

648. Session transition sırasında open SHORT taşımak riskli mi?

649. Weekend likidite koşulları scalp ve swing’i farklı etkiliyor mu?

650. Funding maliyetinden bağımsız olarak funding event’i state transition yaratıyor mu?

651. Belirli boundary öncesinde forced exit ekonomik mi?

Burada amaç haber tahmini değil; yapısal olarak farklı piyasa sınırlarını ayırmak.

---

# 12. Timestamp ve veri-latency bütünlüğü

Özellikle OI, funding, basis ve cross-exchange feature’lar gerçek zamanda aynı hızda gelmez.

### DATA_AVAILABILITY_LATENCY

652. Feature’ın event timestamp’i ile sisteme ulaşma timestamp’i aynı mı?

653. OI slope gerçekte kaç saniye/dakika gecikmeli biliniyor?

654. Funding/basis update stale olabilir mi?

655. Book, trades ve liquidation timestamp’leri aynı clock’a hizalı mı?

656. Exchange timestamp ile local receive timestamp farkı ne?

657. Out-of-order mesajlar düzeltiliyor mu?

658. Duplicate liquidation mesajları event geometry’yi şişiriyor mu?

659. Missing data gerçekten rastgele mı, yoksa yüksek stres anlarında mı eksiliyor?

660. Data-quality bozukken görünen alpha aslında feed artifact olabilir mi?

661. `feature_known_at` alanı bütün feature’larda mevcut mu?

Her feature:

```text
event_ts
exchange_ts
receive_ts
processed_ts
feature_known_at
staleness_ms
coverage_state
```

taşımalı.

Bu, lookahead dışında **pratik bilgi gecikmesini** de yakalar.

---

# 13. Venue ve cross-exchange yapı

Tek exchange event’i bütün market event’i olmayabilir.

### VENUE_PROPAGATION

662. Event Binance kaynaklı mı, market-wide mı?

663. Başka exchange liquidation’ları önce mi başladı?

664. Binance event’i lider mi, takipçi mi?

665. Cross-exchange price lead/lag yön sonucunu değiştiriyor mu?

666. Event tek venue’ye özgüyse mean-reversion ihtimali daha mı yüksek?

667. Market-wide cascade daha uzun continuation mı üretiyor?

668. Cross-exchange basis divergence failed-fade LONG’u tahmin ediyor mu?

669. Feed yalnız tek venue gösterdiği için cycle başlangıcı geç tespit ediliyor olabilir mi?

670. Venue-specific liquidity vacuum ile global exhaustion ayrılabilir mi?

Bu data henüz yoksa `BLOCKED_BY_CROSS_VENUE_DATA` olmalı.

---

# 14. Çoklu test ve discovery governance

Artık yüzlerce soru ve binlerce child test olacak. En büyük araştırma riski alpha’dan çok **multiple testing** hâline geliyor.

### MULTIPLE_TESTING_GOVERNANCE

671. Aynı soru ailesinde kaç varyant test edildi?

672. En iyi sonuç search-space büyüklüğüne göre düzeltiliyor mu?

673. Family-level false-discovery kontrolü var mı?

674. Tekil p-value yerine hierarchical evidence kullanılmalı mı?

675. Deflated Sharpe veya probability of backtest overfitting hesaplanıyor mu?

676. Threshold komşuluklarında sonuç stabil mi?

677. En iyi threshold yalnız tek noktada mı çalışıyor?

678. Route sonucu researcher degrees of freedom’e ne kadar duyarlı?

679. Yeni soru gerçekten yeni hipotez mi, eski sonucun yeniden paketlenmesi mi?

680. Minimum ekonomik etki istatistiksel anlamlılıktan önce frozen mı?

681. Her question family için ayrı experiment budget gerekli mi?

Zorunlu kayıt:

```text
variants_tested
effective_trials
family_id
family_adjusted_significance
threshold_stability
researcher_freedom_score
```

Bu kadar geniş bir AMI sistemi için bu katman zorunlu.

---

# 15. Edge lifecycle ve decay

Forward validation yalnız “tekrar etti mi?” dememeli; edge’in zaman içinde yaşayıp yaşamadığını ölçmeli.

### EDGE_LIFECYCLE

682. Rolling expectancy zamanla stabil mi?

683. Edge belirli bir tarihte başladı veya bitti mi?

684. Structural break noktaları var mı?

685. Son 30/60/90 gün eski döneme göre farklı mı?

686. Frequency sabitken edge magnitude düşüyor mu?

687. Edge magnitude sabitken frequency düşüyor mu?

688. Execution maliyeti arttığı için mi edge kayboluyor?

689. Route hangi koşulda dormant olmalı?

690. Hangi yeni veri geldikten sonra yeniden açılmalı?

691. Forward sonucu historical’dan kademeli mi, ani mi ayrışıyor?

692. Bir alpha’nın “ölü”, “uyuyan” ve “rejim dışı” durumları ayrılmalı mı?

Önerilen statüler:

```text
ACTIVE_EVIDENCE
DECAYING
DORMANT
REGIME_ABSENT
STRUCTURALLY_BROKEN
INSUFFICIENT_RECENT_SAMPLE
```

---

# 16. Confidence calibration ve abstention

Action ranking’in kendisi yeterli değil; tahminin güvenilirliği de ölçülmeli.

### DECISION_CONFIDENCE

693. SHORT olasılığı %70 denilen state’ler gerçekten yaklaşık %70 mi sonuçlanıyor?

694. Confidence ile ekonomik EV monoton mu?

695. LONG ve SHORT action değerleri birbirine çok yakınsa WAIT seçilmeli mi?

696. Model belirsizliği ile market belirsizliği ayrılabilir mi?

697. Hangi state’lerde sistem bilinçli olarak abstain etmeli?

698. WAIT’in value-of-information değeri nedir?

699. Bir confirmation daha beklemek EV’yi mi artırıyor, hareketi mi kaçırıyor?

700. Confidence düşük fakat payoff asimetrisi yüksekse trade edilmeli mi?

701. N küçük olduğunda action ranking shrink edilmeli mi?

702. Conflicting timeframe durumunda confidence otomatik düşmeli mi?

Bu, AMI’nin “her durumda cevap üretme” zorunluluğunu kaldırır.

---

# 17. Capacity ve notional sensitivity

Sizing’i live’a açmadan da araştırma seviyesinde alpha kapasitesi ölçülmeli.

### EXECUTION_CAPACITY

703. Sonuç 35 dolar, 1.000 dolar ve 10.000 dolar notional’da aynı mı?

704. Scalp alpha hangi notional’dan sonra slippage nedeniyle kayboluyor?

705. Book depth’e göre maximum executable size nedir?

706. Partial fill riski notional ile nasıl değişiyor?

707. Büyük notional entry event’in kendi path’ini etkileyebilir mi?

708. LONG ve SHORT kapasitesi asimetrik mi?

709. Thin session route’ları yalnız mikro-size alpha mı?

710. Route yüksek bps üretip düşük capacity nedeniyle ekonomik olarak sınırlı mı?

Bu bölüm:

```text
RESEARCH_CAPACITY_ONLY
```

olmalı; size permission üretmemeli.

---

# 18. Tail sequence ve streak riski

Tek trade tail riskine ek olarak kayıpların kümelenmesi incelenmeli.

### SEQUENCE_RISK

711. Loser’lar aynı regime veya birkaç gün içinde kümeleniyor mu?

712. Arka arkaya stoplar aynı yanlış state’in tekrar trade edilmesinden mi geliyor?

713. Cycle dedup sonrası losing streak uzuyor mu?

714. SHORT route birkaç büyük winner’a mı bağımlı?

715. LONG route birkaç squeeze winner’a mı bağımlı?

716. Top-k removed dışında bottom-k cluster analizi ne gösteriyor?

717. En kötü haftada action ranking nasıl davranıyor?

718. Regime değişiminde sistem eski aksiyonu tekrar tekrar seçiyor mu?

719. Cooldown trade-level değil state-failure-level olmalı mı?

720. Bir route son X trade yerine son X bağımsız cycle’a göre durdurulmalı mı?

---

# 19. Birbirine zıt sinyallerin çözümü

Portfolio interaction var, fakat tek cycle içindeki çelişkili evidence için açık bir hiyerarşi de gerekir.

### EVIDENCE_CONFLICT_RESOLUTION

721. Microstructure SHORT, higher-timeframe LONG diyorsa hangi aksiyon seçilmeli?

722. Direction ve horizon ayrı ayrı mı tahmin edilmeli?

723. SHORT scalp + LONG swing aynı anda doğru olabilir mi?

724. Mevcut LONG korunurken hedge benzeri kısa scalp teorik olarak ayrı mı değerlendirilmelidir?

725. BTC weakness ile ETH failed-fade LONG çatışırsa WAIT mi?

726. Price confirmation ile flow confirmation çatıştığında hangisi öncelikli?

727. Event geometry güçlü fakat structural location kötü ise trade iptal edilmeli mi?

728. Evidence conflict score oluşturulmalı mı?

729. Conflict arttıkça size değil, permitted action set daraltılmalı mı?

730. “Direction uncertain, volatility opportunity high” ayrı bir state mi?

Çok önemli fikir:

```text
direction
horizon
timing
```

tek tahmin olmamalı.

Örneğin:

```text
1h SHORT
24h LONG
```

aynı anda doğru olabilir.

---

# En kritik 8 ekleme

Bunların hepsini bir anda açmak yerine şu sırayı öneririm:

## P0 — Araştırma bütünlüğü

1. **Signal aging**
2. **Event overlap / state reset**
3. **Censoring**
4. **Data availability latency**

Bunlar düzelmeden bazı timing sonuçları teknik olarak yanlış yorumlanabilir.

## P1 — Ekonomik alpha

5. **Scalp–swing route separation**
6. **Competing-risk hold**
7. **Progress-adjusted management**
8. **Structural location**

Bence gerçek yeni alpha en çok burada çıkabilir.

## P2 — Gerçek uygulanabilirlik

9. Entry mechanics ve fill probability
10. Scheduled boundaries
11. Capacity
12. Venue propagation

## P3 — Epistemik güvenlik

13. Multiple-testing governance
14. Edge lifecycle
15. Confidence calibration
16. Sequence risk

# En önemli üç yeni üst hipotez

### Hipotez 1 — Timing problemi aslında “signal age” problemidir

T+30 entry kötü olduğu için değil, **T+30’da seçilen event’lerin çoğunda sinyal ekonomik olarak yaşlandığı** için negatif olabilir.

### Hipotez 2 — SHORT scalp ve SHORT continuation aynı alpha değildir

İlk 15–30 dakikalık düşüş mikrostructure displacement olabilir. Multi-hour continuation ise ancak structural location + regime transition + higher-timeframe alignment varsa oluşabilir.

### Hipotez 3 — Hold problemi sabit süreyle çözülemez

Doğru exit:

```text
time elapsed
+ progress achieved
+ reclaim hazard
+ opposing-flow arrival
+ regime transition
```

birleşimiyle belirlenebilir.

# En büyük potansiyel eksik keşif

Bence şu soruyu özellikle merkeze almalısınız:

> **Event’in nerede gerçekleştiği, event’in ne kadar büyük olduğundan daha önemli olabilir mi?**

Şu ana kadarki negatif unconditional SHORT sonuçları ile dar hücrelerde görülen olumlu davranışı açıklayabilecek en güçlü adaylardan biri bu:

```text
Büyük event + range ortası
→ noise / failed fade / continuation LONG

Daha küçük event + mature extension + structural resistance
→ clean SHORT scalp

Event + UP→RANGE transition
→ LONG exit

Event + RANGE→DOWN transition
→ SHORT continuation
```

Bu eklemelerden sonra sistem yalnız LONG/SHORT tahmini yapan bir araştırma katmanı değil; **sinyalin yaşını, piyasa konumunu, trade horizon’ını ve hangi olay gerçekleştiğinde fikrinden vazgeçmesi gerektiğini bilen bir cycle decision model** hâline gelir.

---

# Appendix N — Advanced Remaining-Gaps Register

The following families complete the evidence-safety and dynamic-decision layer that remains after the cycle, timing and execution extensions.

## N.1 Evidence contamination and holdout independence

- Record the dataset split on which every hypothesis was born.
- Mark all splits viewed before freezing the hypothesis.
- Prevent a hypothesis inspired by an untouched split from treating that split as independent confirmation.
- Track the amount of untouched-data budget remaining.
- Restrict result-informed historical discoveries to a forward-confirmation ceiling.

Required fields:

```text
hypothesis_birth_ts
hypothesis_origin_split
splits_seen_before_freeze
contaminated_splits
eligible_validation_splits
fresh_forward_required
maximum_evidence_ceiling
```

## N.2 Evidence dependency graph

- Quantify shared events, cycles, features and outcomes across claims.
- Replace raw claim counts with effective independent evidence counts.
- Prevent one underlying result from appearing as many independent Knowledge Objects.

## N.3 Causal structure and endogeneity

- Maintain a causal-assumption registry.
- Distinguish predictive association, mechanism consistency and causal support.
- Track confounders such as structural location, BTC movement, liquidity, OI, funding, session and shocks.
- Treat event-conditioned LONG studies as potentially selection-biased.

## N.4 Label uncertainty

- Store primary and secondary path labels.
- Preserve ambiguous cases.
- Record classification horizon and taxonomy version.
- Propagate label uncertainty into action confidence.

## N.5 State dwell time

- Model how transition risk depends on the duration of LONG_MATURE, RANGE, SHORT_SCALP and other states.
- Compare memoryless and duration-aware transition models.

## N.6 OOD and novelty

- Measure distance to historical support and nearest-cycle similarity.
- Separate novel regimes, novel execution environments and data-quality novelty.
- Default to abstention when support is weak.

## N.7 Uncertainty decomposition

```text
epistemic
market
data quality
execution
label
regime
```

The same total confidence can imply different actions depending on which uncertainty dominates.

## N.8 Hierarchical shrinkage

- Partially pool tiny cells toward parent populations.
- Use independent cycle N rather than event N as the primary support count.
- Compare asset-specific and partially pooled models.

## N.9 Dynamic policy evaluation

- Freeze sequential policies before validation.
- Prevent per-event hindsight-best action selection.
- Record decisions, available information and realized regret.
- Compare simple policies against complex policies.

## N.10 Cross-asset factor decomposition

- Separate market beta from asset-specific residual movement.
- Test whether ETH BUY-fade outcomes are BTC-driven, market-wide or idiosyncratic.
- Study dispersion and cross-asset liquidation intensity.

## N.11 Market-structure versioning

- Version exchange APIs, fee schedules, contract specifications, tick sizes and event definitions.
- Avoid comparing structurally incompatible historical periods without adjustment.

## N.12 Order-book resilience

- Research refill speed, cancellation intensity, queue persistence and depth recovery.
- Prefer neutral behavioural labels over unsupported manipulation claims.

## N.13 Exogenous-shock isolation

- Identify exchange outages, depegs, unexpected headlines, exploits and index anomalies.
- Separate normal-market and shock populations.
- Define recovery rules before normal state is restored.

## N.14 Derivatives surface

- Add futures curve, basis term structure, IV, skew, expiry and gamma context when data exist.
- Keep related questions blocked rather than approximating unavailable data.

## N.15 Benchmark and regret

- Compare every complex route against NO_TRADE, WAIT, simple trend baselines, fixed T45 and the current best route.
- Report complexity-adjusted and execution-adjusted incremental value.

## N.16 Non-ergodic capital path

- Measure expected log growth, drawdown duration, recovery time, loss clustering and ruin risk.
- Keep this research separate from sizing permission.

## N.17 Active data acquisition

- Rank data feeds by questions unblocked and expected information gain.
- Include implementation, storage and latency costs.
- Never enable new collectors automatically.

## N.18 Failure meta-analysis

- Study common rejection reasons and repeated failure patterns.
- Detect renamed old hypotheses.
- Extract anti-alpha and veto knowledge from rejected experiments.

## N.19 Asset and venue transferability

- Test transfer rather than assuming it.
- Use verdicts ASSET_SPECIFIC, TRANSFERABLE_WITH_RECALIBRATION, CROSS_ASSET_STABLE or FAILED_TRANSFER.

## N.20 Human research bias

- Log which results researchers viewed.
- Mark post-hoc, result-informed and independently replicated hypotheses.
- Record manual overrides and reviewer agreement.

---

# Appendix O — Canonical High-Level Question Families

The complete registry should preserve unique question IDs while organising them under the following high-level graph:

```text
Q001–Q024    Existing BUY-fade evidence and main structural thesis
Q025–Q085    Pre-event LONG genesis, maturity, exhaustion and LONG opportunity
Q086–Q147    Early, T0, delayed and post-event SHORT entries
Q148–Q175    Silence onset, maturity, breakdown and permitted use
Q176–Q218    SHORT horizon, exit and management
Q219–Q243    Stop taxonomy and SHORT re-entry
Q244–Q299    SHORT→LONG, LONG exit/re-entry and LONG→SHORT transitions
Q300–Q335    Multi-timeframe, regime and general opposite-liquidation management
Q336–Q395    Historical replication, forward validation, data gaps, execution and question governance
Q396–Q534    Position-aware action, failed-fade LONG, unconditional LONG, cycle/path/mechanism/action-value extensions
Q535–Q730    Signal aging, market clock, route separation, hold hazards, location, execution, overlap, censoring, latency, capacity and conflict
Q731–Q866    Evidence independence, causal safety, OOD, uncertainty, policy, market structure, meta-research and human bias
```

Every child question should inherit:

- cycle state;
- position state;
- timeframe;
- horizon;
- data requirements;
- independent cycle requirement;
- contamination status;
- and permission ceiling.

---

# Appendix P — Final Version 0.3 Declaration

AMI Version 0.3 defines an intelligence system that must know not only:

```text
what happened,
what may happen next,
and which action appears valuable,
```

but also:

```text
when the information became knowable,
how long it remains alive,
where the event occurred,
which market cycle it belongs to,
which position state is affected,
which evidence is independent,
which uncertainty dominates,
which action is executable,
and when the system should abstain.
```

The final governing principle is:

> AMI must earn the right not only to act, but also to believe that its evidence is independent, current, executable and applicable.

