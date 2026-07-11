# AMI — Artificial Market Intelligence
## Whitepaper, Scientific Constitution and Canonical Engineering Specification

**Document:** `AMI_WHITEPAPER_AND_SYSTEM_SPECIFICATION_v0.2.md`  
**Version:** `0.2.0`  
**Status:** `FOUNDATIONAL WHITEPAPER / LIVING SPECIFICATION`  
**Initial research laboratory:** S34 liquidation and cascade intelligence  
**Long-term scope:** General-purpose, multi-market Artificial Market Intelligence  
**Date:** 2026-07-02

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

# Appendix H — Applied Change Log (Living Document Patches)

> Appendix G protokolü uyarınca uygulanan değişiklikler. Detaylı mühendislik logu:
> `docs/ami/AMI_CHANGELOG.md`. Bu bölüm whitepaper'ın kendi soy kütüğüdür.

## PATCH-0001

```yaml
change_id: AMI-CHG-0001
date: 2026-07-02
author: claude (operatör onaylı)
section_changed: Part XVII Faz 0-5 — spesifikasyondan çalışan koda
type: ADD
reason: Appendix F build brief'in ilk implementasyonu
new_evidence: reports/research/s34/AMI_PHASE_VALIDATION.md (gerçek veri, Definition-of-Done 10/10)
affected_knowledge: 9 S34 Knowledge Object tohumlandı (K-S34-*), 12 mezarlık kaydı, 10 backlog sorusu
status_change: whitepaper 0.2.0 -> 0.2.1 (spec değişmedi; implementasyon durumu eklendi)
implementation_change: >
  ami/ paketi (constitution, enums, knowledge graph + audit + failure archive,
  epistemic governor, multi-TF state engine, structure transition matrix,
  trade lifecycle engine, research OS + prereg freeze, decision trace,
  seed_s34, run_phase_checks). Testler 17/17. Store'lar: data/ami/*.
validation_required: E-MECHCOMP-FWD-001 forward kanıt birikimi (dondurulmuş)
```

Faz durumu (2026-07-02): **Faz 0-5 ✅ inşa edildi ve doğrulandı; Faz 6-9 başlanmadı**
(ayrıntı: `docs/ami/AMI_ROADMAP.md`).

Uygulama sırasında spesifikasyona eklenen ampirik notlar:
- Part XVI §87 dersleri Knowledge Object olarak kodlandı; `bk_refill` REGIME_LIMITED,
  `bk_pull` HOLDOUT_VALIDATED statüsüyle tohumlandı (§87.3 doğrulaması).
- §9 veri-kalite durumları gerçek veride işletildi: `vol_state` STALE örneği canlı yakalandı;
  OI/spot toplaması `data/oi_spot_poller.py` ile yeniden başlatıldı (§87.7).

## PATCH-0002

```yaml
change_id: AMI-CHG-0002
date: 2026-07-02
author: claude (operatör onaylı Paket 1-3)
section_changed: Part XV (validation/forward), Part X (Research OS pratiği)
type: ADD
reason: Automated Forward Evidence Pipeline + Adversarial Suite + ilk preregistered deney
new_evidence: >
  AMI_FORWARD_EVIDENCE.md (2 frozen binding aktif), AMI_MUTATION_REPORT.md (20/20 ihlal
  yakalandı), AMI_MFE50_EXPERIMENT.md (FALSIFIES — dürüst null sonuç).
affected_knowledge: >
  K-S34-HOUR17-001 + K-S34-MECH-COMPOSITE-001 forward-validating akışa bağlandı;
  MFE50 tek-feature ayrıştırma hipotezi Failure Archive'a (NO_EDGE, retry: state-transition
  dizileri + yeni prereg).
status_change: whitepaper 0.2.1 -> 0.2.2; hiçbir bilgi terfi etmedi, hiçbir operasyonel izin verilmedi
implementation_change: >
  ami/research/forward_pipeline.py (R1 freeze-sonrası, R2 versiyon/hash değişiminde
  BINDING_INVALID, R3 trade-başına tek evidence, R4 provenance zorunlu, R5 pipeline izin
  vermez, R6 live dokunulmaz); ami/mutation_suite.py 20 senaryo; E-MFE50-001 prereg.
validation_required: forward birikim n>=20/binding; MFE50 retry ancak YENİ prereg ile
```

Ampirik not (Part VIII §44'e): +50bps milestone'unda mevcut 10 tek-feature'ın hiçbiri
continuation'ları korurken giveback'i ayıramadı (TRAIN kısıtı cc>=0.85 sağlanamadı);
LOCK_ALL/EXIT_ALL kontrolleri de baseline'ın altında (1205/585 vs 1313). MFE State
Classifier'ın v1'i state-TRANSITION dizilerine ihtiyaç duyuyor — tek-anlık feature yetmiyor.

## PATCH-0003

```yaml
change_id: AMI-CHG-0003
date: 2026-07-02
author: claude (operatör onaylı Faz 6A)
section_changed: Part XI §64 (Latent State Engine) — spesifikasyondan ilk koşuma
type: ADD
reason: Faz 6A research-only latent state discovery tamamlandı
new_evidence: AMI_PHASE6A_LATENT.md — E-LATENT6A-001 (frozen prereg), 15/15 mutation
affected_knowledge: Failure Archive += NO_STABLE_STATE (kronolojik occupancy kayması)
status_change: whitepaper 0.2.2 -> 0.2.3; hiçbir latent çıktıya operasyonel izin verilmedi
implementation_change: ami/latent/* — CUSUM + seeded k-means + numpy-HMM; outcome-leakage yapısal engelli
validation_required: retry yalnız YENİ prereg ile (daha uzun veri / rejim-koşullu)
```

Ampirik not (§64'e): 5m grid üzerinde k=4 latent state seed/perturbasyon açısından güçlü
stabil (ARI 0.85/0.99) çıktı fakat kronolojik validasyonda occupancy 0.14×-4.99× kaydı —
"latent state'ler var ama REJİME BAĞLI" bulgusu. LS-003 (keskin satış: ret1h −1.9σ, %90
downtrend) validasyon rallisinde neredeyse kayboldu; LS-004 (fiyat-düşüşsüz SELL-stres,
5.9σ) 5× büyüdü. Whitepaper §4.3 "duration matters" ve §22 rejim-bağımlılığı öngörüsüyle
tutarlı: state taksonomisi rejim-koşullu tanımlanmalı. Sonraki latent denemesi bu yüzden
daha uzun tarih VE rejim-katmanlı kabul kriteri ile preregister edilmeli.

## PATCH-0004

```yaml
change_id: AMI-CHG-0004
date: 2026-07-03
author: claude (operatör onaylı Faz 6A-R)
section_changed: Part XI §64-65 + Part XII §12 (temporal decay/drift) — ilk drift altyapısı
type: ADD
reason: 6A kronolojik kırılmasının attribution'ı + rejim-koşullu latent test
new_evidence: AMI_PHASE6AR_REGIME.md (PASS), AMI_DRIFT_MONITOR.md (UNUSABLE-canlı), 14/14 mutation
affected_knowledge: K-LATENT-REGIME-001 (trend=UP scoped, max SHADOW)
status_change: whitepaper 0.2.3 -> 0.2.4; operasyonel izin YOK; Faz 6B kapalı
implementation_change: RegimeDefiner + DriftMonitor (STABLE/WARNING/SHIFTED/UNUSABLE; öneri-only)
validation_required: trend=UP bandının forward izlemi; mekanizma adlandırması ayrı prereg ister
```

Ampirik notlar:
- 6A occupancy kırılması VERİ değil PİYASA kaymasıydı: rv/stress/buyliq/spread/trades
  PSI 0.16-7.4, missingness ~sabit; trend karışımı DEĞİŞMEDİ → kayma likidasyon/vol
  yapısında. §12 "regime-dependent decay" öngörüsü ilk kez ölçülerek doğrulandı.
- Latent state'ler rejim İÇİNDE yaşıyor: yalnız trend=UP walk-forward persistent.
  State taksonomisi rejim-scoped tanımlanmalı (§4.3/§22 ile tutarlı).
- Latent+rejim kombinasyonu risk (mdd −1363→−416) daraltıyor ama alpha eklemiyor
  (top-winner bağımlı) → §5.5 "risk intelligence ≠ alpha intelligence" ayrımı pratikte.

## PATCH-0005

```yaml
change_id: AMI-CHG-0005
date: 2026-07-03
author: claude (operatör onaylı Faz 6A-R2)
section_changed: Part XI §64-65 (latent) + §5.5 (risk vs alpha intelligence) — risk/applicability doğrulaması
type: ADD
reason: "regime+latent risk azaltıyor" hipotezi frekans-normalize kontrollerle test edildi
new_evidence: AMI_PHASE6AR2_RISK.md (FALSIFIES/INSUFFICIENT_SAMPLE), 13/13 mutation (toplam 79)
affected_knowledge: Failure Archive += riskapp overlay (INSUFFICIENT_SAMPLE, retry=forward>=6ay yeni prereg)
status_change: whitepaper 0.2.4 -> 0.2.5; operasyonel izin YOK; Faz 6B kapalı
implementation_change: matched-count blocked bootstrap + random-veto + regime-only kontrol
  çerçevesi ve 13 yapısal guard (ami/latent/risk_applicability.py) — gelecek risk deneyleri
  için yeniden kullanılabilir
validation_required: aday overlay'in forward'da seçim üretip üretmediği izlenmeli
```

Ampirik notlar:
- §5.5'in "risk intelligence ≠ alpha intelligence" ayrımına üçüncü katman eklendi:
  **görünür risk azalması ≠ risk intelligence.** 6A-R'deki mdd −1363→−416 daralması
  per-fold dürüst refit altında YENİDEN ÜRETİLEMEDİ; kaynak ALL-era-fit artifact +
  hipotez-penceresi + düşük frekansın mekanik etkisiydi. Frekans-normalize kontrol
  (eşit-N matched-count/random-veto) olmadan MDD karşılaştırması yapısal olarak yasaklandı.
- Rejim kayması altında latent overlay'in birincil kırılma modu yön hatası değil
  **applicability çöküşü**: aday katman validation erasında seçim üretemiyor
  (n_cand 9→1/2/1). Drift'in operasyonel maliyeti önce seçim-kapasitesinde görünüyor.
- Satüre alarm dersi (§12 drift): referans-pencere drift monitörü rejim kaymasından sonra
  SÜREKLİ UNUSABLE veriyor (13/13, fp-suspension 0.69) — koruyucu applicability sinyali
  için referans penceresinin adaptif/rolling olması gerekir; sabit-referans alarm
  "her şey değişti"den fazlasını söyleyemiyor.

## PATCH-0006

```yaml
change_id: AMI-CHG-0006
date: 2026-07-03
author: claude (operatör onaylı C-BUY-FADE + 8A paketi)
section_changed: Part X (Research OS pratiği) + §16 lookahead disiplini — route-yapısal şablon
type: ADD
reason: mevcut bir shadow route'un yapısal yolunun (genesis/timing/horizon/TF-bağlam/
  silence/management/re-entry) tam preregistered incelemesi — şablon niteliğinde
new_evidence: BUYFADE_STRUCTURAL.md + BUYFADE_REENTRY.md (FALSIFIES ×2; silence-info istisnası)
affected_knowledge: K-BUYFADE-SILENCE-INFO-001; failure archive += 5
status_change: whitepaper 0.2.5 -> 0.2.6; operasyonel izin YOK; live/shadow route değişmedi
implementation_change: yeniden kullanılabilir guard seti (15) + 24 mutation; NOT_AVAILABLE
  disiplini (OI/basis/L2 proxy'siz raporlandı)
validation_required: silence-info forward izlemi; yeni prereg adayları DR-0006'da
```

Ampirik notlar:
- **Shadow-N küçükken route istatistiği genellemez:** N=26 +2.8bps shadow sonucu, 391-event
  tarihsel replay'de −9.5bps'e döndü. Route promotion'ları için tarihsel replay tabanı zorunlu.
- **Silence lookahead dersi İKİNCİ KEZ doğrulandı (BUY tarafında):** T+30m'de bilinen bilgi
  T0 filtresine sızarsa +26bps'lik hayali edge oluşuyor; kademeli-bilinir versiyonları
  (s1m..s10m) gerçek ama zayıf ve delayed-entry ile yakalanamıyor. SELL-side saga (§26-27
  oturum kayıtları) ile birlikte bu artık YAPISAL kural: "cascade-sonrası sessizlik" ailesi
  bilgi taşır ama giriş-anında bilinemez — kullanım alanı geç-aşama yönetim/risk.
- **Re-entry churn hipotezi (H-RE-NULL) ilk ölçümde doğrulandı:** aynı yapısal cycle içinde
  ikinci giriş, fee+adverse-selection sonrası tüm cooldown'larda negatif; random-timing
  kontrolünden ayırt edilemez. Tek istisna adayı stop-taksonomisi BAD_TIMING alt-sınıfı.

## PATCH-0007

```yaml
change_id: AMI-CHG-0007
date: 2026-07-03
author: claude (operatör onaylı silence-exit paketi)
section_changed: §16 lookahead disiplini + Part X — survivor-bias-safe exit-timing şablonu
type: ADD
reason: T+30m'de bilinir hale gelen bilginin (silence) YÖNETİM katmanında dürüst testi
new_evidence: BUYFADE_SILENCE_EXIT.md (REJECTED[econ] + T45_EXIT_ROBUST)
affected_knowledge: failure archive += silence-exit overlay; silence-info KO kapsamı netleşti
status_change: whitepaper 0.2.6 -> 0.2.7; izin YOK; route değişmedi
implementation_change: survivor-universe + pre-T30-use + breakdown-causal + manage-closed +
  realized-only + fee-extension + route-mutation guard'ları (yeniden kullanılabilir)
validation_required: bd_first_buy50 forward gözlemcisi (operatör onaylı, observation-only)
```

Ampirik notlar:
- **Geç-bilinir bilginin yönetim değeri sınırlı çıktı:** silence T+30m'de doğrulandığında
  hareketin ~%90'ı tamamlanmış (T30 medyan unrealized +22bps; T30→45 katkı +0.6..+5.4).
  "Bilgi gerçek ama geç" durumunda doğru soru giriş değil çıkıştı — cevap: mevcut T+45
  zaten robust.
- **Frozen econ eşiği görevini yaptı:** 9 kriterden 8'ini geçen aday (bd_first_buy50,
  üç split'te pozitif incremental) val-econ'da (+1.37<3bps) dürüstçe reddedildi; ayrıca
  noisy kontrolünde de çalışması "silence-koşullu" iddiasını çürüttü — kontrol setlerinin
  değeri: mekanizma etiketi yanlışsa PASS bile yanlış bilgi üretirdi.
- **Üçüncü bağımsız teyit:** cascade-sonrası sessizlik ailesi GİRİŞ sinyali değildir
  (T+30 observer entry −13/−16). SELL-side (§26-27) + BUY-side struct + bu deney.
