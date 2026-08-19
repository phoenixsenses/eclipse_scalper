# E-DER Measurement Sensitivity Contract V1

**Date:** 2026-08-19  
**Status:** `DRAFT FOR INDEPENDENT REVIEW — NOT YET FROZEN FOR EXECUTION`  
**Scope:** Historical Track A specification only  
**Authority:** Data Feasibility Audit V1 and frozen 25-event manifest

This document freezes outcome-blind measurement choices before the next historical descriptive proxy diagnostic. It does not execute an analysis, fit a model, select alpha, repair matching, or authorize an exit.

## A. Estimand

The estimand is the time-ordered conditional one-minute-OHLCV response of frozen E-DER events under an observed forced-liquidation pressure proxy. The primary mechanism question is whether higher observed adverse SELL pressure is accompanied by less adverse conditional OHLCV response. This is a response-anomaly estimand, not true executed-flow impact, quote-mid impact, absorption, causality, or a trading signal.

The independent statistical unit is the E-DER event. Multiple minutes and horizons within an event are dependent repeated measurements and must never inflate N beyond 25 events.

## B. Observable proxy

Historical forceOrder storage retains exchange event time `E` as `ts_ms`, order trade time `o.T` as `trade_time_ms`, symbol, side, order price `o.p`, original quantity `o.q`, and `notional=o.p*o.q`. It does not retain `ap`, fill quantities `l/z`, status `X`, raw payload, or receive time.

Every liquidation-derived quantity is therefore an `OBSERVED FORCED-LIQUIDATION PRESSURE PROXY`. It is not complete executed liquidation volume, true forced SELL volume, or an exact liquidation tape. `q_parent` and `q_echo` are normalized pressure proxies, not executed-liquidation shares.

Historical price response is a `ONE-MINUTE-OHLCV RESPONSE / IMPACT SURPRISE PROXY`. It is not quote-mid impact.

## C. Primary measurement

### C1. Pressure

The continuity-preserving primary pressure representation is the side-specific window sum of `p*q`, because repository provenance verifies that frozen E-DER `running_notional`, `q_parent`, and `q_echo` use stored `notional=p*q`. Normalization, where required by the preserved specification, uses exact trailing one-minute kline quote volume and must be named explicitly.

### C2. Clock

The primary historical forceOrder clock is `E=ts_ms`, matching the frozen anchor construction and historical research bucketing. This preserves continuity; it is not a claim that E is the economically superior clock.

### C3. Response

The benchmark-compatible primary target is exact log return in bps from `OPEN(t)` to `OPEN(t+h)`. The existing exporter proves this target in its frozen preregistration and implementation. Output names must include `OHLCV_PROXY` or equally explicit language.

### C4. Frozen horizons

For generic expected-response and residual measurements, preserve the existing artifact grid: `1, 5, 15, 30, 60` minutes. These are not newly selected. Interpret them as:

- 1/5m: immediate OHLCV response;
- 15/30m: digestion/transition response;
- 60m: longer response path.

The historical Channel A snapshot grid `5, 15, 30, 60, 120, 180` is separate provenance and must not be silently merged with the generic benchmark grid. No new horizon may be added in this contract.

## D. Admissible sensitivity variants

Predeclared pressure variants are:

1. `sum(p*q)` — primary;
2. `sum(q)` — original-quantity sensitivity;
3. observed message count — message-process sensitivity.

They are conceptually distinct proxies and none is true executed volume. `sum(q)` must not be pooled across symbols without an explicit unit-safe normalization. Message count must not be interpreted as number of liquidated accounts or taker orders.

The mandatory clock sensitivity is separate bucketing by `o.T`. E and T outputs remain parallel. No optimized hybrid is admissible. Receive-time sensitivity is `NOT IDENTIFIABLE HISTORICALLY`.

Admissible OHLCV sensitivities, only if timestamped without leakage, are:

- OPEN-to-OPEN log response — primary benchmark-compatible target;
- close-based response under an explicitly stated completed-bar timing rule;
- completed-bar adverse/favorable high/low range, named as range sensitivity rather than impact.

No OHLC4 or close value may be silently substituted for a quote mid. If material conclusions differ among semantically admissible pressure, clock, or response representations, report the disagreement; never select the most profitable variant.

## E. Invalid or forbidden variants

The following are invalid under Track A:

- treating `p*q`, `q`, or message count as executed liquidation volume;
- manufacturing event-symbol aggressor flow, quotes, spread, depth, OFI, MLOFI, replenishment, cancellations, or hidden liquidity from candles;
- E/T hybrids selected after outcome inspection;
- receive-time reconstruction;
- return-selected windows, horizons, normalizations, thresholds, labels, scores, or data-quality weights;
- composite “resilience scores”;
- multiplying Impact Surprise by a reliability/data-quality weight;
- using future bars, fixed +240m outcome, win rate, subset profitability, or Impact Surprise association to define a state;
- converting a descriptive percentile into a p-value or trading threshold.

## F. Source-quality rules

Every required pressure-support interval receives exactly one source-quality status:

- `VERIFIED OUTAGE`: a documented zero-row/outage interval intersects support; the pressure measurement is invalid for primary inference and retained only as a flagged missing/invalid record.
- `VERIFIED COMPLETE SUPPORT`: a source-specific receipt directly proves the required support under a defined completeness contract.
- `NO KNOWN OUTAGE BUT COMPLETENESS NOT PROVEN`: no verified outage intersects support, but forceOrder sparsity and incomplete gap logging prevent a completeness claim.

Zero forceOrder messages alone do not prove an outage. Absence from the legacy gap registry does not prove completeness. No outcome-tuned quality score is permitted.

Audit V1 identified verified liquidation outages on 2026-04-28..06-05 and 2026-07-06 10:06:39Z..07-10 11:24:38Z, including zero-row days 07-07..09. Later execution must use the exact audit evidence, not infer new outages from silence.

## G. Clock rules

For every event/window and every E-versus-T comparison, later execution must report:

- number and fraction of messages changing minute bucket;
- pressure difference by required window;
- event-level measurement disagreement;
- whether any descriptive state/label direction changes.

E and T are not interchangeable: Audit V1 found 1,722,630 of 1,722,645 keeper rows differ, with observed E−T differences of 1..10,825 ms. Clock choice must never be selected by downstream result quality.

## H. Response definitions and alignment

The frozen event timing remains:

- `base=floor(anchor/60s)*60s+60s`;
- entry=`base+31m` exact OPEN;
- fixed boundary=`base+240m` exact OPEN;
- event path=`base+0..240m` inclusive;
- an observation’s later outcome, if separately evaluated, begins at the next exact OPEN.

Every response must state observation OPEN, target OPEN, horizon, and whether the target stays inside the frozen boundary. No uncompleted boundary candle high/low is admissible. The +240m outcome remains firewalled from measurement construction and selection.

## I. Existing generic OOS benchmark reuse

The only presently provenance-supported benchmark contract is:

- source contract: `reports/research/s34/S35_E_DER_RAW_RESEARCH_DATASET_EXPORT_PREREG_V1.md` plus its time-axis erratum;
- implementation: `tools/research_s35_e_der_raw_dataset_export.py`;
- universe: all symbols in `xsec_klines.db` with exact eligible feature/label bars;
- grid: UTC-aligned 15-minute panel;
- horizons: 1/5/15/30/60m;
- target: OPEN(t)→OPEN(t+h) log return in bps;
- model: standardized linear ridge, unpenalized intercept, fixed alpha=1.0, no model comparison;
- features known through t−1: own 5m and 30m log returns, own 30m realized volatility, log1p trailing 5m quote volume, BTC 30m return and realized volatility, trailing 5m signed and absolute observed-liquidation pressure divided by quote volume;
- chronology: UTC Monday-aligned seven-day test folds; expanding earlier training data; label end strictly before fold start; minimum 7 days and 10,000 rows;
- contamination: same-symbol panel rows overlapping `[anchor−60m, fixed+60m]` excluded from training and residual calibration;
- residual: actual minus prediction;
- calibration: same symbol/horizon, genuine OOS residuals with `label_end_ms < observation_open_ms`; minimum 20 prior rows;
- artifacts: 3,534,890 generic OOS rows, 30,125 event-path rows and 55 horizon-fold fits, recorded in the sealed raw-export receipt.

No random CV or in-sample residual may be called Impact Surprise. No benchmark refit or model substitution is authorized here. Any property not reproducibly contained in the frozen contract, implementation, model-fit metadata, OOS artifact, and receipt is `UNKNOWN` rather than an invitation to reconstruct a favorable benchmark.

Purge/overlap note: contamination exclusion and strict label-end chronology are verified in the contract/code. No additional general dependence purge or embargo beyond these recorded rules is claimed; that property is `UNKNOWN / NOT ESTABLISHED` for broader overlapping panel labels.

## J. Common-market adjustment

Exact BTC one-minute paths exist for all 25 events. Preserve the existing BTC feature adjustment described above and any separately established BTC/common-market OHLCV response adjustment only when its exact formula/provenance is named. Do not optimize beta or control construction against E-DER outcomes. Alternative market controls require a new preregistered decision.

## K. Required reporting format

Future Track A output must report separately:

1. event identity;
2. observed liquidation-pressure proxy;
3. pressure measurement variant;
4. clock basis;
5. source-quality status;
6. OHLCV response and exact alignment;
7. generic expected response;
8. `OHLCV_PROXY_IMPACT_SURPRISE = actual − predicted`;
9. strictly prior-OOS descriptive percentile;
10. BTC/common-market-adjusted response, with provenance;
11. measurement sensitivity/disagreement;
12. mechanism interpretation limit.

Residual percentiles are descriptive, not p-values. Existing 90th/95th percentile observations are exploratory/post-hoc and cannot become frozen trading thresholds from the 25 events.

## L. Multiplicity and dependence

All inference and summaries use event-level aggregation or dependence-aware procedures. Never treat event-minute or event-horizon rows as independent. Report event counts, medians and tail sensitivity. If multiple frozen comparisons are formally tested, state the family and use a prespecified correction such as Holm; do not enlarge the family after seeing results.

## M. Stop and downgrade conditions

Later execution must stop the affected measurement or downgrade interpretation when:

- a required pressure interval intersects a `VERIFIED OUTAGE`;
- a benchmark prediction is not genuinely prior chronological OOS;
- event identity differs from the frozen manifest;
- a required OHLCV bar is missing;
- admissible variants reverse the substantive response anomaly;
- E/T disagreement is material and resolution would require result-driven choice;
- required metric provenance cannot be reproduced;
- output naming would imply quote-mid, aggressive-flow, LOB, executed-volume, absorption, or causal identification.

A failed contract may not be “solved” by inventing a profitable feature.

## N. Forbidden tuning and claim language

Do not select or modify any E-DER threshold, entry, exit, horizon, feature threshold, pressure representation, clock, matching distance, model, calibration, state, or label based on E-DER returns. Do not inspect +240m outcomes to decide which variant is useful.

Historical Track A may not say: absorption confirmed, seller exhaustion confirmed, market makers absorbed liquidation, order book replenished, hidden liquidity caused rebound, aggressive-flow surprise, true liquidation volume, OFI/MLOFI state, or quote-mid impact.

Preferred language: observed forced-liquidation pressure proxy; one-minute-OHLCV response anomaly; conditional response anomaly; measurement-sensitive; measurement-robust; compatible with; not supported; not identifiable.

## Matching boundary

Matching is not repaired or rerun by this contract. Current state: 25/25 `POOR_OVERLAP`; liquidation coordinate MAD=0 made current normalization invalid on that coordinate. This does not establish that economic controls do not exist. Matching supplies no evidence for or against E-DER until support and normalization are repaired outcome-blind under a separate authorization.

## Contract activation

This V1 draft becomes executable only after independent review records a dated freeze decision. Until then, the historical Measurement Sensitivity analysis and proxy diagnostic remain `STOP`. Dynamic exits remain `STOP / NOT AUTHORIZED` regardless of contract review.
