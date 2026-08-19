# E-DER Hypothesis Ledger

Statuses describe scientific state, not trading permission. The frozen 25 events are exploratory/post-hoc for hypotheses formulated after Channel A.

## H1 — Channel A deterioration rule

- **Status:** `REJECTED / FAILED AS FROZEN ORDERING`
- **Question:** Does renewed post-entry SELL liquidation identify thesis deterioration?
- **Null:** The frozen ordering does not separate subsequent outcomes in the predicted direction.
- **Alternative:** The frozen ordering separates deterioration as preregistered.
- **Required data:** Frozen Channel A snapshots and unchanged fixed-boundary outcomes.
- **Falsification:** Observed ordering failed and sometimes reversed.
- **Forbidden tuning:** Reverse or retune the rule after seeing returns.
- **Historical role:** Completed failed diagnostic; preserved.
- **Forward role:** None unless a new independently motivated preregistration is authorized.

## H2 — Post-entry flow-impact compression

- **Status:** `EXPLORATORY / POST-HOC HISTORICAL`
- **Question:** Does increasing adverse forced-liquidation pressure become less price-effective after entry?
- **Null:** Conditional downside response does not weaken as observed adverse pressure rises.
- **Alternative:** Conditional downside response weakens as observed adverse pressure rises.
- **Required data:** Track A pressure/OHLCV proxies historically; Track B aggressive flow and quote/L2 data prospectively.
- **Falsification:** No compression, reversed response, or loss of pattern across admissible measurement variants.
- **Forbidden tuning:** Choose windows, clocks, pressure definitions, horizons, states, or thresholds using E-DER returns.
- **Historical role:** Descriptive proxy-only, after the active contract is reviewed/frozen.
- **Forward role:** Prospective rich-microstructure confirmation.

## H3 — Liquidation-pressure-proxy Flow Surprise

- **Status:** `ACTIVE, PROXY ONLY`
- **Question:** Is observed forceOrder pressure unusual relative to an outcome-blind prior expectation?
- **Null:** Observed pressure is consistent with the frozen prior expectation.
- **Alternative:** Observed pressure is unusually high or low under that expectation.
- **Required data:** forceOrder E/T, p, original q, deterministic quality status and strictly prior calibration.
- **Falsification:** No genuinely prior reference, outage-intersecting support, or material definition/clock reversal.
- **Forbidden tuning:** Choose p*q, q, count, clock, window, model, or cutoff by +240m return.
- **Historical role:** Track A measurement primitive only.
- **Forward role:** Legacy-compatible view alongside richer flow measurements.

## H4 — OHLCV Impact Surprise

- **Status:** `ACTIVE, PROXY ONLY`
- **Question:** Is one-minute-OHLCV price response unusual relative to a genuinely chronological generic expectation?
- **Null:** Actual OPEN-to-OPEN response is consistent with prior-OOS expectation.
- **Alternative:** Actual response is an unusually positive or negative residual.
- **Required data:** Exact one-minute OHLCV, frozen benchmark target/features/folds and strictly prior OOS residual calibration.
- **Falsification:** Benchmark is not genuine prior OOS, required bars are missing, or admissible measurement variants reverse the anomaly.
- **Forbidden tuning:** Refit/select model, target, horizon, percentile, feature or state by E-DER outcomes.
- **Historical role:** Track A conditional response anomaly; not quote-mid impact.
- **Forward role:** Legacy-compatible comparator to quote-mid response.

## H5 — Aggressive-flow Flow Surprise

- **Historical status:** `UNSUPPORTED`
- **Forward status:** `PROSPECTIVE`
- **Question:** Is signed aggressive event-symbol flow unexpected after E-DER entry?
- **Required data:** Raw/aggregate trade identifiers, side semantics, exchange and receive clocks, candidate-symbol coverage.
- **Falsification:** No surprise after a frozen strictly prior null or instability across valid feed views.
- **Forbidden tuning:** Return-selected flow definitions or horizons.
- **Historical role:** None for frozen 25.
- **Forward role:** Track B after collector validation.

## H6 — Displayed-book resilience

- **Historical status:** `UNSUPPORTED`
- **Forward status:** `PROSPECTIVE`
- **Question:** Does sequence-valid displayed liquidity remain stable or recover during adverse flow?
- **Required data:** Level-A L2, valid reconstruction, spread/depth state and timing provenance.
- **Falsification:** Spread expands, bid state weakens, or reconstruction quality fails during the specified state.
- **Forbidden tuning:** Outcome-selected levels, windows, score weights, or resilience thresholds.
- **Historical role:** None for frozen 25.
- **Forward role:** Track B mechanism evidence, not participant/causal identification.

## H7 — Exact replenishment

- **Historical status:** `NOT IDENTIFIABLE`
- **Forward status:** `REQUIRES LEVEL-A DATA`
- **Question:** Is removed displayed bid quantity replaced under sequence-valid observation?
- **Required data:** Incremental L2, trades, continuity, explicit operational definition.
- **Falsification:** No qualifying re-addition under the frozen definition or invalid book sequence.
- **Forbidden tuning:** Call sampled L1 ratios or generic quantity increases exact replenishment.
- **Historical role:** None.
- **Forward role:** Potential displayed-book process only; removal cause may remain unidentified.

## H8 — Hidden-liquidity absorption

- **Historical status:** `NOT IDENTIFIABLE`
- **Forward note:** Displayed Level-A L2 alone does not necessarily identify hidden liquidity.
- **Question:** Does non-displayed liquidity explain low adverse price response?
- **Required data:** Additional execution/venue evidence capable of distinguishing hidden from displayed and alternative mechanisms.
- **Falsification:** A uniquely discriminating design rejects hidden-liquidity implications; public displayed data alone is insufficient.
- **Forbidden tuning:** Infer hidden liquidity from weak price movement alone.
- **Historical role:** Mechanism-compatible story only, never a finding.
- **Forward role:** Remains potentially not identifiable even with Track B core data.

## H9 — Dynamic exit

- **Status:** `STOP / NOT AUTHORIZED`
- **Question:** None active.
- **Forbidden tuning:** Any exit simulation, TP/SL design, threshold optimization or use of proxy states as trading rules.
- **Historical role:** Prior exit work is research history only.
- **Forward role:** Requires a separate future preregistered authorization after mechanism and measurement gates.

## H10 — Measurement Sensitivity Contract V1

- **Status:** `ACTIVE STAGE GATE`
- **Question:** Are Track A conclusions stable across semantically admissible, outcome-blind measurement choices?
- **Null:** Substantive interpretation is not robust across admissible variants or required provenance fails.
- **Alternative:** The descriptive anomaly is measurement-robust within the frozen contract.
- **Required data:** Audit-supported forceOrder and OHLCV, benchmark provenance, quality flags, frozen event manifest.
- **Falsification:** Any contract stop condition.
- **Forbidden tuning:** Any selection using E-DER return, subset profitability, win rate or future information.
- **Historical role:** Must be independently reviewed and frozen before execution.
- **Forward role:** Supplies legacy-emulation comparison rules.
