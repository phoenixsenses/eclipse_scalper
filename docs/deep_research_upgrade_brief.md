# Eclipse Scalper Deep Research Brief

This document is a practical map of the current `eclipse_scalper` bot so you can plan major upgrades with less guesswork and faster iteration.

## 1) What the bot is (today)

Eclipse Scalper is a modular execution system for crypto scalping where alpha generation and execution reliability are intentionally separated.

- Alpha side: strategy signal generation and confidence scoring.
- Reliability side: order routing, reconcile, belief/debt posture, adaptive guards, and telemetry loops.
- Operational side: bootstrap/runner orchestration, data cache loops, and notification/reporting tools.

Core principle in current design: "bounded wrongness with fast recovery" instead of assuming perfect exchange consistency.

## 2) High-level architecture

Main control planes and files:

- Launch/orchestration
  - `eclipse_scalper/main.py`
  - `eclipse_scalper/bot/runner.py`
  - `eclipse_scalper/execution/bootstrap.py`
  - `eclipse_scalper/execution/bot_factory.py`

- Core runtime state + market loops
  - `eclipse_scalper/bot/core.py`
  - `eclipse_scalper/execution/data_loop.py`
  - `eclipse_scalper/execution/guardian.py`

- Entry/exit execution
  - `eclipse_scalper/execution/entry_loop.py`
  - `eclipse_scalper/execution/entry.py`
  - `eclipse_scalper/execution/exit.py`
  - `eclipse_scalper/execution/position_manager.py`

- Reliability and control
  - `eclipse_scalper/execution/reconcile.py`
  - `eclipse_scalper/execution/belief_controller.py`
  - `eclipse_scalper/execution/belief_evidence.py`
  - `eclipse_scalper/execution/reliability_gate_runtime.py`
  - `eclipse_scalper/execution/adaptive_guard.py`
  - `eclipse_scalper/execution/anomaly_guard.py`

- Router and state consistency
  - `eclipse_scalper/execution/order_router.py`
  - `eclipse_scalper/execution/replace_manager.py`
  - `eclipse_scalper/execution/state_machine.py`
  - `eclipse_scalper/execution/event_journal.py`
  - `eclipse_scalper/execution/intent_allocator.py`
  - `eclipse_scalper/execution/intent_ledger.py`

- Strategy
  - `eclipse_scalper/strategies/eclipse_scalper.py`
  - `eclipse_scalper/strategies/risk.py`

- Exchange adapters
  - `eclipse_scalper/exchanges/binance.py`
  - `eclipse_scalper/exchanges/coinbase.py`
  - `eclipse_scalper/exchanges/mock.py`

- Observability tooling (very extensive)
  - `eclipse_scalper/tools/*`
  - `eclipse_scalper/docs/observability_contract.md`
  - `eclipse_scalper/docs/telemetry_runbook.md`

## 3) Runtime flow (what happens in a live session)

1. `main.py` resolves mode/dry-run and starts runner.
2. `bot/runner.py` does startup guards (version checks, keys, mode, dry-run order block patching) and supervises loops.
3. `execution/bootstrap.py` wires cfg/state/exchange/data, optionally rebuilds state from exchange, and starts guardian/data/entry/exit manager loops.
4. Strategy (`strategies/eclipse_scalper.py`) computes directional signal + confidence with many dynamic gates.
5. Entry decision layer (`execution/entry_decision.py` from `entry_loop.py`) applies risk posture and commit checks.
6. Router (`execution/order_router.py`) submits/cancels/replaces orders with retry policy, idempotency handling, and telemetry.
7. Reconcile (`execution/reconcile.py`) compares local vs exchange truth, repairs drift, updates belief debt metrics.
8. Belief controller (`execution/belief_controller.py`) clamps entry risk knobs based on debt/growth/hysteresis.
9. Exit and protection paths remain available even when entry posture degrades.
10. Telemetry tools classify anomalies and can feed back into adaptive/recovery guards.

## 4) Strategy model today

Signal engine in `strategies/eclipse_scalper.py` is a large composite rule system:

- Momentum + trend + VWAP + volatility regime + structure filters.
- Optional session/time filters, data staleness checks, trend-confirm gates.
- Confidence score built from weighted votes and post-calibration.
- Debug/force-entry modes for plumbing tests.
- Audit CSV support for near-miss and signal diagnostics.

Strength: many knobs for market regime adaptation.
Risk: very high parameter surface area can create overfitting and hard-to-reason interactions.

## 5) Reliability model today

Reliability is not a side feature; it is a full subsystem:

- Reconcile tracks mismatch/debt and emits structured state events.
- Belief controller transforms debt to guard knobs (entries tighter, exits protected).
- Router supports idempotent cancel handling and bounded retry/replace behavior.
- Runtime/adaptive/anomaly guards can pause or tighten entries based on telemetry drift/spikes.

This is a major strength and already closer to production-grade execution discipline than typical retail bots.

## 6) What is strong already

- Clear separation between alpha, execution, and reliability layers.
- Extensive failure handling around order lifecycle ambiguity.
- Large test surface in `tools/test_*` for execution/risk/telemetry invariants.
- Rich telemetry/reporting scripts for both local and scheduled workflows.
- Multi-exchange abstraction already exists (`binance`, `coinbase`, `mock`).

## 7) Current bottlenecks to faster progress

1. Monolithic modules
- `entry_loop.py`, `order_router.py`, and `reconcile.py` are very large and carry many concerns.

2. Configuration sprawl
- High env-var count makes behavior powerful but hard to reason about globally.

3. Limited formal parameter governance
- Many knobs but no single canonical experiment registry tying knob changes to outcome metrics.

4. Possible architectural duplication
- Runner/bootstrap/core overlap can create lifecycle complexity and subtle race conditions.

5. Telemetry richness > decision synthesis
- You collect a lot of signals; fewer condensed "operator decisions" are generated automatically.

## 8) Big upgrade roadmap (high leverage)

### Phase A: Architecture hardening (fast win)

- Split `entry_loop.py` into:
  - opportunity scanner
  - risk/guard gate evaluator
  - order submission executor
  - post-submit handlers (partial fill/protection)
- Split `order_router.py` into:
  - validation + intent shaping
  - retry/error policy
  - exchange adapter executor
  - telemetry/journal hooks
- Keep behavior identical initially; refactor behind stable public functions.

Outcome: safer iteration velocity and smaller blast radius for new features.

### Phase B: Research operating system (big multiplier)

Add an "experiment contract":

- `experiment_id`, config hash, market regime snapshot, metrics snapshot.
- Every backtest/live micro run emits one standardized report row.
- Build comparison tool to rank experiments by robustness, not just raw pnl.

Outcome: upgrades become scientific instead of intuition-only.

### Phase C: Adaptive intelligence layer

- Convert current threshold-heavy confidence into hybrid model:
  - keep rule engine as base signal
  - add calibration model (online or rolling) to map signal context -> expected edge
- Add per-symbol regime memory:
  - trend/chop/vol clusters
  - best-performing strategy profile per cluster
- Add dynamic capital allocator across symbols using confidence reliability and recent fill quality.

Outcome: more stable edge, faster adaptation without manually retuning 100+ knobs.

### Phase D: Execution speed and fill quality

- Introduce latency budget tracking per stage (`signal -> decision -> order accepted`).
- Add micro-batching of market data transforms to reduce repeated indicator recalculation.
- Build venue-specific execution profiles (binance vs coinbase) with policy presets.
- Add slippage-aware order type switching (market/limit/post-only variants where supported).

Outcome: better realized edge and lower execution drag.

## 9) Concrete near-term upgrades (next 2-4 weeks)

1. Create `execution/contracts.py` and define typed dataclasses for:
- entry intent
- router request
- router result
- reconcile summary
- guard knobs snapshot

2. Add one canonical "run summary" artifact per session:
- net pnl, drawdown, win rate
- fill/slippage stats
- blocked-entry reasons histogram
- guard mode timeline summary

3. Build a minimal parameter registry:
- map each env var to owner module, default, safety impact, and test coverage.

4. Add fault-injection replay harness:
- replay selected telemetry slices through router/reconcile/guard pipelines to validate behavior under known incidents.

## 10) Research questions worth deep focus

- Which 10 parameters explain most variance in risk-adjusted pnl?
- Which guard events are predictive vs reactive?
- Does confidence calibration drift by symbol/regime/time-of-day?
- Where is realized edge lost: signal quality, entry timing, fill quality, or exit policy?
- How much of performance variance comes from execution quality vs strategy quality?

## 11) Suggested KPI stack for major upgrades

Primary:

- return/drawdown ratio
- max drawdown
- weekly stability of Sharpe-like metric
- execution slippage per symbol
- guard-trigger rate with recovery time

Reliability:

- mismatch debt minutes/day
- reconcile repair success ratio
- unknown-order resolution time
- protection coverage gap seconds

Speed:

- p95 signal-to-order latency
- p95 order submit-to-ack latency
- cycle runtime per loop

## 12) Where to start first

If your goal is fastest meaningful progress:

1. Refactor boundaries (without changing logic) in `entry_loop.py` and `order_router.py`.
2. Add experiment registry + run summary artifact.
3. Use your existing telemetry tooling to rank top failure/drag contributors.
4. Only then start alpha-model upgrades.

This sequence prevents "new strategy on unstable execution" and gives you compounding engineering speed.
