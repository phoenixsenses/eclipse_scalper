# Market Microstructure Physics Roadmap

## Current State (Repo-grounded)
- Deterministic replay/eval pipeline exists: `tools/replay_strategy.py`, `tools/eval_run.py`, `tools/sweep_eval.py`, `tools/walkforward_eval.py`, `tools/walkforward_sweep.py`.
- Execution simulation layer exists: `execution/sim/min_exec_sim.py`, `execution/sim/price_oracle.py`.
- Reliability and ops controls exist: `execution/health_gate.py`, `execution/alpha_gate.py`, `tools/health_check.py`, `tools/ops_smoke.py`, watchdog and PID identity work.
- Strategy adapters exist: baseline and `micro_edge_pocket`.
- Missing from physics stack: explicit state reconstruction artifact, impact/propagator layer, and regime-segmented physical diagnostics tied to decisions/fills.

## Target State
- Every research run emits a deterministic state trajectory artifact:
  - `state_vector.jsonl`
  - `S(t) = [order_flow_imbalance, spread_proxy, liquidity_pressure, trade_rate, vol_proxy]`
- Strategy and execution metrics can be attributed to market-state transitions.
- Impact and propagation costs become first-class metrics in eval/sweep/walk-forward outputs.

## Phased Build Plan

### Phase 2A (Now): State Reconstruction MVP
- Add `tools/state_reconstruct.py`.
- Build deterministic state vectors from existing replay events/tables:
  - `agg_trades`, `mark_prices`, `liquidations`.
- Emit `state_vector.jsonl` for each eval run.
- Add tests for deterministic and schema-light behavior.

### Phase 2B: State Feature Expansion
- Extend state vector with:
  - short/medium horizon return gradients,
  - liquidation pressure imbalance,
  - short-term volatility clustering.
- Add per-symbol diagnostics summary in `metrics.json` and `summary.md`.

### Phase 3: Physics Signal Modules
- Add modular signals under `tools/strategies/physics_*`:
  - order-flow momentum,
  - liquidity vacuum,
  - intensity shock,
  - spread compression.
- Route through replay adapter interface with explainable decision metadata.

### Phase 4: Impact + Propagation MVP
- Add `execution/sim/impact.py`:
  - square-root impact estimate,
  - simple propagator decay model.
- Report `impact_cost_sum`, `impact_half_life_proxy`, `impact_dominates_count`.

### Phase 5: Simulator Realism Pass
- Add queue-position proxy and adverse-fill decomposition hooks.
- Persist per-fill attribution tags for post-trade diagnostics.

### Phase 6: Regime-Conditioned Physics Evaluation
- Segment state/signal/impact metrics by regime in walk-forward runs.
- Add regime-conditioned stability score and alpha gate inputs.

### Phase 7: Live Runtime State Engine
- Stream state reconstruction online from collector DB/cache.
- Emit live state heartbeat and degradation reasons for entry gating.

### Phase 8: Self-Improving Research Loop
- Automated feature search on state vectors.
- Stability-first selection and promotion (`runs/latest` and `last_good`) with strict gates.

## File-Level Patch Map (Immediate)
- Add: `tools/state_reconstruct.py`
- Update: `tools/eval_run.py` (write `state_vector.jsonl`)
- Add: `tests/test_state_reconstruct.py`
- Update: `tests/test_eval_run_outputs.py` (new artifact assertion)

## Verification Commands
```powershell
python -m py_compile tools\state_reconstruct.py tools\eval_run.py tests\test_state_reconstruct.py tests\test_eval_run_outputs.py
pytest -q tests\test_state_reconstruct.py tests\test_eval_run_outputs.py
python -m tools.eval_run --db data/microstructure.db --symbols BTCUSDT --start 2026-03-01T00:00:00Z --end 2026-03-01T00:10:00Z --strategy baseline --run-dir runs\eval\physics_mvp
```

## Local Verification Checklist
- `state_vector.jsonl` exists and is non-empty.
- Same slice/config emits byte-identical `state_vector.jsonl`.
- `summary.md` lists `state_vector.jsonl` in artifacts.
