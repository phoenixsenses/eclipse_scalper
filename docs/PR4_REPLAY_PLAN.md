# PR-4 Replay Fidelity Plan

## Goal
- Add deterministic replay parity tooling to compare simulated fills with paper/live fills.
- Keep existing strategy logic unchanged; this is diagnostics and calibration infrastructure.

## Scope
- `src/microphys/replay/replayer.py`
- `tools/replay_parity_report.py`
- `tests/replay/test_replay_determinism.py`

## Build Steps
1. Normalize simulated rows (jsonl/json) and live rows (SQLite `trades`) into one canonical schema.
2. Deterministically sort by `(symbol, side, ts, event_id)` to remove input-order effects.
3. Match simulated vs live rows using nearest timestamp within `match_window_sec`.
4. Compute parity metrics:
   - `match_rate_vs_sim`
   - `fill_rate_delta`
   - `mean_abs_dt_sec`
   - `mean_fill_delay_delta_sec`
   - `mean_pnl_bps_delta`
   - `mean_adverse_bps_delta`
5. Emit JSON + Markdown report for calibration workflow.

## Acceptance
- Replay results are deterministic under shuffled input ordering.
- Tool runs from CLI and writes both outputs under `reports/`.
- Tests cover both direct in-memory comparison and file-based loading.

## Verification
- `python -m py_compile src/microphys/replay/replayer.py tools/replay_parity_report.py`
- `pytest -q tests/replay/test_replay_determinism.py`

