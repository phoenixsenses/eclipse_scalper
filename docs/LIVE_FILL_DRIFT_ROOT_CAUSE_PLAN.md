# Live Fill Drift Root-Cause Plan

## Objective

Find why live/paper fills diverge from replay/backtest expectations, then close that gap without changing validated signal logic.

## Scope

- Execution realism only: latency, queue position, adverse selection, data linkage.
- No changes to pocket filters, confidence gates, or regime signal logic.
- Use existing artifacts and tools first, then calibrate.

## Root-Cause Categories

1. Timestamp/linkage mismatch
- Sim and live events fail to match due to time base, symbol/side normalization, or join window.

2. Latency modeling drift
- Real feed/order latency regime differs from current model.

3. Queue/hazard miscalibration
- Queue depletion/join dynamics mismatch causes fill-rate/adverse drift.

4. Evidence quality issues
- Too few matched samples, noisy days, or missing artifacts.

## Standard Daily Runbook

1. Run calibration + diagnostics + root-cause in one command:
```powershell
python -m tools.daily_execution_calibration --symbol ETHUSDT --days 14 --run-root-cause 1
```

2. Optional standalone root-cause rerun:
```powershell
python -m tools.live_fill_drift_root_cause --run-pipeline
```

3. Canary expansion gate (7-day rule):
```powershell
python -m tools.evaluate_canary_expansion_gate --report-dir reports/daily --window-days 7 --max-top-score 0.5
```

4. Single-command daily runner (prints `CANARY_EXPANSION=GO|HOLD`):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_daily_canary_gate.ps1 -Symbol ETHUSDT -Days 14 -WindowDays 7 -MaxTopScore 0.5
```

3. Review outputs:
- `reports/REPLAY_PARITY_REPORT.md`
- `reports/EXECUTION_HEALTH.md`
- `reports/TOXICITY_REPORT.md`
- `reports/POST_ROLLOUT_AUDIT.md`
- `reports/LIVE_FILL_DRIFT_ROOT_CAUSE.md`
- `reports/daily/YYYY-MM-DD_LIVE_FILL_DRIFT_ROOT_CAUSE.md`
- `reports/CANARY_EXPANSION_GATE.md`

## Weekly Decision Gates

1. Fill-rate prediction error improvement >= 25% vs baseline.
2. Adverse-selection MAE improvement >= 20% vs baseline.
3. No contract/FSM violations.
4. P95 fill delay stays below operational target.

If any gate fails:
- Keep `EXEC_*` flags in canary mode.
- Refit latency/queue parameters with last 7 days.
- Re-run daily root-cause analysis for 3 consecutive days before expanding rollout.

## Phase Plan

### Phase 1 - Measure (Week 1)

- Freeze metric contract and artifact paths.
- Ensure daily pipeline writes all required reports.
- Start root-cause ranking from measured artifacts, not assumptions.

Exit:
- Daily reports generated for 5/5 trading days.
- No missing artifact errors.

### Phase 2 - Calibrate (Week 2-3)

- Refit latency distributions by session.
- Refit queue/hazard parameters by regime/time bucket.
- Track drift trend daily.

Exit:
- Fill-rate error trend improving.
- Adverse drift trend improving.

### Phase 3 - Validate (Week 4)

- Run parity + diagnostics in canary mode (`EXEC_*` flags on for selected symbol only).
- Compare against control path.

Exit:
- All weekly decision gates pass for 7 consecutive days.

### Phase 4 - Rollout (Week 5-6)

- Expand symbol set gradually.
- Keep rollback toggle ready.
- Audit post-rollout every day.

Exit:
- No critical incidents and no material drift regressions.

## Failure Triage Order

1. Check `REPLAY_PARITY_REPORT` match coverage.
2. Check `EXECUTION_HEALTH` latency/timing drift.
3. Check `TOXICITY_REPORT` and adverse deltas.
4. Check `POST_ROLLOUT_AUDIT` fail flags.
5. Only then tune parameters.

## Notes

- This workflow is execution-only and safe to run while paper trading continues.
- It does not mutate live strategy logic.
