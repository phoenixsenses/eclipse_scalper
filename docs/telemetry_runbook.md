# Telemetry Runbook

This runbook covers day-to-day telemetry workflow checks, RED_LOCK smoke validation, and fast incident triage for `eclipse_scalper`.

## Preconditions

- GitHub CLI is installed and authenticated:
  - `gh auth status`
- Repo target:
  - `phoenixsenses/eclipse_scalper`
- Workflow:
  - `.github/workflows/telemetry-dashboard.yml`

## Trigger And Watch A Manual Run

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml -R phoenixsenses/eclipse_scalper
gh run list -R phoenixsenses/eclipse_scalper --workflow .github/workflows/telemetry-dashboard.yml --limit 1
gh run watch -R phoenixsenses/eclipse_scalper <run_id>
```

For a single-run chained smoke (escalation + reset with built-in assertions), trigger:

```bash
gh workflow run .github/workflows/telemetry-smoke.yml -R phoenixsenses/eclipse_scalper
```

## Local Safe Profile Preflight

Use the safe launcher preflight before any live run:

```powershell
$env:RUN_PREFLIGHT_ONLY = "1"
.\run-bot-ps2.ps1
```

Optional local smoke assertion before startup:

```powershell
$env:RUN_SMOKE = "1"
.\run-bot-ps2.ps1
```

Preflight blocks startup when key constraints are unsafe (symbol list, leverage/notional caps, margin mode, loss limits, correlation config, reliability coupling thresholds).

## RED_LOCK Escalation Smoke

Use this to validate that notifier state escalates to `critical` when the RED_LOCK streak reaches threshold.

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_red_lock_event=true \
  -f simulate_red_lock_seed_streak=1 \
  -f expected_notifier_level=critical \
  -f expected_recovery_stage=RED_LOCK \
  -f expected_red_lock_streak=2
```

Expected in workflow logs:

- `Run telemetry dashboard notifier` includes `Notify decision: send reason=level_transition:normal->critical`
- `Inspect notifier state` includes:
  - `level=critical`
  - `recovery_stage_latest=RED_LOCK`
  - `recovery_red_lock_streak=2` (with default threshold `2`)

## Recovery Reset Smoke

Use this to validate streak reset when recovery stage is not `RED_LOCK`.

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_recovery_stage_override=POST_RED_WARMUP \
  -f simulate_red_lock_seed_streak=2 \
  -f expected_notifier_level=normal \
  -f expected_recovery_stage=POST_RED_WARMUP \
  -f expected_red_lock_streak=0
```

Expected in workflow logs:

- `Inspect notifier state` includes:
  - `recovery_stage_latest=POST_RED_WARMUP`
  - `recovery_red_lock_streak=0`
- Notifier decision is usually `normal_unchanged` unless other signals degrade state.
- `Assert smoke notifier expectations` step must pass.

## Fast Incident Triage

If workflow fails:

1. Identify failed step:
   - `gh run view -R phoenixsenses/eclipse_scalper <run_id> --json jobs`
2. Pull failing logs:
   - `gh run view -R phoenixsenses/eclipse_scalper <run_id> --log-failed`
3. Classify issue:
   - Dependency/runtime mismatch in workflow
   - Script argument mismatch
   - Missing optional artifacts/tools
   - Telemetry input empty/stale
4. Fix with minimal blast radius:
   - Prefer workflow-step change over broad dependency changes.
   - Keep notifier state persistence and `Inspect notifier state` intact.
5. Re-run workflow and confirm `conclusion=success`.

## Operational Signals To Watch

From `Inspect notifier state`:

- `level`
- `recovery_stage_latest`
- `recovery_red_lock_streak`
- `last_decision_reason`
- `last_decision_sent`

From notifier step:

- `Notify decision` line for transition reason.

From artifacts:

- `logs/telemetry_dashboard_notify_state.json`
- `logs/reliability_gate.txt`
- `logs/telemetry_alert_summary.txt`
- `logs/telemetry_dashboard_page.html`

## Dashboard Threshold Tuning

The top strip in `telemetry_dashboard_page.html` now color-codes reliability counters using env thresholds.

- Debt: `TELEMETRY_DASHBOARD_DEBT_WARN_SEC`, `TELEMETRY_DASHBOARD_DEBT_CRIT_SEC`
- Mismatch streak: `TELEMETRY_DASHBOARD_MISMATCH_STREAK_WARN`, `TELEMETRY_DASHBOARD_MISMATCH_STREAK_CRIT`
- Coverage gap: `TELEMETRY_DASHBOARD_COVERAGE_GAP_WARN_SEC`, `TELEMETRY_DASHBOARD_COVERAGE_GAP_CRIT_SEC`
- Refresh budget blocked: `TELEMETRY_DASHBOARD_REFRESH_BLOCKED_WARN`, `TELEMETRY_DASHBOARD_REFRESH_BLOCKED_CRIT`
- Force override count: `TELEMETRY_DASHBOARD_FORCE_OVERRIDE_WARN`, `TELEMETRY_DASHBOARD_FORCE_OVERRIDE_CRIT`

Runtime reliability gate current/peak thresholds:

- `RELIABILITY_GATE_MAX_POSITION_MISMATCH`:
  - Current position mismatch cap used for immediate posture degrade.
- `RELIABILITY_GATE_MAX_POSITION_MISMATCH_PEAK`:
  - Peak position mismatch cap over the gate window (degrades even if current recovers).
- `RELIABILITY_GATE_MAX_ORPHAN_COUNT`:
  - Current orphan action cap from rebuild decisions.
- `RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT`:
  - Current cap for restart `intent_id` collisions detected during rebuild/orphan classification.
- `RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS`:
  - Current protection coverage gap cap (seconds).
- `RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS_PEAK`:
  - Peak coverage gap cap over the gate window.

Operational intent:

- `current` thresholds enforce live, immediate safety posture.
- `peak` thresholds prevent instant re-open after recent instability and support recover-slow behavior.
- Runtime coupling remains entry-only; exits are not blocked by gate thresholds.
- Non-zero `intent_collision_count` should be treated as restart-state ambiguity and investigated before expecting entry recovery.

Related refresh-budget controls (runtime):

- `RECONCILE_STOP_REFRESH_BUDGET_WINDOW_SEC`, `RECONCILE_STOP_REFRESH_MAX_PER_WINDOW`
- `RECONCILE_TP_REFRESH_BUDGET_WINDOW_SEC`, `RECONCILE_TP_REFRESH_MAX_PER_WINDOW`
- Notifier escalation thresholds:
  - `PROTECTION_REFRESH_BUDGET_BLOCKED_CRITICAL_THRESHOLD`
  - `PROTECTION_REFRESH_FORCE_OVERRIDE_CRITICAL_THRESHOLD`
- Controller decay/warmup controls:
  - `BELIEF_PROTECTION_REFRESH_DECAY_SEC`
  - `BELIEF_PROTECTION_REFRESH_RECOVER_SEC`
  - `BELIEF_PROTECTION_REFRESH_WARMUP_NOTIONAL_SCALE`
  - `BELIEF_PROTECTION_REFRESH_WARMUP_LEVERAGE_SCALE`
  - `BELIEF_PROTECTION_REFRESH_WARMUP_MIN_CONF_EXTRA`
  - `BELIEF_PROTECTION_REFRESH_WARMUP_COOLDOWN_EXTRA_SEC`

Correlation threshold semantics (shared across summary + notifier):

- `CORR_STRESS_THRESHOLD`:
  - Trigger count for stress-regime correlation pressure events.
  - Used by `tools/telemetry_alert_summary.py --corr-stress-threshold`.
  - Used by `tools/telemetry_dashboard_notify.py --corr-stress-threshold`.
- `CORR_PRESSURE_THRESHOLD`:
  - Absolute latest `corr_pressure` trigger.
  - Used by `tools/telemetry_alert_summary.py --corr-pressure-threshold`.
  - Used by `tools/telemetry_dashboard_notify.py --corr-pressure-threshold`.
- `CORR_EXIT_PNL_DROP_THRESHOLD`:
  - Absolute `avg_pnl` delta trigger where stress regime underperforms normal regime.
  - Used by `tools/telemetry_alert_summary.py --corr-exit-pnl-drop-threshold`.
  - Used by `tools/telemetry_dashboard_notify.py --corr-exit-pnl-drop-threshold`.
- Backward compatibility:
  - `CORR_STRESS_CRITICAL_THRESHOLD` and `CORR_PRESSURE_CRITICAL_THRESHOLD` are still accepted as fallback/default input.
  - `CORR_EXIT_PNL_DROP_CRITICAL_THRESHOLD` is still accepted as fallback/default input.
  - `--corr-stress-critical-threshold`, `--corr-pressure-critical-threshold`, and `--corr-exit-pnl-drop-critical-threshold` remain as legacy aliases in notifier.

Recommended approach:

1. Keep defaults for one full day of live telemetry.
2. Tighten WARN first (not CRIT) if you need earlier visibility.
3. Lower CRIT only when you have repeated incidents requiring faster clamp triage.

## Incident Playbook: Intent Collision Spike

Use this when `intent_collision_count` appears in reliability gate output or notifier escalates due to collision signals.

Primary controls:

- Runtime gate threshold:
  - `RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT`
- Notifier immediate escalation:
  - `INTENT_COLLISION_CRITICAL_THRESHOLD`
  - CLI: `--intent-collision-critical-threshold`
- Notifier streak escalation:
  - `INTENT_COLLISION_CRITICAL_STREAK`
  - CLI: `--intent-collision-critical-streak`

Expected behavior:

- Entry posture clamps quickly (`allow_entries=false` via runtime gate / belief controller).
- Exits remain unaffected (entry-only safety coupling invariant).
- Notifier state persists:
  - `reliability_intent_collision_count`
  - `intent_collision_streak`
  - `level=critical` when threshold or streak condition is met.

Immediate checks:

1. Confirm reliability gate signal:
   - `intent_collision_count` in `logs/reliability_gate.txt`
2. Confirm runtime posture:
   - latest `execution.belief_state` includes `guard_cause_tags` with `runtime_gate_intent_collision`
3. Confirm notifier state:
   - `logs/telemetry_dashboard_notify_state.json` shows `intent_collision_streak` and current `level`

Recovery/unlock checks before expecting entry re-enable:

1. `intent_collision_count` returns to zero for fresh gate windows.
2. Runtime gate cause summary no longer includes collision contributors.
3. Standard unlock conditions continue to pass:
   - healthy ticks
   - journal coverage target
   - contradiction clear window
   - protection gap target

## Workflow Dispatch Presets: Collision Escalate Then Clear

Use these presets for one-click smoke validation of collision escalation and reset behavior.

Escalate (expect critical + streak=1):

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_intent_collision_count=1 \
  -f expected_notifier_level=critical \
  -f expected_intent_collision_streak=1
```

Clear/reset (expect streak reset to 0; level may remain critical only if other independent signals are active):

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_intent_collision_clear=true \
  -f expected_intent_collision_streak=0
```

Optional strict clear check (only when you expect no other critical contributors):

```bash
gh workflow run .github/workflows/telemetry-dashboard.yml \
  -R phoenixsenses/eclipse_scalper \
  -f simulate_intent_collision_clear=true \
  -f expected_notifier_level=normal \
  -f expected_intent_collision_streak=0
```

## Escalation Rules

- If `level=critical` repeats unexpectedly without RED_LOCK smoke input:
  - inspect real `execution.belief_state` events and guard conditions in telemetry.
- If state is missing:
  - verify cache restore/save steps and notifier state write path:
    - `TELEMETRY_DASHBOARD_NOTIFY_STATE_PATH=logs/telemetry_dashboard_notify_state.json`
- If Telegram alerts are missing but state transitions occur:
  - verify `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` secrets.

## Quick Reference

- Workflow URL filter:
  - `https://github.com/phoenixsenses/eclipse_scalper/actions/workflows/telemetry-dashboard.yml`
- Latest run:
  - `gh run list -R phoenixsenses/eclipse_scalper --workflow .github/workflows/telemetry-dashboard.yml --limit 1`
