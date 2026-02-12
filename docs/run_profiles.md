# Run Profiles

## Safe Live Profile (`run-bot-ps2.ps1`)

This profile is the hardened launcher for low-risk live operation:

- single-symbol focus by default (`DOGEUSDT`)
- low leverage / low fixed notional
- first-live allowlist constraints
- runtime reliability coupling enabled (reliability gate + reconcile-first posture pressure)

It is designed to refuse unsafe launches before `execution.bootstrap` starts.

## What Preflight Validates

Before launch, preflight validates:

- `SCALPER_DRY_RUN` is `0` or `1`
- `ACTIVE_SYMBOLS` is non-empty
- `MARGIN_MODE=isolated`
- `LEVERAGE <= SAFE_PROFILE_MAX_LEVERAGE` (default `1`)
- `FIXED_NOTIONAL_USDT > 0`
- if `FIRST_LIVE_SAFE=1`: allowlist exists and notional cap is respected
- `MAX_DAILY_LOSS_PCT` and `MAX_DRAWDOWN_PCT` are present and in safe ranges
- correlation group configuration parses cleanly
- telemetry/log parent directories exist

If any check fails, startup exits non-zero with actionable error lines.

## Runtime Reliability Coupling

The profile explicitly enables:

- `RUNTIME_RELIABILITY_COUPLING=1`
- `RELIABILITY_GATE_*` thresholds
- `BELIEF_RUNTIME_GATE_*` warmup/recovery knobs
- `BELIEF_RECONCILE_FIRST_GATE_*` thresholds

This coupling only tightens **entry posture**. It does not alter exit invariants.

Reliability gate reporting is windowed by default:

- `RELIABILITY_GATE_WINDOW_SECONDS=14400` (4h rolling view)
- `RELIABILITY_GATE_STALE_SECONDS=900` (warn when gate file is stale in preflight)
- use this with `tools/reliability_gate.py --window-seconds ...` to avoid stale historical debt dominating current posture.

Current + peak threshold defaults for runtime coupling:

- `RELIABILITY_GATE_MAX_POSITION_MISMATCH=1` (current)
- `RELIABILITY_GATE_MAX_POSITION_MISMATCH_PEAK=2` (peak within window)
- `RELIABILITY_GATE_MAX_ORPHAN_COUNT=0` (current)
- `RELIABILITY_GATE_MAX_INTENT_COLLISION_COUNT=0` (current)
- `RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS=0` (current)
- `RELIABILITY_GATE_MAX_COVERAGE_GAP_SECONDS_PEAK=30` (peak within window)

Interpretation:

- `current` thresholds enforce immediate safety posture.
- `peak` thresholds catch recent instability even after current values recover.
- runtime degrade remains entry-only; exits stay exempt.

Run profile launch now refreshes the gate snapshot before bootstrap:

- `python tools/reliability_gate_refresh.py --window-seconds $env:RELIABILITY_GATE_WINDOW_SECONDS --allow-missing`

## Operator Clarity

Preflight prints:

- mode (`DRY-RUN` or `LIVE`)
- symbols, notional, leverage, margin mode
- first-live allowlist/cap state
- daily loss and drawdown limits
- correlation caps
- reliability gate / reconcile-first thresholds
- warmup/recovery knobs and quick recovery guidance

## Entry Block Diagnosis

When entries are blocked, check:

- reliability gate degradation (`runtime_gate_degraded`)
- reconcile-first pressure (`reconcile_first_gate_*`)
- belief-controller mode (`ORANGE`/`RED`)
- per-symbol guard overlays (`symbol_belief_debt_sec`)
- correlation budget/caps

## Recovery Steps (`ORANGE` / `RED`)

1. Inspect `logs/reliability_gate.txt` for mismatch/coverage/contradiction contributors.
2. If `intent_collision_count` is non-zero, inspect `rebuild.orphan_decision` and `rebuild.summary` events to confirm no stale client-order-id reuse across distinct live order ids.
3. Verify telemetry and journal freshness (`logs/telemetry.jsonl`, journal path).
4. Resolve data quality issues (stale feeds, missing journal transitions, contradiction spikes).
5. Wait for warmup timers (`BELIEF_RUNTIME_GATE_RECOVER_SEC`, `BELIEF_POST_RED_WARMUP_SEC`) to pass.
6. Confirm posture re-enters `YELLOW`/`GREEN` before expecting normal entry throughput.

## Run Commands

Preflight only:

```powershell
$env:RUN_PREFLIGHT_ONLY = "1"
.\run-bot-ps2.ps1
```

Smoke + start:

```powershell
$env:RUN_SMOKE = "1"
.\run-bot-ps2.ps1
```

Full start:

```powershell
.\run-bot-ps2.ps1
```

One-shot maintenance (safe rebuild + single reconcile tick, then exit):

```powershell
$env:SCALPER_DRY_RUN = "1"
$env:BOOT_REBUILD_ON_START = "1"
$env:BOOT_MAINTENANCE_ONESHOT = "1"
python -m execution.bootstrap
```
