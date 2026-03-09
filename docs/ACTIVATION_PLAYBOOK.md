# Activation Playbook — ETHUSDT Micro Pocket
*Target date: ~2026-03-13*

## Context

Validated pocket: **ETHUSDT, h=60, imb>=0.85, micro_edge_v3_passive_alpha**
Pre-regime results (before Mar 1): win_rate=60.5%, filled_n=38, avg_net=+1.23e-04 bps — GO.
Mar 1–6 trending rally broke fill economics → all candidates fail with `fees_dominate`.
The 7D lookback clears that block on **~Mar 13–14**.

---

## Step 0 — Prerequisites (do these once, already done)

| Item | Status |
|---|---|
| `execution/event_lane_gate.py` in shadow mode | Done (PR #22) |
| `tools/check_event_lanes.py` | Done |
| `tools/pocket_promotion_checklist.py` | Done |
| `tools/watch_regime_recovery.py` | Done |
| Shadow gate env flags set | Must set on target machine |

Set shadow mode on the live machine now (safe, no blocking):
```powershell
$env:ENTRY_EVENT_LANE_GATE_ENABLED = "1"
$env:ENTRY_EVENT_LANE_GATE_SHADOW  = "1"
$env:ENTRY_EVENT_LANE_GATE_DB      = "data/microstructure.db"
```

---

## Step 1 — Mar 13 morning: Gate check

```bash
py -3 -m tools.check_event_lanes --db data/microstructure.db
```

**ALLOWED** → proceed to Step 2.
**BLOCKED** → wait. Check again in a few hours. Use watchdog:
```bash
py -3 -m tools.watch_regime_recovery --db data/microstructure.db --consecutive 2
```
Do not proceed until you see `[READY]` with consecutive=2.

---

## Step 2 — Re-validation

Run the ranker on the 5 pre-validated candidates:
```bash
py -3 -m tools.rank_passive_pockets_forward \
  --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_IMB05.md \
  --db data/microstructure.db \
  --lookback-min 10080 --splits 3 \
  --out-md  reports/REVALIDATE_POST_REGIME_7D.md \
  --out-json reports/REVALIDATE_POST_REGIME_7D.json
```

**Expected result:** ≥1 candidate with pass_rate ≥ 0.50 and positive NPA.
**If all fail again:** regime has not fully cleared. Wait another day and check `--lookback-min 4320` (3D) as a probe.

---

## Step 3 — Promotion checklist

```bash
py -3 -m tools.pocket_promotion_checklist --db data/microstructure.db
```

Expected output after Step 2 succeeds:
```
[PASS] event_lane_gate      : gate=ALLOWED
[PASS] revalidation_pass    : pass_rate=0.XX
[SKIP] fill_density         : report not found (acceptable)
[PASS] market_regime        : rolling_1h_return=+0.00X (below 1.5% threshold)

Overall: HOLD (1 SKIP)
```

HOLD with only `fill_density` SKIP is acceptable — proceed.
**Stop** if any item is FAIL.

---

## Step 4 — Shadow log review (1–2 hours)

Keep bot running in shadow mode. Watch the gate log:
```bash
py -3 -m tools.review_event_lane_gate_shadow \
  --telemetry-path logs/telemetry.jsonl --symbol ETHUSDT --last-min 120
```

Look for:
- `would_block_rate` < 30% during normal market hours → gate not firing excessively
- No false-block pattern (gate blocking on clearly calm buckets)

If shadow behavior looks clean, proceed.

---

## Step 5 — Activate live gate

```powershell
$env:ENTRY_EVENT_LANE_GATE_SHADOW = "0"   # Shadow off — gate now blocks
# ENTRY_EVENT_LANE_GATE_ENABLED stays "1"
```

Restart the entry loop. The gate is now live.

Verify in logs:
```
[event_lane_gate] symbol=ETHUSDT gate_status=blocked shadow=False
[event_lane_gate] symbol=ETHUSDT gate_status=allowed shadow=False
```

---

## Step 6 — Monitor first 24h

```bash
# Every few hours:
py -3 -m tools.review_event_lane_gate_shadow \
  --telemetry-path logs/telemetry.jsonl --last-min 60

# Full daily report:
py -3 -m tools.daily_research_report --db data/microstructure.db
```

Watch for:
- Fill rate not collapsing vs pre-regime baseline (expected ~48–62%)
- Gross NPA positive after 50+ fills
- Gate not blocking > 40% of signals during normal hours

---

## Rollback

If results deteriorate within 48h:

```powershell
$env:ENTRY_EVENT_LANE_GATE_ENABLED = "0"  # Gate fully off
```

No code change needed. Gate returns to inactive state on next signal.

---

## Key thresholds (pre-regime baseline)

| Metric | Target |
|---|---|
| Win rate | ≥ 55% |
| Fill rate (touch) | ~76–79% |
| Fill rate (joint) | ~38–40% |
| Gross NPA per fill | ≥ +1.0e-04 |
| Break-even fee | ~0.8 bps/leg |
| Gate block rate (shadow) | < 30% in normal hours |

---

## If pocket does NOT recover by Mar 16

Options in priority order:
1. Extend lookback to 14D (`--lookback-min 20160 --splits 2`) — needs 30+ days of data
2. Test h=120 imb>=0.85 separately (different fill economics)
3. Revisit regime-conditioned strategy (SELL_UP only) — requires regime runtime flag

---

## Files reference

| File | Purpose |
|---|---|
| `reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_IMB05.md` | 5 validated candidates (input to ranker) |
| `reports/REVALIDATE_POST_REGIME_7D.md` | Post-regime re-validation result (created in Step 2) |
| `docs/EVENT_LANE_GATE_SHADOW_RUNBOOK.md` | Shadow gate env flags and review procedure |
| `execution/event_lane_gate.py` | Gate implementation |
| `execution/entry_loop.py` | Hook location (~line 2431) |
