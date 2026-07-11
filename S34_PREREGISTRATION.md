# S34 Live Paper Validation — Pre-Registration

**Status:** amended 2026-06-28 (Amendment 9). BUY-continuation anchor contamination quarantined.

**Pre-registration variants (independent N=100 validations):**
- `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30` — clock set Amendment 4, N=0
- `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30` — clock set Amendment 4, N=0
- `SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30` — clock set Amendment 7, N=0

**Exploratory telemetry (logging only, not pre-registered):**
- `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30`
- `ETH_BUY_LIQ_LONG_200K_BTC_PRE15_TP120_SL40_BE30_DELAY60`
- `ETH_BUY_LIQ_LONG_500K_NEGTREND_STRETCHED_TP60_SL40_BE30`

**SELL-continuation exploratory forward validation (Amendment 6+, clocks start 2026-06-26):**
- `ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40`
- `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40`
- `SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30`
- `SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40` — added Amendment 8

**Deprecated / audit-only (no new paper positions):**
- `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` - `archived_contaminated`, old N reset to 0; prior forward evidence relied on terminal cluster attribution stamped too early.
- `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` - `archived_contaminated`, old N reset to 0; primary anchor audit did not validate after de-lookahead.

**Date:** 2026-06-11 (amended 2026-06-28)
**Strategy under test:** S34 — liquidation-momentum BUY (taker entry), live paper. ETHUSDT, SOLUSDT, BTCUSDT.
**Purpose:** commit, in advance, to the exact criteria that will decide whether S34 has a real, tradeable edge — so that no decision is influenced by data seen after this document is written.

This document is binding. Any deviation (re-tuning, peeking at the holdout, changing the runner mid-collection) voids the sample and restarts the clock. The whole value of pre-registration is that it predates the data.

---

## 0. Why this measurement is trustworthy now

The validation only counts data produced under the corrected apparatus. The following were fixed before this sample begins, and the sample is defined as trades collected *after* all of them were in place:

1. Chronological evaluation bug fixed — no replay-state rewind; an exit can only trigger at or after the state that caused it.
2. Executable fill model — entry fills at ask, taker SL/BE/time exits fill at bid, real bookTicker quotes (no mark-price idealization).
3. `NO_FILL_DATA` quarantine — trades with no real bid/ask at fill time are skipped, not silently priced at zero spread.
4. Corrected cost attribution — `net = gross − entry_adverse − exit_adverse − spread − fee`; the old mislabeled "spread" (mark-to-fill drift) is now its own column; true bid/ask spread is computed from real quotes.

**Excluded from the sample:** P013 (pre-fill-model, corrupted) and P056 (N=1, collected during apparatus stabilization). The validation clock starts at the first trade after this document's date under the frozen config.

---

## Amendment 9 - Feature Availability Contract v1 / BUY Continuation Quarantine (2026-06-28)

**Basis:** The anchor-integrity audit found that the apparent ETH BUY continuation edge was dominated by terminal cluster information stamped at an earlier entry timestamp. `cluster_notional` is only knowable after the cluster has developed, so using full terminal notional at `first_ts` violates the feature availability contract.

**Phase 0 action:** `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` and `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` are moved to `archived_contaminated`. Old forward samples are treated as `lookahead_contaminated=true` unless they explicitly carry and use a knowable `threshold_cross_ts_ms` entry anchor. Their acceptance counters are reset to 0 and prior N values must not be cited as validation evidence.

**Contract adopted:** For any candidate entry at wall-clock time `T`, every consumed entry feature must satisfy `knowable_at_ts <= T`. `TERMINAL_CLUSTER` and `FORWARD_OUTCOME` features are illegal in entry vectors. Violations raise `LookaheadViolation`; they are not warnings.

**Implementation status:** `tools/s34_feature_availability.py` defines the feature classes, `FeatureValue`, `LookaheadViolation`, registry writer, and runner entry gate. `tools/s34_shadow_paper_runner.py` now applies the gate before creating paper trades and hard-blocks the two archived BUY continuation rules from new paper positions.

**Discovery status:** No clean ETH BUY continuation pre-registration is open until the feature availability registry is green and the candidate is re-derived from running, knowable features.

---

## Amendment 8 — SOL 100K SELL→SHORT Exploratory (2026-06-26)

**Basis:** 120-day real-fill TP/SL sweep for SOLUSDT 100K SELL liq → SHORT:

| Config | N | Median | 2nd-Half | WR | No-Fill |
|---|---|---|---|---|---|
| TP60/SL30/BE40 | 82 | +37.9 bps | +47.8 bps | 57% | 20% |

Signal quality is strong: second-half median (+47.8 bps) substantially exceeds first-half, confirming non-degrading edge. No-fill rate 20% — markedly lower than ETH routes, clean fill model. Priority=15 ensures SOL SELL 200K (priority=10) fires first on any same-cascade event; 100K only opens when 200K threshold was not reached.

**Rule (frozen):**
```python
S34Rule(name="SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40",
        symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=100_000.0, tp_bps=60.0, sl_bps=30.0, be_trigger_bps=40.0,
        use_global_regime=False, priority=15)
```

**Same exploratory pass criteria as Amendment 6 (SELL_EXP):**

| Criterion | Value |
|---|---|
| Target N | 30 closed trades |
| Pass P1: median net | > 0 |
| Pass P2: top3-removed cumulative | > 0 |
| Pass P3: day spread | >= 8 distinct calendar days |

A pass authorizes elevation to full N=100 pre-registration if warranted. Does not constitute a main S34 pre-registration pass.

**Frozen runner SHA256:** `037b2fe0774a6f2f88534163aa99d6e882dce617a879d76120d88250358a1197`

---

## Amendment 7 — SOL 100K BUY→LONG Pre-Registration (2026-06-26)

**Basis:** 120-day TP/SL sweep — N=75, median +47.1 bps, 2nd-half +48.3 bps, WR=60%, no-fill 23%. Nearly identical quality to SOL 200K. Added to `DEFAULT_RULES` with priority=15 (evaluated before 200K on same cascade).

**Rule (frozen):**
```python
S34Rule(name="SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30",
        symbol="SOLUSDT", threshold_usd=100_000.0,
        tp_bps=60.0, sl_bps=40.0, be_trigger_bps=30.0,
        use_global_regime=False, priority=15)
```

Same N=100 / 40-60 split / K1-K4 kill criteria as other main pre-reg variants. Clock starts 2026-06-26.

**Frozen runner SHA256:** `2d0589d3347b8e5da63e1a84d0b15139c5bd92db7f01b8266cfb9e721698d16b`

---

## Amendment 6 — SELL-Continuation Exploratory Forward Validation (2026-06-26)

**Basis:** 120-day real-fill sweep shows SELL liq → SHORT is near-symmetric to BUY liq → LONG for ETH 500K and SOL 200K:

| Route | Median | 2nd-Half | WR | No-Fill |
|---|---|---|---|---|
| ETH 500K BUY→LONG | +48.5 bps | +50.2 | 62% | 42% |
| ETH 500K SELL→SHORT | +50.6 bps | +51.1 | 63% | 42% |
| SOL 200K BUY→LONG | +49.0 bps | +48.6 | 65% | 25% |
| SOL 200K SELL→SHORT | +49.9 bps | +50.9 | 64% | 20% |

TP60 optimal on both sides. BTC SELL side weaker (+7 bps vs +42 BUY) — not added.

**New routes added to `DEFAULT_RULES`:**
```python
S34Rule(name="ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
        liq_side="SELL", direction="SHORT",
        threshold_usd=500_000.0, tp_bps=60.0, sl_bps=40.0, be_trigger_bps=40.0,
        use_global_regime=False, priority=10)

S34Rule(name="SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30",
        symbol="SOLUSDT", liq_side="SELL", direction="SHORT",
        threshold_usd=200_000.0, tp_bps=60.0, sl_bps=30.0, be_trigger_bps=30.0,
        use_global_regime=False, priority=10)
```

**These are NOT main S34 pre-registration variants.** They are separate "SELL-continuation exploratory validation" with lighter pass criteria:

| Criterion | Value |
|---|---|
| Target N | 30 closed trades |
| Pass: median net | > 0 |
| Pass: top3-removed cumulative | > 0 |
| Pass: day spread | ≥ 8 distinct calendar days |
| Monitor: no-fill rate | logged, flag if > 35% |
| Monitor: cross-direction conflict | logged per event (see below) |

A pass here does not constitute a main S34 pre-registration pass. It authorizes elevation to full N=100 pre-registration if warranted.

**Cross-direction conflict logging:** When a SELL SHORT opens while a BUY LONG is open on the same symbol (or vice versa), the runner logs a `s34_shadow_paper.cross_direction_conflict` event and tags the trade with `cross_direction_conflict=True`. This does NOT block the trade — both can be open simultaneously — but allows post-hoc analysis of whether same-cascade BUY+SELL openings are correlated or independent.

**Runner invocation:** unchanged — no new CLI flags. New rules activate automatically on next loop cycle.

**Frozen repo HEAD at amendment:** `2d8a48269d2729731f202adb5fe8b7a040b3c91e`
**Frozen runner SHA256:** `189a9c8c82861d82ae0d514ec023c1db75fc62770523d0710e971c144d53d3f8`

---

## Amendment 5 — Route Map Reconciliation (2026-06-25)

**Basis:** Pre-registration document was inconsistent with runner state. Runner already had `DEPRECATED_PAPER_RULES = frozenset({"ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"})` — a hard paper block preventing new positions — but the document still described 50K/TP120 as the primary validation strategy. Additionally, the runner SHA256 in §1 (`f877...`) was computed against an older version of the runner; the current file hash is updated below.

**Changes in this amendment:**

1. **50K/TP120 status formalized:** `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` is Deprecated / Audit-Only. The `DEPRECATED_PAPER_RULES` block was added to the runner in a prior session after weak forward performance (WR 30%, avg net −9.8 bps over 10 trades). The rule remains in `DEFAULT_RULES` for signal/prediction/intelligence ledger logging, but cannot open new shadow paper positions. This status is now reflected in the document header.

2. **Full route map defined:** All 7 routes in `DEFAULT_RULES` are classified:
   - **Pre-registration (3 routes):** ETH 500K/daytrend, SOL 200K, BTC 1M/distributed — independent N=100 validations per §2–§5.
   - **Exploratory telemetry (3 routes):** ETH 200K/TP60, ETH 200K/BTC_PRE15, ETH 500K/negtrend_stretched — logging trades and signals, no pre-registration commitment, not used for §4 decision.
   - **Deprecated/audit-only (1 route):** ETH 50K/TP120 — paper-blocked, signals logged only.

3. **Runner SHA256 updated:** The frozen runner SHA in §1 is superseded. The current runner file SHA256 is:
   `cd3584a64a1cbe8076faf7709beafeb8bcda8d470d8127dec0951290bca8feaa`
   This hash covers the runner as it stands with the full 7-rule `DEFAULT_RULES` and the `DEPRECATED_PAPER_RULES` block. Any further code change restarts the clock for all three pre-reg variants and requires a new amendment.

4. **Sections §1–§7 scope:** Those sections define the validation protocol. Where they refer to "the strategy" or "a variant" in singular, they apply independently to each of the three pre-registration variants. A pass in §4 is evaluated per variant — one variant passing does not affect the others.

**Frozen repo HEAD at amendment:** `2d8a48269d2729731f202adb5fe8b7a040b3c91e`
**Frozen runner SHA256:** `cd3584a64a1cbe8076faf7709beafeb8bcda8d470d8127dec0951290bca8feaa`

---

## Amendment 4 — Parallel Pre-Registration: SOL 200K and BTC 1000K (2026-06-25)

**Basis:** TP/SL sweep across 120-day real bookTicker backtest confirms both variants viable at standard fees:
- `SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30`: N=49, median +49.0 bps, WR=65%, 2nd-half median +48.6 bps
- `BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30`: N=83, median +42.3 bps, WR=60%, 2nd-half median +40.4 bps

Both variants are already instrumented in `DEFAULT_RULES` with no code changes required. All trades accumulated prior to this amendment date are excluded from their validation samples — clocks start at 2026-06-25.

**Variant definitions (frozen — any change restarts the clock for that variant):**

SOL 200K:
```python
S34Rule(
    name="SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
    symbol="SOLUSDT",
    threshold_usd=200_000.0,
    tp_bps=60.0,
    sl_bps=40.0,
    be_trigger_bps=30.0,
    use_global_regime=False,
    priority=10,
)
```

BTC 1000K (distributed quality filter applied — only events where no single order exceeds 50% of the cluster notional):
```python
S34Rule(
    name="BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
    symbol="BTCUSDT",
    threshold_usd=1_000_000.0,
    tp_bps=60.0,
    sl_bps=30.0,
    be_trigger_bps=30.0,
    use_global_regime=False,
    max_single_liq_share_pct=50.0,
    priority=10,
)
```

**Each variant is validated independently** under the same §2–§5 protocol as ETH 500K:
- N=100 target, calibration=first 40, holdout=next 60
- Kill criteria K1–K4 identical (see §5)
- Decision rule §4 applied separately per variant: a pass on one does not imply a pass on another
- Variants are not combined into a portfolio for the purpose of this validation

**Runner invocation (unchanged — no config modification):**
```
python tools/s34_shadow_paper_runner.py --loop --quality-gate-enabled --quality-gate-min-eclipse 42.0
```
The quality gate is ETH-only today (permissive fallback for SOL/BTC while detector enrichment is unavailable for those symbols). This is noted — if the detector is extended to SOL/BTC mid-collection, it does not change the sample definition unless a new amendment is filed.

**Frozen repo HEAD at amendment:** `2d8a48269d2729731f202adb5fe8b7a040b3c91e`

---

## Amendment 3 — Variant Switch to 500K/TP60 (2026-06-25)

**Basis:** Post-launch review of 10 valid ETH 50K/TP120 trades (June 16–22, all with real dates): WinR 30%, avg net −9.8 bps. Parallel runner data shows `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30` at N=16, WinR 75%, avg net +37.5 bps over the same window. The 50K/TP120 variant is structurally inferior: low signal threshold creates noise, TP120 gives back gains. The 10 pre-reg trades are in the calibration slice (< 40) and have not triggered any kill criterion check yet.

**Change:** Variant of record switches from `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` to `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30`. Validation clock resets to N=0 at this amendment date (2026-06-25). The 10 prior 50K/TP120 trades and 16 prior 500K/TP60 trades are all excluded from the new sample — none were collected after this amendment.

**Why 500K/TP60 is cleaner:**
- $500K threshold filters noise — only large institutional liquidation events
- `use_global_regime=False` — no range_pct gate; fires in any market condition
- TP60 exits faster, capturing the initial momentum impulse before reversal
- Signal frequency: ~1.5 trades/day → N=100 in ~2 months

**New frozen runner invocation:**
```
python tools/s34_shadow_paper_runner.py --loop --quality-gate-enabled --quality-gate-min-eclipse 42.0
```
Runner already tracks this variant — no code changes. The `--quality-gate-enabled` flag from Amendment 2 is retained (permissive fallback active until detector is rebuilt).

**Frozen repo HEAD at amendment:** `2d8a48269d2729731f202adb5fe8b7a040b3c91e`
**Frozen runner SHA256:** `386452600d81598bdd6b746aa89d4f4f4bee0a27a56b180f280a609324e37a74`

---

## Amendment 2 — Quality Gate Added Before Validation Clock Runs (2026-06-24)

**Basis:** Bucket analysis of 61 pre-registration-window signals (all collected before the validation clock started on 2026-06-11, last signal 2026-04-14). N=0 valid trades have been collected under the current pre-registration. This amendment is filed before any validation data exists; it does not retroactively modify any outcome.

**Change:** The runner adds `--quality-gate-enabled` flag. The gate reads `confidence_band` and `eclipse_score` from `detector_signals` and skips signals that fail:
- `confidence_band = 'standard'` → skip (36/61 pre-reg signals, avg 15m return -0.054, WinR 55.6%)
- `eclipse_score < 42.0` → skip (bottom two quartiles, avg 15m return -0.188, WinR 44%)

**Permissive fallback:** if no `detector_signals` row exists within ±5 minutes of the liquidation cluster, the gate passes (does not block). This ensures the gate only fires when the detector has enriched the signal.

**Effect on sample definition:** Only signals passing the quality gate are counted as valid observations in the pre-registration sample. The variant of record remains `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` with `--quality-gate-enabled --quality-gate-min-eclipse 42.0` (standard confidence blocked).

**Runner invocation (frozen):**
```
python tools/s34_shadow_paper_runner.py --loop --quality-gate-enabled --quality-gate-min-eclipse 42.0
```

---

## Amendment 1 — S34 Variant of Record (2026-06-12) [superseded by Amendment 3]

~~S34 of record for this validation is the single runner variant: `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30`~~

Superseded by Amendment 3 (2026-06-25). The variant of record is now `ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30`. All other variants remain exploratory telemetry only. One valid 500K/TP60 closed position equals one validation observation.

---

## 1. Collection protocol (frozen)

- Runner: `s34_shadow_paper_runner.py`.
- Frozen repo HEAD: `2d8a48269d2729731f202adb5fe8b7a040b3c91e`.
- Frozen runner file SHA256: `f87748884c7cdcb1cba6e97b8afd184a802c92dbe2103bd1110bdc3c91aca591`.
- Frozen runner config SHA256: `39a037a2c65967781e08575fe1753aa0c5df4d70bdc74b658c7e326f1c8fe6a8`.
- No changes to thresholds (BE/TP/SL), signal config, fill model, or attribution logic during collection. A code or config change of any kind voids the sample and restarts from N=0.
- A **valid closed trade** = a position opened from a real signal on a regime-eligible day, filled with real bookTicker data, and closed by the runner with a reconciling cost decomposition (`net = gross − entry_adverse − exit_adverse − spread − fee` to float tolerance). Quarantined (`NO_FILL_DATA`) signals are logged but are not trades.
- Feeds must stay live throughout (watchdog GREEN, bookTicker and liquidation streams writing). A feed outage during collection does not void prior trades but pauses accumulation.

## 2. Sample structure

- **Target total N = 100** valid closed trades.
- **Calibration slice = first 40** trades. May be inspected — but only for the kill checks in §5 and for estimating friction distributions. **Not for tuning.**
- **Holdout = next 60** trades. **Locked.** No holdout statistic is computed or viewed until N = 100 is reached. The decision in §4 is run once, on the holdout, after the full 60 are collected.

## 3. Friction model

Per-trade `net_bps` already nets all real friction via the corrected attribution, so the decision is taken directly on the net distribution — there is no separately-assumed break-even bar to set after the fact. The calibration slice is used only to:

- estimate the distributions of `entry_adverse_bps`, `exit_adverse_bps` (bucketed by exit reason: SL vs BE vs TP), `spread`, and `fee`;
- sanity-check that attribution stays coherent at larger N;
- evaluate the structural-cost kill (§5).

Known/expected components for reference: fee ≈ 8.0 bps round trip (taker); true bid/ask spread ≈ negligible (~0.06 bps observed); entry adverse selection = the taker momentum-chase cost (the open structural question this sample answers).

## 4. Primary decision rule (run once, on the holdout, at N = 100)

S34 **passes** only if **both** hold (logical AND):

1. **Economic significance.** Mean holdout `net_bps` > 0, and the lower bound of a one-sided 95% bootstrap confidence interval (10,000 resamples) on mean `net_bps` is > 0. A positive point estimate alone is not sufficient.
2. **Statistical significance, deflated.** The Deflated Sharpe Ratio of the holdout per-trade net returns clears the 95% threshold (probability that the true Sharpe > 0 ≥ 0.95), with the **number of trials declared honestly** from the research log — every distinct S34/OHLCV/signal configuration ever tested counts toward the deflation. Pull the exact count; if unavailable, use a conservative floor of N_trials = 50.

If either condition fails, S34 does not pass.

## 5. Kill criteria (staged)

**Calibration-stage early stops** (checked once the 40-trade calibration slice is complete):

- **K1 — net not viable in-sample.** Mean calibration `net_bps` ≤ 0. If the edge is not even positive in the calibration slice after real friction, halt and reassess before spending the holdout.
- **K2 — structurally untradeable as taker.** Median `entry_adverse_bps` in calibration ≥ mean |`gross_bps`|. If the taker entry-chase cost alone consumes the directional edge, the signal cannot be traded as a taker regardless of gross quality — kill.

**Data-integrity kill** (monitored throughout):

- **K3 — quarantine selection bias.** If the `NO_FILL_DATA` quarantine rate exceeds 25% **and** quarantines are significantly correlated with cascade intensity (i.e. the runner is systematically dropping the most violent moments — exactly when the signal fires), the surviving sample is biased toward calm conditions and is not representative. Halt, fix collection, restart the clock.

**Holdout-stage terminal kill:**

- **K4 — fails validation.** The §4 AND test fails on the holdout. S34 is retired in its current form.

## 6. What a pass authorizes — and what it does not

A holdout pass is **necessary, not sufficient** for deploying real capital. It authorizes progression to the next alpha-factory stage — walk-forward across distinct regimes and a Probability-of-Backtest-Overfitting check — and/or a minimal-size live confirmation. It does **not** authorize scaled live capital directly. The holdout answers one question (does the clean signal survive honest cost on out-of-sample paper data), not the deployment question.

## 7. Discipline clauses

- Do not view, compute, or summarize any holdout statistic before N = 100.
- Do not tune any parameter on the calibration slice; calibration is read-only for the §5 checks and friction estimation.
- Any change to runner code or config restarts the clock at N = 0 and requires a new frozen hash.
- Report secondary diagnostics (win rate, gross distribution, entry/exit adverse distributions, exit-reason mix, quarantine rate) for understanding — but they are **not** decision rules. Only §4 decides.
