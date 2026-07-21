# S34 Live Paper Validation — Pre-Registration

**Status:** locked at N=1 (one clean closed trade: P056), before the validation sample exists.
**Date:** 2026-06-11
**Strategy under test:** S34 — ETHUSDT liquidation-momentum BUY (taker entry), live paper.
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

## Amendment 1 — S34 Variant of Record (2026-06-12)

S34 of record for this validation is the single runner variant:

`ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30`

The `ETH_BUY_LIQ_LONG_200K_TP60_SL40_BE30` variant remains exploratory telemetry only and is excluded from the validation sample. The runner may continue recording both variants for research continuity, but the pre-registered validation counters, calibration checks, holdout decision, bootstrap CI, and DSR operate only on the 50K/TP120 variant.

Under this definition, one valid 50K/TP120 closed position equals one validation observation. The multi-variant duplicate/cluster issue is therefore removed from the validation sample without changing runner behavior, signal thresholds, TP/SL/BE thresholds, fill model, or cost attribution.

---

## Amendment 6 - SELL Continuation Exploratory Validation (2026-06-26)

The following SELL liquidation continuation routes are exploratory paper routes only. They are not part of the original 50K/TP120 pre-registration sample and do not count toward the N=100 holdout decision in Section 4.

- `ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40`
- `ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40`
- `SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30`

Rationale before forward collection:

- ETH 500K SELL->SHORT research sweep: TP60/SL40/BE40, median about +50.6 bps, second-half about +51.1 bps, positive days 23/28.
- ETH 1000K SELL->SHORT research sweep: TP80/SL40/BE40, median about +62.0 bps, second-half about +70.1 bps, positive days 21/26, lower no-fill rate than ETH 500K.
- SOL 200K SELL->SHORT research sweep: TP60/SL30/BE30, median about +49.9 bps, second-half about +50.9 bps, positive days 13/17.

Forward validation rule for each SELL route:

- Target initial N = 30 clean closed paper trades per route.
- A clean trade requires real BOOK_TICKER entry and exit fills and a reconciling cost decomposition.
- Pass conditions at N=30: median net bps > 0, top3-removed cumulative net bps > 0, and trades spread across at least 8 distinct UTC days.
- `NO_FILL_DATA`, cross-direction conflict events, and same-cascade interactions with BUY routes must be reported, not silently ignored.
- Passing N=30 only authorizes promotion to a full N=100 pre-registration. It does not authorize live capital.

No live order placement is authorized by this amendment.

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
