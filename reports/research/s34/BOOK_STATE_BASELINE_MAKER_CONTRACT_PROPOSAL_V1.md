# BOOK STATE — BASELINE MAKER CONTRACT (P_BASE) — **PROPOSAL V1**

**Status token:** `BOOK_STATE_BASELINE_CONTRACT_PROPOSED_PENDING_OPERATOR_SIGNOFF_AND_REVIEW`
**Role of this document:** proposal author ONLY. Nothing here is accepted, frozen, run, or deployed.
**Resolves against:** `BOOK_STATE_H1_H2_BOUNDED_PREREGISTRATION_V1` §1.3 blocker
`BOOK_STATE_PREREG_BASELINE_CONTRACT_UNRESOLVED`.
**Date:** 2026-07-19 · **Symbol:** ETHUSDT · **Own quote size:** 0.5 ETH (Stage-1 `OWN`)
**Touches NO execution/risk code.** This is an *offline research-simulation* spec, not a live-engine change.

> The author does **not** ratify. This document proposes concrete values so the operator can decide
> §1.3 in one pass. Every risk-domain number below is a **proposal**, not a fact, and becomes part of
> the frozen prereg `P_BASE` **only after** operator sign-off **and** an independent review — and it
> **must be frozen before the holdout opens**, otherwise it is a post-hoc tunable parameter.

---

## 0. FRAMING — why this unblocks §1.3 without touching the `DOKUNMA` surface

The H1/H2 validation is an **offline holdout** over historical `book_ticker` + `agg_trades`
(§5 of the prereg). `P_BASE` and `P_H1` are **simulation policies scored on recorded data**, exactly
like the Stage-1 fill-quality harness — which the acceptance stamp already certified as read-only
("NO inventory accounting" was a *scope* statement, not a missing capability).

Therefore:

- Defining `P_BASE` = writing a **simulation baseline spec**, evaluated by an offline scorer. It does
  **not** modify `execution/`, `risk/`, `brain/`, or the live `s34_v_engine`/`s34_state_machine`
  executors. The `DOKUNMA` guardrail on *live code* is not crossed.
- The operator still owns the numbers, because inventory limit / unwind / stop are **risk-domain
  choices** even in simulation — they change what the primary estimand *means*.
- The one genuinely-live artifact that resembles a maker (`tools/s34_v_engine_live_executor.py`,
  a **single-side directional maker-LIMIT entry**: initial offset O20, cancel/replace to O5 after
  300 s if unfilled, reduce-only emergency stop) is **not** a two-sided quoting engine and is **not**
  proposed as `P_BASE`. It is cited only as a repo-grounded precedent for cadence/cancel/stop
  conventions, so the values below are anchored, not invented.

---

## 1. STRATEGIC FORK — operator picks ONE

| | **Option A (recommended)** | **Option B** |
|---|---|---|
| Define an offline sim `P_BASE` (this doc) | YES | NO |
| Primary estimand `Delta_EV_H1` (inventory-adjusted) | **computable** | not computable |
| Fallback scope | — | markout-level **secondary** estimands only (§7 of prereg) |
| Acceptance basis | full H1 success/failure criteria (§11) | **none** — secondary alone is not an acceptance basis |
| Cost | must freeze ~7 risk-domain values now | zero new decisions, but H1 stays unfalsifiable as a *risk module* |

**Recommendation: Option A.** The whole point of H1 is "does S3 attacked-side suppression raise the
maker engine's net expectancy" — that question is only answerable with an inventory/unwind model.
Option B leaves H1 permanently `PROMISING_BUT_UNRESOLVED` at best. The values below are deliberately
**minimal and conservative** so Option A costs the operator minutes, not a redesign.

---

## 2. PROPOSED §1.3 FIELD VALUES (offline sim `P_BASE`)

Each row: **recommended default** · alternative(s) · rationale/grounding. All are the operator's call;
the ★ risk-domain rows are the ones that most need an explicit operator decision.

| # | Field | Recommended default | Alternatives | Rationale / grounding |
|---|---|---|---|---|
| 1 | **Re-quote cadence** | **250 ms evaluation tick** (= primary action latency); a filled/expired quote is re-assessed and re-sent on the next tick if conditions still hold | event-driven per book-update, capped at latency | Matches the mechanism/action latency the prereg already froze (§8, 250 ms). `s34_v_engine`'s 300 s cadence is far too coarse for a 1 s-window microstructure test. |
| 2 | **Cancellation** | Cancel a resting quote when **(a)** lifetime 1000 ms expires unfilled (→ re-quote next tick), **(b)** H1 suppression is active on that side, or **(c)** the side's inventory limit is reached | add explicit adverse-move cancel (rejected: introduces a tuned trigger H1 doesn't need) | Lifetime 1000 ms is the Stage-1 primary (`PRIMARY_LIFE`); `s34_v_engine` has an explicit cancel/replace precedent. No offset/widening (consistent with H1). |
| 3 | ★ **Inventory limit** (max abs position) | **±2.0 ETH** (= 4 clips of 0.5) | conservative **±0.5 ETH** (single-clip, flatten-before-requote); loose **±5.0 ETH** | Small enough to keep the sim realistic; 4 clips gives skew room without letting inventory dominate PnL. **Pure risk-domain — operator number.** |
| 4 | ★ **Inventory skew** | **Hard one-sided throttle**: when signed inventory reaches **+50 % of the limit** on a side, stop *adding* to that side (only the reducing side quotes) until back under 50 % | linear **price** skew (rejected: adds a quote offset the prereg forbids for H1) | Simplest defensible rule; keeps the "cancel-only, no offset" invariant intact. Price-based skew would confound H1's suppression effect. |
| 5 | **Passive exit** | Acquired inventory is exited **passively via the opposite-side best-of-book quote** (natural maker mean-reversion), scored with the **same F3** fill model | dedicated passive exit at +k ticks (rejected: adds a tuned param) | This *is* the default maker unwind path; reuses the accepted F3 mechanics, no new degrees of freedom. |
| 6 | ★ **Market-unwind** (forced flatten) | Taker flatten if **(a)** inventory at hard limit AND still growing, or **(b)** inventory held **> 60 s** without passive exit. Cost = taker fee (§3) + half-spread slippage at best-of-book | thresholds **30 s / 120 s** as preregistered sensitivities | Bounds inventory risk without a tuned PnL trigger. `s34_v_engine`'s reduce-only emergency stop is the live analogue; offline we model it as this deterministic rule. **Risk-domain — operator number.** |
| 7 | ★ **Stop / risk guardrails** | Per-quote: none beyond lifetime. Portfolio: the §6 market-unwind **is** the backstop. **Report** drawdown / max-inventory / holding-duration but do **NOT** add a PnL-gated daily stop for the offline holdout | add a daily max-loss gate (rejected for offline: a results-contingent tuned param) | A PnL-gated stop chosen with holdout visibility would be a §14 prohibited adaptation. Keep the backstop mechanical and pre-committed. |

---

## 3. UNWIND COST MODEL — one missing input

- **Maker fee:** 1.0 bps — config-verified (`config/costs.py DEFAULT_MAKER_FEE_BPS`, env `MAKER_FEE_BPS`),
  already frozen in prereg §8, with 0 / 2 bps sensitivities.
- **Taker fee (for market-unwind, §2 row 6):** **NOT present** in `config/costs.py` (only the maker
  constant exists). This is the single external input the operator must supply so unwind cost is not
  fabricated. Options: (a) point to the live venue taker fee in `.env`/exchange config; (b) declare a
  frozen taker assumption + sensitivity band. Until supplied, the unwind term is **parameter-open** and
  the primary cannot be finalized.
- **Slippage on unwind:** half-spread at best-of-book at unwind time (deterministic from recorded book),
  no hidden L2 (consistent with F3).

---

## 4. WHAT THIS PROPOSAL DOES **NOT** GRANT

No maker alpha · no policy profitability · no deployability · no holdout result · no live-code change ·
no authorization to run the validation. It only *proposes* the `P_BASE` contract so the prereg's
primary estimand becomes definable.

---

## 5. GATE

1. **Operator decision:** pick the fork (§1), then sign / edit the ★ risk-domain values (§2 rows 3, 4,
   6, 7) and supply the taker input (§3). These are `execution/risk`-domain choices even in simulation.
2. **Independent review** of the resulting frozen `P_BASE` contract (fresh-context, read-only) — same
   discipline as the prereg chain.
3. Only after 1 + 2: fold the accepted `P_BASE` into the prereg as §1.3 RESOLVED, clear the
   `BOOK_STATE_PREREG_BASELINE_CONTRACT_UNRESOLVED` token, and **freeze it before opening the holdout**.

**Author did not ratify. No holdout, no simulation, no PnL, no live-code change performed.**
