# CASCADE RESPONSE SURFACE — UNIFIED REGIME-CONDITIONAL POST-SELL-CASCADE MODEL — BOUNDED PREREGISTRATION V1

**Overall token:** `CASCADE_RESPONSE_UNIFIED_PREREGISTERED_PENDING_INDEPENDENT_REVIEW`
**Role of this document:** preregistration author ONLY. Nothing here is run, fit, or accepted.
**Date authored:** 2026-07-19 · **Symbol:** ETHUSDT · **Author phase:** authoring (next gate = independent fresh-context review)

> Freezes the design of ONE unified post-cascade response model **before** any forward result is seen.
> Grants **no** alpha, **no** deployability. Producing it does not authorize running it.

---

## 0. MOTIVATION & THE UNIFICATION CLAIM (carried from SYSTEM_STATE §148/§149)

Three separately-buried leads are, mechanistically, **one bet**: *SELL-liquidation cascade → a
mean-reverting price response whose direction/strength is regime-conditional.*

- `echo_30_90 + regime` — post-cascade **bounce (LONG)** in favorable regime (btc4h<0), 30–90 min window. Survivor, mc 0.004, but **never forward/holdout-validated** (§147).
- swing **reversal harvest** — post-cascade bounce as a base rate (**53–65 % WR**); buried by DATA_GAP(~2 mo)+PARAM (calibration/hold want *opposite* regime params — §148).
- `HOUR17` direct-**SHORT** — the regime where the bounce **fails → continuation (SHORT)**; died underpowered (FWER **p=0.0523**, 10h **p=0.0781** — just above 0.05; §148).

**Unification claim:** these are three cells of a single **E[post-cascade return | regime]** surface.
Fragmenting the fixed cascade population into per-lead sub-slices (N=26 / 93 / ~2 mo) is *itself* a
cause of their underpower. One surface, tested once, uses every cascade for the whole model.

**What unification does and does NOT buy (honest, per §149 correction):**
- It does **NOT** add data. The liquidation cascade population is at its historical max (~167 ungated
  independent cycles, Feb-15→Jul-02 minus the permanent May gap; liquidations are un-backfillable;
  §149 C1). History is **not** deepenable.
- It **DOES** buy **forward-data efficiency**: each un-burned forward cascade updates the *entire*
  regime surface instead of one fragment — ~3× the information per scarce forward cascade. That
  efficiency is the entire quantitative case for this model over running the three leads separately.

---

## 1. BURNED-DATA DECLARATION (this is the load-bearing discipline)

**The entire ~167-cycle historical population (Feb-15 → Jul-02, minus the May gap) is BURNED.** It was
used to discover echo, hour17, and reversal. Therefore:

- The historical cascades may be used **only** to FREEZE the model specification below (regime cuts,
  horizons, direction map). This is **in-sample design, disclosed, NOT validation.**
- Pre-specifying regime cells that we already know worked (hour≥17, btc<0) on burned data is
  **circular** and is **prohibited as evidence**. The frozen spec's *only* honest test is a **forward
  holdout** of cascades occurring **strictly after** the cutoff.

**`FROZEN_CUTOFF_TS`** = the DB max ingest timestamp at authoring, to be pinned exactly at review:
provisional `2026-07-19T12:00:00Z`. All data at or before it is burned. DB opened **read-only**
(`file:…?mode=ro` + `PRAGMA query_only=1`; **not** `immutable=1` — it corrupts reads on the live-append
DB, §149).

**Primary holdout:** the first **N_MIN = 40 independent post-cutoff cascade cycles** (day-clustered).
At ~5.7 cascades/month this is ~7 months (≈2027-02). No interim result is opened before 40 accrue.
If not reached → `CASCADE_RESPONSE_HOLDOUT_UNDERPOWERED`.

---

## 2. POPULATION & INDEPENDENCE (frozen)

- **Trigger:** ETHUSDT SELL liquidation, `notional ≥ 200_000` (matches the deployed anchor scale).
- **Cascade anchoring + clustering:** the **already-accepted** knowable-anchor min-gap construction
  (the same one behind the 167-cycle / anchor→cycle 0.66 recompute, §104). **No new clustering.**
- **Independence unit:** the independent cascade **cycle** (post min-gap merge), day-clustered for
  inference. Report raw anchors, cycles, and distinct days.
- **Excluded:** the May transport-outage window (2026-04-25 → 2026-06-06) — zero coverage, not zero
  cascades (§149).

---

## 3. REGIME DIMENSIONS (theory-driven, PRE-FROZEN, disclosed as informed by prior discovery)

The conditioning vector is fixed here and **may not be changed after the holdout opens** (§11). Each
dimension has a **theoretical** rationale; disclosure: the *specific cut points* are informed by prior
(burned) discovery, which is exactly why they are tested only forward.

| Dim | Definition (frozen) | Theory | Prior-discovery disclosure |
|---|---|---|---|
| **Session** | UTC hour bucket: ASIA (0–7), EU (7–13), US (13–24) | liquidity/participant mix differs | hour17 (US-late) was where bounce failed |
| **Trend** | sign of btc4h AND btc7d at T0 | cascade into downtrend ≠ into range | echo required btc4h<0 ∨ btc7d<0 |
| **Vol** | realized ETH vol pre-cascade, HIGH/LOW vs frozen median | high-vol cascades over/undershoot | reversal hold was regime-sensitive |

**Frozen direction map (the unification hypothesis, pre-committed):**
- Favorable-bounce cells (downtrend, non-US-late) → **LONG** candidate.
- Failed-bounce cells (US-late / adverse trend) → **SHORT** candidate.
- All other cells → **NO-TRADE** (the model is allowed, indeed required, to abstain).

No cell's direction may be flipped based on holdout results.

---

## 4. ESTIMAND (frozen)

Primary response: **fee-net return from T0 entry over a 6 h hold** (matches echo/hour17 hold; fee =
**5 bps** mark-fill, the standard research cost). Secondary horizons (descriptive only): 1 h, 2 h, 30–90 min echo window.

```
Primary estimand:  E_holdout[ net_return(cell.direction) ]  aggregated over pre-specified LONG/SHORT cells,
                   minus the NAIVE-ALL-CASCADE baseline (flat T0 entry, no regime) — which is a proven
                   TRUE_NULL (failure_archive #1, mc 0.88).  The regime surface must BEAT that null.
```

**Primary question:** does the pre-frozen regime-conditional direction map produce positive fee-net
expectancy on **un-burned** forward cascades, in excess of the naive-all-cascade null?

---

## 5. INFERENCE (frozen)

- Independence unit = cascade cycle; **day-clustered bootstrap** + **leave-one-day-out** + **paired
  day-level permutation** (sign-flip on cell direction).
- **Multiple-testing:** max-stat FWER across the regime-cell family (the grid is precisely where hour17
  failed; this is mandatory, not optional).
- Primary CI = day-clustered 95 %.
- The discovery-phase per-lead statistics (echo mc, hour17 p, reversal WR) are **NOT** re-admitted as
  evidence — they are burned design inputs.

---

## 6. SUCCESS / FAILURE CRITERIA

**SUPPORTED** requires ALL of:
1. holdout net expectancy of the frozen direction map > 0;
2. day-clustered CI95 lower bound ≥ 0;
3. beats the naive-all-cascade null (criterion of incremental regime value);
4. survives max-stat FWER across the regime grid;
5. LODO sign-stable across most days;
6. not carried by a single day / few cascades;
7. the profitable cells are the **pre-specified** ones (no TEST-selected cells).

**PROMISING_BUT_UNRESOLVED:** point estimate positive + direction map correct-signed, but CI includes 0 or forward-power insufficient.

**NOT_SUPPORTED:** expectancy ≤ 0, or gain only from a TEST-selected cell, or fails FWER, or does not beat naive-all-cascade.

---

## 7. ALLOWED VERDICT TOKENS

`CASCADE_RESPONSE_REGIME_CONDITIONAL_SUPPORTED` ·
`CASCADE_RESPONSE_REGIME_CONDITIONAL_PROMISING_BUT_UNRESOLVED` ·
`CASCADE_RESPONSE_REGIME_CONDITIONAL_NOT_SUPPORTED` ·
`CASCADE_RESPONSE_HOLDOUT_UNDERPOWERED`

This preregistration is **not** an alpha acceptance.

---

## 8. PROHIBITED ADAPTATIONS (once the holdout opens)

Change regime dims or cut points · add/remove a regime cell · flip any cell's direction on holdout
result · change the 6 h hold or 5 bps fee · select a horizon · drop the naive-null comparison · pool
pre/post-gap to inflate N · re-admit discovery statistics as evidence · extend/shorten the holdout ·
open before 40 independent post-cutoff cycles.

**No automatic validation start** on completion. Next gate = independent, fresh-context preregistration review.

---

## 9. INDEPENDENT-REVIEW HANDOFF

Reviewer must confirm (non-exhaustive):
- §0 unification claim honest; §149-corrected "no deepening, forward-efficiency only" carried without drift.
- §1 burned-data discipline sound: the 167 cascades are design-only; the ONLY validation is the forward holdout; circularity of pre-specified cells is disclosed, not hidden.
- §2 population reuses the accepted anchoring (no new clustering); May gap excluded.
- §3 regime dims + direction map are pre-frozen and internally consistent; the "informed by prior discovery" disclosure is explicit (this is the model's central epistemic risk).
- §4 estimand requires beating the naive-all-cascade TRUE_NULL; fee 5 bps; 6 h hold.
- §5 FWER across the regime grid is mandatory; discovery stats not re-admitted.
- §6–§8 criteria/prohibitions pre-committed and complete.

**Operator gate items:** (1) confirm the forward-holdout framing (accept that a clean verdict is ~7 months out at ~5.7 cascades/mo, the frequency ceiling); (2) sign the independent-review verdict before frozen-final.

**This turn performed no holdout, no fit, no PnL, no forward evaluation, no deployment. Author did not ratify.**
