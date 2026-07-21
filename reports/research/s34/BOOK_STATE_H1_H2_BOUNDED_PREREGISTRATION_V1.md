# BOOK STATE H1/H2 — BOUNDED PREREGISTRATION (V1)

**Overall token:** `BOOK_STATE_H1_H2_VALIDATION_PREREGISTERED_CORRECTED_AWAITING_REREVIEW`
(was `..._PENDING_INDEPENDENT_REVIEW`; independent review verdict `BOOK_STATE_H1_H2_PREREG_CORRECTIVE_REQUIRED`, 1 blocking finding fixed — see §16 CORRECTION LOG)
**Blocking condition raised at authoring:** `BOOK_STATE_PREREG_BASELINE_CONTRACT_UNRESOLVED` (see §1)
**Role of this document:** preregistration author ONLY. Nothing here is run, simulated, or accepted.
**Date authored:** 2026-07-19 · **Correction C1:** 2026-07-19 · **Symbol:** ETHUSDT · **Phase:** corrected, awaiting independent re-review (author does not ratify)

> This preregistration freezes the *next* validation experiment **before** any holdout, PnL,
> markout, or policy result is seen. It grants **no** alpha, **no** deployability, **no**
> profitability. Producing it does not authorize running it.

---

## 0. ACCEPTED BASIS (carried forward verbatim, not re-opened)

Canonical starting point: **`BOOK_STATE_STAGE1_ACCEPTED`** (operator sign-off 2026-07-18;
chain: implementation → `BOOK_STATE_STAGE1_INDEPENDENT_REVIEW_ACCEPTED` → operator acceptance).

Accepted evidence boundary (geometry / methodology / evidence ONLY):

- S0–S5 book-state geometry is causal and reproducible.
- **S3 resilient-refill** — the only robustly incremental-adverse state vs matched pre-state controls:
  - BID incremental markout ≈ **−0.584 bps @1s**
  - ASK incremental markout ≈ **−0.410 bps @1s**
  - BID attribution residual ≈ **−0.397 bps** (survives vol + spread + imbalance)
- **S2 liquidity-vacuum** — the same incremental adverse geometry is **NOT** present (clean, well-powered null).
- Fill-ordering mechanism (F3 primary, n_full):
  - **S3**: move-before-fill **1705** ≫ fill-before-move **293** (price walks; stale adverse fills)
  - **S2**: fill-before-move **1044** ≫ move-before-fill **352** (fill-before-move benign; liquidity demand)

These are **discovery geometry**. The following are explicitly **NOT** accepted and remain open:
maker alpha · policy profitability · S3 gate deployability · S2 quoting profitability · any holdout result.

> **Disclosure note (carried, not adjudicated):** the accepted Stage-1 record contains a minor
> internal labeling difference for the S2 non-dominant ordering count (acceptance stamp "1044:352"
> vs discovery report "1044 ≫ 376"). The mechanistically load-bearing fact — S2 dominated by
> fill-before-move (benign) — is consistent across both. As author I carry the ACCEPTED BASIS
> numbers exactly as the operator stated them and do **not** re-derive or resolve the 352/376 count.

### 0.1 F1–F4 disclosures (carried into this preregistration in full)

- **F1** — the "1 s canonical merge" did **not** actually reduce anchors at the 2 s discovery grid
  (0 anchors removed). Independence is carried by **day-cluster** inference, not by the 1 s merge.
  → In this validation, the claim "1 s merge provides independence" is **prohibited** (§10).
- **F2** — matched-control reuse in discovery was high (e.g. 4439/9020 at cap 6) and SMD was not
  tabulated. → A **control-balance SMD table is MANDATORY** in this validation phase (§9).
- **F3** — legacy `spread_bps` / `spread_z` config fields are **dead config** (superseded by ticks).
  → They must remain unused; no re-activation (§4, §14).
- **F4** — the S3 result is **fill-conditional** on a rare (~5.9 %) fill event; it is an
  adverse-tail lower bound, **not** an unconditional all-quote-opportunities alpha claim.

---

## 1. BASELINE MAKER CONTRACT — **PARTIALLY UNRESOLVED**

### 1.1 Finding that governs this section

The Stage-1 "maker" is a **read-only fill-quality simulation harness** (own passive quote used to
*measure* markout). It is **not** a deployed maker engine. Repository inspection confirms:

- The live execution surface (`tools/s34_state_machine_live_executor.py`, `execution/`, `risk/`,
  `brain/`) is a **directional liquidation-cascade scalper**. A repo-wide search for a quoting /
  market-making engine (`place_quote`, `post_only`, `maker_engine`, `inventory_skew`, `unwind`, …)
  returns **no** deployed maker engine.
- The Stage-1 acceptance stamp states verbatim: **"NO holdout, NO policy PnL, NO inventory
  accounting, NO deployment performed."**

Therefore an "existing maker engine" as a single frozen, deployable object **does not exist**, and
the components required to compute the **primary estimand** (§6 — *inventory-adjusted* net PnL) —
inventory limit, inventory skew, passive exit, market-unwind condition, stop/risk guardrails —
were **never defined** by any accepted artifact.

Per §1's non-fabrication rule, the author **must not invent** these. The disciplined outcome is:
freeze the fields Stage 1 pins byte-exact, and raise the token below for the rest.

### 1.2 RESOLVED fields (byte-exact from accepted Stage-1 artifacts)

| Field | Frozen value | Source |
|---|---|---|
| Quoted symbol | ETHUSDT | `br_lib.py` `SYM` |
| Quote sides | BID + ASK (paired) | Stage-1 census (both sides) |
| Placement | **best-of-book only** (no below-best; below-best `NOT_IDENTIFIABLE`) | `eval_cell`, F3 |
| Own quote size | **0.5 ETH** | `br_lib.py` `OWN = 0.5` |
| Quote lifetime | {500, 1000, 5000} ms; **primary 1000 ms** | `br_lib.py` `LIFETIMES`, `PRIMARY_LIFE` |
| Latency | {50, 100, 250, 500} ms; **primary 250 ms** | `br_lib.py` `LATENCIES`, `PRIMARY_LAT` |
| Fill model | F3 queue-conservative primary; F2 secondary; F1 diagnostic-only | `eval_cell` (§8) |
| Fee schedule | **1.0 bps maker** (primary), 0 / 2 bps sensitivities | `config/costs.py DEFAULT_MAKER_FEE_BPS` (env `MAKER_FEE_BPS`, default 1.0); Stage-1 fee sweep 0/1/2 |
| Markout snapshot staleness | placement ≤ 2000 ms, mid ≤ 3000 ms | `br_lib.py` `STALE_SNAP`, `STALE_MID` |

### 1.3 UNRESOLVED fields → `BOOK_STATE_PREREG_BASELINE_CONTRACT_UNRESOLVED`

The following baseline-contract fields are **undefined** in every accepted artifact and are **not**
fabricated here. They are also **execution/risk-domain** objects behind the `DOKUNMA` guardrail,
requiring operator sign-off to define:

- **Update / re-quote cadence** of an actual engine (Stage 1 used a fixed research grid + quote
  lifetime, never an engine cadence).
- **Cancellation behavior** (when/why a resting quote is pulled outside the frozen lifetime).
- **Inventory limit** (max absolute position).
- **Inventory skew** (quote adjustment as inventory builds).
- **Passive exit** (how an acquired position is exited).
- **Market-unwind condition** (forced flatten trigger and cost model).
- **Stop / risk guardrails** (per-quote and portfolio-level).

### 1.4 Consequence for the primary estimand

`Delta_EV_H1` (§6) is defined as *inventory-adjusted net PnL(P_H1) − inventory-adjusted net
PnL(P_BASE)*. Without a defined inventory / passive-exit / unwind model, **inventory-adjusted net
PnL is not computable**. The `inventory / unwind effect` decomposition term (§7) is likewise
undefined. Therefore:

> **The holdout for H1's primary estimand cannot be opened until the operator resolves §1.3.**
> This preregistration freezes everything else and hands the §1.3 gap to the operator as the single
> blocking item. Until it is resolved, only the **markout-level** secondary estimands (§7:
> avoided adverse markout, missed profitable fills, spread-capture loss, fill/cancel-race,
> quote uptime, fill count) are computable — and those alone are **not** an acceptance basis.

`P_BASE` = the existing maker engine (once §1.3 is resolved).
`P_H1` = identical to `P_BASE` except the attacked-side quote is temporarily suppressed during S3.

---

## 2. H1 — S3 ATTACKED-SIDE SUPPRESSION (exact causal definition, frozen)

When S3 resilient-refill is classified **causally for the first time** on a book side:

- If **SELL-attacked** book side → suppress the **BID** quote.
- If **BUY-attacked** book side → suppress the **ASK** quote.
- The **opposite-side** quote keeps baseline behavior.
- **No** quote offset and **no** widening — the suppressed quote is only **cancelled / not re-sent**.

Attacked-side mapping (from Stage-1 adverse-pressure convention `adv_BID = −imb`, `adv_ASK = +imb`):
the attacked side is the maker side facing adverse selection in S3 — i.e. the side whose adverse
imbalance is realized. Frozen exactly as Stage-1 `assign_states` computed it; no new mapping.

**Primary action latency:** 250 ms. **Sensitivities:** 100 ms, 500 ms.

**Primary re-entry rule:**
- If S3 is **not** observed for **two consecutive** causal evaluations, the attacked-side quote re-opens.
- Evaluation cadence = **baseline engine cadence** (resolved with §1.3).
- **Maximum suppression duration = 5 s.** If still S3 at 5 s, the event is reported **separately** as
  `DATA/STATE unresolved`; the duration is **never** extended based on results.

**Primary markout mechanism window:** 1 s.
**Secondary descriptive windows:** 500 ms, 2 s, 5 s.
**Primary policy evaluation window:** event start → **+5 s**.

---

## 3. H2 — S2 EXCLUSION RULE (frozen)

H2 is **not** an independent profit hypothesis. Exact rule:

> S2 liquidity-vacuum **alone does not trigger** the H1 suppression gate.

During S2: baseline maker behavior is preserved; **no** automatic suppression is applied. Only if a
**causal S2 → S3 transition** occurs does H1 engage **at that moment** (on S3, not on S2).

H2 makes **no** claim that S2 quoting is profitable, that S2 carries spread-capture alpha, or that
S2 is safe in every regime. H2's sole purpose: prevent the S3 risk gate from being needlessly
widened onto S2.

---

## 4. STATE FREEZE (S2 & S3 byte-exact from accepted Stage-1 artifacts)

State definitions are frozen **exactly** as in `br_lib.py` (`assign_states`, `qbin`,
`compute_state_feat`, `_side_stats_1w`) and the frozen bins `br_bins.json`. **No** new calibration,
**no** new quantiles, **no** new features. SHA-256 in the hash manifest (`§ artifact 3`).

**Frozen mechanics (recorded for reviewer; source of truth = hashed files):**
- Canonical window **W = 1000 ms**; anchor = last book update ≤ anchor ts.
- Spread in **ticks** (`TICK = 0.01`), stationary; `normal_spr = ticks ≤ 1`, `wide_spr = ticks ≥ 2`.
- Imbalance sign: `adv_BID = −imb`, `adv_ASK = +imb` (adverse to the maker side).
- Quantity key: `bid_qty` for BID, `ask_qty` for ASK.
- `qbin`: 6 bands from 5 edges `[Q25,Q50,Q75,Q90,Q99]`; NaN/None → −1.
- Frozen bin edges (`br_bins.json`, self_sha256 `b443a97c…`):
  `dep_frac [.009271,.127985,.482031,.749859,.974995]` ·
  `dep_vel [11.10225,58.1955,270.14425,998.858,3211.4662]` ·
  `refill_vel [8.842,51.9555,276.08175,1719.337,6431.57665]` ·
  `qty [47.18125,95.9635,164.29575,262.9875,531.7708]` ·
  `cr_cnt [4,8,23,65,197]` · `consec [0,0,0,1,4]` ·
  `adv [-.531428,0,.531428,.853064,.992398]`. Spread NOT binned (ticks).
- **S2 (LIQUIDITY_VACUUM):** `consuming ∧ thin ∧ (wide_spr ∨ adverse_mp) ∧ weak_refill`.
- **S3 (RESILIENT_REFILL):** `consuming ∧ refilling ∧ stable_mp`.
- Predicate definitions (frozen): `consuming = b_depv≥3`, `consumed = b_depf≥2`,
  `refilling = b_refv≥3 ∧ rd_ratio≥0.8`, `weak_refill = b_refv≤1 ∨ rd_ratio<0.3`,
  `thin = b_qty≤1`, `adverse_mp = b_adv≥3`, `stable_mp = b_adv≤2`, `high_osc = b_cr≥4`.
- **Precedence (exact, top-down):** S2 → S4 → S3 → S1 → S5 → S0 (first match wins; single tie-break by order).
- **Exit condition:** state is re-evaluated each causal step from past-only book; no persistence /
  hysteresis beyond the frozen predicates. (H1's suppression persistence is a *policy* layer §2,
  not a state change.)

Dead config **F3**: `spread_bps` / `spread_z` remain unused (not reactivated).

---

## 5. VALIDATION POPULATION & BURNED-DATA DECLARATION

**Burned cutoff:** `FROZEN_CUTOFF_TS = 1784320049654 = 2026-07-17T20:27:29.654Z`.
All data at **or before** this timestamp is burned for: discovery, calibration, model development,
and this preregistration's design. DB `D:/eclipse_scalper/data/microstructure.db` opened
**read-only** (`file:…?mode=ro` + `PRAGMA query_only=1`); the cutoff (not file size) is the
immutability anchor.

**Primary untouched holdout:** the first **40 usable, independent calendar days** that occur
**strictly after** `FROZEN_CUTOFF_TS`.

**Usable day** (all must hold): joint `book_ticker` + `agg_trades` coverage · no serious collection
gap · state / fill reconstruction feasible · minimum quote opportunity present.

**Holdout accumulation rule:** days accumulate forward from the cutoff; a day enters the holdout only
once complete and usable. **No interim PnL / markout / policy result is opened before 40 usable days
are complete.** If 40 usable days are not reached → verdict `BOOK_STATE_H1_H2_HOLDOUT_UNDERPOWERED`.

---

## 6. PRIMARY ESTIMAND

H1 is tested as a **risk module**, not a standalone alpha.

```
Delta_EV_H1 = inventory-adjusted net PnL(P_H1) − inventory-adjusted net PnL(P_BASE)
```

Paired counterfactual: same event, **same market path**, difference of the two policies.

**Primary question:** during S3, does attacked-side suppression **increase the net expectancy** of
the existing maker engine?

H1 is **NOT** required to: be absolutely PnL-positive alone · beat a NO-QUOTE strategy · make the
whole engine positive · be symmetric in BUY vs SELL magnitude · be positive at every horizon.

> **Computability caveat (from §1.4):** the primary estimand requires the §1.3 inventory / unwind /
> exit model. It is frozen here but **cannot be evaluated** until the operator resolves that gap.

---

## 7. SECONDARY ESTIMANDS & MANDATORY DECOMPOSITION

Report separately: avoided adverse markout · missed profitable fills · spread-capture loss ·
fill-before-cancel · cancel-before-fill · quote uptime · fill count · inventory exposure · inventory
holding duration · unwind cost · max inventory · drawdown contribution.

**Mandatory decomposition (must sum to the primary):**
```
  avoided adverse-selection
− missed profitable fills
− cancel/fill-race loss
− inventory/unwind effect
− fee effect
= Delta_EV_H1
```

**H2 secondary comparison (preregistered, NOT run this turn):** `P_H1_S3_ONLY` vs `P_H1_S3_PLUS_S2`.

---

## 8. FEE / LATENCY / FILL-MODEL TABLE (frozen)

| Item | Primary | Secondary / sensitivity | Notes |
|---|---|---|---|
| Fill model | **F3 queue-conservative** | F2 trade-through | F1 = upper-bound diagnostic ONLY, **not** acceptance evidence |
| F3 mechanics | best-of-book only; visible queue-ahead at placement; own 0.5 ETH; cumulative opposite aggressive volume; **no hidden L2** | — | `eval_cell` thr `Q+OWN` |
| Latency | **250 ms** | 100 ms, 500 ms | mechanism / action latency |
| Fee | **1.0 bps maker** (config-verified) | 0 bps, 2 bps | primary ruling must not rest on a single fee assumption |
| Deep-bid / below-best | — | — | `NOT_IDENTIFIABLE`, out of scope |

Fee source verified in repo: `config/costs.py` → `DEFAULT_MAKER_FEE_BPS` (env `MAKER_FEE_BPS`,
default **1.0**). Because a single canonical live value exists but Stage-1 swept 0/1/2, the primary
ruling is reported at 1 bps **and** must be shown robust across {0, 1, 2} bps.

---

## 9. CONTROLS & MANDATORY CONTROL-BALANCE (SMD) TABLE

Primary estimand is a **paired same-path** comparison, so a matched control is **not** a primary
acceptance requirement. If a secondary mechanism-attribution control is used, covariates are
**pre-state only**: UTC hour · past volatility · past spread (ticks) · past ETH return · past imbalance.

**Prohibited as matching covariates** (they ARE the S3 mechanism): refill · depletion ·
move-before-fill · post-state microprice movement · future fill · future markout.

**Mandatory control-balance SMD table** (required by disclosure **F2**): standardized mean
difference **before** matching · SMD **after** matching · control reuse · unmatched rate ·
event–control time distance · same-day reuse · top-control concentration.

---

## 10. INDEPENDENCE & INFERENCE (frozen)

**Primary inference unit: calendar day.** Also report: raw state hits · canonical S3 events · fill
opportunities · actual fills · distinct days · effective non-overlapping events.

Canonical event merge = the Stage-1-accepted construction. **Prohibited claim (F1):** "the 1 s merge
provides independence." Independence is established by: **day-clustered bootstrap · leave-one-day-out
· paired day-level aggregation.**

Primary CI: **day-clustered 95 %**. Secondary: 90 %. Permutation: **day-level paired**.

**Multiple-testing family (exactly these four, nothing else):**
1. H1 primary 250 ms · 2. latency sensitivity 100 ms · 3. latency sensitivity 500 ms · 4. H2 exclusion comparison.

The discovery-phase state × side × horizon grid is **NOT** re-admitted into the testing family.

---

## 11. SUCCESS / FAILURE CRITERIA

**H1 SUPPORTED** requires ALL of:
1. `Delta_EV_H1` point estimate > 0
2. day-clustered CI95 lower bound ≥ 0
3. avoided adverse-selection **exceeds** missed profitable fills
4. contribution preserved after inventory / unwind
5. preserved at 250 ms latency
6. leave-one-day-out sign stable across most days
7. result not carried by a single day or a few events
8. preserved under the **F3** primary fill model

**H1 PROMISING_BUT_UNRESOLVED:** point estimate positive **and** mechanical decomposition in the
right direction, **but** CI includes 0 or day-power insufficient.

**H1 NOT SUPPORTED:** `Delta_EV ≤ 0` · or gain comes only from fee/churn accounting · or missed
profitable fills erase the gain · or it disappears after inventory / unwind.

**H2 SUPPORTED** iff adding S2 to the suppression gate **reduces or fails to improve** net
contribution vs the S3-only policy. H2 may **never** be written as "S2 quoting is alpha."

---

## 12. ALLOWED VERDICT TOKENS

**H1:** `BOOK_STATE_H1_POLICY_CONTRIBUTION_SUPPORTED` ·
`BOOK_STATE_H1_POLICY_PROMISING_BUT_UNRESOLVED` ·
`BOOK_STATE_H1_POLICY_NOT_SUPPORTED` · `BOOK_STATE_H1_HOLDOUT_UNDERPOWERED`

**H2:** `BOOK_STATE_H2_EXCLUSION_SUPPORTED` · `BOOK_STATE_H2_EXCLUSION_UNRESOLVED` ·
`BOOK_STATE_H2_EXCLUSION_NOT_SUPPORTED`

**Overall (this document):** `BOOK_STATE_H1_H2_VALIDATION_PREREGISTERED_PENDING_INDEPENDENT_REVIEW`.
**Authoring blocker:** `BOOK_STATE_PREREG_BASELINE_CONTRACT_UNRESOLVED` (§1.3).

This preregistration is **not** an alpha acceptance.

---

## 13. REQUIRED PREREGISTRATION ARTIFACTS (index)

1. Human-readable preregistration MD — **this file**.
2. Machine-readable contract — `BOOK_STATE_H1_H2_BOUNDED_PREREGISTRATION_V1.json`.
3. State-artifact hash manifest — `BOOK_STATE_H1_H2_STATE_ARTIFACT_HASH_MANIFEST_V1.json`.
4. Baseline maker contract — §1 (partially UNRESOLVED, token raised).
5. Burned-data declaration — §5.
6. Holdout accumulation rule — §5.
7. Primary/secondary estimand table — §6, §7.
8. Fee / latency / fill-model table — §8.
9. Control-balance SMD requirement — §9.
10. Allowed verdict tokens — §12.
11. Prohibited adaptations list — §14.
12. Independent-review handoff — §15.

---

## 14. PROHIBITED ADAPTATIONS (once the holdout is opened)

Forbidden after holdout opening: change S3/S2 thresholds · change state features · change quote
action · change suppression duration · change re-entry condition · change primary latency · select a
fee · select a fill model · drop bad days · create new subgroups · flip policy direction based on
BUY/SELL result · select a markout horizon · extend/shorten the holdout · add any policy beyond
H1/H2.

**No automatic validation start** on completion of this preregistration. The next gate is a full
independent, fresh-context preregistration review.

---

## 15. INDEPENDENT-REVIEW HANDOFF

**State of chain:** authoring complete → **awaiting independent fresh-context review** (read-only).
The reviewer receives: this MD, the JSON contract, the hash manifest, and read-only access to the
Stage-1 artifacts (session `39231408` scratchpad) whose SHA-256 are pinned in artifact 3.

**Reviewer must confirm (non-exhaustive):**
- §0 accepted basis and F1–F4 are carried without drift.
- §1 baseline-contract resolution is honest: resolved fields match Stage-1 byte-exact; UNRESOLVED
  fields are genuinely undefined (no fabricated inventory/unwind model); the token is warranted.
- §4 state freeze matches the hashed `br_lib.py` / `br_bins.json` exactly.
- §5 burned-cutoff and 40-usable-day holdout rule leak no post-cutoff information into design.
- §6–§11 estimands, decomposition, controls (SMD mandatory), inference, and criteria are internally
  consistent and pre-committed.
- §14 prohibitions are complete.

**Operator gate items:**
1. Resolve §1.3 (baseline inventory / unwind / exit / risk contract) — required before any
   inventory-adjusted PnL holdout may open. This touches `execution/` / `risk/` (guardrail
   `DOKUNMA`) and needs operator sign-off.
2. Sign off the independent review verdict before this preregistration is treated as frozen-final.

**This turn performed no holdout, no simulation, no PnL/markout, no threshold change, no new state,
no deployment. Author did not ratify.**

---

## 16. CORRECTION LOG

### C1 — 2026-07-19 · S2 fill-ordering label inversion (BLOCKING, fixed)

**Source:** independent fresh-context read-only review (verdict `BOOK_STATE_H1_H2_PREREG_CORRECTIVE_REQUIRED`,
1 blocking + 3 non-blocking findings; hash-drift CLEAN 10/10).

**Blocking finding:** §0 (MD) and the JSON contract labeled S2 fill-ordering as
`move-before-fill 1044 ≫ fill-before-move 352`. The hashed Stage-1 source (`BR_STAGE1_DISCOVERY_REPORT.md`
§ "S2 vacuum: fill-before-move ≫ move-before-fill (1044:352) → fills are benign"; acceptance stamp
"S2 1044:352") establishes S2 as **fill-before-move 1044 ≫ move-before-fill 352 (benign)**. The digits
were correct but the two labels were swapped — inverting the exact benign/adverse ordering that H2 (§3)
depends on. The MD line was also internally self-contradictory (labeled 1044 "move-before-fill" while its
own parenthetical said "fill-before-move benign").

**Fix applied (digits unchanged, labels corrected only):**
- §0 line: `S2: fill-before-move 1044 ≫ move-before-fill 352 (fill-before-move benign; liquidity demand)`.
- JSON `accepted_basis.fill_ordering_f3_nfull.S2`: `{"fill_before_move": 1044, "move_before_fill": 352}`.
- S3 was already correct (`move-before-fill 1705 ≫ fill-before-move 293`) — untouched.
- The 352-vs-376 non-dominant-count disclosure (§0 note) is unaffected and retained.

**Non-blocking findings carried forward (NOT changed, disclosed for re-reviewer):**
- N1 (F2): the "4439/9020 at cap 6" reuse numerator is not independently confirmable from
  `br_ctrl_map.json` (9020 and reuse_cap=6 are); the mandatory-SMD consequence stands regardless.
- N2 (§1.1): "no deployed maker engine" is substantively correct (maker-primitive grep = 0), but a
  single-side directional maker-LIMIT entry exists (`tools/s34_v_engine_live_executor.py`). It is **not**
  the two-sided P_BASE quoting engine, so the §1.3 `UNRESOLVED` token stands; §1.1 wording may optionally
  be narrowed at operator discretion.

**New phase token:** `BOOK_STATE_H1_H2_VALIDATION_PREREGISTERED_CORRECTED_AWAITING_REREVIEW`.
**Next gate:** independent fresh-context **re-review** (separate reviewer) → then operator sign-off.
Author does not ratify the correction.
