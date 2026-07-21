# S34 ECHO FORWARD EVALUATOR — DESIGN / PRE-REGISTRATION V1

**Status:** `DESIGN_ACCEPTED` → Part 1 impl → review=CORRECTIVE(B1/B2/B3) → correction → re-review=**ACCEPT** → **Part 1 `MEASURED_COST_LEDGER_V2_ACCEPTED`** (2026-07-20; full gated chain complete, 10/10 tests, corrected code restarted live). **Part 2 (evaluator) = `ACCEPTED`** (2026-07-20): `tools/research_s34_echo_forward_evaluator.py` + `tests/test_echo_forward_evaluator.py` (8/8). Independent review = **ACCEPT** (no blockers; frozen-rule fidelity, no-overlap, seeded bootstrap, quarantine exclusion, gate logic, no lookahead, fail-closed all verified). 4 NON-BLOCKING: NB1 REFUTE also N-gated (safe interpretation — ratify), NB2 harden fail-closed if baseline missing, NB3 add DEGRADED/boundary tests, NB4 quarantine_rate excludes unmeasurable-but-not-quarantined rows. Built while forward N=0 (safest anti-lookahead timing). NOT wired into start_eclipse (on-demand). **FULL MACHINE (Part 1 + Part 2) COMPLETE.**
**Date:** 2026-07-20 · **Author:** Opus 4.8 [1M] · **Scope:** read-only, research-only
**Relates:** SYSTEM_STATE §166/§167/§168/§169/§170 · memory `project_echo_hour17_tailgate_jul2026`,
`project_echo_forward_lead_jul2026`, `project_canonical_alpha_status_jul2026`

---

## 0. Purpose (one line)
A read-only machine that, each run, scores echo_30_90+regime (CAUSAL) **forward** paper results
against a frozen burned baseline across all timeframes × stops × the hour17 tail-gate, and emits a
per-cell verdict (`ACCUMULATING / CONFIRMED / DEGRADED / REFUTED`) under **pre-registered** rules —
turning the §168/§169/horizon-sweep investigation into an automatic lead→alpha adjudicator.

**The machine produces EVIDENCE, it does not BLESS/DEPLOY.** Promotion stays with operator + the gated
review chain. No trade/order/executor/scheduler control. Fail-closed.

## 1. Inputs (strictly separated — no re-mine)
- **FORWARD (the only evidence):** `reports/shadow/hold_horizon_forward_ledger.jsonl` — echo_causal
  OPEN/RESOLVE rows, 6 horizons × {nostop,−150,−300}, each RESOLVE tagged `hour_utc`,`is_hour17`
  (§170), `net_bps`,`net_bps_s150`,`net_bps_s300`. Currently N=0 (post-2026-07-20 only).
- **BURNED BASELINE (reference only, never counted as forward):** `S34_HOLD_HORIZON_SWEEP.json`
  (frozen at generated_utc 2026-07-20T13:29:43Z). Used ONLY to set the comparison bar.

The evaluator must NEVER recompute the burned universe as if it were forward. Burned = read-only
constant. Forward = the accumulating ledger. Hard wall between them.

## 2. Pre-committed PRIMARY hypotheses (frozen — everything else is descriptive/secondary)
To avoid the machine becoming a 36-cell forking-path scanner, only TWO cells are confirmatory:

- **P1 — base echo edge @ 6h nostop.** In-sample honest anchor = **no-overlap avg +19.3 bps net**
  (noovN=57, noovWR=56.1). (6h chosen because in-sample mc=0.000, WF5/5, best short-horizon noov sum
  +1099. This selection is itself burned — that's WHY it must be forward-tested, not trusted.)
- **P2 — hour17 tail-gate (§169).** PRIMARY METRIC = **tail-rate** (only §169-significant quantity,
  perm p=0.018; avg-uplift was n.s. p=0.156). In-sample: hour≥17 tail-rate = 0/28; hour<17 = 14/90 ≈15.6%.

All other cells (2h/4h/12h/24h/48h × 3 stops × all/gated) are computed and displayed but labeled
`SECONDARY_DESCRIPTIVE`; to be read confirmatorily they require Bonferroni α/36 (see §5).

## 3. Frozen decision rules (per cell) — THE numbers requiring operator ratification
All verdicts use **no-overlap (noov)** obs (raw N is overlap-inflated — §167). Bootstrap CI = 95%,
10k resamples, seed=1234 (fixed for reproducibility/resume-safety).

| Verdict | Condition (frozen) |
|---|---|
| `ACCUMULATING` | noovN < **N_MIN=20** for the cell → no confirm/refute allowed yet (power gate) |
| `CONFIRMED` | noovN≥20 AND avg-net bootstrap **CI lower-bound > 0** AND point-avg ≥ **0.40 × in-sample noov-avg** |
| `DEGRADED` | noovN≥20 AND avg-net > 0 but CI includes 0 OR point-avg < 0.40× in-sample → "alive but weak, keep accumulating" |
| `REFUTED` | noovN≥20 AND avg-net bootstrap **CI upper-bound ≤ 0** (edge gone net-of-cost) |

**P2 (tail-gate) overlay verdict (frozen):**
| | Condition |
|---|---|
| `GATE_CONFIRMED` | hour≥17 cell noovN≥20 AND forward tail-rate ≤ **5%** AND hour<17 forward tail-rate ≥ **10%** (gate protects) |
| `GATE_REFUTED` | hour≥17 forward tail-rate ≥ **10%** (≈ base rate → gate gives no protection) |
| `GATE_ACCUMULATING` | otherwise |

`net-of-cost` = FEE 5bps (both sides) + funding drag per hold (already in ledger `net_bps`).
`in-sample noov-avg` per cell taken from the frozen baseline (P1 primary = +19.3 → CONFIRM point-bar ≈ +7.7).

## 4. Integrity guards (MUST — credibility depends on these)
1. **Outage/gap quarantine (§166 lesson):** reject any RESOLVE whose wall-clock hold deviates from its
   intended horizon by **>5%** (`|actual_hold_ms − target_ms|/target_ms > 0.05`) → tagged
   `QUARANTINED_OUTAGE`, excluded from all stats. (This is exactly the fake +900bps@~49h artifact class.)
2. **No-overlap is primary** everywhere; raw N shown only as context, never a verdict basis.
3. **Fail-closed:** ledger missing/empty/malformed → emit `NO_DATA`/`ACCUMULATING`, NEVER fabricate a number.
4. **Underlying-feed gap check:** if the microstructure feed had a gap spanning an OPEN→RESOLVE window,
   quarantine that trade (`QUARANTINED_FEEDGAP`).
5. **Determinism:** seeded bootstrap; no `Date.now()`/`random` without seed; same ledger → same scorecard.

## 5. Multiple-comparison handling
- Confirmatory = P1, P2 only (pre-committed). Family-wise error controlled by pre-commitment, not scanning.
- Secondary cells displayed with per-cell stats + `SECONDARY_DESCRIPTIVE` label; a footnote states the
  Bonferroni threshold (α=0.05/36 ≈ 0.0014) required to read any secondary cell as confirmatory.

## 6. Outputs
- `reports/research/s34/ECHO_FORWARD_SCORECARD.json` + `.md` — per-cell forward stats vs baseline + verdict,
  regenerated each run (idempotent, run-once; NO daemon, NO parallel process — guardrail).
- Read-only panel on leads monitor dashboard `:8771` (`tools/s34_leads_monitor_dashboard.py`) surfacing
  the scorecard (secondary/diagnostic surface; does NOT supersede canonical dashboard).
- Verdict TRANSITIONS (e.g. ACCUMULATING→CONFIRMED/REFUTED) surfaced to operator; recorded to
  SYSTEM_STATE only on material transition (autonomy-bounded governance recording).

## 7. What the machine CANNOT do (explicit)
- Cannot bless/deploy/size/trade. Cannot change any threshold, gate, horizon, or the live executor.
- Cannot re-select the horizon/stop from forward data (P1/P2 are frozen here; a NEW primary needs a
  NEW versioned pre-registration + its own gated review).
- Cannot recompute burned data as forward.

## 8. Echo-overall kill / refutation conditions (frozen)
- If P1 (6h nostop) reaches `REFUTED` on ≥N_MIN noov obs → echo base edge falsified forward; escalate.
- If P2 reaches `GATE_REFUTED` → §169 hour17 tail-gate claim dies (as pre-registered in §169).
- If quarantine fraction > 30% of resolved trades sustained → the FORWARD ITSELF is untrustworthy
  (infrastructure problem, not strategy signal) → halt verdicts, flag operator (repeat of §166 risk).

## 9. Architecture (for the implementation phase — NOT built yet)
- Standalone `tools/research_s34_echo_forward_evaluator.py`, read-only, run-once, py_compile clean.
- Pure-stdlib + numpy for bootstrap; reads two files; writes two files; optional dashboard-panel read.
- Unit test `tests/test_echo_forward_evaluator.py` (synthetic ledger, monkeypatched, DB never opened;
  ≤2 test files/call, `--basetemp` scratchpad, `-p no:cacheprovider`).

---

---

## V2 AMENDMENT (2026-07-20) — MEASURED-COST FORWARD + SOURCE OUTAGE QUARANTINE (operator-confirmed scope)

Operator decision: **"gerçek forward olsun hepsi" → measured-cost PAPER** (no real orders; costs measured,
not assumed). This closes the two holes found in the current ledgers: (a) fill = `mark_price` + hardcoded
`FEE_BPS=5` (unmeasured slippage), (b) outage quarantine exists only in the DASHBOARD display (`hold>7h`),
NOT at the source ledger — `_mark_at` returns arbitrarily-stale marks during a feed gap (the §166 fake-+900
mechanism can still write contaminated rows).

**Grounding (verified, not assumed):** main-DB table `book_ticker` exists with `bid_price, ask_price,
bid_qty, ask_qty, mid_price, spread_pct, book_imbalance, bid_depth_usd`; ETHUSDT live (latest 2026-07-20
15:00 UTC). Measured-cost fill is feasible.

### V2-A. Measured-cost fill (LONG; replaces mark→mark)
- **Entry = `ask_price`** at signal ts (buy lifts the offer). **Exit = `bid_price`** at target ts (sell hits
  the bid). → the bid/ask **spread is actually paid and recorded**, not assumed.
- `net_bps = (bid_exit − ask_entry)/ask_entry × 1e4 − COMMISSION_BPS`.
- **`COMMISSION_BPS = 5.0`** (operator-confirmed: standard round-trip taker; SEPARATE from spread, which is
  now measured — no double count, since the old 5 was a pure fee assumption that never included spread).
- Record per row: `ask_entry, bid_exit, spread_bps_entry, spread_bps_exit, bid_depth_usd_entry`.
- **Depth/impact honesty flag:** compare ORDER_NOTIONAL (read-only, from live config; NEVER modified) to
  `bid_depth_usd`/top-of-book qty. If notional > top-of-book → tag `TOP_OF_BOOK_INSUFFICIENT` (fill is an
  optimistic floor; real walk-the-book impact not modeled). This surfaces impact rather than hiding it.
- Stop variants (−150/−300) recomputed on the measured path (min bid over hold), same crossing convention.

### V2-B. Source outage quarantine (in the LEDGER, per-horizon)
A RESOLVE is `QUARANTINED_FEEDGAP` (excluded from every stat, never summed) if ANY of:
1. Nearest `book_ticker` to the **entry** ts is > **60s** away (stale entry quote), OR
2. Nearest `book_ticker` to the **exit** (target) ts is > **60s** away (stale exit quote — the §166 class), OR
3. `book_ticker` has a gap > **5 min** anywhere inside the OPEN→RESOLVE window (path/stop untrustworthy), OR
4. Wall-clock hold deviates from target horizon by > **5%** (`|actual−target|/target > 0.05`).
Quarantined rows are LOGGED (count + reason) so the operator sees the contamination rate, not a silent drop.
If quarantine fraction > **30%** sustained → forward infra is untrustworthy → halt verdicts, flag operator (§8).

### V2-C. Scope of code change (guardrail check)
- Modifies `tools/research_s34_hold_horizon_forward_ledger.py` (and mirrors into echo ledger if needed):
  add book_ticker read, spread-crossing fill, quarantine tags, extra recorded fields. **Forward paper only.**
- **NOT touched:** `execution/`, `risk/`, `brain/`, `.env`, `s34_state_machine_live_executor.py`, leverage,
  ORDER_NOTIONAL, sizing. DB stays `mode=ro`. No orders. Read-only + additive logging.
- Backward note: rows written before V2 keep `mark`-based net; V2 rows carry `cost_model:"measured_v2"` so
  the evaluator never mixes the two cost regimes.

---

## OPERATOR SIGN-OFF GATE (freeze ALL of these before implementation)
**Evaluator (V1):**
- **N_MIN = 20** noov obs before any confirm/refute verdict (≈ ~2 months at 6h ~11 noov/mo).
- **CONFIRM bar = CI-LB>0 AND avg ≥ 0.40 × in-sample-noov-avg** (P1 point-bar ≈ +7.7 bps).
- **Tail-gate CONFIRM ≤5% / REFUTE ≥10%** on hour≥17 forward.
- **Primary cells = {6h nostop; 6h × hour17 gate}; rest descriptive.**

**Measured-cost ledger (V2, operator-confirmed):**
- **Fill = entry@ask / exit@bid** (spread measured & paid), **COMMISSION_BPS = 5.0** on top.
- **Outage quarantine at source:** entry/exit quote > 60s stale, in-window gap > 5min, OR hold deviation > 5%.
- **Depth flag** when ORDER_NOTIONAL exceeds top-of-book.

On `DESIGN_ACCEPTED`, phase 2 = implementation (V2 ledger FIRST, then V1 evaluator — evaluator needs measured
rows). Each is a separate pass through: implement → independent review → correction → re-review → acceptance,
per CLAUDE.md gated chain. No two phases merged.
