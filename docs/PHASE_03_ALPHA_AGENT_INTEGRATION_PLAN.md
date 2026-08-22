# Phase 03 — integrating eclipse_scalper as the Eclipse Alpha Agent

**Status:** PLAN ONLY. No Scalper code has been changed. No platform code has been changed.
**Prerequisites met:** §290 website PASS (`348dc4ad`), §291 platform PASS (`92aeab11`), both
independent gates closed (§295).
**Out of scope for this phase, by operator instruction:** live execution, NATS, persistence,
FastAPI, and every agent other than Alpha.

This document exists to be argued with before anything is built. If a section is wrong, it is
cheaper to find out here.

---

## 1. What is actually being integrated

`D:\eclipse_scalper` is not one program. `start_eclipse.ps1` starts **27 named roles**, and only a
minority of them are "alpha" in the Eclipse sense. Grouping them by the agent they will eventually
belong to is the first honest step, because it shows that "make the Scalper the Alpha Agent" is
really "carve one agent out of a system that already contains four."

| Future agent | Existing roles |
|---|---|
| **Data Guardian** (not this phase) | `bookticker_collector`, `oi_spot_poller`, `collector_supervisor`, `heartbeat_watchdog`, `feed_gap_logger`, `liq_anomaly_monitor` |
| **Alpha Engine** (this phase) | `s34_state_machine_shadow_runner`, `echo_forward_ledger`, `echo_multilane_forward`, `hold_horizon_forward_ledger`, `liq_tip_forward`, `s34_bucket_live_harness`, `s34_v_engine_v02_shadow_mirror`, `e_der_v1_forward_runner`, `e_der_forward_core_collector`, `e_der_candidate_rich_collector`, `e_der_forward_universe`, `e_der_network_v2_observer` |
| **Execution Gateway** (blocked) | `s34_state_machine_live_executor` — **guarded file, operator sign-off required, default OFF** |
| **Observability** (not this phase) | `metric_snapshot_logger`, `s34_canonical_dashboard`, `s34_leads_monitor_dashboard`, `s34_live_chart`, `orderflow_chart`, `s34_replay` |
| Retired / other | `s34_shadow_paper_runner`, `liquidation_silence_scheduler` |

**Nothing in this table moves in Phase 03.** All 27 roles keep running exactly as they do today.

---

## 2. The one thing that could go badly wrong

> **The seal and the bus are in direct tension, and the bus is the more dangerous of the two.**

CLAUDE.md seals `reports/shadow/echo_forward_ledger.jsonl`, `hold_horizon_forward_ledger.jsonl`
and the derived surfaces (§239). On a **qualifying arm** no aggregate may be produced at all —
"sum/avg/WR/tail/max/min/MFE-capture … or any summary that IMPLIES a criterion". The precedent for
violating it is CT-011, and it was me.

An event bus is an aggregation machine. The moment a sealed arm's outcomes are published:

* Observability subscribes to `eclipse.>` by design and would aggregate them as telemetry.
* The Master Center journals **every** event it can read, and its grant is `eclipse.>`.
* Any future subscriber inherits the leak without anyone deciding to grant it.

The platform's own permission model does **not** save us here, because the Master Center is
*supposed* to read everything for the journal. So the protection has to be at the source.

**Design rule for Phase 03, non-negotiable:**

1. The Alpha Agent may publish **candidates** (a T0 intention). It may **never** publish an
   outcome — no `net_bps`, no resolved fill, no `CLOSE` record, no derived field from one.
2. `eclipse.alpha.trade_candidate` carries no outcome field, which is already true of the frozen
   schema. **Do not add one.**
3. The forward ledgers keep writing to disk exactly as today. The bus is an *additional*,
   outcome-free surface — never a replacement for, or a mirror of, the sealed files.
4. Sealed-arm evaluation stays where it is: files, evaluator, pre-declared sample sizes.

This is the single most important sentence in the document: **the ledger is the record; the bus is
a notification.**

---

## 3. Contract mapping

Scalper vocabulary → the frozen `eclipse-shared` contract.

| Scalper concept | Eclipse contract | Notes / risk |
|---|---|---|
| `anchor_ts_ms` (ms epoch) | `TradeCandidate.anchor_id` (str) | Stable, already the natural key. Rendered as a string. |
| arm name (`echo_30_90+regime`, `E-DER V1/V2/V3`, `hour17`) | `TradeCandidate.arm` | Must be the **frozen** name. Changing a condition creates a new arm — this is already the repo's rule and the platform's. |
| arm version | `TradeCandidate.arm_version` | The Scalper does not currently version arms explicitly. **Gap — see §6.** |
| `ETHUSDT` | `TradeCandidate.symbol` | Echo ledger is single-symbol today. |
| LONG | `TradeCandidate.direction` | |
| `HOLD_MS` (4 h) | `TradeCandidate.horizon_minutes` (240) | Internal only. Never published to the website (W1). |
| `gates_t0`, `qualified_t0`, `indicators` | `TradeCandidate.context` (`dict[str,str]`) | **T0 only.** `qualified_full` and `noisy_T30m` are T+30m lookahead and must **never** enter a candidate. |
| `entry_mark` | `context`, or omitted | A mark price, not a size. Safe, but adds nothing at T0 — recommend omitting. |
| `net_bps`, `CLOSE`, `path_min_bps`, `be_ratio_*` | **nothing — must not be published** | §2. |
| feed gap / outage window | `eclipse.data.quality_degraded` | Belongs to Data Guardian, a later phase. |
| collector liveness | registry heartbeat | Not an event. |

**Fields the platform requires that the Scalper does not currently produce:** `candidate_id`,
`arm_version`. Both are trivial to mint, but minting them is a *code change* and therefore not part
of this phase.

---

## 4. What the platform will enforce, and what that changes

These are consequences of the frozen invariants, not choices:

* **Trusted clock (P2).** `received_at` comes from the bus; a candidate whose `occurred_at` is in
  the future is refused. Fine for forward-only operation — and it means **historical backfill onto
  the bus is impossible by construction.** Phase 03 is forward-only. Good: that matches the repo's
  own forward-only discipline.
* **Schema boundary (P1).** A malformed or over-full payload is refused, not logged. A candidate
  carrying a stray `size` field would raise. This is the point.
* **Health, transitively (P10/P11).** The Alpha Agent will declare a dependency on the Data
  Guardian. When the feed is stale, **Alpha cannot publish at all.** That is a genuine behavioural
  difference from today, where the ledger keeps writing and contamination is filtered afterwards.
  It is arguably an improvement — feed outage is a known contamination source (§191) — but it is a
  **decision**, see §6.
* **Declaration binds (P8).** Alpha declares `eclipse.alpha.trade_candidate` and nothing else. A
  later operator grant cannot widen that without a re-registration.

---

## 5. Staged delivery

Each stage is separately reviewable and separately abandonable. **Only 3A is complete.**

| Stage | Deliverable | Touches Scalper code? |
|---|---|---|
| **3A** | This document: role audit, contract mapping, risk register | **No** |
| **3B** | `eclipse-shared` available to the Scalper as a dependency; a read-only adapter module that *constructs* a validated `TradeCandidate` from an existing anchor. No transport, no emission, no behaviour change | New file only |
| **3C** | The Alpha Agent's `AgentRegistration` descriptor — identity, declaration, health, dependencies — validated against the protocol. Still no transport | New file only |
| **3D** | Parity harness: for a replayed window of existing ledger rows, prove the adapter yields **exactly one candidate per anchor**, never one for a sealed outcome, and never a lookahead field | Tests only |
| **3E** | Transport. **Gated — requires NATS, which is out of scope, and its own review** | — |

**3B and 3C add files; they do not modify any existing Scalper file.** That is deliberate: it keeps
the diff reviewable and means Phase 03 cannot break a running role.

---

## 6. Decisions required before 3B

These change the work materially and are not mine to make.

1. **Which arm goes first?** Recommend **one** frozen arm, not all. `echo_30_90+regime` has the
   longest forward history; the E-DER arms are the operator's current focus. Starting with more
   than one multiplies the seal surface for no extra learning.
2. **Seal posture.** Recommended and assumed above: candidates are publishable, outcomes never
   are. If the answer is instead *"no sealed arm touches the bus at all"*, then 3B must select a
   non-sealed arm or the phase becomes contract-only.
3. **Feed-gap behaviour.** Fail-closed (no candidate while a dependency is stale, per P11) or
   emit-with-quality-flag? Recommend **fail-closed** — it matches the frozen invariant and the
   repo's outage discipline — but it is a behaviour change and deserves an explicit yes.
4. **Arm versioning.** The Scalper has no explicit `arm_version`. Recommend deriving it from the
   frozen prereg identity rather than inventing a new counter.
5. **Dependency mechanism.** Install `eclipse-shared` into the Scalper environment, or path-insert
   it? Recommend **install**, because a path-insert quietly couples two repositories.

---

## 7. Guardrails this phase operates under

Unchanged and binding:

* **Do not touch** `tools/s34_state_machine_live_executor.py`, `.env`, `execution/`, `risk/`,
  `brain/`. Reading is permitted; modification requires operator sign-off.
* Leverage, `ORDER_NOTIONAL` and sizing are immutable.
* No parallel Python/PowerShell processes; research scripts run sequentially.
* pytest: ≤2 files per call, `--basetemp` to scratchpad, `-p no:cacheprovider`.
* Frozen research arms are preserved exactly. Phase 03 changes no arm, no threshold, no gate.
* Process termination by **exact owned PID only** (`docs/OPERATOR_PROCESS_SAFETY.md`).
* Phase 01–02 platform invariants are frozen (§295). If integration appears to need one changed,
  that is a signal the integration is wrong, not the invariant.

---

## 8. Acceptance criteria for 3B, written before it is built

So the next independent review has something to check against:

1. No existing Scalper file is modified.
2. No role's behaviour changes; `start_eclipse.ps1` is untouched.
3. The adapter constructs a `TradeCandidate` that validates against the frozen schema, including
   rejecting a payload carrying `size`/`quantity`/`notional`/`leverage`/`venue`.
4. The adapter has **no** code path that reads `net_bps`, `qualified_full`, `noisy_T30m`,
   `exit_mark` or any CLOSE-record field. Enforced by a test, not by inspection.
5. No transport, no socket, no broker client, no persistence.
6. The Scalper does not import `eclipse_master_center`. Only `eclipse_shared`.
7. Tests pass under the repo's pytest constraints.
