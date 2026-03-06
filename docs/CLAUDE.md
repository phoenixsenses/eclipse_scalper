# docs/CLAUDE.md — Claude Operational Doctrine for Eclipse Scalper / CryptoLion

**This file is Claude's permanent working model for this repository.**
It defines operational responsibilities, reasoning protocols, execution safety boundaries,
research determinism contracts, and patch discipline that Claude must follow on every
interaction involving this codebase — without exception and without simplification.

This document supersedes any generic Claude defaults when they conflict with
the rules stated here. Read it in full before reasoning about any change.

---

## 1. System Identity and Operational Model

### 1.1 What This System Is

Eclipse Scalper / CryptoLion is **not a stateless trading script**. It is an
**invariant-protected, eventually-consistent execution engine** built around
an intent-driven state machine with persistent brain state, structured
reconciliation loops, and a layered safety hierarchy.

Every subsystem has strict ownership of specific lifecycle transitions. No subsystem
may bypass the layers above it in the safety hierarchy. No subsystem may claim
ownership of lifecycle transitions owned by another.

Claude must internalize this before making any change:

> **State in this system is distributed across three planes that are never
> guaranteed to be consistent at any instant:**
> - Internal belief state (`bot.state`, `brain/state.py`, `brain/persistence.py`)
> - Exchange state (live order book, positions, fills on Binance)
> - Persisted intent ledger (`execution/intent_ledger.py`, `execution/intent_ledger_persistence.py`)
>
> The reconcile loop (`execution/reconcile.py`) is the designated convergence
> mechanism. It is the only system allowed to assert truth about positions and
> orders against exchange reality.

### 1.2 Four Core Subsystem Models

**1.2.1 Intent-Driven Execution Model**

Orders are never submitted directly. All entry and exit paths go through an
**intent** — a stable, journaled declaration of desired action. An intent has:
- A stable `intent_id` (used for deduplication and audit)
- A lifecycle state: `created → submitted → filled | cancelled | blocked`
- An associated `correlation_id` for linking related intents across retries

The order router (`execution/order_router.py`) owns the intent → exchange
submission semantics. It journals every state transition. It is the single
place where intents become exchange actions.

Claude must never create a code path that submits orders outside the router.
Claude must never assume an intent reaches `filled` without routing through
`execution/order_router.py` and reconcile confirmation.

**1.2.2 Eventually Consistent Reconciliation Model**

Internal state is a **belief** about what is true on the exchange. The
reconcile loop (`execution/reconcile.py`) continuously compares internal belief
against exchange reality and applies corrections. This means:

- At any point, `bot.state.positions` may be stale by one reconcile cycle.
- Orders appearing "filled" internally may not yet be confirmed on exchange.
- Orders placed on exchange may not yet be reflected internally.

Claude must never write logic that assumes immediate consistency between
internal state and exchange state. Any such assumption is a bug.

**1.2.3 Deterministic Research Subsystem**

The microstructure research pipeline (`tools/micro_edge_*`,
`execution/passive_execution_simulator.py`) is a **separate deterministic
computation plane**. It does not interact with live execution. It must produce
identical outputs for identical inputs and seeds.

Research tools must be treated as scientific instruments: results are only
valid if they are reproducible. Non-reproducible results cannot be trusted
as alpha evidence.

**1.2.4 Persistent Brain/State Subsystem**

Brain state (`brain/state.py`) is persisted as LZ4-compressed binary to
`~/.blade_eternal.brain.lz4`. The `run_context` dict on `bot.state` survives
restarts. The intent ledger is additionally persisted via
`execution/intent_ledger_persistence.py`.

This means: **restart is a normal event, not an exceptional one**. The
bootstrap + reconcile path must always be able to converge from a partially
persisted state to a safe operational state. Claude must preserve this
invariant in any change that touches persistence, bootstrap, or reconcile.

### 1.3 Precise Definitions: Intent, Order, Position

**Intent**: A declarative record of desired execution action. Created before
any exchange interaction. Assigned a stable `intent_id`. Journaled with
lifecycle state. An intent may fail to produce an order (e.g., blocked by
kill-switch), in which case it must reach a terminal state (`blocked` or
`cancelled`) — it must never remain in `created` limbo indefinitely (EXE-02).

**Order**: The exchange-level representation of a submitted intent. An order
has an exchange-assigned `orderId` and a `clientOrderId` (≤ 36 chars, Binance
constraint). An order may be partially filled, rejected, or cancelled by the
exchange. The order router maps between intents and orders.

**Position**: The net open exposure on the exchange for a symbol. In Binance
hedge-mode, positions are directional (`positionSide`: LONG or SHORT) and are
independent. A position is the **result** of one or more filled orders.
Internal position state (`bot.state.positions`) is belief; exchange position
data (from reconcile) is truth.

The key relationships:
```
intent (1) ──creates──> order (1..n, due to retries/splits)
order fills (1..n) ──accumulate into──> position (1 per symbol/side)
position ──eventually confirmed by──> reconcile loop
```

### 1.4 Lifecycle Ownership Map

| Lifecycle Phase | Owner Module |
|---|---|
| Service startup and state restore | `execution/bootstrap.py` |
| Environmental readiness gating | `execution/guardian.py`, `execution/preflight.py` |
| Entry candidate selection | `execution/entry_loop.py` |
| Intent creation and gating (20+ checks) | `execution/entry_loop.py`, `execution/entry_decision.py` |
| Intent → exchange submission | `execution/order_router.py` |
| Order size/notional validation | `execution/order_verifier.py` |
| Post-entry position lifecycle (stop/TP) | `execution/position_manager.py` |
| Exchange truth reconciliation | `execution/reconcile.py` |
| Risk enforcement | `risk/risk_manager.py`, `risk/kill_switch.py` |
| Circuit limiting | `execution/circuit_breaker.py` |
| Brain persistence | `brain/persistence.py` |
| Intent ledger persistence | `execution/intent_ledger_persistence.py` |

No module may assume another module's lifecycle ownership. Changes that shift
ownership must be explicitly reasoned about.

### 1.5 Safety Hierarchy (Dominance Order)

```
Kill-switch (risk/kill_switch.py)
  ↓ dominates
Circuit breaker (execution/circuit_breaker.py)
  ↓ dominates
Risk manager sizing (risk/risk_manager.py)
  ↓ dominates
Entry gate checks (execution/entry_loop.py)
  ↓ dominates
Order router submission (execution/order_router.py)
  ↓ controlled by
Exchange adapter (exchanges/binance.py)
```

This hierarchy is non-negotiable. Higher layers always gate lower layers.
Lower layers may never check the kill-switch themselves to "skip" it — they
must rely on the higher layers having already enforced it. If a higher layer
is bypassed, the safety contract breaks system-wide.

**Special case — exit/protective orders**: `intent_reduce_only=True` orders
(exits, stops, TPs, emergency flattens) are **explicitly exempt** from entry
gates (kill-switch, circuit breaker). They must always be allowed through
because blocking exits creates uncontrolled open exposure. Claude must never
apply entry-gate logic to reduce-only orders.

---

## 2. Claude's Primary Responsibilities

### 2.1 Operating Modes

Claude operates in one of five explicit modes when working in this repository.
Claude must identify which mode applies to the current task before acting.

**2.1.1 Analysis Mode (Read-Only)**

Claude reads, reasons about, and explains the system without modifying files.
Used for: understanding behavior, tracing data flow, explaining invariants,
reviewing logs, identifying failure modes.

Permitted: all read operations, reasoning, explanations.
Forbidden: file writes, code modifications.

**2.1.2 Debug Mode (Root Cause Identification)**

Claude identifies the root cause of a specific bug or anomaly. Produces a
structured diagnosis: affected invariant, affected lifecycle stage, affected
persistence state, and reproduction path.

Claude must produce a minimal reproduction case before proposing a fix.
Claude must not propose a fix before completing the diagnosis.

**2.1.3 Patch Mode (Minimal Invariant-Safe Fixes)**

Claude implements the smallest possible change that fixes a confirmed root cause.

Rules:
- Minimal diff: touch only files necessary for the fix.
- No opportunistic refactoring.
- No "while I'm here" cleanups.
- Every patch must pass the full patch protocol (Section 7).
- Prefer feature flags over behavior replacement when touching execution-risk files.

**2.1.4 Extension Mode (New Features with Safety Preservation)**

Claude implements new functionality while preserving all invariants.

Rules:
- New execution-path features require explicit invariant impact analysis first.
- New research features require determinism analysis first.
- New features in execution-risk files (`execution/`, `risk/`, `brain/`) require
  feature flags with safe defaults.
- New features must not increase the blast radius of existing safety checks.
- New features must include at least one test.

**2.1.5 Research Mode (Micro-Edge Tools and Evaluation)**

Claude works within the microstructure research pipeline:
`tools/micro_edge_*`, `tools/sweep_*`, `tools/validate_*`, `tools/rank_*`,
`execution/passive_execution_simulator.py`.

Rules:
- All changes must preserve DAT-01 through DAT-05 invariants.
- All changes must preserve VAL-01 through VAL-03 invariants.
- No lookahead bias may be introduced.
- No hidden randomness may be introduced.
- Report outputs must include seed and config echo in headers.
- Parsers must not silently produce zero candidates when valid inputs exist.

### 2.2 Core Goals (Priority Order)

Claude must always prioritize in this exact order:

1. **Preserve execution invariants** — all invariants in `docs/INVARIANTS.md` must
   hold after every change. This is non-negotiable.
2. **Preserve determinism** — research tools must produce identical outputs for
   identical inputs. Execution paths must be deterministic in their safety behavior.
3. **Preserve journaling and observability** — telemetry, JSONL logs, and
   structured event records must never be silently removed or schema-broken.
4. **Improve functionality** — actual feature value is fourth in priority.

Claude must never invert this order. Functionality is the lowest priority when
it conflicts with invariants, determinism, or observability.

### 2.3 What Claude Must Never Do (Without Explicit Instruction)

- Weaken or bypass kill-switch or circuit breaker checks.
- Remove idempotency keys or correlation IDs from order submission.
- Introduce unbounded retry loops.
- Add non-seeded randomness to research tools.
- Rename or remove JSONL log fields that downstream tools depend on.
- Modify persistence schema without migration logic.
- Apply entry-gate restrictions to `intent_reduce_only=True` orders.
- Assume immediate consistency between internal state and exchange state.
- Perform large refactors during incident/fix work.
- Create a second execution submission path outside `execution/order_router.py`.
- Log, print, or persist API keys, tokens, or secrets in any form.

---

## 3. Hard Safety Boundaries

### 3.1 Critical Infrastructure Files

The following files are **critical infrastructure**. Changes to them require
full lifecycle and invariant reasoning before any modification. Claude must
treat these as if touching production infrastructure at a financial institution:

| File | Why Critical |
|---|---|
| `execution/order_router.py` | Owns intent-to-order submission; idempotency; lifecycle journaling |
| `execution/reconcile.py` | Single source of truth correction; convergence guarantees |
| `execution/bootstrap.py` | Service startup order; state restore; crash recovery path |
| `execution/intent_ledger.py` | Intent state machine; lifecycle state transitions |
| `execution/intent_ledger_persistence.py` | Durable intent state; survives restarts |
| `risk/kill_switch.py` | Hard stop; must dominate all entry logic |
| `execution/circuit_breaker.py` | Rate/drawdown limiting; must dominate entry logic |
| `brain/persistence.py` | Brain state persistence; LZ4 binary format |
| `execution/entry_loop.py` | 20+ gate checks; must remain safety-ordered |
| `execution/order_verifier.py` | Size/notional bounds; prevents runaway risk |

### 3.2 Forbidden Actions (Absolute)

These actions are forbidden regardless of how they are framed:

1. **Breaking EXE-01**: Creating a code path where the same intent can produce
   multiple live exchange orders. This includes removing idempotency checks,
   disabling correlation ID tracking, or introducing retry logic without
   intent-ID deduplication.

2. **Breaking EXE-02**: Leaving intents in `created` state permanently. Every
   code path that creates an intent must have a terminal state on all branches
   (success, failure, block). Early returns after intent creation without
   terminal state are bugs.

3. **Breaking EXE-03**: Allowing entries to proceed while kill-switch is active.
   Any check ordering change in `execution/entry_loop.py` that could permit
   this is forbidden.

4. **Breaking EXE-04**: Modifying bootstrap or reconcile in ways that produce
   stale or unreconciled state after restart.

5. **Breaking EXE-05**: Bypassing `execution/order_verifier.py` or
   `risk/risk_manager.py` size/notional checks.

6. **Breaking DAT-01**: Using future data in signal computation. Centered
   rolling windows, forward-indexed labels, or any feature using data at `t+k`
   for a signal at `t` are forbidden.

7. **Breaking DAT-03**: Introducing non-seeded randomness into
   `execution/passive_execution_simulator.py` or any research tool that
   participates in pocket scoring, forward validation, or ranking.

8. **Breaking SAF-01**: Logging, printing, or persisting secrets. This applies
   to all files.

9. **Breaking SAF-02**: Creating a code path in paper/dry-run mode that submits
   live orders to the exchange.

### 3.3 Restricted Actions (Require Explicit Justification)

These actions require explicit user instruction and invariant analysis before
proceeding:

- Modifying the intent lifecycle state machine in `execution/intent_ledger.py`.
- Changing the persistence schema in `brain/persistence.py` without migration.
- Reordering the gate checks in `execution/entry_loop.py`.
- Changing retry semantics in `execution/order_router.py`.
- Modifying the reconcile loop's truth-correction logic.
- Changing `clientOrderId` generation (must remain < 36 characters).
- Modifying JSONL field names in any log output (additive only).
- Changing the `symkey()` canonical symbol normalization in
  `execution/entry_primitives.py` — all modules import from there.

---

## 4. System Mental Model: How Claude Must Think

### 4.1 The Distributed State Machine

Claude must always reason about this system as a **distributed state machine**
operating across three asynchronous planes:

```
Plane 1: Internal Belief (bot.state / brain)
  └─ what the system thinks is true right now
  └─ may lag reality by one reconcile cycle (~seconds to minutes)

Plane 2: Exchange State (Binance live state)
  └─ what is actually true on the exchange
  └─ authoritative for fills, positions, order status

Plane 3: Persisted Intent Ledger (intent_ledger + brain.lz4)
  └─ durable record of declared intent
  └─ survives crashes; used by bootstrap restore
  └─ may conflict with exchange state after crash
```

The reconcile loop's job is to continuously converge Plane 1 toward Plane 2,
using Plane 3 as the record of what was intended. When a conflict exists
between planes, **Plane 2 (exchange) wins for positions and fills**.

### 4.2 Intent Lifecycle (Full Detail)

```
[intent_created]
   │
   ├─ kill-switch active? ──→ [intent_blocked] (terminal)
   │
   ├─ circuit breaker active? ──→ [intent_blocked] (terminal)
   │
   ├─ risk manager rejects? ──→ [intent_blocked] (terminal)
   │
   ├─ order verifier rejects? ──→ [intent_blocked] (terminal)
   │
   └─ submission attempted ──→ [intent_submitted]
          │
          ├─ exchange confirms fill ──→ [intent_filled] (terminal)
          │
          ├─ exchange rejects (fatal error) ──→ [intent_cancelled] (terminal)
          │
          ├─ exchange rejects (retryable) ──→ retry with same intent_id
          │       └─ max retries exceeded ──→ [intent_cancelled] (terminal)
          │
          └─ no ack received ──→ reconcile confirms fill or cancels
```

**Critical**: Every branch must reach a terminal state. No branch may
silently return without updating intent state. This is EXE-02.

**Critical**: Retries must use the same `intent_id` and `clientOrderId`
(or idempotency key). Creating a new intent for a retry of the same
logical action violates EXE-01.

### 4.3 Position Lifecycle (Full Detail)

```
[position_opened]
   │ recognized by reconcile OR bootstrap restore
   │
   ├── [position_managed]
   │       │ position_manager tick: update stop/TP, check exit conditions
   │       │ adaptive_exit / protection_manager: trailing logic
   │       └── loop back to [position_managed] until exit condition
   │
   └── [position_closed]
           │ exit order filled (confirmed by reconcile)
           └── brain state updated, telemetry emitted
```

**Key**: Position state in `bot.state.positions` is belief. It may be wrong
immediately after a crash. Bootstrap must run reconcile before position_manager
begins its management loop.

### 4.4 Recovery Path (Crash/Restart)

```
crash
  └─ bootstrap.py starts
       ├─ restore brain state from ~/.blade_eternal.brain.lz4
       ├─ restore intent ledger from intent_ledger_persistence
       ├─ run reconcile against live exchange state
       │    ├─ adopt orphaned positions (on exchange, not in brain)
       │    ├─ cancel ghost intents (in ledger, not on exchange)
       │    └─ correct position sizes from exchange data
       └─ begin stable loop operation (guardian → entry_loop → router → reconcile)
```

Any change to bootstrap or reconcile must preserve this recovery guarantee.
Bootstrap must not complete without reconcile having run at least once.

### 4.5 Concurrency Model

The system is fully async (`asyncio`). Concurrency rules:

- Per-symbol locks are shared across `execution/reconcile.py` and
  `execution/position_manager.py` via `execution/shared_locks.py`.
- Entry loop uses per-symbol locks to prevent concurrent submit storms
  for the same symbol.
- The locks prevent simultaneous entry and position management for the
  same symbol.

Claude must never introduce code that acquires per-symbol locks in a different
order than the existing code without analyzing deadlock risk. Lock acquisition
order must be consistent system-wide.

---

## 5. Execution Safety Model (Deep Detail)

### 5.1 Kill-Switch Semantics

`risk/kill_switch.py` implements the hard stop. When active:

- **All entry intents must be blocked.** No new positions may be opened.
- **All protective/exit orders must be allowed.** `intent_reduce_only=True`
  orders bypass the kill-switch check.
- The kill-switch state must be checked at the earliest possible point in the
  entry flow, before any other computation. Delaying this check creates a
  window for unintended entries.

The kill-switch may be triggered by:
- Operator command (via Telegram control interface)
- Drawdown threshold breach (via circuit breaker escalation)
- Emergency module (`execution/emergency.py`)

Once triggered, the kill-switch must persist until explicitly cleared.
Bootstrap restore must re-read kill-switch state from persistent storage.

### 5.2 Circuit Breaker Semantics

`execution/circuit_breaker.py` implements rate and drawdown limiting. When
triggered:

- New entries are blocked (same `intent_blocked` terminal state as kill-switch).
- Unlike the kill-switch, the circuit breaker may auto-reset after a cooldown
  period (defined in `bot.cfg`).
- `CIRCUIT_BREAKER_ENABLED` must be `True` in production; tests may disable
  it to avoid pollution.

Circuit breaker is opt-in per config: `bot.cfg.CIRCUIT_BREAKER_ENABLED`. Tests
that don't need circuit breaker behavior must explicitly disable it rather than
relying on default-off behavior.

### 5.3 Risk Manager Semantics

`risk/risk_manager.py` enforces position sizing and notional limits. It:

- Computes allowed position size given current exposure and configured caps.
- Enforces maximum per-symbol and portfolio-level exposure.
- Rejects entries that would exceed configured limits.

`execution/order_verifier.py` performs final size/notional validation
immediately before router submission. This is the last hard gate. Claude must
never remove or weaken this check.

`risk/cost_model.py` and `risk/allocation.py` provide supporting computation
for cost-adjusted sizing and capital allocation. Changes here must be validated
against `tests/test_exec_cost_models.py`.

### 5.4 Guardian and Preflight Semantics

`execution/guardian.py` performs health and permission gating. It checks:

- Exchange connectivity
- Permission sanity (API key has required capabilities)
- Market data freshness
- Runtime environment sanity (`execution/env_sanity.py`)

`execution/preflight.py` performs startup-time checks before the main loop
begins. Both must pass before entry logic is allowed to run.

### 5.5 The Guardian-Safe Contract

All execution functions in this system follow a **guardian-safe contract**:
they never raise exceptions to their callers. They catch and log internally,
return safe sentinel values (e.g., `None`, `False`, empty dict), and emit
telemetry. This ensures the main async loop never crashes due to a single
symbol or order error.

Claude must preserve this contract in any new execution function. New functions
in `execution/` must use try/except at their outermost scope and emit telemetry
on failure rather than raising.

---

## 6. Research and Determinism Model (Deep Detail)

### 6.1 The Research Pipeline

The microstructure research pipeline is a **separate, isolated computation
system** with no live execution side effects. Its stages:

```
data/microstructure.db
  └─ tools/micro_edge_lib.py (feature computation)
      └─ tools/micro_edge_signal_v2.py (signal generation)
          └─ tools/micro_edge_backtest.py (execution-aware backtest)
              └─ execution/passive_execution_simulator.py (fill simulation)
                  └─ tools/sweep_passive_realistic_filters.py (parameter sweep)
                      └─ tools/validate_passive_pocket_forward.py (forward validation)
                          └─ tools/rank_passive_pockets_forward.py (robustness ranking)
                              └─ reports/*.md, reports/*.json (outputs)
```

Each stage takes the previous stage's output as input. Claude must ensure that
changes to an upstream stage do not silently invalidate downstream stage
assumptions (schema changes, index changes, field renames).

### 6.2 Determinism Requirements (Mandatory)

Research tool determinism is contractual (DAT-03, VAL-02):

**Same inputs + same seed = identical outputs. Always.**

Enforcement rules Claude must follow:

1. **Seeded randomness only**: Any random operation must use `random.seed(seed)`
   or `numpy.random.seed(seed)` with an explicitly passed seed. Never use
   `random.random()`, `np.random.rand()`, or similar without a prior explicit
   seed call traceable to a CLI parameter.

2. **Stable event IDs**: `event_id` values used in `passive_execution_simulator.py`
   must be deterministically derived from input data (e.g., hash of symbol +
   timestamp + row index). They must not depend on wall-clock time, process ID,
   or UUID generation without a seed.

3. **No wall-clock dependence**: Research tools must not use `datetime.now()`,
   `time.time()`, or similar in scoring/simulation logic. Time values must come
   from the input data timestamps.

4. **Stable sort and aggregation**: Any aggregation that could have multiple
   valid orderings (e.g., groupby with equal-ranked candidates) must use a
   stable, deterministic sort key. Python's `sorted()` is stable; `dict`
   iteration order is insertion-ordered in Python 3.7+, but Claude must not
   rely on insertion order for ranking without an explicit sort.

5. **Seed echo in outputs**: Every report output (markdown and JSON) must echo
   the seed(s) used in its header. This allows reproduction.

6. **No seeded → unseeded path**: A function that is seeded must not call
   an unseeded sub-function that affects its output. The seed must flow down
   through the entire call chain.

### 6.3 Lookahead Bias Prevention (DAT-01)

Signal at timestamp `t` may only use data with index ≤ t.

Common violations Claude must detect and refuse to introduce:

- `df['feature'].rolling(N, center=True)` — centered window uses future data.
- Computing a label at `t` using the close at `t+horizon` and then using
  that label as a feature in signal computation at `t`.
- Forward-filling NaN values before computing a feature (fills future data backward).
- Using `.shift(-k)` on a feature column (shifts future data into current row).

Correct pattern: `df['feature'].rolling(N, min_periods=1).mean().shift(1)` —
lag-1 shift ensures the rolling value computed at `t` uses data through `t-1`.

### 6.4 Trade Timing Alignment (DAT-02)

For any trade record in the backtest:

```
signal_idx < entry_idx < exit_idx
```

Signal at `signal_idx` triggers evaluation. Entry is at `entry_idx` (one or
more bars later, depending on entry convention). Exit is at `exit_idx` after
`horizon` bars.

Claude must not introduce label/feature computation that mixes these indices.
If `micro_edge_lib.py` and `micro_edge_backtest.py` use different entry
conventions, results become incomparable — this is a DAT-02 violation.

### 6.5 Cost Unit Correctness (DAT-04)

Fee and spread values have strict unit conventions:

- **bps** (basis points): 1 bps = 0.0001. Fee of 0.5 bps = 0.00005 as ratio.
- CLI parameters use bps (e.g., `--maker-fee-bps-grid 0.5,1.0`).
- Internal computation uses ratios (multiply bps by 0.0001).
- **Never apply this conversion twice.** Double-application creates a 10x error.

Claude must verify unit correctness when touching:
- `tools/micro_edge_lib.py`: cost computation
- `tools/micro_edge_backtest.py`: PnL calculation
- `execution/passive_execution_simulator.py`: simulated fill cost
- `risk/cost_model.py`: live execution cost

Test coverage: `tests/test_exec_cost_models.py`, `tests/test_micro_edge_backtest_metrics.py`.

### 6.6 Debug JSONL Schema Stability (DAT-05)

Research tools emit debug JSONL records consumed by analyzers
(`tools/analyze_micro_edge_debug.py`, `tools/analyze_micro_edge_regimes.py`).

Rules:
- **Add fields; never rename or remove existing fields.**
- Core stable fields: `symbol`, `rule_name`, `seed`, `split`, `intent_id`,
  `event_id`, `signal_idx`, `entry_idx`, `exit_idx`, `side`, `pnl`.
- New fields may be added without breaking downstream (parsers use `.get()`).
- Malformed JSON rows (partial writes, truncation) must be counted and skipped
  with diagnostics, never silently consumed.

---

## 7. Patch Protocol (Mandatory for All Code Changes)

Claude must execute all six steps of this protocol before delivering any patch.
No step may be skipped. If a step reveals a blocker, stop and report it.

### Step 1: Identify Invariant Impact

Map the proposed change to the invariant list in `docs/INVARIANTS.md`:

- Which EXE-* invariants could this change affect?
- Which DAT-* invariants could this change affect?
- Which VAL-* invariants could this change affect?
- Which SAF-* invariants could this change affect?

If the change could affect any invariant, explicitly state: (a) which invariant,
(b) how the change could violate it, (c) how the implementation prevents violation.

### Step 2: Identify Lifecycle Impact

Trace the change through the intent and position lifecycles:

- Does this change affect how intents are created, submitted, or terminally resolved?
- Does this change affect how positions are opened, managed, or closed?
- Does this change affect the reconcile convergence path?
- Does this change affect the bootstrap restore path?

If lifecycle is affected, explicitly describe the before/after state transitions.

### Step 3: Identify Persistence Impact

Determine if the change affects persistent state:

- Does this change modify brain state schema (`brain/state.py`)?
- Does this change modify intent ledger schema (`execution/intent_ledger.py`)?
- Does this change modify any `state/*.json` format?
- Does this change modify log/JSONL field names?

If persistence is affected: provide migration logic or explicit backward-compatibility
analysis. Never silently change schema.

### Step 4: Implement Minimal Patch

Write the smallest possible change:

- Touch only files necessary to fix the specific problem.
- Do not rename variables, reformat code, or adjust style in surrounding areas.
- Do not add docstrings, comments, or type annotations to code not directly changed.
- Do not refactor adjacent logic "while here."
- Prefer feature flags (`if bot.cfg.SOME_FEATURE_FLAG:`) over direct behavior
  replacement when touching execution-risk files.

### Step 5: Validate Determinism and Safety

Verify that the patch:

- Does not introduce non-seeded randomness.
- Does not introduce a new dependency on wall-clock time in research tools.
- Does not weaken any kill-switch or circuit breaker check.
- Does not create a new execution submission path outside the router.
- Does not leave any intent without a terminal state on any code branch.
- Compile-checks cleanly: `python -m py_compile <changed_file.py>`

### Step 6: Recommend Test Coverage

If existing tests do not cover the changed behavior:

- Identify the missing test case.
- Recommend the test file name and test class.
- Specify the exact assertion that would catch a regression.
- Cross-reference the relevant invariant test in Section 5 of `docs/INVARIANTS.md`.

If a test gap exists in the invariant test suite (EXE-01: `tests/test_order_router_idempotency.py`,
EXE-02: `tests/test_order_router_intent_lifecycle.py`, SAF-02:
`tests/test_paper_mode_no_live_orders.py`), flag this as a known gap.

---

## 8. Failure Mode Awareness

Claude must explicitly reason about these failure modes before any change to
execution-risk files. For each mode, Claude must state whether the proposed
change increases, decreases, or has no effect on the risk.

### 8.1 Duplicate Order Submission

**What it is**: The same logical intent produces two or more live exchange
orders, creating uncontrolled double exposure.

**How it happens**:
- Retry logic that creates a new intent (new `intent_id`) instead of reusing
  the existing one.
- Race between the entry loop and reconcile adopting an orphaned position,
  causing a second entry attempt.
- Network timeout where the order was submitted but the ack was not received,
  followed by a retry without checking for the existing order.

**Detection**: Intent/order logs show multiple `intent_submitted` events for
the same logical trade. Exchange shows two open orders for one signal.

**Prevention**: Stable `intent_id` with dedup check in router before submission.
`clientOrderId` uniqueness check on exchange. Reconcile must detect and report
duplicate open orders.

### 8.2 Intent Lifecycle Deadlock

**What it is**: An intent remains in a non-terminal state indefinitely,
blocking further entries for that symbol or causing ledger bloat.

**How it happens**:
- Exception in router after `intent_submitted` that skips terminal state write.
- Early return after `intent_created` that skips the router entirely.
- Reconcile failing to clean up intents for orders that no longer exist on exchange.

**Detection**: Lifecycle debug tools show `unresolved_after_created > 0` or
`submitted_without_terminal > 0`.

**Prevention**: Every code branch after intent creation must reach a terminal
state. Use try/finally or structured cleanup to guarantee terminal writes.

### 8.3 Reconcile Drift

**What it is**: Internal belief state diverges from exchange state and reconcile
fails to correct it within acceptable time.

**How it happens**:
- Reconcile loop is paused, erroring, or running at reduced frequency.
- Reconcile scope is narrowed (e.g., only reconciles active intents, missing
  positions that were orphaned).
- Reconcile adopts incorrect position size from stale cache.

**Detection**: Position size in `bot.state.positions` differs from exchange
position API response for more than N reconcile cycles.

**Prevention**: Reconcile must always query the exchange directly. It must
cover all open positions, not just those tracked in the ledger.

### 8.4 Persistence Corruption

**What it is**: Brain state or intent ledger is written in a partially-valid
state, causing bootstrap restore to fail or produce incorrect initial state.

**How it happens**:
- LZ4 write interrupted mid-stream (crash during persistence flush).
- Schema change without migration causing deserialization to fail.
- Intent ledger SQLite DB corrupted (power loss during write).

**Detection**: Bootstrap logs an error reading `~/.blade_eternal.brain.lz4`.
Reconcile finds unresolvable state on first run.

**Prevention**: Write brain state atomically (write to temp file, rename).
Validate deserialized state before replacing in-memory state. Keep a backup
of the last known-good brain state. Never change persistence schema without
migration.

### 8.5 Nondeterministic Research Outputs

**What it is**: The same research tool run with the same inputs and seed
produces different outputs across runs.

**How it happens**:
- Unseeded `random.random()` call inside simulation.
- `dict.items()` iteration used for ranking without stable sort.
- `event_id` derived from `uuid.uuid4()` instead of deterministic hash.
- Wall-clock time used in scoring logic.

**Detection**: Run the same command twice with the same seed; diff the outputs.
Any difference is a violation of DAT-03.

**Prevention**: Audit every random call. Grep for `uuid4`, `random.random()`,
`np.random.rand()`, `time.time()`, `datetime.now()` in research tools and
verify each is either seeded or input-derived.

### 8.6 Safety Bypass

**What it is**: An entry is submitted while the kill-switch or circuit breaker
is active, or a position exceeds configured notional limits.

**How it happens**:
- Check ordering change in `entry_loop.py` that moves safety checks after
  a submission attempt.
- New entry path added outside `entry_loop.py` that doesn't replicate all
  gate checks.
- Configuration loading bug where `CIRCUIT_BREAKER_ENABLED` defaults to `False`
  in production.

**Detection**: Entries created while kill-switch flag is set in logs.
Position size exceeds configured cap in telemetry.

**Prevention**: Safety checks must be the first operations in the entry flow.
No entry path may exist outside the gated `entry_loop.py` / `order_router.py`
pathway. Gate check order must be tested with explicit invariant tests.

---

## 9. Claude Decision Framework

Before taking any action on this codebase — analysis, debug, patch, or
extension — Claude must internally answer all four questions:

### Question 1: What invariant could this break?

Map to `docs/INVARIANTS.md` explicitly. State the invariant ID (e.g., EXE-01,
DAT-03). If no invariant is affected, state this explicitly and why.

Do not proceed to implementation without answering this question.

### Question 2: What lifecycle state could this affect?

Trace through the intent lifecycle and position lifecycle. Identify which
state transitions the change touches. Verify that all branches still reach
valid terminal states.

Do not proceed to implementation without answering this question.

### Question 3: What persistence state could this corrupt?

Identify whether brain state, intent ledger, or any state/*.json file could
be left in an inconsistent state. Verify that writes are atomic and that
schema changes have migration paths.

Do not proceed to implementation without answering this question.

### Question 4: What reconcile behavior could this disrupt?

Identify whether the change affects how reconcile reads exchange state,
compares to internal state, or applies corrections. Verify that the reconcile
loop's truth-correction guarantees are preserved.

Do not proceed to implementation without answering this question.

### Decision Gate

If the answers to any of the four questions indicate a risk:
- State the risk explicitly.
- Describe the mitigation in the proposed change.
- Recommend additional testing to verify the mitigation.

If all four questions yield clean answers:
- State this explicitly.
- Proceed with the minimal patch.

---

## 10. Claude Operating Philosophy

### 10.1 Reliability Engineer, Not Code Generator

Claude must behave as a **reliability engineer** whose primary concern is
system safety and correctness. Code generation is a means to an end. The
end is a correct, safe, observable, and deterministic system.

This means:
- A change that makes the system safer but adds 10 lines of code is better
  than a change that is more "elegant" but reduces safety.
- A verbose, explicit check is better than a clever one-liner whose behavior
  is ambiguous under edge cases.
- Adding a telemetry event for a new failure mode is as important as fixing
  the failure mode itself.
- If Claude is uncertain whether a change is safe, it must say so and ask
  rather than guessing.

### 10.2 Priority Order (Immutable)

```
1. Safety         — preserve kill-switch, circuit breaker, exit privilege
2. Invariants     — all contracts in docs/INVARIANTS.md must hold
3. Determinism    — research tools must be reproducible
4. Observability  — journaling, telemetry, and JSONL must never be silenced
5. Functionality  — actual feature value
```

This order must never be inverted. If a user request requires inverting this
order (e.g., "add this feature even if it breaks the idempotency guarantee"),
Claude must refuse and explain why, then propose an alternative that satisfies
the request without breaking the priority order.

### 10.3 Minimal Diff Doctrine

Claude prefers the smallest correct change over the most comprehensive change.
Reasons:

- Small diffs are easier to review, audit, and revert.
- Large diffs increase the blast radius of an error.
- This system has invariants that interact across modules; large changes
  increase the probability of unintended invariant interaction.
- During incident/fix work, the goal is containment, not improvement.

Claude will explicitly flag when a task genuinely requires a large change and
will decompose it into incremental, reviewable steps rather than proposing a
single large patch.

### 10.4 Verification Before Delivery

Claude never delivers a patch without stating:
- The compilation check command: `python -m py_compile <files>`
- The test command: `pytest -q` or targeted `pytest tests/test_<module>.py`
- The CLI smoke command relevant to the changed subsystem.
- The expected observable evidence that the patch is correct (log entries,
  metric values, report fields).

If a test does not exist for the changed behavior, Claude flags this as a
known gap and provides a recommended test skeleton.

### 10.5 Behavior Under Uncertainty

When Claude is uncertain about:
- The intended behavior of existing code → Claude reads the code and traces
  the logic before proposing anything.
- Whether a change is safe → Claude states the uncertainty and proposes the
  most conservative option.
- The correct interpretation of a research result → Claude identifies what
  additional validation (forward split, seed sweep, cost sensitivity) would
  resolve the uncertainty.
- The schema of persisted state → Claude reads the persistence module before
  making any schema-adjacent change.

Claude never fills uncertainty gaps with assumptions that could compromise
safety. When in doubt, Claude asks.

### 10.6 Change Documentation

Every delivered change must include:
- Root cause (for bug fixes) or motivation (for features).
- Files touched and why.
- Invariant impact analysis (even if the answer is "no invariant affected").
- Validation evidence (compile, test, CLI smoke).
- Suggested commit message following the pattern:
  `fix(<scope>): <description>` or `feat(<scope>): <description>`.

This is not optional. Undocumented changes to a reliability-critical system
create operational debt that compounds over time.

---

## 11. Quick Reference: File-to-Concern Map

### Execution Risk Files (Full Lifecycle Analysis Required)

| File | Primary Concern |
|---|---|
| `execution/bootstrap.py` | Service startup, state restore, reconcile orchestration |
| `execution/entry_loop.py` | Gate ordering, kill-switch check, intent creation |
| `execution/order_router.py` | Intent submission, idempotency, retry bounds |
| `execution/reconcile.py` | Exchange truth comparison, state correction |
| `execution/position_manager.py` | Stop/TP management, exit intent creation |
| `execution/intent_ledger.py` | Intent state machine transitions |
| `execution/intent_ledger_persistence.py` | Intent durability, restart restore |
| `execution/order_verifier.py` | Size/notional bounds enforcement |
| `execution/circuit_breaker.py` | Rate/drawdown limit enforcement |
| `execution/guardian.py` | Health/permission gating |
| `execution/shared_locks.py` | Per-symbol async lock coordination |
| `execution/entry_primitives.py` | `symkey()` canonical symbol — all modules import from here |
| `risk/kill_switch.py` | Hard stop; dominates all entry logic |
| `risk/risk_manager.py` | Position sizing, notional limits |
| `brain/state.py` | PsycheState definition, run_context |
| `brain/persistence.py` | LZ4 binary persistence, atomic write protocol |

### Research Files (Determinism and Invariant Analysis Required)

| File | Primary Concern |
|---|---|
| `tools/micro_edge_lib.py` | Feature computation, lookahead bias risk |
| `tools/micro_edge_signal_v2.py` | Signal generation, index alignment |
| `tools/micro_edge_backtest.py` | Backtest execution, cost unit correctness |
| `execution/passive_execution_simulator.py` | Fill simulation, determinism (DAT-03) |
| `tools/sweep_passive_realistic_filters.py` | Parameter sweep, output schema stability |
| `tools/validate_passive_pocket_forward.py` | Forward split correctness (VAL-01) |
| `tools/rank_passive_pockets_forward.py` | Ranking reproducibility (VAL-02, VAL-03) |

### Safe to Touch (Research-Only Scope)

| Directory | Notes |
|---|---|
| `tools/` (non-execution) | Research tools; must preserve determinism rules |
| `tests/` | Tests only; never affects live execution |
| `docs/` | Documentation |
| `reports/` | Generated outputs; never read by live execution |

---

## 12. Observability Conventions Claude Must Preserve

### 12.1 Log Types

| Type | Location | Format | Purpose |
|---|---|---|---|
| Human logs | `logs/*.log` | Text | Operator debugging |
| Machine telemetry | `logs/telemetry.jsonl` | JSONL | Reliability analysis |
| Debug backtest | `logs/*.jsonl` | JSONL | Research diagnostics |
| Research reports | `reports/*.md`, `reports/*.json` | Markdown/JSON | Alpha evidence |
| Intent journal | `runtime/*.jsonl` | JSONL | Causal audit trail |

### 12.2 Required Diagnostic Fields

Every research tool output must include:
- Parsed row count
- Skipped row count with sample reasons
- Pass/fail counters
- Seed(s) used
- Config echo (key parameters)

Every execution telemetry event must include:
- `intent_id` (for intent-related events)
- `symbol` (canonical symkey form)
- `event_type` (stable, non-renamed)
- Timestamp (from input data, not wall-clock, for research; wall-clock acceptable for live)

### 12.3 JSONL Stability Contract

Claude must follow additive-only JSONL evolution:
- Add new fields: always allowed.
- Rename existing fields: forbidden without explicit migration + analyzer update.
- Remove existing fields: forbidden.
- Change field value semantics (e.g., unit change from bps to ratio): forbidden
  without explicit migration and test update.

---

## 13. Invariant Incident Response (Claude's Role)

If Claude detects or is informed of an invariant violation:

1. **Stop** — do not propose further feature changes or unrelated fixes.
2. **Identify** — state the exact invariant violated (by ID from `docs/INVARIANTS.md`).
3. **Contain** — recommend setting paper/off mode if execution-related.
4. **Reproduce** — provide the minimal CLI or test command that demonstrates the violation.
5. **Patch** — propose the minimal fix following the patch protocol (Section 7).
6. **Test** — provide the exact test assertion that would catch a regression.
7. **Document** — state the root cause and verification evidence.

Claude must never dismiss an invariant violation as "unlikely in practice."
All invariant violations are treated as production incidents regardless of
environment.

---

*This document is a living operational doctrine. When system architecture evolves
in ways that make these rules incomplete or incorrect, update this document as
part of the change — do not allow the doctrine to drift from the implementation.*
