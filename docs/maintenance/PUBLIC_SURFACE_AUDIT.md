# PUBLIC SURFACE AUDIT

**Date:** 2026-08-24 · **Scope:** read-only audit of everything the public GitHub
repository shows a visitor · **Method:** the repository's own files, not its narrative.

This audit was written before anything was changed. It exists because the README is
not evidence about the repository — it is a claim about it, and the two had drifted
apart. Every statement below was checked against a file, a `git ls-files` entry or a
workflow definition. Where a claim could not be checked it is marked `UNCERTAIN`
rather than repeated.

---

## 0. What was measured

| Quantity | Value | How |
|---|---|---|
| Commits on `main` | 496 | `git rev-list --count main` |
| Commits on the working branch | 525 | `git rev-list --count HEAD` |
| Tracked files | 19,040 | `git ls-files` |
| Tracked files excluding the nested `eclipse_scalper/localtests` snapshot | 4,865 | see finding S-1 |
| Tracked Python modules (excluding that snapshot) | 1,704 | `git ls-files '*.py'` |
| Tracked test modules under `tests/` | 595 | `git ls-files 'tests/*.py'` |
| Tracked docs (`docs/**/*.md`) | 137 | `git ls-files` |
| First / last commit on `main` | 2026-01-13 / 2026-07-14 | `git log` |
| CI workflows | 4 | `.github/workflows/` |

The working branch is `feature/eclipse-website`. Local `main` (`cdeb9009`) and
`origin/main` (`f9008c9b`) are **not** the same commit, and the working branch differs
from `origin/main` by **1,398 files**. That difference is the subject of finding P-1,
which is the most consequential item in this audit.

---

## 1. What the current README gets right

These are true and should survive the rewrite:

- **The safety hierarchy is real.** `risk/kill_switch.py`, `execution/circuit_breaker.py`,
  `execution/entry_loop.py`, `execution/order_router.py`, `execution/order_verifier.py`
  all exist and sit in the order the README draws them.
- **The reconcile / restart story is real.** `execution/bootstrap.py`,
  `execution/reconcile.py`, `execution/intent_ledger.py`,
  `execution/intent_ledger_persistence.py`, `execution/position_manager.py` exist.
- **The invariant document is real and is genuinely good.** `docs/INVARIANTS.md`
  states contracts with *how it breaks* / *how to detect* / *how it is enforced*,
  which is a stronger form than most projects publish.
- **Determinism and no-lookahead are stated as contracts, not aspirations** (DAT-01,
  DAT-03, VAL-01) and each names its test file.
- **The escalation ladder** (simulation → paper → micro capital → live) is the right
  public posture and matches the launcher's default-off live flag.
- **The disclaimer is honest** and stronger than most trading repositories carry.

## 2. What is stale

| # | Claim in README | Reality |
|---|---|---|
| T-1 | `Python 3.13` badge | Every CI job pins **3.11** or **3.12**. 3.13.9 is only the local interpreter. The badge states a version the project does not test on. |
| T-2 | `License MIT` badge | **There is no `LICENSE` file in the repository.** See P-4 — this is the single most serious factual defect on the public surface. |
| T-3 | `execution/entry_decision.py` in the architecture diagram | Does not exist. The nearest real modules are `execution/entry_gates.py`, `execution/entry_signals.py`, `execution/entry_sizing.py`. |
| T-4 | `python dashboard/backend.py` in Quick Start | Does not exist. The backend is a package: `dashboard/backend/app.py`. **The published start command is wrong.** |
| T-5 | `risk/` described as "Kill-switch, circuit breaker, risk manager" | `risk/` contains exactly one module, `kill_switch.py`. The circuit breaker lives in `execution/`, and there is no `risk/risk_manager.py` — the only `risk_manager` in the codebase is a *regime* risk manager resolved out of a runtime container in `execution/entry_loop.py` and `execution/exit.py`. `docs/INVARIANTS.md` §EXE-05 also names `risk/risk_manager.py`. |
| T-6 | `exchanges/` described as "Binance adapter + paper trading adapter" | `exchanges/` holds `base.py`, `binance.py`, `coinbase.py`, `mock.py`, `validator.py`. There is no `exchanges/paper_trading.py`; paper mode is a *profile and dry-run guard*, not an adapter module. `docs/INVARIANTS.md` §SAF-02 also names the non-existent file. |
| T-7 | `strategies/` described as "Signal logic (Eclipse Scalper strategy)" | `strategies/` now contains `alpha/`, `indicators/`, `regime/`, `signals/` and `risk.py` alongside `eclipse_scalper.py`. |
| T-8 | The project-structure tree | Omits `ami/` (143 tracked files), `src/` (114), `core/` (12), `runs/`, `runtime/`, `monitoring/`, `web/`, `scripts/`, `integrations/`. These are not minor: `ami/` is an entire research-governance subsystem. |
| T-9 | `docs/INVARIANTS.md` §5 lists EXE-01, EXE-02 and SAF-02 as **TODO** test gaps | All three test files now exist: `tests/test_order_router_idempotency.py`, `tests/test_order_router_intent_lifecycle.py`, `tests/test_paper_mode_no_live_orders.py`. The invariant doc understates its own coverage. |
| T-10 | "The repository contains the core engine and research tools." | It also contains 1,839 tracked report files, an AMI governance subsystem, a static website, an observatory service and a 19k-file tracked tree. The sentence describes a much smaller project. |
| T-11 | **Systemic, not just the README** — see the sweep below | Eight module paths cited across the *tracked* public documentation do not exist. |

### T-11 · module paths cited in public docs that do not exist

The README's three bad paths (T-3 / T-4 / T-5 / T-6) are not isolated typos. Sweeping
every tracked `.md` for module paths and checking each against the filesystem:

| Cited path | Exists? | Where it is cited | Nearest real thing |
|---|---|---|---|
| `risk/risk_manager.py` | no | `README.md`, `docs/INVARIANTS.md` (EXE-05), `docs/CLAUDE.md`, `docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` | sizing lives in `execution/entry_sizing.py` and `execution/regime_sizer.py`; the only `risk_manager` in code is a *regime* risk manager held in a runtime container |
| `exchanges/paper_trading.py` | no | `docs/INVARIANTS.md` (SAF-02), `docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` | paper mode is a profile plus a dry-run guard, not an adapter module; `exchanges/mock.py` is the non-live adapter |
| `execution/entry_decision.py` | no | `README.md`, `docs/CLAUDE.md` | `execution/entry_gates.py`, `execution/entry_signals.py`, `execution/entry_sizing.py` |
| `execution/entry_primitives.py` | no | `docs/CLAUDE.md`, `docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` | `symkey()` is defined in `execution/runtime_helpers.py:46` |
| `execution/adaptive_exit.py` | no | `docs/CLAUDE.md` | `execution/exit.py`, `execution/adaptive_guard.py` |
| `risk/cost_model.py` | no | `docs/CLAUDE.md` (twice, including a test cross-reference) | `core/fee_model.py`, `config/costs.py` |
| `risk/allocation.py` | no | `docs/CLAUDE.md` | no equivalent found |
| `dashboard/backend.py` | no | `README.md` | `dashboard/backend/app.py` |

This matters more than a broken link. `docs/CLAUDE.md` and `docs/INVARIANTS.md` are
*operating contracts* — they tell a reader (human or agent) where a safety check lives.
An invariant whose stated enforcement point does not exist cannot be verified by
following the document. **The correct fix is to the rule, not to the eight instances:**
a link/path check over tracked Markdown belongs in CI alongside the existing gates.

This audit does **not** edit `docs/INVARIANTS.md`, `docs/CLAUDE.md` or
`docs/ECLIPSE_SCALPER_CODEX_GUIDE_TR.md` — they are contract documents and are outside
the permitted change set (§9/§10). The finding is recorded for the owner. Only the
README's own four instances are fixed here, because the README is being rewritten from
zero regardless.

## 3. What is misleading

| # | Item | Why it misleads |
|---|---|---|
| M-1 | A **"Top Alpha Pocket"** section naming a symbol and a horizon, with a threshold triple and a four-column touch / fill / hit / adverse table | It reads as *the current edge of the system*. It is a development-era result from a research line that later work superseded. Presenting a development PASS as the project's headline is exactly the confusion the rest of the repository is built to prevent. |
| M-2 | **The regime GO / NO-GO table** ("SELL + UP regime … **GO**") | `GO` here meant *passed a development screen*. On a front page it reads as *validated and deployable*. |
| M-3 | **A single break-even maker-fee figure** | One break-even number implies a settled cost model. The repository's own contradiction register records that the round-trip fee constant is **inconsistent across active code paths**, and that results computed under different fee bases are not cross-comparable. |
| M-4 | **"forward-validated alpha pockets"** in the tagline | Forward validation is a *procedure the repository runs*, not a property any route has been shown to have. The tagline asserts the outcome. |
| M-5 | The architecture ASCII diagram terminating at `reports/*.md · reports/*.json` | Implies the research pipeline's output is a report. The actual terminal object is a *verdict under a frozen contract*, and the machinery that produces it (contracts, prereg, evaluators, replication) is absent from the diagram entirely. |
| M-6 | `Status: Research / Paper Trading` badge next to a GO table | The badge says "research"; the content says "here is the winning configuration". Readers resolve that contradiction in favour of whichever they wanted to believe. |

## 4. What no longer represents the project

The README describes a **scalper with a research pipeline attached**. The repository is
a **research system with an execution layer attached**. Concretely, none of the
following appear on the public front page at all:

1. **Preregistration and frozen contracts.** `docs/research/contracts/`,
   `S34_PREREGISTRATION.md`, and per-study prereg artifacts freeze the rule *before*
   the outcome is opened.
2. **A hypothesis ledger and a decision log** (`docs/research/HYPOTHESIS_LEDGER.md`,
   `docs/research/DECISION_LOG.md`) that keep scientific state and record why choices
   changed, with superseded statements marked rather than erased.
3. **A canonical research operating order** — `docs/research/README.md` states an
   explicit loading order and the rule that *no result silently redefines the Bible*.
4. **Data-feasibility auditing as a gate.** `docs/research/audits/data_feasibility_v1/`
   establishes what is *observable* before anything is measured.
5. **Observability and join-health discipline** — the repository treats an unobserved
   window as unobserved, not as zero, and requires joint coverage across feeds to be
   published before a multi-feed result is read.
6. **An epistemic governance subsystem** (`ami/`, 143 tracked modules) with a
   constitution, knowledge/failure stores, mutation suites and decision records.
7. **A graveyard that is enforced** — `FAILURE_ARCHIVE.md` plus a protocol rule that
   closed ideas are not re-tested. Publishing what you refuse to re-run is a stronger
   signal than publishing what passed.
8. **Chaos and reliability testing in CI** — three named chaos scenarios plus a
   reliability gate that is checked *both* for passing on a good fixture and for
   **failing on a degraded one**. A rule that never fires reads exactly like a rule
   that passes; the repository already tests for that and never said so publicly.
9. **A machine-checked publication policy** (`web/tools/check_policy.py`) that refuses
   performance figures, thresholds, horizon suffixes, ranking vocabulary and health
   claims — and is mutation-tested with 21 deliberate violations.
10. **A staged independent-review lifecycle** — implementation → independent review →
    correction → independent re-review → acceptance, each phase closing with its own
    verdict token.

## 5. Publication-sensitive material currently exposed

Detailed treatment is in [`PUBLICATION_RISK_REGISTER.md`](PUBLICATION_RISK_REGISTER.md).
Summary of what the audit found:

| # | Finding | Severity |
|---|---|---|
| P-1 | The working branch would newly publish **849 markdown/CSV/JSON files** that are not on `origin/main`, including `docs/protocols/S34_*_V0_1.md` — seven mini-protocols that state complete frozen rules: symbol, trigger threshold in USDT, depth band, prior-trend threshold, limit offset in bps, fill condition and exit horizon, with the formulas. | **BLOCKING** |
| P-2 | The current README itself publishes, counted mechanically: **four threshold comparisons**, **four feature formulas**, the passive fill model as source with its offset and depth-proxy expressions, a horizon in a heading and again in a CLI example, and **fourteen performance figures** — eight touch/fill/hit/adverse quantities, four regime screening quantities, and two cost figures. Under the repository's own published content policy (`web/README.md`) every one is a "never publish" item. | **HIGH** |
| P-3 | 816 tracked files under `reports/research/s34/` are already public and contain bps figures, win rates and prereg thresholds. Pre-existing, not introduced here. | **MEDIUM — owner decision** |
| P-4 | An `MIT` license badge with no `LICENSE` file. | **HIGH — legal, owner decision** |
| P-5 | `SYSTEM_STATE.md` is already public and would grow from 846 KB to 2.71 MB. | **MEDIUM — owner decision** |

## 6. Broken or stale links and commands

| Item | Status |
|---|---|
| `python dashboard/backend.py` | **Broken** — no such file (T-4). |
| `powershell … .\scripts\start_paper_trading.ps1` | Valid — file exists. |
| `python -m execution.bootstrap` | Module exists. Not executed during this audit. |
| `python -m tools.validate_passive_pocket_forward …` | Module exists. Flags not executed. |
| `python -m tools.rank_passive_pockets_forward --candidates-md reports/FILTER_SWEEP_V3_21D_ETH_h120_ADV1p2.md` | Both module and report file exist — but the example hard-codes a development-era sweep artifact and a horizon, which is itself a published threshold. |
| `cd dashboard/frontend && npm run dev` | Valid — `dev`, `build`, `typecheck`, `test` scripts all present. |
| `docs/INVARIANTS.md` link | Valid. |
| `docs/eclipse_scalper_thumbnail.svg` | Exists (5,350 bytes). Teal/amber terminal aesthetic, unrelated to the Eclipse design system now used by `web/`. |

## 7. Verified current architecture

Checked to exist on disk, at these paths:

```
execution/        60 tracked modules   bootstrap · guardian · entry_loop · entry_gates
                                       entry_signals · entry_sizing · order_router
                                       order_verifier · order_validation · intent_ledger
                                       intent_ledger_persistence · reconcile · replace_manager
                                       position_manager · position_lock · circuit_breaker
                                       protection_manager · emergency · flatten_intent
                                       belief_controller · belief_evidence · event_journal
                                       event_lane_gate · health_gate · health_monitor
                                       reliability_gate_runtime · telemetry · preflight
                                       passive_execution_simulator · state_machine · sim/ · live/
risk/              1 module            kill_switch.py
exchanges/         5 modules           base · binance · coinbase · mock · validator
brain/             3 modules           state · persistence · performance_memory
core/             12 modules           fee_model · micro_features · micro_signal · regime
                                       regime_risk · latency_profiler · order_placement
strategies/                            eclipse_scalper.py + alpha/ indicators/ regime/ signals/
tools/           701 tracked files     research CLIs, dashboards, monitors, gates
ami/             143 tracked files     constitution · governance · knowledge · research
                                       states · lifecycle · decision · warehouse · storage
src/             114 tracked files     src/eclipse · src/microphys
dashboard/        92 tracked files     backend/ (app · aggregator · canonical_state ·
                                       freshness · data_sources) + frontend/ (React/Vite/TS)
web/              22 tracked files     static site, no build, no external requests
tests/           595 tracked modules   incl. tests/legacy_tools/ (45 modules)
docs/            137 tracked docs      incl. docs/research/ governance tree
```

CI, verified from `.github/workflows/`:

| Workflow | File | Jobs |
|---|---|---|
| CI Tests | `ci-tests.yml` | Dashboard Frontend Gate · Dashboard Backend Smoke · Chaos Required (3-way matrix) · Execution Invariants and Gate · PR Reliability Comment · Chaos Full Suites (Nightly) |
| Ops Smoke | `ops-smoke.yml` | Offline bootstrap + ops tools smoke |
| Telemetry Dashboard Snapshot | `telemetry-dashboard.yml` | scheduled snapshot + notifier smoke |
| Telemetry Smoke Assertions | `telemetry-smoke.yml` | notifier smoke chain |

The three required chaos scenarios are named in the matrix:
`ack-after-fill-recovery`, `cancel-unknown-idempotent`, `replace-race-single-exposure`.
The `Execution Invariants and Gate` job runs ten unit suites and then runs
`tools/reliability_gate.py` **twice** — once expecting a pass on a clean fixture, once
expecting a **non-zero exit** on a fixture with missing journal coverage.

## 8. Verified public-safe capabilities

Safe to describe on the front page, at the level of *what exists*, with no figures:

- layered execution safety with a stated dominance order and a reduce-only exemption
- idempotent order routing keyed on a stable intent id
- an intent ledger with terminal-state completeness as a contract
- restart and reconcile convergence toward exchange truth
- deterministic, seeded research simulation
- no-lookahead as a tested contract, not a convention
- true forward splits as a tested contract
- cost-unit correctness as a tested contract
- chaos scenarios and a negatively-tested reliability gate in CI
- preregistration, frozen contracts, hypothesis ledger, decision log, failure archive
- data-feasibility auditing, coverage and join-health gates
- observability-before-measurement ordering
- a machine-checked, mutation-tested publication policy
- a staged independent-review lifecycle with verdict tokens
- read-only operator dashboards with no order/trade control path

## 9. Files this work may safely change

Public documentation and public assets only:

```
README.md                      rewritten from zero
docs/public/**                 new — audit, risk register, method, architecture,
                               reproducibility, historical context, metadata
                               recommendations, doc map, policy checker
docs/assets/**                 new — SVG visual system
CONTRIBUTING.md                new
SECURITY.md                    new
```

## 10. Files that must remain untouched

Everything else. Explicitly, and without exception:

```
SYSTEM_STATE.md                       the single master state file
CLAUDE.md · AGENTS.md                 operating contracts
docs/INVARIANTS.md                    a contract document; its staleness (T-9) is
                                      recorded here, not edited here
docs/research/**                      Bible, decision log, hypothesis ledger,
                                      contracts, audits, manifests
reports/**                            every research artifact and receipt
docs/protocols/**                     frozen mini-protocols
web/**                                the reviewed public site and its checker
execution/ risk/ brain/ strategies/   strategy, execution and risk logic
exchanges/ core/ tools/ ami/ src/
data/** runtime/** state/** runs/**   data, runtime state, locks, ledgers
.env*                                 secrets
.github/workflows/**                  CI definitions
tests/**                              the test suite
```

No collector was started, no database was queried, no backtest was run, no research
outcome was opened, no runtime process was touched, and nothing was committed or
pushed during this audit.

---

## Appendix A — repository hygiene observations (recorded, not acted on)

| # | Observation | Note |
|---|---|---|
| S-1 | `eclipse_scalper/localtests/**` — a **nested copy of the repository name** holding 14,175 tracked files of per-run `metrics.json`. It is 74% of the tracked tree and it is what a visitor sees when they browse the repository root. `.gitignore` lists `eclipse_scalper/localtests/`, so the ignore rule is present but the files were tracked before it. | Owner decision. Untracking is a history/size decision, not a documentation one. |
| S-2 | 30+ `.pytest_tmp_*` directories in the working tree, untracked but visible locally. | Cosmetic. |
| S-3 | `.env`, `.env.draft`, `.env.paper`, `.env.paper.dual`, `.env.s34_live.example` exist on disk; `.gitignore` covers `.env` and `.env.*`. `git ls-files` returns none of them. **No secret file is tracked.** | Verified clean. |
| S-4 | `docs/OPERATOR_PROCESS_SAFETY.md` is referenced by the operating contract but is **untracked** — it cannot be linked from a public README. | Owner decision. |
| S-5 | `.github/CODEOWNERS`, issue templates and a PR template already exist. | Left untouched. |
