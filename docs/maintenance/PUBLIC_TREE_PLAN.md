# PUBLIC REPOSITORY RECONSTRUCTION

**Date:** 2026-08-24 · **Scope:** every tracked file in the repository, classified for
publication · **Status:** decided and reported. **Nothing has been moved, deleted or
untracked, and nothing has been committed or pushed.**

Round one rewrote the README. This round asks the question the README rewrite exposed:
*the front page was leaking — what else is?*

The answer is that the README was not the leak. It was the visible part of one.

---

## 0. Headline

| | Files |
|---|---:|
| Tracked today | 19,040 |
| **Stays public** | **1,300** |
| Leaves the public remote | 17,740 |
| — of which regenerable scratch and runtime | 15,555 |
| — of which internal material with real value | 2,185 |

Root directory: **68 tracked files today, 15 in the target tree.**
Eleven top-level directories disappear from the public remote entirely.

The reduction looks drastic and mostly is not: 15,555 of the 17,740 departures are
generated artifacts that should never have been tracked. The part that matters — the
research corpus, the governance subsystem, the frozen rules — is 2,185 files.

## 1. Method

`docs/maintenance/tools/classify_public_surface.py` reads `git ls-files` and assigns every
path exactly one disposition, with its reason recorded next to it. Ordered path rules
decide first; then the file's **contents** can override its location.

| Disposition | Meaning | Files |
|---|---|---:|
| `PUBLIC_KEEP` | already public-safe; stays as it is | 584 |
| `PUBLIC_REWRITE` | belongs in public, not in its current form | 6 |
| `CURATED_PUBLIC` | a named subset stays; the bulk does not | 8 |
| `DO_NOT_TOUCH` | stays public, and this work may not edit it | 702 |
| `INTERNAL_ONLY` | real value, must be preserved — privately | 2,185 |
| `REMOVE_FROM_PUBLIC` | generated or scratch; no archival value | 15,555 |

The per-file manifest is regenerated on demand rather than committed — `.gitignore`
excludes `*.csv`, and a half-megabyte list of paths is an execution aid, not a public
document:

```bash
python docs/maintenance/tools/classify_public_surface.py --csv manifest.csv
python docs/maintenance/tools/classify_public_surface.py --label INTERNAL_ONLY
python docs/maintenance/tools/classify_public_surface.py --explain <path>
```

**Location is a hypothesis; content is evidence.** Three refinements were needed before
the classifier was worth trusting, and each came from checking its output rather than
from reasoning about its design:

**(a) A name is not a value.** The first pass escalated 208 files. Reading them showed
almost all were schemas and field accessors — code that *reads* a `hit_rate` field
publishes nothing. Every pattern now requires a numeric literal bound to the thing it
names. That took 208 down to 78.

**(b) A number in a program may be a fixture.** Of those 78, a test asserting a round
`delta_vs_baseline: 0.10` is inventing a number, while a gate file stating a hit rate to
four decimal places is publishing a result. No pattern separates the two. So a number in
a **data artifact** (`.md`, `.json`, `.csv`) escalates the file, and a number in **source
code** is flagged for a human instead of being decided automatically. 114 source files
carry that flag — 72 tests, 37 tools, 3 data, 2 dashboard. **They are not classified;
they are queued.**

**(c) The scanner's own false negative.** A figure scan cleared 94 of 412 research
reports as carrying no figure, threshold or horizon. One of those 94 was opened and read.
It is a full calibration grid — win rates, medians, means across a parameter sweep — plus
an absolute path on the operator's machine and a database size in bytes. Every pattern
missed it because its numbers sit in **unlabelled table columns**; the column *header* was
the tell.

That finding changed a decision rather than just a pattern, and it is the most important
result in this document. It is treated separately in §4.

## 2. What leaves, and why

### 2a. Generated and scratch — 15,555 files

| Path | Files | Why |
|---|---:|---|
| `eclipse_scalper/**` | 14,175 | a nested directory sharing the repository's own name: a mis-rooted copy of per-run artifacts and fixtures. 74% of the tracked tree, and the first thing a visitor sees in the root |
| `localtests/**` | 385 | per-run local test artifacts |
| `runs/**` | 267 | sweep, eval and walk-forward run artifacts — regenerable *and* result-bearing |
| `runtime/**` | 223 | runtime state |
| `tmp/**` | 43 | scratch |
| `reports/test_*`, `reports/_runs`, `reports/plots`, `reports/phase24_demo` | ~110 | generated test-run and demo output |
| `state/locks/*` | 2 | a lock file and a pid file. These must never be tracked under any circumstances |
| `test.py`, `signal_check.py`, `tmp_sqlite_test.txt` | 3 | scratch at the repository root |

`.gitignore` already lists `eclipse_scalper/localtests/` and `localtests/`. The rule
exists; the files were tracked before it did, and a gitignore entry does not untrack
anything already in the index.

### 2b. Internal material with real value — 2,185 files

| Path | Files | Why |
|---|---:|---|
| `reports/**` | 1,839 | the research corpus and its governance closures — see §4 |
| `ami/**` | 143 | the epistemic governance subsystem: question registries, seeded rules, research state |
| `src/microphys/**` | 72 | the measurement library the open research lanes are built on |
| root `*.md` | 50 | internal roadmaps, registers, gap analyses, session reports, preregistrations, whitepapers |
| `docs/**` (minus the curated 8) | ~90 | handoffs, per-person status notes, lane specs, alert contracts |
| `docs/research/**` | 24 | the research bible, decision log, hypothesis ledger, contracts, audit evidence |
| `docs/protocols/**` | 7 | frozen mini-protocols — **operator decision, 2026-08-24** |
| derived artifacts | 19 | candidate specifications, calibration and selection manifests |
| gate and scoreboard state | 3 | frozen gate thresholds and a paper scoreboard |
| research tooling | ~380 | encodes rules, thresholds and cost constants **in code** |
| `pine/**` | 2 | a Pine strategy proxy is a literal trading rule |
| `research/**`, `deep_discovery/**` | 4 | lead-lag and stage research scripts with their reports |
| `CLAUDE.md`, `AGENTS.md`, `docs/CLAUDE.md`, `.agents/`, `.claude/` | 5 | agent operating contracts, carrying seal state, unseal stamps and alpha status |
| `SYSTEM_STATE.md` | 1 | **operator decision, 2026-08-24** — replaced publicly by [`PROJECT_STATUS.md`](../public/PROJECT_STATUS.md) |

### 2c. Three leaks bigger than `docs/protocols/`

Round one flagged `docs/protocols/` as the blocking item. It was not the largest one, and
two of these three are **already on the public remote today**.

**A gate file — a complete executable rule, and already public.** In one artifact: the
rule's name, its per-symbol threshold set, the bucket size, the horizon, the minimum
sample count, the cooldown — **and its hit rate, its baseline, and the delta between
them**. It is simultaneously the rule and the result. The frozen protocols at least had
the decency to be documents; this is the machine-readable version.

**Derived candidate specifications — also already public.** Each line names the feature
column, the quantile bounds, the horizon, the cooldown, the side and the entry model. It
is a list of trading rules in a line-delimited format.

**The research tooling — the rules as code, also already public.** Roughly 380 modules
with fee constants, thresholds and frozen-rule logic written into them, including a live
executor. Withdrawing the frozen protocol *documents* while leaving that code in place
would be theatre: the documents describe what the code already does.

Per-symbol fill-model parameters and a paper scoreboard belong to the same class.

## 3. What stays, and why it is enough

The public repository keeps **1,300 files** and loses nothing that makes it worth
looking at.

| Path | Files | Note |
|---|---:|---|
| `execution/`, `risk/`, `brain/`, `exchanges/`, `strategies/`, `core/`, `bot/` | ~180 | the engine. **Scanned and clean** — no rule specification, no threshold constant, no result. This is the strongest thing on the public surface and it survives the audit untouched |
| `tests/` | ~600 | the test suite, including the chaos and invariant suites CI requires |
| `tools/` (non-research) | ~320 | operational and diagnostic tooling |
| `dashboard/`, `web/`, `notifications/`, `integrations/`, `monitoring/`, `scripts/`, `config/`, `utils/`, `data/` (code), `src/eclipse/` | ~180 | operational subsystems and the static site |
| `docs/public/`, `docs/assets/` | ~20 | the purpose-written public documentation and its visual system |
| `docs/` curated set | 8 | `ARCHITECTURE` · `INVARIANTS` · `EXECUTION_CONTRACTS` · `OPS_RUNBOOK` · `ENV_REFERENCE` · `MICROSTRUCTURE_DATA_CONTRACT` · `DEBUG_OPERATIONS` · `PAPER_TRADING_ARCHITECTURE` |
| `.github/` | 11 | CI workflows and templates |
| root | 15 | see below |

**The engine being clean is the load-bearing result of this audit.** The showcase is the
execution architecture, the invariant contracts, the chaos testing and the reliability
gate — and none of that has to be withdrawn. What leaves is the research *content*: the
rules, the outcomes, and the machinery that produced them.

### Target root

```
README.md              CONTRIBUTING.md        SECURITY.md
requirements.txt       pytest.ini             .gitignore   .dockerignore
Dockerfile             docker-compose.yml
main.py                settings.py
start_eclipse.ps1      stop_eclipse.ps1       status_eclipse.ps1
run-bot.ps1            run-bot-ps2.ps1        run-bot-eth-test.ps1
```

Fifteen files, from sixty-eight. The fifty-three that leave are internal planning
documents that were never public artifacts — roadmaps, gap analyses, reconciliation
matrices, session reports, whitepapers, preregistrations, per-batch status files.

### Directories that disappear entirely

`ami/` · `reports/` · `runs/` · `runtime/` · `localtests/` · `eclipse_scalper/` ·
`state/` · `tmp/` · `pine/` · `research/` · `deep_discovery/` · `.claude/`

## 4. The research corpus: why nothing is curated out of it

The instruction was to select a public-safe sample from the 816 tracked S34 reports and
send the rest private. **That recommendation is withdrawn, and the whole corpus goes
`INTERNAL_ONLY`.**

A figure scan over the 412 markdown reports found **318 carrying a figure, threshold or
horizon, and 94 carrying none**. The 94 looked like the curation shortlist. One was
opened and read before recommending it, and it turned out to be a parameter-sweep
calibration grid — outcome columns across a grid of configurations — with an absolute
local path and a database size alongside it.

So the scan's false-negative rate on its own shortlist is not zero, and one sample is not
enough to estimate it. Two options follow: read all 412 by hand, or publish none of them.
Reading 412 dense internal reports to extract material that a purpose-written page
already covers is not a proportionate use of anyone's attention.

The public replacement is written from scratch and reviewed as public writing:
[`RESEARCH_METHOD.md`](../public/RESEARCH_METHOD.md) and
[`HISTORICAL_RESEARCH_CONTEXT.md`](../public/HISTORICAL_RESEARCH_CONTEXT.md).

**None of this is deletion.** The corpus is research evidence and it stays — on a private
remote, where a scan being imperfect costs nothing.

## 5. The 114 flagged source files

Not classified. Queued for a human, because the classifier cannot tell a synthetic
fixture from a real constant and should not pretend to.

| Where | Files | The question to answer |
|---|---:|---|
| `tests/` | 72 | is this fixture invented, or copied from a real configuration? |
| `tools/` | 37 | is this constant a default, or a frozen parameter? |
| `data/`, `dashboard/` | 5 | is this a schema example, or a live value? |

One worked example of why this needs eyes: a ranking test fixture carries a symbol, a
horizon, a minimum imbalance, a minimum trade intensity and a maximum spread — the exact
shape of the real rule, with plausible magnitudes. Whether the numbers are the real ones
or a near-miss cannot be settled by regular expression, and the difference matters.

Regenerate the manifest and filter on its `needs_review` column to work through them.

## 6. Execution order

For the owner, when authorised. Nothing below has been done.

1. **Create the private destination first.** A private remote, or a private repository.
   Nothing is untracked until its contents are somewhere else. `INTERNAL_ONLY` means
   *preserved elsewhere*, never *deleted*.
2. **Push the full current tree there**, so the internal history survives intact.
3. **Work the 114 flagged source files** (§5). Their outcome may move a handful of files
   between dispositions.
4. **Untrack, in two passes** — `git rm --cached -r`, which removes from the index and
   leaves the working tree alone:
   - pass one: `REMOVE_FROM_PUBLIC` (15,555). Low risk, no destination needed.
   - pass two: `INTERNAL_ONLY` (2,185), only after step 2 is confirmed.
5. **Extend `.gitignore`** so nothing returns by accident.
6. **Fix the `PUBLIC_REWRITE` six** — `README.md`, `docs/research/README.md`,
   `web/README.md` (it publishes an absolute local path), and the three launcher scripts
   (role names, symbol lists and ports).
7. **Fix the eight `CURATED_PUBLIC` documents** — each cites module paths that do not
   resolve. See the audit's §T-11.
8. **Re-run both checkers**, then re-read the tree as a stranger would.
9. **Only then** commit, branch, push, open a pull request.

Steps 4 and 5 stop *future* exposure. They do nothing about what is already in the
published history — [`GIT_HISTORY_LEAKAGE.md`](GIT_HISTORY_LEAKAGE.md) covers that
separately, and it changes what step 9 should be.

## 7. What this round did not do

- did not move, delete or untrack any file
- did not edit any `INTERNAL_ONLY` or `DO_NOT_TOUCH` file
- did not commit, branch, push or open a pull request
- did not change any repository setting
- did not start a collector, query a database, run a backtest or open a research outcome
- did not touch a running process
