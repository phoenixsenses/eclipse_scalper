# Contributing

Eclipse is a reliability-critical research system. A change here is held to a higher bar
than a change to an ordinary application, and the bar is not about code style.

Read this before opening anything. It is short, and most of it is unusual.

---

## Before you start

> **No licence has been granted.** There is no `LICENSE` file in this repository, so
> default copyright applies. Until the owner adds one, the terms under which a
> contribution could be accepted and redistributed are undefined. **Open an issue before
> writing code**, and please do not send substantial work in the meantime.

## The four questions

Every change — a fix, a feature, a refactor — is expected to answer these before any code
is written. They are the project's operating doctrine, not a review checklist bolted on
afterwards.

1. **What invariant could this break?** Name it by ID from
   [`docs/INVARIANTS.md`](docs/INVARIANTS.md) (`EXE-*`, `DAT-*`, `VAL-*`, `SAF-*`). If
   none, say so and say why.
2. **What lifecycle state could this affect?** Trace the intent and position lifecycles.
   Verify every branch still reaches a terminal state.
3. **What persistence state could this corrupt?** Brain state, intent ledger, `state/*.json`,
   JSONL field names. Schema changes need migration, not hope.
4. **What reconcile behaviour could this disrupt?** Reconcile is the only component
   permitted to assert truth about positions and fills.

If any answer indicates risk: state the risk, describe the mitigation in the change, and
name the test that verifies the mitigation.

## Minimal diff

Smallest correct change. Specifically:

- touch only the files necessary
- no opportunistic refactoring, no reformatting, no "while I'm here"
- no docstrings or type annotations added to code you did not otherwise change
- prefer a feature flag with a safe default over replacing behaviour, when the change
  touches `execution/`, `risk/` or `brain/`

Small diffs are not a stylistic preference. Invariants in this system interact across
modules, and a large diff raises the probability of an unintended interaction faster than
it raises the value delivered.

## Things that are never accepted

Regardless of framing, benchmark, or how much cleaner the result is:

- a second order-submission path outside `execution/order_router.py`
- removing an idempotency key or a correlation id
- an intent left without a terminal state on any branch
- an unbounded retry loop
- entry-gate logic applied to a `reduce_only` protective exit
- non-seeded randomness in any research tool
- wall-clock time in scoring or simulation logic
- a renamed or removed JSONL field (additive only)
- a persistence schema change without migration
- any code that logs, prints or persists a secret
- code that assumes internal state and exchange state agree at a given instant

## Research contributions

Additional rules apply in `tools/`, `src/`, and `execution/passive_execution_simulator.py`:

- **No lookahead.** A signal at `t` uses only data at index ≤ `t`. Centred windows,
  negative shifts, and forward-fill-before-feature are the three that actually happen.
- **Determinism.** Same inputs plus same seed produce identical outputs. Seeds flow down
  the whole call chain and are echoed into report headers.
- **Event ids** derive from input data, never from a fresh UUID or the wall clock.
- **Cost units.** The bps-to-ratio conversion is applied exactly once.
- **Parsers report their counters** — rows parsed, rows skipped with sample reasons,
  pass/fail counts. A parser that silently returns zero candidates is a defect, not an
  empty result.

If a contribution produces a *result*, it also needs a preregistration frozen before the
outcome was opened, and a statement of what would refute it. A result computed on a window
that also produced the rule is a development observation, and it is labelled as one.

See [`docs/public/RESEARCH_METHOD.md`](docs/public/RESEARCH_METHOD.md).

## Documentation and the public surface

Anything under `README.md`, `docs/public/` or `docs/assets/` is public-facing and falls
under the publication policy stated in
[`docs/maintenance/PUBLICATION_RISK_REGISTER.md`](docs/maintenance/PUBLICATION_RISK_REGISTER.md).

**Never publish:** entry or exit rules · offsets · horizons · thresholds · feature
definitions · formulas · any performance figure (bps, win rate, profit factor, drawdown,
totals, or a comparison implying one) · rankings or comparisons between arms · anything
derived from a sealed forward arm in any aggregated form · hostnames, IPs, ports,
credentials, real network layout, live positions.

**Never claim health.** Nothing in Eclipse is running for a public reader. No label may
read `Active`, `Healthy` or `Running`, and green / amber / red may never carry the state
of a component. Use `accepted` / `building` / `design` / `planned` / `research` /
`refuted` / `parked`.

Check it rather than eyeballing it:

```bash
python docs/maintenance/tools/check_public_docs.py             # README + docs/public + docs/assets
python docs/maintenance/tools/check_public_docs.py --self-test  # 29 mutants, all must be caught
```

**If you extend the checker, add mutants for the new rule and re-run the self-test.** A
rule that never fires reads exactly like a rule that passes — that failure mode has
already happened here once, and was caught only because the mutants were re-run.

## Before you open a pull request

```bash
python -m py_compile <every file you changed>
pytest -q                                    # or targeted files while iterating
python docs/maintenance/tools/check_public_docs.py --self-test   # if you touched public docs
```

Run at most a couple of test files per `pytest` invocation while iterating; the suite is
large, and some environments need `--basetemp` pointed at a writable scratch directory.

CI will additionally run the three required chaos scenarios, the execution invariant
suites and the reliability gate.

## Pull request description

State, in this order:

1. root cause (for a fix) or motivation (for a feature)
2. files touched, and why each one
3. invariant impact — even if the answer is "none, because …"
4. validation evidence — the compile command, the test command, and their output
5. any known gap you are leaving behind

A change that cannot describe its own invariant impact is not ready, however correct the
code is.

## Review

Substantive changes go through:

```
implementation → independent review → correction → independent re-review → acceptance
```

The phases run separately. A review is read-only and changes nothing; corrections happen
in their own phase. This is not bureaucracy — the value of a review is its independence,
and an author who produces and approves an artifact in one pass has approved nothing.

Time pressure is not a reason to compress the chain.

## Reporting a security issue

Do not open a public issue. See [`SECURITY.md`](SECURITY.md).
