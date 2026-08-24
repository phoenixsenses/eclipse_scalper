# SOURCE PUBLICATION REVIEW

**Date:** 2026-08-24 · **Subject:** the 114 source files the classifier flagged and
refused to decide · **Method:** the matched lines were extracted and read, then each file
was judged by what it is for — not by whether a number appears in it.

The classifier stops at source code on purpose. A number in a data artifact is data; a
number in a program may be a synthetic fixture, an initialiser, a generic engineering
constant, or a real research parameter, and no regular expression separates those four.
This is the pass that does.

---

## 0. Outcome

| Verdict | Files |
|---|---:|
| `INTERNAL_RESEARCH_IP` | 92 |
| `PUBLIC_SAFE_AFTER_SANITIZATION` | 11 |
| `PUBLIC_SAFE` | 11 |
| `UNCERTAIN_MANUAL_REVIEW` | 0 |

Nothing was left uncertain. Where a judgement was close, the file was excluded — an
allowlist should fail towards silence.

---

## 1. The rule that decided 63 of the 72 tests

**A test follows its subject.**

Reading a test's fixtures to decide whether it may be published is the wrong question.
A fixture of two round numbers is obviously invented; a fixture carrying a horizon, a
minimum imbalance, a minimum intensity and a maximum spread is obviously shaped like a
real rule — and whether its values *are* the real ones cannot be settled by looking at
them. But the module a test imports is not ambiguous at all.

So the test suite is resolved by import closure: a test is published only if everything
it imports, transitively, is also published. That rule is implemented in the mirror
builder rather than applied by hand, so it cannot drift.

Applied to the 72 flagged tests: **63 inherit `INTERNAL_RESEARCH_IP`** because they test
the AMI governance subsystem, the research tooling, or a sealed lane. Applied to the full
suite, it drops 510 of 651 tests and keeps 141.

The nine that did not resolve that way were read individually:

| File | Verdict | Why |
|---|---|---|
| `tests/test_analyze_cost_breakdown.py` | `INTERNAL_RESEARCH_IP` | fixture binds a horizon; subject is a research cost tool |
| `tests/test_fit_adverse_model.py` | `INTERNAL_RESEARCH_IP` | fixture carries a populated threshold object — three quantile bounds and a horizon — in the shape the live gate file uses |
| `tests/test_dashboard_live_metrics_api.py` | `INTERNAL_RESEARCH_IP` | subject is a dashboard adapter over internal ledgers |
| `tests/test_dashboard_paper_bucket_ledger.py` | `INTERNAL_RESEARCH_IP` | same |
| `tests/test_dashboard_shadow_observatory.py` | `INTERNAL_RESEARCH_IP` | same |
| `tests/test_dashboard_shadow_paper_activity.py` | `INTERNAL_RESEARCH_IP` | same; fixtures also carry shadow-runner command lines |
| `tests/test_notifications_extended.py` | `PUBLIC_SAFE` | a single round `win_rate` in a notifier fixture; subject is the notifier |
| `tests/test_pid_registry_identity.py` | `PUBLIC_SAFE` | repeated `pid: 1234`; a process-identity test |
| `tests/test_verify_data_layer_status.py` | `PUBLIC_SAFE` | synthetic pids and collector command strings; subject is the data layer |

## 2. The 37 flagged tools

`tools/` is excluded by default, so the question for each was whether it earns an
exception. Most did not, and the reasons cluster.

**Real results written into source — `INTERNAL_RESEARCH_IP`.** These carry measured
values as literals or in prose, not as computation:

- a diagnostics module whose plot caption states per-symbol win rates, trigger sizes and
  a daily drift figure, all real
- a second diagnostics module stating a win rate together with a take-profit and a
  stop-loss — a result *and* a rule in one comment
- an alpha-decision reporter whose table contains a rejected route, its fill rate and the
  two expected values being compared
- a presentation builder carrying worked examples with real averages
- a candidate-availability audit quoting a median against a lookahead comparison
- a milestone experiment whose population line names trigger sizes and whose claim
  template emits a feature, a direction and a threshold
- a fill-calibration audit binding a maker fee as a module constant

**Report generators — `INTERNAL_RESEARCH_IP`.** Roughly twenty modules whose output *is*
a results table: their markdown headers enumerate outcome columns. They publish no value
themselves, but they exist to produce ones, and their column vocabulary describes the
research programme's scoring in detail.

**Operational summarisers — `PUBLIC_SAFE`.** A daily report and a paper-trade summariser
also emit tables with a win-rate column, and were kept. The distinction is what they read:
a database the public does not have, supplied by the operator. A summariser over absent
data publishes a shape, not a result. One of them flags an anomaly below a fixed win-rate
threshold; that is an operational alerting bound, not a strategy parameter.

**Infrastructure — `PUBLIC_SAFE`.** Health state, ingestion checks, collection health,
data-readiness checks, trade replay, run summary, microstructure-contract validation.
Zero research signals in all of them.

**One tool was excluded for a single line.** A live gate checker states a complete pocket
rule — a horizon and three thresholds — twice, once in its docstring and once in its
output. Its only caller states the same rule in *its* docstring. Both were excluded
rather than redacted: removing a rule from two modules that exist to evaluate that rule
would leave modules whose purpose had been deleted, and the one caller imports them
lazily behind a fallback the codebase already ships.

## 3. The five others

| File | Verdict | Action |
|---|---|---|
| `dashboard/backend/adapters/execmgmt_stopprot.py` | `INTERNAL_RESEARCH_IP` | binds a real historical worst-fill figure as a module constant. Excluded with the rest of the dashboard |
| `dashboard/backend/data_sources.py` | `INTERNAL_RESEARCH_IP` | parses internal ledger formats. Excluded |
| `data/setup_scheduled_tasks.ps1` | `PUBLIC_SAFE_AFTER_SANITIZATION` | absolute install root → script-relative |
| `data/start_collector.bat` | `PUBLIC_SAFE_AFTER_SANITIZATION` | same |
| `data/start_diary.bat` | `PUBLIC_SAFE_AFTER_SANITIZATION` | same |

## 4. What the review found that the flag list did not

Three leaks arrived from outside the 114, and each is worth recording because each shows
a different blind spot.

**A research lane's configuration file.** A JSON config carrying frozen window
parameters — parent windows, anchor bucketing, anchor gap. It matched no numeric pattern
because its keys are lane vocabulary, and no path rule because it sits in `config/`
alongside ordinary settings.

**Two collectors whose docstrings are research provenance.** Both name the study they
were built for, one names its preregistration and its estimand, one cites a master-state
section. Not a number anywhere. Prose is a leak channel and the numeric scanners are
blind to it.

**An adapter and a publishing bus for a sealed arm**, plus the tests around them.

All five were caught by a **content veto** added afterwards: a build-time refusal that
scans the *post-sanitization* text of every allowlisted file for research provenance —
study ids, master-state sections, preregistration and estimand vocabulary, frozen window
parameters, sealed-arm names, internal report paths — and refuses to build if any
survives. The allowlist decides; the veto can overrule it. Neither alone would have been
enough.

The veto also caught eight engine and tool files citing internal research reports in
comments. Those are not leaks of content, but they are dangling references that advertise
the internal layout, so one declared rule replaces the path with a neutral phrase. No
behaviour changes.

## 5. What is deliberately not claimed here

- **This review is not a proof of absence.** It is one careful pass over files a scanner
  pointed at, plus a veto that catches a class the scanner cannot see. Both are
  fallible, and the mirror is small enough to be read by a person before it is published.
- **A `PUBLIC_SAFE` verdict is about research content, not about code quality.** Nothing
  here says a published module is correct, useful, or well-written.
- **The eleven sanitized files differ from their internal originals.** Every difference
  is declared in `public_allowlist.json` with its reason, and the build fails if a
  declared redaction stops matching — so a leaked line cannot survive a rename of the
  text around it.
