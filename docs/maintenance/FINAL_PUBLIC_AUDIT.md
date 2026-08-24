# FINAL PUBLIC AUDIT

**Date:** 2026-08-24 · **Subject:** the candidate release artifact, reviewed as a
stranger would meet it rather than as the operator who built it.

---

## Verdicts

| Dimension | Verdict |
|---|---|
| Visual quality | **PASS**, after one substantive fix |
| First screen | **PASS** |
| Technical credibility | **PASS** |
| Publication safety | **PASS**, after three findings closed in this pass |
| README coherence | **PASS** |
| Repository-root coherence | **PASS**, after a structural split |

## What the mirror is

| | |
|---|---:|
| Files | **390** |
| Size | **4.3 MB** |
| Root items | **29** (11 files, 18 directories) |
| Largest file | 126 KB (`execution/entry_loop.py`) |
| Test modules | 156 |
| Execution modules | 59 |
| Published tools | 30 |
| Documentation files | 34, of which 7 are SVG assets |
| Git history | none — the tree carries no `.git` |
| Licence | none — a standing owner decision |

---

## 1. Visual quality

The seven assets were rendered and looked at, not merely validated. That distinction
produced the one substantive finding of this pass.

**The hero was not an eclipse.** The occulting disc was filled with a gradient close
enough to the page ground that it disappeared, so the motif read as *a thin circle with a
gradient stroke* — a ring, not an eclipse. The whole identity rests on that shape and it
was not landing. Every mechanical check passed the file: geometry inside the viewBox, ids
resolving, no external reference, no text overflow. None of them can see that a picture is
not of the thing it is supposed to be.

Three changes fixed it:

- a **halo** — light spilling past the limb, which is what makes an occulted disc read as
  occulted rather than as a circle;
- a **denser body** than the ground it sits on, so the disc is a body;
- **removing the price ladder**. At README width it read as scratches beside the ring, and
  the idea it was gesturing at is carried properly by the microstructure diagram.

The social preview received the same treatment. Re-rendered, the hero now reads as an
eclipse at full width and at thumbnail size.

**Coherence across the set.** One ground, one hairline weight, one type pairing, one
radius. Blue, cyan and violet appear only as category accents; no labelled node in any
diagram carries a health colour; the single warm tone lives only in the corona, where it
labels nothing. The architecture diagram gained a **`PRIVATE RESEARCH ESTATE`** bracket
so no reader can mistake a private plane for a shipped one.

**What it is not:** not a Bloomberg imitation, not a fake terminal, not cyberpunk, no
fabricated metrics, no health chips, no third-party artwork.

## 2. First screen

A visitor meets, in order: the wordmark, one sentence of what Eclipse is, the five verbs,
four badges, and four statements about why the project exists. No implementation trivia,
no badge wall, and nothing that needs a private research result to be impressive.

## 3. Technical credibility

A senior engineer can see, and run:

- the intent lifecycle and the single order-submission path
- order-router idempotency and intent-lifecycle completeness, as tests
- restart and reconcile convergence
- the risk hierarchy and its one deliberate exemption
- deterministic, seeded simulation
- three named chaos scenarios
- **a reliability gate exercised in both directions** — passing on a clean fixture and
  required to fail on a degraded one. Verified in the mirror: exit 0 and exit 2.
- 156 test modules, and CI that references only files present here

No proprietary logic was added to make the codebase look larger. The engine was scanned
and is clean of rule specifications, threshold constants and results — which is why it
could be published whole.

## 4. Publication safety

Three findings were closed during this pass. All three came from reading rather
than from scanning, and the third was caught on the pushed repository while it was
still private — which is what the private-first step is for.

**Exact leak paths in the maintenance documents.** Two of them tabulated the specific
files in the old public history that carry an executable rule, with the dates they
entered. Honest, and also a lookup: it converts *somewhere in the record* into *this file,
this date*. The counts and dates are what a reader needs to judge the exposure; the paths
were reduced to categories.

**The classifier's fixtures.** Its self-test table named individual internal artifacts by
path. It also has nothing to classify in this tree. Excluded, with the reason recorded.

**A threshold triple inside a code span** — found on the fresh clone of the pushed
repository, while it was still private. `SOURCE_PUBLICATION_REVIEW.md` quoted a test
fixture verbatim to illustrate that a fixture can be shaped like a real rule, and in doing
so printed a horizon and three thresholds in the very page arguing they must not be
printed. The checker had passed it: inline code was being stripped before the content
rules ran, so a value inside backticks was invisible to them.

The strip exists so a document can *name* a banned label — write `Active` while stating
the ban. It was never meant to hide a *value*. The rule now applies to the raw line for
every value rule and to the stripped line only for the health rule, and a mutant covers
it. This is the fourth time in this work that a check passed something a person then
caught by reading, and the pattern is consistent: the checks catch classes, and reading
catches the instance that does not look like its class.

Standing checks, all green in the mirror:

```
publication checker      clean, 27 files
mutation self-test       29/29 caught
secret scan              no .env, no key material
absolute-path scan       none
DB / runtime / lock scan none beyond synthetic test fixtures
rule / candidate scan    none
compileall               clean
```

## 5. README coherence

Reordered so the reading flow matches what a visitor needs: identity, why, method,
lifecycle, architecture, safety, data, validation, reproducibility, frontier, module map,
quick start, CI, documentation, status, roadmap, and what Eclipse refuses to claim. The
page ends on the refusals and the disclaimer rather than on a roadmap.

Every path, command, badge and link was checked against the tree that actually ships. The
`DAT-*` and `VAL-*` invariant tests import research tooling and therefore live with it —
the README says so rather than shipping tests that cannot run.

Length: ~31 KB, dense, scannable.

## 6. Repository-root coherence

29 root items, each with a reason. No internal planning documents, no agent files, no
research receipts.

The structural change this pass made: `docs/` now separates **visitor documentation**
from **release-engineering tooling**.

```
docs/
  public/       7 documents written to be read from outside
  maintenance/  the allowlist, the checkers, and the record of what they caught
  assets/       the visual system
  + 8 curated operational and contract documents
```

Before the split, a visitor opening `docs/public/` met a publication risk register, a
surface audit, a tree plan and a source-publication review — the repository reading as an
internal release-engineering workspace rather than as a product. The machinery is still
public, because a policy nobody can check is a claim; it is just no longer in the path of
someone trying to understand what Eclipse does.

## 7. Outstanding defects

| # | Item | Severity | Note |
|---|---|---|---|
| 1 | No `LICENSE`. Default copyright applies; no permissions are conveyed | open by decision | stated in the README rather than papered over |
| 2 | A test in the source repository does not parse — an indentation error at a `try`. Excluded from the mirror; the fix is one line | internal, not a release blocker | the syntax gate that found it now refuses any unparsable file |
| 3 | The social preview must be rasterised to PNG before upload; no rasteriser is installed here | manual step | source, script and two fallbacks documented |
| 4 | `web/` is publication-ready and deliberately not in this repository | owner decision | one allowlist line if that changes |
| 5 | The `DAT-*` / `VAL-*` invariants are stated here but their tests are private | structural, disclosed | a consequence of the split, not a gap in the contracts |

None blocks publication.

## 8. What this audit did not do

- did not re-derive the allowlist decisions; those are in `PUBLIC_TREE_PLAN.md` and
  `SOURCE_PUBLICATION_REVIEW.md`
- did not touch the source repository's tracked files
- did not query a database, start a collector, run a backtest, or touch a process
- did not commit, push, or change any repository setting
