# GIT HISTORY LEAKAGE

**Date:** 2026-08-24 · **Question:** untracking a file stops it being published from now
on. What about what has already been published?

Measured against `origin/main` — the branch the public actually sees — not against the
local working branch. Nothing here has been acted on.

---

## 1. The good news, measured

**No `.env` file has ever been committed, on any branch, at any point.**

```
git log --all --diff-filter=A --name-only -- '.env' '.env.*'   →  no results
```

`.gitignore` covers `.env` and `.env.*`, and the glob catches the example files too. No
credential, key or token has been through this history. That is the one class of leak
that cannot be walked back, and it did not happen.

**Two of the three highest-risk paths were never public at all.**

| Path | Commits on `origin/main` | In `origin/main` today |
|---|---:|---|
| `docs/protocols/` | 0 | no |
| `AGENTS.md` | 0 | no |

The seven frozen mini-protocols — round one's blocking finding — **have never been pushed**.
They exist only on the local working branch. There is nothing to remediate: the fix is to
not push them, which is already the plan.

## 2. What is actually in the published history

Deliberately by category rather than by path. The counts and the dates are what a
reader needs in order to judge the exposure; the file names would only shorten
someone else's search, and this document exists to be honest, not to be a lookup.

| What | Commits on `origin/main` | First appeared | In HEAD |
|---|---:|---|---|
| A gate file: a rule and its measured result in one artifact | 1 | 2026-02-20 | yes |
| Fill-model parameters, and a paper scoreboard | 2 | 2026-02-20 | yes |
| Lock and pid files | 1 | 2026-02-20 | yes |
| Derived candidate specifications and a live plan | 2 | 2026-03-05 | yes |
| A strategy proxy in a charting language | 1 | 2026-03-05 | yes |
| The README's threshold rule | 1 | 2026-03-08 | yes |
| The historical report corpus | 26 | 2026-06-30 | yes |
| Research and shadow tooling | many | 2026-06-30 | yes |
| The governance subsystem | 31 | 2026-07-05 | yes |
| The master state record | 63 | 2026-07-05 | yes |
| The agent operating contract | 3 | — | yes |

`origin/main` holds **518 commits**, beginning 2026-01-13. **122 of them touch at least
one path on this list.**

Three things follow.

**The exposure is old.** The earliest offending commit is 2026-02-20 — the thirty-eighth
day of the project. Everything after it is downstream of at least one leak.

**Most of it entered once and never moved.** Eight of the thirteen rows are single
commits. That is a small blast radius per artifact, and a large one in aggregate, because
each is present in every commit that followed it.

**The README threshold rule entered in exactly one commit** — `d34c8c5e`, 2026-03-08,
*"docs(readme): rewrite README with full architecture and research pipeline"* — and was
never removed. It is live in `origin/main` today. Round one's rewrite is the first time
it is taken out, which means it has been continuously public for roughly five and a half
months and is in every clone, fork and cache taken during that time.

## 3. What a history rewrite would actually cost

A rewrite (`git filter-repo`, or BFG) that removed every path above would have to run
from 2026-02-20 onward. Since that is commit ~40 of 518, this is not "rewriting the tail"
— it is **rewriting the repository**.

What that buys, and what it costs:

| | |
|---|---|
| Buys | the values disappear from the canonical remote's history |
| Costs | every commit hash after 2026-02-20 changes |
| | every existing clone and fork is orphaned and must be re-cloned |
| | open pull requests and branches break |
| | any mirror, cache or archive taken before the rewrite still holds the old objects |
| | the repository's own history — 518 commits of real engineering — becomes untrustworthy as a record, because it no longer matches what happened |

That last cost is the one that matters most for a project whose entire public argument is
that it does not edit its own record. `REPRODUCIBILITY.md` states the rule in its own
words: *superseded statements are marked as superseded, not erased*, and *a correction
never edits the source it corrects*. A history rewrite is exactly the move that rule
forbids, applied to the repository itself.

## 4. Recommendation: a clean public repository, not a rewrite

Build a **new public repository** from the `PUBLIC_KEEP` / `PUBLIC_REWRITE` /
`CURATED_PUBLIC` / `DO_NOT_TOUCH` set — 1,300 files — with no inherited history. Keep the
current repository as the private one, intact, history and all.

| | Rewrite in place | Clean public repository |
|---|---|---|
| Old values reachable from the public remote | no | no |
| Existing clones and forks | orphaned | untouched (they point at what becomes the private repo) |
| Internal history preserved | mutilated | fully intact, privately |
| Rewrites the project's own record | yes | no |
| Consistent with the project's stated discipline | no | yes |
| Reversible | no | yes |
| Effort | high, and delicate | moderate, and boring |

Two honest limitations, because this recommendation is not a containment guarantee:

- **Neither option recovers what is already out.** Clones, forks, caches and archives
  taken over the last five and a half months hold what they hold. Both options are
  forward-looking, and the clean-repository option does not claim otherwise.
- **The values in question are development-era and superseded.** The README rule and the
  gate file come from a research line that later work replaced, and the project's own
  record says no route is validated. That lowers the value of what leaked without making
  the leak acceptable.

If the owner disagrees and wants the rewrite, it is a separate, carefully-planned
operation with a fresh backup taken first — not something to fold into a documentation
pass.

## 5. Sequencing

The recommendation changes what round three's final step should be. Round one ended with
*"commit, branch, push, PR"*. If a clean public repository is chosen, there is no push to
`origin/main` at all — the sequence becomes:

1. make the current remote private
2. create the new public repository
3. populate it from the classified tree, as a single initial commit or a small curated
   set
4. re-run both checkers against it
5. read it as a stranger, then publish

Setting an existing GitHub repository to private, and creating a new one, are both
account-level actions. They are the owner's to take.

## 6. Commands used

Recorded so the numbers can be re-derived rather than taken on trust.

```bash
git log --all --diff-filter=A --name-only -- '.env' '.env.*'
git log origin/main --oneline -- <path> | wc -l
git log origin/main --diff-filter=A --format="%ad %h" --date=short --reverse -- <path> | head -1
git cat-file -e origin/main:<path>
git log origin/main --oneline -S"<the threshold string>" -- README.md
git rev-list origin/main --count
```
