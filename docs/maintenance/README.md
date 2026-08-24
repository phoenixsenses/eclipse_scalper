# Maintenance

**This subtree is not for visitors.** It is the release engineering behind the repository
— how it is assembled, what may appear in it, and the record of the decisions that shaped
it. Nothing here explains what Eclipse does; [`../public/`](../public/) does that.

It is kept in the open rather than deleted for one reason: a repository that publishes a
policy should also publish the machinery that enforces it, and the record of what that
machinery caught. A policy nobody can check is a claim.

---

## The machinery

| File | What it does |
|---|---|
| [`public_allowlist.json`](public_allowlist.json) | the allowlist this repository is assembled from — every include, exclusion, named exception, declared boundary and sanitization, each with its reason |
| [`tools/build_public_mirror.py`](tools/build_public_mirror.py) | assembles the tree from that allowlist; refuses to build on an undeclared boundary, on research provenance in any file, or on a file that does not parse |
| [`tools/check_public_docs.py`](tools/check_public_docs.py) | checks the documentation against the publication policy; mutation-tested against 28 deliberate violations |

A third tool, the classifier that decided each file's disposition in the first place,
runs against the source repository and is not shipped here: it has nothing to classify in
this tree, and its fixtures name individual internal artifacts by path.

Run these from the repository root:

```bash
python docs/maintenance/tools/check_public_docs.py             # 0 = clean
python docs/maintenance/tools/check_public_docs.py --self-test  # inject violations, all must be caught
```

## The record

| Document | What it answers |
|---|---|
| [`PUBLICATION_RISK_REGISTER.md`](PUBLICATION_RISK_REGISTER.md) | what may be published, what may not, and what is still an owner decision |
| [`SOURCE_PUBLICATION_REVIEW.md`](SOURCE_PUBLICATION_REVIEW.md) | the file-by-file review of every source file a scanner flagged, and what the scanner missed |
| [`PUBLIC_TREE_PLAN.md`](PUBLIC_TREE_PLAN.md) | every file in the source repository, classified, with the target tree |
| [`PUBLIC_SURFACE_AUDIT.md`](PUBLIC_SURFACE_AUDIT.md) | what the repository showed before this work, measured rather than assumed |
| [`GIT_HISTORY_LEAKAGE.md`](GIT_HISTORY_LEAKAGE.md) | what a published history retains after a file is removed, measured |
| [`REPOSITORY_METADATA_RECOMMENDATIONS.md`](REPOSITORY_METADATA_RECOMMENDATIONS.md) | the exact description, topics and social preview to set |
| [`FINAL_PUBLIC_AUDIT.md`](FINAL_PUBLIC_AUDIT.md) | this repository reviewed as a stranger would meet it, before it was published |

## The one rule for anyone extending this

**A rule that never fires reads exactly like a rule that passes.**

Every checker here is mutation-tested: deliberate violations are injected and every one
must be caught. If you widen a rule, add mutants for it and re-run the self-test. That
discipline exists because a widened rule was once silently broken and reported a clean
surface for a while — caught only because the mutants were re-run afterwards.

## What these documents describe, and do not

They describe a *separation*: this repository is a public engineering framework, and the
research estate it was separated from is private. They do not describe the research
estate, name its contents, or quantify anything in it.

Where one of them says a leak was found, it says what *kind* of thing it was and where the
check now sits — never the value. A document explaining why a figure must not be published
does not get to print the figure.
