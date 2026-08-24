# PUBLIC REPOSITORY PROVENANCE

What this repository is, where it comes from, and what it deliberately does not contain.

---

## What you are looking at

Eclipse is two things that live in two places.

**A public engineering framework** — this repository. An execution and risk engine, its
safety contracts, its test suite, its CI gates, and the documentation that explains how
the system decides what is true.

**A private research estate** — not this repository. The measurement lanes, the frozen
rule specifications, the preregistrations, the outcome ledgers, the governance
subsystem, and the report corpus they produce.

That split is deliberate, and it is the architecture rather than an omission. A framework
whose purpose is to keep *information*, *mechanism*, *economic value*, *execution
feasibility* and *operational value* apart can be shown in full. The specific rules it has
tested, and what they returned, cannot be — and would not be worth much to a reader
anyway, since the project's own record says no route is validated.

## How this repository is assembled

It is **not** a copy of the internal repository with files deleted. It is a separate
publication artifact, built from an explicit allowlist:

```
docs/maintenance/public_allowlist.json     what may be published, each entry with its reason
docs/maintenance/tools/build_public_mirror.py   assembles the tree from that list
```

The direction matters. With a blacklist, forgetting an entry publishes something you did
not intend. With an allowlist, forgetting an entry publishes nothing. An unlisted path is
excluded by default.

Four rules do most of the work:

**Start from what must work.** The seed is the engine, plus the modules CI actually
invokes, plus their first-party import closure — not "everything that looks safe".

**A test follows its subject.** A test is published only if everything it imports,
transitively, is also published. This resolves the test suite by what it tests rather
than by scanning its fixtures for numbers, which is the only reliable way to tell an
invented value from a copied one.

**A boundary is documented, not fabricated.** If a published module reaches something
unpublished, the build refuses. Where the reach is a lazy import the code already guards
with its own fallback, the boundary is declared and the tool verifies it really is lazy.
Nothing is stubbed to make a check pass.

**Content can overrule location.** A build-time veto scans every allowlisted file for
research provenance and refuses to publish it even when the allowlist said yes. It exists
because the most instructive leaks found during this work were prose, not numbers — a
collector docstring naming the study it was built for, a configuration file whose keys
are lane vocabulary.

## History

This repository begins with a **new root commit**. The internal repository's history is
not imported.

Two honest consequences:

- The internal engineering history — hundreds of commits of real work — is not visible
  here. That is a cost of the split, accepted deliberately.
- **A new repository does not retract anything already published.** Material that was
  public before this separation remains in whatever clones, forks, caches and archives
  were taken. Nothing about this repository claims otherwise. It changes what is
  authoritative going forward; it does not reach backwards.

The internal repository's history is likewise **not rewritten**. Rewriting it would break
every existing clone, and — for a project whose stated discipline is that a correction
never edits the source it corrects — would be that rule violated on the repository
itself.

## What is not here, and why

| Absent | Why |
|---|---|
| Research reports and their outcomes | the research estate; results under an open contract are sealed until their evaluator opens them |
| Frozen rule specifications and protocols | entry and exit rules, offsets, horizons, thresholds, feature definitions |
| Derived alpha candidates and gate parameters | executable rules in machine-readable form |
| The research and shadow tooling | encodes rules, thresholds and cost constants in code |
| The epistemic governance subsystem | question registries, hypothesis state, failure archive |
| The operator dashboard | its aggregator reads internal research artifacts; a stripped version would be placeholder functionality, so the boundary is stated instead |
| The master state record | an operator's working log. [`PROJECT_STATUS.md`](PROJECT_STATUS.md) is its public counterpart |
| Runtime state, ledgers, databases, locks | operational state, and none of it belongs in a repository |
| Agent operating contracts | carry seal state and research status |

## What that means when you read the code

Some modules here have a lazy import or a configuration hook that points at something
this repository does not contain. Those are **real boundaries**, not broken code: in each
case the existing module already handles the absence through its own fail-closed path,
because the codebase was written on the assumption that any dependency can be missing.
The declared list is in `documented_boundaries` in the allowlist.

You can read every safety contract, run the full published test suite offline, and
reproduce every CI gate. What you cannot do from this repository is reconstruct a
strategy — and that is the intended outcome, not a gap.

## Verifying this repository

Everything below is offline and needs no credential, no exchange connection and no
database:

```bash
pip install -r requirements.txt
pytest -q
python docs/maintenance/tools/check_public_docs.py
```

The publication policy this repository is written under is stated in
[`PUBLICATION_RISK_REGISTER.md`](../maintenance/PUBLICATION_RISK_REGISTER.md) and enforced by that
checker, which is mutation-tested against deliberate violations.

## Licence

**No licence has been granted.** There is no `LICENSE` file, so default copyright applies
and no permissions are conveyed. This is a deliberate open decision, not an oversight —
see the note in the README.
