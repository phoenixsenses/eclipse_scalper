# PROJECT STATUS

The public statement of where Eclipse is. It replaces `SYSTEM_STATE.md` on the public
surface — that file is the operator's working record, written for an operator, and it
stays private.

**Last reviewed:** 2026-08-24.

State vocabulary, identical to the one the public site uses:

`accepted` built **and** passed an independent review gate · `building` exists as code,
under construction · `design` specified, not built · `planned` neither · `research` an
open question · `refuted` closed by a test, kept on the record · `parked` not refuted,
blocked on something that does not exist yet

`accepted` describes the review state of code. It says nothing about a running thing and
nothing about a market result.

---

## Where things stand

| Area | State | What that means here |
|---|---|---|
| Execution engine | `building` | intent lifecycle, gates, router, verifier, reconcile and position management exist and are exercised by CI |
| Execution safety contracts | `building` | the invariants are stated, and each names its enforcing test |
| Chaos and reliability gates | `building` | three required chaos scenarios plus a reliability gate that is tested in both directions |
| Research method and governance | `building` | preregistration, frozen contracts, hypothesis ledger, decision log, failure archive |
| Measurement and coverage gates | `building` | observability, joint coverage and target semantics checked before a result is read |
| Forward and fresh-epoch observation | `building` | accumulating under sealed contracts |
| Operator dashboards | `building` | not in this repository — an operator surface over the private estate |
| Publication policy and its checker | `accepted` | mutation-tested, and independently reviewed |
| Public repository surface | `accepted` | assembled from an explicit allowlist; every file decided with a recorded reason |
| Cross-market portability | `research` | pooled cross-sectional testing rather than per-symbol selection |
| Live capital deployment | `planned` | live execution is off by default and requires an explicit launcher flag |

## What is being worked on

Concept level. Outcomes under an open contract are sealed until their evaluator opens
them, and none appear here.

- **Level-1 queue state and price innovation** — what the top of the book can and cannot
  say about the next price, posed as a first-passage question rather than a prediction
  problem.
- **Measurement fidelity versus target semantics** — separating *the record is faithful*
  from *the record is about the right object*.
- **The endogenous market clock** — whether activity-driven variance belongs to a symbol
  or to a common market clock. Where it is common, it is a risk-state feature and never
  an alpha.
- **Order-flow memory and event arrival** — whether arrivals cluster beyond what
  seasonality already explains.
- **Execution timing and cost realism** — what a fill model must get right before an
  execution result means anything.
- **Risk governors** — path risk rather than tail incidence, and what a governor would
  have to demonstrate to be worth having.
- **Decision-value identification** — whether a ranking metric identifies decision value
  *at all*. Prior to asking whether one metric beats another, and repeatedly the binding
  constraint.
- **Fresh-epoch replication discipline** — keeping an untouched window genuinely
  untouched, which is mostly a bookkeeping problem and harder than it sounds.

## What is closed

Closed ideas are archived with the condition that closed them, and the protocol forbids
re-testing them. The archive itself is internal — publishing a list of what did not work
is still publishing what was tried — but the *shape* of what closes is public, because it
is the most instructive part of the record:

- routes closed because their **oracle ceiling sat under the cost of trading them**, with
  the direction itself intact. Being right about the sign is not the same as having a
  trade.
- routes closed because a result did not survive a **multiplicity correction across the
  whole programme**, despite a clean permutation test and a clean walk-forward.
- questions closed as **`NOT IDENTIFIED`** — the estimand was not identified, the supply
  of independent units was insufficient, or the decision object the result would inform
  did not exist. The study design, not the market, was the binding constraint.
- an execution-model assumption closed because a **measurement contradicted it**: a price
  level empties through cancellation as readily as through trading, so displayed quantity
  is not a countdown.

## What Eclipse does not claim, as of this review

- No route is claimed to be validated. Not one.
- No component is claimed to be running, healthy or profitable.
- The fee basis is not uniform across all historical code paths; it is tracked as an open
  contradiction, so results computed on different bases are not cross-comparable.
- Development-era results are not evidence about the future, and the project's own record
  contains results that looked clean and did not survive correction.

## Recent structural work

| When | What |
|---|---|
| 2026-08-24 | This repository separated from the internal research estate and assembled from an explicit allowlist. See [`PUBLIC_REPOSITORY_PROVENANCE.md`](PUBLIC_REPOSITORY_PROVENANCE.md) |
| 2026-08-24 | Documentation rewritten under a machine-checked publication policy, mutation-tested against deliberate violations |
| 2026-08-24 | Every source file a scanner flagged reviewed by hand; the record is in [`SOURCE_PUBLICATION_REVIEW.md`](../maintenance/SOURCE_PUBLICATION_REVIEW.md) |

## How this file is maintained

It is a **summary**, not a log. It carries no outcome, no figure, no threshold, no
horizon and no sealed aggregate, and it is checked mechanically:

```bash
python docs/maintenance/tools/check_public_docs.py
```

A component has one state. When it changes it changes everywhere in the same pass — two
surfaces disagreeing about a component is how a project ends up telling two stories about
itself.
