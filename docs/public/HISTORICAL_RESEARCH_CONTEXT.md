# HISTORICAL RESEARCH CONTEXT

> **`HISTORICAL / DEVELOPMENT / NON-CURRENT`**
>
> This page describes an earlier generation of Eclipse's research. It is kept because
> deleting superseded work is how a project loses the ability to explain itself — not
> because any of it is the current state of the system. **Nothing on this page should be
> read as a claim about what Eclipse does now, or about what works.**
>
> Every threshold, formula, horizon, verdict and measured value from the original write-up
> has been removed. What remains is the *shape* of the work and what it turned out to
> mean. See [`PUBLICATION_RISK_REGISTER.md`](../maintenance/PUBLICATION_RISK_REGISTER.md) §P-2 for why.

---

## What the old front page said

Until this rewrite, the repository's README presented Eclipse as a scalper for a single
perpetual-futures symbol, organised around:

- a small set of one-second bucket features computed from tick-level order flow
- a filter sweep that searched combinations of those features for a favourable subset
- a passive-fill simulator that modelled limit-order execution rather than assuming it
- a forward split with a cost-sensitivity sweep before any configuration was promoted
- a regime split that conditioned the same configuration on a rolling directional state

The page then displayed the winning configuration, its screening results, a per-variant
verdict column, and a break-even cost figure.

## Why that page is no longer the front page

Four reasons, in increasing order of importance.

**1. It published what the project's own policy forbids.** Eclipse's public content
policy — written for the website, machine-checked, and mutation-tested — prohibits
publishing entry rules, offsets, horizons, thresholds, feature definitions, formulas and
performance figures. The README predated that policy and was never brought under it.

**2. The numbers were stale.** They came from a research line that later work
superseded. A visitor was taking away a picture of the system that was many research
generations out of date.

**3. A screening verdict was being read as a validated one.** The verdict column meant
*this configuration passed a development screen*. On a front page, next to a threshold
rule, it reads as *this is the edge, and here is how to run it*. Those are not the same
statement, and the distinction between them is the single thing this repository exists
to maintain.

**4. It was the wrong headline.** The most defensible thing Eclipse has built is not a
configuration. It is the machinery that decides whether a configuration means anything —
preregistration, coverage gates, falsification, fresh replication, independent review.
Leading with a filter triple buried all of it.

## What the old line actually taught

The findings that survived are methodological, and they are worth more than the
configuration was.

**A filter sweep manufactures its own winner.** Search a wide enough space of feature
combinations and the best one looks excellent by construction. The number that matters is
not the winner's score but how many independent ideas the search consumed to find it —
and when that correction is applied to a whole programme rather than to one reported
family, results that looked clean stop being significant.

**A fill model is a hypothesis, not an accounting detail.** Modelling limit execution
with a depth proxy was the right instinct. A later measurement showed the underlying
assumption — that a price level empties as aggressive flow consumes the displayed size —
does not hold: levels also empty through cancellation and are replaced through refill.
An execution result is only as good as the fill model beneath it, and that model is
itself something to be tested rather than assumed.

**A regime split multiplies the search space.** Conditioning a configuration on a market
state does not add an independent check. It doubles the number of things tried, which
makes the surviving cell *more* likely to be noise, not less.

**Cost is a gate, not a footnote.** A break-even figure is only meaningful when the cost
basis is settled. Eclipse's contradiction register shows it was not uniform across the
code that produced those numbers. This is the origin of a rule now applied first rather
than last: compute the value available to a *perfect* forecaster before building any
predictor, and close the route if that ceiling sits under the cost of trading it.

## Where the historical record lives

Not here. The failure archive, the research-wave history, the decision log, the hypothesis
ledger, the prior canonical versions of the research bible and the report corpus itself
are all part of the private research estate.

That is not tidying. Those files are research receipts and they are kept, in full, with
the version history that makes them auditable — just not on a public surface, because a
list of what was tried is still a disclosure of what was tried. What survived them is
method, and method is what this page and
[`RESEARCH_METHOD.md`](RESEARCH_METHOD.md) publish.

## The rule that came out of all of this

> A development PASS is not a validated result.
> A validated result is not an economic one.
> An economic result is not an executable one.
> An executable result is not an operationally durable one.

Each arrow is a separate test, and most ideas stop at the first one.
