# Security

## Reporting a vulnerability

**Do not open a public issue.**

Use GitHub's private vulnerability reporting on this repository
(*Security → Report a vulnerability*). If that is unavailable to you, open a public issue
containing **only** the words "security report, requesting a private channel" and no
technical detail, and wait to be contacted.

Please include, in the private channel: what you found, how to reproduce it, what an
attacker gains, and the commit you tested against.

This is a personal research project rather than a funded product. There is no bounty, and
there is no guaranteed response time. What you will get is an honest answer about whether
the finding is real and what is being done about it.

---

## What this repository is, for threat-modelling purposes

Being specific about this saves everyone time:

- It is **research and execution software**, not a hosted service. There is no server
  operated for third parties, no user accounts, no multi-tenancy and no public API.
- The operator dashboards are **not in this repository**, so there is no network service
  here to attack. Internally they bind to loopback and expose no order, cancel or
  position-control endpoint.
- **Live execution is off by default** and requires an explicit launcher flag.

## Findings that are in scope

- anything that could cause an unintended live order, or bypass the kill switch, circuit
  breaker, sizing bounds or order verifier
- anything that could cause paper or dry-run mode to reach the exchange (`SAF-02`)
- a path by which a credential or token could be logged, printed, persisted or committed
  (`SAF-01`)
- a dependency vulnerability with a plausible path to exploitation **in this repository's
  actual usage** — not a raw scanner listing
- anything that would let a contributor's change silently disable a safety check

## Findings that are out of scope

- research results being wrong, weak or unprofitable. That is a research question, not a
  security one, and the repository already says no route is validated
- missing hardening on a surface that is never exposed to a network
- scanner output with no exploitation path in this codebase
- the absence of a licence file — real, known, and tracked as an owner decision in
  [`docs/maintenance/PUBLICATION_RISK_REGISTER.md`](docs/maintenance/PUBLICATION_RISK_REGISTER.md)

---

## Secrets

`SAF-01` is a hard contract: API keys, tokens and secrets must never appear in a log, a
report, a commit or a printed line.

Verified at the time of the last public-surface audit: **`git ls-files` returns no `.env`
file of any kind**, including `.env.example`. `.gitignore` covers `.env` and `.env.*`, and
the glob catches the example files too. Supply your own environment file locally; the key
names are documented in [`docs/ENV_REFERENCE.md`](docs/ENV_REFERENCE.md).

If you believe a secret has been committed at any point in this repository's history,
report it privately using the process above rather than opening an issue — history is
public, and an issue is a pointer.

## Publication safety

This repository treats **disclosure of research content** as a security-adjacent concern
with its own policy and its own machine check. What may and may not be published is
defined in
[`docs/maintenance/PUBLICATION_RISK_REGISTER.md`](docs/maintenance/PUBLICATION_RISK_REGISTER.md)
and enforced by a mutation-tested checker.

If you find published material that breaches that policy — a threshold, a formula, a
horizon, a performance figure, a sealed aggregate, a hostname or a port — that is a valid
report, and the private channel above is the right place for it.

## Operational safety

One operational rule is worth stating publicly because it came from a real incident here:

> Process cleanup is never written as *"everything except the protected PID"*. It is
> always written as *"only the exact PIDs this job started"*.

A filtered mass-kill pattern once terminated 24 processes, took down the whole runtime
stack, and destroyed forward-only observations that cannot be recovered. Forward-only data
has no backup by construction — once the window passes, it is gone. If you contribute
anything that stops processes, it enumerates its own PIDs.
