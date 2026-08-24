# PUBLICATION RISK REGISTER

**Date:** 2026-08-24 · **Scope:** what the public repository discloses, and what it
would disclose if the current working branch were merged.

This register exists because the repository already has a written, machine-checked
answer to the question *"what may be published?"* — `web/README.md`, enforced by
`web/tools/check_policy.py` and mutation-tested with 21 deliberate violations. That
policy governs the website. **It was never applied to the README or to `docs/`.** The
findings below are what happens when a reviewed policy covers one surface and not the
adjacent one.

The policy's own words, quoted so the standard is visible rather than paraphrased:

> **Never publish** — entry/exit rules, offsets, horizons, thresholds, feature
> definitions, formulas · horizon suffixes in names · any ranking or comparison between
> arms · any performance figure — bps, win rate, profit factor, drawdown, totals, or a
> comparison that implies one · anything derived from a sealed forward arm, in any
> aggregated form · hostnames, IPs, ports, credentials, real network layout, live positions.

## Risk classification used below

| Label | Meaning |
|---|---|
| `PUBLIC_SAFE` | May appear on the front page as written. |
| `RESEARCH_SENSITIVE` | Conceptually publishable; the specific numbers or rules are not. |
| `FROZEN_PRIVATE` | A frozen rule, contract, sealed window or evaluation boundary. Never publishable while frozen. |
| `STALE_OR_SUPERSEDED` | True once, no longer the state of the system. Publishable only under an explicit historical label. |
| `UNCERTAIN` | Could not be settled from the repository without opening something this work is not permitted to open. |

---

## P-1 · `FROZEN_PRIVATE` · **BLOCKING** — seven frozen mini-protocols would be newly published

**Artifact:** `docs/protocols/` — 7 files, none present on `origin/main`:

```
docs/protocols/AMI_S34_MASTER_EXECUTION_PROTOCOL_v1.1.md
docs/protocols/S34_ETH_SELL_DEEP_V_FADE_V0_1.md
docs/protocols/S34_STATE_MACHINE_V0_1.md
docs/protocols/S34_STOP_TIGHTEN_V0_1.md
docs/protocols/S34_V02_H4_SHADOW_PROTOCOL.md
docs/protocols/S34_V_ENGINE_V0_1.md
docs/protocols/S34_WINNER_EXTENSION_V0_1.md
```

**Risk:** each is a *Frozen Rule* table. Without quoting the values, one of these files
states, in a single table: the symbol, the liquidation side, a trigger threshold in
USDT notional, a depth band in bps, a prior-trend threshold in bps, the entry model,
the limit offset in bps, the conservative fill condition in bps, and the exit horizon —
followed by the two formulas that compute its features. That is a complete, executable
rule specification. It is simultaneously every category the policy forbids: entry/exit
rules, offsets, horizons, thresholds, feature definitions and formulas.

**Recommended public treatment:** do not publish. Keep the protocol directory out of
the public branch, or move it behind a private remote. If the *existence* of frozen
mini-protocols is worth showing — and it is, because freezing a rule before observation
is the discipline being demonstrated — publish the **shape** of a protocol (status
token, protocol id, purpose, "Frozen Rule" as a heading, permission label, evidence
pointer, refutation condition) with the value column removed. `docs/public/RESEARCH_METHOD.md`
does exactly this.

**Manual decision required: YES.** Whether `docs/protocols/` may reach a public branch
at all is an owner decision, not a documentation one.

---

## P-2 · `RESEARCH_SENSITIVE` + `STALE_OR_SUPERSEDED` · **HIGH** — the current README

**Artifact:** `README.md` on `origin/main`, live now.

**Risk:** the README publishes, in one screen:

| What | Policy category breached |
|---|---|
| A three-condition threshold rule with numeric bounds | thresholds |
| Four feature definitions written as formulas | feature definitions, formulas |
| The passive fill model as executable source, including the offset expression and the depth-proxy expression | formulas, offsets |
| A horizon in the section heading and again in the CLI example | horizons |
| Touch rate, fill rate, hit rate and adverse-path figures for two sides | performance figures |
| Regime pass rates and a per-strategy score column | performance figures |
| A GO / NO-GO / MARGINAL verdict column across four strategy variants | ranking between arms |
| A break-even fee figure | performance figure, and see P-7 |

**Aggravating factor:** the numbers are also *stale*. They come from a development-era
research line that later work superseded. So the page leaks and misinforms at the same
time — a visitor takes away both a rule they should not have and a picture of the
system that is years of research out of date.

**Recommended public treatment:** removed from the README. The *existence* of a
threshold-driven pocket study is described in
[`HISTORICAL_RESEARCH_CONTEXT.md`](../public/HISTORICAL_RESEARCH_CONTEXT.md) with every value,
formula and verdict stripped, under an explicit historical label.

**Manual decision required:** NO — this work removes them. But note that removing them
from the README does not remove them from git history; see P-8.

---

## P-3 · `RESEARCH_SENSITIVE` · **MEDIUM** — 816 tracked S34 report files are already public

**Artifact:** `reports/research/s34/**` — 816 files tracked, already on `origin/main`.

**Risk:** a sample of the first 200 found bps figures in 20 of them, with one audit file
carrying 18 separate bps quantities. Prereg documents in the same directory state
threshold values. This is a far larger disclosure surface than the README and it
predates this work.

**Mitigating factor:** everything *after* S34 is private. Verified: of the 87
directories under `reports/research/`, only `s34/` and `scalper_stack/` have tracked
files. The E-DER lanes, the S36–S92 study series and the current L1 queue-race lane are
**all untracked**, i.e. not public.

**Recommended public treatment:** no action in this pass, and specifically **no
deletion** — these are research receipts and deleting them to tidy a front page would
destroy evidence. Two options for the owner: (a) accept the exposure as historical and
label the directory, or (b) move the historical report corpus to a private remote and
keep only the index. The new README does not link into this directory, so the rewrite
does not amplify it.

**Manual decision required: YES.**

---

## P-4 · **HIGH — legal** — an MIT badge with no licence

**Artifact:** `README.md` badge row. `ls LICENSE* COPYING*` returns nothing.

**Risk:** the repository advertises MIT terms it has not granted. A reader may rely on
that badge. Simultaneously, with no `LICENSE` file, default copyright applies and no
one has any grant at all — so the badge is both an over-grant in appearance and a
non-grant in fact.

**Recommended public treatment:** the badge is **removed** by this work. The README now
carries an explicit "no licence has been granted" line under *What Eclipse Deliberately
Does Not Claim*. Adding a licence is deliberately **not** done here — licensing is an
owner decision and picking one silently would be worse than the badge.

**Manual decision required: YES** — choose a licence and add a `LICENSE` file, or
confirm the repository stays all-rights-reserved.

---

## P-5 · `RESEARCH_SENSITIVE` · **MEDIUM** — `SYSTEM_STATE.md` is public and growing

**Artifact:** `SYSTEM_STATE.md`. On `origin/main`: 846 KB. On the working branch:
2.71 MB.

**Risk:** it is the master state file and it is written for an operator, not for a
reader. It contains verdict tokens, per-study conclusions, effect directions, sample
counts and, in places, magnitudes. It is already public, so this is an increase in
exposure rather than a new one.

**Recommended public treatment:** owner decision. The new README does **not** link to
it. Note that the file is also the operating contract's primary artifact — moving it
private has an operational cost, so this is a genuine trade-off rather than an
oversight to be fixed.

**Manual decision required: YES.**

---

## P-6 · `RESEARCH_SENSITIVE` · **MEDIUM** — 849 further docs/CSV/JSON would be newly published

**Artifact:** the working branch adds 1,398 files relative to `origin/main`; 849 of them
are `.md`, `.csv` or `.json`. Breakdown by top-level directory:

| Directory | Newly added files |
|---|---|
| `reports/` | 824 |
| `tools/` | 343 |
| `docs/` | 50 |
| `tests/` | 49 |
| `src/` | 41 |
| `web/` | 22 |
| everything else | 69 |

Within `docs/`, the newly-published set includes `docs/research/ECLIPSE_RESEARCH_BIBLE.md`
(1,430 lines, 23 occurrences of a bps quantity), `docs/research/DECISION_LOG.md`,
`docs/research/HYPOTHESIS_LEDGER.md`, the full `data_feasibility_v1` audit with its CSV
evidence exports, and `docs/ECHO_SIGNAL_DEV_INDICATORS.md`.

**Risk:** mixed. The governance *structure* here is the project's best public material
— `docs/research/README.md` in particular is excellent and safe. The *evidence tables*
inside the Bible and the audit CSVs are not.

**Recommended public treatment:** review `docs/research/**` file by file before this
branch merges. This work links only to `docs/research/README.md`, which was read in
full and contains no performance figure, no threshold and no formula.

**Manual decision required: YES.**

---

## P-7 · `PUBLIC_SAFE` to *state*, `RESEARCH_SENSITIVE` to *quantify* — the fee-constant contradiction

**Artifact:** `CONTRADICTION_REGISTER.md` and the fee constants in active research code.

**Risk:** the repository's own register records that the round-trip taker fee constant
differs between active code paths, that no fee key is defined in the environment file,
and that results computed on different bases are **not cross-comparable** until the
real tier is confirmed in writing.

**Why it appears in a publication register:** the current README publishes a break-even
fee figure as though the cost model were settled. Publishing a derived number that
rests on an openly-registered contradiction is a stronger error than publishing the
number alone.

**Recommended public treatment:** the *fact* that cost-unit correctness is a tested
invariant, and that fee-base inconsistency is tracked as a live contradiction, is
excellent public material and is now in the README's *Deliberately Does Not Claim*
section. No figure accompanies it.

**Manual decision required:** NO.

---

## P-8 · `STALE_OR_SUPERSEDED` · **LOW, but permanent** — history retains what the README drops

**Artifact:** git history of `README.md` on the public remote.

**Risk:** removing the threshold triple and the results tables from the working copy
does not remove them from the public repository. They remain retrievable from every
prior commit, and from any fork or mirror already taken.

**Recommended public treatment:** treat P-2's values as **already disclosed** and plan
accordingly — do not rely on the rewrite as a containment measure. History rewriting is
not recommended and is not attempted here: it breaks every existing clone and fork, and
the values are development-era and superseded, so the cost exceeds the benefit.

**Manual decision required:** NO, unless the owner disagrees with that trade-off.

---

## P-9 · `PUBLIC_SAFE` · **INFORMATIONAL** — secrets are clean

Verified: `git ls-files` returns **no** `.env` file of any kind — not even
`.env.example`. `.gitignore` covers `.env` and `.env.*`, and the glob catches the
example files too. `.env`, `.env.draft`, `.env.example`, `.env.paper`, `.env.paper.dual`
and `.env.s34_live.example` all exist on disk and are all untracked.

Worth stating explicitly because it cuts the other way as well: `.env.example` carries
default sizing values, a leverage setting and a margin mode. Because nothing matching
`.env*` is tracked, none of that is public — but it also means **a new contributor has
no key reference in the repository**. `docs/ENV_REFERENCE.md` is tracked and serves
that role instead; `CONTRIBUTING.md` points there rather than at a `.env` template.

No credential, key, token, hostname, IP or port was introduced by this work. The new
README contains no network address of any kind.

---

## P-10 · `UNCERTAIN` — infrastructure quantities already in tracked docs

`REPOSITORY_RUNTIME_AUDIT.md` (tracked, public) states database sizes in GB, absolute
disk free space, and a per-store inventory with last-write dates. Under the site
policy, "real network layout" is forbidden but storage inventory is not explicitly
named. This is flagged rather than resolved.

**Recommended public treatment:** the new README carries no storage figure, no disk
path and no port number. Whether the existing audit document should keep them is left
open.

**Manual decision required: YES, if the owner considers storage scale sensitive.**

---

## Standing rule adopted by this work

Every claim considered for the README was put through one question before it was
written:

> Could this reveal a deployable threshold, private alpha logic, a sealed forward rule,
> a fresh evaluation window, a frozen economic decision rule, an unpublished edge
> magnitude, private runtime configuration, sensitive dataset provenance, or operator
> information?

Where the answer was yes, the claim was not published — including where the claim was
scientifically the most impressive thing available. The most interesting result in the
repository is, by construction, the one most likely to fail that test.

The README and every file under `docs/public/` are checked mechanically against this
standard by [`tools/check_public_docs.py`](tools/check_public_docs.py), which is
mutation-tested the same way `web/tools/check_policy.py` is. A rule that never fires
reads exactly like a rule that passes.
